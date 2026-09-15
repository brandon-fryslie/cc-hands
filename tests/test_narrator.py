"""A finished turn is heard: the Stop hook's turn read from the transcript, summarised, and spoken with the session's name."""

import asyncio
import shutil
from pathlib import Path

import pytest
from aiohttp import web
from loguru import logger
from pipecat.frames.frames import Frame, TTSSpeakFrame

from hands.core.effects import Summarise
from hands.core.events import Ended, Joined, Prompted, Stopped
from hands.core.session import Membership, SessionId
from hands.core.turn import Budget
from hands.sessions.audit import Entry, Failure, Recounted, failures_to
from hands.sessions.registry import Sessions
from hands.voice.narrator import narrate, recount
from hands.voice.pipeline import OpenAICompatibleBackend
from hands.voice.summary import SummaryFailed, summariser

FIXTURE = Path(__file__).parent / "fixtures" / "turn.jsonl"
BUDGET = Budget(opening=100, said=100, input=100, result=100, steps=10)
SID = SessionId("s1")


async def test_a_session_that_stops_is_heard_by_its_title_saying_what_the_turn_did(tmp_path: Path) -> None:
    transcript = tmp_path / "s1.jsonl"
    shutil.copy(FIXTURE, transcript)
    recorded: list[Entry] = []
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=recorded.append)
    shown: list[str] = []

    async def summarise(turn: str) -> str:
        shown.append(turn)
        return "Loaded the repo conventions and hit an API error."

    frames: asyncio.Queue[Frame] = asyncio.Queue()
    narrating = asyncio.create_task(narrate(sessions, summarise, frames.put, recorded.append, BUDGET))
    try:
        await sessions.apply(Joined(Membership(SID, pid=4242, cwd=Path("/code/cc-hands"), transcript=transcript), "startup"))
        await sessions.apply(Prompted(SID, at=1.0))
        await sessions.apply(Stopped(SID))
        spoken = await asyncio.wait_for(frames.get(), 5.0)
    finally:
        narrating.cancel()
    assert isinstance(spoken, TTSSpeakFrame)
    assert spoken.text == "Hands-free interactive coding agent architecture: Loaded the repo conventions and hit an API error."
    assert spoken.append_to_context
    [turn] = shown
    assert turn.startswith("The user asked:\nI'd like you to go a bit further") and "Claude used Bash (Inspect repo layout and remotes)" in turn
    assert Recounted(SID, "Loaded the repo conventions and hit an API error.") in recorded


async def test_a_session_that_ends_as_its_turn_is_summarised_is_heard_ending_after_that_turn(tmp_path: Path) -> None:
    transcript = tmp_path / "s1.jsonl"
    shutil.copy(FIXTURE, transcript)
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    thinking = asyncio.Event()

    async def summarise(turn: str) -> str:
        await thinking.wait()
        return "Hit an API error."

    frames: asyncio.Queue[Frame] = asyncio.Queue()
    narrating = asyncio.create_task(narrate(sessions, summarise, frames.put, lambda _: None, BUDGET))
    try:
        await sessions.apply(Joined(Membership(SID, pid=4242, cwd=Path("/code/cc-hands"), transcript=transcript), "startup"))
        await sessions.apply(Prompted(SID, at=1.0))
        await sessions.apply(Stopped(SID))
        # `claude -p` exits the moment its turn stops, so the end lands while the model is still summarising.
        await sessions.apply(Ended(SID, "other"))
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(frames.get(), 0.2)
        thinking.set()
        spoken = [await asyncio.wait_for(frames.get(), 5.0), await asyncio.wait_for(frames.get(), 5.0)]
    finally:
        narrating.cancel()
    assert [frame.text for frame in spoken if isinstance(frame, TTSSpeakFrame)] == [
        "Hands-free interactive coding agent architecture: Hit an API error.",
        "The session Hands-free interactive coding agent architecture is gone.",
    ]


async def test_a_turn_that_cannot_be_summarised_is_said_to_have_failed_and_logged(tmp_path: Path) -> None:
    recorded: list[Entry] = []

    # A port nothing listens on: the model is unreachable the way a stopped inferno is.
    unreachable = summariser(OpenAICompatibleBackend(base_url="http://127.0.0.1:9/v1", model="m"), "Summarise.", max_tokens=50, timeout=5.0)
    sink = logger.add(failures_to(recorded.append), level="ERROR", filter="hands")
    try:
        spoken = await recount(Summarise(SID, FIXTURE), "cc-hands", unreachable, recorded.append, BUDGET)
    finally:
        logger.remove(sink)
    assert isinstance(spoken, TTSSpeakFrame)
    assert spoken.text == "cc-hands finished a turn, and I could not summarise it."
    assert not spoken.append_to_context
    [failure] = recorded
    assert isinstance(failure, Failure) and "APIConnectionError: Connection error." in failure.message


async def test_a_missing_transcript_is_said_to_have_failed_too(tmp_path: Path) -> None:
    async def never(turn: str) -> str:
        raise AssertionError("nothing to summarise")

    spoken = await recount(Summarise(SID, tmp_path / "gone.jsonl"), "cc-hands", never, lambda _: None, BUDGET)
    assert isinstance(spoken, TTSSpeakFrame) and spoken.text == "cc-hands finished a turn, and I could not summarise it."


async def test_a_session_that_stops_before_any_prompt_says_nothing(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('{"type":"ai-title","aiTitle":"x"}\n')

    async def never(turn: str) -> str:
        raise AssertionError("nothing to summarise")

    assert await recount(Summarise(SID, transcript), "cc-hands", never, lambda _: None, BUDGET) is None


async def openai_server(content: str | None) -> tuple[web.AppRunner, str, list[dict[str, object]]]:
    """A chat completions endpoint that answers every request with content, and keeps what it was asked."""
    asked: list[dict[str, object]] = []

    async def complete(request: web.Request) -> web.Response:
        asked.append(await request.json())
        return web.json_response(
            {
                "id": "c1", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    return runner, f"http://127.0.0.1:{port}/v1", asked


async def test_the_openai_compatible_summariser_sends_the_instruction_and_the_turn_and_returns_the_text() -> None:
    runner, url, asked = await openai_server("  Fixed the test.  ")
    try:
        summarise = summariser(OpenAICompatibleBackend(base_url=url, model="m"), "Summarise.", max_tokens=50, timeout=5.0)
        assert await summarise("The user asked:\nfix it") == "Fixed the test."
    finally:
        await runner.cleanup()
    [request] = asked
    assert request["model"] == "m" and request["max_tokens"] == 50
    assert request["messages"] == [{"role": "system", "content": "Summarise."}, {"role": "user", "content": "The user asked:\nfix it"}]


async def test_a_summary_with_nothing_in_it_is_a_failure() -> None:
    runner, url, _ = await openai_server(None)
    try:
        summarise = summariser(OpenAICompatibleBackend(base_url=url, model="m"), "Summarise.", max_tokens=50, timeout=5.0)
        with pytest.raises(SummaryFailed):
            await summarise("The user asked:\nfix it")
    finally:
        await runner.cleanup()
