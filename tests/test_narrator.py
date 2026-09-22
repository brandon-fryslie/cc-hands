"""A finished turn is heard: the Stop hook's turn read from the transcript, summarised, and spoken with the session's name."""

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest
from aiohttp import web
from loguru import logger
from pipecat.frames.frames import Frame, TTSSpeakFrame

from hands.core.delta import Changed, Delta
from hands.core.events import Ended, Joined, Prompted, Stopped
from hands.core.session import Membership, SessionId
from hands.core.turn import Budget
from hands.sessions.audit import Entry, Failure, Recounted, failures_to
from hands.sessions.registry import Sessions
from hands.sessions.tail import Tails
from hands.voice.narrator import narrate, recount
from hands.voice.pipeline import OpenAICompatibleBackend
from hands.voice.summary import SummaryFailed, summariser

FIXTURE = Path(__file__).parent / "fixtures" / "turn.jsonl"
BUDGET = Budget(opening=100, said=100, input=100, result=100, steps=10, files=10, commits=10, changes=500)
SID = SessionId("s1")


@dataclass
class Registry:
    """As much of the session registry as the tail asks about."""

    member: Membership

    def live_members(self) -> list[Membership]:
        return [self.member]

    def membership(self, session: SessionId) -> Membership | None:
        return self.member if session == self.member.id else None


def tailing(transcript: Path) -> Tails:
    """A tail following one session, as the daemon's does from the registry."""
    return Tails(Registry(Membership(SID, pid=4242, cwd=Path("/code/cc-hands"), transcript=transcript)))


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
    narrating = asyncio.create_task(narrate(sessions, Tails(sessions), summarise, frames.put, recorded.append, BUDGET))
    try:
        await sessions.apply(Joined(Membership(SID, pid=4242, cwd=Path("/code/cc-hands"), transcript=transcript), "startup"))
        await sessions.apply(Prompted(SID, at=1.0))
        await sessions.apply(Stopped(SID, None))
        spoken = await asyncio.wait_for(frames.get(), 5.0)
    finally:
        narrating.cancel()
    assert isinstance(spoken, TTSSpeakFrame)
    assert spoken.text == "Hands-free interactive coding agent architecture: Loaded the repo conventions and hit an API error."
    assert spoken.append_to_context
    [turn] = shown
    assert turn.startswith("The user asked:\nI'd like you to go a bit further") and "(Inspect repo layout and remotes)" in turn
    # The fixture turn is two blocks of text around three commands, none of which moved the repository, so
    # the narration leaves those two topics to be opened and plays neither.
    assert Recounted(SID, "Loaded the repo conventions and hit an API error.", ("what it said", "the commands"), ()) in recorded


async def test_a_session_that_ends_as_its_turn_is_summarised_is_heard_ending_after_that_turn(tmp_path: Path) -> None:
    transcript = tmp_path / "s1.jsonl"
    shutil.copy(FIXTURE, transcript)
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    thinking = asyncio.Event()

    async def summarise(turn: str) -> str:
        await thinking.wait()
        return "Hit an API error."

    frames: asyncio.Queue[Frame] = asyncio.Queue()
    narrating = asyncio.create_task(narrate(sessions, Tails(sessions), summarise, frames.put, lambda _: None, BUDGET))
    try:
        await sessions.apply(Joined(Membership(SID, pid=4242, cwd=Path("/code/cc-hands"), transcript=transcript), "startup"))
        await sessions.apply(Prompted(SID, at=1.0))
        await sessions.apply(Stopped(SID, None))
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


async def test_a_turn_that_stops_again_after_another_hook_blocked_its_stop_tells_only_what_is_new(tmp_path: Path) -> None:
    transcript = tmp_path / "s1.jsonl"
    prompt = '{"type":"user","uuid":"u1","message":{"role":"user","content":"fix it"}}'
    looked = '{"type":"assistant","uuid":"u2","message":{"content":[{"type":"text","text":"Looked."}]}}'
    call = '{"type":"assistant","uuid":"u3","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"pytest"}}]}}'
    result = '{"type":"user","uuid":"u4","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"1 passed"}]}}'
    fixed = '{"type":"assistant","uuid":"u5","message":{"content":[{"type":"text","text":"Fixed."}]}}'
    transcript.write_text(f"{prompt}\n{looked}\n")
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    shown: list[str] = []

    async def summarise(turn: str) -> str:
        shown.append(turn)
        return f"summary {len(shown)}"

    frames: asyncio.Queue[Frame] = asyncio.Queue()
    narrating = asyncio.create_task(narrate(sessions, Tails(sessions), summarise, frames.put, lambda _: None, BUDGET))
    try:
        await sessions.apply(Joined(Membership(SID, pid=4242, cwd=Path("/code/cc-hands"), transcript=transcript), "startup"))
        await sessions.apply(Prompted(SID, at=1.0))
        await sessions.apply(Stopped(SID, "Looked."))
        await asyncio.wait_for(frames.get(), 5.0)
        with transcript.open("a") as more:
            more.write(f"{call}\n{result}\n{fixed}\n")
        await sessions.apply(Stopped(SID, "Fixed."))
        await asyncio.wait_for(frames.get(), 5.0)
        # A third Stop with nothing written since is not a turn to tell.
        await sessions.apply(Stopped(SID, "Fixed."))
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(frames.get(), 0.2)
    finally:
        narrating.cancel()
    # The second telling carries the opening as context rather than as the request. Handed it as the request,
    # a small model answers it again: heard live on 2026-09-21 as a second summary restating the first half.
    assert shown == [
        "The user asked:\nfix it\n\nClaude said:\nLooked.",
        "This turn has already been reported once, up to and including its first step, and none of that may be"
        " reported again. For context only, this is what opened it:\nThe user asked:\nfix it\n"
        "Report only what it did after that, below.\n\nClaude ran pytest\nOutput: 1 passed\n\nClaude said:\nFixed.",
    ]


async def test_a_turn_that_cannot_be_summarised_is_said_to_have_failed_and_logged(tmp_path: Path) -> None:
    recorded: list[Entry] = []

    # A port nothing listens on: the model is unreachable the way a stopped inferno is.
    unreachable = summariser(OpenAICompatibleBackend(base_url="http://127.0.0.1:9/v1", model="m"), "Summarise.", max_tokens=50, timeout=5.0)
    sink = logger.add(failures_to(recorded.append), level="ERROR", filter="hands")
    try:
        spoken = await recount(tailing(FIXTURE), SID, None, "cc-hands", unreachable, recorded.append, BUDGET, Delta())
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

    spoken = await recount(tailing(tmp_path / "gone.jsonl"), SID, None, "cc-hands", never, lambda _: None, BUDGET, Delta())
    assert isinstance(spoken, TTSSpeakFrame) and spoken.text == "cc-hands finished a turn, and I could not summarise it."


async def test_a_session_that_stops_before_any_prompt_says_nothing(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('{"type":"ai-title","aiTitle":"x"}\n')

    async def never(turn: str) -> str:
        raise AssertionError("nothing to summarise")

    assert await recount(tailing(transcript), SID, None, "cc-hands", never, lambda _: None, BUDGET, Delta()) is None


async def test_a_turn_that_only_a_shell_command_changed_is_still_told_by_what_the_repository_says(tmp_path: Path) -> None:
    """A formatter run from a shell command leaves a turn whose steps say nothing about the files it rewrote.

    The turn is told anyway, and what it is told from is git: without this the user hears that a command ran
    and never hears that it rewrote forty files, which is the whole result of the turn.
    """
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('{"uuid":"u1","type":"user","message":{"role":"user","content":"run the formatter"}}\n')
    shown: list[str] = []

    async def summarise(turn: str) -> str:
        shown.append(turn)
        return "it reformatted the whole package"

    delta = Delta(files=(Changed("src/a.py", 12, 9), Changed("src/b.py", 3, 3)), commits=(), patch="@@\n-x\n+y\n")
    spoken = await recount(tailing(transcript), SID, None, "cc-hands", summarise, lambda _: None, BUDGET, delta)

    # What git says is said after the summary and out of the narration's own words: the two files are the whole
    # result of the turn, and no step of it names them.
    assert isinstance(spoken, TTSSpeakFrame) and spoken.text == "cc-hands: it reformatted the whole package. It left two files different."
    [rendered] = shown
    assert "src/a.py +12 -9" in rendered and "src/b.py +3 -3" in rendered


async def test_a_turn_that_did_nothing_and_changed_nothing_is_still_silent(tmp_path: Path) -> None:
    """The delta is a reason to speak, not an excuse to: a turn with neither steps nor changes has no news."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('{"uuid":"u1","type":"user","message":{"role":"user","content":"hello"}}\n')

    async def never(turn: str) -> str:
        raise AssertionError("nothing to summarise")

    assert await recount(tailing(transcript), SID, None, "cc-hands", never, lambda _: None, BUDGET, Delta()) is None


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


async def test_a_model_that_answers_with_nothing_is_said_to_have_failed_rather_than_spoken_as_a_stop() -> None:
    """The whole chain, because its two halves were pinned separately and the join between them was not: a
    model that answers with nothing raises out of the summariser, and `recount` is what catches it.

    Left to reach the narration, an empty reply becomes a headline of no sentences, which is given a stop so it
    does not run into what git says next — and the turn is spoken as a lone period. The reason that cannot
    happen is that the summariser refuses empty text at the boundary, so the narration is never handed any
    [LAW:single-enforcer]. This is the test that says so, rather than a second empty check inland.
    """
    runner, url, _ = await openai_server(None)
    recorded: list[Entry] = []
    sink = logger.add(failures_to(recorded.append), level="ERROR", filter="hands")
    try:
        summarise = summariser(OpenAICompatibleBackend(base_url=url, model="m"), "Summarise.", max_tokens=50, timeout=5.0)
        spoken = await recount(tailing(FIXTURE), SID, None, "cc-hands", summarise, recorded.append, BUDGET, Delta())
    finally:
        logger.remove(sink)
        await runner.cleanup()
    assert isinstance(spoken, TTSSpeakFrame)
    assert spoken.text == "cc-hands finished a turn, and I could not summarise it."
    [failure] = recorded
    assert isinstance(failure, Failure) and "SummaryFailed: the model returned no summary" in failure.message
