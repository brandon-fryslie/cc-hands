"""A finished turn is heard: the Stop's turn read from the transcript and handed to the model, with the reply that ended
it, to say in its own words."""

import asyncio
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest
from loguru import logger
from pipecat.frames.frames import Frame, LLMMessagesAppendFrame, TTSSpeakFrame

from hands.core import delta as repository
from hands.core.delta import Changed, Delta
from hands.core.events import Ended, Joined, Prompted, Stopped
from hands.core.session import Membership, PromptId, RequestId, SessionId
from hands.sessions.audit import Entry, Failure, Recounted, failures_to
from hands.sessions.registry import Sessions
from hands.sessions.home import Home
from hands.sessions.overlays import Overlays
from hands.sessions.summaries import Summaries, summaries
from hands.sessions.tail import Tails
from hands.voice.narrator import REPLY_SHOWN, Recounts, narrate, recount
from hands.voice.speech import Narrated, Pushed, Tailed
from hands.voice.pipeline import AnthropicBackend, OpenAICompatibleBackend
from hands.voice.summary import SummaryFailed, summariser

from conftest import ServeChat
from hands.core.status import Stamp

# When hands heard a Stop, on the clock Claude Code stamps a status with.
STOP_HEARD = Stamp(1500)
STOP_REQUEST = RequestId("stop")

FIXTURE = Path(__file__).parent / "fixtures" / "turn.jsonl"
SID = SessionId("s1")


def on() -> Summaries:
    return "on"


@dataclass
class Registry:
    """As much of the session registry as the tail asks about."""

    member: Membership

    def live_members(self) -> list[Membership]:
        return [self.member]

    def status_read(self, session: SessionId) -> bool:
        return True

    def now(self) -> float:
        return 0.0

    def stamp(self) -> Stamp:
        return Stamp(0)

    def membership(self, session: SessionId) -> Membership | None:
        return self.member if session == self.member.id else None


# The prompt id the captured turn's records carry.
TURN = PromptId("1dd65461-0644-432f-88e5-42aaeb27cbde")


def tailing(transcript: Path) -> Tails:
    """A tail following one session, as the daemon's does from the registry."""
    return Tails(Registry(Membership(SID, pid=4242, cwd=Path("/code/cc-hands"), transcript=transcript)))


def handed(frame: Frame | None) -> str:
    """What a model reached through Pipecat is handed to say: one message, and the model asked to answer it."""
    assert isinstance(frame, LLMMessagesAppendFrame) and frame.run_llm
    match frame.messages:
        case [{"role": "user", "content": str() as content}]:
            return content
        case other:
            raise AssertionError(f"not one message from hands: {other!r}")


def _prompt(prompt: str, text: str) -> str:
    return json.dumps({"type": "user", "uuid": f"{prompt}-u", "promptId": prompt, "message": {"role": "user", "content": text}}, separators=(",", ":"))


def _said(uuid: str, text: str) -> str:
    return json.dumps({"type": "assistant", "uuid": uuid, "message": {"content": [{"type": "text", "text": text}]}}, separators=(",", ":"))


def said_turn(tmp_path: Path, reply: str) -> Path:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(f"{_prompt('p1', 'fix the test')}\n{_said('u2', reply)}\n")
    return transcript


async def test_a_finished_turn_is_handed_to_the_model_with_its_reply_and_the_session_named(tmp_path: Path) -> None:
    """The bug this closes: a turn told by a summariser beside the model left the model not knowing what the user heard.
    Handed to the model as a turn, what it says of it is in its own history."""
    transcript = tmp_path / "s1.jsonl"
    shutil.copy(FIXTURE, transcript)
    recorded: list[Entry] = []
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=recorded.append)
    frames: asyncio.Queue[Frame] = asyncio.Queue()
    narrating = asyncio.create_task(narrate(sessions, Tails(sessions), Pushed(), frames.put, recorded.append, on, Overlays(Home(tmp_path / "home")), Recounts()))
    try:
        await sessions.apply(Joined(Membership(SID, pid=4242, cwd=Path("/code/cc-hands"), transcript=transcript), "startup"))
        await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=TURN))
        await sessions.apply(Stopped(SID, None, mode=None, prompt=TURN, again=False, heard=STOP_HEARD, request=STOP_REQUEST))
        told = handed(await asyncio.wait_for(frames.get(), 5.0))
    finally:
        narrating.cancel()
    assert told.startswith("[hands] The Claude Code session cc-hands finished a turn. The last thing it said was:\n\n")
    # The fixture's closing text, whole: the session's own account of what it did.
    assert "API Error" in told
    assert told.endswith("in one or two spoken sentences, naming the session. It asks the user nothing.")
    [recounted] = [entry for entry in recorded if isinstance(entry, Recounted)]
    assert recounted == Recounted(SID, told, ("what it said", "the commands"), (), "summaries", opened="Asked")


async def test_the_brain_takes_a_finished_turn_as_a_narration_and_never_through_the_pipelines_context(tmp_path: Path) -> None:
    told = await recount(tailing(said_turn(tmp_path, "Fixed it.")), SID, PromptId("p1"), None, "cc-hands", lambda _: None, Delta(), "summaries", Recounts(), Tailed())
    assert isinstance(told, Narrated) and "The last thing it said was:\n\nFixed it.\n\n" in told.text
    assert told.unsaid == "cc-hands finished a turn, and I could not tell it."


async def test_a_narration_the_brain_cannot_take_says_only_that_it_could_not_be_told(tmp_path: Path) -> None:
    """Nothing of a turn is said as written past the brain: a question said bare was answered by a user whose brain never heard it."""
    told = await recount(tailing(said_turn(tmp_path, "Fixed it. Want me to push it?")), SID, PromptId("p1"), None, "cc-hands", lambda _: None, Delta(), "summaries", Recounts(), Tailed())
    assert isinstance(told, Narrated) and told.unsaid == "cc-hands finished a turn, and I could not tell it."


async def test_a_turn_interrupted_mid_work_is_handed_on_with_the_last_thing_it_said(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    call = '{"type":"assistant","uuid":"u3","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"pytest"}}]}}'
    result = '{"type":"user","uuid":"u4","promptId":"p1","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"1 passed"}]}}'
    stop = '{"type":"user","uuid":"u5","promptId":"p1","message":{"role":"user","content":[{"type":"text","text":"[Request interrupted by user]"}]}}'
    transcript.write_text(f"{_prompt('p1', 'fix it')}\n{_said('u2', 'Fixed the parser; running the tests.')}\n{call}\n{result}\n{stop}\n")
    told = handed(await recount(tailing(transcript), SID, PromptId("p1"), None, "cc-hands", lambda _: None, Delta(), "summaries", Recounts(), Pushed()))
    assert "The last thing it said was:\n\nFixed the parser; running the tests.\n\nFrom its record, hands adds: You interrupted it." in told


async def test_a_turn_waiting_on_an_answer_is_handed_on_with_the_question_the_daemon_found_for_the_model_to_ask(tmp_path: Path) -> None:
    told = handed(await recount(tailing(said_turn(tmp_path, "Fixed it. Want me to push it?")), SID, PromptId("p1"), None, "cc-hands", lambda _: None, Delta(), "summaries", Recounts(), Pushed()))
    assert told.endswith("so end by asking it, with what it refers to, so they can answer without looking at the screen: It said: Want me to push it?")


async def test_what_is_handed_of_a_long_reply_is_bounded(tmp_path: Path) -> None:
    """Every turn told grows the brain's history toward compaction."""
    told = handed(await recount(tailing(said_turn(tmp_path, "x" * (REPLY_SHOWN * 3))), SID, PromptId("p1"), None, "cc-hands", lambda _: None, Delta(), "summaries", Recounts(), Pushed()))
    assert "x" * REPLY_SHOWN + "... (cut short)" in told and "x" * (REPLY_SHOWN + 1) not in told


async def test_a_session_that_ends_after_its_turn_is_heard_ending_after_that_turn(tmp_path: Path) -> None:
    transcript = tmp_path / "s1.jsonl"
    shutil.copy(FIXTURE, transcript)
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    frames: asyncio.Queue[Frame] = asyncio.Queue()
    narrating = asyncio.create_task(narrate(sessions, Tails(sessions), Pushed(), frames.put, lambda _: None, on, Overlays(Home(tmp_path / "home")), Recounts()))
    try:
        await sessions.apply(Joined(Membership(SID, pid=4242, cwd=Path("/code/cc-hands"), transcript=transcript), "startup"))
        await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=TURN))
        await sessions.apply(Stopped(SID, None, mode=None, prompt=TURN, again=False, heard=STOP_HEARD, request=STOP_REQUEST))
        # `claude -p` exits the moment its turn stops.
        await sessions.apply(Ended(SID, "other"))
        first, second = await asyncio.wait_for(frames.get(), 5.0), await asyncio.wait_for(frames.get(), 5.0)
    finally:
        narrating.cancel()
    assert "finished a turn" in handed(first)
    assert isinstance(second, TTSSpeakFrame) and second.text == "The session cc-hands is gone."


async def test_a_turn_held_until_asked_for_is_told_once(tmp_path: Path) -> None:
    """Held, it is told: a later telling of the same turn has nothing new, whatever the switch says by then."""
    tails = tailing(said_turn(tmp_path, "Fixed it. Want me to push it?"))
    recorded: list[Entry] = []
    recounts = Recounts()
    assert await recount(tails, SID, PromptId("p1"), None, "cc-hands", recorded.append, Delta(), "on request", recounts, Tailed()) is None
    assert await recount(tails, SID, PromptId("p1"), None, "cc-hands", recorded.append, Delta(), "summaries", Recounts(), Tailed()) is None
    [recounted] = [entry for entry in recorded if isinstance(entry, Recounted)]
    assert (recounted.delivered, recounted.told) == ("on request", recounts.of(SID))


async def test_a_turn_a_slash_command_opened_is_logged_as_commanded_rather_than_asked(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    skill = json.dumps({"type": "user", "uuid": "c1", "promptId": "p1", "message": {"role": "user", "content": "<command-message>ship</command-message>\n<command-name>/ship</command-name>\n<command-args>it</command-args>"}}, separators=(",", ":"))
    transcript.write_text(f"{skill}\n{_said('u2', 'Shipped.')}\n")
    recorded: list[Entry] = []
    await recount(tailing(transcript), SID, PromptId("p1"), None, "cc-hands", recorded.append, Delta(), "summaries", Recounts(), Tailed())
    assert [entry.opened for entry in recorded if isinstance(entry, Recounted)] == ["Commanded"]


async def test_a_switch_that_cannot_be_read_is_logged(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.summaries.write_text("yes\n")
    transcript = said_turn(tmp_path, "Fixed it. Want me to push it?")
    failed: asyncio.Queue[Failure] = asyncio.Queue()
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    sink = logger.add(failures_to(lambda entry: failed.put_nowait(entry) if isinstance(entry, Failure) else None), level="ERROR", filter="hands")
    narrating = asyncio.create_task(narrate(sessions, Tails(sessions), Pushed(), asyncio.Queue[Frame]().put, lambda _: None, lambda: summaries(home), Overlays(Home(tmp_path / "home")), Recounts()))
    try:
        await sessions.apply(Joined(Membership(SID, pid=4242, cwd=Path("/code/cc-hands"), transcript=transcript), "startup"))
        await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p1")))
        await sessions.apply(Stopped(SID, "Fixed it. Want me to push it?", mode=None, prompt=PromptId("p1"), again=False, heard=STOP_HEARD, request=STOP_REQUEST))
        failure = await asyncio.wait_for(failed.get(), 5.0)
    finally:
        narrating.cancel()
        logger.remove(sink)
    assert "cannot read whether spoken summaries are on" in failure.message and "neither on nor off" in failure.message


async def test_a_turn_that_stops_again_after_another_hook_blocked_its_stop_tells_only_what_is_new(tmp_path: Path) -> None:
    transcript = tmp_path / "s1.jsonl"
    call = '{"type":"assistant","uuid":"u3","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"pytest"}}]}}'
    result = '{"type":"user","uuid":"u4","promptId":"p1","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"1 passed"}]}}'
    transcript.write_text(f"{_prompt('p1', 'fix it')}\n{_said('u2', 'Looked.')}\n")
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    frames: asyncio.Queue[Frame] = asyncio.Queue()
    narrating = asyncio.create_task(narrate(sessions, Tails(sessions), Pushed(), frames.put, lambda _: None, on, Overlays(Home(tmp_path / "home")), Recounts()))
    try:
        await sessions.apply(Joined(Membership(SID, pid=4242, cwd=Path("/code/cc-hands"), transcript=transcript), "startup"))
        await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p1")))
        await sessions.apply(Stopped(SID, "Looked.", mode=None, prompt=PromptId("p1"), again=False, heard=STOP_HEARD, request=STOP_REQUEST))
        first = handed(await asyncio.wait_for(frames.get(), 5.0))
        with transcript.open("a") as more:
            more.write(f"{call}\n{result}\n{_said('u5', 'Fixed.')}\n")
        await sessions.apply(Stopped(SID, "Fixed.", mode=None, prompt=PromptId("p1"), again=True, heard=STOP_HEARD, request=STOP_REQUEST))
        second = handed(await asyncio.wait_for(frames.get(), 5.0))
        # A third Stop with nothing written since is not a turn to tell.
        await sessions.apply(Stopped(SID, "Fixed.", mode=None, prompt=PromptId("p1"), again=True, heard=STOP_HEARD, request=STOP_REQUEST))
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(frames.get(), 0.2)
    finally:
        narrating.cancel()
    assert "said was:\n\nLooked.\n\n" in first
    assert "said was:\n\nFixed.\n\n" in second


async def test_a_missing_transcript_is_said_to_have_failed_and_logged(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    sink = logger.add(failures_to(recorded.append), level="ERROR", filter="hands")
    try:
        spoken = await recount(tailing(tmp_path / "gone.jsonl"), SID, None, None, "cc-hands", recorded.append, Delta(), "summaries", Recounts(), Pushed())
    finally:
        logger.remove(sink)
    assert isinstance(spoken, TTSSpeakFrame) and spoken.text == "cc-hands finished a turn, and I could not read it."
    assert not spoken.append_to_context
    [failure] = recorded
    assert isinstance(failure, Failure) and "FileNotFoundError" in failure.message


async def test_a_session_that_stops_before_any_prompt_says_nothing(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('{"type":"ai-title","aiTitle":"x"}\n')
    assert await recount(tailing(transcript), SID, None, None, "cc-hands", lambda _: None, Delta(), "summaries", Recounts(), Pushed()) is None


async def test_a_turn_that_only_a_shell_command_changed_is_still_told_by_what_the_repository_says(tmp_path: Path) -> None:
    """A formatter run from a shell command leaves a turn whose steps say nothing about the files it rewrote.

    The turn is told anyway, and what it is told from is git: without this the user hears that a command ran
    and never hears that it rewrote forty files, which is the whole result of the turn.
    """
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('{"uuid":"u1","type":"user","message":{"role":"user","content":"run the formatter"}}\n')
    delta = Delta(files=(Changed("src/a.py", 12, 9), Changed("src/b.py", 3, 3)), commits=(), patch="@@\n-x\n+y\n")
    told = handed(await recount(tailing(transcript), SID, None, None, "cc-hands", lambda _: None, delta, "summaries", Recounts(), Pushed()))
    assert "finished a turn. It said nothing. From its record, hands adds: It left two files different. Tell the user" in told


async def test_a_push_and_a_pull_request_no_step_recorded_are_told_and_land_on_the_audit_line(tmp_path: Path) -> None:
    """A push made in a script leaves no step and no file; read off the repository, it is what the turn did."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('{"uuid":"u1","type":"user","message":{"role":"user","content":"ship it"}}\n')
    delta = Delta(changes=(repository.Pushed("fix"), repository.PullRequested(7, "https://x/7", "created")))
    recorded: list[Entry] = []
    told = handed(await recount(tailing(transcript), SID, None, None, "cc-hands", recorded.append, delta, "summaries", Recounts(), Pushed()))
    assert "From its record, hands adds: It pushed fix and created a pull request. Tell the user" in told
    [recounted] = [entry for entry in recorded if isinstance(entry, Recounted)]
    assert "It pushed fix and created a pull request." in recounted.told


async def test_a_turn_that_did_nothing_and_changed_nothing_is_still_silent(tmp_path: Path) -> None:
    """The delta is a reason to speak, not an excuse to: a turn with neither steps nor changes has no news."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('{"uuid":"u1","type":"user","message":{"role":"user","content":"hello"}}\n')
    assert await recount(tailing(transcript), SID, None, None, "cc-hands", lambda _: None, Delta(), "summaries", Recounts(), Pushed()) is None


async def test_a_turn_stopped_before_it_did_anything_is_handed_to_the_model_as_interrupted(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        '{"type":"user","promptId":"p1","message":{"role":"user","content":"Write an essay."}}\n'
        '{"type":"user","promptId":"p1","message":{"role":"user","content":[{"type":"text","text":"[Request interrupted by user]"}]}}\n'
    )
    told = handed(await recount(tailing(transcript), SID, None, None, "cc-hands", lambda _: None, Delta(), "summaries", Recounts(), Pushed()))
    assert "It said nothing. From its record, hands adds: You interrupted it. Tell the user" in told


async def test_the_openai_compatible_summariser_sends_the_instruction_and_the_turn_with_its_key_and_returns_the_text(
    chat_server: ServeChat,
) -> None:
    server = await chat_server("  Fixed the test.  ")
    summarise = summariser(OpenAICompatibleBackend(base_url=server.url, api_key="k", model="m"), "Summarise.", max_tokens=50, timeout=5.0)
    assert await summarise("The user asked:\nfix it") == "Fixed the test."
    [request] = server.asked
    assert request["model"] == "m" and request["max_tokens"] == 50
    assert request["messages"] == [{"role": "system", "content": "Summarise."}, {"role": "user", "content": "The user asked:\nfix it"}]
    assert server.keys == ["k"]


async def test_the_anthropic_summariser_sends_the_instruction_and_the_turn_with_its_key_to_its_url_and_returns_the_text(
    chat_server: ServeChat,
) -> None:
    server = await chat_server("  Fixed the test.  ")
    summarise = summariser(AnthropicBackend(base_url=server.anthropic_url, api_key="k", model="m"), "Summarise.", max_tokens=50, timeout=5.0)
    assert await summarise("The user asked:\nfix it") == "Fixed the test."
    [request] = server.asked
    assert request["model"] == "m" and request["max_tokens"] == 50 and request["system"] == "Summarise."
    assert request["messages"] == [{"role": "user", "content": "The user asked:\nfix it"}]
    assert server.keys == ["k"]


async def test_a_summary_with_nothing_in_it_is_a_failure(chat_server: ServeChat) -> None:
    server = await chat_server(None)
    summarise = summariser(OpenAICompatibleBackend(base_url=server.url, api_key="k", model="m"), "Summarise.", max_tokens=50, timeout=5.0)
    with pytest.raises(SummaryFailed):
        await summarise("The user asked:\nfix it")


def test_a_summary_turns_thinking_off_only_for_a_model_that_would_think_unasked() -> None:
    from anthropic import omit

    from hands.voice.summary import thinking

    assert thinking("claude-sonnet-5") == {"type": "disabled"}
    # Haiku does not think unless asked, and Opus 5.5 rejects thinking turned off, so neither is sent the setting.
    assert thinking("claude-haiku-4-5") is omit
    assert thinking("claude-opus-5-5") is omit
