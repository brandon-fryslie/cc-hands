"""A finished turn is heard: the Stop's turn read from the transcript and handed to the model, with the reply that ended
it, to say in its own words."""

import asyncio
import json
import shutil
from dataclasses import dataclass
from typing import cast
from pathlib import Path

import pytest
from loguru import logger
from pipecat.frames.frames import Frame, TTSSpeakFrame

from hands.core import delta as repository
from hands.core.delta import Changed, Delta
from hands.core.events import Ended, Joined, Prompted, Stopped
from hands.core.drive import DriveStopped, StartDrive
from hands.core.session import Drive, Membership, PromptId, RequestId, SessionId
from hands.sessions.audit import Entry, Failure, failures_to
from hands.sessions.wide import WideEvent, begun
from hands.sessions.registry import Sessions
from hands.sessions.home import Home
from hands.sessions.overlays import Overlays
from hands.core.attention import Attention, Spoken, Steering, Withheld
from hands.sessions.attention import attention
from hands.sessions.tail import Tails
from hands.core.pending import Finished, News, Pending
from hands.voice.narrator import Recount, Recounts, narrate, recount
from hands.voice.speech import ToAct, ToAsk, ToTell, REPLY_SHOWN, Aloud, Names, Narrated, Unprompted, frames, sent, told
from hands.voice.utterance import Utterance, Utterances

from hands.core.status import Stamp

# When hands heard a Stop, on the clock Claude Code stamps a status with.
STOP_HEARD = Stamp(1500)
STOP_REQUEST = RequestId("stop")

FIXTURE = Path(__file__).parent / "fixtures" / "turn.jsonl"
SID = SessionId("s1")


def on() -> Attention:
    return Attention(finished="full")


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


def unprompted(frame: Frame) -> Pending:
    """What the narrator queued for the floor."""
    assert isinstance(frame, Unprompted)
    return frame.pending


def rendered(pending: Pending, names: Names) -> tuple[Frame, ...]:
    """The frames the floor sends to say `pending`, a line hands says as written shown as the frame that says it."""
    return tuple(frame.spoken if isinstance(frame, Aloud) else frame for frame in sent(frames(pending, names), ()))


def heard() -> Utterance:
    """An utterance as the narrator opens one, for a test that hands it to a stage of the narration directly."""
    return Utterance(begun())


def utterances(recorded: list[Entry]) -> list[WideEvent]:
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "utterance"]


def said(told: Pending | None) -> Frame:
    """The frame the floor makes of what the narrator told, as it lets it go."""
    assert told is not None
    [frame] = rendered(told, lambda _: "cc-hands")
    return frame


def handed(told: Pending | None) -> str:
    """What the brain is handed to say, as a turn of its own."""
    frame = said(told)
    assert isinstance(frame, Narrated)
    return frame.text


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
    narrating = asyncio.create_task(narrate(sessions, Utterances(recorded.append), Tails(sessions), frames.put, on, Overlays(Home(tmp_path / "home")), Recounts()))
    try:
        await sessions.apply(Joined(Membership(SID, pid=4242, cwd=Path("/code/cc-hands"), transcript=transcript), "startup"))
        await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=TURN))
        await sessions.apply(Stopped(SID, None, mode=None, prompt=TURN, again=False, heard=STOP_HEARD, request=STOP_REQUEST))
        queued = await asyncio.wait_for(frames.get(), 5.0)
        told = handed(unprompted(queued))
    finally:
        narrating.cancel()
    assert told.startswith(f"[hands] The Claude Code session cc-hands (id {SID}) finished a turn. The last thing it said was:\n\n")
    # The fixture's closing text, whole: the session's own account of what it did.
    assert "API Error" in told
    assert told.endswith("in one or two spoken sentences, naming the session. It asks the user nothing.")
    [utterance] = cast(Unprompted, queued).utterances
    reply = utterance.facts.pop("reply")
    assert utterance.facts == {
        "session": SID,
        "heard": utterance.facts["heard"],
        "delivered": Spoken("full", "finished"),
        "facts": "",
        "topics": ("what it said", "the commands"),
        "questions": (),
        "opened": "Asked",
        "subagents": (),
        "unread_subagents": (),
    }
    assert isinstance(reply, str) and reply in told


async def test_the_brain_takes_a_finished_turn_as_a_narration_and_never_through_the_pipelines_context(tmp_path: Path) -> None:
    told = said(await recount(tailing(said_turn(tmp_path, "Fixed it.")), SID, PromptId("p1"), None, heard(), Delta(), Spoken("full", "finished"), Recounts()))
    assert isinstance(told, Narrated) and "The last thing it said was:\n\nFixed it.\n\n" in told.text
    assert told.unsaid == "cc-hands finished a turn, and I could not tell it."


async def test_a_narration_the_brain_cannot_take_says_only_that_it_could_not_be_told(tmp_path: Path) -> None:
    """Nothing of a turn is said as written past the brain: a question said bare was answered by a user whose brain never heard it."""
    told = said(await recount(tailing(said_turn(tmp_path, "Fixed it. Want me to push it?")), SID, PromptId("p1"), None, heard(), Delta(), Spoken("full", "finished"), Recounts()))
    assert isinstance(told, Narrated) and told.unsaid == "cc-hands finished a turn, and I could not tell it."


async def test_a_turn_interrupted_mid_work_is_handed_on_with_the_last_thing_it_said(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    call = '{"type":"assistant","uuid":"u3","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"pytest"}}]}}'
    result = '{"type":"user","uuid":"u4","promptId":"p1","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"1 passed"}]}}'
    stop = '{"type":"user","uuid":"u5","promptId":"p1","message":{"role":"user","content":[{"type":"text","text":"[Request interrupted by user]"}]}}'
    transcript.write_text(f"{_prompt('p1', 'fix it')}\n{_said('u2', 'Fixed the parser; running the tests.')}\n{call}\n{result}\n{stop}\n")
    told = handed(await recount(tailing(transcript), SID, PromptId("p1"), None, heard(), Delta(), Spoken("full", "finished"), Recounts()))
    assert "The last thing it said was:\n\nFixed the parser; running the tests.\n\nFrom its record, hands adds: You interrupted it." in told


async def test_a_turn_waiting_on_an_answer_is_handed_on_with_the_question_the_daemon_found_for_the_model_to_ask(tmp_path: Path) -> None:
    told = handed(await recount(tailing(said_turn(tmp_path, "Fixed it. Want me to push it?")), SID, PromptId("p1"), None, heard(), Delta(), Spoken("full", "finished"), Recounts()))
    assert told.endswith("so end by asking it, with what it refers to, so they can answer without looking at the screen: It said: Want me to push it?")


async def test_what_is_handed_of_a_long_reply_is_bounded(tmp_path: Path) -> None:
    """Every turn told grows the brain's history toward compaction."""
    told = handed(await recount(tailing(said_turn(tmp_path, "x" * (REPLY_SHOWN * 3))), SID, PromptId("p1"), None, heard(), Delta(), Spoken("full", "finished"), Recounts()))
    assert "x" * REPLY_SHOWN + "... (cut short)" in told and "x" * (REPLY_SHOWN + 1) not in told


def test_a_turn_told_again_in_one_fold_is_one_turn_that_went_on() -> None:
    """A turn that went on past its Stop is told twice; folded, it is still one turn, and the next one is another."""
    first, more, next_turn = News(PromptId("p1"), "Fixed it.", "", "", (), frozenset()), News(PromptId("p1"), "Pushed it.", "", "", (), frozenset()), News(PromptId("p2"), "Done.", "", "", (), frozenset())
    shown = told(SID, "cc-hands", (first, more, next_turn), "full")
    assert "finished 2 turns." in shown and "\n\nThen it went on: The last thing it said was:\n\nPushed it." in shown
    assert "Then, in the turn after that: The last thing it said was:\n\nDone." in shown
    assert "finished a turn." in told(SID, "cc-hands", (News(None, "a", "", "", (), frozenset()),), "full") and "finished 2 turns." in told(SID, "cc-hands", (News(None, "a", "", "", (), frozenset()), News(None, "b", "", "", (), frozenset())), "full")


def test_turns_folded_into_one_telling_share_its_bound() -> None:
    """Folding five long turns hands the model no more of their replies than one long turn told alone."""
    news = tuple(News(None, str(each) * (REPLY_SHOWN * 3), "", "", (), frozenset()) for each in range(5))
    shown = told(SID, "cc-hands", news, "full")
    assert all(str(each) * (REPLY_SHOWN // 5) + "... (cut short)" in shown for each in range(5)) and "0" * (REPLY_SHOWN // 5 + 1) not in shown


@pytest.mark.parametrize("set_to", [Attention(finished="full", ended="off"), Attention(finished="full", quiet="on")])
async def test_a_session_ending_is_not_said_with_endings_off_or_while_quiet_and_its_line_says_why(tmp_path: Path, set_to: Attention) -> None:
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    frames: asyncio.Queue[Frame] = asyncio.Queue()
    recorded: list[Entry] = []
    said_unasked = Utterances(recorded.append)
    keeping = asyncio.create_task(said_unasked.keep())
    narrating = asyncio.create_task(narrate(sessions, said_unasked, Tails(sessions), frames.put, lambda: set_to, Overlays(Home(tmp_path / "home")), Recounts()))
    try:
        await sessions.apply(Joined(Membership(SID, pid=4242, cwd=Path("/code/cc-hands"), transcript=tmp_path / "s1.jsonl"), "startup"))
        await sessions.apply(Ended(SID, "other"))
        async with asyncio.timeout(5):
            while not utterances(recorded):
                await asyncio.sleep(0.01)
    finally:
        narrating.cancel()
        keeping.cancel()
    assert frames.empty()
    [ending] = utterances(recorded)
    assert (ending.outcome, ending.facts["fate"], ending.facts["attention"], ending.facts["route"], ending.facts["session"]) == ("ok", "noted", set_to, "note", SID)


def test_a_turn_told_briefly_asks_the_model_for_a_few_words_and_still_for_what_the_session_asks() -> None:
    news = (News(None, "Fixed it.", "", "Want me to push it?", (), frozenset()),)
    brief, full = told(SID, "cc-hands", news, "brief"), told(SID, "cc-hands", news, "full")
    assert "in a few words which session finished and the one thing it did" in brief and "one or two spoken sentences" in full
    assert brief.endswith("Want me to push it?") and full.endswith("Want me to push it?")


async def test_a_session_that_ends_after_its_turn_is_heard_ending_after_that_turn(tmp_path: Path) -> None:
    transcript = tmp_path / "s1.jsonl"
    shutil.copy(FIXTURE, transcript)
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    frames: asyncio.Queue[Frame] = asyncio.Queue()
    narrating = asyncio.create_task(narrate(sessions, Utterances(lambda _: None), Tails(sessions), frames.put, on, Overlays(Home(tmp_path / "home")), Recounts()))
    try:
        await sessions.apply(Joined(Membership(SID, pid=4242, cwd=Path("/code/cc-hands"), transcript=transcript), "startup"))
        await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=TURN))
        await sessions.apply(Stopped(SID, None, mode=None, prompt=TURN, again=False, heard=STOP_HEARD, request=STOP_REQUEST))
        # `claude -p` exits the moment its turn stops.
        await sessions.apply(Ended(SID, "other"))
        first, second = unprompted(await asyncio.wait_for(frames.get(), 5.0)), unprompted(await asyncio.wait_for(frames.get(), 5.0))
    finally:
        narrating.cancel()
    assert "finished a turn" in handed(first)
    second = said(second)
    assert isinstance(second, TTSSpeakFrame) and second.text == "The session cc-hands is gone."


async def test_a_turn_held_until_asked_for_is_told_once(tmp_path: Path) -> None:
    """Held, it is told: a later telling of the same turn has nothing new, whatever is set by then."""
    tails = tailing(said_turn(tmp_path, "Fixed it. Want me to push it?"))
    recounts = Recounts()
    held, again = heard(), heard()
    assert await recount(tails, SID, PromptId("p1"), None, held, Delta(), Withheld("off"), recounts) is None
    assert await recount(tails, SID, PromptId("p1"), None, again, Delta(), Spoken("full", "finished"), Recounts()) is None
    assert held.facts["delivered"] == Withheld("off") and [(telling.reply, telling.facts) for telling in cast(Recount, recounts.of(SID)).tellings] == [(held.facts["reply"], held.facts["facts"])]
    assert "reply" not in again.facts


async def test_a_turn_held_until_asked_for_whose_transcript_cannot_be_read_says_nothing_and_holds_the_failure(tmp_path: Path) -> None:
    recounts = Recounts()
    assert await recount(tailing(tmp_path / "gone.jsonl"), SID, PromptId("p1"), None, heard(), Delta(), Withheld("off"), recounts) is None
    assert recounts.of(SID) == Recount(PromptId("p1"), (), unread=True)
    recounts.put(SID, PromptId("p1"), News(None, "read after all", "", "", (), frozenset()))
    assert recounts.of(SID) == Recount(PromptId("p1"), (News(None, "read after all", "", "", (), frozenset()),))
    recounts.unread(SID, PromptId("p1"))
    assert recounts.of(SID) == Recount(PromptId("p1"), (News(None, "read after all", "", "", (), frozenset()),), unread=True)


def test_a_turn_told_again_is_held_whole_and_the_next_turn_replaces_it_even_with_nothing_to_tell() -> None:
    recounts = Recounts()
    first, then, another = News(None, "first", "", "", (), frozenset()), News(None, "then", "", "", (), frozenset()), News(None, "another", "", "", (), frozenset())
    recounts.put(SID, PromptId("p1"), first)
    recounts.put(SID, PromptId("p1"), then)
    recounts.put(SID, PromptId("p1"), None)
    assert recounts.of(SID) == Recount(PromptId("p1"), (first, then))
    recounts.put(SID, PromptId("p2"), None)
    assert recounts.of(SID) == Recount(PromptId("p2"), ())
    recounts.put(SID, None, News(None, "unnamed", "", "", (), frozenset()))
    recounts.put(SID, None, another)
    assert recounts.of(SID) == Recount(None, (another,))


async def test_a_turn_a_slash_command_opened_is_logged_as_commanded_rather_than_asked(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    skill = json.dumps({"type": "user", "uuid": "c1", "promptId": "p1", "message": {"role": "user", "content": "<command-message>ship</command-message>\n<command-name>/ship</command-name>\n<command-args>it</command-args>"}}, separators=(",", ":"))
    transcript.write_text(f"{skill}\n{_said('u2', 'Shipped.')}\n")
    utterance = heard()
    await recount(tailing(transcript), SID, PromptId("p1"), None, utterance, Delta(), Spoken("full", "finished"), Recounts())
    assert utterance.facts["opened"] == "Commanded"


async def test_a_setting_that_cannot_be_read_is_logged(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.attention.write_text("yes\n")
    transcript = said_turn(tmp_path, "Fixed it. Want me to push it?")
    failed: asyncio.Queue[Failure] = asyncio.Queue()
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    sink = logger.add(failures_to(lambda entry: failed.put_nowait(entry) if isinstance(entry, Failure) else None), level="ERROR", filter="hands")
    narrating = asyncio.create_task(narrate(sessions, Utterances(lambda _: None), Tails(sessions), asyncio.Queue[Frame]().put, lambda: attention(home), Overlays(Home(tmp_path / "home")), Recounts()))
    try:
        await sessions.apply(Joined(Membership(SID, pid=4242, cwd=Path("/code/cc-hands"), transcript=transcript), "startup"))
        await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p1")))
        await sessions.apply(Stopped(SID, "Fixed it. Want me to push it?", mode=None, prompt=PromptId("p1"), again=False, heard=STOP_HEARD, request=STOP_REQUEST))
        failure = await asyncio.wait_for(failed.get(), 5.0)
    finally:
        narrating.cancel()
        logger.remove(sink)
    assert "cannot read what hands is set to say unprompted" in failure.message and "is not JSON" in failure.message


async def test_a_turn_that_stops_again_after_another_hook_blocked_its_stop_tells_only_what_is_new(tmp_path: Path) -> None:
    transcript = tmp_path / "s1.jsonl"
    call = '{"type":"assistant","uuid":"u3","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"pytest"}}]}}'
    result = '{"type":"user","uuid":"u4","promptId":"p1","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"1 passed"}]}}'
    transcript.write_text(f"{_prompt('p1', 'fix it')}\n{_said('u2', 'Looked.')}\n")
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    frames: asyncio.Queue[Frame] = asyncio.Queue()
    narrating = asyncio.create_task(narrate(sessions, Utterances(lambda _: None), Tails(sessions), frames.put, on, Overlays(Home(tmp_path / "home")), Recounts()))
    try:
        await sessions.apply(Joined(Membership(SID, pid=4242, cwd=Path("/code/cc-hands"), transcript=transcript), "startup"))
        await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p1")))
        await sessions.apply(Stopped(SID, "Looked.", mode=None, prompt=PromptId("p1"), again=False, heard=STOP_HEARD, request=STOP_REQUEST))
        first = handed(unprompted(await asyncio.wait_for(frames.get(), 5.0)))
        with transcript.open("a") as more:
            more.write(f"{call}\n{result}\n{_said('u5', 'Fixed.')}\n")
        await sessions.apply(Stopped(SID, "Fixed.", mode=None, prompt=PromptId("p1"), again=True, heard=STOP_HEARD, request=STOP_REQUEST))
        second = handed(unprompted(await asyncio.wait_for(frames.get(), 5.0)))
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
        utterance = heard()
        spoken = said(await recount(tailing(tmp_path / "gone.jsonl"), SID, None, None, utterance, Delta(), Spoken("full", "finished"), Recounts()))
    finally:
        logger.remove(sink)
    assert utterance.failure is not None and "FileNotFoundError" in utterance.failure
    assert isinstance(spoken, TTSSpeakFrame) and spoken.text == "cc-hands finished a turn, and I could not read it."
    assert not spoken.append_to_context
    [failure] = recorded
    assert isinstance(failure, Failure) and "FileNotFoundError" in failure.message


async def test_a_session_that_stops_before_any_prompt_says_nothing(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('{"type":"ai-title","aiTitle":"x"}\n')
    assert await recount(tailing(transcript), SID, None, None, heard(), Delta(), Spoken("full", "finished"), Recounts()) is None


async def test_a_turn_that_only_a_shell_command_changed_is_still_told_by_what_the_repository_says(tmp_path: Path) -> None:
    """A formatter run from a shell command leaves a turn whose steps say nothing about the files it rewrote.

    The turn is told anyway, and what it is told from is git: without this the user hears that a command ran
    and never hears that it rewrote forty files, which is the whole result of the turn.
    """
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('{"uuid":"u1","type":"user","message":{"role":"user","content":"run the formatter"}}\n')
    delta = Delta(files=(Changed("src/a.py", 12, 9), Changed("src/b.py", 3, 3)), commits=(), patch="@@\n-x\n+y\n")
    told = handed(await recount(tailing(transcript), SID, None, None, heard(), delta, Spoken("full", "finished"), Recounts()))
    assert "finished a turn. It said nothing. From its record, hands adds: It left two files different. Tell the user" in told


async def test_a_push_and_a_pull_request_no_step_recorded_are_told_and_land_on_the_audit_line(tmp_path: Path) -> None:
    """A push made in a script leaves no step and no file; read off the repository, it is what the turn did."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('{"uuid":"u1","type":"user","message":{"role":"user","content":"ship it"}}\n')
    delta = Delta(changes=(repository.Pushed("fix"), repository.PullRequested(7, "https://x/7", "created")))
    utterance = heard()
    told = handed(await recount(tailing(transcript), SID, None, None, utterance, delta, Spoken("full", "finished"), Recounts()))
    assert "From its record, hands adds: It pushed fix and created a pull request. Tell the user" in told
    assert "It pushed fix and created a pull request." in cast(str, utterance.facts["facts"])


async def test_a_turn_that_did_nothing_and_changed_nothing_is_still_silent(tmp_path: Path) -> None:
    """The delta is a reason to speak, not an excuse to: a turn with neither steps nor changes has no news."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('{"uuid":"u1","type":"user","message":{"role":"user","content":"hello"}}\n')
    assert await recount(tailing(transcript), SID, None, None, heard(), Delta(), Spoken("full", "finished"), Recounts()) is None


async def test_a_turn_stopped_before_it_did_anything_is_handed_to_the_model_as_interrupted(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        '{"type":"user","promptId":"p1","message":{"role":"user","content":"Write an essay."}}\n'
        '{"type":"user","promptId":"p1","message":{"role":"user","content":[{"type":"text","text":"[Request interrupted by user]"}]}}\n'
    )
    told = handed(await recount(tailing(transcript), SID, None, None, heard(), Delta(), Spoken("full", "finished"), Recounts()))
    assert "It said nothing. From its record, hands adds: You interrupted it. Tell the user" in told


async def test_a_driven_sessions_turn_is_handed_to_the_brain_to_act_on_with_finished_turns_off(tmp_path: Path) -> None:
    """Off is the default for finished turns: without the drive's row in the table the turn would wait to be asked for,
    and the drive would stall with nobody told."""
    transcript = tmp_path / "s1.jsonl"
    shutil.copy(FIXTURE, transcript)
    recorded: list[Entry] = []
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=recorded.append)
    frames: asyncio.Queue[Frame] = asyncio.Queue()
    narrating = asyncio.create_task(narrate(sessions, Utterances(recorded.append), Tails(sessions), frames.put, Attention, Overlays(Home(tmp_path / "home")), Recounts()))
    try:
        await sessions.apply(Joined(Membership(SID, pid=4242, cwd=Path("/code/cc-hands"), transcript=transcript), "startup"))
        await sessions.drive(StartDrive(SID, "keep fixing tests until they pass"))
        await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=TURN))
        await sessions.apply(Stopped(SID, None, mode=None, prompt=TURN, again=False, heard=STOP_HEARD, request=STOP_REQUEST))
        queued = await asyncio.wait_for(frames.get(), 5.0)
    finally:
        narrating.cancel()
    frame = said(unprompted(queued))
    assert isinstance(frame, Narrated) and isinstance(frame.handed, ToAct)
    assert 'You are driving cc-hands under the user\'s standing order: "keep fixing tests until they pass". You have sent it 0 of the 20 prompts' in frame.text
    assert f"call drive_send with session {SID}" in frame.text
    [utterance] = cast(Unprompted, queued).utterances
    assert utterance.facts["delivered"] == Steering(Drive("keep fixing tests until they pass", 0), Withheld("off"))


def test_an_undriven_turn_is_not_marked_driven() -> None:
    news = News(PromptId("p1"), "Fixed it.", "", "", (), frozenset())
    frame = said(Finished(SID, (news,), "full"))
    assert isinstance(frame, Narrated) and frame.handed == ToTell(SID)


def test_a_driven_turn_that_asks_the_user_stops_the_drive_and_puts_the_question() -> None:
    news = News(PromptId("p1"), "Should I delete the legacy migration?", "", "Should I delete the legacy migration?", (), frozenset())
    text = told(SID, "billing", (news,), Steering(Drive("keep billing going", 2), Spoken("brief", "finished")))
    assert f"Call stop_driving with session {SID}" in text and "Should I delete the legacy migration?" in text
    assert "drive_send" not in text


def test_a_driven_turn_the_user_set_held_is_acted_on_in_silence() -> None:
    news = News(PromptId("p1"), "Fixed one test.", "", "", (), frozenset())
    text = told(SID, "billing", (news,), Steering(Drive("keep billing going", 2), Withheld("quiet")))
    assert "call drive_send" in text and "say nothing to the user" in text


async def test_a_driven_turn_that_cannot_be_read_ends_the_drive_and_says_so(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=recorded.append)
    frames: asyncio.Queue[Frame] = asyncio.Queue()
    narrating = asyncio.create_task(narrate(sessions, Utterances(recorded.append), Tails(sessions), frames.put, Attention, Overlays(Home(tmp_path / "home")), Recounts()))
    try:
        await sessions.apply(Joined(Membership(SID, pid=4242, cwd=Path("/code/cc-hands"), transcript=tmp_path / "gone.jsonl"), "startup"))
        await sessions.drive(StartDrive(SID, "keep fixing tests until they pass"))
        await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=TURN))
        await sessions.apply(Stopped(SID, None, mode=None, prompt=TURN, again=False, heard=STOP_HEARD, request=STOP_REQUEST))
        queued = await asyncio.wait_for(frames.get(), 5.0)
    finally:
        narrating.cancel()
    spoken = said(unprompted(queued))
    assert isinstance(spoken, TTSSpeakFrame) and spoken.text == "cc-hands finished a turn, and I could not read it, so I stopped driving it."
    assert sessions.driven(SID) is None
    [utterance] = cast(Unprompted, queued).utterances
    assert utterance.facts["drive_ended"] == DriveStopped(SID, Drive("keep fixing tests until they pass", 0))


def test_a_driven_turn_that_asks_the_user_moves_their_focus_and_a_failed_one_says_the_drive_waits() -> None:
    asks = News(PromptId("p1"), "Delete it?", "", "Delete the legacy migration?", (), frozenset())
    plain = News(PromptId("p1"), "Fixed one.", "", "", (), frozenset())
    steering = Steering(Drive("keep billing going", 2), Spoken("brief", "finished"))
    asked = said(Finished(SID, (asks,), steering))
    acted = said(Finished(SID, (plain,), steering))
    assert isinstance(asked, Narrated) and asked.handed == ToAsk(SID, Drive("keep billing going", 2))
    assert isinstance(acted, Narrated) and acted.handed == ToAct(SID, Drive("keep billing going", 2))
    assert acted.unsaid == "cc-hands finished a turn while I was driving it, and I could not act on it."
