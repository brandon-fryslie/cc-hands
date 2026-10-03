"""What hands says unprompted is one control: each turn a session finishes is told through one summary, as it finishes
as finished turns are set to be told or the session is watched, and when the user asks for it otherwise, as a muted
one's and a quiet hands' always are. What a session asks is said for every one, whatever is set."""

import asyncio
import json
import subprocess
import zlib
from pathlib import Path
from typing import cast

import pytest
from loguru import logger
from pipecat.frames.frames import Frame, LLMMessagesAppendFrame

from hands.core.attention import Attention, Delivery, Level, Overlay, Spoken, Switch, Withheld
from hands.core.delta import Delta
from hands.core.events import Joined, Prompted, StatusReported, Stopped
from hands.core.session import Membership, PromptId, RequestId, SessionId
from hands.core.status import Idle, Report, Stamp
from hands.sessions.hookconfig import LAUNCHER, PLUGIN_DIR
from hands.sessions.audit import Entry, Failure, Recounted, Refocused, failures_to
from hands.sessions.focus import focused, set_focus
from hands.sessions.home import Home
from hands.sessions.overlays import Overlays
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions
from hands.sessions.attention import attention, set_attention
from hands.sessions.tail import Tails
from hands.voice.refocus import Refocus
from hands.voice.narrator import Recount, Recounts, delivery, narrate, recount
from hands.core.pending import News, Pending
from hands.voice import speech
from hands.voice.speech import Pushed, Told, Unprompted, frames
from hands.voice.tools import attention_tool, tell_turn_tool, set_overlay_tool

ONE = SessionId("one")
REPLY = "Pushed the fix. Want me to open a pull request?"


def membership(tmp_path: Path, name: str) -> Membership:
    return Membership(SessionId(name), pid=zlib.crc32(name.encode()), cwd=Path("/code") / name, transcript=tmp_path / f"{name}.jsonl")


def finished_turn(member: Membership) -> None:
    """A transcript holding one finished turn, which ended on a question."""
    asked = {"type": "user", "uuid": "p1-u", "promptId": "p1", "message": {"role": "user", "content": "fix it"}}
    said = {"type": "assistant", "uuid": "u2", "message": {"content": [{"type": "text", "text": REPLY}]}}
    member.transcript.write_text(f"{json.dumps(asked, separators=(",", ":"))}\n{json.dumps(said, separators=(",", ":"))}\n")


def handed(told: Frame | Pending | None) -> str:
    """What the floor hands the model of what the narrator told, as it lets it go: one message, and the model asked to answer it."""
    pending = told.pending if isinstance(told, Unprompted) else told
    assert pending is not None and not isinstance(pending, Frame)
    [frame, told] = frames(pending, Pushed(), names=lambda id: id)
    assert isinstance(frame, LLMMessagesAppendFrame) and frame.run_llm and isinstance(told, Told)
    match frame.messages:
        case [{"role": "user", "content": str() as content}]:
            return content
        case other:
            raise AssertionError(f"not one message from hands: {other!r}")


@pytest.mark.parametrize(
    ("finished", "quiet", "overlay", "delivered"),
    [
        ("full", "off", "normal", Spoken("full", "finished")),
        ("brief", "off", "normal", Spoken("brief", "finished")),
        ("full", "off", "watched", Spoken("full", "watched")),
        ("brief", "off", "watched", Spoken("brief", "watched")),
        ("off", "off", "watched", Spoken("full", "watched")),
        ("off", "off", "normal", Withheld("off")),
        ("full", "off", "muted", Withheld("muted")),
        ("off", "off", "muted", Withheld("muted")),
        ("full", "on", "normal", Withheld("quiet")),
        ("off", "on", "watched", Withheld("quiet")),
        ("full", "on", "muted", Withheld("muted")),
    ],
)
def test_a_turn_is_told_as_finished_turns_are_set_or_the_session_watched_and_otherwise_muted_or_quiet_when_asked(
    finished: Level, quiet: Switch, overlay: Overlay, delivered: Delivery
) -> None:
    assert delivery(Attention(finished=finished, quiet=quiet), overlay) == delivered


async def test_every_way_a_turn_is_told_tells_the_one_summary(tmp_path: Path) -> None:
    """The summary is the thing to iterate on, so there is one: told unasked or asked for, the model is handed the same."""
    member = membership(tmp_path, "one")
    finished_turn(member)
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(member, "startup"))
    told: dict[Delivery, str] = {}
    for delivered in (Spoken("full", "finished"), Spoken("full", "watched"), Withheld("off"), Withheld("muted"), Withheld("quiet")):
        recounts = Recounts()
        frame = await recount(Tails(sessions), member.id, PromptId("p1"), None, lambda _: None, Delta(), delivered, recounts)
        asked = await tell_turn_tool(sessions, recounts, Refocus(sessions, Home(tmp_path), lambda _: None)).body(session=member.id)
        assert asked == {"turn": "\n\n".join(speech.told(member.id, "one", (telling,), "full") for telling in cast(Recount, recounts.of(member.id)).tellings), "now": "not reported yet"}
        told[delivered] = handed(frame) if frame is not None else str(asked["turn"])
        assert (frame is None) == isinstance(delivered, Withheld)
    assert len(set(told.values())) == 1
    [summary] = set(told.values())
    assert summary.endswith(f"so they can answer without looking at the screen: It said: {REPLY.split('. ')[1]}")


async def test_a_turn_is_asked_for_only_of_a_running_session_and_one_with_nothing_to_tell_says_so(tmp_path: Path) -> None:
    member = membership(tmp_path, "one")
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    recounts = Recounts()
    asked = tell_turn_tool(sessions, recounts, Refocus(sessions, Home(tmp_path), lambda _: None))
    assert await asked.body(session=member.id) == {"error": "no running session has the id 'one'; take one from list_sessions"}
    await sessions.apply(Joined(member, "startup"))
    recounts.put(member.id, PromptId("p1"), None)
    assert await asked.body(session=member.id) == {
        "turn": "[hands] The Claude Code session one finished a turn with nothing in it hands could tell. Tell the user so.",
        "now": "not reported yet",
    }
    recounts.put(member.id, PromptId("p1"), News(None, "Told.", "", "", (), frozenset()))
    recounts.unread(member.id, PromptId("p1"))
    assert (await asked.body(session=member.id))["turn"] == f"{speech.told(member.id, 'one', (News(None, 'Told.', '', '', (), frozenset()),), 'full')}\n\n[hands] hands could not read the rest of the turn the Claude Code session one finished. Tell the user so."


async def test_the_narrator_tells_a_watched_session_s_turn_and_holds_an_unwatched_one_s_until_asked(tmp_path: Path) -> None:
    watched, other = membership(tmp_path, "watched"), membership(tmp_path, "other")
    home = Home(tmp_path / "home")
    entries: list[Entry] = []
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=entries.append)
    for each in (watched, other):
        finished_turn(each)
        await sessions.apply(Joined(each, "startup"))
        await sessions.apply(StatusReported(each.id, Report(Idle(), Stamp(1)), at=1.0))
    assert await set_overlay_tool(sessions, Overlays(home)).body(session=watched.id, overlay="watched") == {"readback": "I'll tell you each turn watched finishes."}
    queued: asyncio.Queue[Frame] = asyncio.Queue()
    recounts = Recounts()
    narrating = asyncio.create_task(narrate(sessions, Tails(sessions), queued.put, entries.append, Attention, Overlays(home), recounts))
    try:
        for each in (other, watched):
            await sessions.apply(Prompted(each.id, at=2.0, mode=None, prompt=PromptId("p1")))
            await sessions.apply(Stopped(each.id, REPLY, mode=None, prompt=PromptId("p1"), again=False, heard=Stamp(1500), request=RequestId(f"stop-{each.id}")))
        told = handed(await asyncio.wait_for(queued.get(), 5.0))
    finally:
        narrating.cancel()
    assert queued.empty()
    assert [speech.told(watched.id, "watched", (telling,), "full") for telling in cast(Recount, recounts.of(watched.id)).tellings] == [told] and "session watched (id watched) finished a turn" in told
    assert [telling.reply for telling in cast(Recount, recounts.of(other.id)).tellings] == [REPLY]
    assert [(entry.session, entry.delivered) for entry in entries if isinstance(entry, Recounted)] == [(other.id, Withheld("off")), (watched.id, Spoken("full", "watched"))]


async def test_a_session_whose_overlay_cannot_be_read_is_told_as_a_normal_one_and_the_reason_is_logged(tmp_path: Path) -> None:
    member = membership(tmp_path, "unreadable")
    finished_turn(member)
    home = Home(tmp_path / "home")
    home.overlays.mkdir(parents=True)
    home.overlay(member.id).write_text("loud\n")
    entries: list[Entry] = []
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=entries.append)
    await sessions.apply(Joined(member, "startup"))
    sink = logger.add(failures_to(entries.append), level="ERROR", filter="hands")
    narrating = asyncio.create_task(narrate(sessions, Tails(sessions), asyncio.Queue[Frame]().put, entries.append, Attention, Overlays(home), Recounts()))
    try:
        await sessions.apply(Prompted(member.id, at=2.0, mode=None, prompt=PromptId("p1")))
        await sessions.apply(Stopped(member.id, REPLY, mode=None, prompt=PromptId("p1"), again=False, heard=Stamp(1500), request=RequestId("stop")))
        async with asyncio.timeout(5):
            while not any(isinstance(entry, Recounted) for entry in entries):
                await asyncio.sleep(0.01)
    finally:
        narrating.cancel()
        logger.remove(sink)
    [recounted] = [entry for entry in entries if isinstance(entry, Recounted)]
    assert recounted.delivered == Withheld("off")
    assert any(isinstance(entry, Failure) and "cannot read the overlay of session unreadable" in entry.message and "loud" in entry.message for entry in entries)


async def test_a_turn_asked_for_moves_the_focus_to_its_session_as_a_turn_told_as_it_finishes_does(tmp_path: Path) -> None:
    member = membership(tmp_path, "one")
    home = Home(tmp_path)
    set_focus(home, SessionId("other"))
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(member, "startup"))
    recounts = Recounts()
    recounts.put(member.id, PromptId("p1"), News(None, "Told.", "", "", (), frozenset()))
    recorded: list[Entry] = []
    await tell_turn_tool(sessions, recounts, Refocus(sessions, home, recorded.append)).body(session=member.id)
    assert focused(home) == member.id
    # [LAW:nothing-unseen] recorded as a move made as a turn is told is, so its line and level are the one kind.
    assert recorded == [Refocused(member.id, "moved", None)]


async def test_a_turn_asked_for_before_any_has_finished_is_said_to_be_missing(tmp_path: Path) -> None:
    member = membership(tmp_path, "one")
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(member, "startup"))
    result = await tell_turn_tool(sessions, Recounts(), Refocus(sessions, Home(tmp_path), lambda _: None)).body(session=member.id)
    assert "error" in result and "has finished since hands started" in str(result["error"])


async def test_what_is_said_unprompted_is_set_by_voice_in_one_call_and_holds_across_a_restart(tmp_path: Path) -> None:
    setting = attention_tool(home := Home(tmp_path)).body
    result = await setting(changes=[{"kind": "finished", "level": "brief"}, {"kind": "ended", "level": "off"}])
    assert result == {
        "readback": "I tell each finished turn in a few words. I tell each step the focused session takes. I don't say when a session ends. What needs your answer is always said."
    }
    assert attention(Home(tmp_path)) == Attention(finished="brief", ended="off")
    # Quiet holds the rest and leaves its levels as they were, so talking again is back to them.
    await setting(changes=[{"kind": "quiet", "level": "on"}])
    assert attention(home) == Attention(finished="brief", ended="off", quiet="on")
    await setting(changes=[{"kind": "quiet", "level": "off"}])
    assert attention(home) == Attention(finished="brief", ended="off")


async def test_asking_what_is_set_changes_nothing_and_says_it(tmp_path: Path) -> None:
    result = await attention_tool(Home(tmp_path)).body(changes=[])
    assert result == {
        "readback": "Finished turns wait until you ask, except a watched session's. I tell each step the focused session takes. I say when a session ends. What needs your answer is always said."
    }
    # Read and never written, so it cannot land over a change made meanwhile.
    assert not Home(tmp_path).attention.exists()


async def test_a_change_missing_its_level_is_refused_to_the_model_with_nothing_set(tmp_path: Path) -> None:
    result = await attention_tool(home := Home(tmp_path)).body(changes=[{"kind": "quiet"}])
    assert result == {"error": "missing field 'level'"}
    assert not home.attention.exists()


async def test_a_kind_turned_on_is_all_of_it_and_a_level_a_kind_does_not_have_is_refused_with_nothing_set(tmp_path: Path) -> None:
    setting = attention_tool(home := Home(tmp_path)).body
    await setting(changes=[{"kind": "progress", "level": "on"}])
    assert attention(home).progress == "full"
    result = await setting(changes=[{"kind": "finished", "level": "off"}, {"kind": "ended", "level": "brief"}])
    assert result == {"error": "'brief' is no level of ended; it is one of on, off"}
    assert attention(home) == Attention()


def test_the_model_is_offered_only_the_kinds_and_levels_there_are(tmp_path: Path) -> None:
    items = cast(dict[str, dict[str, object]], cast(dict[str, object], attention_tool(Home(tmp_path)).properties["changes"]["items"])["properties"])
    assert (items["kind"]["enum"], items["level"]["enum"]) == (["finished", "progress", "ended", "quiet"], ["off", "brief", "full", "on"])


def test_a_file_holding_anything_but_settings_is_refused_and_a_kind_never_written_is_at_its_default(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.attention.write_text('{"finished": "brief"}')
    assert attention(home) == Attention(finished="brief")
    for written in ("loud", '["on"]', '{"finished": "loud"}', '{"volume": "full"}'):
        home.attention.write_text(written)
        with pytest.raises(Rejected):
            attention(home)


@pytest.mark.parametrize(
    ("overlay", "readback"),
    [
        ("watched", "I'll tell you each turn dropped finishes."),
        ("normal", "I'll hold dropped's turns until you ask for one."),
        ("muted", "dropped is muted: I'll hold its turns until you ask. It still speaks when it needs your answer."),
    ],
)
async def test_setting_a_session_s_overlay_says_what_is_told_of_it_and_holds(tmp_path: Path, overlay: Overlay, readback: str) -> None:
    member = membership(tmp_path, "dropped")
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(member, "startup"))
    result = await set_overlay_tool(sessions, Overlays(Home(tmp_path / "home"))).body(session=member.id, overlay=overlay)
    assert result == {"readback": readback}
    assert Overlays(Home(tmp_path / "home")).of(member.id) == overlay


@pytest.mark.parametrize(
    ("set_to", "readback"),
    [
        (Attention(finished="full"), "I'll tell you each turn dropped finishes, as I tell every session's."),
        (
            Attention(finished="full", quiet="on"),
            "For now I'm keeping quiet and holding dropped's turns. After that, I'll tell you each turn dropped finishes, as I tell every session's.",
        ),
        (Attention(quiet="on"), "For now I'm keeping quiet and holding dropped's turns. After that, I'll hold dropped's turns until you ask for one."),
    ],
)
async def test_a_normal_session_s_readback_says_how_its_turns_are_told_as_set(tmp_path: Path, set_to: Attention, readback: str) -> None:
    """The readback is the delivery: unwatching a session with finished turns told does not stop its turns, and says so."""
    member = membership(tmp_path, "dropped")
    home = Home(tmp_path / "home")
    set_attention(home, set_to)
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(member, "startup"))
    result = await set_overlay_tool(sessions, Overlays(home)).body(session=member.id, overlay="normal")
    assert result == {"readback": readback}


async def test_an_overlay_set_with_the_switch_unreadable_is_set_and_read_back_as_the_narrator_will_deliver_it(tmp_path: Path) -> None:
    member = membership(tmp_path, "dropped")
    home = Home(tmp_path / "home")
    home.attention.parent.mkdir(parents=True, exist_ok=True)
    home.attention.write_text("loud\n")
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(member, "startup"))
    result = await set_overlay_tool(sessions, Overlays(home)).body(session=member.id, overlay="normal")
    assert result == {"readback": "I'll hold dropped's turns until you ask for one."}
    assert Overlays(home).of(member.id) == "normal"


def test_set_overlay_offers_the_model_only_the_overlays_there_are(tmp_path: Path) -> None:
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    assert set_overlay_tool(sessions, Overlays(Home(tmp_path))).properties["overlay"]["enum"] == ["normal", "watched", "muted"]


async def test_an_overlay_the_model_names_that_is_none_is_refused_and_nothing_is_set(tmp_path: Path) -> None:
    member = membership(tmp_path, "one")
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(member, "startup"))
    result = await set_overlay_tool(sessions, Overlays(Home(tmp_path / "home"))).body(session=member.id, overlay="quiet")
    assert result == {"error": "'quiet' is no overlay; it is one of normal, watched, muted"}
    assert not (tmp_path / "home" / "overlays").exists()


async def test_a_muted_session_s_turn_is_held_with_finished_turns_told_and_told_when_asked(tmp_path: Path) -> None:
    """Muting holds a session's Stops: nothing is said as it finishes, and its turn is told when the user asks for it."""
    muted, other = membership(tmp_path, "muted"), membership(tmp_path, "other")
    home = Home(tmp_path / "home")
    entries: list[Entry] = []
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=entries.append)
    for each in (muted, other):
        finished_turn(each)
        await sessions.apply(Joined(each, "startup"))
        await sessions.apply(StatusReported(each.id, Report(Idle(), Stamp(1)), at=1.0))
    await set_overlay_tool(sessions, Overlays(home)).body(session=muted.id, overlay="muted")
    queued: asyncio.Queue[Frame] = asyncio.Queue()
    recounts = Recounts()
    narrating = asyncio.create_task(narrate(sessions, Tails(sessions), queued.put, entries.append, lambda: Attention(finished="full"), Overlays(home), recounts))
    try:
        for each in (muted, other):
            await sessions.apply(Prompted(each.id, at=2.0, mode=None, prompt=PromptId("p1")))
            await sessions.apply(Stopped(each.id, REPLY, mode=None, prompt=PromptId("p1"), again=False, heard=Stamp(1500), request=RequestId(f"stop-{each.id}")))
        told = handed(await asyncio.wait_for(queued.get(), 5.0))
    finally:
        narrating.cancel()
    assert queued.empty() and "session other (id other) finished a turn" in told
    assert [(entry.session, entry.delivered) for entry in entries if isinstance(entry, Recounted)] == [(muted.id, Withheld("muted")), (other.id, Spoken("full", "finished"))]
    asked = await tell_turn_tool(sessions, recounts, Refocus(sessions, Home(tmp_path), lambda _: None)).body(session=muted.id)
    assert "session muted (id muted) finished a turn" in str(asked["turn"]) and REPLY.split(". ")[0] in str(asked["turn"])


def test_an_overlay_holds_across_a_restart_and_a_session_never_set_is_normal(tmp_path: Path) -> None:
    Overlays(Home(tmp_path)).set(ONE, "muted")
    restarted = Overlays(Home(tmp_path))
    assert (restarted.of(ONE), restarted.of(SessionId("other"))) == ("muted", "normal")
    restarted.set(ONE, "normal")
    assert Overlays(Home(tmp_path)).of(ONE) == "normal"


def test_an_overlay_file_holding_anything_else_is_refused(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.overlays.mkdir()
    home.overlay(ONE).write_text("loud\n")
    with pytest.raises(Rejected):
        Overlays(home).of(ONE)


async def test_a_session_that_is_not_running_cannot_have_its_overlay_set(tmp_path: Path) -> None:
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    result = await set_overlay_tool(sessions, Overlays(Home(tmp_path))).body(session="../escape", overlay="watched")
    assert "error" in result
    assert not (tmp_path / "overlays").exists()


PLUGIN_ROOT = Path(__file__).resolve().parent.parent / PLUGIN_DIR


def skill(home: Home, cwd: Path, path: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    """`/hands:attention` as the skill runs it: the plugin's launcher, in the session's directory, with no venv."""
    environment = {"HANDS_HOME": str(home.root), "PATH": path, "HOME": str(cwd)}
    return subprocess.run([PLUGIN_ROOT / LAUNCHER, "-m", "hands.sessions.attention", *arguments], env=environment, cwd=cwd, capture_output=True, text=True)


def test_the_skill_sets_what_is_said_unprompted_and_says_what_is_set(tmp_path: Path, python312: str) -> None:
    home = Home(tmp_path / "home")
    asked = skill(home, tmp_path, python312)
    assert (asked.returncode, asked.stderr) == (0, "") and asked.stdout.startswith("Finished turns wait until you ask")
    quiet = skill(home, tmp_path, python312, "Quiet", "On", "progress", "brief")
    assert (quiet.returncode, quiet.stderr) == (0, "") and "I'm keeping quiet for now" in quiet.stdout
    assert attention(home) == Attention(progress="brief", quiet="on")


@pytest.mark.parametrize(
    ("arguments", "refused"),
    [
        (("quiet",), "hands attention: expected a kind and its level, in pairs, got 'quiet'\n"),
        (("volume", "up"), "hands attention: 'volume' is no kind of thing hands says unprompted; it is one of finished, progress, ended, quiet\n"),
    ],
)
def test_the_skill_refuses_what_is_no_setting_and_leaves_what_is_set_alone(tmp_path: Path, python312: str, arguments: tuple[str, ...], refused: str) -> None:
    home = Home(tmp_path / "home")
    set_attention(home, Attention(finished="full"))
    result = skill(home, tmp_path, python312, *arguments)
    assert (result.returncode != 0, result.stdout, result.stderr) == (True, "", refused)
    assert attention(home) == Attention(finished="full")
