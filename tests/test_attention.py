"""A session's waiting is said only for a session the user asked to hear from; what a session asks is said for every one."""

import asyncio
import zlib
from pathlib import Path

import pytest
from pipecat.frames.frames import Frame, TTSSpeakFrame

from hands.core.attention import routed
from hands.core.effects import Asking, Expired, Heard, Narrate, Speak, WaitingForYou
from hands.core.events import Joined, StatusReported, Waited
from hands.core.session import Membership, Permission, RequestId, SessionId
from hands.core.status import Idle, Report, Stamp
from hands.sessions.audit import Entry, Routed
from hands.sessions.home import Home
from hands.sessions.overlays import Overlays
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions
from hands.voice.speech import Pushed, relay
from hands.voice.tools import watch_session_tool

ONE = SessionId("one")
ASKED = Permission("Bash", {})


def membership(tmp_path: Path, name: str) -> Membership:
    return Membership(SessionId(name), pid=zlib.crc32(name.encode()), cwd=Path("/code") / name, transcript=tmp_path / f"{name}.jsonl")


@pytest.mark.parametrize(
    ("heard", "normal", "watched"),
    [
        (Speak(WaitingForYou(ONE, asking=False)), False, True),
        (Speak(WaitingForYou(ONE, asking=True)), False, True),
        (Speak(Expired(ONE, ASKED)), True, True),
        (Narrate(Asking(ONE, RequestId("r"), ASKED)), True, True),
    ],
)
def test_only_a_watched_session_s_waiting_is_passed_on_and_what_it_asks_always_is(heard: Heard, normal: bool, watched: bool) -> None:
    assert (routed(heard, "normal"), routed(heard, "watched")) == (normal, watched)


def test_an_overlay_holds_across_a_restart_and_a_session_never_set_is_normal(tmp_path: Path) -> None:
    Overlays(Home(tmp_path)).set(ONE, "watched")
    restarted = Overlays(Home(tmp_path))
    assert (restarted.of(ONE), restarted.of(SessionId("other"))) == ("watched", "normal")
    restarted.set(ONE, "normal")
    assert Overlays(Home(tmp_path)).of(ONE) == "normal"


def test_an_overlay_file_holding_anything_else_is_refused(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.overlays.mkdir()
    home.overlay(ONE).write_text("loud\n")
    with pytest.raises(Rejected):
        Overlays(home).of(ONE)


async def test_a_stop_is_announced_for_the_session_the_user_asked_about_and_not_for_the_other(tmp_path: Path) -> None:
    watched, other, dropped = membership(tmp_path, "watched"), membership(tmp_path, "other"), membership(tmp_path, "dropped")
    overlays = Overlays(Home(tmp_path / "home"))
    entries: list[Entry] = []
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=entries.append)
    for each in (watched, other, dropped):
        await sessions.apply(Joined(each, "startup"))
        await sessions.apply(StatusReported(each.id, Report(Idle(), Stamp(1)), at=1.0))
    watching = watch_session_tool(sessions, overlays).body
    assert await watching(session=watched.id, watch=True) == {"readback": "I'll tell you when watched is waiting for you."}
    await watching(session=dropped.id, watch=True)
    assert await watching(session=dropped.id, watch=False) == {"readback": "I won't tell you when dropped stops, only when it asks you something."}

    queued: list[Frame] = []

    async def queue_frame(frame: Frame) -> None:
        queued.append(frame)

    relaying = asyncio.create_task(relay(sessions, overlays, Pushed(), queue_frame, entries.append))
    try:
        for each in (watched, other, dropped):
            await sessions.apply(Waited(each.id))
        await _routed(entries, 3)
    finally:
        relaying.cancel()

    assert [frame.text for frame in queued if isinstance(frame, TTSSpeakFrame)] == ["watched is waiting for you."]
    assert [(entry.heard, entry.overlay, entry.passed) for entry in entries if isinstance(entry, Routed)] == [
        (Speak(WaitingForYou(watched.id, asking=False)), "watched", True),
        (Speak(WaitingForYou(other.id, asking=False)), "normal", False),
        (Speak(WaitingForYou(dropped.id, asking=False)), "normal", False),
    ]


async def test_a_session_that_is_not_running_cannot_be_watched(tmp_path: Path) -> None:
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    result = await watch_session_tool(sessions, Overlays(Home(tmp_path))).body(session="../escape", watch=True)
    assert "error" in result
    assert not (tmp_path / "overlays").exists()


async def _routed(entries: list[Entry], count: int) -> None:
    """Until the relay has routed `count` things: it runs as its own task, as in the daemon."""
    async with asyncio.timeout(2):
        while sum(isinstance(entry, Routed) for entry in entries) < count:
            await asyncio.sleep(0)
