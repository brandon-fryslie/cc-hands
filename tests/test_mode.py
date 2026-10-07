"""Mode readback: the permission_mode each hook reports is the session's mode, and list_sessions says it."""

from pathlib import Path
from typing import get_args

import pytest

from hands.core.effects import Allow
from hands.core.events import Joined, PermissionRequested, Prompted, Stopped
from hands.core.session import Membership, Mode, PromptId, Permission, PermissionMode, RequestId, SessionId, UnknownMode
from hands.sessions.registry import Sessions
from hands.voice.readback import spoken_mode
from hands.voice.tools import describe_listing

from hands.core.status import Stamp

# When hands heard a Stop, on the clock Claude Code stamps a status with.
STOP_HEARD = Stamp(1500)
STOP_REQUEST = RequestId("stop")

SID = SessionId("s1")
ONE = Membership(SID, pid=4242, cwd=Path("/code/a"), transcript=Path("/nonexistent/s1.jsonl"))


@pytest.mark.parametrize(
    ("mode", "said"),
    [
        ("default", "manual mode"),
        ("acceptEdits", "accept edits mode"),
        ("plan", "plan mode"),
        ("auto", "auto mode"),
        ("dontAsk", "don't ask mode"),
        ("bypassPermissions", "bypass permissions mode"),
        (UnknownMode("ultraplan"), "a mode hands does not know, named ultraplan"),
    ],
)
def test_each_mode_is_said_as_the_sessions_own_footer_names_it(mode: Mode, said: str) -> None:
    assert spoken_mode(mode) == said


def test_every_mode_claude_code_has_is_said_by_name() -> None:
    assert all(spoken_mode(mode) for mode in get_args(PermissionMode))


async def test_a_mode_changed_at_the_keyboard_is_listed_at_the_sessions_next_hook() -> None:
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(ONE, "startup"))
    assert describe_listing(sessions.live()[0])["mode"] == "not reported yet"
    await sessions.apply(Prompted(SID, at=1.0, mode="default", prompt=PromptId("p1")))
    await sessions.apply(Stopped(SID, "ok", mode="default", prompt=PromptId("p1"), again=False, heard=STOP_HEARD, request=STOP_REQUEST))
    assert describe_listing(sessions.live()[0])["mode"] == "manual mode"
    # Shift-tab at the prompt fires no hook; the next prompt reports where it landed.
    await sessions.apply(Prompted(SID, at=2.0, mode="acceptEdits", prompt=PromptId("p1")))
    assert describe_listing(sessions.live()[0])["mode"] == "accept edits mode"


async def test_a_voice_answer_keeps_the_mode_the_session_reported() -> None:
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(ONE, "startup"))
    await sessions.apply(Prompted(SID, at=1.0, mode="plan", prompt=PromptId("p1")))
    request = PermissionRequested(SID, at=2.0, request=RequestId("r1"), on=Permission("Bash", {}), mode="plan")
    await sessions.apply(request)
    await sessions.answer(RequestId("r1"), Allow())
    assert describe_listing(sessions.live()[0])["mode"] == "plan mode"
