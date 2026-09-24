"""Mode readback: the permission_mode each hook reports is the session's mode, list_sessions says it, and a change reaches the intermediary unspoken."""

from pathlib import Path
from typing import cast

import pytest
from pipecat.frames.frames import LLMMessagesAppendFrame

from hands.core.effects import Allow, ModeChanged, Note
from hands.core.events import Joined, PermissionRequested, Prompted, Stopped
from hands.core.session import Membership, Mode, Permission, RequestId, SessionId, UnknownMode
from hands.sessions.registry import Sessions
from hands.voice.readback import spoken_mode
from hands.voice.speech import frame
from hands.voice.tools import describe_listing

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


def test_a_mode_change_is_put_in_the_context_without_asking_the_model_to_speak() -> None:
    noted = frame(Note(ModeChanged(SID, "acceptEdits")), names=lambda _: "auth refactor")
    assert isinstance(noted, LLMMessagesAppendFrame) and noted.run_llm is False
    [message] = noted.messages
    assert "auth refactor is now in accept edits mode" in str(cast(dict[str, object], message)["content"])


async def test_a_mode_changed_at_the_keyboard_is_listed_and_noted_at_the_sessions_next_hook() -> None:
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(ONE, "startup"))
    assert describe_listing(sessions.live()[0])["mode"] == "not reported yet"
    await sessions.apply(Prompted(SID, at=1.0, mode="default"))
    await sessions.apply(Stopped(SID, "ok", mode="default"))
    assert describe_listing(sessions.live()[0])["mode"] == "manual mode"
    # Shift-tab at the prompt fires no hook; the next prompt reports where it landed.
    await sessions.apply(Prompted(SID, at=2.0, mode="acceptEdits"))
    assert describe_listing(sessions.live()[0])["mode"] == "accept edits mode"
    assert await sessions.heard() == Note(ModeChanged(SID, "acceptEdits"))


async def test_a_voice_answer_keeps_the_mode_the_session_reported() -> None:
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(ONE, "startup"))
    await sessions.apply(Prompted(SID, at=1.0, mode="plan"))
    request = PermissionRequested(SID, at=2.0, request=RequestId("r1"), on=Permission("Bash", {}), mode="plan")
    await sessions.apply(request)
    await sessions.answer(RequestId("r1"), Allow())
    assert describe_listing(sessions.live()[0])["mode"] == "plan mode"
