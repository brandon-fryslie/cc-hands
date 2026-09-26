"""Commands and interrupts as a table: the session as the registry holds it, the request, and what came of it. No I/O."""

from dataclasses import replace
from pathlib import Path

import pytest

from hands.core.effects import Command, Key, Type
from hands.core.keyboard import Interrupt, KeyboardRequest, NothingRunning, SendCommand, decide
from hands.core.reach import AtItsDialog, SessionEnded, UnknownSession, Unwrapped
from hands.core.session import (
    AtDialog,
    Blocked,
    CommandName,
    Gone,
    Idle,
    Membership,
    Permission,
    PromptText,
    Registry,
    RequestId,
    Session,
    SessionId,
    SessionState,
    Submitted,
    Working,
)

SOCKET = Path("/tmp/fritter-1/session.sock")
ONE = Membership(SessionId("s1"), pid=1, cwd=Path("/code/a"), transcript=Path("/t/s1.jsonl"), fritter=SOCKET)
COMPACT = Command(CommandName("compact"), None)
BLOCKED = Blocked(on=Permission(tool="Bash", input={"command": "ls"}), request=RequestId("r"), deadline=9.0, warned=False)


def registry(state: SessionState, member: Membership = ONE) -> Registry:
    return Registry(permission_deadline=60.0, sessions={ONE.id: Session(member, state, mode=None, turn=None)}, drafts={})


@pytest.mark.parametrize("state", [Idle(), Submitted(since=1.0), Working(since=1.0)])
def test_a_command_is_typed_as_itself_whatever_the_session_is_doing(state: SessionState) -> None:
    assert decide(registry(state), SendCommand(ONE.id, COMPACT)) == Type(ONE.id, SOCKET, 1, COMPACT)


@pytest.mark.parametrize("state", [BLOCKED, AtDialog(on=BLOCKED.on)])
def test_a_session_at_a_dialog_is_sent_no_command(state: SessionState) -> None:
    assert decide(registry(state), SendCommand(ONE.id, COMPACT)) == AtItsDialog(ONE.id)


@pytest.mark.parametrize("state", [Submitted(since=1.0), Working(since=1.0), BLOCKED, AtDialog(on=BLOCKED.on)])
def test_an_interrupt_presses_escape_even_at_a_dialog(state: SessionState) -> None:
    assert decide(registry(state), Interrupt(ONE.id)) == Type(ONE.id, SOCKET, 1, Key("escape"))


def test_a_session_at_its_prompt_has_nothing_to_interrupt() -> None:
    assert decide(registry(Idle()), Interrupt(ONE.id)) == NothingRunning(ONE.id)


@pytest.mark.parametrize("request_", [SendCommand(ONE.id, COMPACT), Interrupt(ONE.id)])
def test_an_ended_session_is_typed_nothing_wrapped_or_not(request_: KeyboardRequest) -> None:
    assert decide(registry(Gone()), request_) == SessionEnded(ONE.id)
    assert decide(registry(Gone(), replace(ONE, fritter=None)), request_) == SessionEnded(ONE.id)


@pytest.mark.parametrize("request_", [SendCommand(ONE.id, COMPACT), Interrupt(ONE.id)])
def test_a_session_nobody_wrapped_is_refused_by_name(request_: KeyboardRequest) -> None:
    assert decide(registry(Working(since=1.0), replace(ONE, fritter=None)), request_) == Unwrapped(ONE.id)


@pytest.mark.parametrize("request_", [SendCommand(SessionId("s2"), COMPACT), Interrupt(SessionId("s2"))])
def test_a_session_that_never_joined_is_named_unknown(request_: KeyboardRequest) -> None:
    assert decide(registry(Idle()), request_) == UnknownSession(SessionId("s2"))


def test_a_command_is_typed_with_its_slash_and_its_arguments_behind_a_space() -> None:
    assert COMPACT.typed == "/compact"
    assert Command(CommandName("model"), PromptText("opus")).typed == "/model opus"
