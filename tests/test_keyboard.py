"""Commands and interrupts as a table: the session as the registry holds it, the request, and what came of it. No I/O."""

from dataclasses import replace
from pathlib import Path

import pytest

from hands.core.effects import Command, Key, Type
from hands.core.keyboard import Interrupt, KeyboardRequest, NothingRunning, SendCommand, decide
from hands.core.reach import AtItsDialog, SessionEnded, UnknownSession, Unwrapped
from hands.core.session import (
    CommandName,
    Gone,
    Idle,
    Membership,
    Opened,
    PromptId,
    PromptText,
    Registry,
    Session,
    SessionId,
    SessionState,
    Unreported,
    Running,
)
from hands.core.status import Busy, Going, Shell, Stamp, UnknownReason, Waiting

SOCKET = Path("/tmp/fritter-1/session.sock")
ONE = Membership(SessionId("s1"), pid=1, cwd=Path("/code/a"), transcript=Path("/t/s1.jsonl"), fritter=SOCKET)
COMPACT = Command(CommandName("compact"), None)


def running(going: Going = Busy()) -> Running:
    return Running(going, Stamp(1), idled=Stamp(1))


AT_DIALOG = running(Waiting("permission prompt"))


IDLE = Idle(Stamp(1), due=61.0, after=None)


def registry(state: SessionState, member: Membership = ONE) -> Registry:
    return Registry(permission_deadline=60.0, sessions={ONE.id: Session(member, state, mode=None)}, drafts={})


@pytest.mark.parametrize("state", [IDLE, running(), running(Shell()), Unreported()])
def test_a_command_is_typed_as_itself_whatever_the_session_is_doing(state: SessionState) -> None:
    assert decide(registry(state), SendCommand(ONE.id, COMPACT)) == Type(ONE.id, SOCKET, 1, COMPACT)


@pytest.mark.parametrize("state", [AT_DIALOG, running(Waiting(UnknownReason("a dialog this version does not know")))])
def test_a_session_at_a_dialog_is_sent_no_command(state: SessionState) -> None:
    assert decide(registry(state), SendCommand(ONE.id, COMPACT)) == AtItsDialog(ONE.id)


@pytest.mark.parametrize("state", [running(), AT_DIALOG, Unreported()])
def test_an_interrupt_presses_escape_even_at_a_dialog(state: SessionState) -> None:
    assert decide(registry(state), Interrupt(ONE.id)) == Type(ONE.id, SOCKET, 1, Key("escape"))


def test_a_session_at_its_prompt_has_nothing_to_interrupt() -> None:
    assert decide(registry(IDLE), Interrupt(ONE.id)) == NothingRunning(ONE.id)


def test_a_prompt_still_in_its_hooks_is_interrupted_though_its_session_is_read_idle() -> None:
    before = Registry(permission_deadline=60.0, sessions={ONE.id: Session(ONE, IDLE, mode=None, turn=Opened(PromptId("p1")))}, drafts={})
    assert decide(before, Interrupt(ONE.id)) == Type(ONE.id, SOCKET, 1, Key("escape"))


@pytest.mark.parametrize("request_", [SendCommand(ONE.id, COMPACT), Interrupt(ONE.id)])
def test_an_ended_session_is_typed_nothing_wrapped_or_not(request_: KeyboardRequest) -> None:
    assert decide(registry(Gone()), request_) == SessionEnded(ONE.id)
    assert decide(registry(Gone(), replace(ONE, fritter=None)), request_) == SessionEnded(ONE.id)


@pytest.mark.parametrize("request_", [SendCommand(ONE.id, COMPACT), Interrupt(ONE.id)])
def test_a_session_nobody_wrapped_is_refused_by_name(request_: KeyboardRequest) -> None:
    assert decide(registry(running(), replace(ONE, fritter=None)), request_) == Unwrapped(ONE.id)


@pytest.mark.parametrize("request_", [SendCommand(SessionId("s2"), COMPACT), Interrupt(SessionId("s2"))])
def test_a_session_that_never_joined_is_named_unknown(request_: KeyboardRequest) -> None:
    assert decide(registry(IDLE), request_) == UnknownSession(SessionId("s2"))


def test_a_command_is_typed_with_its_slash_and_its_arguments_behind_a_space() -> None:
    assert COMPACT.typed == "/compact"
    assert Command(CommandName("model"), PromptText("opus")).typed == "/model opus"
