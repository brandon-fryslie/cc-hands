"""Commands and interrupts as a table: the session as the registry holds it, the request, and what came of it. No I/O."""

from dataclasses import replace
from pathlib import Path

import pytest

from hands.core.effects import Command, Fritter, Key, Type
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
    Told,
    Turn,
    Unreported,
    Running,
)
from hands.core import status
from hands.core.status import Busy, Going, Shell, Stamp, UnknownReason, Waiting
from hands.core.tmux import NotInTmux, Pane

SOCKET = Path("/tmp/fritter-1/session.sock")
ONE = Membership(SessionId("s1"), pid=1, cwd=Path("/code/a"), transcript=Path("/t/s1.jsonl"), fritter=SOCKET)
FRITTER = Fritter(SOCKET, 1)
PANE = Pane(Path("/tmp/tmux-501/default"), "%3", "work", 1)
COMPACT = Command(CommandName("compact"), None)


def running(going: Going = Busy()) -> Running:
    return Running(going, Stamp(1), idled=Stamp(1))


AT_DIALOG = running(Waiting("permission prompt"))


IDLE = Idle(status.Idle(), Stamp(1), after=None)
# At its prompt with a background shell command still running.
SHELLING = Idle(Shell(), Stamp(1), after=None)


def registry(state: SessionState, member: Membership = ONE, turn: Turn = Told()) -> Registry:
    return Registry(permission_deadline=60.0, sessions={ONE.id: Session(member, state, mode=None, turn=turn)}, drafts={})


def gone(member: Membership = ONE) -> Registry:
    return Registry(permission_deadline=60.0, sessions={ONE.id: Gone(member)}, drafts={})


@pytest.mark.parametrize("state", [IDLE, running(), SHELLING, Unreported()])
def test_a_command_is_typed_as_itself_whatever_the_session_is_doing(state: SessionState) -> None:
    assert decide(registry(state), SendCommand(ONE.id, COMPACT), NotInTmux()) == Type(ONE.id, FRITTER, COMPACT)


@pytest.mark.parametrize("state", [AT_DIALOG, running(Waiting(UnknownReason("a dialog this version does not know")))])
def test_a_session_at_a_dialog_is_sent_no_command(state: SessionState) -> None:
    assert decide(registry(state), SendCommand(ONE.id, COMPACT), NotInTmux()) == AtItsDialog(ONE.id)


@pytest.mark.parametrize("state", [running(), AT_DIALOG, Unreported()])
def test_an_interrupt_presses_escape_even_at_a_dialog(state: SessionState) -> None:
    assert decide(registry(state), Interrupt(ONE.id), NotInTmux()) == Type(ONE.id, FRITTER, Key("escape"))


@pytest.mark.parametrize("state", [IDLE, SHELLING])
@pytest.mark.parametrize("turn", [Told(), Opened(PromptId("p1"))])
def test_a_session_at_its_prompt_has_nothing_to_interrupt_whatever_turn_was_heard(state: SessionState, turn: Turn) -> None:
    """Claude Code's status alone says whether anything runs: it is set busy before a prompt's hooks run. A shell
    command running in the background is no turn, and Escape at the prompt does not stop it."""
    assert decide(registry(state, turn=turn), Interrupt(ONE.id), NotInTmux()) == NothingRunning(ONE.id)


@pytest.mark.parametrize("request_", [SendCommand(ONE.id, COMPACT), Interrupt(ONE.id)])
def test_an_ended_session_is_typed_nothing_wrapped_or_not(request_: KeyboardRequest) -> None:
    assert decide(gone(), request_, NotInTmux()) == SessionEnded(ONE.id)
    assert decide(gone(replace(ONE, fritter=None)), request_, NotInTmux()) == SessionEnded(ONE.id)


@pytest.mark.parametrize("request_", [SendCommand(ONE.id, COMPACT), Interrupt(ONE.id)])
def test_a_session_nobody_wrapped_is_refused_by_name(request_: KeyboardRequest) -> None:
    assert decide(registry(running(), replace(ONE, fritter=None)), request_, NotInTmux()) == Unwrapped(ONE.id, NotInTmux())


def test_a_session_nobody_wrapped_is_typed_into_through_its_tmux_pane() -> None:
    unwrapped = registry(running(), replace(ONE, fritter=None))
    assert decide(unwrapped, SendCommand(ONE.id, COMPACT), PANE) == Type(ONE.id, PANE, COMPACT)
    assert decide(unwrapped, Interrupt(ONE.id), PANE) == Type(ONE.id, PANE, Key("escape"))


def test_a_session_in_a_pane_at_a_dialog_is_sent_no_command_and_one_at_its_prompt_no_interrupt() -> None:
    assert decide(registry(AT_DIALOG, replace(ONE, fritter=None)), SendCommand(ONE.id, COMPACT), PANE) == AtItsDialog(ONE.id)
    assert decide(registry(IDLE, replace(ONE, fritter=None)), Interrupt(ONE.id), PANE) == NothingRunning(ONE.id)


@pytest.mark.parametrize("request_", [SendCommand(SessionId("s2"), COMPACT), Interrupt(SessionId("s2"))])
def test_a_session_that_never_joined_is_named_unknown(request_: KeyboardRequest) -> None:
    assert decide(registry(IDLE), request_, NotInTmux()) == UnknownSession(SessionId("s2"))


def test_a_command_is_typed_with_its_slash_and_its_arguments_behind_a_space() -> None:
    assert COMPACT.typed == "/compact"
    assert Command(CommandName("model"), PromptText("opus")).typed == "/model opus"
