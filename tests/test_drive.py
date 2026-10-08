"""Drives as a table: registry before, request, and what came of it. No I/O."""

from dataclasses import replace
from pathlib import Path

import pytest

from hands.core.attention import Attention, Overlay, Spoken, Steering, Withheld, delivery
from hands.core.drive import SENDS, DriveSending, DriveStopped, Driving, LastSend, NotDriven, Send, StartDrive, StopDrive, decide
from hands.core.effects import Fritter, Text, Type
from hands.core.events import Died
from hands.core.reach import AtItsDialog, SessionEnded, UnknownSession, Unwrapped
from hands.core.reducer import reduce
from hands.core.session import Drive, Gone, Idle, Membership, PromptText, Registry, Running, Session, SessionId, SessionState, Staged
from hands.core import status
from hands.core.status import Busy, Stamp, Waiting
from hands.core.tmux import NotInTmux, Pane

ONE = Membership(SessionId("s1"), pid=1, cwd=Path("/code/a"), transcript=Path("/t/s1.jsonl"))
WRAPPED = replace(ONE, fritter=Path("/tmp/fritter-1/session.sock"))
ORDER = "keep fixing tests until they all pass"
NEXT = PromptText("fix the two failing parser tests")
PANE = Pane(Path("/tmp/tmux-501/default"), "%1", "work", 0)
IDLE = Idle(status.Idle(), Stamp(1), after=None)
AT_DIALOG = Running(Waiting("permission prompt"), Stamp(1), idled=Stamp(1))
FRITTER = Fritter(Path("/tmp/fritter-1/session.sock"), 1)
DOCS_DRAFT = Staged(PromptText("fix the broken links"), ())


def registry(state: SessionState = IDLE, drives: dict[SessionId, Drive] | None = None, member: Membership = WRAPPED) -> Registry:
    return Registry(permission_deadline=60.0, sessions={ONE.id: Session(member, state, mode=None)}, drafts={}, drives=drives or {})


def driven(sends: int = 0, state: SessionState = IDLE) -> Registry:
    return registry(state, {ONE.id: Drive(ORDER, sends)})


def test_starting_a_drive_holds_the_order() -> None:
    assert decide(registry(), StartDrive(ONE.id, ORDER)) == (driven(), Driving(ONE.id, Drive(ORDER, 0), replaced=None))


def test_a_new_order_replaces_the_old_and_counts_afresh() -> None:
    assert decide(driven(3), StartDrive(ONE.id, "ship it")) == (registry(drives={ONE.id: Drive("ship it", 0)}), Driving(ONE.id, Drive("ship it", 0), replaced=Drive(ORDER, 3)))


def test_stopping_lets_the_order_go() -> None:
    assert decide(driven(2), StopDrive(ONE.id)) == (registry(), DriveStopped(ONE.id, Drive(ORDER, 2)))


@pytest.mark.parametrize("request_", [StopDrive(ONE.id), DriveSending(ONE.id, NEXT, PANE)])
def test_a_session_never_driven_is_refused_and_nothing_is_typed(request_: StopDrive | DriveSending) -> None:
    assert decide(registry(), request_) == (registry(), NotDriven(ONE.id))


def test_a_send_under_a_drive_types_the_prompt_and_counts_it() -> None:
    assert decide(driven(), DriveSending(ONE.id, NEXT, PANE)) == (driven(1), Send(Type(ONE.id, FRITTER, Text(NEXT))))


def test_the_send_that_spends_the_order_ends_the_drive() -> None:
    assert decide(driven(SENDS - 1), DriveSending(ONE.id, NEXT, PANE)) == (registry(), LastSend(Type(ONE.id, FRITTER, Text(NEXT)), Drive(ORDER, SENDS)))


def test_a_driven_session_at_a_dialog_is_not_typed_into() -> None:
    assert decide(driven(state=AT_DIALOG), DriveSending(ONE.id, NEXT, PANE)) == (driven(state=AT_DIALOG), AtItsDialog(ONE.id))


def test_a_driven_session_with_no_way_to_type_into_it_says_why() -> None:
    before = registry(drives={ONE.id: Drive(ORDER, 0)}, member=ONE)
    assert decide(before, DriveSending(ONE.id, NEXT, NotInTmux())) == (before, Unwrapped(ONE.id, NotInTmux()))


def test_a_send_never_touches_a_staged_draft() -> None:
    before = replace(driven(), drafts={ONE.id: DOCS_DRAFT})
    after, _ = decide(before, DriveSending(ONE.id, NEXT, PANE))
    assert after.drafts == {ONE.id: DOCS_DRAFT}


def test_an_unknown_session_cannot_be_driven() -> None:
    empty = Registry(permission_deadline=60.0, sessions={}, drafts={})
    assert decide(empty, StartDrive(ONE.id, ORDER)) == (empty, UnknownSession(ONE.id))


def test_an_ended_session_cannot_be_driven() -> None:
    ended = Registry(permission_deadline=60.0, sessions={ONE.id: Gone(WRAPPED)}, drafts={})
    assert decide(ended, StartDrive(ONE.id, ORDER)) == (ended, SessionEnded(ONE.id))


def test_a_drive_ends_with_its_session() -> None:
    after, _ = reduce(driven(state=Running(Busy(), Stamp(1), idled=Stamp(1))), Died(WRAPPED))
    assert after.drives == {}


@pytest.mark.parametrize("attention", [Attention(), Attention(quiet="on"), Attention(finished="full")])
@pytest.mark.parametrize("overlay", ["normal", "watched", "muted"])
def test_a_driven_turn_reaches_the_brain_whatever_is_set_for_the_ear(attention: Attention, overlay: Overlay) -> None:
    assert delivery(attention, overlay, Drive(ORDER, 1)) == Steering(Drive(ORDER, 1))


def test_without_a_drive_the_settings_decide_as_before() -> None:
    assert delivery(Attention(), "normal", None) == Withheld("off")
    assert delivery(Attention(finished="brief"), "normal", None) == Spoken("brief", "finished")
