"""Drafts as a table: registry before, request, and what came of it. No I/O."""

from dataclasses import replace
from pathlib import Path

import pytest

from hands.core.drafts import (
    AmendDraft,
    Decidable,
    DiscardDraft,
    DraftAmended,
    DraftDiscarded,
    DraftStaged,
    NothingStaged,
    Sending,
    StageDraft,
    decide,
)
from hands.core.effects import Fritter, Text, Type
from hands.core.reach import AtItsDialog, SessionEnded, UnknownSession, Unwrapped
from hands.core.session import (
    Gone,
    Idle,
    Membership,
    PromptText,
    Registry,
    Resolution,
    Session,
    SessionId,
    SessionState,
    Staged,
    Unreported,
    Running,
)
from hands.core import status
from hands.core.status import Busy, Going, Shell, Stamp, UnknownReason, Waiting
from hands.core.tmux import NotInTmux, Pane

ONE = Membership(SessionId("s1"), pid=1, cwd=Path("/code/a"), transcript=Path("/t/s1.jsonl"))
TWO = Membership(SessionId("s2"), pid=2, cwd=Path("/code/b"), transcript=Path("/t/s2.jsonl"))
FIX = Staged(PromptText("/fix the auth middleware"), (Resolution("auth middleware", "authMiddleware.ts"),))
BETTER = Staged(PromptText("fix the token helper"), ())


WRAPPED = replace(ONE, fritter=Path("/tmp/fritter-1/session.sock"))


def running(going: Going = Busy()) -> Running:
    return Running(going, Stamp(1), idled=Stamp(1))


AT_DIALOG = running(Waiting("permission prompt"))


IDLE = Idle(status.Idle(), Stamp(1), after=None)
# At its prompt with a background shell command still running.
SHELLING = Idle(Shell(), Stamp(1), after=None)


def registry(state: SessionState = IDLE, drafts: dict[SessionId, Staged] | None = None, member: Membership = ONE) -> Registry:
    return Registry(permission_deadline=60.0, sessions={ONE.id: Session(member, state, mode=None)}, drafts=drafts or {})


def gone(drafts: dict[SessionId, Staged] | None = None, member: Membership = ONE) -> Registry:
    return Registry(permission_deadline=60.0, sessions={ONE.id: Gone(member)}, drafts=drafts or {})


def staged(state: SessionState = IDLE) -> Registry:
    return registry(state, {ONE.id: FIX})


@pytest.mark.parametrize("state", [IDLE, running(), AT_DIALOG])
def test_staging_holds_the_draft(state: SessionState) -> None:
    assert decide(registry(state), StageDraft(ONE.id, FIX)) == (staged(state), DraftStaged(ONE.id, FIX, replaced=None))


def test_staging_over_a_draft_replaces_it_and_says_so() -> None:
    assert decide(staged(), StageDraft(ONE.id, BETTER)) == (registry(drafts={ONE.id: BETTER}), DraftStaged(ONE.id, BETTER, replaced=FIX))


def test_amending_replaces_the_draft_and_keeps_both_versions_for_the_readback() -> None:
    assert decide(staged(), AmendDraft(ONE.id, BETTER)) == (registry(drafts={ONE.id: BETTER}), DraftAmended(ONE.id, before=FIX, after=BETTER))


@pytest.mark.parametrize("request_", [AmendDraft(ONE.id, BETTER), DiscardDraft(ONE.id)])
def test_with_nothing_staged_there_is_nothing_to_amend_or_discard(request_: Decidable) -> None:
    assert decide(registry(), request_) == (registry(), NothingStaged(ONE.id))


def test_discarding_clears_the_draft_even_after_the_session_ended() -> None:
    assert decide(gone({ONE.id: FIX}), DiscardDraft(ONE.id)) == (gone(), DraftDiscarded(ONE.id, FIX))


@pytest.mark.parametrize("request_", [StageDraft(ONE.id, BETTER), AmendDraft(ONE.id, BETTER)])
def test_an_ended_session_takes_no_draft(request_: Decidable) -> None:
    assert decide(gone({ONE.id: FIX}), request_) == (gone({ONE.id: FIX}), SessionEnded(ONE.id))


@pytest.mark.parametrize("request_", [StageDraft(TWO.id, FIX), AmendDraft(TWO.id, FIX), DiscardDraft(TWO.id)])
def test_a_session_that_never_joined_is_named_unknown(request_: Decidable) -> None:
    assert decide(staged(), request_) == (staged(), UnknownSession(TWO.id))


def test_a_discard_touches_only_its_own_session() -> None:
    both = Registry(
        permission_deadline=60.0,
        sessions={ONE.id: Session(ONE, IDLE, mode=None), TWO.id: Session(TWO, IDLE, mode=None)},
        drafts={ONE.id: FIX, TWO.id: BETTER},
    )
    after, _ = decide(both, DiscardDraft(TWO.id))
    assert after.drafts == {ONE.id: FIX}


def wrapped(state: SessionState = IDLE, drafts: dict[SessionId, Staged] | None = None) -> Registry:
    return registry(state, drafts, member=WRAPPED)


@pytest.mark.parametrize("state", [IDLE, running(), SHELLING, Unreported()])
def test_a_send_is_typed_into_the_fritter_that_wrapped_the_session_and_the_draft_is_gone_at_once(state: SessionState) -> None:
    typed = Type(ONE.id, Fritter(Path("/tmp/fritter-1/session.sock"), pid=1), input=Text(FIX.text))
    assert decide(wrapped(state, {ONE.id: FIX}), Sending(ONE.id, NotInTmux())) == (wrapped(state), typed)


def test_text_is_typed_behind_a_space_so_a_leading_sigil_is_read_as_text() -> None:
    assert Text(FIX.text).typed == " /fix the auth middleware"


def test_a_session_nobody_wrapped_is_refused_by_name_and_keeps_its_draft() -> None:
    assert decide(staged(), Sending(ONE.id, NotInTmux())) == (staged(), Unwrapped(ONE.id, NotInTmux()))


@pytest.mark.parametrize("state", [IDLE, running(), Unreported()])
def test_a_send_to_a_session_nobody_wrapped_is_typed_into_its_tmux_pane_and_the_draft_is_gone_at_once(state: SessionState) -> None:
    pane = Pane(Path("/tmp/tmux-501/default"), "%3", "work", 1)
    assert decide(registry(state, {ONE.id: FIX}), Sending(ONE.id, pane)) == (registry(state), Type(ONE.id, pane, Text(FIX.text)))


@pytest.mark.parametrize("state", [AT_DIALOG, running(Waiting(UnknownReason("a dialog this version does not know")))])
def test_a_session_at_a_dialog_is_sent_nothing_and_keeps_its_draft(state: SessionState) -> None:
    before = wrapped(state, {ONE.id: FIX})
    assert decide(before, Sending(ONE.id, NotInTmux())) == (before, AtItsDialog(ONE.id))


def test_an_ended_session_is_sent_nothing() -> None:
    before = gone({ONE.id: FIX}, WRAPPED)
    assert decide(before, Sending(ONE.id, NotInTmux())) == (before, SessionEnded(ONE.id))


def test_with_nothing_staged_there_is_nothing_to_send() -> None:
    assert decide(wrapped(), Sending(ONE.id, NotInTmux())) == (wrapped(), NothingStaged(ONE.id))
