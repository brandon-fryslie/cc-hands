"""Drafts as a table: registry before, request, and what came of it. No I/O."""

from dataclasses import replace
from pathlib import Path

import pytest

from hands.core.drafts import (
    AmendDraft,
    DiscardDraft,
    DraftAmended,
    DraftDiscarded,
    DraftRequest,
    DraftSent,
    DraftStaged,
    MaybeSent,
    NothingStaged,
    NotSent,
    SendDraft,
    SentBefore,
    SessionEnded,
    StageDraft,
    StillSending,
    UnknownSession,
    Unwrapped,
    AtItsDialog,
    decide,
    land,
)
from hands.core.effects import Landed, Landing, MaybeTyped, NotTyped, Text, Type
from hands.core.session import (
    AtDialog,
    Blocked,
    Draft,
    Gone,
    Idle,
    Membership,
    Permission,
    PromptText,
    Registry,
    RequestId,
    Resolution,
    Sending,
    Session,
    SessionId,
    SessionState,
    Staged,
    Submitted,
    Unsure,
    Working,
)

ONE = Membership(SessionId("s1"), pid=1, cwd=Path("/code/a"), transcript=Path("/t/s1.jsonl"))
TWO = Membership(SessionId("s2"), pid=2, cwd=Path("/code/b"), transcript=Path("/t/s2.jsonl"))
FIX = Staged(PromptText("/fix the auth middleware"), (Resolution("auth middleware", "authMiddleware.ts"),))
BETTER = Staged(PromptText("fix the token helper"), ())
BLOCKED = Blocked(on=Permission(tool="Bash", input={"command": "ls"}), request=RequestId("r"), deadline=9.0, warned=False)


SOCKET = Path("/tmp/fritter-1/session.sock")
WRAPPED = replace(ONE, fritter=SOCKET)
TYPE_FIX = Type(ONE.id, SOCKET, pid=1, input=Text(FIX.text))


def registry(state: SessionState = Idle(), drafts: dict[SessionId, Draft] | None = None, member: Membership = ONE) -> Registry:
    return Registry(permission_deadline=60.0, sessions={ONE.id: Session(member, state, mode=None, turn=None)}, drafts=drafts or {})


def staged(state: SessionState = Idle()) -> Registry:
    return registry(state, {ONE.id: FIX})


def wrapped(state: SessionState = Idle(), draft: Draft = FIX) -> Registry:
    return registry(state, {ONE.id: draft}, member=WRAPPED)


@pytest.mark.parametrize("state", [Idle(), Working(since=1.0), BLOCKED])
def test_staging_holds_the_draft(state: SessionState) -> None:
    assert decide(registry(state), StageDraft(ONE.id, FIX)) == (staged(state), DraftStaged(ONE.id, FIX, replaced=None))


def test_staging_over_a_draft_replaces_it_and_says_so() -> None:
    assert decide(staged(), StageDraft(ONE.id, BETTER)) == (registry(drafts={ONE.id: BETTER}), DraftStaged(ONE.id, BETTER, replaced=FIX))


def test_amending_replaces_the_draft_and_keeps_both_versions_for_the_readback() -> None:
    assert decide(staged(), AmendDraft(ONE.id, BETTER)) == (registry(drafts={ONE.id: BETTER}), DraftAmended(ONE.id, before=FIX, after=BETTER))


@pytest.mark.parametrize("request_", [AmendDraft(ONE.id, BETTER), DiscardDraft(ONE.id)])
def test_with_nothing_staged_there_is_nothing_to_amend_or_discard(request_: DraftRequest) -> None:
    assert decide(registry(), request_) == (registry(), NothingStaged(ONE.id))


def test_discarding_clears_the_draft_even_after_the_session_ended() -> None:
    assert decide(staged(Gone()), DiscardDraft(ONE.id)) == (registry(Gone()), DraftDiscarded(ONE.id, FIX))


@pytest.mark.parametrize("request_", [StageDraft(ONE.id, BETTER), AmendDraft(ONE.id, BETTER)])
def test_an_ended_session_takes_no_draft(request_: DraftRequest) -> None:
    assert decide(staged(Gone()), request_) == (staged(Gone()), SessionEnded(ONE.id))


@pytest.mark.parametrize("request_", [StageDraft(TWO.id, FIX), AmendDraft(TWO.id, FIX), DiscardDraft(TWO.id)])
def test_a_session_that_never_joined_is_named_unknown(request_: DraftRequest) -> None:
    assert decide(staged(), request_) == (staged(), UnknownSession(TWO.id))


def test_a_discard_touches_only_its_own_session() -> None:
    both = Registry(
        permission_deadline=60.0,
        sessions={ONE.id: Session(ONE, Idle(), mode=None, turn=None), TWO.id: Session(TWO, Idle(), mode=None, turn=None)},
        drafts={ONE.id: FIX, TWO.id: BETTER},
    )
    after, _ = decide(both, DiscardDraft(TWO.id))
    assert after.drafts == {ONE.id: FIX}


def test_text_is_typed_behind_a_space_so_a_leading_sigil_is_read_as_text() -> None:
    assert Text(PromptText("/fix it")).typed == " /fix it"


@pytest.mark.parametrize("state", [Idle(), Submitted(since=1.0), Working(since=1.0)])
def test_a_send_types_the_draft_into_the_fritter_that_wrapped_its_session_and_holds_it_meanwhile(state: SessionState) -> None:
    assert decide(wrapped(state), SendDraft(ONE.id)) == (wrapped(state, Sending(FIX)), TYPE_FIX)


def test_a_session_nobody_wrapped_is_refused_by_name_and_keeps_its_draft() -> None:
    assert decide(staged(), SendDraft(ONE.id)) == (staged(), Unwrapped(ONE.id))


@pytest.mark.parametrize("state", [BLOCKED, AtDialog(on=BLOCKED.on)])
def test_a_session_at_a_dialog_is_sent_nothing_because_the_dialog_would_take_it_as_an_answer(state: SessionState) -> None:
    assert decide(wrapped(state), SendDraft(ONE.id)) == (wrapped(state), AtItsDialog(ONE.id))


def test_an_ended_session_is_sent_nothing() -> None:
    assert decide(wrapped(Gone()), SendDraft(ONE.id)) == (wrapped(Gone()), SessionEnded(ONE.id))


def test_with_nothing_staged_there_is_nothing_to_send() -> None:
    assert decide(registry(member=WRAPPED), SendDraft(ONE.id)) == (registry(member=WRAPPED), NothingStaged(ONE.id))


@pytest.mark.parametrize("request_", [StageDraft(ONE.id, BETTER), AmendDraft(ONE.id, BETTER), DiscardDraft(ONE.id), SendDraft(ONE.id)])
def test_a_draft_being_sent_is_left_alone_until_the_send_lands(request_: DraftRequest) -> None:
    sending = wrapped(draft=Sending(FIX))
    assert decide(sending, request_) == (sending, StillSending(ONE.id))


@pytest.mark.parametrize(
    ("landing", "after", "outcome"),
    [
        (Landed(), None, DraftSent(ONE.id, FIX)),
        (NotTyped("no"), FIX, NotSent(ONE.id, "no")),
        (MaybeTyped("half"), Unsure(FIX, "half"), MaybeSent(ONE.id, "half")),
    ],
)
def test_what_came_of_the_typing_decides_whether_the_draft_is_gone_sendable_or_held_back(
    landing: Landing, after: Draft | None, outcome: object
) -> None:
    left = registry(member=WRAPPED) if after is None else wrapped(draft=after)
    assert land(wrapped(draft=Sending(FIX)), TYPE_FIX, landing) == (left, outcome)


def test_a_draft_that_may_already_be_in_the_session_is_not_sent_again() -> None:
    unsure = wrapped(draft=Unsure(FIX, "half"))
    assert decide(unsure, SendDraft(ONE.id)) == (unsure, SentBefore(ONE.id, "half"))


@pytest.mark.parametrize(
    ("request_", "outcome"),
    [
        (StageDraft(ONE.id, BETTER), DraftStaged(ONE.id, BETTER, replaced=FIX)),
        (AmendDraft(ONE.id, BETTER), DraftAmended(ONE.id, before=FIX, after=BETTER)),
    ],
)
def test_staging_or_amending_a_draft_that_may_have_been_sent_makes_it_sendable_again(request_: DraftRequest, outcome: object) -> None:
    after, said = decide(wrapped(draft=Unsure(FIX, "half")), request_)
    assert (after, said) == (wrapped(draft=BETTER), outcome)
    assert decide(after, SendDraft(ONE.id))[1] == replace(TYPE_FIX, input=Text(BETTER.text))


def test_a_draft_that_may_have_been_sent_can_be_discarded() -> None:
    assert decide(wrapped(draft=Unsure(FIX, "half")), DiscardDraft(ONE.id)) == (registry(member=WRAPPED), DraftDiscarded(ONE.id, FIX))


def test_a_landing_for_a_draft_not_being_sent_is_a_bug_said_out_loud() -> None:
    with pytest.raises(RuntimeError, match="not being sent"):
        land(wrapped(), TYPE_FIX, Landed())
