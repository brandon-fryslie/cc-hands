"""Drafts as a table: registry before, request, and what came of it. No I/O."""

from pathlib import Path

import pytest

from hands.core.drafts import (
    AmendDraft,
    DiscardDraft,
    DraftAmended,
    DraftDiscarded,
    DraftRequest,
    DraftStaged,
    NothingStaged,
    SessionEnded,
    StageDraft,
    UnknownSession,
    decide,
)
from hands.core.session import (
    Blocked,
    Gone,
    Idle,
    Membership,
    Permission,
    PromptText,
    Registry,
    RequestId,
    Resolution,
    Session,
    SessionId,
    SessionState,
    Staged,
    Working,
)

ONE = Membership(SessionId("s1"), pid=1, cwd=Path("/code/a"), transcript=Path("/t/s1.jsonl"))
TWO = Membership(SessionId("s2"), pid=2, cwd=Path("/code/b"), transcript=Path("/t/s2.jsonl"))
FIX = Staged(PromptText("/fix the auth middleware"), (Resolution("auth middleware", "authMiddleware.ts"),))
BETTER = Staged(PromptText("fix the token helper"), ())
BLOCKED = Blocked(on=Permission(tool="Bash", input={"command": "ls"}), request=RequestId("r"), deadline=9.0, warned=False)


def registry(state: SessionState = Idle(), drafts: dict[SessionId, Staged] | None = None) -> Registry:
    return Registry(permission_deadline=60.0, sessions={ONE.id: Session(ONE, state)}, drafts=drafts or {})


def staged(state: SessionState = Idle()) -> Registry:
    return registry(state, {ONE.id: FIX})


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
        sessions={ONE.id: Session(ONE, Idle()), TWO.id: Session(TWO, Idle())},
        drafts={ONE.id: FIX, TWO.id: BETTER},
    )
    after, _ = decide(both, DiscardDraft(TWO.id))
    assert after.drafts == {ONE.id: FIX}
