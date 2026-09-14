"""Drafts as a table: registry before, request, and what came of it. No I/O."""

from pathlib import Path

import pytest

from hands.core.drafts import (
    AmendDraft,
    AwaitingPermission,
    DiscardDraft,
    DraftAmended,
    DraftDiscarded,
    DraftRequest,
    DraftSent,
    DraftStaged,
    NothingStaged,
    OutsideTmux,
    SendDraft,
    SessionEnded,
    StageDraft,
    UnknownSession,
    decide,
)
from hands.core.effects import Audit, Sending, Text, Type
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
    TmuxPane,
    Working,
)

PANE = TmuxPane("%3")
ONE = Membership(SessionId("s1"), pid=1, pane=PANE, cwd=Path("/code/a"), transcript=Path("/t/s1.jsonl"))
TWO = Membership(SessionId("s2"), pid=2, pane=TmuxPane("%4"), cwd=Path("/code/b"), transcript=Path("/t/s2.jsonl"))
HEADLESS = Membership(SessionId("s1"), pid=1, pane=None, cwd=Path("/code/a"), transcript=Path("/t/s1.jsonl"))
FIX = Staged(PromptText("/fix the auth middleware"), (Resolution("auth middleware", "authMiddleware.ts"),))
BETTER = Staged(PromptText("fix the token helper"), ())
BASH = Permission(tool="Bash", input={"command": "ls"})
BLOCKED = Blocked(on=BASH, request=RequestId("r"), deadline=9.0)


def registry(
    state: SessionState = Idle(), membership: Membership = ONE, drafts: dict[SessionId, Staged] | None = None
) -> Registry:
    return Registry(permission_timeout=60.0, sessions={membership.id: Session(membership, state)}, drafts=drafts or {})


def staged(state: SessionState = Idle(), membership: Membership = ONE) -> Registry:
    return registry(state, membership, {ONE.id: FIX})


@pytest.mark.parametrize("state", [Idle(), Working(since=1.0), BLOCKED])
def test_staging_holds_the_draft_and_types_nothing(state: SessionState) -> None:
    assert decide(registry(state), StageDraft(ONE.id, FIX)) == (staged(state), DraftStaged(ONE.id, FIX, replaced=None), [])


def test_staging_over_a_draft_replaces_it_and_says_so() -> None:
    after = registry(drafts={ONE.id: BETTER})
    assert decide(staged(), StageDraft(ONE.id, BETTER)) == (after, DraftStaged(ONE.id, BETTER, replaced=FIX), [])


def test_amending_replaces_the_draft_and_keeps_both_versions_for_the_readback() -> None:
    after = registry(drafts={ONE.id: BETTER})
    assert decide(staged(), AmendDraft(ONE.id, BETTER)) == (after, DraftAmended(ONE.id, before=FIX, after=BETTER), [])


@pytest.mark.parametrize("state", [Idle(), Working(since=1.0)])
def test_sending_audits_then_types_the_draft_and_clears_it(state: SessionState) -> None:
    # A working session queues the prompt itself, so it is sent exactly as an idle one is.
    effects = [Audit(Sending(ONE.id, PANE, FIX.text)), Type(PANE, Text(FIX.text))]
    assert decide(staged(state), SendDraft(ONE.id)) == (registry(state), DraftSent(ONE.id, FIX), effects)


def test_sending_to_a_session_at_a_permission_dialog_is_refused_and_the_draft_kept() -> None:
    before = staged(BLOCKED)
    assert decide(before, SendDraft(ONE.id)) == (before, AwaitingPermission(ONE.id, BASH), [])


@pytest.mark.parametrize("request_", [AmendDraft(ONE.id, BETTER), DiscardDraft(ONE.id), SendDraft(ONE.id)])
def test_with_nothing_staged_there_is_nothing_to_amend_discard_or_send(request_: DraftRequest) -> None:
    assert decide(registry(), request_) == (registry(), NothingStaged(ONE.id), [])


def test_discarding_clears_the_draft_even_after_the_session_ended() -> None:
    assert decide(staged(Gone()), DiscardDraft(ONE.id)) == (registry(Gone()), DraftDiscarded(ONE.id, FIX), [])


@pytest.mark.parametrize("request_", [StageDraft(ONE.id, BETTER), AmendDraft(ONE.id, BETTER), SendDraft(ONE.id)])
def test_an_ended_session_takes_no_draft_and_is_sent_nothing(request_: DraftRequest) -> None:
    before = staged(Gone())
    assert decide(before, request_) == (before, SessionEnded(ONE.id), [])


@pytest.mark.parametrize("request_", [StageDraft(ONE.id, BETTER), AmendDraft(ONE.id, BETTER), SendDraft(ONE.id)])
def test_a_session_outside_tmux_cannot_be_typed_into(request_: DraftRequest) -> None:
    before = staged(membership=HEADLESS)
    assert decide(before, request_) == (before, OutsideTmux(ONE.id), [])


@pytest.mark.parametrize(
    "request_", [StageDraft(TWO.id, FIX), AmendDraft(TWO.id, FIX), DiscardDraft(TWO.id), SendDraft(TWO.id)]
)
def test_a_session_that_never_joined_is_named_unknown(request_: DraftRequest) -> None:
    assert decide(staged(), request_) == (staged(), UnknownSession(TWO.id), [])


def test_a_send_touches_only_its_own_session() -> None:
    both = Registry(
        permission_timeout=60.0,
        sessions={ONE.id: Session(ONE, Idle()), TWO.id: Session(TWO, Idle())},
        drafts={ONE.id: FIX, TWO.id: BETTER},
    )
    after, _, _ = decide(both, SendDraft(TWO.id))
    assert after.drafts == {ONE.id: FIX}
