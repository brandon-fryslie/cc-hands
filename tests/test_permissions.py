"""Answering a permission as a table: registry before, answer, and what came of it. No I/O."""

from pathlib import Path

import pytest

from hands.core.effects import Allow, Decision, Deny, Reply
from hands.core.permissions import AnswerPermission, NotWaiting, PermissionAnswered, answer
from hands.core.session import Blocked, Gone, Idle, Membership, Permission, Registry, RequestId, Session, SessionId, SessionState, Working

ONE = Membership(SessionId("s1"), pid=1, cwd=Path("/code/a"), transcript=Path("/t/s1.jsonl"))
TWO = Membership(SessionId("s2"), pid=2, cwd=Path("/code/b"), transcript=Path("/t/s2.jsonl"))
BASH = Permission(tool="Bash", input={"command": "rm -r build"})
REQUEST = RequestId("r1")


def registry(*sessions: Session) -> Registry:
    return Registry(permission_deadline=60.0, sessions={s.membership.id: s for s in sessions}, drafts={})


@pytest.mark.parametrize("decision", [Allow(), Deny("use git clean instead")])
def test_an_answer_replies_to_the_waiting_hook_and_the_turn_carries_on(decision: Decision) -> None:
    before = registry(Session(ONE, Blocked(on=BASH, request=REQUEST, deadline=61.0, warned=True)), Session(TWO, Idle()))
    assert answer(before, AnswerPermission(REQUEST, decision, at=20.0)) == (
        registry(Session(ONE, Working(since=20.0)), Session(TWO, Idle())),
        PermissionAnswered(ONE.id, BASH, decision),
        [Reply(ONE.id, REQUEST, decision)],
    )


@pytest.mark.parametrize(
    "state",
    [Idle(), Working(since=1.0), Gone(), Blocked(on=BASH, request=RequestId("another"), deadline=61.0, warned=False)],
)
def test_an_answer_to_a_request_nobody_waits_on_changes_nothing(state: SessionState) -> None:
    before = registry(Session(ONE, state))
    assert answer(before, AnswerPermission(REQUEST, Allow(), at=20.0)) == (before, NotWaiting(REQUEST), [])


def test_a_request_is_answered_once() -> None:
    once, _, _ = answer(registry(Session(ONE, Blocked(on=BASH, request=REQUEST, deadline=61.0, warned=False))), AnswerPermission(REQUEST, Allow(), at=2.0))
    assert answer(once, AnswerPermission(REQUEST, Deny("no"), at=3.0)) == (once, NotWaiting(REQUEST), [])
