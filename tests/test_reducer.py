"""The session lifecycle as a table: registry before, event, registry after, effects. No I/O."""

from dataclasses import replace
from pathlib import Path

import pytest

from hands.core.effects import Audit, Unregistered
from hands.core.events import Ended, Event, Joined, PermissionRequested, Prompted, SessionEvent, Stopped
from hands.core.reducer import reduce
from hands.core.session import (
    Blocked,
    Gone,
    Idle,
    Membership,
    Permission,
    Registry,
    RequestId,
    Session,
    SessionId,
    SessionState,
    TmuxPane,
    Working,
)

TIMEOUT = 60.0
ONE = Membership(SessionId("s1"), pid=4242, pane=TmuxPane("%3"), cwd=Path("/code/a"), transcript=Path("/t/s1.jsonl"))
TWO = Membership(SessionId("s2"), pid=5353, pane=None, cwd=Path("/code/b"), transcript=Path("/t/s2.jsonl"))
BASH = Permission(tool="Bash", input={"command": "ls"})

STATES: list[SessionState] = [
    Idle(),
    Working(since=1.0),
    Blocked(on=BASH, request=RequestId("r0"), deadline=61.0),
    Gone(),
]


def registry(*sessions: Session) -> Registry:
    return Registry(permission_timeout=TIMEOUT, sessions={s.membership.id: s for s in sessions})


def holding(state: SessionState) -> Registry:
    return registry(Session(ONE, state))


def test_a_start_registers_the_session_idle() -> None:
    assert reduce(registry(), Joined(ONE)) == (holding(Idle()), [])


@pytest.mark.parametrize("before", STATES)
@pytest.mark.parametrize(
    ("event", "after"),
    [
        (Joined(ONE), Idle()),
        (Prompted(ONE.id, at=5.0), Working(since=5.0)),
        (Stopped(ONE.id), Idle()),
        (
            PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), permission=BASH),
            Blocked(on=BASH, request=RequestId("r1"), deadline=5.0 + TIMEOUT),
        ),
        (Ended(ONE.id), Gone()),
    ],
)
def test_every_state_takes_each_event_to_its_state(before: SessionState, event: Event, after: SessionState) -> None:
    assert reduce(holding(before), event) == (holding(after), [])


def test_a_restarted_session_replaces_its_membership() -> None:
    moved = replace(ONE, pid=7777, pane=None)
    assert reduce(holding(Working(since=1.0)), Joined(moved)) == (registry(Session(moved, Idle())), [])


@pytest.mark.parametrize(
    "event",
    [
        Prompted(ONE.id, at=5.0),
        Stopped(ONE.id),
        PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), permission=BASH),
        Ended(ONE.id),
    ],
)
def test_an_event_for_a_session_that_never_joined_changes_nothing_and_is_audited(event: SessionEvent) -> None:
    before = registry(Session(TWO, Idle()))
    assert reduce(before, event) == (before, [Audit(Unregistered(event))])


def test_an_event_moves_only_its_own_session() -> None:
    before = registry(Session(ONE, Idle()), Session(TWO, Idle()))
    after, _ = reduce(before, Prompted(TWO.id, at=2.0))
    assert after == registry(Session(ONE, Idle()), Session(TWO, Working(since=2.0)))


def test_live_is_every_session_that_has_not_ended() -> None:
    both = registry(Session(ONE, Blocked(on=BASH, request=RequestId("r"), deadline=9.0)), Session(TWO, Gone()))
    assert both.live() == [Session(ONE, Blocked(on=BASH, request=RequestId("r"), deadline=9.0))]
