"""The session lifecycle as a table: registry before, event, registry after, effects. No I/O."""

from dataclasses import replace
from pathlib import Path

import pytest

from hands.core.effects import AfterEnd, Audit, Unregistered
from hands.core.events import Ended, Event, Joined, PermissionRequested, Prompted, SessionEvent, StartSource, Stopped
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

LIVE: list[SessionState] = [
    Idle(),
    Working(since=1.0),
    Blocked(on=BASH, request=RequestId("r0"), deadline=61.0),
]
SESSION_EVENTS: list[SessionEvent] = [
    Prompted(ONE.id, at=5.0),
    Stopped(ONE.id),
    PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), permission=BASH),
    Ended(ONE.id),
]


def registry(*sessions: Session) -> Registry:
    return Registry(permission_timeout=TIMEOUT, sessions={s.membership.id: s for s in sessions}, drafts={})


def holding(state: SessionState) -> Registry:
    return registry(Session(ONE, state))


def test_a_start_registers_the_session_idle() -> None:
    assert reduce(registry(), Joined(ONE, "startup")) == (holding(Idle()), [])


@pytest.mark.parametrize("before", LIVE)
@pytest.mark.parametrize(
    ("event", "after"),
    [
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


@pytest.mark.parametrize("before", LIVE)
def test_a_compacted_session_keeps_its_state_and_takes_the_new_membership(before: SessionState) -> None:
    moved = replace(ONE, pid=7777, pane=None)
    assert reduce(holding(before), Joined(moved, "compact")) == (registry(Session(moved, before)), [])


@pytest.mark.parametrize("before", [*LIVE, Gone()])
@pytest.mark.parametrize("source", ["startup", "resume", "clear"])
def test_any_start_but_compaction_is_at_the_prompt(before: SessionState, source: StartSource) -> None:
    # a session resumed after a crash never sent the Stop or SessionEnd the registry is still waiting for
    assert reduce(holding(before), Joined(ONE, source)) == (holding(Idle()), [])


def test_an_ended_session_compacting_is_idle() -> None:
    assert reduce(holding(Gone()), Joined(ONE, "compact")) == (holding(Idle()), [])


@pytest.mark.parametrize("event", SESSION_EVENTS)
def test_an_ended_session_ignores_late_hooks_and_audits_them(event: SessionEvent) -> None:
    assert reduce(holding(Gone()), event) == (holding(Gone()), [Audit(AfterEnd(event))])


@pytest.mark.parametrize("event", SESSION_EVENTS)
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
