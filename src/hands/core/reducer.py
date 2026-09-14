"""The session lifecycle as one pure function."""

from dataclasses import replace

from hands.core.effects import (
    AfterEnd,
    Audit,
    Deny,
    Effect,
    Narrate,
    PermissionAsked,
    PermissionDeadlineNear,
    PermissionExpired,
    Reply,
    Speak,
    Unregistered,
    Withdraw,
)
from hands.core.events import Ended, Event, Joined, PermissionRequested, Prompted, SessionEvent, StartSource, Stopped, Tick
from hands.core.session import Blocked, Gone, Idle, Instant, Registry, Session, SessionId, SessionState, Working

# How long before a permission's deadline the one warning is spoken.
WARNING_LEAD_SECONDS = 10.0

# What the agent reads when nobody answered in time.
EXPIRED_MESSAGE = (
    "Nobody answered this permission request by voice before its deadline, so hands denied it. "
    "Do not retry it until the user asks."
)


def reduce(registry: Registry, event: Event) -> tuple[Registry, list[Effect]]:
    """One event in; the next registry and the effects it calls for out. No I/O."""
    # [LAW:effects-at-boundaries] time arrives inside the event and the deadline
    # inside the registry, so a deadline is arithmetic on values, never a clock read.
    match event:
        case Joined(membership=membership, source=source):
            previous = registry.sessions.get(membership.id)
            state = _started(source, previous)
            before = None if previous is None else previous.state
            return registry.put(Session(membership, state)), _transition(membership.id, before, state)
        case Prompted(at=at):
            return _enter(registry, event, Working(since=at))
        case Stopped():
            return _enter(registry, event, Idle())
        case PermissionRequested(at=at, request=request, permission=permission):
            deadline = at + registry.permission_deadline
            return _enter(registry, event, Blocked(on=permission, request=request, deadline=deadline, warned=False))
        case Ended():
            return _enter(registry, event, Gone())
        case Tick(at=at):
            return _ticked(registry, at)


def _started(source: StartSource, previous: Session | None) -> SessionState:
    # Compaction starts a session again in the middle of a turn, so what it was
    # doing carries over. Every other start sits at the prompt, whatever the
    # registry last heard: a session resumed after a crash was never told it stopped.
    match (source, previous):
        case ("compact", Session(state=Working() | Blocked() as state)):
            return state
        case _:
            return Idle()


def _enter(registry: Registry, event: SessionEvent, state: SessionState) -> tuple[Registry, list[Effect]]:
    match registry.sessions.get(event.session):
        case None:
            # [LAW:no-silent-failure] an event for a session that never joined is a record, not a drop.
            return registry, [Audit(Unregistered(event))]
        case Session(state=Gone()):
            # Ended is final until the session starts again; a hook that lands late cannot revive it.
            return registry, [Audit(AfterEnd(event))]
        case Session(membership=membership, state=before):
            return registry.put(Session(membership, state)), _transition(membership.id, before, state)


def _transition(session: SessionId, before: SessionState | None, after: SessionState) -> list[Effect]:
    # [LAW:dataflow-not-control-flow] every change of state passes through here, so no event
    # can leave a hook waiting or a request unspoken by taking a path that forgot to.
    match (before, after):
        case (Blocked(request=held), Blocked(request=asked)) if held == asked:
            # Compaction kept the session waiting on the same request.
            return []
        case (Blocked(request=held), Blocked(request=asked, on=permission)):
            return [Reply(session, held, Withdraw()), Narrate(PermissionAsked(session, asked, permission))]
        case (Blocked(request=held), _):
            # The session moved on without a voice answer, most often because the user answered its
            # dialog at the keyboard; the waiting hook is let go so it cannot decide a settled question.
            return [Reply(session, held, Withdraw())]
        case (_, Blocked(request=asked, on=permission)):
            return [Narrate(PermissionAsked(session, asked, permission))]
        case _:
            return []


def _ticked(registry: Registry, at: Instant) -> tuple[Registry, list[Effect]]:
    after, effects = registry, list[Effect]()
    for session in registry.sessions.values():
        state, due = _deadline(session.membership.id, session.state, at)
        after = after.put(Session(session.membership, state))
        effects += due
    return after, effects


def _deadline(session: SessionId, state: SessionState, at: Instant) -> tuple[SessionState, list[Effect]]:
    match state:
        case Blocked(on=permission, request=request, deadline=deadline) if at >= deadline:
            # [LAW:no-silent-failure] silence never approves: an unanswered request is denied, and said to be.
            return Working(since=at), [Reply(session, request, Deny(EXPIRED_MESSAGE)), Speak(PermissionExpired(session, permission))]
        case Blocked(on=permission, deadline=deadline, warned=False) if at >= deadline - WARNING_LEAD_SECONDS:
            return replace(state, warned=True), [Speak(PermissionDeadlineNear(session, permission, remaining=deadline - at))]
        case _:
            return state, []
