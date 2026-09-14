"""The session lifecycle as one pure function."""

from hands.core.effects import AfterEnd, Audit, Effect, Unregistered
from hands.core.events import Ended, Event, Joined, PermissionRequested, Prompted, SessionEvent, StartSource, Stopped
from hands.core.session import Blocked, Gone, Idle, Registry, Session, SessionState, Working


def reduce(registry: Registry, event: Event) -> tuple[Registry, list[Effect]]:
    """One event in; the next registry and the effects it calls for out. No I/O."""
    # [LAW:effects-at-boundaries] time arrives inside the event and the timeout
    # inside the registry, so a deadline is arithmetic on values, never a clock read.
    match event:
        case Joined(membership=membership, source=source):
            return registry.put(Session(membership, _started(source, registry.sessions.get(membership.id)))), []
        case Prompted(at=at):
            return _enter(registry, event, Working(since=at))
        case Stopped():
            return _enter(registry, event, Idle())
        case PermissionRequested(at=at, request=request, permission=permission):
            deadline = at + registry.permission_timeout
            return _enter(registry, event, Blocked(on=permission, request=request, deadline=deadline))
        case Ended():
            return _enter(registry, event, Gone())


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
        case Session(membership=membership):
            return registry.put(Session(membership, state)), []
