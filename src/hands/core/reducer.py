"""The session lifecycle as one pure function."""

from hands.core.effects import Audit, Effect, Unregistered
from hands.core.events import Ended, Event, Joined, PermissionRequested, Prompted, SessionEvent, Stopped
from hands.core.session import Blocked, Gone, Idle, Registry, Session, SessionState, Working


def reduce(registry: Registry, event: Event) -> tuple[Registry, list[Effect]]:
    """One event in; the next registry and the effects it calls for out. No I/O."""
    # [LAW:effects-at-boundaries] time arrives inside the event and the timeout
    # inside the registry, so a deadline is arithmetic on values, never a clock read.
    match event:
        case Joined(membership=membership):
            # A resumed or compacted session starts again: its membership is
            # replaced whole, never merged with the record it had before.
            return registry.put(Session(membership, Idle())), []
        case Prompted(at=at):
            return _enter(registry, event, Working(since=at))
        case Stopped():
            return _enter(registry, event, Idle())
        case PermissionRequested(at=at, request=request, permission=permission):
            deadline = at + registry.permission_timeout
            return _enter(registry, event, Blocked(on=permission, request=request, deadline=deadline))
        case Ended():
            return _enter(registry, event, Gone())


def _enter(registry: Registry, event: SessionEvent, state: SessionState) -> tuple[Registry, list[Effect]]:
    match registry.sessions.get(event.session):
        case None:
            # [LAW:no-silent-failure] an event for a session that never joined is a record, not a drop.
            return registry, [Audit(Unregistered(event))]
        case Session(membership=membership):
            return registry.put(Session(membership, state)), []
