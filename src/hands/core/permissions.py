"""What the user decides about a permission request, what came of it, and the one function that decides."""

from dataclasses import dataclass

from hands.core.effects import Decision, Effect, Reply
from hands.core.session import Blocked, Instant, Permission, Registry, RequestId, Session, SessionId, Working


@dataclass(frozen=True)
class AnswerPermission:
    request: RequestId
    decision: Decision
    at: Instant


@dataclass(frozen=True)
class PermissionAnswered:
    session: SessionId
    permission: Permission
    decision: Decision


@dataclass(frozen=True)
class NotWaiting:
    """No session waits on this request: it was answered, withdrawn, or denied at its deadline."""

    request: RequestId


PermissionOutcome = PermissionAnswered | NotWaiting


def answer(registry: Registry, request: AnswerPermission) -> tuple[Registry, PermissionOutcome, list[Effect]]:
    """One answer in; the next registry, what came of it, and the reply it calls for out. No I/O."""
    waiting = [session for session in registry.sessions.values() if _waits_on(session, request.request)]
    match waiting:
        case [Session(membership=membership, state=Blocked(on=permission))]:
            # The turn carries on from here, with the tool run or refused.
            answered = registry.put(Session(membership, Working(since=request.at)))
            return answered, PermissionAnswered(membership.id, permission, request.decision), [Reply(membership.id, request.request, request.decision)]
        case _:
            return registry, NotWaiting(request.request), []


def _waits_on(session: Session, request: RequestId) -> bool:
    match session.state:
        case Blocked(request=held):
            return held == request
        case _:
            return False
