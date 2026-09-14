"""Everything that can happen to the registry: parsed at the edges, reduced here."""

from dataclasses import dataclass

from hands.core.session import Instant, Membership, Permission, RequestId, SessionId


@dataclass(frozen=True)
class Joined:
    membership: Membership


@dataclass(frozen=True)
class Prompted:
    session: SessionId
    at: Instant


@dataclass(frozen=True)
class Stopped:
    session: SessionId


@dataclass(frozen=True)
class PermissionRequested:
    session: SessionId
    at: Instant
    request: RequestId
    permission: Permission


@dataclass(frozen=True)
class Ended:
    session: SessionId


# Events about a session the registry must already know; a join is how it comes to.
SessionEvent = Prompted | Stopped | PermissionRequested | Ended
Event = Joined | SessionEvent
