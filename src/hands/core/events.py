"""Everything that can happen to the registry: parsed at the edges, reduced here."""

from dataclasses import dataclass
from typing import Literal

from hands.core.session import Instant, Membership, Permission, RequestId, SessionId


StartSource = Literal["startup", "resume", "clear", "compact"]


@dataclass(frozen=True)
class Joined:
    membership: Membership
    source: StartSource


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


@dataclass(frozen=True)
class Tick:
    """The one clock the reducer hears: deadlines are compared against it, never against a timer."""

    at: Instant


# Events about a session the registry must already know; a join is how it comes to.
SessionEvent = Prompted | Stopped | PermissionRequested | Ended
Event = Joined | SessionEvent | Tick
