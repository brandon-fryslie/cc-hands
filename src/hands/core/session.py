"""A session's membership and lifecycle state, and the registry that holds them."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NewType, Self

SessionId = NewType("SessionId", str)
RequestId = NewType("RequestId", str)
TmuxPane = NewType("TmuxPane", str)
Instant = float  # monotonic seconds


@dataclass(frozen=True)
class Membership:
    """Which process and pane a session is, as the shim recorded it at SessionStart."""

    id: SessionId
    pid: int
    pane: TmuxPane | None  # None when Claude Code runs outside tmux
    cwd: Path
    transcript: Path


@dataclass(frozen=True)
class Permission:
    tool: str
    input: Mapping[str, object]


# [LAW:types-are-the-program] a session is in exactly one of these, and each
# carries only what is true of that state: a working session has a start, a
# blocked one has the request it waits on and when that request expires.
@dataclass(frozen=True)
class Idle:
    pass


@dataclass(frozen=True)
class Working:
    since: Instant


@dataclass(frozen=True)
class Blocked:
    on: Permission
    request: RequestId
    deadline: Instant


@dataclass(frozen=True)
class Gone:
    pass


SessionState = Idle | Working | Blocked | Gone


@dataclass(frozen=True)
class Session:
    membership: Membership
    state: SessionState


@dataclass(frozen=True)
class Registry:
    permission_timeout: float
    sessions: Mapping[SessionId, Session]

    def put(self, session: Session) -> Self:
        return replace(self, sessions={**self.sessions, session.membership.id: session})

    def live(self) -> list[Session]:
        return [session for session in self.sessions.values() if not isinstance(session.state, Gone)]
