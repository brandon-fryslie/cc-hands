"""What the reducer asks the edges to do. Adapters perform these and nothing else."""

from dataclasses import dataclass

from hands.core.events import SessionEvent
from hands.core.session import Permission, RequestId, SessionId


@dataclass(frozen=True)
class Unregistered:
    """An event named a session that never joined, so it changed nothing."""

    event: SessionEvent


@dataclass(frozen=True)
class AfterEnd:
    """An event arrived for a session that had already ended, so it changed nothing."""

    event: SessionEvent


AuditRecord = Unregistered | AfterEnd


@dataclass(frozen=True)
class Audit:
    record: AuditRecord


@dataclass(frozen=True)
class Allow:
    """The tool runs."""


@dataclass(frozen=True)
class Deny:
    """The tool does not run, and the agent reads the message."""

    message: str


@dataclass(frozen=True)
class Withdraw:
    """hands lets go of the request without deciding it, so Claude Code's own dialog is the only answer left."""


# [LAW:types-are-the-program] what a person can decide is narrower than what the daemon can reply:
# nothing the user or the model says can produce a Withdraw, and nothing but Allow runs a tool.
Decision = Allow | Deny
HookReply = Decision | Withdraw


@dataclass(frozen=True)
class Reply:
    """The answer to a blocking PermissionRequest hook, sent back down the socket it is waiting on."""

    session: SessionId
    request: RequestId
    reply: HookReply


@dataclass(frozen=True)
class PermissionAsked:
    """A session stopped to ask; the intermediary explains the request and asks the user."""

    session: SessionId
    request: RequestId
    permission: Permission


@dataclass(frozen=True)
class PermissionDeadlineNear:
    session: SessionId
    permission: Permission
    remaining: float  # seconds


@dataclass(frozen=True)
class PermissionExpired:
    """Nobody answered by voice in time, so hands replied deny."""

    session: SessionId
    permission: Permission


@dataclass(frozen=True)
class SessionGone:
    """A session ended without the user ending it at the keyboard: its terminal closed, or its process died."""

    session: SessionId


Announcement = PermissionDeadlineNear | PermissionExpired | SessionGone


@dataclass(frozen=True)
class Speak:
    """Said as written, with no model in the way."""

    announcement: Announcement


@dataclass(frozen=True)
class Narrate:
    """Handed to the intermediary to explain in its own words, and to act on what the user answers."""

    moment: PermissionAsked


Heard = Speak | Narrate
Effect = Audit | Reply | Heard
