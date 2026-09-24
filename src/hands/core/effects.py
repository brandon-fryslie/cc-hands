"""What the reducer asks the edges to do. Adapters perform these and nothing else."""

from dataclasses import dataclass
from pathlib import Path

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
class WaitingForYou:
    """A session finished its turn a while ago and nobody has answered it."""

    session: SessionId


Announcement = PermissionDeadlineNear | PermissionExpired | WaitingForYou


@dataclass(frozen=True)
class Speak:
    """Said as written, with no model in the way."""

    announcement: Announcement


@dataclass(frozen=True)
class Narrate:
    """Handed to the intermediary to explain in its own words, and to act on what the user answers."""

    moment: PermissionAsked


Heard = Speak | Narrate


@dataclass(frozen=True)
class Summarise:
    """A session finished a turn: what the tail has not told of it is summarised by the model and spoken.

    The transcript is not named here because the tail is already following it [LAW:one-source-of-truth].
    """

    session: SessionId
    closing: str | None


@dataclass(frozen=True)
class SessionGone:
    """A session ended without the user ending it at the keyboard: its terminal closed, or its process died."""

    session: SessionId


# [LAW:no-ambient-temporal-coupling] what a session did and that it ended are told in the order they happened:
# a summary takes seconds, so an end spoken at once would be heard before the last turn it ends.
Story = Summarise | SessionGone


@dataclass(frozen=True)
class Snapshot:
    """Where a session's repository stands as its turn begins, so that what the turn changes can be read against it.

    Carries where the session works rather than leaving it to be looked up: where that is, is the registry's to
    say [LAW:one-source-of-truth], and saying it here is what lets the reader be built without one.
    """

    session: SessionId
    cwd: Path


@dataclass(frozen=True)
class Compare:
    """What a session's turn changed, read against the snapshot its start took.

    Told apart from Summarise and done before it, because a summary is made one at a time and takes seconds:
    read then, the repository would already hold whatever the next turn had started doing.
    """

    session: SessionId


# What a turn did to the repository it ran in, which no record of the session need name: a formatter, a code
# generator, or a `sed` in a shell command changes files that no step reports.
Repository = Snapshot | Compare

Effect = Audit | Reply | Heard | Story | Repository
