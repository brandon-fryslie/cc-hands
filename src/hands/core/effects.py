"""What the reducer asks the edges to do. Adapters perform these and nothing else."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hands.core.events import SessionEvent
from hands.core.session import Blocker, Mode, PromptId, RequestId, SessionId


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
class Answers:
    """What the user chose for each question a session asked, in the order it asked them: a label, or their own words."""

    chosen: tuple[str, ...]


# Where an approved plan leaves plan mode for: back to the mode the session had before it planned, which is what
# ExitPlanMode does by itself, or one of the two modes named by the plan dialog's "Yes, auto-accept edits" and "Yes,
# manually approve edits".
ModeAfterPlan = Literal["resume", "acceptEdits", "default"]


@dataclass(frozen=True)
class Approve:
    """The plan is approved, and the session leaves plan mode for this mode."""

    mode: ModeAfterPlan


@dataclass(frozen=True)
class KeepPlanning:
    """The plan is sent back: the agent reads what the user wants changed, and stays in plan mode."""

    message: str


@dataclass(frozen=True)
class AllowWith:
    """The tool runs with this input in place of the one it asked with: how a question's answers reach it."""

    input: Mapping[str, object]


@dataclass(frozen=True)
class Withdraw:
    """hands lets go of the request without deciding it, so Claude Code's own dialog is the only answer left."""


# [LAW:types-are-the-program] what a person can decide is not what the daemon replies: nothing the user or the
# model says can produce a Withdraw, and answers become an AllowWith only against the question they answer.
Decision = Allow | Deny | Answers | Approve | KeepPlanning
HookReply = Allow | AllowWith | Approve | Deny | Withdraw


@dataclass(frozen=True)
class Reply:
    """The answer to a blocking PermissionRequest hook, sent back down the socket it is waiting on."""

    session: SessionId
    request: RequestId
    reply: HookReply


@dataclass(frozen=True)
class Asking:
    """A session stopped to ask; the intermediary explains what it asks and puts it to the user."""

    session: SessionId
    request: RequestId
    on: Blocker


@dataclass(frozen=True)
class DeadlineNear:
    session: SessionId
    on: Blocker
    remaining: float  # seconds


@dataclass(frozen=True)
class Expired:
    """Nobody answered by voice in time: a permission was denied, a question left to its dialog."""

    session: SessionId
    on: Blocker


@dataclass(frozen=True)
class WaitingForYou:
    """A session finished its turn a while ago and nobody has answered it."""

    session: SessionId


Announcement = DeadlineNear | Expired | WaitingForYou


@dataclass(frozen=True)
class Speak:
    """Said as written, with no model in the way."""

    announcement: Announcement


@dataclass(frozen=True)
class Narrate:
    """Handed to the intermediary to explain in its own words, and to act on what the user answers."""

    moment: Asking


@dataclass(frozen=True)
class ModeChanged:
    """A session reported a permission mode other than the one it reported before."""

    session: SessionId
    mode: Mode


@dataclass(frozen=True)
class Note:
    """Put in the intermediary's context and not spoken: the model knows, and says nothing of it until asked."""

    fact: ModeChanged


Heard = Speak | Narrate | Note


@dataclass(frozen=True)
class Summarise:
    """A session finished a turn: what the tail has not told of it is summarised by the model and spoken.

    The transcript is not named here because the tail is already following it [LAW:one-source-of-truth].
    """

    session: SessionId
    # [LAW:no-ambient-temporal-coupling] the turn that ended, by any prompt id its records carry: the tail may have read
    # the next prompt by the time this is told, and the turn it is on then is not this one. None where nothing named
    # it, and the tail tells the turn it is on.
    turn: PromptId | None
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
