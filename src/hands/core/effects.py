"""What the reducer asks the edges to do. Adapters perform these and nothing else."""

from dataclasses import dataclass

from hands.core.events import SessionEvent
from hands.core.session import PromptText, SessionId, TmuxPane


@dataclass(frozen=True)
class Unregistered:
    """An event named a session that never joined, so it changed nothing."""

    event: SessionEvent


@dataclass(frozen=True)
class AfterEnd:
    """An event arrived for a session that had already ended, so it changed nothing."""

    event: SessionEvent


@dataclass(frozen=True)
class Sending:
    """A draft the user approved, recorded before a key of it is typed."""

    session: SessionId
    pane: TmuxPane
    text: PromptText


AuditRecord = Unregistered | AfterEnd | Sending


@dataclass(frozen=True)
class Audit:
    record: AuditRecord


# [LAW:types-are-the-program] what goes into a pane decides its own escaping, so no
# code path looks at the first character to find out what the input is.
@dataclass(frozen=True)
class Text:
    """A prompt, submitted as text even when it starts with /, @, or !."""

    body: PromptText


Input = Text


@dataclass(frozen=True)
class Type:
    pane: TmuxPane
    input: Input


Effect = Audit | Type
