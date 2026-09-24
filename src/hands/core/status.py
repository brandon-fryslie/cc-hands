"""What Claude Code itself says a session is doing, as it publishes it in the session's own file.

Claude Code 2.1.280 to 2.1.282 keep ~/.claude/sessions/<pid>.json for every interactive session and rewrite its
`status` as the session moves: busy while a turn or a `!` command runs, waiting at a dialog, idle at the prompt. It is the session's own word on whether a turn is running, which no hook and no transcript record says
for every way a turn can stop.
"""

from dataclasses import dataclass
from typing import Literal, NewType

# Claude Code's statusUpdatedAt: wall-clock milliseconds since the epoch, rewritten each time it sets the status, even to
# the one it already had.
Stamp = NewType("Stamp", int)


@dataclass(frozen=True)
class Idle:
    pass


@dataclass(frozen=True)
class Busy:
    pass


# The waitingFor reasons 2.1.282 writes.
Reason = Literal["permission prompt", "input needed"]


@dataclass(frozen=True)
class UnknownReason:
    """A waitingFor this version of hands does not know, kept by its name so it is said rather than guessed at."""

    name: str


@dataclass(frozen=True)
class Waiting:
    """At a dialog that needs the user: a permission prompt, or a question such as AskUserQuestion."""

    reason: Reason | UnknownReason


@dataclass(frozen=True)
class Shell:
    """Seen live on 2.1.280, not yet tied to what the session was doing: a command typed after `!` reports busy (2.1.282)."""


@dataclass(frozen=True)
class Unknown:
    """A status this version of hands does not know, kept by its name: never read as idle, or as anything else."""

    name: str


# [LAW:types-are-the-program] one variant a status, each carrying only what is true of it: only a waiting session has a reason.
Status = Idle | Busy | Waiting | Shell | Unknown


@dataclass(frozen=True)
class Report:
    """One status as Claude Code set it, and when."""

    status: Status
    stamp: Stamp
