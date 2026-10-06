"""What Claude Code itself says a session is doing, as it publishes it in the session's own file.

Claude Code 2.1.280 to 2.1.289 keep ~/.claude/sessions/<pid>.json for every interactive session and rewrite its
`status` as the session moves: busy while a turn, a `!` command or a background subagent runs, waiting at a dialog,
idle at the prompt. It is the session's own word on whether it is at work, which no hook and no transcript record says
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
    """A turn or a `!` command runs; or a subagent it started in the background does, from its launch until the turn
    that reports it back ends, though every turn between has stopped (2.1.289)."""


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
    """At its prompt, with a shell command it ran in the background still running: Claude Code writes shell where it
    would write idle while any of its background bash tasks has not ended (2.1.289). No turn runs; the task's end is
    news to the session, which opens one. Claude Code's own session list labels it working, for the task; hands says
    what the session does, which is wait at its prompt, where a typed prompt runs at once."""


@dataclass(frozen=True)
class Unknown:
    """A status this version of hands does not know, kept by its name: never read as idle, or as anything else."""

    name: str


# [LAW:types-are-the-program] one variant a status, each carrying only what is true of it: only a waiting session has a reason.
Status = Idle | Busy | Waiting | Shell | Unknown
# At its prompt, where a typed prompt runs at once: no turn runs, whatever its background shells do.
AtPrompt = Idle | Shell
# Every other status: whatever the session is doing, Claude Code does not say it is at its prompt; busy can be, while a
# background subagent works, which only its turns and its subagents' reports tell (session.Delegating).
Going = Busy | Waiting | Unknown


@dataclass(frozen=True)
class Report:
    """One status as Claude Code set it, and when."""

    status: Status
    stamp: Stamp
