"""Everything that can happen to the registry: parsed at the edges, reduced here."""

from dataclasses import dataclass
from typing import Literal

from hands.core.session import Blocker, Instant, Membership, FinishedCall, Mode, PromptId, RequestId, SessionId
from hands.core.status import Report, Stamp


StartSource = Literal["startup", "resume", "clear", "compact"]
# Why Claude Code 2.1.270 says a session ended. `other` is what a closed terminal reports.
EndReason = Literal["clear", "resume", "logout", "prompt_input_exit", "bypass_permissions_disabled", "other"]


@dataclass(frozen=True)
class Joined:
    membership: Membership
    source: StartSource


@dataclass(frozen=True)
class Attached:
    """A membership file names a running process: a session from before the daemon started, or one whose start hook never arrived."""

    membership: Membership


@dataclass(frozen=True)
class Died:
    """A session's process is no longer running, and no hook said the session ended."""

    membership: Membership


@dataclass(frozen=True)
class MovedOn:
    """A session's process runs, but holds another session now: a /clear or a resume in it, whose end hook never arrived."""

    membership: Membership


@dataclass(frozen=True)
class Prompted:
    session: SessionId
    at: Instant
    # The permission_mode the hook carried; None only when it carried none, which 2.1.281's never do.
    mode: Mode | None
    # [LAW:no-ambient-temporal-coupling] the prompt_id that names the turn this prompt opens. Only the opening names the
    # turn: every hook of a background subagent carries the prompt_id of the turn that started it, even after that
    # turn stopped and another opened (2.1.281), so a turn read off any later hook could be one already over.
    prompt: PromptId


@dataclass(frozen=True)
class Stopped:
    session: SessionId
    closing: str | None  # the reply the turn closed with, as the Stop hook carries it; None when there was none
    # The permission_mode the hook carried; None only when it carried none, which 2.1.281's never do.
    mode: Mode | None
    # The prompt_id of the turn that stopped: the id it last went on under, and a turn a background task's notification
    # opened has its own, which its UserPromptSubmit carried too (2.1.281).
    prompt: PromptId


@dataclass(frozen=True)
class Interrupted:
    """The user stopped the turn at the keyboard, with Escape or Ctrl-C: no Stop hook fires for that, so it is read
    from the record Claude Code writes in the transcript instead."""

    session: SessionId
    # The turn the record says it stopped: only the untold turn it names waits for it to be told.
    prompt: PromptId
    at: Instant  # when the record was read


@dataclass(frozen=True)
class Taken:
    """Claude Code took the prompt: the transcript holds a record of its turn, which it writes only once the prompt's
    hooks are done and it was not cancelled.

    At the prompt it is a turn no hook opened, running as Claude Code says: a message queued while a turn ran, taken
    once that turn's Stop lands; a `!` command, which Claude answers once it has run; a command such as /compact
    (2.1.282). What happened in it is what is told, which for a command is nothing."""

    session: SessionId
    prompt: PromptId
    # When Claude Code wrote the record, on the clock it stamps a status with, so an idle it set after it can be told
    # from one it set before. None when the record carried no time, which 2.1.282's always do, or one the tail could not
    # read and said so: either way it opens nothing.
    written: Stamp | None
    at: Instant  # when the record was read


@dataclass(frozen=True)
class Continued:
    """Claude went on answering under another prompt's id without the turn ending: a queued message taken in mid-turn,
    or flushed by an Escape, carries its own new id from then on, and no hook ever names it (2.1.281). Read from the
    transcript, where the user's side of every record Claude answers carries the id it answers under."""

    session: SessionId
    # The id it was answering under, so a record read after the next prompt opened moves nothing.
    was: PromptId
    now: PromptId


@dataclass(frozen=True)
class Waited:
    """Claude Code says the session has sat at its prompt since its turn ended, and nobody has typed: idle_prompt."""

    session: SessionId


@dataclass(frozen=True)
class StatusReported:
    """Claude Code set the session's status: read from the file it keeps for the session each time its stamp moves, so a
    status set again to what it was is heard, and so is a busy that came and went between two reads."""

    session: SessionId
    report: Report
    at: Instant  # when it was read


@dataclass(frozen=True)
class PermissionRequested:
    session: SessionId
    at: Instant
    request: RequestId
    on: Blocker
    # The permission_mode the hook carried; None only when it carried none, which 2.1.281's never do.
    mode: Mode | None


@dataclass(frozen=True)
class ToolFinished:
    """A tool call ran to its end, or failed. The call is named as the request to run it was, so the two can be matched."""

    session: SessionId
    at: Instant
    call: FinishedCall
    # The permission_mode the hook carried; None only when it carried none, which 2.1.281's never do.
    mode: Mode | None


@dataclass(frozen=True)
class Ended:
    session: SessionId
    reason: EndReason


@dataclass(frozen=True)
class Abandoned:
    """The hook waiting on a permission reply went away before one was decided, so nothing can be answered."""

    session: SessionId
    request: RequestId
    at: Instant


@dataclass(frozen=True)
class Tick:
    """The one clock the reducer hears: deadlines are compared against it, never against a timer."""

    at: Instant


# Events about a session the registry must already know; a join is how it comes to.
SessionEvent = Prompted | Stopped | Interrupted | Taken | Continued | Waited | StatusReported | PermissionRequested | ToolFinished | Ended
# What a session's transcript says of its turn that none of its hooks do.
Transcribed = Taken | Interrupted | Continued
# What the liveness sweep saw in one membership file.
Observed = Attached | Died | MovedOn
Event = Joined | Observed | SessionEvent | Abandoned | Tick
