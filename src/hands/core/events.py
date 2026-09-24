"""Everything that can happen to the registry: parsed at the edges, reduced here."""

from dataclasses import dataclass
from typing import Literal

from hands.core.session import Blocker, Instant, Membership, FinishedCall, Mode, RequestId, SessionId


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


@dataclass(frozen=True)
class Stopped:
    session: SessionId
    closing: str | None  # the reply the turn closed with, as the Stop hook carries it; None when there was none
    # The permission_mode the hook carried; None only when it carried none, which 2.1.281's never do.
    mode: Mode | None


@dataclass(frozen=True)
class Waited:
    """Claude Code says the session has sat at its prompt since its turn ended, and nobody has typed: idle_prompt."""

    session: SessionId


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
SessionEvent = Prompted | Stopped | Waited | PermissionRequested | ToolFinished | Ended
# What the liveness sweep saw in one membership file.
Observed = Attached | Died | MovedOn
Event = Joined | Observed | SessionEvent | Abandoned | Tick
