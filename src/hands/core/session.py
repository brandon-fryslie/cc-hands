"""A session's membership and lifecycle state, and the registry that holds them."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NewType, Self

SessionId = NewType("SessionId", str)
RequestId = NewType("RequestId", str)
Instant = float  # monotonic seconds

# Prompt text that holds no control characters, so typing it into a session
# cannot press a key the text does not name. Made only where the model's words are parsed.
PromptText = NewType("PromptText", str)


@dataclass(frozen=True)
class Membership:
    """Which process a session is, and where it works, as the shim recorded it at SessionStart."""

    id: SessionId
    pid: int
    cwd: Path
    transcript: Path


@dataclass(frozen=True)
class Permission:
    tool: str
    input: Mapping[str, object]


@dataclass(frozen=True)
class Option:
    label: str
    description: str | None


@dataclass(frozen=True)
class AskedQuestion:
    question: str
    # Empty for a question that takes the user's own words rather than a choice.
    options: tuple[Option, ...]
    several: bool  # more than one option may be chosen


@dataclass(frozen=True)
class Question:
    """AskUserQuestion, waiting on the user's answers."""

    asked: tuple[AskedQuestion, ...]
    # The tool input as it was asked, which the answers are written back into.
    input: Mapping[str, object]


@dataclass(frozen=True)
class Plan:
    """ExitPlanMode, waiting on the user to approve the plan or send it back to be planned again."""

    text: str


# Everything a session stops for arrives through the same PermissionRequest hook.
Blocker = Permission | Question | Plan


@dataclass(frozen=True)
class PlanApproved:
    """ExitPlanMode ran, or failed as it ran: either way its plan was approved, at its dialog or by voice, and the dialog
    is gone. What ran no longer carries the plan."""


# A tool call that ran, named as the request to run it was so the two can be matched.
FinishedCall = Permission | Question | PlanApproved


# [LAW:types-are-the-program] a session is in exactly one of these, and each
# carries only what is true of that state: a working session has a start, a
# blocked one has the request it waits on and when that request expires.
@dataclass(frozen=True)
class Idle:
    # [LAW:no-ambient-temporal-coupling] one idle period is one Idle value: the nudge is spoken once because speaking
    # it is this value changing, and every way into Idle builds a fresh one, so the next period can be nudged again.
    nudged: bool = False


@dataclass(frozen=True)
class Working:
    since: Instant


@dataclass(frozen=True)
class Blocked:
    on: Blocker
    request: RequestId
    deadline: Instant
    # [LAW:no-ambient-temporal-coupling] the warning is spoken once because speaking it is this
    # value changing, not a timer that could fire twice.
    warned: bool


@dataclass(frozen=True)
class AtDialog:
    """Still at its dialog after hands let go of the hook at the deadline, so only the keyboard can answer it now."""

    on: Blocker


@dataclass(frozen=True)
class Gone:
    pass


SessionState = Idle | Working | Blocked | AtDialog | Gone


@dataclass(frozen=True)
class Session:
    membership: Membership
    state: SessionState


@dataclass(frozen=True)
class Resolution:
    """A spoken phrase the model turned into something exact, such as a file name."""

    heard: str
    meant: str


@dataclass(frozen=True)
class Staged:
    """A draft waiting for the user's word: the text to send and how it was resolved."""

    text: PromptText
    resolutions: tuple[Resolution, ...]


@dataclass(frozen=True)
class Registry:
    permission_deadline: float  # seconds from a permission request to its default deny
    sessions: Mapping[SessionId, Session]
    # [LAW:types-are-the-program] a session with no entry has nothing staged; there is no empty draft.
    drafts: Mapping[SessionId, Staged]

    def put(self, session: Session) -> Self:
        return replace(self, sessions={**self.sessions, session.membership.id: session})

    def stage(self, session: SessionId, draft: Staged) -> Self:
        return replace(self, drafts={**self.drafts, session: draft})

    def unstage(self, session: SessionId) -> Self:
        return replace(self, drafts={id: draft for id, draft in self.drafts.items() if id != session})

    def live(self) -> list[Session]:
        return [session for session in self.sessions.values() if not isinstance(session.state, Gone)]
