"""A session's membership and lifecycle state, and the registry that holds them."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, NewType, Self

SessionId = NewType("SessionId", str)
RequestId = NewType("RequestId", str)
# Claude Code's prompt_id: every hook of one turn carries the id of the prompt that opened it, and so does every
# transcript record of that turn, which is how a record read from the file is matched to the turn a hook opened.
PromptId = NewType("PromptId", str)
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


# The permission modes Claude Code 2.1.281 has, named as every hook's permission_mode names them. Shift-tab cycles
# them at the keyboard, and no hook fires when it does: a change is heard at the session's next hook.
PermissionMode = Literal["default", "acceptEdits", "plan", "auto", "dontAsk", "bypassPermissions"]


@dataclass(frozen=True)
class UnknownMode:
    """A permission_mode this version of hands does not know, kept by its name so it is said rather than guessed at."""

    name: str


Mode = PermissionMode | UnknownMode


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
    # When hands says the session is waiting on its own clock, for an idle period Claude Code sends no idle_prompt for:
    # one a turn the user interrupted began (2.1.281). None where idle_prompt will say it, or already has.
    due: Instant | None = None


@dataclass(frozen=True)
class Submitted:
    """Sent from the prompt while its UserPromptSubmit hooks run: Claude Code has not taken it, and never will if the user
    presses Escape before they finish, which puts it back in the input box with no hook and no record to say so (2.1.281)."""

    since: Instant
    # [LAW:no-ambient-temporal-coupling] the prompt this one was sent over while that one was still Submitted:
    # cancelled, or taken and ended before any record of it was read. Only its own record can say which, and that is
    # read after this, so the prompt is kept until then, and a record of it ends it as the turn it was. None where there was none.
    over: PromptId | None = None


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


SessionState = Idle | Submitted | Working | Blocked | AtDialog | Gone


@dataclass(frozen=True)
class Session:
    membership: Membership
    state: SessionState
    # [LAW:one-source-of-truth] the permission_mode of the last hook that carried one. None until one does:
    # SessionStart, idle_prompt, and SessionEnd carry none (verified live on 2.1.281).
    mode: Mode | None
    # [LAW:no-ambient-temporal-coupling] the prompt_id of the last prompt the session was given, which names the turn a
    # working or blocked session is in. What ends a turn from outside its hooks names the turn it ends, so it can never
    # end the one after it, however late it is read. None until a prompt is heard.
    turn: PromptId | None
    # Every other id the running turn has been read going on under: a flush's is taken seconds before Claude answers
    # under it and the turn is moved to it, and a message queued in between carries it (2.1.281).
    taken: frozenset[PromptId] = frozenset()
    # [LAW:no-ambient-temporal-coupling] the ids of the last turn hands ended before its Stop was heard, because a
    # later prompt or a later turn's record showed it over: that Stop, applied late, ends nothing.
    ended: frozenset[PromptId] = frozenset()


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
