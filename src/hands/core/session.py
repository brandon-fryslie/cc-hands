"""A session's membership and lifecycle state, and the registry that holds them."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, NewType, Self

from hands.core.status import Report, Stamp

SessionId = NewType("SessionId", str)
RequestId = NewType("RequestId", str)
# Claude Code's prompt_id: every hook of one turn carries the id of the prompt that opened it, and so does every
# transcript record of that turn, which is how a record read from the file is matched to the turn a hook opened.
PromptId = NewType("PromptId", str)
Instant = float  # monotonic seconds

# Prompt text that holds no control characters, so typing it into a session
# cannot press a key the text does not name. Made only where the model's words are parsed.
# A newline is not a control character here: bracketing carries it into the message, and it
# is the carriage return that would submit a half-written one. A tab is, because nothing
# carries a tab - it is in the Keystroke vocabulary below and is sent by name.
PromptText = NewType("PromptText", str)

# The named chords a session can be sent, as distinct from text. Which bytes each one is
# belongs to whatever does the typing, not here; this is the vocabulary hands speaks.
Keystroke = Literal["escape", "enter", "ctrl_c", "ctrl_u", "up", "down", "tab", "shift_tab"]


@dataclass(frozen=True)
class Membership:
    """Which process a session is, and where it works, as the shim recorded it at SessionStart."""

    id: SessionId
    pid: int
    cwd: Path
    transcript: Path
    # Where hands can type into this session: the control socket of the fritter that
    # wrapped it, which published the address to the process in FRITTER_SOCKET.
    #
    # [LAW:types-are-the-program] Absent, and absent in the type, for a session started
    # outside fritter. Such a session can be listed, read and spoken about like any
    # other; it simply cannot be typed into, and the type says so rather than leaving a
    # caller to find out by writing to a path that is not there.
    fritter: Path | None = None


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
    # Whether the turn that left it here ended on a question or an offer, so the nudge can say it has one rather than
    # only that it waits: its dialog question left unanswered, or its closing reply asking. See `_asking` in the reducer.
    asking: bool = False


@dataclass(frozen=True)
class Submitted:
    """Sent from the prompt while its UserPromptSubmit hooks run: Claude Code has not taken it, and never will if the user
    presses Escape before they finish, which puts it back in the input box with no hook and no record to say so, and sets
    the session idle (2.1.282)."""

    since: Instant


@dataclass(frozen=True)
class Working:
    since: Instant
    # Its question dialog was closed unanswered, by an Escape at it, which kills the hook and fires no post-tool hook
    # and no Stop, and it has run nothing since: what the turn is waiting on if it ends here. See `_asking` in the reducer.
    unanswered: bool = False


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
class Untold:
    """A turn Claude Code said is over and hands has not told yet. Claude Code sets idle before the transcript says how
    the turn ended (an interrupt's record lands ~100 ms after, 2.1.282), so the turn is told once that record or its Stop
    is read, or a turn after it opens, or at `by` with what was read by then."""

    turn: PromptId | None
    by: Instant


@dataclass(frozen=True)
class Session:
    membership: Membership
    state: SessionState
    # [LAW:one-source-of-truth] the permission_mode of the last hook that carried one. None until one does:
    # SessionStart, idle_prompt, and SessionEnd carry none (verified live on 2.1.281).
    mode: Mode | None
    # [LAW:no-ambient-temporal-coupling] the id the session's last turn goes by: its prompt's, or the one Claude went on
    # answering under, which names the turn a busy session is in. A Stop ends a busy session's turn only when it names
    # it, so one applied late never ends the turn after it; at the prompt, a Stop of this turn ends it again only when
    # the turn went on after another Stop hook blocked its Stop (see _ends). None until a prompt is heard.
    turn: PromptId | None
    # Every other id the running turn has been read going on under: a flush's is taken seconds before Claude answers
    # under it and the turn is moved to it, and a message queued in between carries it (2.1.281).
    taken: frozenset[PromptId] = frozenset()
    # [LAW:one-source-of-truth] what Claude Code last said the session is doing, as it said it. None until it is read.
    report: Report | None = None
    # When Claude Code last set the session idle, of the statuses read: a prompt taken under an id no hook named opens a
    # turn only when written since. Until an idle is read, when it set the first status read. None until any is read, so
    # a transcript read from its start opens nothing.
    idled: Stamp | None = None
    # A message the user sent while the running turn ran, waiting behind it: Claude Code runs it once the turn's Stop hook
    # returns, under an id no hook names (2.1.282). False at the prompt.
    queued: bool = False
    # [LAW:no-ambient-temporal-coupling] the one wait on the transcript, as a value the clock settles: None when every
    # turn that ended has been told.
    untold: Untold | None = None


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
