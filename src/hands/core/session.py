"""A session's membership and lifecycle state, and the registry that holds them."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, NewType, Self

from hands.core.progress import Doing, Gathering
from hands.core.status import Going, Stamp
from hands.core.turn import AgentTask

SessionId = NewType("SessionId", str)
RequestId = NewType("RequestId", str)
# Claude Code's prompt_id: every hook of one turn carries the id of the prompt that opened it, and so does every
# transcript record of that turn, which is how a record read from the file is matched to the turn a hook opened.
PromptId = NewType("PromptId", str)
Instant = float  # monotonic seconds

# Prompt text that holds no control characters, so typing it into a session
# cannot press a key the text does not name. Made where the model's words are parsed, which
# refuses what does not fit, and by `pasted`, which renders hands' own text to fit.
# A newline is not a control character here: bracketing carries it into the message, and it
# is the carriage return that would submit a half-written one. A tab is, because nothing
# carries a tab - it is in the Keystroke vocabulary below and is sent by name.
# It does not end with a backslash, which would make the Return that sends it a newline.
PromptText = NewType("PromptText", str)

# Every C0 and C1 control character but newline: each would press a key if it were typed.
KEYSTROKES = re.compile(r"[\x00-\x09\x0b-\x1f\x7f-\x9f]")
# Half of a character that was cut in two, as Claude Code cuts a string mid-emoji: it is no character, and cannot be
# written as bytes for a terminal or a command line.
HALVES = re.compile(r"[\ud800-\udfff]")
# What a terminal is told rather than shown: colour, cursor moves, and the titles and modes set around them.
ESCAPES = re.compile(r"\x1b(\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(\x07|\x1b\\)|[()*+].|[@-Z\\-_0-9=>])")


def pasted(text: str) -> PromptText:
    """Text hands wrote or read, rendered to be typed as the characters it shows: what a terminal is told dropped, line
    ends as newlines, tabs as the spaces they stand for, any other control character and any half of a character
    spelled out, and a closing backslash kept from the Return behind it."""
    # [LAW:parse-dont-validate] the words the model chose are refused when they do not fit, so it can choose again; text
    # that is only being passed on - a turn to summarise, a note - is made to fit, since there is no one to ask.
    shown = ESCAPES.sub("", text).replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
    keyless = KEYSTROKES.sub(lambda control: f"\\x{ord(control.group()):02x}", shown)
    spelled = HALVES.sub(lambda half: f"\\u{ord(half.group()):04x}", keyless)
    return PromptText(f"{spelled} " if spelled.endswith("\\") else spelled)


# The named chords a session can be sent, as distinct from text. Which bytes each one is
# belongs to whatever does the typing, not here; this is the vocabulary hands speaks.
Keystroke = Literal["escape", "enter", "ctrl_c", "ctrl_u", "up", "down", "tab", "shift_tab"]

# A slash command's name without its slash, such as `compact` or a plugin's `memento:ceiling`: word characters, colons,
# and hyphens, so typing it after a slash cannot type anything but the command. Made only where the model's words are parsed.
CommandName = NewType("CommandName", str)


def identifier(cwd: Path, name: str | None) -> str:
    """A session as it is spoken and addressed: its project, then its name, as one identifier; its project alone before it has a name."""
    return cwd.name if name is None else f"{cwd.name}, {name}"


@dataclass(frozen=True)
class Membership:
    """Which process a session is, and where it works, as the shim recorded it at SessionStart or at the first hook of a session that fired none."""

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


@dataclass(frozen=True)
class Held:
    """At a dialog whose PermissionRequest hook hands holds open, waiting on a reply by voice."""

    on: Blocker
    request: RequestId
    deadline: Instant
    # [LAW:no-ambient-temporal-coupling] the warning is spoken once because speaking it is this
    # value changing, not a timer that could fire twice.
    warned: bool


@dataclass(frozen=True)
class LetGo:
    """At a dialog hands let go of at its deadline, so only the keyboard can answer it now."""

    on: Blocker


# The dialog the session is at, or left behind it, as its hooks tell it: Claude Code's idle ends it.
Dialog = Held | LetGo


# [LAW:one-source-of-truth] whether a session runs, waits at a dialog, or sits at its prompt is Claude Code's status, and
# only a status read moves a session between these. Hooks and records say which turn it is and what it did (see Turn),
# never whether it runs. [LAW:types-are-the-program] each variant carries only what is true under it.
@dataclass(frozen=True)
class Unreported:
    """Heard of, with no status read for it yet."""


@dataclass(frozen=True)
class Idle:
    """Claude Code says the session is at its prompt."""

    stamp: Stamp
    # The turn last heard when the period began. An idle read with another turn heard since begins a new period, though
    # no busy was read between, as for a turn that ran between two reads.
    after: PromptId | None


@dataclass(frozen=True)
class Running:
    """Claude Code says a turn or a `!` command runs, or it waits at a dialog, or it said a status hands does not know."""

    status: Going
    stamp: Stamp
    # When Claude Code last set it idle before this run: a record written before then is of a turn over by then. None
    # when it was first read running, as when hands follows it mid-turn: no idle of it was read, and the tail hands on
    # no record from before the turn it is in.
    idled: Stamp | None


SessionState = Unreported | Idle | Running


def status_stamp(state: SessionState) -> Stamp | None:
    """When Claude Code set the status the session is held in; None when no status has been read for it."""
    match state:
        case Idle(stamp=stamp) | Running(stamp=stamp):
            return stamp
        case Unreported():
            return None


# [LAW:types-are-the-program] the turn as hooks and records tell it, one phase at a time, each carrying the ids the turn
# goes by: its prompt's, or the one Claude went on answering under, and every other id it was read going on under (a
# flush's is taken seconds before Claude answers under it and the turn is moved to it, and a message queued in between
# carries it, 2.1.281). A Stop ends only a turn it names, so one applied late never ends the turn after it.
@dataclass(frozen=True)
class Opened:
    """A turn a prompt or its record opened, and no end of it heard."""

    turn: PromptId
    others: frozenset[PromptId] = frozenset()
    # A message the user sent while it ran, waiting behind it: Claude Code runs it once the turn's Stop hook returns,
    # under an id no hook names (2.1.282).
    queued: bool = False
    # [LAW:types-are-the-program] progress lives only on a turn that runs: whatever a turn gathered and did not tell
    # is let go of with it as it ends, since its result is told instead.
    gathering: Gathering | None = None
    # The call the turn made last, for anyone asking what the session is doing; None before it has made one.
    latest: Doing | None = None


@dataclass(frozen=True)
class Untold:
    """A turn Claude Code said is over and hands has not told yet. Claude Code sets idle before the transcript says how
    the turn ended (an interrupt's record is written 37 ms after, 2.1.283), so the turn is told once that record or its
    Stop is read, or a turn after it opens, or with what was read once the transcript has been read through `by`."""

    turn: PromptId
    others: frozenset[PromptId]
    # On Claude Code's clock, from the idle it set: [LAW:no-ambient-temporal-coupling] how late hands reads the transcript
    # moves when the telling goes out, never what it holds.
    by: Stamp


@dataclass(frozen=True)
class Told:
    """The last turn is told: None until one is. At the prompt, a Stop of this turn ends it again only when the turn went
    on after another Stop hook blocked its Stop (see _turned)."""

    turn: PromptId | None = None
    others: frozenset[PromptId] = frozenset()


Turn = Opened | Untold | Told


def ids(turn: Turn) -> frozenset[PromptId]:
    """Every id the turn goes by: none for the Told a session starts in, before any turn is."""
    return turn.others if turn.turn is None else turn.others | {turn.turn}


@dataclass(frozen=True)
class Unnamed:
    """A Stop under an id no record hands has read names yet. Claude Code writes a turn's records before it fires the
    turn's Stop, and they reach the transcript within ~40 ms of it (2.1.285, measured), so the transcript says whose it
    is: the turn a record carried onto its id, or opened under it. Held until one is read, or until the transcript is
    read through `by` without one."""

    prompt: PromptId
    closing: str | None
    again: bool
    by: Stamp
    # The hook Claude Code waits on meanwhile, let go once the Stop is decided; None once it stopped waiting.
    hook: RequestId | None


@dataclass(frozen=True)
class Session:
    membership: Membership
    state: SessionState
    # [LAW:one-source-of-truth] the permission_mode of the last hook that carried one. None until one does:
    # SessionStart, idle_prompt, and SessionEnd carry none (verified live on 2.1.281).
    mode: Mode | None
    turn: Turn = Told()
    dialog: Dialog | None = None
    # Every id the turns before this one went by, each told as the turn after it replaced it: a late Stop naming one ends
    # nothing, in whatever phase the turn now is. [LAW:types-are-the-program] kept here, and not on the turn, so every
    # phase answers "was this turn told" the same way, across a restart too.
    earlier: frozenset[PromptId] = frozenset()
    # [LAW:no-ambient-temporal-coupling] Stops heard before the records that say whose they are, in the order heard.
    unnamed: tuple[Unnamed, ...] = ()
    # What each subagent did that nobody has been told of yet. Kept here, and not on the turn: a subagent run in the
    # background works on after the turn that started it ends, and its progress is news whatever phase its parent is in.
    subagents: Mapping[AgentTask, Gathering] = field(default_factory=dict[AgentTask, Gathering])


@dataclass(frozen=True)
class Gone:
    """A session that ended. [LAW:types-are-the-program] it has no status, turn, or dialog: its last turn was told and its
    held hook let go as it ended, and it starts again as a new Session."""

    membership: Membership


# Every session the registry has heard of, live or ended.
Known = Session | Gone


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
    sessions: Mapping[SessionId, Known]
    # [LAW:types-are-the-program] a session with no entry has nothing staged; there is no empty draft.
    drafts: Mapping[SessionId, Staged]

    def put(self, session: Known) -> Self:
        return replace(self, sessions={**self.sessions, session.membership.id: session})

    def stage(self, session: SessionId, draft: Staged) -> Self:
        return replace(self, drafts={**self.drafts, session: draft})

    def unstage(self, session: SessionId) -> Self:
        return replace(self, drafts={id: draft for id, draft in self.drafts.items() if id != session})

    def live(self) -> list[Session]:
        return [session for session in self.sessions.values() if isinstance(session, Session)]
