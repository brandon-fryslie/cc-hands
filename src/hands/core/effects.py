"""What the reducer asks the edges to do. Adapters perform these and nothing else."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hands.core.events import SessionEvent
from hands.core.occurrences import Occurrence
from hands.core.progress import Doing
from hands.core.session import Blocker, CommandName, Keystroke, PromptId, PromptText, RequestId, SessionId
from hands.core.tmux import Pane
from hands.core.turn import AgentId, AgentTask


@dataclass(frozen=True)
class Unregistered:
    """An event named a session that never joined, so it changed nothing."""

    event: SessionEvent


@dataclass(frozen=True)
class AfterEnd:
    """An event arrived for a session that had already ended, so it changed nothing."""

    event: SessionEvent


@dataclass(frozen=True)
class Unmatched:
    """A Stop that ended nothing: its turn was told already, or no record naming its id was read by the time the
    transcript was read through it. The Stop itself is the line of the event it came in."""

    session: SessionId
    prompt: PromptId


@dataclass(frozen=True)
class Unclosed:
    """A reply on the wire that ended nothing: its turn was told already, or the session is in no turn its id names. The
    reply itself is the line of the event it came in."""

    session: SessionId
    prompt: PromptId


@dataclass(frozen=True)
class Holding:
    """A Stop under an id no record read yet names, held until one is: Claude Code waits on its hook meanwhile."""

    session: SessionId
    prompt: PromptId


@dataclass(frozen=True)
class Unsettled:
    """A Stop still held when its session ended or started again: nothing read from here on says whose it was."""

    session: SessionId
    prompt: PromptId


@dataclass(frozen=True)
class Overtaken:
    """Subagents launched in the background that an idle Claude Code set after their launch says are over, since it sets
    none while one works: a launch read after that idle, or one whose report had not been read when the idle was."""

    session: SessionId
    agents: frozenset[AgentId]


AuditRecord = Unregistered | AfterEnd | Unmatched | Unclosed | Holding | Unsettled | Overtaken


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
    """hands lets go of the hook without deciding anything: a permission request is left to Claude Code's own dialog,
    and a Stop lets Claude Code go on."""


# [LAW:types-are-the-program] what a person can decide is not what the daemon replies: nothing the user or the
# model says can produce a Withdraw, and answers become an AllowWith only against the question they answer.
Decision = Allow | Deny | Answers | Approve | KeepPlanning
HookReply = Allow | AllowWith | Approve | Deny | Withdraw


@dataclass(frozen=True)
class Reply:
    """The answer to a hook Claude Code waits on, a PermissionRequest or a Stop, sent back down the socket it is waiting on."""

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
    request: RequestId
    on: Blocker
    remaining: float  # seconds


@dataclass(frozen=True)
class Expired:
    """Nobody answered by voice in time: a permission was denied, a question left to its dialog."""

    session: SessionId
    on: Blocker


Announcement = DeadlineNear | Expired


@dataclass(frozen=True)
class Speak:
    """Said as written, with no model in the way."""

    announcement: Announcement


@dataclass(frozen=True)
class Narrate:
    """Handed to the intermediary to explain in its own words, and to act on what the user answers."""

    moment: Asking


@dataclass(frozen=True)
class Progress:
    """A running turn's calls and text, gathered until they settled: how the user hears of them is decided where the focus and the
    session's overlay are read, not here."""

    session: SessionId
    # Whose calls they are: the turn's own, by every id the turn they were made in goes by, so its result is known for
    # theirs wherever it stands; or a subagent's, said as the work of the call that started it.
    of: frozenset[PromptId] | AgentTask
    doings: tuple[Doing, ...]
    # The text the turn wrote among them, as it was displayed; empty when it wrote none.
    written: str


@dataclass(frozen=True)
class Tell:
    """Something a session's hook said happened, for the user to hear as its kind is set: how is decided where the
    settings and the session's overlay are read, not here."""

    session: SessionId
    occurrence: Occurrence


Heard = Speak | Narrate | Progress | Tell


@dataclass(frozen=True)
class Summarise:
    """A session finished a turn: what the tail has not told of it is handed to the model to say.

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
    # Whether this is a turn hands told before, going on after another Stop hook blocked its Stop: read against where
    # the reading of its last Stop found the repository, since no prompt marked where the part going on began.
    again: bool


# What a turn did to the repository it ran in, which no record of the session need name: a formatter, a code
# generator, or a `sed` in a shell command changes files that no step reports.
Repository = Snapshot | Compare

Effect = Audit | Reply | Heard | Story | Repository


@dataclass(frozen=True)
class Text:
    """A prompt, typed into a session's input and sent with Return."""

    prompt: PromptText

    @property
    def typed(self) -> PromptText:
        """The prompt behind a space.

        [LAW:single-enforcer] Claude Code reads `/`, `@` and `!` at the start of a prompt as a command, a file mention
        and shell mode, and behind a space as the characters they are. Always a space, so nothing anywhere asks what a
        prompt starts with. Measured on 2.1.283: the transcript's record keeps the space.
        """
        return PromptText(f" {self.prompt}")


@dataclass(frozen=True)
class Command:
    """A slash command, typed into a session's input as `/name` and its arguments, and sent with Return."""

    name: CommandName
    args: PromptText | None

    @property
    def word(self) -> str:
        """The command with its slash, which is what makes Claude Code run it rather than read it as a prompt."""
        return f"/{self.name}"

    @property
    def typed(self) -> PromptText:
        """The command and its arguments, as they read in the input."""
        match self.args:
            case None:
                return PromptText(self.word)
            case args:
                return PromptText(f"{self.word} {args}")


@dataclass(frozen=True)
class Key:
    """One named chord, pressed in a session as the user would press it."""

    key: Keystroke


# [LAW:types-are-the-program] what is typed into a session, as what it is: the variant decides whether a leading sigil
# is escaped, so nothing that types asks what the input starts with.
Input = Text | Command | Key


@dataclass(frozen=True)
class Fritter:
    """The fritter that wrapped a session: it listens at `socket`, and types only into process `pid`, the one it wrapped."""

    socket: Path
    pid: int


# What types into a session: the fritter that wrapped it, or tmux, into the pane it runs in.
Writer = Fritter | Pane


@dataclass(frozen=True)
class Type[I: Input]:
    """Typed into a session by its writer.

    Not in Effect: a draft, command, or interrupt request emits it, and what came of it is the request's answer.
    """

    session: SessionId
    writer: Writer
    input: I


@dataclass(frozen=True)
class Typed[I: Input]:
    """The session's writer typed the input into it."""

    session: SessionId
    input: I


@dataclass(frozen=True)
class NotTyped[I: Input]:
    """The session's writer could not be reached, or refused, or could not write: `reason` says which."""

    session: SessionId
    input: I
    reason: str
