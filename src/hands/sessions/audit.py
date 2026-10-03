"""The audit log: one JSON line for everything the daemon did, heard, said, and failed at, appended by the daemon alone.

    uv run hands log        # the newest lines, then each new one as it is written

Each line is a value from this module or from the core, encoded the same way: its type's name under
"type" and its fields beside it, nested values alike, with the wall-clock time the line was written
under "at", and whether it tells of something that went wrong under "level", between the two. Nothing here
decides what happened; it records what the rest of the daemon already decided.

[LAW:domain-language] it is a segmented log, in the terms Kafka gave the shape: a directory of segments, each named by
its base offset, the log offset of its first byte. Lines are appended to the active segment; a line that would take it
past SEGMENT_BYTES rolls the log to a new segment, which opens with a Rolled line, and retention deletes every segment
older than the one just closed. Two segments are kept, so the log never holds more than twice SEGMENT_BYTES, unless
one line alone is longer. A segment is never renamed, and once a later one exists it is closed: nothing is appended
to it again.
"""

import fcntl
import json
import os
import re
import traceback
from bisect import bisect_right
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal, assert_never, cast

from loguru import logger

if TYPE_CHECKING:
    # Defined only in loguru's type stubs.
    from loguru import Message

from hands.core.attention import Amount, Attention, Delivery, EndedRoute, Overlay, Route
from hands.core.delta import Branched, PullRequested, Pushed
from hands.core.effects import AfterEnd, Allow, AuditRecord, Deny, Effect, Heard, Holding, Input, Type, Unclosed, Unmatched, Unregistered, Unsettled
from hands.core.events import Event
from hands.core.session import SessionId
from hands.core.wire import Exchanged, Garbled, Held, Reached, Uncopied, Unreached
from hands.sessions.model_facts import ModelFact


@dataclass(frozen=True)
class Applied:
    """An event that changed the registry or called for an effect. One that did neither, such as a quiet tick, is not a line."""

    event: Event


@dataclass(frozen=True)
class Performed:
    effect: Effect


@dataclass(frozen=True)
class Typing:
    """What is about to be typed into a session: written before the typing, so the log holds every send."""

    effect: Type[Input]


@dataclass(frozen=True)
class EffectFailed:
    effect: Effect
    error: str


@dataclass(frozen=True)
class LLMChosen:
    """The model a run reaches and the server it reaches it on, from the configuration it started with, and the brain's
    subscription account (None for a keyed variant); never its key."""

    backend: str
    base_url: str
    model: str
    account: str | None


@dataclass(frozen=True)
class SettingsRead:
    """Where a run's settings came from: the home's config.toml, or None where it has none and every setting is its
    default; and the Whisper model they name. The backend they name is LLMChosen."""

    path: str | None
    whisper_model: str


@dataclass(frozen=True)
class VoiceChosen:
    """The voice a run starts speaking in: the one the user kept, or the default where they kept none."""

    voice: str


@dataclass(frozen=True)
class ProxyListening:
    """Where the wire proxy took requests for this run: the url a Claude Code process's ANTHROPIC_BASE_URL is set to."""

    url: str
    upstream: str


@dataclass(frozen=True)
class TapListening:
    """Where hands took the copies of wrapped sessions' exchanges for this run: the socket each session's fritter dials."""

    path: Path


@dataclass(frozen=True)
class DisplayListening:
    """Where hands took the text Claude Code displays for this run: the URL the plugin's MessageDisplay hook posts to."""

    url: str


@dataclass(frozen=True)
class CopiesLost:
    """Copies of a session's exchanges that its fritter could not hand to hands, told with the first copy since that it
    did: hands was down, or was not reading them. The session's own exchanges went on regardless."""

    session: str | None
    lost: int


@dataclass(frozen=True)
class McpConnected:
    """A client of hands' MCP server opened with it: who it says it is, and the protocol version it asked for."""

    client: Mapping[str, object]
    protocol: str


@dataclass(frozen=True)
class BrainLaunched:
    """The brain's process started, with the login and working directory it was given."""

    pid: int
    config_dir: Path
    cwd: Path
    model: str


@dataclass(frozen=True)
class BrainOffered:
    """The tools the brain's requests offer the model, from the first of its requests and each that offers others: what
    its own setup gave it, beside hands' tools."""

    tools: tuple[str, ...]


@dataclass(frozen=True)
class BrainRefused:
    """A dialog the brain's Claude Code would have opened, answered no by hands at its hook: nobody is at its keyboard.
    A permission is a BrainPermission, held while the user is asked."""

    prompt: str | None  # the prompt id of the turn that asked; none for what asked between turns
    dialog: str  # its hook: Elicitation, from an MCP server
    asker: str | None  # what asked, as the hook names it: the MCP server


@dataclass(frozen=True)
class BrainPermission:
    """A permission the brain's own setup asked about, held at its hook while the user was asked, and how it was settled:
    by their plain yes, by their other words, by nobody answering in time, or by the turn's end. `seconds` is how long it was held."""

    prompt: str | None  # the prompt id of the turn that asked
    tool: str
    decision: Allow | Deny
    seconds: float


@dataclass(frozen=True)
class BrainAsked:
    """The typed end of a brain turn: what was typed into it, at this line's time."""

    text: str


@dataclass(frozen=True)
class BrainAnswered:
    """The end of a brain turn, at this line's time: its Stop hook, its StopFailure hook with what failed it, or the Escape
    hands pressed to stop it. Its words are on the wire."""

    prompt: str  # the prompt id Claude Code gave the turn, which every hook of the turn and its transcript records carry
    error: str | None


@dataclass(frozen=True)
class AsideAnswered:
    """A side question hands asked in the background, of a slim Claude Code started for it alone, and its answer from
    the wire, or why it has none. `session` is that Claude Code's, which its exchanges on the wire carry. `waited` is how
    long, in seconds, the question waited behind the ones asked before it, and `seconds` how long its Claude Code ran: 0 for a question whose asker left before its turn, which
    had none."""

    question: str
    reply: str
    failed: bool
    session: SessionId
    waited: float
    seconds: float


@dataclass(frozen=True)
class ResultsStubbed:
    """The brain's history crossed a batch boundary: the calls whose results go as a line from now on, and those that
    go whole because no sentence had been said of them by then."""

    stubbed: tuple[str, ...]
    unsaid: tuple[str, ...]


@dataclass(frozen=True)
class BrainInterrupted:
    """The user barged in on a brain turn: the tools it had in flight, and whether it was told to stop at once.

    Not stopped means a tool whose effect must land was running: it runs to its end, and the turn's next request is held.
    """

    running: tuple[str, ...]
    stopped: bool


# Whose turn the brain answered: the user's words, or what hands handed it to tell.
Asker = Literal["user", "hands"]


@dataclass(frozen=True)
class BrainSpoke:
    """What a brain turn handed to the speaker, and the exchanges on the wire its words came from; `readbacks` is what hands
    said for it once its next request was held. `asker` is whose turn it was: the user's words, or hands' narration.
    `waited` is how long, in seconds, the turn waited in its lane for the brain before it was written. `failed` is what the
    turn failed of: the error its latest answer on the wire said, or a model that answered it with nothing; None for a
    turn that did not fail."""

    exchanges: tuple[str, ...]
    text: str
    readbacks: tuple[str, ...]
    interrupted: bool
    asker: Asker
    waited: float
    failed: ModelFact | None


@dataclass(frozen=True)
class BrainExited:
    """The brain's process ended: its exit code, and the last of what it showed on its terminal."""

    code: int
    shown: str


@dataclass(frozen=True)
class Transcribed:
    """What the user said in one turn, as it went into the intermediary's context."""

    text: str


@dataclass(frozen=True)
class Unsaid:
    """A segment Whisper transcribed and hands dropped as not said, with the scores it was dropped for."""

    text: str
    no_speech_prob: float
    compression_ratio: float
    avg_logprob: float


@dataclass(frozen=True)
class HoldHeard:
    """What Whisper made of one hold: what it took as said, None where nothing was, and each segment it dropped."""

    hold: int
    said: str | None
    dropped: tuple[Unsaid, ...]


@dataclass(frozen=True)
class Primed:
    """The words Whisper was primed with for one hold, oldest first, and how long reading them took. `focus` is the
    session whose repository was read, None where no running session is focused; `failed` says why the focus or its
    repository gave no words where either could not be read."""

    focus: SessionId | None
    words: tuple[str, ...]
    failed: str | None
    seconds: float


@dataclass(frozen=True)
class Replied:
    """What the intermediary said in one turn."""

    text: str
    interrupted: bool


@dataclass(frozen=True)
class CutOff:
    """The user barged in: the sentence that was playing, None when the speaker was quiet, and how many cut-off readings
    are now waiting to be gone back to."""

    sentence: str | None
    waiting: int


@dataclass(frozen=True)
class Called:
    """A tool the intermediary called, with the arguments it gave and the result it was handed back."""

    tool: str
    arguments: Mapping[str, object]
    result: Mapping[str, object]


@dataclass(frozen=True)
class Announced:
    """A fact the system channel gave the user, and whether it was spoken or, with speech down, posted to the screen."""

    text: str
    via: Literal["speech", "screen"]


@dataclass(frozen=True)
class Yielded:
    """What hands had to say unprompted, let go by the floor: the kind of each thing that came, in the order it came,
    the kind of each thing told of it, in the order told, and how long the user's turn held it, 0 when no turn was
    open. `held` longer than `told` is something folded into a session's telling or no longer waiting on the user."""

    held: tuple[str, ...]
    told: tuple[str, ...]
    waited: float


@dataclass(frozen=True)
class Relayed:
    """Something a session said to the user, passed on to the pipeline."""

    heard: Heard


@dataclass(frozen=True)
class Routed:
    """Which way a session's progress went: played, briefly or in full, for the focused session, or noted for the
    model; and what hands was set to say unprompted, the focus, and the overlay that decided it, as they were read."""

    session: SessionId
    attention: Attention
    focused: bool
    overlay: Overlay
    route: Route


@dataclass(frozen=True)
class EndedRouted:
    """Whether a session's ending was said, or left to the log for catch_up; and what hands was set to say unprompted
    that decided it, as it was read."""

    session: SessionId
    attention: Attention
    route: EndedRoute


RefocusOutcome = Literal["moved", "ended", "failed"]


@dataclass(frozen=True)
class Refocused:
    """A session's turn or question was told, so the focus moved to it: what the user says next is taken first as said to
    that session. `outcome` says what came of it: `moved` focused it; `ended` found it no longer running, so the focus
    stayed; `failed` could not write the focus, which `failed` says why, and the focus stayed."""

    session: SessionId
    outcome: RefocusOutcome
    failed: str | None


@dataclass(frozen=True)
class ProgressTold:
    """Progress routed to be played, as it went to the floor: how much text it held, the clause that text was summarised
    as or why it could not be, and whether its turn still ran once it was ready, since progress of a turn that has ended
    is not played: its result is told instead."""

    session: SessionId
    written: int  # characters of text, 0 for a burst of calls alone
    explained: str | None
    failed: str | None
    current: bool
    # How much of it the user set to be said: briefly, a burst is said by its text alone, and one with none is not said.
    amount: Amount


@dataclass(frozen=True)
class Recounted:
    """What hands told of a turn a session finished, and what the narration left the user able to ask for.

    `reply` is the last thing the session said in the turn and `facts` what hands adds from its record: what the model
    is handed to say in its own words, under the session's name as it is when told. `delivered` is how it reached the
    user and what decided it: told as the turn finished, and how much of it, or held for when they ask (tell_turn).
    `topics` is every part of the turn's narration that was built and not played — its sections, and what it
    asked through a dialog and is no longer waiting on — which makes this line the
    one place a developer who cannot see the screen can find out what "more on that" has to open. `questions`
    is what the session is waiting on an answer to. `opened`
    is the kind of thing that opened the turn — Asked, Notified, Commanded, or Shelled — so a turn the user's own
    command opened is told apart from one they asked for. `subagents` names each subagent that reported back in the
    turn and whose own transcript was read for it, and `unread` each one whose transcript could not be read, so a turn
    told without a subagent's work is told apart from one no subagent reported back to.
    """

    session: str
    reply: str | None
    facts: str
    topics: tuple[str, ...]
    questions: tuple[str, ...]
    delivered: Delivery
    opened: str
    subagents: tuple[str, ...]
    unread: tuple[str, ...]


@dataclass(frozen=True)
class Summarised:
    """One pass of the summary store over a project's backlog: how much of it was already said, and what this pass said.

    `unsaid` is what is still without a sentence when the pass ends: things the summariser failed on or left out, and
    the parents above them. `left_out` names what a reply gave no sentence for, and `stray` counts the reply lines
    that named nothing asked, so a model that skips items reads apart from one whose calls failed.
    """

    project: str
    outcome: Literal["said", "partial"]
    things: int
    known: int
    said: int
    unsaid: int
    rounds: int
    calls: int
    failed_calls: int
    left_out: tuple[str, ...]
    stray: int
    seconds: float


@dataclass(frozen=True)
class TurnsSummarised:
    """One pass of the summary store over finished turns of a session that read_session found unsaid.

    `known` is how many turns it was handed that had been said since they were queued, and `asked` how many it asked
    the summariser for; `left_out` names those a reply gave no sentence for, and `stray` counts the reply lines that
    named nothing asked.
    """

    session: str
    outcome: Literal["said", "partial"]
    known: int
    asked: int
    said: int
    calls: int
    failed_calls: int
    left_out: tuple[str, ...]
    stray: int
    seconds: float


# Whether the forge was asked about the pull requests of the branch a turn pushed, and what came of it: not asked
# where the turn pushed nothing or pushed its remote's default branch, which no pull request is opened from; absent
# where there is no gh to ask; unanswered where it was too slow, and refused where gh would not list or answered in
# a shape hands does not read.
Forge = Literal["unasked", "absent", "answered", "unanswered", "refused"]

DeltaReadOutcome = Literal["unmarked", "dropped", "read", "failed", "cancelled"]


@dataclass(frozen=True)
class DeltaRead:
    """One reading of what a turn changed in its repository, one per turn that stopped.

    `outcome` is "unmarked" where nothing marked where the turn began (no repository, or one that could not be read),
    "dropped" where too many readings were already waiting to be told, "read" where git answered, and "failed" or
    "cancelled" where the reading did not finish. `seconds` is what the narrator may have waited through for it.
    """

    session: str
    outcome: DeltaReadOutcome
    commits: int
    files: int
    changes: tuple[Pushed | Branched | PullRequested, ...]
    forge: Forge
    forge_seconds: float
    seconds: float


NamingOutcome = Literal["renamed", "kept", "unread", "failed", "refused"]


@dataclass(frozen=True)
class Named:
    """One judging of a session's name after a turn it finished.

    `outcome` says what came of it: `renamed` decided a new name, given at the session's next prompt; `kept` found the
    name it has still fits; `unread` could not read the name it has; `failed` had no answer from the model; `refused`
    had an answer that is not a name of three words at most, which `reply` holds.
    """

    session: str
    outcome: NamingOutcome
    before: str | None
    name: str | None
    reply: str | None
    error: str | None
    seconds: float


@dataclass(frozen=True)
class NameGiven:
    """A name handed to Claude Code in the reply to a session's prompt, which sets the session's title."""

    session: str
    name: str


@dataclass(frozen=True)
class NameWithheld:
    """A name hands decided and did not give at the session's prompt: its title is no longer the one the name was decided
    against, `held` (set since, by the user's /rename), or it could not be read, which `error` says."""

    session: str
    name: str
    against: str | None
    held: str | None
    error: str | None


@dataclass(frozen=True)
class BacklogUnread:
    """A pass of the summary store that could not start: lit would not hand over the project's backlog, and why."""

    project: str
    error: str
    seconds: float


@dataclass(frozen=True)
class SettingsEdited:
    """The home's config.toml changed while hands ran: the run restarts on it, or, where it was `refused`, says why
    and runs on the settings it started with."""

    path: str
    refused: str | None


@dataclass(frozen=True)
class Restarting:
    """The daemon was asked to restart, or its settings were edited (SettingsEdited): it has stopped, and starts again
    as pid `pid`, the same process, from the code and configuration on disk now."""

    pid: int


@dataclass(frozen=True)
class Rolled:
    """The first line of a segment: the log rolled to it at log offset `base`, and retention deleted the segments whose
    base offsets are `deleted`."""

    base: int
    deleted: tuple[int, ...]


@dataclass(frozen=True)
class Failure:
    """An error the daemon logged: the module and function that logged it, what it said, the file and line it was logged
    at, and, when it was logged with an exception, that exception and each it was raised from or while handling, the
    first cause first, each named and then followed by the frames it came up through, the raising one last."""

    source: str
    message: str
    where: str
    trace: tuple[str, ...]


Entry = (
    AuditRecord
    | Applied
    | Performed
    | Typing
    | EffectFailed
    | LLMChosen
    | SettingsRead
    | SettingsEdited
    | VoiceChosen
    | ProxyListening
    | TapListening
    | DisplayListening
    | CopiesLost
    | Exchanged
    | McpConnected
    | BrainLaunched
    | BrainOffered
    | BrainRefused
    | BrainPermission
    | BrainAsked
    | BrainAnswered
    | AsideAnswered
    | ResultsStubbed
    | BrainInterrupted
    | BrainSpoke
    | BrainExited
    | Transcribed
    | HoldHeard
    | Primed
    | Replied
    | CutOff
    | Called
    | Announced
    | Yielded
    | Relayed
    | Routed
    | EndedRouted
    | Refocused
    | ProgressTold
    | Recounted
    | Summarised
    | TurnsSummarised
    | DeltaRead
    | Named
    | NameGiven
    | NameWithheld
    | BacklogUnread
    | Restarting
    | Rolled
    | Failure
)
Record = Callable[[Entry], None]
Level = Literal["error", "info"]


def level(entry: Entry) -> Level:
    """Whether a line tells of something that went wrong: a Failure; an effect, a backlog read, a turn's reading or a
    name that failed; an exchange the API refused or never answered, whose stream hands could not read, or whose copy
    broke off; a tool that answered with an error; or a brain turn or side question that came to nothing."""
    # [LAW:one-source-of-truth] the one place a line is judged an error, so a reader finds every error by one field and
    # never by an "error" deep in a body the API sent. [LAW:types-are-the-program] every kind of line is named here,
    # so a record added to Entry is judged here before pyright passes, rather than read as info by default.
    match entry:
        case Failure() | EffectFailed() | BacklogUnread():
            return "error"
        case Exchanged(reply=reply):
            return _reply_level(reply)
        case BrainAnswered(error=error) | NameWithheld(error=error):
            return "info" if error is None else "error"
        case BrainSpoke(failed=fact):
            return "info" if fact is None else "error"
        case AsideAnswered(failed=failed):
            return "error" if failed else "info"
        case Called(result=result):
            return "error" if "error" in result else "info"
        case DeltaRead(outcome=outcome):
            return "error" if outcome == "failed" else "info"
        case Named(outcome=outcome):
            return "error" if outcome in ("unread", "failed", "refused") else "info"
        case ProgressTold(failed=failed):
            return "info" if failed is None else "error"
        case Refocused(outcome=outcome):
            return "error" if outcome == "failed" else "info"
        case Primed(failed=failed) | SettingsEdited(refused=failed):
            return "info" if failed is None else "error"
        case (
            Unregistered() | AfterEnd() | Unmatched() | Unclosed() | Holding() | Unsettled()
            | Applied() | Performed() | Typing() | LLMChosen() | SettingsRead() | VoiceChosen() | ProxyListening() | TapListening() | DisplayListening() | CopiesLost()
            | McpConnected() | BrainLaunched() | BrainOffered() | BrainRefused() | BrainPermission() | BrainAsked() | ResultsStubbed() | BrainInterrupted() | BrainExited()
            | Transcribed() | HoldHeard() | Replied() | CutOff() | Announced() | Yielded() | Relayed() | Routed() | EndedRouted() | Recounted() | Summarised()
            | TurnsSummarised() | NameGiven() | Restarting() | Rolled()
        ):
            return "info"
        case _:
            assert_never(entry)


def _reply_level(reply: Reached | Unreached | Held | Uncopied) -> Level:
    match reply:
        case Unreached() | Uncopied() | Reached(body=Garbled()):
            return "error"
        case Reached(status=status):
            return "error" if status >= 400 else "info"
        case Held():
            return "info"


# How large a segment grows before the log rolls: two on disk at most, a few days of every session's exchanges.
SEGMENT_BYTES = 32 * 1024 * 1024

_SEGMENT = re.compile(r"(\d{20})\.jsonl")
# The segments as a shell glob matches them, and no other file in the directory.
SEGMENT_GLOB = f"{'[0-9]' * 20}.jsonl"


def segment(directory: Path, base: int) -> Path:
    """The segment whose first byte is at log offset base."""
    return directory / f"{base:020d}.jsonl"


def segments(directory: Path) -> list[int]:
    """The base offsets of the log's segments, oldest first: none while there is no log."""
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return []
    return sorted(int(found[1]) for name in names if (found := _SEGMENT.fullmatch(name)))


class AuditLog:
    def __init__(self, directory: Path, clock: Callable[[], datetime], segment_bytes: int = SEGMENT_BYTES) -> None:
        # The log holds what every tapped session said, as the tap's socket does: the user's alone to read.
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o700)
        self._directory = directory
        self._clock = clock
        self._segment_bytes = segment_bytes

    def record(self, entry: Entry) -> None:
        # [LAW:single-enforcer] the log watches what the daemon does and never changes it: a line it cannot encode or
        # write is lost here, not a send's answer, a tool's result, a permission's question, or a background task.
        try:
            body = _body(entry)
        except TypeError as error:
            # [LAW:no-silent-failure] a bug in what was recorded: logged as an error, it is a Failure line, whose fields always encode.
            logger.error(f"the audit log cannot encode a {type(entry).__name__} line: {error}")
            return
        try:
            # Made again for each line, as a segment's file is: a log deleted under a running daemon begins again.
            self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            with _exclusive(self._directory):
                # [LAW:no-ambient-temporal-coupling] read under the lock that orders the writes, and once for a line and the
                # Rolled line in front of it, so "at" never runs backwards down the log.
                at = json.dumps(self._clock().isoformat(timespec="milliseconds"))
                line = _stamped(body, at)
                # [LAW:one-source-of-truth] the active segment is the newest on disk, listed for each line: no writer holds
                # a copy of it that another writer's roll, or a roll that failed partway, could leave behind.
                bases = segments(self._directory)
                active = max(bases, default=0)
                size = _size(segment(self._directory, active))
                rolled = Rolled(active + size, tuple(bases[:-1])) if size > 0 and size + len(line) > self._segment_bytes else None
                if rolled is not None:
                    active, line = rolled.base, _stamped(_body(rolled), at) + line
                # Opened for each line, so a line is on disk when record returns.
                with open(segment(self._directory, active), "a+b", opener=_private) as log:
                    log.write(_ending(log) + line)
                # Retention deletes after the Rolled line naming what it deletes is on disk: a roll that failed deleted nothing.
                for base in () if rolled is None else rolled.deleted:
                    segment(self._directory, base).unlink(missing_ok=True)
        except OSError as error:
            # [LAW:no-silent-failure] said on stderr, as a warning: an error would be sent back to the log that just failed.
            logger.warning(f"the audit log {self._directory} failed at a {type(entry).__name__} line: {error}")


@contextmanager
def _exclusive(directory: Path) -> Generator[None]:
    """One writer at a time, across threads and processes: every line is on disk before the roll that closes its segment,
    which is what lets a reader trust a closed one. An flock on the directory, let go as it is closed."""
    # [LAW:single-enforcer] the one lock on the log's writes, for a second daemon as for a second thread.
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _ending(log: BinaryIO) -> bytes:
    """A newline to end a line that a write which failed partway left torn at the end of log, so the next line is whole."""
    end = log.seek(0, os.SEEK_END)
    if end == 0:
        return b""
    log.seek(end - 1)
    return b"" if log.read(1) == b"\n" else b"\n"


def _stamped(body: str, at: str) -> bytes:
    # Half of a character cut in two is no UTF-8: it is written as its JSON escape, which reads back as itself.
    return f'{{"at": {at}, {body[1:]}\n'.encode("utf-8", errors="backslashreplace")


def _body(entry: Entry) -> str:
    """The line for entry without its time, which is written in front of it as it goes onto the log."""
    return json.dumps({"level": level(entry), **encoded(entry)}, ensure_ascii=False)


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _private(path: str, flags: int) -> int:
    return os.open(path, flags, 0o600)


def encoded(value: object) -> dict[str, object]:
    """A dataclass as JSON: its type's name under "type", then its fields."""
    # [LAW:dataflow-not-control-flow] one encoding for every entry, event, and effect, so a new variant needs no code here.
    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError(f"an audit entry is a dataclass, not {type(value).__name__}")
    return {"type": type(value).__name__, **{field.name: _json(getattr(value, field.name)) for field in fields(value)}}


def _json(value: object) -> object:
    match value:
        case None | bool() | int() | float() | str():
            return value
        case Path():
            return str(value)
        case Enum():
            return _json(value.value)
        case Mapping():
            return {str(key): _json(item) for key, item in cast(Mapping[object, object], value).items()}
        case list() | tuple():
            return [_json(item) for item in cast(list[object] | tuple[object, ...], value)]
        case set() | frozenset():
            # Sorted, so a set is written the same way on every line it is on.
            return sorted((_json(item) for item in cast(set[object] | frozenset[object], value)), key=repr)
        case _ if is_dataclass(value) and not isinstance(value, type):
            return encoded(value)
        case _:
            # [LAW:no-silent-failure] a value the log cannot write is a bug to hear about, not a line to guess at.
            raise TypeError(f"the audit log cannot encode a {type(value).__name__}: {value!r}")


def failures_to(record: Record) -> "Callable[[Message], None]":
    """A loguru sink that writes every error it is given as a Failure line."""

    def sink(message: "Message") -> None:
        logged = message.record
        exception = logged["exception"]
        detail = "" if exception is None or exception.value is None else f": {type(exception.value).__name__}: {exception.value}"
        record(
            Failure(
                source=f"{logged['name']}:{logged['function']}",
                message=f"{logged['message']}{detail}",
                where=f"{logged['file'].path}:{logged['line']}",
                trace=() if exception is None or exception.value is None else _trace(exception.value),
            )
        )

    return sink


def _trace(error: BaseException) -> tuple[str, ...]:
    chain: list[BaseException] = []
    link: BaseException | None = error
    while link is not None and link not in chain:
        chain.append(link)
        link = link.__cause__ or (None if link.__suppress_context__ else link.__context__)
    return tuple(line for cause in reversed(chain) for line in (f"{type(cause).__name__}: {cause}", *_frames(cause)))


def _frames(error: BaseException) -> tuple[str, ...]:
    # Without the source lines, which nothing here reads: looking them up opens every frame's file inside the sink.
    frames = traceback.StackSummary.extract(traceback.walk_tb(error.__traceback__), lookup_lines=False)
    return tuple(f"{frame.filename}:{frame.lineno} in {frame.name}" for frame in frames)


def tail(directory: Path, count: int) -> tuple[list[str], int]:
    """The last count complete lines of the log, and the log offset just past them, where following it begins."""
    bases = segments(directory)
    if not bases:
        return [], 0
    lines, end = _lines(segment(directory, bases[-1]), 0)
    for base in reversed(bases[:-1]):
        if len(lines) >= count:
            break
        lines = _lines(segment(directory, base), 0)[0] + lines
    return (lines[-count:] if count > 0 else []), bases[-1] + end


def backwards(directory: Path) -> Iterator[str]:
    """Every complete line of the log, newest first, reading an older segment only once the caller asks past the newer."""
    for base in reversed(segments(directory)):
        yield from reversed(_lines(segment(directory, base), 0)[0])


def follow(directory: Path, offset: int, poll: Callable[[], None]) -> Iterator[str]:
    """Each complete line written to the log past offset, calling poll between reads, for as long as the caller asks."""
    while True:
        lines, offset = _past(directory, offset)
        yield from lines
        poll()


def _past(directory: Path, offset: int) -> tuple[list[str], int]:
    """The complete lines past offset in the segment that holds it, and the log offset to read from next."""
    # [LAW:no-ambient-temporal-coupling] listed before the segment is read: a segment with a later one in this listing was
    # closed before it, so the read has every line the segment will ever hold, and reading goes on at the later one.
    bases = segments(directory)
    if not bases:
        return [], offset
    # An offset this log never handed out - behind retention, or past the end of or mid-line in a log begun again from
    # zero - goes on at the oldest segment kept. Behind retention, the Rolled line naming the segment missed is ahead. A
    # log begun again that has a line ending just before the offset is read on from there, its start unseen.
    held = bisect_right(bases, offset)
    if held == 0 or not _starts_a_line(segment(directory, bases[held - 1]), offset - bases[held - 1]):
        offset, held = bases[0], 1
    base = bases[held - 1]
    lines, end = _lines(segment(directory, base), offset - base)
    return lines, (bases[held] if held < len(bases) else base + end)


def _starts_a_line(path: Path, at: int) -> bool:
    """Whether a line of path begins at byte at: every offset a reader is given is a segment's base or follows a newline."""
    if at == 0:
        return True
    try:
        with path.open("rb") as log:
            log.seek(at - 1)
            return log.read(1) == b"\n"
    except FileNotFoundError:
        return False


def _lines(path: Path, start: int) -> tuple[list[str], int]:
    """The complete lines in path from start on, and the offset just past them: a line still being written is left for
    the next read. A segment retention has deleted holds none."""
    try:
        with path.open("rb") as log:
            log.seek(start)
            data = log.read()
    except FileNotFoundError:
        return [], start
    end = data.rfind(b"\n") + 1
    # A write that failed partway can cut a character in two; its torn line reads with U+FFFD in place of the half, and
    # is then no JSON, as every torn line is, rather than taking every other line of the segment down with it.
    return data[:end].decode("utf-8", errors="replace").splitlines(), start + end
