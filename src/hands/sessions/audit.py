"""The audit log: one JSON line for everything the daemon did, heard, said, and failed at, appended by the daemon alone.

    uv run hands log        # the newest lines, then each new one as it is written

Each line is a value from this module or from the core, encoded the same way: its type's name under
"type" and its fields beside it, nested values alike, with the wall-clock time the line was written
under "at", and whether it tells of something that went wrong under "level", between the two. Nothing here
decides what happened; it records what the rest of the daemon already decided.

Log rotation keeps it bounded: a line that would take the log past LIMIT bytes first moves it to its retired name,
replacing the one retired before, and starts a new log with a Retired line. The two never hold more than twice LIMIT,
unless one line alone is longer than LIMIT.
"""

import json
import os
import threading
import traceback
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal, assert_never, cast

from loguru import logger

if TYPE_CHECKING:
    # Defined only in loguru's type stubs.
    from loguru import Message

from hands.core.attention import Delivery
from hands.core.delta import Branched, PullRequested, Pushed
from hands.core.effects import AfterEnd, AuditRecord, Effect, Heard, Holding, Input, Type, Unclosed, Unmatched, Unregistered, Unsettled
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
class ProxyListening:
    """Where the wire proxy took requests for this run: the url a Claude Code process's ANTHROPIC_BASE_URL is set to."""

    url: str
    upstream: str


@dataclass(frozen=True)
class TapListening:
    """Where hands took the copies of wrapped sessions' exchanges for this run: the socket each session's fritter dials."""

    path: Path


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
    """What hands had to say unprompted while the user's turn was open, passed on as it closed: each frame's kind, in
    order, and how long the turn held the floor."""

    held: tuple[str, ...]
    waited: float


@dataclass(frozen=True)
class Relayed:
    """Something a session said to the user, passed on to the pipeline."""

    heard: Heard


@dataclass(frozen=True)
class Recounted:
    """What hands told of a turn a session finished, and what the narration left the user able to ask for.

    `told` is the summary the model is handed to say in its own words, and `delivered` how it reached the user: told
    as the turn finished, or held for when they ask (tell_turn).
    `topics` is every part of the turn's narration that was built and not played — its sections, and what it
    asked through a dialog and is no longer waiting on — which makes this line the
    one place a developer who cannot see the screen can find out what "more on that" has to open. `questions`
    is what the session is waiting on an answer to, kept apart so a log reader need not find it in `told`. `opened`
    is the kind of thing that opened the turn — Asked, Notified, Commanded, or Shelled — so a turn the user's own
    command opened is told apart from one they asked for.
    """

    session: str
    told: str
    topics: tuple[str, ...]
    questions: tuple[str, ...]
    delivered: Delivery
    opened: str


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
class Restarting:
    """The daemon was asked to restart: it has stopped, and starts again as pid `pid`, the same process, from the code
    and configuration on disk now."""

    pid: int


@dataclass(frozen=True)
class Retired:
    """The first line of a new log: the one before it reached the bound and was moved to `path` at `size` bytes."""

    path: Path
    size: int


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
    | ProxyListening
    | TapListening
    | CopiesLost
    | Exchanged
    | McpConnected
    | BrainLaunched
    | BrainAsked
    | BrainAnswered
    | AsideAnswered
    | ResultsStubbed
    | BrainInterrupted
    | BrainSpoke
    | BrainExited
    | Transcribed
    | Replied
    | CutOff
    | Called
    | Announced
    | Yielded
    | Relayed
    | Recounted
    | Summarised
    | TurnsSummarised
    | DeltaRead
    | Named
    | NameGiven
    | NameWithheld
    | BacklogUnread
    | Restarting
    | Retired
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
        case (
            Unregistered() | AfterEnd() | Unmatched() | Unclosed() | Holding() | Unsettled()
            | Applied() | Performed() | Typing() | LLMChosen() | ProxyListening() | TapListening() | CopiesLost()
            | McpConnected() | BrainLaunched() | BrainAsked() | ResultsStubbed() | BrainInterrupted() | BrainExited()
            | Transcribed() | Replied() | CutOff() | Announced() | Yielded() | Relayed() | Recounted() | Summarised()
            | TurnsSummarised() | NameGiven() | Restarting() | Retired()
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


# How large the log grows before it is retired: two of these on disk at most, a few days of every session's exchanges.
LIMIT = 32 * 1024 * 1024


def retired(log: Path) -> Path:
    """Where the log is moved when it reaches the bound: the lines just older than its own."""
    return log.with_name(f"{log.name}.1")


class AuditLog:
    def __init__(self, path: Path, clock: Callable[[], datetime], limit: int = LIMIT) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._clock = clock
        self._limit = limit
        # [LAW:no-ambient-temporal-coupling] errors are recorded from whichever thread logged them; one writer at a time
        # means no line lands in a log after it was retired, where a reader that has moved on would never see it.
        self._writing = threading.Lock()
        # The log holds what every tapped session said, as the tap's socket does: the user's alone to read.
        if path.exists():
            path.chmod(0o600)

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
            with self._writing:
                # [LAW:no-ambient-temporal-coupling] read under the lock that orders the writes, and once for a line and the
                # Retired line in front of it, so "at" never runs backwards down the log.
                at = json.dumps(self._clock().isoformat(timespec="milliseconds"))
                line = _stamped(body, at)
                size = _size(self._path)
                if size > 0 and size + len(line) > self._limit:
                    os.replace(self._path, retired(self._path))
                    line = _stamped(_body(Retired(retired(self._path), size)), at) + line
                # Opened for each line, so a line is on disk when record returns and a log moved aside is started again.
                with open(self._path, "ab", opener=_private) as log:
                    log.write(line)
        except OSError as error:
            # [LAW:no-silent-failure] said on stderr, as a warning: an error would be sent back to the log that just failed.
            logger.warning(f"the audit log {self._path} lost a {type(entry).__name__} line: {error}")


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


@dataclass(frozen=True)
class Position:
    """How far into which file the log has been read. A file with another inode is another log, whatever its size."""

    inode: int  # 0 while there is no log
    offset: int


START = Position(inode=0, offset=0)


def tail(path: Path, count: int) -> tuple[list[str], Position]:
    """The last count whole lines of the log and the one retired before it, and the position just past them, where
    following it begins."""
    lines, position = _read(path, START)
    if len(lines) < count:
        older, was = _read(retired(path), START)
        # The file under the retired name may be the log just read, moved aside since: its lines are already here.
        lines = (older if was.inode != position.inode else []) + lines
    return (lines[-count:] if count > 0 else []), position


def follow(path: Path, position: Position, poll: Callable[[], None]) -> Iterator[str]:
    """Each whole line written to the log past position, calling poll between looks, for as long as the caller asks."""
    while True:
        lines, now = _read(path, position)
        # A log that moved since the last look left its last lines under its retired name, and takes no more once the
        # new one has been opened. One that did not move has them all in lines. One retired twice between looks is gone,
        # with its last lines.
        yield from _rest(retired(path), position) if now.inode != position.inode else []
        yield from lines
        position = now
        poll()


def _rest(path: Path, since: Position) -> list[str]:
    """The whole lines past since when path is the file since was read in; none when it is another, or there is none."""
    try:
        with path.open("rb") as log:
            if os.fstat(log.fileno()).st_ino != since.inode:
                return []
            return _whole_lines(log, since.offset)[0]
    except FileNotFoundError:
        return []


def _read(path: Path, since: Position) -> tuple[list[str], Position]:
    try:
        with path.open("rb") as log:
            status = os.fstat(log.fileno())
            # [LAW:one-source-of-truth] the offset means something only in the file it was read from, just past a newline.
            # A log moved aside, or cut short in place, is read again from its first line, never from the middle of one.
            # One cut short and regrown past the offset between looks, with a newline where the old one was, is not
            # told apart: lines are skipped, but none is split.
            same = status.st_ino == since.inode and status.st_size >= since.offset
            if same and since.offset > 0:
                log.seek(since.offset - 1)
                same = log.read(1) == b"\n"
            lines, end = _whole_lines(log, since.offset if same else 0)
    except FileNotFoundError:
        return [], START
    return lines, Position(status.st_ino, end)


def _whole_lines(log: BinaryIO, offset: int) -> tuple[list[str], int]:
    """The whole lines from offset on, and the offset just past them: a line still being written is left for the next look."""
    log.seek(offset)
    data = log.read()
    end = data.rfind(b"\n") + 1
    return data[:end].decode("utf-8").splitlines(), offset + end
