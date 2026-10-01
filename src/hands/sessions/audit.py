"""The audit log: one JSON line for everything the daemon did, heard, said, and failed at, appended by the daemon alone.

    uv run hands log        # the newest lines, then each new one as it is written

Each line is a value from this module or from the core, encoded the same way: its type's name under
"type" and its fields beside it, nested values alike, with the wall-clock time the line was written
under "at". Nothing here decides what happened; it records what the rest of the daemon already decided.
"""

import json
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from loguru import logger

if TYPE_CHECKING:
    # Defined only in loguru's type stubs.
    from loguru import Message

from hands.core.effects import AuditRecord, Effect, Input, Type
from hands.core.events import Event
from hands.core.session import SessionId
from hands.core.wire import Exchanged
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
    turn ended in error of, as its latest answer on the wire said; None for a turn that did not."""

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
class Called:
    """A tool the intermediary called, with the arguments it gave and the result it was handed back."""

    tool: str
    arguments: Mapping[str, object]
    result: object


@dataclass(frozen=True)
class Announced:
    """A fact the system channel gave the user, and whether it was spoken or, with speech down, posted to the screen."""

    text: str
    via: Literal["speech", "screen"]


@dataclass(frozen=True)
class Recounted:
    """What hands told of a turn a session finished, and what the narration left the user able to ask for.

    `told` is what went out: handed to the model to say in its own words when `by_model`, and said as written when not.
    `topics` is every part of the turn's narration that was built and not played — its sections, and what it
    asked through a dialog and is no longer waiting on — which makes this line the
    one place a developer who cannot see the screen can find out what "more on that" has to open. `questions`
    is what the session is waiting on an answer to, kept apart so a log reader need not find it in `told`.
    """

    session: str
    told: str
    topics: tuple[str, ...]
    questions: tuple[str, ...]
    by_model: bool


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


@dataclass(frozen=True)
class BacklogUnread:
    """A pass of the summary store that could not start: lit would not hand over the project's backlog, and why."""

    project: str
    error: str
    seconds: float


@dataclass(frozen=True)
class Failure:
    """An error the daemon logged: where it was raised and what it said."""

    source: str
    message: str


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
    | Called
    | Announced
    | Recounted
    | Summarised
    | TurnsSummarised
    | BacklogUnread
    | Failure
)
Record = Callable[[Entry], None]


class AuditLog:
    def __init__(self, path: Path, clock: Callable[[], datetime]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._clock = clock
        # The log holds what every tapped session said, as the tap's socket does: the user's alone to read.
        if path.exists():
            path.chmod(0o600)

    def record(self, entry: Entry) -> None:
        # [LAW:single-enforcer] the log watches what the daemon does and never changes it: a line it cannot encode or
        # write is lost here, not a send's answer, a tool's result, a permission's question, or a background task.
        try:
            line = json.dumps({"at": self._clock().isoformat(timespec="milliseconds"), **encoded(entry)}, ensure_ascii=False)
        except TypeError as error:
            # [LAW:no-silent-failure] a bug in what was recorded: logged as an error, it is a Failure line, whose fields always encode.
            logger.error(f"the audit log cannot encode a {type(entry).__name__} line: {error}")
            return
        try:
            # Opened for each line, so a line is on disk when record returns and a log moved aside is started again.
            with open(self._path, "a", encoding="utf-8", opener=_private) as log:
                log.write(line + "\n")
        except OSError as error:
            # [LAW:no-silent-failure] said on stderr, as a warning: an error would be sent back to the log that just failed.
            logger.warning(f"the audit log {self._path} lost a {type(entry).__name__} line: {error}")


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
        record(Failure(source=f"{logged['name']}:{logged['function']}", message=f"{logged['message']}{detail}"))

    return sink


@dataclass(frozen=True)
class Position:
    """How far into which file the log has been read. A file with another inode is another log, whatever its size."""

    inode: int  # 0 while there is no log
    offset: int


START = Position(inode=0, offset=0)


def tail(path: Path, count: int) -> tuple[list[str], Position]:
    """The last count whole lines of the log, and the position just past them, where following it begins."""
    lines, position = _read(path, START)
    return (lines[-count:] if count > 0 else []), position


def follow(path: Path, position: Position, poll: Callable[[], None]) -> Iterator[str]:
    """Each whole line written to the log past position, calling poll between looks, for as long as the caller asks."""
    while True:
        lines, position = _read(path, position)
        yield from lines
        poll()


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
            offset = since.offset if same else 0
            log.seek(offset)
            data = log.read()
    except FileNotFoundError:
        return [], START
    # A line still being written is left for the next look.
    end = data.rfind(b"\n") + 1
    return data[:end].decode("utf-8").splitlines(), Position(status.st_ino, offset + end)
