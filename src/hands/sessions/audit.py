"""The audit log: one JSON line for everything hands did, heard, said, and failed at, appended by the daemon and by each
`hands` command, one line at a time under one lock.

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
import io
import itertools
import json
import os
import re
from bisect import bisect_right
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal, assert_never, cast

from loguru import logger

if TYPE_CHECKING:
    # Defined only in loguru's type stubs.
    from loguru import Message

from hands.core.effects import Input, Type
from hands.core.session import SessionId
from hands.core.trace import Span
from hands.core.wire import Ending, Exchanged, Garbled, Held, Reached, Uncopied, Unfinished, Unreached
from hands.sessions.wide import WideEvent, chain


@dataclass(frozen=True)
class Typing:
    """What is about to be typed into a session: written before the typing, so the log holds every send, and the span of
    the unit of work that typed it, whose event says why."""

    effect: Type[Input]
    span: Span


@dataclass(frozen=True)
class TypingFailed:
    """What a Typing line said was about to be typed, and why it was not: the send never reached its session."""

    effect: Type[Input]
    reason: str


@dataclass(frozen=True)
class CopiesLost:
    """Copies of a session's exchanges that its fritter could not hand to hands, told with the first copy since that it
    did: hands was down, or was not reading them. The session's own exchanges went on regardless."""

    session: str | None
    lost: int


@dataclass(frozen=True)
class Transcribed:
    """What the user said in one turn, as it went to the brain."""

    text: str


@dataclass(frozen=True)
class Unsaid:
    """A segment Whisper transcribed and hands dropped as not said, with the scores it was dropped for."""

    text: str
    compression_ratio: float
    avg_logprob: float


@dataclass(frozen=True)
class Levels:
    """How loud one hold's audio was, as its mean power in dBFS, None where it was digital silence: as the microphone
    captured it, and as Whisper heard it, with the echo canceller between the two. What the canceller took out is the
    difference; at the phone, heard through no canceller, the two are the same."""

    captured_dbfs: float | None
    heard_dbfs: float | None


@dataclass(frozen=True)
class HoldHeard:
    """What Whisper made of one hold: what it took as said, None where nothing was, each segment it dropped, and how
    loud the hold was before and after the echo canceller, which tells a transcript made of the reply's echo left over
    (loud captured, quiet heard) from one of the room with nobody speaking (quiet both)."""

    hold: int
    said: str | None
    dropped: tuple[Unsaid, ...]
    levels: Levels


@dataclass(frozen=True)
class ByHand:
    """A hold the owner's hand opened, at the key, a button, or the phone: theirs, whoever else is in the room. `taught`
    says it taught the owner's voiceprint, a hold of the desk's key long enough to; `similarity` is how like the print
    it was before, by cosine, the owner's own measure for tuning SAME_VOICE, none where it was not heard or no print yet."""

    taught: bool
    similarity: float | None


@dataclass(frozen=True)
class Matched:
    """A hold the voice opened, whose voice is the user's: as like their voiceprint as `similarity`, by cosine."""

    similarity: float


@dataclass(frozen=True)
class Other:
    """A hold the voice opened, whose voice is someone else's in the room: only as like the user's voiceprint as
    `similarity`, by cosine."""

    similarity: float


@dataclass(frozen=True)
class Untold:
    """A hold the voice opened that could not be told apart, and is taken as the user's, as every hold was before hands
    told voices apart: no voiceprint has been taught yet."""

    why: Literal["no voiceprint"]


@dataclass(frozen=True)
class Untellable:
    """A hold whose voice could not be told for `error`: its words are given as the user's, as an Untold hold's are, and
    the failure is an error in the log rather than the words lost."""

    error: str


# Whose voice a hold was in (hands.voice.speakers).
Speaker = ByHand | Matched | Other | Untold | Untellable


@dataclass(frozen=True)
class Voiced:
    """Whose voice one hold with words in it was in, and how many seconds of it were heard to say so."""

    hold: int
    speaker: Speaker
    seconds: float


@dataclass(frozen=True)
class Primed:
    """The words Whisper was primed with for one hold, oldest first, the prompt tokens they come to, and how long reading
    them took. `focus` is the session whose repository was read, None where no running session is focused; `failed`
    says why the focus or its repository gave no words where either could not be read."""

    focus: SessionId | None
    words: tuple[str, ...]
    tokens: int
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


# When a user turn cuts off what hands is saying: as the hold that opens it opens, Pipecat's VAD start; or once Whisper
# hears words in it, Pipecat's transcription start.
TurnStart = Literal["on the hold", "on words"]


@dataclass(frozen=True)
class UserTurn:
    """A user turn as it ended: when the edge that opened it had it cut off what hands was saying, and how many seconds
    after it opened it did; None where it never did, a turn the voice opened that Whisper heard no words in."""

    start: TurnStart
    cut: float | None


@dataclass(frozen=True)
class Announced:
    """A fact the system channel gave the user, and whether it was spoken or, with speech down, posted to the screen."""

    text: str
    via: Literal["speech", "screen"]



# Where a cue went: the phone, the desk's speaker, or nowhere, with no desk speaker attached to play it on.
Played = Literal["phone", "desk", "unattached"]


@dataclass(frozen=True)
class Cued:
    """A cue for silence handed to the speaker: its line, how many times it was owed since it last played, how long
    the first of those was held, by speech or by its spacing, and where it went."""

    line: str
    folded: int
    waited: float
    played: Played





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
class SettingsEdited:
    """The home's config.toml changed while hands ran: the run restarts on it, or, where it was `refused`, says why
    and runs on the settings it started with."""

    path: str
    refused: str | None


# The OTLP signals each wide event is sent to the collector as: a span, for the trace store, and a log record, for the
# event store.
Signal = Literal["traces", "logs"]


@dataclass(frozen=True)
class Exported:
    """A batch of wide events sent to the collector as one OTLP signal, each named by its span id; how long the send
    took; and why the collector did not take it, None where it took every event: it could not be reached, it refused
    the request, it rejected events in it without saying which, or hands stopped before the batch could be sent. Each
    is in the log still. `warning` is what the collector said of a batch it took whole, as OTLP's partial success with
    nothing rejected says it."""

    collector: str
    signal: Signal
    spans: tuple[str, ...]
    duration_ms: float
    error: str | None
    warning: str | None


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
    Typing
    | TypingFailed
    | SettingsEdited
    | CopiesLost
    | Exchanged
    | Transcribed
    | HoldHeard
    | Voiced
    | Primed
    | Replied
    | CutOff
    | UserTurn
    | Announced
    | Cued
    | Refocused
    | Rolled
    | Failure
    | WideEvent
    | Exported
)
Record = Callable[[Entry], None]
Level = Literal["error", "info"]


def level(entry: Entry) -> Level:
    """Whether a line tells of something that went wrong: a Failure; a unit of work that failed, a start refused among them; a batch of wide events the collector did not take; an exchange the API refused or never answered, whose stream hands could not read, or whose copy
    broke off."""
    # [LAW:one-source-of-truth] the one place a line is judged an error, so a reader finds every error by one field and
    # never by an "error" deep in a body the API sent. [LAW:types-are-the-program] every kind of line is named here,
    # so a record added to Entry is judged here before pyright passes, rather than read as info by default.
    match entry:
        case Failure() | TypingFailed() | Exported(error=str()) | Voiced(speaker=Untellable()):
            return "error"
        case Exchanged(reply=reply):
            return _reply_level(reply)
        case WideEvent(outcome=outcome):
            return "error" if outcome == "failed" else "info"
        case Refocused(outcome=outcome):
            return "error" if outcome == "failed" else "info"
        case Primed(failed=failed) | SettingsEdited(refused=failed):
            return "info" if failed is None else "error"
        case (
            Typing() | Exported() | CopiesLost()
            | Transcribed() | HoldHeard() | Voiced() | Replied() | CutOff() | UserTurn() | Announced() | Cued() | Rolled()
        ):
            return "info"
        case _:
            assert_never(entry)


def _reply_level(reply: Ending) -> Level:
    match reply:
        case Unreached() | Uncopied() | Unfinished() | Reached(body=Garbled()):
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
        try:
            directory.mkdir(parents=True, exist_ok=True)
            directory.chmod(0o700)
        except OSError as error:
            # [LAW:single-enforcer] as a line it cannot write is: a log it cannot make costs the work it watches nothing,
            # and each line it then fails at says so.
            logger.warning(f"the audit log {directory} cannot be made the user's alone: {error}")
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
    return {"type": type(value).__name__, **{field.name: jsonable(getattr(value, field.name)) for field in fields(value)}}


def jsonable(value: object) -> object:
    """value as JSON holds it: a dataclass encoded, a time in ISO 8601, a duration in milliseconds, a set sorted."""
    match value:
        case None | bool() | int() | float() | str():
            return value
        case Path():
            return str(value)
        case datetime():
            return value.isoformat(timespec="milliseconds")
        case timedelta():
            # [LAW:one-source-of-truth] in milliseconds, the log's one unit of time, as duration_ms and heartbeat_ms are.
            return value / timedelta(milliseconds=1)
        case Enum():
            return jsonable(value.value)
        case Mapping():
            return {str(key): jsonable(item) for key, item in cast(Mapping[object, object], value).items()}
        case list() | tuple():
            return [jsonable(item) for item in cast(list[object] | tuple[object, ...], value)]
        case set() | frozenset():
            # Sorted, so a set is written the same way on every line it is on.
            return sorted((jsonable(item) for item in cast(set[object] | frozenset[object], value)), key=repr)
        case _ if is_dataclass(value) and not isinstance(value, type):
            return encoded(value)
        case _:
            # [LAW:no-silent-failure] a value the log cannot write is a bug to hear about, not a line to guess at.
            raise TypeError(f"the audit log cannot encode a {type(value).__name__}: {value!r}")


# The extra a logged line is bound with when it says a line the audit log already holds.
_AUDITED = "audited"


def said(error: str) -> None:
    """Say on the terminal an error the audit log already holds, as a failed unit of work's event: logged, and never
    written again as a Failure line beside it."""
    # At its caller's line, as the terminal names where each line was said.
    logger.opt(depth=1).bind(**{_AUDITED: True}).error(error)


def failures_to(record: Record) -> "Callable[[Message], None]":
    """A loguru sink that writes every error it is given as a Failure line, but one that says a line already written."""

    def sink(message: "Message") -> None:
        logged = message.record
        # [LAW:one-source-of-truth] the error is the line it says, never a Failure line beside it.
        if _AUDITED in logged["extra"]:
            return
        exception = logged["exception"]
        detail = "" if exception is None or exception.value is None else f": {type(exception.value).__name__}: {exception.value}"
        record(
            Failure(
                source=f"{logged['name']}:{logged['function']}",
                message=f"{logged['message']}{detail}",
                where=f"{logged['file'].path}:{logged['line']}",
                trace=() if exception is None or exception.value is None else chain(exception.value),
            )
        )

    return sink


def tail(directory: Path, count: int) -> tuple[list[str], int]:
    """The last count complete lines of the log, and the log offset just past them, where following it begins."""
    bases = segments(directory)
    if not bases:
        return [], 0
    *closed, active = bases
    # [LAW:no-ambient-temporal-coupling] the end is found once, and the lines are read back from it: a line written
    # while they are read is past the offset, and is following's to tell.
    with _held(segment(directory, active)) as log:
        end = _complete(log)
        older = (line for base in reversed(closed) for line in _newest_first(segment(directory, base)))
        newest = list(itertools.islice(itertools.chain(_before(log, end), older), max(count, 0)))
    return newest[::-1], active + end


def forwards(directory: Path) -> Iterator[str]:
    """Every complete line of the log, oldest first, reading a segment only once the caller asks past the one before it."""
    for base in segments(directory):
        yield from _lines(segment(directory, base), 0)[0]


def backwards(directory: Path) -> Iterator[str]:
    """Every complete line of the log, newest first, reading no further back than the caller asks."""
    for base in reversed(segments(directory)):
        yield from _newest_first(segment(directory, base))


def follow(directory: Path, offset: int, poll: Callable[[], None]) -> Iterator[str]:
    """Each complete line written to the log past offset, calling poll between reads, for as long as the caller asks."""
    while True:
        lines, offset = past(directory, offset)
        yield from lines
        poll()


def past(directory: Path, offset: int) -> tuple[list[str], int]:
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
    return [_text(line) for line in data[:end].split(b"\n")[:-1]], start + end


def _newest_first(path: Path) -> Iterator[str]:
    """The complete lines of path, newest first."""
    # [LAW:no-ambient-temporal-coupling] the end is found in the file the lines are read back from: opened once, it is
    # the same segment for both, whatever its path comes to name.
    with _held(path) as log:
        yield from _before(log, _complete(log))


def _complete(log: BinaryIO) -> int:
    """How many bytes of log are complete lines: a line still being written is not among them."""
    size = log.seek(0, os.SEEK_END)
    return next((start + block.rfind(b"\n") + 1 for start, block in _back(log, size) if b"\n" in block), 0)


def _before(log: BinaryIO, end: int) -> Iterator[str]:
    """The line of log that ends at byte end, where one does, and each before it, newest first."""
    if end == 0:
        return
    # The blocks of the line being read that lie past the block in hand, the nearest first.
    later: list[bytes] = []
    # The newline at end - 1 ends the newest line: every newline before it ends an older one.
    for _, block in _back(log, end - 1):
        head, *rest = block.split(b"\n")
        if rest:
            yield _text(rest.pop() + b"".join(reversed(later)))
            yield from map(_text, reversed(rest))
            later = []
        later.append(head)
    yield _text(b"".join(reversed(later)))


# How much of a segment is read at a time, going back from its end.
_BLOCK = 64 * 1024


def _back(log: BinaryIO, end: int) -> Iterator[tuple[int, bytes]]:
    """The bytes of log before end a block at a time, the block nearest end first, each with the offset it begins at:
    reading back from the end costs what is read, and not the segment's size."""
    for past in range(end, 0, -_BLOCK):
        start = max(0, past - _BLOCK)
        log.seek(start)
        yield start, log.read(past - start)


@contextmanager
def _held(path: Path) -> Generator[BinaryIO]:
    """path, open to read. A segment retention has deleted holds nothing."""
    try:
        log: BinaryIO = path.open("rb")
    except FileNotFoundError:
        log = io.BytesIO()
    with log:
        yield log


def _text(line: bytes) -> str:
    # [LAW:one-source-of-truth] a line ends at the newline its writer ended it with, and nowhere else: U+2028 and U+0085
    # are written raw inside a line's JSON. A write that failed partway can cut a character in two; its torn line reads
    # with U+FFFD in place of the half, and is then no JSON, as every torn line is.
    return line.decode("utf-8", errors="replace")
