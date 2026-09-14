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
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from loguru import logger

if TYPE_CHECKING:
    # Defined only in loguru's type stubs.
    from loguru import Message

from hands.core.effects import AuditRecord, Effect
from hands.core.events import Event


@dataclass(frozen=True)
class Applied:
    """An event that changed the registry or called for an effect. One that did neither, such as a quiet tick, is not a line."""

    event: Event


@dataclass(frozen=True)
class Performed:
    effect: Effect


@dataclass(frozen=True)
class EffectFailed:
    effect: Effect
    error: str


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
class Failure:
    """An error the daemon logged: where it was raised and what it said."""

    source: str
    message: str


Entry = AuditRecord | Applied | Performed | EffectFailed | Transcribed | Replied | Called | Announced | Failure
Record = Callable[[Entry], None]


class AuditLog:
    def __init__(self, path: Path, clock: Callable[[], datetime]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._clock = clock

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
            with self._path.open("a", encoding="utf-8") as log:
                log.write(line + "\n")
        except OSError as error:
            # [LAW:no-silent-failure] said on stderr, as a warning: an error would be sent back to the log that just failed.
            logger.warning(f"the audit log {self._path} lost a {type(entry).__name__} line: {error}")


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
