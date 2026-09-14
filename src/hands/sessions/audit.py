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
        line = json.dumps({"at": self._clock().isoformat(timespec="milliseconds"), **encoded(entry)}, ensure_ascii=False)
        # Opened for each line, so a line is on disk when record returns and a log moved aside is started again.
        with self._path.open("a", encoding="utf-8") as log:
            log.write(line + "\n")


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


def tail(path: Path, count: int) -> tuple[list[str], int]:
    """The last count whole lines of the log, and the offset just past them, where following it begins."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return [], 0
    end = data.rfind(b"\n") + 1
    lines = data[:end].decode("utf-8").splitlines()
    return (lines[-count:] if count > 0 else []), end


def follow(path: Path, offset: int, poll: Callable[[], None]) -> Iterator[str]:
    """Each whole line written to the log past offset, calling poll between looks, for as long as the caller asks."""
    while True:
        try:
            with path.open("rb") as log:
                if os.fstat(log.fileno()).st_size < offset:
                    # Moved aside and started again: the new log is read from its first line.
                    offset = 0
                log.seek(offset)
                data = log.read()
        except FileNotFoundError:
            offset, data = 0, b""
        # A line still being written is left for the next look.
        end = data.rfind(b"\n") + 1
        yield from data[:end].decode("utf-8").splitlines()
        offset += end
        poll()
