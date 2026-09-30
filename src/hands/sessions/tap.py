"""The tap: each wrapped session's exchanges with the API, as its fritter copies them to hands, read as the wire's values.

    fritter --tap ANTHROPIC_BASE_URL=<upstream> --tap-to <home>/wire.sock -- claude ...

A copy is one connection of JSON lines, in the order the exchange happened: the request, then the reply's head and each
chunk of its bytes as they came, then its end, or instead that the upstream was never reached (fritter/tap.go). Each is
read into the Sent, Heard, and Exchanged the proxy makes of the brain's exchanges, through the same reading, and told to
the same observer. Nothing goes back: the session never waited on its copy, and hands has no say in its exchanges.
"""

import asyncio
import base64
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from loguru import logger

from hands.core.wire import Exchanged, Garbled, Heard, Observed, Reached, Seconds, Sent, Uncopied, Unreached, WireEvent
from hands.sessions.audit import CopiesLost, Record
from hands.sessions.replies import Reader, reply_reader, sent_of, shielded
from hands.sessions.server import claim_socket

# The longest line a copy may hold: a request carries a session's whole history, base64 in JSON.
LINE_LIMIT = 1 << 28


@dataclass(frozen=True)
class Request:
    at: Seconds
    method: str
    path: str
    headers: Mapping[str, str]
    body: bytes
    lost: int  # copies of this fritter's earlier exchanges that never reached hands


@dataclass(frozen=True)
class Response:
    at: Seconds
    status: int
    headers: Mapping[str, str]


@dataclass(frozen=True)
class Chunk:
    at: Seconds
    data: bytes


@dataclass(frozen=True)
class End:
    """The reply ended: read to its end when `error` is None, and cut short, as `error` says, otherwise."""

    at: Seconds
    error: str | None


@dataclass(frozen=True)
class NoUpstream:
    at: Seconds
    error: str


Line = Request | Response | Chunk | End | NoUpstream


def line_of(raw: bytes) -> Line:
    """One line of a copy, typed; raises ValueError saying what about it is not a line fritter writes."""
    # [LAW:parse-dont-validate] the one place a copy's JSON is read: everything past here is a Line.
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
        line = cast(dict[str, object], data)
        at = _float(line["at"])
        match line["kind"]:
            case "request":
                return Request(at, _str(line["method"]), _str(line["path"]), _headers(line["headers"]), _bytes(line["body"]), _int(line["lost"]))
            case "response":
                return Response(at, _int(line["status"]), _headers(line["headers"]))
            case "bytes":
                return Chunk(at, _bytes(line["bytes"]))
            case "end":
                return End(at, _str(line["error"]) or None)
            case "unreached":
                return NoUpstream(at, _str(line["error"]))
            case other:
                raise ValueError(f"no line is of kind {other!r}")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{error!r} in {raw[:200]!r}") from error


def _float(value: object) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"expected a number, got {type(value).__name__}")
    return float(value)


def _int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"expected an integer, got {type(value).__name__}")
    return value


def _str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected a string, got {type(value).__name__}")
    return value


def _bytes(value: object) -> bytes:
    # Go writes a []byte as base64, and a nil one as null.
    return b"" if value is None else base64.b64decode(_str(value), validate=True)


def _headers(value: object) -> Mapping[str, str]:
    if not isinstance(value, list):
        raise TypeError(f"expected a list of header pairs, got {type(value).__name__}")
    pairs = cast(list[object], value)
    headers: dict[str, str] = {}
    for pair in pairs:
        match pair:
            case [str() as name, str() as given]:
                headers[name] = given
            case _:
                raise TypeError(f"a header is {pair!r}, not a name and a value")
    return headers


async def serve_tap(path: Path, observe: Callable[[Observed], None], record: Record, clock: Callable[[], Seconds]) -> asyncio.Server:
    """Take copies on the socket until the returned server is closed."""
    tell = shielded(observe)

    async def take(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await _copy(reader, tell, record, clock)
        except Exception:
            # [LAW:no-silent-failure] a copy hands cannot read is its own loss: logged with its trace, and the next copy
            # is read as ever. The session had its exchange whatever became of this.
            logger.exception("hands could not read a session's copy of an exchange")
        finally:
            writer.close()

    claim_socket(path)
    server = await asyncio.start_unix_server(take, path=str(path), limit=LINE_LIMIT)
    # The copies are the user's conversations: theirs alone to dial into.
    path.chmod(0o600)
    return server


async def _copy(reader: asyncio.StreamReader, tell: Callable[[Observed], None], record: Record, clock: Callable[[], Seconds]) -> None:
    """One copy read as it arrives, each event told as its bytes come, and the whole exchange told at its end."""
    first = await reader.readline()
    if not first:
        # A fritter that dialled and went before it wrote: it counts the copy as lost, and says so with its next one.
        return
    match line_of(first):
        case Request() as request:
            pass
        case other:
            raise ValueError(f"a copy opened with a {type(other).__name__}, not its request")
    sent: Sent = sent_of(request.headers, request.path, request.body)
    if request.lost:
        record(CopiesLost(sent.session, request.lost))
    tell(sent)
    reply = await _reply(reader, lambda event: tell(Heard(sent.exchange, event)), clock)
    tell(Exchanged(sent.exchange, sent.session, sent.kind, request.method, request.path, len(request.body), (), request.at, request.at, reply))


async def _reply(reader: asyncio.StreamReader, hear: Callable[[WireEvent], None], clock: Callable[[], Seconds]) -> Reached | Unreached | Uncopied:
    reading: tuple[Response, Reader] | None = None
    first: Seconds | None = None
    last: Seconds | None = None
    size = 0
    while True:
        raw = await reader.readline()
        try:
            line = line_of(raw) if raw else None
        except ValueError as error:
            logger.exception("a copy line hands cannot read")
            line, broken = None, f"a line hands cannot read: {error}"
        else:
            broken = "the copy broke off: its fritter ended, or hands could not keep reading"
        match (line, reading):
            case (Response() as head, None):
                reading = (head, reply_reader(head.headers, hear))
            case (Chunk(at=at, data=data), (_, reader_)):
                first = at if first is None else first
                last = at
                size += len(data)
                reader_.feed(data)
            case (End(at=at, error=error), (head, reader_)):
                body = reader_.finish() if error is None else Garbled(error)
                return Reached(head.status, head.at if first is None else first, at if last is None else last, size, body)
            case (NoUpstream(at=at, error=error), None):
                return Unreached(error, at)
            case (None, (head, _)):
                return Reached(head.status, head.at if first is None else first, head.at if last is None else last, size, Garbled(broken))
            case (None, None):
                return Uncopied(broken, clock())
            case (unexpected, _):
                raise ValueError(f"a copy line {unexpected!r} out of its order")
