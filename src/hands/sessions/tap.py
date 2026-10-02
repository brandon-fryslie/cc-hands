"""The tap: each wrapped session's exchanges with the API, as its fritter copies them to hands, read as the wire's values.

    fritter --tap <upstream> --tap-ca NODE_EXTRA_CA_CERTS --tap-to <home>/wire.sock -- claude ...

A copy is one connection of JSON lines, in the order the exchange happened: the request, then the reply's head and each
chunk of its bytes as they came, then its end, or instead that the upstream was never reached (fritter/tap.go). Each is
read into the Sent, Heard, and Exchanged the proxy makes of the brain's exchanges, through the same reading, and told to
the same observer. Nothing goes back: the session never waited on its copy, and hands has no say in its exchanges.
"""

import asyncio
import base64
import binascii
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from hands.core.events import Closed
from hands.core.session import SessionId
from hands.core.wire import Exchanged, Garbled, Heard, MainTurn, Message, Observed, Reached, Seconds, Sent, Streamed, Text, Uncopied, Unreached, WireEvent
from hands.sessions.audit import CopiesLost, Record
from hands.sessions.payload import Payload, Rejected
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
    """One line of a copy, typed; raises Rejected saying what about it is not a line fritter writes."""
    # [LAW:parse-dont-validate] the one place a copy's JSON is read: everything past here is a Line.
    try:
        line = Payload.parse(raw)
        at = line.number("at")
        match line.text("kind"):
            case "request":
                return Request(at, line.text("method"), line.text("path"), _headers(line.items("headers")), _bytes(line.optional_text("body")), line.integer("lost"))
            case "response":
                return Response(at, line.integer("status"), _headers(line.items("headers")))
            case "bytes":
                return Chunk(at, _bytes(line.optional_text("bytes")))
            case "end":
                return End(at, line.text("error") or None)
            case "unreached":
                return NoUpstream(at, line.text("error"))
            case other:
                raise Rejected(f"no line is of kind {other!r}")
    except Rejected as error:
        raise Rejected(f"{error} in {raw[:200]!r}") from error


def _bytes(value: str | None) -> bytes:
    # Go writes a []byte as base64, and a nil one as null.
    try:
        return b"" if value is None else base64.b64decode(value, validate=True)
    except binascii.Error as error:
        raise Rejected(f"bytes that are not base64: {error}") from error


def _headers(pairs: list[object]) -> Mapping[str, str]:
    headers: dict[str, str] = {}
    for pair in pairs:
        match pair:
            case [str() as name, str() as given]:
                headers[name] = given
            case _:
                raise Rejected(f"a header is {pair!r}, not a name and a value")
    return headers


def moves(observed: Observed) -> tuple[Closed, ...]:
    """What a session's exchange says of its turn: the reply that closed it came back."""
    match observed:
        case Exchanged(session=str() as session, kind=MainTurn(prompt=str() as prompt), reply=Reached(body=Streamed(message=Message(stop_reason="end_turn") as message))):
            # The last text block, as the Stop hook's last_assistant_message and the transcript's last step hold it:
            # Claude Code records each block of a reply on its own.
            texts = [block.text for block in message.content if isinstance(block, Text)]
            return (Closed(SessionId(session), prompt, texts[-1]),) if texts and texts[-1] else ()
        case _:
            # A request, one that names no turn, a subagent's or a fork's, a reply that asks for a tool or never came whole:
            # nothing of the turn's end, which its Stop hook still tells. Each is on its exchange's own audit line.
            return ()


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
    # Off the loop: a request carries its session's whole history, megabytes on a long one, and the voice runs on this loop.
    request, sent = await asyncio.to_thread(_opened, first)
    if request.lost:
        record(CopiesLost(sent.session, request.lost))
    tell(sent)
    reply = await _reply(reader, lambda event: tell(Heard(sent.exchange, event)), clock)
    tell(Exchanged(sent.exchange, sent.session, sent.kind, request.method, request.path, len(request.body), (), request.at, request.at, reply))


def _opened(raw: bytes) -> tuple[Request, Sent]:
    match line_of(raw):
        case Request() as request:
            return request, sent_of(request.headers, request.path, request.body)
        case other:
            raise Rejected(f"a copy opened with a {type(other).__name__}, not its request")


async def _next(reader: asyncio.StreamReader) -> Line | str:
    """The copy's next line, or why it has none."""
    try:
        raw = await reader.readline()
    except (OSError, ValueError) as error:
        # A connection reset, or a line longer than LINE_LIMIT.
        return f"the copy broke off: {error!r}"
    if not raw:
        return "the copy broke off: its fritter ended, or hands could not keep reading"
    try:
        return line_of(raw)
    except Rejected as error:
        logger.exception("a copy line hands cannot read")
        return f"a line hands cannot read: {error}"


async def _reply(reader: asyncio.StreamReader, hear: Callable[[WireEvent], None], clock: Callable[[], Seconds]) -> Reached | Unreached | Uncopied:
    """How the exchange ended, as its copy tells it; a copy that breaks off, however it does, is a reply of its own."""
    reading: tuple[Response, Reader] | None = None
    first: Seconds | None = None
    last: Seconds | None = None
    size = 0
    while True:
        line = await _next(reader)
        match (line, reading):
            case (Response() as head, None):
                reading = (head, reply_reader(head.headers, hear))
                continue
            case (Chunk(at=at, data=data), (_, feeder)):
                first = at if first is None else first
                last = at
                size += len(data)
                feeder.feed(data)
                continue
            case (End(at=at, error=error), (head, feeder)):
                body = feeder.finish() if error is None else Garbled(error)
                return Reached(head.status, head.at if first is None else first, at if last is None else last, size, body)
            case (NoUpstream(at=at, error=error), None):
                return Unreached(error, at)
            case (str() as broken, _):
                pass
            case (unexpected, _):
                broken = f"a {type(unexpected).__name__} line out of its order"
        match reading:
            case None:
                return Uncopied(broken, clock())
            case (head, _):
                return Reached(head.status, head.at if first is None else first, head.at if last is None else last, size, Garbled(broken))
