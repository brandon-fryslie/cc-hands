"""An exchange as hands reads it off the wire: the request classified, and the reply's bytes, decoded as their
Content-Encoding says, into typed events.

Read the same way however hands was handed the bytes: the proxy forwarding them, or fritter's copy of them.
"""

import json
import zlib
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

import brotli
import zstandard
from loguru import logger

from hands.core.wire import Body, Elsewhere, Garbled, Kind, Observed, Seconds, UsageLimitReached, Sent, Unknown, Unkept, WireEvent, assemble, classify, frames, is_stream, parse, session_of, unstreamed


def sent_of(headers: Mapping[str, str], path: str, body: bytes) -> Sent:
    """A request as it leaves for the API, named as a new exchange and classified from its path and body."""
    exchange = uuid4().hex
    session = session_of(headers)
    parsed = _json(body)
    kind = classify(path, parsed)
    if isinstance(kind, Unknown):
        # [LAW:no-silent-failure] the set of shapes fills in from use, so an unknown one is said where it will be seen.
        logger.warning(f"the wire saw {kind.shape} (exchange {exchange}, session {session}); nothing it says will be spoken")
    return Sent(exchange, session, kind, parsed)


def shielded(observe: Callable[[Observed], None]) -> Callable[[Observed], None]:
    """What hears the wire, kept from changing it: an observer that raises loses its own event, not the exchange."""

    def tell(observed: Observed) -> None:
        # [LAW:single-enforcer] what hears the wire never changes it, however the exchange reached hands.
        try:
            observe(observed)
        except Exception:
            logger.exception(f"an observer of the wire failed on a {type(observed).__name__}")

    return tell


def _json(body: bytes) -> object:
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


# The headers the API refuses a subscription's spent usage limit with, and the second its access returns: the body names no
# limit. Claude Code 2.1.285 says the same reset (measured, hands-wire-6ic.gfq).
_LIMIT_STATUS = "anthropic-ratelimit-unified-status"
_LIMIT_RESET = "anthropic-ratelimit-unified-reset"


def spent(status: int, headers: Mapping[str, str]) -> UsageLimitReached | None:
    """The spent usage limit an answer with this status and these headers refuses under; None for any other answer."""
    named = _named(headers)
    match status, named.get(_LIMIT_STATUS):
        case 429, "rejected":
            return UsageLimitReached(_reset(named.get(_LIMIT_RESET, "")))
        case _:
            return None


def _reset(reset: str) -> Seconds | None:
    """The instant a reset header names, or None when it names none a clock can show: read before the proxy passes the
    answer on, so a header the API got wrong cannot fail the answer or the sentence that says it."""
    try:
        return datetime.fromtimestamp(int(reset), UTC).timestamp()
    except (ValueError, OverflowError, OSError):
        return None


class Reader(Protocol):
    def feed(self, plain: bytes, /) -> None: ...
    def finish(self) -> Body: ...


def _named(headers: Mapping[str, str]) -> dict[str, str]:
    """Headers by lowercase name, since HTTP names are case-insensitive whatever mapping carried them."""
    return {name.lower(): value for name, value in headers.items()}


def reply_reader(kind: Kind, headers: Mapping[str, str], hear: Callable[[WireEvent], None]) -> Reader:
    """What reads the reply to a request of this kind with these headers: each event heard as its frame completes, and the
    whole at the end; nothing of a reply from elsewhere than the model's endpoint."""
    named = _named(headers)
    match kind:
        case Elsewhere():
            return _Unread()
        case _:
            return _Decoded(named.get("content-encoding", "identity"), _Events(hear) if is_stream(named.get("content-type", "")) else _Whole(hear))


class _Unread:
    def feed(self, plain: bytes) -> None:
        pass

    def finish(self) -> Body:
        return Unkept()


class _Events:
    """A streamed reply read frame by frame, each event heard as its frame completes, and assembled at the end."""

    def __init__(self, hear: Callable[[WireEvent], None]) -> None:
        self._hear = hear
        self._pending = b""
        self._events: list[WireEvent] = []

    def feed(self, plain: bytes) -> None:
        whole, self._pending = frames(self._pending + plain)
        for frame in whole:
            event = parse(frame)
            self._events.append(event)
            self._hear(event)

    def finish(self) -> Body:
        if self._pending.strip():
            return Garbled(f"the stream ended mid-frame: {self._pending[:200]!r}")
        return assemble(self._events, streamed=True)


class _Whole:
    """Any other reply, read whole: a message sent without a stream, its events heard once it is all here, a count_tokens
    answer, or an error the API sent instead of a message."""

    def __init__(self, hear: Callable[[WireEvent], None]) -> None:
        self._hear = hear
        self._parts: list[bytes] = []

    def feed(self, plain: bytes) -> None:
        self._parts.append(plain)

    def finish(self) -> Body:
        match unstreamed(b"".join(self._parts)):
            case tuple() as events:
                # [LAW:one-source-of-truth] heard and folded as the stream it stands for, so every listener reads the
                # message the one way, however it came.
                for event in events:
                    self._hear(event)
                return assemble(events, streamed=False)
            case unread:
                return unread


class _Decoded:
    """A reader handed the reply's bytes with their Content-Encoding taken off; what cannot be read is the reply's record."""

    def __init__(self, encoding: str, reader: Reader) -> None:
        self._decode = _decoder(encoding)
        self._reader = reader
        self._error: str | None = None

    def feed(self, chunk: bytes) -> None:
        if self._error is not None:
            return
        try:
            self._reader.feed(self._decode(chunk))
        except Exception as error:
            self._fail(error)

    def finish(self) -> Body:
        if self._error is None:
            try:
                return self._reader.finish()
            except Exception as error:
                self._fail(error)
        return Garbled(f"hands could not read it: {self._error}")

    def _fail(self, error: Exception) -> None:
        # [LAW:single-enforcer] what hears the wire never changes it: whatever hands' reading raises, the reply still
        # reaches the client whole. [LAW:no-silent-failure] the failure is the reply's record, and its trace is logged.
        logger.opt(exception=error).warning("the proxy could not read a reply it forwarded")
        self._error = f"{type(error).__name__}: {error}"


def _decoder(encoding: str) -> Callable[[bytes], bytes]:
    match encoding.strip().lower():
        case "" | "identity":
            return lambda chunk: chunk
        case "gzip" | "x-gzip" | "deflate":
            # 32 + 15: a gzip or zlib header, whichever the reply starts with.
            return zlib.decompressobj(32 + zlib.MAX_WBITS).decompress
        # Claude Code asks for every encoding here, so whichever the API picks is one hands reads.
        case "br":
            return brotli.Decompressor().process
        case "zstd":
            # Across frames: a streamed reply may be flushed as many.
            return zstandard.ZstdDecompressor().decompressobj(read_across_frames=True).decompress
        case other:

            def undecodable(_chunk: bytes) -> bytes:
                raise ValueError(f"Content-Encoding {other!r} is not one hands decodes")

            return undecodable
