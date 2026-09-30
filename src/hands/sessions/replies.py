"""An exchange as hands reads it off the wire: the request classified, and the reply's bytes, decoded as their
Content-Encoding says, into typed events.

Read the same way however hands was handed the bytes: the proxy forwarding them, or fritter's copy of them.
"""

import json
import zlib
from collections.abc import Callable, Mapping
from typing import Protocol
from uuid import uuid4

import brotli
import zstandard
from loguru import logger

from hands.core.wire import Body, Garbled, Observed, Sent, Unknown, WireEvent, answered, assemble, classify, frames, is_stream, parse, session_of


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


class Reader(Protocol):
    def feed(self, plain: bytes, /) -> None: ...
    def finish(self) -> Body: ...


def reply_reader(headers: Mapping[str, str], hear: Callable[[WireEvent], None]) -> Reader:
    """What reads a reply with these headers: each event heard as its frame completes, and the whole at the end."""
    named = {name.lower(): value for name, value in headers.items()}
    return _Decoded(named.get("content-encoding", "identity"), _Events(hear) if is_stream(named.get("content-type", "")) else _Whole())


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
        return assemble(self._events)


class _Whole:
    """Any other reply, read whole: a count_tokens answer, or an error the API sent instead of a stream."""

    def __init__(self) -> None:
        self._parts: list[bytes] = []

    def feed(self, plain: bytes) -> None:
        self._parts.append(plain)

    def finish(self) -> Body:
        return answered(b"".join(self._parts))


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
