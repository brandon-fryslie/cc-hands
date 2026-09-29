"""The proxy: a Claude Code process reaches the API through it, and hands hears every request and every byte back.

    ANTHROPIC_BASE_URL=<the proxy's url> claude ...

It forwards each request upstream as it came and streams the reply back byte for byte, reading a copy of the same bytes
into typed events as they pass. It never makes a request of its own and never uses a client's credentials for
anything but that client's own request.
"""

import json
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

import aiohttp
import brotli
import zstandard
from aiohttp import web
from loguru import logger
from multidict import CIMultiDict

from hands.core.wire import (
    Body,
    Exchanged,
    Garbled,
    Heard,
    Observed,
    Reached,
    Seconds,
    Sent,
    Unknown,
    Unreached,
    WireEvent,
    answered,
    assemble,
    classify,
    frames,
    is_stream,
    parse,
    session_of,
)

UPSTREAM = "https://api.anthropic.com"

# Headers that describe one hop's connection rather than the request or reply, so each hop sets its own.
HOP_BY_HOP = frozenset({"connection", "content-length", "host", "keep-alive", "proxy-connection", "te", "trailer", "transfer-encoding", "upgrade"})

Observe = Callable[[Observed], None]


@dataclass(frozen=True)
class Proxy:
    url: str
    runner: web.AppRunner
    client: aiohttp.ClientSession

    async def close(self) -> None:
        await self.runner.cleanup()
        await self.client.close()


async def serve_proxy(upstream: str, observe: Observe, clock: Callable[[], Seconds]) -> Proxy:
    """Listen on a free local port until the returned proxy is closed; its url is what ANTHROPIC_BASE_URL is set to."""
    # The client's own timeouts govern a request: a long reply streams as long as the client keeps reading it.
    # Replies are read as sent, compressed or not, so the client gets the bytes the API wrote; and no header the client
    # left out is added, so the API is asked for no encoding the client did not ask for.
    client = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None), auto_decompress=False, skip_auto_headers=("Accept-Encoding", "Content-Type", "User-Agent")
    )

    def tell(observed: Observed) -> None:
        # [LAW:single-enforcer] what hears the wire never changes it: an observer that raises loses its own event, not the reply.
        try:
            observe(observed)
        except Exception:
            logger.exception(f"an observer of the wire failed on a {type(observed).__name__}")

    async def forward(request: web.Request) -> web.StreamResponse:
        # Read whole whatever its size: the request is classified from its body before any of it goes upstream.
        body = await request.content.read()
        requested_at = clock()
        exchange = uuid4().hex
        session = session_of(request.headers)
        parsed = _json(body)
        kind = classify(request.path_qs, parsed)
        if isinstance(kind, Unknown):
            # [LAW:no-silent-failure] the set of shapes fills in from use, so an unknown one is said where it will be seen.
            logger.warning(f"the proxy saw {kind.shape} (exchange {exchange}, session {session}); nothing it says will be spoken")
        tell(Sent(exchange, session, kind, parsed))

        sent_at = clock()

        def exchanged(reply: Reached | Unreached) -> Exchanged:
            return Exchanged(exchange, session, kind, request.method, request.path_qs, len(body), requested_at, sent_at, reply)

        try:
            reached = await client.request(request.method, upstream + request.path_qs, headers=_end_to_end(request.headers), data=body)
        except (aiohttp.ClientError, OSError) as error:
            tell(exchanged(Unreached(f"{type(error).__name__}: {error}", clock())))
            return web.Response(status=502, text=f"hands' proxy could not reach {upstream}: {error}")
        async with reached:
            response = web.StreamResponse(status=reached.status, reason=reached.reason, headers=_end_to_end(reached.headers))
            if "Content-Length" in reached.headers:
                response.content_length = int(reached.headers["Content-Length"])
            reader = _Decoded(reached.headers.get("Content-Encoding", "identity"), _reader(reached.headers, lambda event: tell(Heard(exchange, event))))
            first: Seconds | None = None
            ended: Seconds | None = None
            size = 0
            # What a reply is recorded as when its handler is cancelled mid-way: the client hung up, or hands is stopping.
            reply: Body = Garbled("the proxy stopped reading the reply: its client hung up or hands stopped")
            try:
                await response.prepare(request)
                async for chunk in reached.content.iter_any():
                    first = clock() if first is None else first
                    size += len(chunk)
                    await response.write(chunk)
                    reader.feed(chunk)
                ended = clock()
                reply = reader.finish()
                await response.write_eof()
            except (aiohttp.ClientError, OSError) as error:
                # Upstream dropped the reply, or the client hung up on it: either way the rest is not coming, and the
                # client's connection ends as the upstream one did.
                reply = Garbled(f"the reply ended after {size} bytes: {type(error).__name__}: {error}")
                raise
            finally:
                last = clock() if ended is None else ended
                tell(exchanged(Reached(reached.status, last if first is None else first, last, size, reply)))
        return response

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", forward)
    # A client that hangs up cancels its handler, and a stopping daemon cancels every handler within a tenth of a
    # second rather than waiting on replies that stream for minutes. Not 0: aiohttp reads a timeout of 0 as none at all.
    runner = web.AppRunner(app, access_log=None, handler_cancellation=True, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    return Proxy(url=f"http://{host}:{port}", runner=runner, client=client)


def _end_to_end(headers: Mapping[str, str]) -> CIMultiDict[str]:
    return CIMultiDict((name, value) for name, value in headers.items() if name.lower() not in HOP_BY_HOP)


def _json(body: bytes) -> object:
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


class _Reader(Protocol):
    def feed(self, plain: bytes) -> None: ...
    def finish(self) -> Body: ...


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


def _reader(headers: Mapping[str, str], hear: Callable[[WireEvent], None]) -> _Reader:
    return _Events(hear) if is_stream(headers.get("Content-Type", "")) else _Whole()


class _Decoded:
    """A reader handed the reply's bytes with their Content-Encoding taken off; what cannot be read is the reply's record."""

    def __init__(self, encoding: str, reader: _Reader) -> None:
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
