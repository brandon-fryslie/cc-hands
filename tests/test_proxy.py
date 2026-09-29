"""The proxy: the client gets the API's bytes unchanged, the API gets the client's request unchanged, and each exchange is one record."""

import asyncio
import contextlib
import gzip
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

import aiohttp
import brotli
import pytest
import zstandard
from aiohttp import web
from loguru import logger

from hands.core.session import SessionId
from hands.core.wire import (
    Answered,
    CountTokens,
    Exchanged,
    Forward,
    Garbled,
    Heard,
    Held,
    Hold,
    MainTurn,
    Observed,
    Reached,
    Route,
    Sent,
    Streamed,
    Text,
    TextDelta,
    Unreached,
    assemble,
    frames,
    parse,
)
from hands.daemon.run import wire_to
from hands.sessions.audit import Entry
from hands.sessions.proxy import Proxy, serve_proxy

REQUEST = (
    b'{"model": "claude-opus-5-5", "tools": [{"name": "Read"}], "stream": true, "messages": '
    b'[{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}]}'
)
HEADERS = {"Authorization": "Bearer the-clients-own", "X-Claude-Code-Session-Id": "s1", "Content-Type": "application/json"}


def sse(event: str, data: str) -> bytes:
    return f"event: {event}\ndata: {data}\n\n".encode()


STREAM = [
    sse("message_start", '{"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":2}}}'),
    sse("content_block_start", '{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}'),
    sse("content_block_delta", '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello "}}'),
    sse("content_block_delta", '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"from the wire"}}'),
    sse("content_block_stop", '{"type":"content_block_stop","index":0}'),
    sse("message_delta", '{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}'),
    sse("message_stop", '{"type":"message_stop"}'),
]


@dataclass
class Upstream:
    """A stand-in for the API: what it was asked, and a handler per path."""

    url: str
    asked: list[tuple[str, dict[str, str], bytes]] = field(default_factory=lambda: [])


def forward(_sent: Sent) -> Route:
    return Forward()


@dataclass
class Wire:
    proxy: Proxy
    seen: list[Observed]
    exchanged: asyncio.Event
    # Where each request goes; a test that holds one sets it before the request is sent.
    route: Callable[[Sent], Route] = forward


Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


@pytest.fixture
async def serve() -> AsyncIterator[Callable[[Handler], Awaitable[tuple[Upstream, Wire]]]]:
    cleanups: list[Callable[[], Awaitable[None]]] = []
    ticks = iter(float(second) for second in range(1_000))

    async def start(handler: Handler) -> tuple[Upstream, Wire]:
        runner = web.AppRunner(web.Application())
        upstream = Upstream(url="")

        async def recorded(request: web.Request) -> web.StreamResponse:
            upstream.asked.append((request.path_qs, dict(request.headers), await request.read()))
            return await handler(request)

        runner.app.router.add_route("*", "/{path:.*}", recorded)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        upstream.url = f"http://127.0.0.1:{runner.addresses[0][1]}"
        seen: list[Observed] = []
        exchanged = asyncio.Event()

        def observe(observed: Observed) -> None:
            seen.append(observed)
            if isinstance(observed, Exchanged):
                exchanged.set()

        def route(sent: Sent) -> Route:
            return made.route(sent)

        proxy = await serve_proxy(upstream.url, observe, route, clock=lambda: next(ticks))
        cleanups.extend([proxy.close, runner.cleanup])
        made = Wire(proxy, seen, exchanged)
        return upstream, made

    yield start
    for cleanup in cleanups:
        await cleanup()


async def streamed(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "request-id": "req_1"})
    await response.prepare(request)
    for chunk in STREAM:
        # Each frame split in two, so the proxy reads frames across chunk boundaries as the API sends them.
        await response.write(chunk[:10])
        await response.write(chunk[10:])
    await response.write_eof()
    return response


async def post(url: str, body: bytes = REQUEST, path: str = "/v1/messages?beta=true") -> tuple[int, dict[str, str], bytes]:
    async with aiohttp.ClientSession(auto_decompress=False) as client:
        async with client.post(url + path, data=body, headers=HEADERS) as response:
            return response.status, dict(response.headers), await response.read()


def only_exchange(wire: Wire) -> Exchanged:
    exchanges = [seen for seen in wire.seen if isinstance(seen, Exchanged)]
    assert len(exchanges) == 1
    return exchanges[0]


async def test_a_stream_passes_byte_for_byte_and_is_one_classified_timed_exchange(serve: Callable[[Handler], Awaitable[tuple[Upstream, Wire]]]) -> None:
    upstream, wire = await serve(streamed)
    status, headers, body = await post(wire.proxy.url)

    assert (status, body, headers["request-id"]) == (200, b"".join(STREAM), "req_1")
    [(path, asked_headers, asked_body)] = upstream.asked
    # The request goes on as it came: its path, its body, and the client's own credentials for its own request.
    assert (path, asked_body, asked_headers["Authorization"]) == ("/v1/messages?beta=true", REQUEST, "Bearer the-clients-own")
    exchange = only_exchange(wire)
    assert (exchange.session, exchange.kind, exchange.method, exchange.path, exchange.request_bytes) == (
        SessionId("s1"), MainTurn(), "POST", "/v1/messages?beta=true", len(REQUEST)
    )
    assert isinstance(exchange.reply, Reached)
    assert exchange.reply.status == 200 and exchange.reply.reply_bytes == len(body)
    assert exchange.requested_at < exchange.sent_at < exchange.reply.first_byte_at < exchange.reply.last_byte_at
    assert isinstance(exchange.reply.body, Streamed)
    assert exchange.reply.body.message.content == (Text("hello from the wire"),)


async def test_the_request_is_heard_before_the_reply_and_text_as_it_arrives(serve: Callable[[Handler], Awaitable[tuple[Upstream, Wire]]]) -> None:
    _, wire = await serve(streamed)
    await post(wire.proxy.url)
    assert isinstance(wire.seen[0], Sent) and wire.seen[0].kind == MainTurn()
    texts = [seen.event.text for seen in wire.seen if isinstance(seen, Heard) and isinstance(seen.event, TextDelta)]
    assert texts == ["hello ", "from the wire"]
    assert isinstance(wire.seen[-1], Exchanged)


async def test_a_compressed_answer_reaches_the_client_compressed_and_is_read_decompressed(serve: Callable[[Handler], Awaitable[tuple[Upstream, Wire]]]) -> None:
    packed = gzip.compress(b'{"input_tokens": 5583}')

    async def counted(_request: web.Request) -> web.Response:
        return web.Response(body=packed, headers={"Content-Type": "application/json", "Content-Encoding": "gzip"})

    _, wire = await serve(counted)
    status, headers, body = await post(wire.proxy.url, path="/v1/messages/count_tokens?beta=true")
    assert (status, body, headers["Content-Encoding"]) == (200, packed, "gzip")
    exchange = only_exchange(wire)
    assert exchange.kind == CountTokens()
    assert isinstance(exchange.reply, Reached) and exchange.reply.body == Answered({"input_tokens": 5583})


def zstd_frames(data: bytes) -> bytes:
    """The reply flushed as two zstd frames, as a stream can be."""
    return zstandard.ZstdCompressor().compress(data[:50]) + zstandard.ZstdCompressor().compress(data[50:])


@pytest.mark.parametrize(("encoding", "pack"), [("br", brotli.compress), ("zstd", zstd_frames)])
async def test_every_encoding_claude_code_asks_for_is_read(
    serve: Callable[[Handler], Awaitable[tuple[Upstream, Wire]]], encoding: str, pack: Callable[[bytes], bytes]
) -> None:
    packed = pack(b"".join(STREAM))

    async def compressed(_request: web.Request) -> web.Response:
        return web.Response(body=packed, headers={"Content-Type": "text/event-stream", "Content-Encoding": encoding})

    _, wire = await serve(compressed)
    assert (await post(wire.proxy.url))[::2] == (200, packed)
    reply = only_exchange(wire).reply
    assert isinstance(reply, Reached) and isinstance(reply.body, Streamed)
    assert reply.body.message.content == (Text("hello from the wire"),)


async def test_a_reply_hands_cannot_read_still_reaches_the_client_whole(serve: Callable[[Handler], Awaitable[tuple[Upstream, Wire]]]) -> None:
    # A frame that is not UTF-8 breaks hands' reading of the stream, and nothing of the client's.
    sent = [*STREAM[:2], b"event: content_block_delta\ndata: \xff\xfe\n\n", *STREAM[2:]]

    async def unreadable(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for chunk in sent:
            await response.write(chunk)
        await response.write_eof()
        return response

    _, wire = await serve(unreadable)
    assert (await post(wire.proxy.url))[::2] == (200, b"".join(sent))
    reply = only_exchange(wire).reply
    assert isinstance(reply, Reached) and isinstance(reply.body, Garbled) and reply.body.reason.startswith("hands could not read it: UnicodeDecodeError")


async def test_no_header_the_client_left_out_is_added_on_the_way_up(serve: Callable[[Handler], Awaitable[tuple[Upstream, Wire]]]) -> None:
    upstream, wire = await serve(streamed)
    async with aiohttp.ClientSession(skip_auto_headers=("Accept-Encoding", "User-Agent", "Content-Type")) as client:
        async with client.post(wire.proxy.url + "/v1/messages", data=REQUEST) as response:
            await response.read()
    [(_, asked_headers, _)] = upstream.asked
    assert {"Accept-Encoding", "User-Agent", "Content-Type"}.isdisjoint(asked_headers)


async def test_an_error_status_passes_through_with_its_body_read(serve: Callable[[Handler], Awaitable[tuple[Upstream, Wire]]]) -> None:
    error = b'{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}'

    async def overloaded(_request: web.Request) -> web.Response:
        return web.Response(status=529, body=error, headers={"Content-Type": "application/json"})

    _, wire = await serve(overloaded)
    assert (await post(wire.proxy.url))[::2] == (529, error)
    reply = only_exchange(wire).reply
    assert isinstance(reply, Reached) and (reply.status, reply.body) == (529, Answered({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}))


async def test_an_api_that_cannot_be_reached_is_a_502_and_an_unreached_exchange() -> None:
    seen: list[Observed] = []
    # A port just released, so nothing listens on it.
    probe = web.AppRunner(web.Application())
    await probe.setup()
    await web.TCPSite(probe, "127.0.0.1", 0).start()
    dead = f"http://127.0.0.1:{probe.addresses[0][1]}"
    await probe.cleanup()
    proxy = await serve_proxy(dead, seen.append, forward, clock=lambda: 7.0)
    try:
        status, _, body = await post(proxy.url)
    finally:
        await proxy.close()
    assert status == 502 and dead.encode() in body
    [exchange] = [observed for observed in seen if isinstance(observed, Exchanged)]
    assert isinstance(exchange.reply, Unreached) and exchange.reply.failed_at == 7.0


async def test_a_client_that_hangs_up_mid_stream_ends_the_upstream_reply_and_is_recorded(serve: Callable[[Handler], Awaitable[tuple[Upstream, Wire]]]) -> None:
    upstream_ended = asyncio.Event()

    async def endless(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(STREAM[0])
        try:
            while True:
                await response.write(sse("ping", '{"type":"ping"}'))
                await asyncio.sleep(0.01)
        except (ConnectionError, aiohttp.ClientConnectionResetError):
            upstream_ended.set()
        return response

    _, wire = await serve(endless)
    async with aiohttp.ClientSession() as client:
        async with client.post(wire.proxy.url + "/v1/messages", data=REQUEST, headers=HEADERS) as response:
            await response.content.readany()
    # The hang-up is the only way the rest stops coming, so both ends are waited on rather than timed.
    await asyncio.wait_for(asyncio.gather(wire.exchanged.wait(), upstream_ended.wait()), timeout=10)
    reply = only_exchange(wire).reply
    assert isinstance(reply, Reached) and isinstance(reply.body, Garbled) and reply.body == Garbled("the proxy stopped reading the reply: its client hung up or hands stopped")


async def test_closing_the_proxy_mid_stream_stops_at_once_and_records_the_reply(serve: Callable[[Handler], Awaitable[tuple[Upstream, Wire]]]) -> None:
    async def endless(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(STREAM[0])
        # Pings until the proxy lets go of the reply, which a write then finds.
        with contextlib.suppress(ConnectionError, aiohttp.ClientConnectionResetError):
            while True:
                await response.write(sse("ping", '{"type":"ping"}'))
                await asyncio.sleep(0.01)
        return response

    _, wire = await serve(endless)
    async with aiohttp.ClientSession() as client:
        async with client.post(wire.proxy.url + "/v1/messages", data=REQUEST, headers=HEADERS) as response:
            await response.content.readany()
            # A stop does not wait on a reply that may stream for minutes.
            await asyncio.wait_for(wire.proxy.close(), timeout=5)
    reply = only_exchange(wire).reply
    assert isinstance(reply, Reached) and reply.body == Garbled("the proxy stopped reading the reply: its client hung up or hands stopped")


async def test_a_chunk_is_heard_before_the_client_can_have_it(serve: Callable[[Handler], Awaitable[tuple[Upstream, Wire]]]) -> None:
    # Whatever the client does once it has the bytes, its result line on stdout included, comes after hands heard them.
    rest = asyncio.Event()

    async def paused(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for chunk in STREAM[:3]:
            await response.write(chunk)
        await rest.wait()
        for chunk in STREAM[3:]:
            await response.write(chunk)
        await response.write_eof()
        return response

    _, wire = await serve(paused)
    async with aiohttp.ClientSession() as client:
        async with client.post(wire.proxy.url + "/v1/messages", data=REQUEST, headers=HEADERS) as response:
            got = b""
            while not got.endswith(STREAM[2]):
                got += await response.content.readany()
            heard = [seen.event for seen in wire.seen if isinstance(seen, Heard)]
            assert TextDelta(0, "hello ") in heard
            rest.set()
            await response.read()


async def test_a_held_request_never_reaches_the_api_and_is_answered_with_the_routes_words(serve: Callable[[Handler], Awaitable[tuple[Upstream, Wire]]]) -> None:
    upstream, wire = await serve(streamed)
    wire.route = lambda _sent: Hold("(stayed silent)")
    status, headers, body = await post(wire.proxy.url)
    assert upstream.asked == []
    assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
    # A whole message with the one text block, ended, as the API would have streamed it.
    whole, rest = frames(body)
    assert rest == b""
    reply = assemble([parse(frame) for frame in whole])
    assert isinstance(reply, Streamed)
    assert (reply.message.model, reply.message.content, reply.message.stop_reason) == ("claude-opus-5-5", (Text("(stayed silent)"),), "end_turn")
    exchange = only_exchange(wire)
    assert exchange.kind == MainTurn() and isinstance(exchange.reply, Held) and exchange.reply.said == "(stayed silent)"
    assert [type(seen) for seen in wire.seen] == [Sent, Exchanged]


async def test_a_held_request_that_did_not_ask_for_a_stream_is_answered_whole(serve: Callable[[Handler], Awaitable[tuple[Upstream, Wire]]]) -> None:
    upstream, wire = await serve(streamed)
    wire.route = lambda _sent: Hold("(interrupted)")
    status, headers, body = await post(wire.proxy.url, REQUEST.replace(b'"stream": true', b'"stream": false'))
    assert upstream.asked == [] and status == 200 and headers["Content-Type"].startswith("application/json")
    answer = json.loads(body)
    assert (answer["content"], answer["stop_reason"], answer["model"]) == ([{"type": "text", "text": "(interrupted)"}], "end_turn", "claude-opus-5-5")


async def test_a_route_that_raises_is_logged_and_the_request_goes_on_as_it_came(serve: Callable[[Handler], Awaitable[tuple[Upstream, Wire]]]) -> None:
    upstream, wire = await serve(streamed)

    def broken(_sent: Sent) -> Route:
        raise RuntimeError("the stage fell over")

    wire.route = broken
    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(message.record["message"]), level="ERROR")
    try:
        status, _, body = await post(wire.proxy.url)
    finally:
        logger.remove(sink)
    assert (status, body) == (200, b"".join(STREAM)) and len(upstream.asked) == 1
    assert [error.startswith("the proxy's route failed") for error in errors] == [True]


def test_the_daemon_keeps_one_audit_line_per_exchange_and_nothing_per_event() -> None:
    lines: list[Entry] = []
    observe = wire_to(lines.append)
    exchange = Exchanged("e1", None, MainTurn(), "POST", "/v1/messages", 1, 1.0, 2.0, Unreached("refused", 3.0))
    observe(Sent("e1", None, MainTurn(), None))
    observe(Heard("e1", TextDelta(0, "hi")))
    observe(exchange)
    assert lines == [exchange]
