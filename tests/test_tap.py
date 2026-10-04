"""The tap: a wrapped session's exchange, copied to hands by its fritter, is read as the proxy reads the brain's."""

import asyncio
import base64
import json
import os
import shutil
import tempfile
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from pathlib import Path

import pytest
from aiohttp import web

from hands.core.events import Closed
from hands.core.session import PromptId, SessionId
from hands.core.wire import Block, Elsewhere, Exchanged, Fork, Garbled, Heard, Kind, MainTurn, Message, Observed, Reached, Sent, Streamed, Subagent, Text, TextDelta, ToolUse, Uncopied, Unkept, Unreached
from hands.sessions.audit import CopiesLost, Entry
from hands.sessions.tap import moves, serve_tap
from hands.sessions.wide import root

REQUEST = (
    b'{"model": "claude-opus-5-5", "tools": [{"name": "Read"}], "stream": true, "messages": '
    b'[{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}]}'
)


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


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def request_line(lost: int = 0) -> dict[str, object]:
    headers = [["X-Claude-Code-Session-Id", "s1"], ["Content-Type", "application/json"]]
    return {"kind": "request", "at": 100.0, "method": "POST", "path": "/v1/messages?beta=true", "headers": headers, "body": b64(REQUEST), "lost": lost}


HEAD: Mapping[str, object] = {"kind": "response", "at": 100.5, "status": 200, "headers": [["Content-Type", "text/event-stream"]]}
CHUNKS: list[Mapping[str, object]] = [{"kind": "bytes", "at": 101.0 + index, "bytes": b64(chunk)} for index, chunk in enumerate(STREAM)]
END: Mapping[str, object] = {"kind": "end", "at": 110.0, "error": ""}


@pytest.fixture
def socket_path() -> Iterator[Path]:
    # Short on purpose: macOS caps a unix socket path near 104 bytes.
    root = Path(tempfile.mkdtemp(prefix="tap-", dir="/tmp")).resolve()
    yield root / "wire.sock"
    shutil.rmtree(root)


class Heard_:
    def __init__(self) -> None:
        self.observed: list[Observed] = []
        self.entries: list[Entry] = []
        self.done = asyncio.Event()

    def observe(self, observed: Observed) -> None:
        self.observed.append(observed)
        if isinstance(observed, Exchanged):
            self.done.set()

    def exchanged(self) -> Exchanged:
        [exchanged] = [observed for observed in self.observed if isinstance(observed, Exchanged)]
        return exchanged


@pytest.fixture
async def tap(socket_path: Path) -> AsyncIterator[Heard_]:
    heard = Heard_()
    server = await serve_tap(socket_path, heard.observe, heard.entries.append, clock=lambda: 999.0)
    yield heard
    server.close()


async def copy(path: Path, lines: Sequence[Mapping[str, object]], raw: bytes = b"") -> None:
    _, writer = await asyncio.open_unix_connection(str(path))
    writer.write(b"".join(json.dumps(line).encode() + b"\n" for line in lines) + raw)
    await writer.drain()
    writer.close()


async def test_a_copied_exchange_is_the_wire_s_values_with_the_session_s_id(tap: Heard_, socket_path: Path) -> None:
    await copy(socket_path, [request_line(), HEAD, *CHUNKS, END])
    await asyncio.wait_for(tap.done.wait(), 5)

    sent = tap.observed[0]
    assert isinstance(sent, Sent) and (sent.session, sent.kind) == (SessionId("s1"), MainTurn(None))
    deltas = [observed.event.text for observed in tap.observed if isinstance(observed, Heard) and isinstance(observed.event, TextDelta)]
    assert deltas == ["hello ", "from the wire"]
    exchanged = tap.exchanged()
    assert (exchanged.exchange, exchanged.session, exchanged.kind, exchanged.path, exchanged.changes) == (sent.exchange, SessionId("s1"), MainTurn(None), "/v1/messages?beta=true", ())
    assert (exchanged.requested_at, exchanged.sent_at, exchanged.request_bytes) == (100.0, 100.0, len(REQUEST))
    match exchanged.reply:
        case Reached(status=200, first_byte_at=101.0, last_byte_at=107.0, reply_bytes=size, body=Streamed(message=message)):
            assert size == sum(len(chunk) for chunk in STREAM)
            assert message.content == (Text("hello from the wire"),) and message.stop_reason == "end_turn"
        case other:
            pytest.fail(f"the reply was read as {other!r}")
    assert tap.entries == []
    # A wrapped session's exchange is made for no unit of hands' work: the root of a trace of its own.
    assert exchanged.span.parent_id is None and (len(exchanged.span.trace_id), len(exchanged.span.span_id)) == (32, 16)


async def test_a_reply_from_elsewhere_than_the_model_s_endpoint_is_kept_as_its_size_alone(tap: Heard_, socket_path: Path) -> None:
    profile = b'{"account": {"email_address": "user@example.com"}, "raw_key": "sk-ant-secret"}'
    await copy(
        socket_path,
        [
            {**request_line(), "method": "GET", "path": "/api/oauth/profile", "body": None},
            {**HEAD, "headers": [["Content-Type", "application/json"]]},
            {"kind": "bytes", "at": 101.0, "bytes": b64(profile)},
            END,
        ],
    )
    await asyncio.wait_for(tap.done.wait(), 5)
    exchanged = tap.exchanged()
    assert exchanged.kind == Elsewhere("/api/oauth/profile")
    assert exchanged.reply == Reached(200, 101.0, 101.0, len(profile), Unkept())


async def test_copies_lost_before_this_one_are_a_line_of_the_session_that_lost_them(tap: Heard_, socket_path: Path) -> None:
    await copy(socket_path, [request_line(lost=3), HEAD, *CHUNKS, END])
    await asyncio.wait_for(tap.done.wait(), 5)
    assert tap.entries == [CopiesLost(SessionId("s1"), 3)]


async def test_a_reply_cut_short_upstream_is_garbled_with_what_cut_it(tap: Heard_, socket_path: Path) -> None:
    await copy(socket_path, [request_line(), HEAD, *CHUNKS[:3], {**END, "error": "the reply ended early: EOF"}])
    await asyncio.wait_for(tap.done.wait(), 5)
    assert isinstance(reply := tap.exchanged().reply, Reached) and reply.body == Garbled("the reply ended early: EOF")


async def test_an_upstream_never_reached_is_unreached(tap: Heard_, socket_path: Path) -> None:
    await copy(socket_path, [request_line(), {"kind": "unreached", "at": 100.2, "error": "connection refused"}])
    await asyncio.wait_for(tap.done.wait(), 5)
    assert tap.exchanged().reply == Unreached("connection refused", 100.2)


async def test_a_copy_that_breaks_off_says_how_far_it_got(tap: Heard_, socket_path: Path) -> None:
    await copy(socket_path, [request_line()])
    await asyncio.wait_for(tap.done.wait(), 5)
    assert isinstance(tap.exchanged().reply, Uncopied)

    tap.done.clear()
    tap.observed.clear()
    await copy(socket_path, [request_line(), HEAD, *CHUNKS[:2]])
    await asyncio.wait_for(tap.done.wait(), 5)
    reply = tap.exchanged().reply
    assert isinstance(reply, Reached) and isinstance(reply.body, Garbled) and "broke off" in reply.body.reason


async def test_a_line_hands_cannot_read_ends_the_copy_as_garbled_and_the_next_copy_is_read(tap: Heard_, socket_path: Path) -> None:
    await copy(socket_path, [request_line(), HEAD], raw=b"not json\n")
    await asyncio.wait_for(tap.done.wait(), 5)
    reply = tap.exchanged().reply
    assert isinstance(reply, Reached) and isinstance(reply.body, Garbled) and "cannot read" in reply.body.reason

    tap.done.clear()
    tap.observed.clear()
    await copy(socket_path, [request_line(), HEAD, *CHUNKS, END])
    await asyncio.wait_for(tap.done.wait(), 5)
    assert isinstance(reply := tap.exchanged().reply, Reached) and isinstance(reply.body, Streamed)


async def test_a_line_out_of_its_order_ends_the_copy_as_broken_and_the_exchange_is_still_told(tap: Heard_, socket_path: Path) -> None:
    await copy(socket_path, [request_line(), CHUNKS[0]])
    await asyncio.wait_for(tap.done.wait(), 5)
    reply = tap.exchanged().reply
    assert isinstance(reply, Uncopied) and "Chunk line out of its order" in reply.reason

    tap.done.clear()
    tap.observed.clear()
    await copy(socket_path, [request_line(), HEAD, HEAD])
    await asyncio.wait_for(tap.done.wait(), 5)
    reply = tap.exchanged().reply
    assert isinstance(reply, Reached) and isinstance(reply.body, Garbled) and "Response line out of its order" in reply.body.reason


async def test_the_socket_is_the_user_s_alone(tap: Heard_, socket_path: Path) -> None:
    assert socket_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(shutil.which("curl") is None, reason="needs curl to be the session")
async def test_a_session_under_a_real_fritter_is_answered_by_the_api_and_heard_on_the_wire(tap: Heard_, socket_path: Path, fritter: Path) -> None:

    async def messages(request: web.Request) -> web.StreamResponse:
        assert request.headers["X-Api-Key"] == "the-sessions-own"
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for chunk in STREAM:
            await response.write(chunk)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/messages", messages)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    body = socket_path.parent / "request.json"
    body.write_bytes(REQUEST)
    session = (
        'curl -sS -X POST -H "X-Api-Key: the-sessions-own" -H "X-Claude-Code-Session-Id: s1" '
        f'--data-binary @{body} "http://{host}:{port}/v1/messages?beta=true"'
    )
    argv = [str(fritter), "--socket-dir", str(socket_path.parent), "--tap", f"http://{host}:{port}", "--tap-ca", "NODE_EXTRA_CA_CERTS", "--tap-to", str(socket_path), "--", "sh", "-c", session]
    controller, terminal = os.openpty()
    process = await asyncio.create_subprocess_exec(*argv, stdin=terminal, stdout=terminal, stderr=terminal, start_new_session=True)
    os.close(terminal)
    printed = b""
    loop = asyncio.get_running_loop()
    while chunk := await loop.run_in_executor(None, _read, controller):
        printed += chunk
    await process.wait()
    os.close(controller)
    await asyncio.wait_for(tap.done.wait(), 5)
    await runner.cleanup()

    assert b"".join(STREAM).decode() in printed.decode().replace("\r\n", "\n")
    exchanged = tap.exchanged()
    assert (exchanged.session, exchanged.kind) == (SessionId("s1"), MainTurn(None))
    assert isinstance(exchanged.reply, Reached) and isinstance(exchanged.reply.body, Streamed)


def _read(fd: int) -> bytes:
    try:
        return os.read(fd, 4096)
    except OSError:  # EIO: the last process holding the terminal let go of it
        return b""


# ── What a session's exchange says of its turn ─────────────────────────────────────────────────────────────────────

TURN = PromptId("p1")


def replied(kind: Kind, stop_reason: str, *content: Block) -> Exchanged:
    message = Message("m1", "claude-opus-5-5", content, stop_reason, {})
    return Exchanged("e1", SessionId("s1"), kind, "POST", "/v1/messages", 1, (), 1.0, 1.0, Reached(200, 2.0, 3.0, 10, Streamed(message)), False, root())


def test_a_reply_that_ends_the_turn_with_text_closes_it_with_its_last_text_as_its_stop_carries_it() -> None:
    thought = replied(MainTurn(TURN), "end_turn", Text("Done."), Text("Pushed."))
    assert moves(thought) == (Closed(SessionId("s1"), TURN, "Pushed."),)


@pytest.mark.parametrize(
    "observed",
    [
        Sent("e1", SessionId("s1"), MainTurn(TURN), {}),
        # Claude Code asks again after an end_turn with no text (2.1.285): the turn goes on.
        replied(MainTurn(TURN), "end_turn"),
        replied(MainTurn(TURN), "tool_use", Text("Reading."), ToolUse("t1", "Read", {})),
        replied(MainTurn(None), "end_turn", Text("Done.")),
        replied(Subagent(), "end_turn", Text("Found it.")),
        replied(Fork(), "end_turn", Text("An aside.")),
    ],
)
def test_what_does_not_end_a_turn_the_request_names_closes_nothing(observed: Observed) -> None:
    assert moves(observed) == ()
