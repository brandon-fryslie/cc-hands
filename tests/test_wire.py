"""The wire read as values: which kind a request is, whose it is, and what a streamed reply said."""

import json

from hands.core.session import SessionId
from hands.core.wire import (
    COMPACTION_OPENING,
    Answered,
    BlockStarted,
    Compaction,
    CountTokens,
    Fork,
    Frame,
    Garbled,
    MainTurn,
    Message,
    Ping,
    RedactedThinking,
    Streamed,
    Text,
    TextDelta,
    Thinking,
    ToolUse,
    Unknown,
    Unparsed,
    answered,
    assemble,
    classify,
    frames,
    is_stream,
    parse,
    session_of,
)

TOOLS: list[object] = [{"name": "Read", "input_schema": {"type": "object"}}]
MARKED = {"type": "ephemeral"}


def said(text: str, marked: bool = False) -> dict[str, object]:
    block: dict[str, object] = {"type": "text", "text": text}
    return {"role": "user", "content": [{**block, "cache_control": MARKED} if marked else block]}


def request(*messages: dict[str, object], tools: list[object] = TOOLS) -> dict[str, object]:
    return {"model": "claude-opus-5-5", "tools": tools, "system": [{"type": "text", "text": "you are"}], "messages": list(messages)}


# The shapes below are the ones Claude Code 2.1.284 sent through the proxy on 2026-09-29: its own loop marks the last
# message, and its compaction, a fork that skips the cache write, marks the one before.


def test_the_loop_marks_its_last_message_and_is_a_main_turn() -> None:
    assert classify("/v1/messages?beta=true", request(said("hi"), said("ok"), said("go on", marked=True))) == MainTurn()


def test_a_fork_marks_the_message_before_its_own_and_is_not_a_main_turn() -> None:
    assert classify("/v1/messages?beta=true", request(said("hi"), said("ok", marked=True), said("side question"))) == Fork()


def test_the_compaction_prompt_makes_a_compaction_whatever_its_marker() -> None:
    last = said(COMPACTION_OPENING + "\n\nYour task is to create a detailed summary")
    assert classify("/v1/messages", request(said("hi"), said("ok", marked=True), last)) == Compaction()


def test_count_tokens_is_known_by_its_path() -> None:
    assert classify("/v1/messages/count_tokens?beta=true", {"messages": [said("a file")]}) == CountTokens()


def test_every_shape_no_rule_matches_is_unknown_and_says_what_it_was() -> None:
    assert classify("/api/hello", None) == Unknown("a request to /api/hello")
    assert classify("/v1/messages", None) == Unknown("a messages request whose body is not a JSON object")
    assert classify("/v1/messages", request()) == Unknown("a messages request with no messages")
    assert classify("/v1/messages", request(said("title this", marked=True), tools=[])) == Unknown(
        "a messages request with no tools, to 'claude-opus-5-5'"
    )
    assert classify("/v1/messages", request(said("a"), said("b"))) == Unknown("a messages request with cache markers on messages [] of 2")
    assert classify("/v1/messages", request(said("a", marked=True), said("b"), said("c"))) == Unknown(
        "a messages request with cache markers on messages [0] of 3"
    )


def test_a_request_names_its_session_in_a_header_of_any_case() -> None:
    assert session_of({"X-Claude-Code-Session-Id": "e4b2b84e"}) == SessionId("e4b2b84e")
    assert session_of({"x-claude-code-session-id": "e4b2b84e"}) == SessionId("e4b2b84e")
    assert session_of({"User-Agent": "curl"}) is None


def sse(event: str, data: dict[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


# A reply with every kind of block hands reads: thinking and its signature, text in pieces, and a tool call whose
# input arrives as JSON split mid-token, with a ping between.
STREAM = b"".join(
    [
        sse("message_start", {"type": "message_start", "message": {"id": "msg_1", "model": "claude-opus-5-5", "usage": {"input_tokens": 3, "output_tokens": 1}}}),
        sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
        sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}}),
        sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}}),
        sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        sse("ping", {"type": "ping"}),
        sse("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
        sse("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Reading é"}}),
        sse("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": " now."}}),
        sse("content_block_stop", {"type": "content_block_stop", "index": 1}),
        sse("content_block_start", {"type": "content_block_start", "index": 2, "content_block": {"type": "redacted_thinking", "data": "opaque"}}),
        sse("content_block_stop", {"type": "content_block_stop", "index": 2}),
        sse("content_block_start", {"type": "content_block_start", "index": 3, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}}}),
        sse("content_block_delta", {"type": "content_block_delta", "index": 3, "delta": {"type": "input_json_delta", "partial_json": '{"file_pa'}}),
        sse("content_block_delta", {"type": "content_block_delta", "index": 3, "delta": {"type": "input_json_delta", "partial_json": 'th": "/a"}'}}),
        sse("content_block_stop", {"type": "content_block_stop", "index": 3}),
        sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 42}}),
        sse("message_stop", {"type": "message_stop"}),
    ]
)

MESSAGE = Message(
    id="msg_1",
    model="claude-opus-5-5",
    content=(Thinking("hmm", "sig"), Text("Reading é now."), RedactedThinking("opaque"), ToolUse("toolu_1", "Read", {"file_path": "/a"})),
    stop_reason="tool_use",
    usage={"input_tokens": 3, "output_tokens": 42},
)


def read_in_pieces(data: bytes, size: int) -> list[Frame]:
    found: list[Frame] = []
    rest = b""
    for start in range(0, len(data), size):
        whole, rest = frames(rest + data[start : start + size])
        found.extend(whole)
    assert rest == b""
    return found


def test_a_stream_reads_the_same_frames_however_its_bytes_are_cut() -> None:
    # One byte at a time splits every frame, every line ending, and the two bytes of the é.
    whole, rest = frames(STREAM)
    assert rest == b""
    for size in (1, 2, 7, 64, len(STREAM)):
        assert read_in_pieces(STREAM, size) == whole


def test_crlf_line_endings_and_comment_lines_read_as_plain_ones() -> None:
    crlf = b": keepalive\r\nevent: ping\r\ndata: {\"type\": \"ping\"}\r\n\r\n"
    assert read_in_pieces(crlf, 1) == [Frame("ping", '{"type": "ping"}')]


def test_a_whole_stream_assembles_into_the_message_it_carried() -> None:
    assert assemble([parse(frame) for frame in frames(STREAM)[0]]) == Streamed(MESSAGE)


def test_text_arrives_as_its_own_event_for_whatever_speaks_it() -> None:
    events = [parse(frame) for frame in frames(STREAM)[0]]
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["Reading é", " now."]
    assert Ping() in events



def test_an_error_mid_stream_garbles_the_reply() -> None:
    cut = STREAM.split(b"event: content_block_start", 1)[0]
    error = sse("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
    assert assemble([parse(frame) for frame in frames(cut + error)[0]]) == Garbled(
        'the API sent an error mid-stream: {"type": "overloaded_error", "message": "Overloaded"}'
    )


def test_a_stream_cut_short_is_garbled_not_a_shorter_message() -> None:
    cut = STREAM.split(b"event: message_stop", 1)[0]
    assert assemble([parse(frame) for frame in frames(cut)[0]]) == Garbled("the stream ended before message_stop")


def test_a_frame_that_cannot_be_read_is_kept_and_garbles_the_reply() -> None:
    odd = Frame("content_block_delta", json.dumps({"type": "content_block_delta", "index": 1, "delta": {"type": "citations_delta"}}))
    assert parse(odd) == Unparsed(odd, "no event is named 'content_block_delta' with this data")
    broken = Frame("message_start", "{not json")
    assert isinstance(parse(broken), Unparsed)
    body = STREAM.replace(b"event: ping", b"event: nothing_known")
    assert assemble([parse(frame) for frame in frames(body)[0]]) == Garbled(
        "a 'nothing_known' frame was not read: no event is named 'nothing_known' with this data"
    )


def test_a_block_hands_does_not_read_garbles_the_reply() -> None:
    body = STREAM.replace(b'"type": "redacted_thinking", "data": "opaque"', b'"type": "server_tool_use", "id": "s1", "name": "web_search"')
    assert assemble([parse(frame) for frame in frames(body)[0]]) == Garbled("a content block of type 'server_tool_use', which hands does not read")


def test_a_block_start_is_parsed_whole() -> None:
    start = parse(Frame("content_block_start", json.dumps({"index": 0, "content_block": {"type": "text", "text": ""}})))
    assert start == BlockStarted(0, {"type": "text", "text": ""})


def test_a_reply_that_is_not_a_stream_is_its_json_or_else_its_text() -> None:
    assert answered(b'{"input_tokens": 5583}') == Answered({"input_tokens": 5583})
    assert answered(b"<html>bad gateway</html>") == Answered("<html>bad gateway</html>")
    assert is_stream("text/event-stream; charset=utf-8")
    assert not is_stream("application/json")
