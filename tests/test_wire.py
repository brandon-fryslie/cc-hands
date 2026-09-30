"""The wire read as values: which kind a request is, whose it is, and what a streamed reply said."""

import json

import pytest
from typing import cast

from hands.core.session import SessionId
from hands.core.wire import (
    ToolAnswer,
    tool_answers,
    COMPACTION_INSTRUCTIONS,
    COMPACTION_OPENING,
    COMPACTION_REMINDER,
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
    Tail,
    SIDE_QUESTION_OPENING,
    Stub,
    Steer,
    tool_calls,
    turns,
    answered,
    edited,
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
# message, and its compaction marks the one before.


def test_the_loop_marks_its_last_message_and_is_a_main_turn() -> None:
    assert classify("/v1/messages?beta=true", request(said("hi"), said("ok"), said("go on", marked=True))) == MainTurn()


def test_a_fork_that_skips_the_cache_write_marks_the_message_before_its_own_and_is_not_a_main_turn() -> None:
    assert classify("/v1/messages?beta=true", request(said("hi"), said("ok", marked=True), said("suggest the next prompt"))) == Fork()


QUESTION = SIDE_QUESTION_OPENING + " You must answer this question directly in a single response.</system-reminder>\n\nwhat did the read say?"


def reply(text: str, marked: bool = False) -> dict[str, object]:
    return {**said(text, marked), "role": "assistant"}


# The three shapes of a side question 2.1.285 sent through the proxy (hands-wire-6ic.l2o), which differ in everything
# but the words the question is wrapped in.
SIDE_QUESTIONS = {
    # Between turns: the question follows the last reply, which carries the marker.
    "between turns": request(said("hi"), reply("hello", marked=True), said(QUESTION)),
    # During a turn: merged into the prompt in flight, and followed by a message of Claude Code's own.
    "during a turn": request(
        said("hi"),
        reply("hello", marked=True),
        {"role": "user", "content": [{"type": "text", "text": "read the file"}, {"type": "text", "text": QUESTION}]},
        {"role": "system", "content": [{"type": "text", "text": "<system-reminder>tokens left</system-reminder>"}]},
    ),
    # Before the first turn has finished: merged into the first prompt, with no marker anywhere.
    "before the first turn": request({"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "text", "text": QUESTION}]}),
}


@pytest.mark.parametrize("shape", SIDE_QUESTIONS)
def test_a_side_question_is_a_fork_by_its_wrapper_wherever_its_marker_lands(shape: str) -> None:
    assert classify("/v1/messages?beta=true", SIDE_QUESTIONS[shape]) == Fork()


def test_a_side_question_quoted_in_history_does_not_make_the_next_turn_a_fork() -> None:
    body = request(said(QUESTION), reply("the read said nothing"), said("thanks", marked=True))
    assert classify("/v1/messages", body) == MainTurn()


def test_the_compaction_prompt_makes_a_compaction_whatever_its_marker() -> None:
    last = said(COMPACTION_OPENING + "\n\nYour task is to create a detailed summary")
    assert classify("/v1/messages", request(said("hi"), said("ok", marked=True), last)) == Compaction()
    # Merged after a prompt just typed, as Claude Code merges adjacent user messages.
    merged: dict[str, object] = {"role": "user", "content": [{"type": "text", "text": "fix the test"}, *cast(list[object], last["content"])]}
    assert classify("/v1/messages", request(said("hi"), said("ok", marked=True), merged)) == Compaction()


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


def test_a_block_of_comments_alone_is_no_frame_and_the_stream_still_assembles() -> None:
    body = STREAM.replace(b"event: ping", b": keepalive\n\nevent: ping")
    assert frames(body)[0] == frames(STREAM)[0]
    assert assemble([parse(frame) for frame in frames(body)[0]]) == Streamed(MESSAGE)


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


def test_a_stream_out_of_order_is_garbled_not_a_message_missing_a_part() -> None:
    def garbled(body: bytes) -> str:
        reply = assemble([parse(frame) for frame in frames(body)[0]])
        assert isinstance(reply, Garbled)
        return reply.reason

    # The tool call never stops, so its input never joined: not a ToolUse with no input.
    assert garbled(STREAM.replace(sse("content_block_stop", {"type": "content_block_stop", "index": 3}), b"")) == "blocks [3] never stopped"
    start = STREAM.split(b"event: content_block_start", 1)[0]
    assert garbled(STREAM.replace(b"event: message_delta", start + b"event: message_delta")) == "a second message_start"
    assert garbled(STREAM + sse("ping", {"type": "ping"})) == "a Ping after message_stop"
    late = sse("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "late"}})
    assert garbled(STREAM.replace(b"event: message_delta", late + b"event: message_delta")) == "a Text delta for block 1, which is not an open Text block"


def test_a_frame_whose_number_json_cannot_hold_is_unparsed() -> None:
    assert isinstance(parse(Frame("message_start", '{"n": ' + "9" * 5000 + "}")), Unparsed)


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


def test_tool_answers_are_every_result_in_the_request_by_the_call_it_answers() -> None:
    body: dict[str, object] = {
        "messages": [
            {"role": "user", "content": "file it"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Filing."},
                {"type": "tool_use", "id": "a", "name": "mcp__hands__stage_draft", "input": {}},
                {"type": "tool_use", "id": "b", "name": "Bash", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": [{"type": "text", "text": '{"readback": "staged"}'}]},
                {"type": "tool_result", "tool_use_id": "b", "content": "exit 1", "is_error": True},
                {"type": "text", "text": "<system-reminder>noted</system-reminder>"},
            ]},
            # Claude Code 2.1.285 follows the results with a message of its own, as the brain's requests showed.
            {"role": "system", "content": [{"type": "text", "text": "<system-reminder>tokens left</system-reminder>", "cache_control": {"type": "ephemeral"}}]},
        ]
    }
    assert tool_answers(body) == (ToolAnswer("a", '{"readback": "staged"}', False, 0), ToolAnswer("b", "exit 1", True, 0))
    assert tool_answers({"messages": [{"role": "user", "content": "hi"}]}) == ()
    assert tool_answers(None) == ()


def test_the_tail_goes_after_every_block_of_the_newest_message_and_nothing_before_it_moves() -> None:
    result: dict[str, object] = {"type": "tool_result", "tool_use_id": "t1", "content": "ok", "cache_control": MARKED}
    tool_result: dict[str, object] = {"role": "user", "content": [result]}
    earlier: list[dict[str, object]] = [said("hi"), {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}]
    sent = request(*earlier, tool_result)
    amended = edited(sent, (Tail("[hands] how they stand"),))
    assert amended == {**sent, "messages": [*earlier, {"role": "user", "content": [result, {"type": "text", "text": "[hands] how they stand"}]}]}
    # Still the loop's own request: the marker has not moved.
    assert classify("/v1/messages", amended) == MainTurn()
    # The request Claude Code keeps in its history is untouched.
    assert tool_result == {"role": "user", "content": [result]}


# A string for content has no block to carry a cache marker, so no main turn's newest message is one.
REFUSED: list[object] = [
    None,
    {"messages": []},
    {"messages": [{"role": "user", "content": []}]},
    {"messages": [{"role": "user"}]},
    {"messages": [{"role": "user", "content": "hi"}]},
]


@pytest.mark.parametrize("body", REFUSED)
def test_a_request_with_no_newest_message_to_append_to_is_refused(body: object) -> None:
    with pytest.raises(ValueError):
        edited(body, (Tail("tail"),))


def result(call: str, text: str) -> dict[str, object]:
    return {"type": "tool_result", "tool_use_id": call, "content": [{"type": "text", "text": text}]}


def called(call: str, name: str = "Read") -> dict[str, object]:
    return {"role": "assistant", "content": [{"type": "tool_use", "id": call, "name": name, "input": {"file_path": f"/{call}"}}]}


def answering(*results: dict[str, object]) -> dict[str, object]:
    return {"role": "user", "content": list(results)}


def test_a_turn_opens_at_each_prompt_and_its_results_are_counted_in_it() -> None:
    body = request(
        said("read a"), called("a"), answering(result("a", "A")), reply("read it"),
        said("read b"), called("b"), answering(result("b", "B")),
        {"role": "system", "content": [{"type": "text", "text": "<system-reminder>tokens left</system-reminder>"}]},
    )
    assert turns(body) == 2
    assert [(answer.call, answer.turn) for answer in tool_answers(body)] == [("a", 0), ("b", 1)]
    assert tool_calls(body) == {"a": ToolUse("a", "Read", {"file_path": "/a"}), "b": ToolUse("b", "Read", {"file_path": "/b"})}
    assert turns(None) == 0


def test_a_stub_replaces_only_the_result_it_names_and_the_request_keeps_its_shape() -> None:
    body = request(said("read a"), called("a"), answering(result("a", "A" * 500), {"type": "text", "text": "note"}), reply("ok"), said("go", marked=True))
    changed = edited(body, (Stub("a", "Read: the file holds A."),))
    assert changed == {
        **body,
        "messages": [
            said("read a"), called("a"),
            answering({**result("a", ""), "content": "Read: the file holds A."}, {"type": "text", "text": "note"}),
            reply("ok"), said("go", marked=True),
        ],
    }
    assert classify("/v1/messages", changed) == MainTurn()


def test_a_steer_replaces_the_newest_compaction_prompt_keeps_what_follows_it_and_it_is_still_a_compaction() -> None:
    quoted = said(COMPACTION_OPENING + " quoted earlier" + COMPACTION_REMINDER)
    after = COMPACTION_INSTRUCTIONS + "keep the draft" + COMPACTION_REMINDER + " Respond with plain text only."
    body = request(quoted, reply("ok", marked=True), said(COMPACTION_OPENING + " summarise the code" + after))
    changed = edited(body, (Steer(COMPACTION_OPENING + " summarise the voice session"),))
    assert changed["messages"] == [quoted, reply("ok", marked=True), said(COMPACTION_OPENING + " summarise the voice session" + after)]
    assert classify("/v1/messages", changed) == Compaction()


def test_a_steer_replaces_a_compaction_prompt_sent_as_a_plain_string() -> None:
    body = request(said("hi"), reply("ok", marked=True), {"role": "user", "content": COMPACTION_OPENING + " summarise" + COMPACTION_REMINDER})
    assert cast(list[dict[str, object]], edited(body, (Steer("voice"),))["messages"])[-1]["content"] == "voice" + COMPACTION_REMINDER


def test_a_steer_refuses_a_compaction_prompt_with_no_closing_reminder() -> None:
    with pytest.raises(ValueError, match="no closing reminder"):
        edited(request(said(COMPACTION_OPENING + " cut short")), (Steer("voice"),))


def test_a_change_with_nothing_to_change_is_refused() -> None:
    with pytest.raises(ValueError, match="no tool result answers call b"):
        edited(request(said("hi", marked=True)), (Stub("b", "Read: B"),))
    with pytest.raises(ValueError, match="no compaction prompt"):
        edited(request(said("hi", marked=True)), (Steer("summarise"),))
