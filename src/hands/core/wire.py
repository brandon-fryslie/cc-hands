"""The wire: what one Claude Code process asks the API and what the API says back, as values.

Bytes and parsed JSON in, typed values out. The proxy in `hands.sessions.proxy` moves the bytes and times them; this
module only reads them. Ported in shape from cc-dump's `pipeline/event_types.py` and `response_assembler.py`, with
one change of rule: nothing it cannot read is defaulted into something it can. A frame it does not know is kept as
`Unparsed`, and a reply built from one is `Garbled`, never a message missing a part.
"""

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import cast

from hands.core.session import SessionId

# ── What a request is ────────────────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MainTurn:
    """The session's own loop asking the model for its next step: the one kind of request whose reply is the session speaking."""


@dataclass(frozen=True)
class Fork:
    """A side request sharing the session's prefix, such as /btw or a prompt suggestion, whose reply never enters its history."""


@dataclass(frozen=True)
class Compaction:
    """The fork that summarises the session's history so the summary can replace it."""


@dataclass(frozen=True)
class CountTokens:
    """A count of what some content would cost in context: a round trip, not a model call."""


@dataclass(frozen=True)
class Unknown:
    """A request no rule recognises, and what about it did not match. Its reply is never spoken."""

    shape: str


Kind = MainTurn | Fork | Compaction | CountTokens | Unknown

# The first words of every compaction request's last message, in Claude Code's services/compact/prompt.ts.
COMPACTION_OPENING = "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools."


def classify(path: str, body: object) -> Kind:
    """Which kind of request this is, from its path and its parsed body alone, decided before any reply exists.

    Claude Code puts exactly one message-level cache marker on a request. Its own loop puts it on the last message;
    a fire-and-forget fork (skipCacheWrite in services/api/claude.ts) puts it on the one before, the last point it
    shares with the loop, and that placement is the fork's mark on the wire. A fork that does not skip the write
    is marked like the loop and is not told apart here.
    """
    # [LAW:dataflow-not-control-flow] every rule is a value test on the request; what none of them matches is Unknown,
    # so a shape no one has seen yet is heard and never spoken.
    match path.split("?", 1)[0]:
        case "/v1/messages/count_tokens":
            return CountTokens()
        case "/v1/messages":
            return _classify_messages(body)
        case other:
            return Unknown(f"a request to {other}")


def _classify_messages(body: object) -> Kind:
    if not isinstance(body, Mapping):
        return Unknown("a messages request whose body is not a JSON object")
    request = cast(Mapping[str, object], body)
    messages = _list(request.get("messages"))
    if not messages:
        return Unknown("a messages request with no messages")
    # Any block: Claude Code merges adjacent user messages, so the compaction prompt can follow a prompt just typed.
    if any(text.startswith(COMPACTION_OPENING) for text in _texts(messages[-1])):
        return Compaction()
    if not _list(request.get("tools")):
        return Unknown(f"a messages request with no tools, to {request.get('model')!r}")
    marked = [index for index, message in enumerate(messages) if _cache_marked(message)]
    last = len(messages) - 1
    if marked == [last]:
        return MainTurn()
    if marked == [last - 1]:
        return Fork()
    return Unknown(f"a messages request with cache markers on messages {marked} of {len(messages)}")


def _list(value: object) -> list[object]:
    return cast(list[object], value) if isinstance(value, list) else []


def _content(message: object) -> object:
    return cast(Mapping[str, object], message).get("content") if isinstance(message, Mapping) else None


def _blocks(message: object) -> list[object]:
    return _list(_content(message))


def _cache_marked(message: object) -> bool:
    return any(isinstance(block, Mapping) and "cache_control" in block for block in _blocks(message))


def _texts(message: object) -> list[str]:
    """A message's text, whether its content is a string or a list of blocks."""
    content = _content(message)
    if isinstance(content, str):
        return [content]
    texts = [cast(Mapping[str, object], block).get("text") for block in _blocks(message) if isinstance(block, Mapping)]
    return [text for text in texts if isinstance(text, str)]


@dataclass(frozen=True)
class ToolAnswer:
    """A tool call's result as a request hands it to the model: the id of the call it answers, its text, and whether it failed."""

    call: str
    text: str
    is_error: bool


def tool_answers(body: object) -> tuple[ToolAnswer, ...]:
    """Every tool result a messages request carries, each by the id of the call it answers.

    A request carries its whole history, so which of these answer the model's last reply is known only to whoever heard
    that reply's calls.
    """
    messages = _list(cast(Mapping[str, object], body).get("messages")) if isinstance(body, Mapping) else []
    results = [cast(Mapping[str, object], item) for message in messages for item in _blocks(message) if isinstance(item, Mapping)]
    return tuple(
        ToolAnswer(call, "".join(_texts(result)), result.get("is_error") is True)
        for result in results
        if result.get("type") == "tool_result" and isinstance(call := result.get("tool_use_id"), str)
    )


SESSION_HEADER = "x-claude-code-session-id"


def session_of(headers: Mapping[str, str]) -> SessionId | None:
    """The Claude Code session a request came from, which its client names on every request, count_tokens included.

    Not metadata.user_id: count_tokens requests carry no metadata, and since Claude Code 2.x that field is a JSON
    string rather than cc-dump's user_<hash>_account_<uuid>_session_<uuid>. None for a client that is not Claude Code.
    """
    found = next((value for name, value in headers.items() if name.lower() == SESSION_HEADER), None)
    return None if found is None else SessionId(found)


# ── The stream back ──────────────────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Frame:
    """One server-sent event as it came off the wire: its event name and its data, before either is read."""

    event: str
    data: str


def frames(buffer: bytes) -> tuple[list[Frame], bytes]:
    """The whole frames at the front of the buffer, and the bytes after them that are not a whole frame yet.

    The rest is handed back with the next chunk appended, so a frame, a line ending, or a UTF-8 character split
    across chunks is read whole once its last byte arrives.
    """
    normal = buffer.replace(b"\r\n", b"\n")
    end = normal.rfind(b"\n\n")
    whole, rest = (normal[: end + 2], normal[end + 2 :]) if end >= 0 else (b"", normal)
    # A block with no data field, a keepalive of comment lines or a blank one, dispatches nothing.
    blocks = [block.split("\n") for block in whole.decode("utf-8").split("\n\n")]
    return [_frame(lines) for lines in blocks if any(line.startswith("data") for line in lines)], rest


def _frame(lines: list[str]) -> Frame:
    fields = [line.partition(":") for line in lines if not line.startswith(":")]
    event = next((value.strip() for name, _, value in fields if name == "event"), "message")
    data = "\n".join(value.removeprefix(" ") for name, _, value in fields if name == "data")
    return Frame(event, data)


@dataclass(frozen=True)
class MessageStarted:
    id: str
    model: str
    usage: Mapping[str, object]


@dataclass(frozen=True)
class BlockStarted:
    """A content block opened, as the API described it before any delta."""

    index: int
    block: Mapping[str, object]


@dataclass(frozen=True)
class TextDelta:
    index: int
    text: str


@dataclass(frozen=True)
class ThinkingDelta:
    index: int
    thinking: str


@dataclass(frozen=True)
class SignatureDelta:
    index: int
    signature: str


@dataclass(frozen=True)
class JsonDelta:
    """A piece of a tool call's input, which is JSON only once every piece has arrived."""

    index: int
    partial_json: str


@dataclass(frozen=True)
class BlockStopped:
    index: int


@dataclass(frozen=True)
class MessageDelta:
    stop_reason: str | None
    usage: Mapping[str, object]


@dataclass(frozen=True)
class MessageStopped:
    pass


@dataclass(frozen=True)
class Ping:
    pass


@dataclass(frozen=True)
class StreamError:
    """The API reporting an error mid-stream, an overload most often, after its 200 was already sent."""

    error: Mapping[str, object]


@dataclass(frozen=True)
class Unparsed:
    """A frame this module cannot read, kept as it came, with why."""

    frame: Frame
    reason: str


WireEvent = (
    MessageStarted
    | BlockStarted
    | TextDelta
    | ThinkingDelta
    | SignatureDelta
    | JsonDelta
    | BlockStopped
    | MessageDelta
    | MessageStopped
    | Ping
    | StreamError
    | Unparsed
)


def parse(frame: Frame) -> WireEvent:
    """The one place a frame's JSON is read: every event downstream is typed, or is Unparsed saying why."""
    # [LAW:single-enforcer] the parse boundary for the response stream; nothing after it reads raw frames.
    try:
        data = json.loads(frame.data)
    except ValueError as error:
        return Unparsed(frame, f"its data is not JSON: {error}")
    if not isinstance(data, dict):
        return Unparsed(frame, "its data is not a JSON object")
    try:
        return _event(frame.event, cast(dict[str, object], data)) or Unparsed(frame, f"no event is named {frame.event!r} with this data")
    except (KeyError, TypeError) as error:
        return Unparsed(frame, f"a field is missing or of the wrong type: {error!r}")


def _event(name: str, data: dict[str, object]) -> WireEvent | None:
    match name:
        case "message_start":
            message = _object(data["message"])
            return MessageStarted(id=_str(message["id"]), model=_str(message["model"]), usage=_object(message.get("usage", {})))
        case "content_block_start":
            return BlockStarted(index=_int(data["index"]), block=_object(data["content_block"]))
        case "content_block_delta":
            return _delta(_int(data["index"]), _object(data["delta"]))
        case "content_block_stop":
            return BlockStopped(index=_int(data["index"]))
        case "message_delta":
            stop = _object(data["delta"]).get("stop_reason")
            return MessageDelta(stop_reason=None if stop is None else _str(stop), usage=_object(data.get("usage", {})))
        case "message_stop":
            return MessageStopped()
        case "ping":
            return Ping()
        case "error":
            return StreamError(error=_object(data["error"]))
        case _:
            return None


def _delta(index: int, delta: Mapping[str, object]) -> WireEvent | None:
    match delta.get("type"):
        case "text_delta":
            return TextDelta(index, _str(delta["text"]))
        case "thinking_delta":
            return ThinkingDelta(index, _str(delta["thinking"]))
        case "signature_delta":
            return SignatureDelta(index, _str(delta["signature"]))
        case "input_json_delta":
            return JsonDelta(index, _str(delta["partial_json"]))
        case _:
            return None


def _object(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"expected an object, got {type(value).__name__}")
    return cast(dict[str, object], value)


def _str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected a string, got {type(value).__name__}")
    return value


def _int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"expected an integer, got {type(value).__name__}")
    return value


# ── The reply, assembled ─────────────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class Thinking:
    thinking: str
    signature: str


@dataclass(frozen=True)
class RedactedThinking:
    data: str


@dataclass(frozen=True)
class ToolUse:
    id: str
    name: str
    input: Mapping[str, object]


Block = Text | Thinking | RedactedThinking | ToolUse


@dataclass(frozen=True)
class Message:
    """A streamed reply put back together: the same message a request without streaming would have been answered with."""

    id: str
    model: str
    content: tuple[Block, ...]
    stop_reason: str | None
    usage: Mapping[str, object]


@dataclass(frozen=True)
class Streamed:
    """A stream that ran to message_stop with every frame read."""

    message: Message


@dataclass(frozen=True)
class Garbled:
    """A stream that cannot be put back together, and why: an error mid-stream, a frame not read, or an early end.

    The bytes still reached the client unchanged; only hands' reading of them failed, so nothing is said from them.
    """

    reason: str


@dataclass(frozen=True)
class Answered:
    """A reply that was not a stream: a count_tokens answer, or an error the API sent instead of a stream."""

    body: object


Body = Streamed | Garbled | Answered


# A block not yet stopped: what it holds so far, and the pieces of a tool call's input not yet joined.
@dataclass(frozen=True)
class _Open:
    block: Block
    json: str


class _Broken(Exception):
    pass


def assemble(events: Sequence[WireEvent]) -> Streamed | Garbled:
    """The reply a whole stream carried, folded from its events in order."""
    try:
        return Streamed(_fold(events))
    except _Broken as broken:
        return Garbled(str(broken))


def _fold(events: Sequence[WireEvent]) -> Message:
    message: Message | None = None
    # [LAW:types-are-the-program] a block is _Open until its stop and a Block after it, so a delta after the stop and a
    # block that never stopped cannot pass for a finished one.
    blocks: dict[int, _Open | Block] = {}
    stopped = False
    for event in events:
        if stopped:
            raise _Broken(f"a {type(event).__name__} after message_stop")
        match event:
            case MessageStarted(id=id, model=model, usage=usage):
                if message is not None:
                    raise _Broken("a second message_start")
                message = Message(id=id, model=model, content=(), stop_reason=None, usage=usage)
            case BlockStarted(index=index, block=block):
                if index in blocks:
                    raise _Broken(f"block {index} started twice")
                blocks[index] = _Open(_opened(block), "")
            case TextDelta(index=index, text=text):
                blocks[index] = _grown(blocks, index, Text, lambda block: replace(block, text=block.text + text))
            case ThinkingDelta(index=index, thinking=thinking):
                blocks[index] = _grown(blocks, index, Thinking, lambda block: replace(block, thinking=block.thinking + thinking))
            case SignatureDelta(index=index, signature=signature):
                blocks[index] = _grown(blocks, index, Thinking, lambda block: replace(block, signature=block.signature + signature))
            case JsonDelta(index=index, partial_json=partial):
                building = blocks.get(index)
                if not isinstance(building, _Open) or not isinstance(building.block, ToolUse):
                    raise _Broken(f"a tool input delta for block {index}, which is not an open tool call")
                blocks[index] = replace(building, json=building.json + partial)
            case BlockStopped(index=index):
                blocks[index] = _closed(blocks, index)
            case MessageDelta(stop_reason=stop_reason, usage=usage):
                if message is None:
                    raise _Broken("a message_delta before message_start")
                message = replace(message, stop_reason=stop_reason, usage={**message.usage, **usage})
            case MessageStopped():
                stopped = True
            case Ping():
                pass
            case StreamError(error=error):
                raise _Broken(f"the API sent an error mid-stream: {json.dumps(error)}")
            case Unparsed(frame=frame, reason=reason):
                raise _Broken(f"a {frame.event!r} frame was not read: {reason}")
    if message is None:
        raise _Broken("the stream ended with no message_start")
    if not stopped:
        raise _Broken("the stream ended before message_stop")
    content = tuple(block for _, block in sorted(blocks.items()) if not isinstance(block, _Open))
    if len(content) != len(blocks):
        raise _Broken(f"blocks {sorted(index for index, block in blocks.items() if isinstance(block, _Open))} never stopped")
    return replace(message, content=content)


def _opened(block: Mapping[str, object]) -> Block:
    try:
        match block.get("type"):
            case "text":
                return Text(_str(block["text"]))
            case "thinking":
                return Thinking(_str(block["thinking"]), _str(block.get("signature", "")))
            case "redacted_thinking":
                return RedactedThinking(_str(block["data"]))
            case "tool_use":
                return ToolUse(_str(block["id"]), _str(block["name"]), {})
            case other:
                raise _Broken(f"a content block of type {other!r}, which hands does not read")
    except (KeyError, TypeError) as error:
        raise _Broken(f"a {block.get('type')!r} block whose start lacks a field: {error!r}") from error


def _grown[B: Block](blocks: Mapping[int, _Open | Block], index: int, kind: type[B], grow: Callable[[B], B]) -> _Open:
    building = blocks.get(index)
    if not isinstance(building, _Open) or not isinstance(building.block, kind):
        raise _Broken(f"a {kind.__name__} delta for block {index}, which is not an open {kind.__name__} block")
    return replace(building, block=grow(building.block))


def _closed(blocks: Mapping[int, _Open | Block], index: int) -> Block:
    building = blocks.get(index)
    if not isinstance(building, _Open):
        raise _Broken(f"content_block_stop for block {index}, which is not open")
    if not isinstance(building.block, ToolUse):
        return building.block
    try:
        parsed = json.loads(building.json or "{}")
    except json.JSONDecodeError as error:
        raise _Broken(f"tool call {building.block.name}'s input is not JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise _Broken(f"tool call {building.block.name}'s input is not a JSON object")
    return replace(building.block, input=cast(dict[str, object], parsed))


def answered(data: bytes) -> Answered:
    """A reply that was not a stream: its JSON when it is JSON, else its text as it came."""
    text = data.decode("utf-8", errors="replace")
    try:
        return Answered(json.loads(text))
    except json.JSONDecodeError:
        return Answered(text)


# A content-type is a stream when it names text/event-stream, whatever parameters follow.
_EVENT_STREAM = re.compile(r"^\s*text/event-stream\b", re.IGNORECASE)


def is_stream(content_type: str) -> bool:
    return _EVENT_STREAM.match(content_type) is not None


# ── One exchange ─────────────────────────────────────────────────────────────────────────────────────────────────────

Seconds = float  # wall-clock seconds since the epoch, so the proxy's times read beside the audit log's own


@dataclass(frozen=True)
class Sent:
    """A request, classified, as it leaves for the API: its body parsed, or None when it is not JSON."""

    exchange: str
    session: SessionId | None
    kind: Kind
    body: object


@dataclass(frozen=True)
class Heard:
    """One event of a streamed reply, as its frame arrived."""

    exchange: str
    event: WireEvent


@dataclass(frozen=True)
class Reached:
    """The API answered: its status, when its first and last bytes arrived, and what the answer said."""

    status: int
    first_byte_at: Seconds
    last_byte_at: Seconds
    reply_bytes: int
    body: Body


@dataclass(frozen=True)
class Unreached:
    """The API was never heard from: the connection failed before a status came back, and the client was told 502."""

    error: str
    failed_at: Seconds


@dataclass(frozen=True)
class Held:
    """hands answered the request itself and the API never saw it: the model was not asked, and is recorded as saying `said`."""

    said: str
    answered_at: Seconds


@dataclass(frozen=True)
class Exchanged:
    """One request and its reply, whole: the wide record of one unit of the proxy's work."""

    exchange: str
    session: SessionId | None
    kind: Kind
    method: str
    path: str
    request_bytes: int
    # What hands appended to the newest message before it went on; empty for a request that went as it came.
    appended: str
    requested_at: Seconds
    sent_at: Seconds
    reply: Reached | Unreached | Held


Observed = Sent | Heard | Exchanged


# ── Where a request goes ─────────────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Forward:
    """The request goes to the API as it came."""


@dataclass(frozen=True)
class Hold:
    """The request is answered by hands and never reaches the API, with `said` as the model's whole reply.

    Kept in the client's history as the model's own words and never spoken. Not empty: to an empty reply Claude Code
    2.1.285 answers "[Your previous response had no visible output. Please continue ...]" and asks again, and its
    history keeps the tool results with no reply after them.
    """

    said: str


@dataclass(frozen=True)
class Append:
    """The request goes to the API with `tail` as one more text block at the end of its newest message.

    After the block carrying the request's cache marker, so the cached prefix is what Claude Code sent, and the next
    request, whose history lacks the tail, misses none of it.
    """

    tail: str


Route = Forward | Append | Hold


def appended(body: object, tail: str) -> Mapping[str, object]:
    """A messages request with `tail` added as a text block after every block of its newest message; nothing before it moves."""
    if not isinstance(body, Mapping):
        raise ValueError("a request with no JSON object for a body has no message to append to")
    request = cast(Mapping[str, object], body)
    messages = _list(request.get("messages"))
    if not messages or not isinstance(messages[-1], Mapping):
        raise ValueError("a request with no newest message has nothing to append to")
    newest = cast(Mapping[str, object], messages[-1])
    content = newest.get("content")
    # A string is one text block written short.
    blocks: list[object] = [{"type": "text", "text": content}] if isinstance(content, str) else _list(content)
    if not blocks:
        raise ValueError(f"the newest message's content is {content!r}, not blocks to append to")
    return {**request, "messages": [*messages[:-1], {**newest, "content": [*blocks, {"type": "text", "text": tail}]}]}
