"""The wire: what one Claude Code process asks the API and what the API says back, as values.

Bytes and parsed JSON in, typed values out. The proxy in `hands.sessions.proxy` moves the bytes and times them; this
module only reads them. Ported in shape from cc-dump's `pipeline/event_types.py` and `response_assembler.py`, with
one change of rule: nothing it cannot read is defaulted into something it can. A frame it does not know is kept as
`Unparsed`, and a reply built from one is `Garbled`, never a message missing a part.
"""

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, cast

from hands.core.session import PromptId, SessionId
from hands.core.trace import Span

# Where Claude Code reaches the API when nothing says otherwise.
UPSTREAM = "https://api.anthropic.com"

# ── What a request is ────────────────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MainTurn:
    """The session's own loop asking the model for its next step: the one kind of request whose reply is the session speaking.

    `prompt` is the turn it asks for, the id its hooks carry as prompt_id; None when the request names none. Claude Code
    names it only when its ANTHROPIC_BASE_URL is Anthropic's API (2.1.285): a session given a gateway, and the brain
    through hands' proxy, name none.
    """

    prompt: PromptId | None


@dataclass(frozen=True)
class Subagent:
    """A subagent's loop asking for its next step: under its session's header, and never the session speaking."""


@dataclass(frozen=True)
class Fork:
    """A side question: a request sharing the session's prefix, as /btw's does, whose reply never enters its history."""


@dataclass(frozen=True)
class Compaction:
    """The fork that summarises the session's history so the summary can replace it."""


@dataclass(frozen=True)
class CountTokens:
    """A count of what some content would cost in context: a round trip, not a model call."""


@dataclass(frozen=True)
class Elsewhere:
    """A request to an endpoint other than the model's - a starting Claude Code's hello, its account and settings, Remote
    Control's worker - at `path`: a round trip, not a model call, and nothing in it is ever spoken."""

    path: str


@dataclass(frozen=True)
class Unknown:
    """A request no rule recognises, and what about it did not match. Its reply is never spoken."""

    shape: str


Kind = MainTurn | Subagent | Fork | Compaction | CountTokens | Elsewhere | Unknown

# The first words of every compaction request's last message, in Claude Code's services/compact/prompt.ts.
COMPACTION_OPENING = "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools."
# The first words Claude Code wraps every side question in, in its utils/sideQuestion.ts.
SIDE_QUESTION_OPENING = "<system-reminder>This is a side question from the user."
# The line Claude Code opens its system prompt with, in its attribution header builder: `key=value;` pairs after it.
BILLING_OPENING = "x-anthropic-billing-header:"


_ENDPOINTS = ("/v1/messages/count_tokens", "/v1/messages")


def classify(path: str, body: object) -> Kind:
    """Which kind of request this is, from its path and its parsed body alone, decided before any reply exists.

    Claude Code's own loop puts its newest message-level cache marker on the last message; a fire-and-forget fork
    (skipCacheWrite in services/api/claude.ts) puts it on the one before, the last point it shares with the loop.
    Compaction and side questions are told by the words Claude Code opens their prompts with, not by where their marker
    lands: a side question asked during a turn is merged into the turn's prompt and followed by another message, and one
    asked before any turn has finished carries no marker at all (hands-wire-6ic.l2o, 2.1.285).
    """
    # [LAW:dataflow-not-control-flow] every rule is a value test on the request; a messages request none of them
    # matches is Unknown, so a shape no one has seen yet is heard and never spoken.
    path = path.split("?", 1)[0]
    # The endpoint ends the path: a gateway named in ANTHROPIC_BASE_URL puts its own path before it.
    match next((endpoint for endpoint in _ENDPOINTS if path.endswith(endpoint)), path):
        case "/v1/messages/count_tokens":
            return CountTokens()
        case "/v1/messages":
            return _classify_messages(body)
        case other if "/v1/messages" in other:
            # [LAW:no-silent-failure] a path of the model's own that is neither endpoint is heard, never passed over.
            return Unknown(f"a request to {other}")
        case other:
            # A session whose API is Anthropic's reaches many of them, Remote Control's every few seconds; only the model's
            # endpoint has shapes still being learned.
            return Elsewhere(other)


def _classify_messages(body: object) -> Kind:
    if not isinstance(body, Mapping):
        return Unknown("a messages request whose body is not a JSON object")
    request = cast(Mapping[str, object], body)
    messages = _list(request.get("messages"))
    if not messages:
        return Unknown("a messages request with no messages")
    # Any block of any message since the last reply: Claude Code merges adjacent user messages, so a prompt of its own
    # can follow a prompt just typed or a tool's result, and a message of its own can follow it (2.1.285).
    if any(text.startswith(COMPACTION_OPENING) for text in asked(request)):
        return Compaction()
    if any(text.startswith(SIDE_QUESTION_OPENING) for text in asked(request)):
        return Fork()
    if not _list(request.get("tools")):
        return Unknown(f"a messages request with no tools, to {request.get('model')!r}")
    marked = [index for index, message in enumerate(messages) if _cache_marked(message)]
    last = len(messages) - 1
    billing = _billing(request)
    # By the newest marker: a working session also marks an earlier message of a long history, which the brain's
    # slim requests never did (2.1.285, markers on messages 5 and 7 of 8).
    if marked and marked[-1] == last and billing.get("cc_is_subagent") == "true":
        # A subagent's requests carry its parent's session header and shape; only this line tells them apart (2.1.285).
        return Subagent()
    if marked and marked[-1] == last:
        prompt = billing.get("cc_prompt_id")
        return MainTurn(None if prompt is None else PromptId(prompt))
    if marked and marked[-1] == last - 1:
        return Fork()
    return Unknown(f"a messages request with cache markers on messages {marked} of {len(messages)}")


def _billing(request: Mapping[str, object]) -> Mapping[str, str]:
    """The pairs of the billing line Claude Code opens its system prompt with; none when it has no such line."""
    system = request.get("system")
    texts = [system] if isinstance(system, str) else [text for block in _mappings(_list(system)) if isinstance(text := block.get("text"), str)]
    line = next((text for text in texts if text.startswith(BILLING_OPENING)), "").partition("\n")[0].removeprefix(BILLING_OPENING)
    pairs = (pair.strip().partition("=") for pair in line.split(";"))
    return {name: value for name, sep, value in pairs if sep}


def asked(body: object) -> tuple[str, ...]:
    """The text of a messages request after the model's last reply: what the request is asking now."""
    messages = _list(cast(Mapping[str, object], body).get("messages")) if isinstance(body, Mapping) else []
    return tuple(text for message in messages[_since_reply(messages) :] for text in _texts(message))


def tool_names(body: object) -> tuple[str, ...]:
    """The names of the tools a messages request offers the model."""
    tools = _list(cast(Mapping[str, object], body).get("tools")) if isinstance(body, Mapping) else []
    return tuple(name for tool in _mappings(tools) if isinstance(name := tool.get("name"), str))


def _since_reply(messages: Sequence[object]) -> int:
    """Where the messages after the model's last reply begin."""
    replies = [index for index, message in enumerate(messages) if _role(message) == "assistant"]
    return replies[-1] + 1 if replies else 0


def _role(message: object) -> object:
    return cast(Mapping[str, object], message).get("role") if isinstance(message, Mapping) else None


def _mappings(values: Sequence[object]) -> list[Mapping[str, object]]:
    """The JSON objects among some values."""
    return [cast(Mapping[str, object], value) for value in values if isinstance(value, Mapping)]


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
    """A tool call's result as a request hands it to the model: the id of the call it answers, its text, whether it
    failed, and the turn it was made in, counted from 0 at the first prompt the request carries."""

    call: str
    text: str
    is_error: bool
    turn: int


def tool_answers(body: object) -> tuple[ToolAnswer, ...]:
    """Every tool result a messages request carries, each by the id of the call it answers.

    A request carries its whole history, so which of these answer the model's last reply is known only to whoever heard
    that reply's calls.
    """
    answers: list[ToolAnswer] = []
    for turn, message in _turned(body):
        results = [item for item in _mappings(_blocks(message)) if item.get("type") == "tool_result"]
        answers.extend(
            ToolAnswer(call, "".join(_texts(result)), result.get("is_error") is True, turn) for result in results if isinstance(call := result.get("tool_use_id"), str)
        )
    return tuple(answers)


def turns(body: object) -> int:
    """How many turns a messages request's history holds, the one in flight included."""
    return max((turn + 1 for turn, _ in _turned(body)), default=0)


def _turned(body: object) -> list[tuple[int, object]]:
    """Each message of a request with the turn it belongs to, counted from 0 at the first prompt."""
    messages = _list(cast(Mapping[str, object], body).get("messages")) if isinstance(body, Mapping) else []
    turned: list[tuple[int, object]] = []
    turn = -1
    for message in messages:
        # A turn opens with a prompt: a message from the user that answers no call.
        turn += _role(message) == "user" and not any(item.get("type") == "tool_result" for item in _mappings(_blocks(message)))
        turned.append((turn, message))
    return turned


def tool_calls(body: object) -> Mapping[str, "ToolUse"]:
    """Every tool call the model made in a request's history, by its id."""
    messages = _list(cast(Mapping[str, object], body).get("messages")) if isinstance(body, Mapping) else []
    uses = [item for message in messages for item in _mappings(_blocks(message)) if item.get("type") == "tool_use"]
    return {
        call: ToolUse(call, name, cast(Mapping[str, object], given))
        for use in uses
        if isinstance(call := use.get("id"), str) and isinstance(name := use.get("name"), str) and isinstance(given := use.get("input"), Mapping)
    }


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
            return _started(_object(data["message"]))
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


def _started(message: Mapping[str, object]) -> MessageStarted:
    """A message's head, as its stream's message_start carries it and as a message sent whole begins."""
    return MessageStarted(id=_str(message["id"]), model=_str(message["model"]), usage=_object(message.get("usage", {})))


def _array(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"expected an array, got {type(value).__name__}")
    return cast(list[object], value)


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
    """A message the model wrote: a streamed reply put back together, or one a request without streaming was answered with."""

    id: str
    model: str
    content: tuple[Block, ...]
    stop_reason: str | None
    usage: Mapping[str, object]


@dataclass(frozen=True)
class Written:
    """A message the model wrote, every part of it read: streamed to message_stop, or sent whole when the request asked
    for no stream, as Claude Code asks again after a stream that fails before any block of it is finished (2.1.289)."""

    message: Message
    streamed: bool


@dataclass(frozen=True)
class Garbled:
    """A reply that cannot be put back together, and why: an error mid-stream, a frame or a part not read, or an early end.

    The bytes still reached the client unchanged; only hands' reading of them failed, so nothing is said from them.
    """

    reason: str


@dataclass(frozen=True)
class Answered:
    """A reply sent whole that is not a message: a count_tokens answer, or an error the API sent instead of one."""

    body: object


@dataclass(frozen=True)
class Unkept:
    """A reply from an endpoint other than the model's, read none of and kept none of: what those hold - the account, a
    key made for it, Remote Control's tokens - is the user's credentials, never the session speaking. Its size is on its
    exchange."""


Body = Written | Garbled | Answered | Unkept


# A block not yet stopped: what it holds so far, and the pieces of a tool call's input not yet joined.
@dataclass(frozen=True)
class _Open:
    block: Block
    json: str


class _Broken(Exception):
    pass


def assemble(events: Sequence[WireEvent], streamed: bool) -> Written | Garbled:
    """The message these events carry, folded in order: a stream's as its frames came, or the ones a message sent whole
    stands for."""
    try:
        return Written(_fold(events), streamed)
    except _Broken as broken:
        return Garbled(str(broken))


def usage_after(usage: Mapping[str, object], delta: Mapping[str, object]) -> Mapping[str, object]:
    """A reply's usage once a message_delta has said its own: each count the delta gives is the reply's total so far and
    replaces the one before, and a count it gives as null is one it did not give."""
    return {**usage, **{name: count for name, count in delta.items() if count is not None}}


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
                message = replace(message, stop_reason=stop_reason, usage=usage_after(message.usage, usage))
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


def unstreamed(data: bytes) -> tuple[WireEvent, ...] | Answered | Garbled:
    """A reply sent whole: the events its stream would have carried when it is a message, so it is heard and assembled as
    one; else its JSON when it is JSON, else its text as it came."""
    text = data.decode("utf-8", errors="replace")
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        return Answered(text)
    match body:
        case {"type": "message"}:
            try:
                return _streamed(cast(dict[str, object], body))
            except (KeyError, TypeError) as error:
                return Garbled(f"a message sent whole lacks a field or has one of the wrong type: {error!r}")
        case _:
            return Answered(body)


def _streamed(message: Mapping[str, object]) -> tuple[WireEvent, ...]:
    """The events a stream of this whole message would have carried, its usage given at its start, whole already."""
    blocks = [_object(block) for block in _array(message["content"])]
    stop = message.get("stop_reason")
    return (
        _started(message),
        *(event for index, block in enumerate(blocks) for event in (*_block_streamed(index, block), BlockStopped(index))),
        MessageDelta(stop_reason=None if stop is None else _str(stop), usage={}),
        MessageStopped(),
    )


def _block_streamed(index: int, block: Mapping[str, object]) -> tuple[WireEvent, ...]:
    """A whole block as its stream would carry it: opened empty, then what it holds as deltas."""
    match block:
        case {"type": "text", "text": str() as text}:
            return BlockStarted(index, {**block, "text": ""}), TextDelta(index, text)
        case {"type": "thinking", "thinking": str() as thinking, "signature": str() as signature}:
            return BlockStarted(index, {**block, "thinking": "", "signature": ""}), ThinkingDelta(index, thinking), SignatureDelta(index, signature)
        case {"type": "tool_use", "input": dict()}:
            return BlockStarted(index, {**block, "input": {}}), JsonDelta(index, json.dumps(block["input"]))
        case _:
            # [LAW:single-enforcer] a block with nothing to stream, or one hands does not read, opens as it came, and
            # the fold reads or refuses it as it would a streamed one.
            return (BlockStarted(index, block),)


# A content-type is a stream when it names text/event-stream, whatever parameters follow.
_EVENT_STREAM = re.compile(r"^\s*text/event-stream\b", re.IGNORECASE)


def is_stream(content_type: str) -> bool:
    return _EVENT_STREAM.match(content_type) is not None


# ── Where a request goes ─────────────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Stub:
    """A tool result sent as one line in place of its text: the id of the call it answers, and the line."""

    call: str
    line: str


@dataclass(frozen=True)
class Steer:
    """A compaction sent with `prompt` in place of the summarisation prompt Claude Code wrote after the last reply.

    What Claude Code writes after its own prompt stays: the /compact arguments or a PreCompact hook's instructions, and
    the closing reminder.
    """

    prompt: str


@dataclass(frozen=True)
class Tail:
    """`text` added as one more text block at the end of the newest message.

    After the block carrying the request's cache marker, so the cached prefix is what Claude Code sent, and the next
    request, whose history lacks the tail, misses none of it.
    """

    text: str


Change = Stub | Steer | Tail


@dataclass(frozen=True)
class Send:
    """The request goes to the API with these changes made to it, and as it came when there are none."""

    changes: tuple[Change, ...] = ()
    # What the client makes of the API refusing it: asks again on its own schedule, or, told the refusal is final, ends
    # there, so the refusal is heard the moment it comes.
    refusal: Literal["retried", "final"] = "retried"
    # [LAW:types-are-the-program] the exchange's span, which its record carries, as the router chose it: inside the unit
    # of work the request was made for, or the root of a trace of its own for one made for none.
    span: Span = field(kw_only=True)


@dataclass(frozen=True)
class Hold:
    """The request is answered by hands and never reaches the API, with `said` as the model's whole reply.

    Kept in the client's history as the model's own words and never spoken. Not empty: to an empty reply Claude Code
    2.1.285 answers "[Your previous response had no visible output. Please continue ...]" and asks again, and its
    history keeps the tool results with no reply after them.
    """

    said: str
    # The exchange's span, as Send's is: a request hands holds is still the part of the unit it was made for.
    span: Span


Route = Send | Hold


def edited(body: object, changes: Sequence[Change]) -> Mapping[str, object]:
    """A messages request with each change made to it; raises ValueError naming a change that has nothing to change."""
    if not isinstance(body, Mapping):
        raise ValueError("a request with no JSON object for a body has nothing to change")
    request = cast(Mapping[str, object], body)
    lines = {change.call: change.line for change in changes if isinstance(change, Stub)}
    messages = [_edited_blocks(message, lambda block: _stubbed(block, lines)) for message in _list(request.get("messages"))]
    stubbed = {item.get("tool_use_id") for message in messages for item in _mappings(_blocks(message)) if item.get("type") == "tool_result"}
    for change in changes:
        # [LAW:no-silent-failure] a change that found nothing to change is a request hands misread, said before it goes.
        match change:
            case Stub(call=call) if call not in stubbed:
                raise ValueError(f"no tool result answers call {call}")
            case Steer(prompt=prompt):
                if not any(text.startswith(COMPACTION_OPENING) for text in asked(request)):
                    raise ValueError("no message since the last reply holds a compaction prompt to steer")
                since = _since_reply(messages)
                messages[since:] = [_steered(message, prompt) for message in messages[since:]]
            case _:
                pass
    tails: list[object] = [{"type": "text", "text": change.text} for change in changes if isinstance(change, Tail)]
    if tails:
        newest = _mappings(messages[-1:])
        if not newest or not _blocks(newest[0]):
            raise ValueError(f"the newest message is {messages[-1:]!r}, not blocks to append to")
        messages[-1] = {**newest[0], "content": [*_blocks(newest[0]), *tails]}
    return {**request, "messages": messages}


# What Claude Code writes after its compaction prompt, in services/compact/prompt.ts: the instructions it was given, if
# any, then the closing reminder.
COMPACTION_INSTRUCTIONS = "\n\nAdditional Instructions:\n"
COMPACTION_REMINDER = "\n\nREMINDER: Do NOT call any tools."


def _edited_blocks(message: object, edit: Callable[[object], object]) -> object:
    match message:
        case {"content": list()}:
            return {**cast(Mapping[str, object], message), "content": [edit(block) for block in _blocks(message)]}
        case _:
            return message


def _stubbed(block: object, lines: Mapping[str, str]) -> object:
    match block:
        case {"type": "tool_result", "tool_use_id": str() as call} if call in lines:
            return {**cast(Mapping[str, object], block), "content": lines[call]}
        case _:
            return block


def _steered(message: object, prompt: str) -> object:
    """A message with its compaction prompt replaced, whether its content is a string or blocks, as `_texts` reads it."""
    match message:
        case {"content": str() as text}:
            return {**cast(Mapping[str, object], message), "content": _steered_text(text, prompt)}
        case _:
            return _edited_blocks(message, lambda block: _steered_block(block, prompt))


def _steered_block(block: object, prompt: str) -> object:
    match block:
        case {"type": "text", "text": str() as text}:
            return {**cast(Mapping[str, object], block), "text": _steered_text(text, prompt)}
        case _:
            return block


def _steered_text(text: str, prompt: str) -> str:
    if not text.startswith(COMPACTION_OPENING):
        return text
    ends = [at for at in (text.find(COMPACTION_INSTRUCTIONS), text.find(COMPACTION_REMINDER)) if at >= 0]
    if not ends:
        raise ValueError("the compaction prompt has no closing reminder to keep")
    return prompt + text[min(ends) :]


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
class UsageLimitReached:
    """The model's API account has reached its usage limit; `returns` is when access comes back, if the API said."""

    returns: Seconds | None


@dataclass(frozen=True)
class Answering:
    """The head of the API's answer, heard before any of it is passed on: its status, and the spent usage limit it refuses
    under when that is why. So what the client does with the answer, its StopFailure hook included, comes after."""

    exchange: str
    status: int
    limit: UsageLimitReached | None


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
class Uncopied:
    """A tapped exchange whose copy broke off before its reply's head reached hands: how it ended is not known here.

    The session had its reply or its error from the upstream whatever became of the copy.
    """

    reason: str
    lost_at: Seconds


@dataclass(frozen=True)
class Exchanged:
    """One request and its reply, whole: the wide record of one unit of the proxy's work."""

    exchange: str
    session: SessionId | None
    kind: Kind
    method: str
    path: str
    request_bytes: int
    # What hands changed of the request before it went on; none for a request that went as it came.
    changes: tuple[Change, ...]
    requested_at: Seconds
    sent_at: Seconds
    reply: Reached | Unreached | Held | Uncopied
    # The client was answered the proxy's own refusal, told final (x-should-retry: false), in place of the API's answer.
    final: bool
    # [LAW:one-source-of-truth] the exchange's own span: in the trace of the unit of work it was made for, so this record
    # is that unit's part and no second one is kept of it, or the root of a trace of its own for a request made as part of
    # none, so every exchange is in the trace record.
    span: Span


Observed = Sent | Heard | Answering | Exchanged
