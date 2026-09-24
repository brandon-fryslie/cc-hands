"""What a session's JSONL transcript knows that its hooks do not carry: one record at a time, as it is written."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from hands.core.session import PromptId
from hands.core.turn import Asked, Interruption, Notified, Opening, Ref
from hands.sessions.payload import Payload, Rejected

# Records are written without spaces, so this finds every title record cheaply.
# It also finds a nested object with that type inside another record, so a
# candidate counts only when the record's own type is the title.
_AI_TITLE_RECORD = b'"type":"ai-title"'


def ai_title(transcript: Path) -> str | None:
    """The newest title Claude Code gave the session, or None before it has named one."""
    try:
        raw = transcript.read_bytes()
    except FileNotFoundError:
        # Claude Code creates the transcript with its first record.
        return None
    # [LAW:no-ambient-temporal-coupling] Claude Code appends while this reads, and
    # a record is whole only once its newline is written, so the tail after the
    # last newline is never parsed.
    *complete, _unfinished = raw.split(b"\n")
    title = None
    for line in complete:
        if _AI_TITLE_RECORD in line:
            record = Payload.parse(line)
            if record.text("type") == "ai-title":
                title = record.text("aiTitle")
    return title


# Only these two record types carry a turn; the rest (attachments, modes, titles, snapshots) are skipped unparsed.
_TURN_RECORDS = (b'"type":"user"', b'"type":"assistant"')


def turn_record(line: bytes) -> Payload | None:
    """The record this line holds if it is one a turn is made of, and None for every other line.

    Raises Rejected for a line that is not a record at all.
    """
    if not any(marker in line for marker in _TURN_RECORDS):
        return None
    record = Payload.parse(line)
    fields = record.fields
    if fields.get("type") not in ("user", "assistant") or fields.get("isSidechain") is True:
        # A subagent's own records are its transcript's, and are narrated there.
        return None
    # [LAW:parse-dont-validate] the message is shaped here, at the one place a line becomes a record, so that
    # reading a turn out of it cannot raise halfway through a record its reader has already begun to consume.
    shaped = message(record)
    # Claude Code's own stand-in for a reply that never came, not something Claude said.
    return None if shaped.get("model") == "<synthetic>" and result_text(shaped.get("content")) == _NO_RESPONSE else record


# What Claude Code writes as the reply to a turn the user stopped at a question: 19 of this machine's interrupts are followed by it.
_NO_RESPONSE = "No response requested."


# The whole of the record Claude Code writes as a turn's last when the user stops it at the keyboard: the second when a
# tool was running, the first otherwise. The same two texts in every version on this machine, 2.1.201 to 2.1.281.
# The text is the mark, not the record's interruptedMessageId: a turn stopped before Claude wrote anything names no
# message, and 162 of the 2,000 such records here carry none.
_INTERRUPTED = ("[Request interrupted by user]", "[Request interrupted by user for tool use]")


def edge_of(record: Payload, mid_tool: bool) -> Opening | Interruption | None:
    """Where this record begins a turn, or cuts the one under way off; None for a record in the middle of one."""
    if record.fields.get("type") == "user" and result_text(message(record).get("content")) in _INTERRUPTED:
        # Written as a user's message with no tool result in it, so it would otherwise read as the next prompt.
        return Interruption(ref_of(record))
    return _opening_of(record, mid_tool)


def prompt_of(record: Payload) -> PromptId | None:
    """The prompt_id of the turn a record belongs to, which Claude Code writes on the user's side of it."""
    value = record.fields.get("promptId")
    return PromptId(value) if isinstance(value, str) else None


def _opening_of(record: Payload, mid_tool: bool) -> Opening | None:
    """What opens a turn: a prompt or a notification Claude Code handed a session that was not waiting on a tool.

    `mid_tool` says whether the record before this one was a tool call or its result.
    """
    if mid_tool:
        # Sent while a tool ran: Claude Code folds it into the turn already under way, whose Stop has not come.
        return None
    fields = record.fields
    # Meta records (skill bodies, command caveats) and compaction's summary are Claude Code's own, not a new request.
    if fields.get("type") != "user" or fields.get("isMeta") is True or fields.get("isCompactSummary") is True:
        return None
    parts = blocks(record)
    match message(record).get("content"):
        case str() as text:
            pass
        case list() if parts and not any(block.get("type") == "tool_result" for block in parts):
            # A prompt with an image or a document attached, or one sent through the SDK. Anything but a tool result,
            # rather than a list of the block kinds known today: a kind added tomorrow would otherwise stop opening the
            # turn it opens, which hands the whole turn to an older opening, where a block nobody named costs the
            # summariser some JSON inside `budget.opening` and nothing else.
            text = result_text(parts)
        case _:
            return None
    ref = ref_of(record)
    match fields.get("origin"):
        case {"kind": "task-notification"}:
            return Notified(ref, text)
        case _:
            return Asked(ref, text)


def holds_a_tool(record: Payload) -> bool:
    """Whether this record is a tool call or a tool's result, which is what makes the record after it mid-turn."""
    return any(block.get("type") in ("tool_use", "tool_result") for block in blocks(record))


def ref_of(record: Payload) -> Ref | None:
    value = record.fields.get("uuid")
    return Ref(value) if isinstance(value, str) else None


def structured_result(record: Payload) -> Mapping[str, object] | None:
    """The result Claude Code wrote beside a record, which it writes for neither an error nor a result its own harness handled."""
    value = record.fields.get("toolUseResult")
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else None


def message(record: Payload) -> Mapping[str, object]:
    value = record.fields.get("message")
    match value:
        case dict():
            return cast(dict[str, object], value)
        case None:
            return {}
        case _:
            raise Rejected(f"a transcript record's message should be an object, got {type(value).__name__}")


def blocks(record: Payload) -> list[Mapping[str, object]]:
    content = message(record).get("content")
    match content:
        case list():
            return [cast(dict[str, object], block) for block in cast(list[object], content) if isinstance(block, dict)]
        case _:
            return []


def result_text(content: object) -> str:
    match content:
        case str():
            return content
        case list():
            parts: list[str] = []
            for block in cast(list[object], content):
                match block:
                    case {"type": "text", "text": str() as text}:
                        parts.append(text)
                    case {"type": "image"}:
                        parts.append("[an image]")
                    case {"type": "document"}:
                        parts.append("[a document]")
                    case _:
                        parts.append(json.dumps(block, ensure_ascii=False))
            return "\n".join(parts)
        case None:
            return ""
        case _:
            return json.dumps(content, ensure_ascii=False)
