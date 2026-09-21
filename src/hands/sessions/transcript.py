"""What a session's JSONL transcript knows that its hooks do not carry."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from hands.core.turn import Asked, Notified, Opening, Said, Step, Turn, Used
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


@dataclass(frozen=True)
class Told:
    """How much of a turn a session has been told: which record opened it, how many of its recorded steps were told,
    and a closing reply told from the Stop hook before Claude Code had written the record of it."""

    opening: str | None
    steps: int
    closing: str | None


# Before a session has been told anything.
UNTOLD = Told(opening=None, steps=0, closing=None)


@dataclass(frozen=True)
class Reading:
    """The part of the newest turn a session has not been told, and what it will have been told once that part is."""

    turn: Turn
    told: Told


def read_turn(transcript: Path, told: Told, closing: str | None) -> Reading | None:
    """What last opened a turn, and the steps after it that `told` does not cover. None before anything has opened one.

    `closing` is the reply the Stop hook carries. Measured over twelve live turns, Claude Code writes that reply's own
    record 46 to 77 ms after the hook fires, so the hook's copy stands in until the record lands and gives way when it does.
    """
    *complete, _unfinished = transcript.read_bytes().split(b"\n")
    records = [Payload.parse(line) for line in complete if any(marker in line for marker in _TURN_RECORDS)]
    turn = [record for record in records if record.fields.get("type") in ("user", "assistant") and record.fields.get("isSidechain") is not True]
    openings = [(index, opening) for index, record in enumerate(turn) if (opening := _opening(record, turn[index - 1] if index else None)) is not None]
    if not openings:
        return None
    start, opening = openings[-1]
    # A Stop another hook blocked lets the same turn go on to a later Stop, which tells the steps the first one did not.
    recorded = _steps(turn[start + 1 :])
    began = _uuid(turn[start])
    heard = told.steps if told.opening == began else 0
    # Claude Code only ever appends, so the record of a stand-in that has since been written is the first step after what
    # was heard; counting it heard too is how the stand-in gives way to its record without the reply being told twice.
    heard += 1 if told.closing is not None and recorded[heard : heard + 1] == [Said(told.closing)] else 0
    stand_in = _stand_in(closing, recorded)
    steps = recorded[heard:] if stand_in is None else [*recorded[heard:], Said(stand_in)]
    return Reading(Turn(opening=opening, steps=tuple(steps)), Told(opening=began, steps=len(recorded), closing=stand_in))


def _stand_in(closing: str | None, recorded: list[Step]) -> str | None:
    """The hook's copy of the closing reply, while the transcript holds no record of it; None once it does.

    [LAW:one-source-of-truth] the transcript is the record of what Claude said, and the copy stands in only until it is written.
    """
    return None if closing is None or closing.strip() in ("", _last_said(recorded)) else closing


def _last_said(steps: list[Step]) -> str | None:
    """The last text Claude wrote among these steps, as the closing reply the Stop hook carries would read."""
    return next((step.text.strip() for step in reversed(steps) if isinstance(step, Said) and step.text.strip()), None)


def _uuid(record: Payload) -> str | None:
    value = record.fields.get("uuid")
    return value if isinstance(value, str) else None


def _opening(record: Payload, previous: Payload | None) -> Opening | None:
    """What opens a turn: a prompt or a notification Claude Code handed a session that was not waiting on a tool."""
    if previous is not None and any(block.get("type") in ("tool_use", "tool_result") for block in _blocks(previous)):
        # Sent while a tool ran: Claude Code folds it into the turn already under way, whose Stop has not come.
        return None
    fields = record.fields
    # Meta records (skill bodies, command caveats) and compaction's summary are Claude Code's own, not a new request.
    if fields.get("type") != "user" or fields.get("isMeta") is True or fields.get("isCompactSummary") is True:
        return None
    blocks = _blocks(record)
    match _message(record).get("content"):
        case str() as text:
            pass
        case list() if blocks and not any(block.get("type") == "tool_result" for block in blocks):
            # A prompt with an image or a document attached, or one sent through the SDK.
            text = _result_text(blocks)
        case _:
            return None
    match fields.get("origin"):
        case {"kind": "task-notification"}:
            return Notified(text)
        case _:
            return Asked(text)


def _steps(records: list[Payload]) -> list[Step]:
    # A call's result arrives in a later record; the step keeps the call's place in the order.
    steps: list[Step | str] = []
    calls: dict[str, tuple[int, str, Mapping[str, object]]] = {}
    results: dict[str, tuple[str, bool]] = {}
    for record in records:
        for block in _blocks(record):
            match block:
                case {"type": "text", "text": str() as text} if record.fields.get("type") == "assistant" and text.strip():
                    steps.append(Said(text))
                case {"type": "tool_use", "id": str() as id, "name": str() as name, "input": dict()}:
                    calls[id] = (len(steps), name, cast(dict[str, object], block["input"]))
                    steps.append(id)
                case {"type": "tool_result", "tool_use_id": str() as id}:
                    results[id] = (_result_text(block.get("content")), block.get("is_error") is True)
                case _:
                    pass
    return [step if not isinstance(step, str) else _used(calls[step][1], calls[step][2], results.get(step)) for step in steps]


def _used(tool: str, input: Mapping[str, object], result: tuple[str, bool] | None) -> Used:
    purpose = input.get("description")
    shown = {key: value for key, value in input.items() if key != "description"}
    match shown:
        case {"command": str() as command} if len(shown) == 1:
            text = command
        case _:
            text = json.dumps(shown, ensure_ascii=False)
    # A call with no result was interrupted, or the turn stopped before its result was written.
    output, failed = result if result is not None else ("(no result)", False)
    return Used(tool=tool, purpose=purpose if isinstance(purpose, str) else None, input=text, result=output, failed=failed)


def _message(record: Payload) -> Mapping[str, object]:
    message = record.fields.get("message")
    match message:
        case dict():
            return cast(dict[str, object], message)
        case None:
            return {}
        case _:
            raise Rejected(f"a transcript record's message should be an object, got {type(message).__name__}")


def _blocks(record: Payload) -> list[Mapping[str, object]]:
    content = _message(record).get("content")
    match content:
        case list():
            return [cast(dict[str, object], block) for block in cast(list[object], content) if isinstance(block, dict)]
        case _:
            return []


def _result_text(content: object) -> str:
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
