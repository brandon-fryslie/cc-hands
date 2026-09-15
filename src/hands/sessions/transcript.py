"""What a session's JSONL transcript knows that its hooks do not carry."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from hands.core.turn import Said, Step, Turn, Used
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


def read_turn(transcript: Path) -> Turn | None:
    """The newest turn: the last prompt the user submitted, and everything Claude said and used after it. None before any prompt."""
    *complete, _unfinished = transcript.read_bytes().split(b"\n")
    records = [Payload.parse(line) for line in complete if any(marker in line for marker in _TURN_RECORDS)]
    turn = [record for record in records if record.fields.get("type") in ("user", "assistant") and record.fields.get("isSidechain") is not True]
    prompts = [index for index, record in enumerate(turn) if _prompt(record) is not None]
    if not prompts:
        return None
    start = prompts[-1]
    return Turn(prompt=cast(str, _prompt(turn[start])), steps=tuple(_steps(turn[start + 1 :])))


def _prompt(record: Payload) -> str | None:
    """A prompt is a user record whose content is a string and that Claude Code did not inject; tool results are lists."""
    match (record.fields.get("type"), _message(record).get("content"), record.fields.get("isMeta")):
        case ("user", str() as text, None | False):
            return text
        case _:
            return None


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
                    case _:
                        parts.append(json.dumps(block, ensure_ascii=False))
            return "\n".join(parts)
        case None:
            return ""
        case _:
            return json.dumps(content, ensure_ascii=False)
