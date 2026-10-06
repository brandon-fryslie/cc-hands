"""What a session's JSONL transcript knows that its hooks do not carry: one record at a time, as it is written."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from hands.core.session import ESCAPES, PromptId
from hands.core.status import Stamp
from hands.core.steps import launched
from hands.core.turn import AgentId, AgentTask, Asked, Commanded, Interruption, Notified, Opening, Ref, Reported, Shelled
from hands.sessions.payload import Payload, Rejected

# Records are written without spaces, so this finds every title record cheaply.
# It also finds a nested object with that type inside another record, so a
# candidate counts only when the record's own type is the title.
_CUSTOM_TITLE_RECORD = b'"type":"custom-title"'


def session_name(transcript: Path) -> str | None:
    """The newest name the session was given, by `/rename` or by hands through a hook, or None before it has one.

    Claude Code's own ai-title is not a name: it runs to twenty-five words and the user never sees it.
    """
    try:
        raw = transcript.read_bytes()
    except FileNotFoundError:
        # Claude Code creates the transcript with its first record.
        return None
    # [LAW:no-ambient-temporal-coupling] Claude Code appends while this reads, and
    # a record is whole only once its newline is written, so the tail after the
    # last newline is never parsed.
    *complete, _unfinished = raw.split(b"\n")
    name = None
    for line in complete:
        if _CUSTOM_TITLE_RECORD in line:
            record = Payload.parse(line)
            if record.text("type") == "custom-title":
                name = record.text("customTitle")
    return name


# Only these record types carry a turn; the rest (modes, titles, snapshots, every other attachment) are skipped unparsed.
# A command the user ran, and what it printed, are written as either a user record or a system local_command one; a
# background task's notification that arrived while the session was working, as an attachment (`reported_of`); and the
# records Claude Code writes as a turn ends (`Ended`).
_TURN_RECORDS = (b'"type":"user"', b'"type":"assistant"', b'"subtype":"local_command"', b'"commandMode":"task-notification"', b'"subtype":"stop_hook_summary"', b'"subtype":"turn_duration"')
# The records Claude Code writes when a turn ends, one after the other: what its Stop hooks did, and how long the turn
# took. Written for the turn that launched a subagent in the background too, though the session stays busy until the
# subagent's report is answered. Seen on 2.1.289; the second is missing from about one stop in five.
_ENDED = ("stop_hook_summary", "turn_duration")


def turn_record(line: bytes) -> Payload | None:
    """The record this line holds if it is one a turn is made of, and None for every other line.

    Raises Rejected for a line that is not a record at all.
    """
    if not any(marker in line for marker in _TURN_RECORDS):
        return None
    record = Payload.parse(line)
    fields = record.fields
    match fields.get("type"), fields.get("subtype"), fields.get("content"):
        case "system", "local_command", str():
            return record
        case "system", "local_command", other:
            raise Rejected(f"a local_command record's content should be a string, got {type(other).__name__}")
        case "system", str() as subtype, _ if subtype in _ENDED:
            return record
        case ("user" | "assistant"), _, _:
            pass
        case "attachment", _, _:
            # Shaped here for the reason a message is, below.
            return None if reported_of(record) is None else record
        case _:
            return None
    # [LAW:parse-dont-validate] the message is shaped here, at the one place a line becomes a record, so that
    # reading a turn out of it cannot raise halfway through a record its reader has already begun to consume.
    shaped = message(record)
    # Claude Code's own stand-in for a reply that never came, not something Claude said.
    return None if shaped.get("model") == "<synthetic>" and result_text(shaped.get("content")) == _NO_RESPONSE else record


def reported_of(record: Payload) -> Reported | None:
    """The notification this record carries, if it is one Claude Code handed a session mid-turn; None for every other record.

    Claude Code writes such a notification as an attachment of the turn under way, `queued_command` with the
    notification's own markup as its prompt, where one handed to a waiting session is a user record that opens a turn.
    Seen on 2.1.288. Raises Rejected for one with no prompt that is text.
    """
    match record.fields.get("type"), record.fields.get("attachment"):
        case "attachment", {"type": "queued_command", "commandMode": "task-notification", "prompt": str() as text}:
            return Reported(ref_of(record), text, _reporter(text))
        case "attachment", {"type": "queued_command", "commandMode": "task-notification"}:
            raise Rejected("a mid-turn notification carries no prompt that is text")
        case _:
            return None


# What Claude Code writes as the reply to a turn the user stopped at a question: 19 of this machine's interrupts are followed by it.
_NO_RESPONSE = "No response requested."


# The whole of the record Claude Code writes as a turn's last when the user stops it at the keyboard: the second when a
# tool was running, the first otherwise. The same two texts in every version on this machine, 2.1.201 to 2.1.281.
# The text is the mark, not the record's interruptedMessageId: a turn stopped before Claude wrote anything names no
# message, and 162 of the 2,000 such records here carry none.
_INTERRUPTED = ("[Request interrupted by user]", "[Request interrupted by user for tool use]")


@dataclass(frozen=True)
class Printed:
    """What Claude Code printed for a command the user ran, written as a record of its own after the command's.

    Neither a request nor a step: it is the command's, the record `of` names as its parent, and joins the opening that
    record made (`printed`).
    """

    of: Ref | None
    output: str


@dataclass(frozen=True)
class Typed:
    """A command written as the words typed: a skill run in a fork of its own, and /compact, whose record Claude Code
    writes once it has run, under another prompt id and naming nothing of these words. That record is this command
    (`recorded`)."""

    command: Commanded


@dataclass(frozen=True)
class Ended:
    """Claude Code's record that the turn under way ended: last of it, unless a Stop hook sent Claude on."""


def edge_of(record: Payload, mid_tool: bool) -> Opening | Printed | Typed | Interruption | Ended | None:
    """Where this record begins a turn, cuts the one under way off, ends it, or carries what a command printed; None for
    a record in the middle of a turn."""
    if record.fields.get("type") == "system" and record.fields.get("subtype") in _ENDED:
        return Ended()
    if record.fields.get("type") == "user" and result_text(message(record).get("content")) in _INTERRUPTED:
        # Written as a user's message with no tool result in it, so it would otherwise read as the next prompt.
        return Interruption(ref_of(record))
    return _opening_of(record, mid_tool)


def prompt_of(record: Payload) -> PromptId | None:
    """The prompt_id of the turn a record belongs to, which Claude Code writes on the user's side of it."""
    value = record.fields.get("promptId")
    return PromptId(value) if isinstance(value, str) else None


def written_of(record: Payload) -> Stamp | None:
    """When Claude Code wrote the record, in the epoch milliseconds it stamps a status with; None for a record with no time.

    Raises Rejected for a time that cannot be read.
    """
    match record.fields.get("timestamp"):
        case None:
            return None
        case str() as stamp:
            try:
                written = datetime.fromisoformat(stamp)
            except ValueError as error:
                raise Rejected(f"a transcript record's timestamp {stamp!r} is not a time: {error}") from error
            if written.tzinfo is None:
                # Read as this machine's local time, it would be hours off the epoch Claude Code stamps a status in.
                raise Rejected(f"a transcript record's timestamp {stamp!r} names no zone")
            return Stamp(round(written.timestamp() * 1000))
        case other:
            raise Rejected(f"a transcript record's timestamp should be a string, got {type(other).__name__}")


def _opening_of(record: Payload, mid_tool: bool) -> Opening | Printed | Typed | None:
    """What opens a turn: a prompt or a notification Claude Code handed a session that was not waiting on a tool.

    `mid_tool` says whether the record before this one was a tool call or its result.
    """
    if mid_tool:
        # Sent while a tool ran: Claude Code folds it into the turn already under way, whose Stop has not come.
        return None
    fields = record.fields
    ref = ref_of(record)
    if fields.get("type") == "system" and fields.get("subtype") == "local_command":
        # A local_command record holds nothing but what the user ran or what it printed (`turn_record`).
        return _ran(ref, cast(str, fields["content"]), ref_of(record, "parentUuid"))
    # Meta records (skill bodies, command caveats) and compaction's summary are Claude Code's own, not a new request.
    if fields.get("type") != "user" or fields.get("isMeta") is True or fields.get("isCompactSummary") is True:
        return None
    parts = blocks(record)
    match message(record).get("content"):
        case str() as text:
            ran = _ran(ref, text, ref_of(record, "parentUuid")) or _typed(record, text)
            if ran is not None:
                return ran
        case list() if parts and not any(block.get("type") == "tool_result" for block in parts):
            # A prompt with an image or a document attached, or one sent through the SDK. Anything but a tool result,
            # rather than a list of the block kinds known today: a kind added tomorrow would otherwise stop opening the
            # turn it opens, which hands the whole turn to an older opening, where a block nobody named costs the
            # summariser some JSON inside `budget.opening` and nothing else.
            text = result_text(parts)
        case _:
            return None
    match fields.get("origin"):
        case {"kind": "task-notification"}:
            return Notified(ref, text, _reporter(text))
        case _:
            return Asked(ref, text)


def launched_of(record: Payload) -> AgentId | None:
    """The subagent this record's result says Claude started in the background; None for every other record."""
    structured = structured_result(record)
    return None if structured is None else launched(structured)


def notified_of(record: Payload) -> AgentId | None:
    """The task whose notification this record carries, at the prompt or mid-turn; None for every other record."""
    # [LAW:one-source-of-truth] read as the turns read it: mid-turn by `reported_of`, at the prompt by the opening.
    match reported_of(record) or _opening_of(record, mid_tool=False):
        case Reported(text=text) | Notified(text=text):
            task = _TASK.search(text)
            return None if task is None else AgentId(task.group(1))
        case _:
            return None


def _reporter(notification: str) -> AgentTask | None:
    """The subagent a notification is the report of; None for a notification from a background command or a monitor."""
    task, agent = _TASK.search(notification), _AGENT.search(notification)
    return None if task is None or agent is None else AgentTask(AgentId(task.group(1)), agent.group(1))


# The task a notification is about, which Claude Code names only inside the notification's own markup, and, where the
# task is a subagent's, the job it was given: `Agent "/code-review high 124" finished`, or `... was stopped by Claude`.
_TASK = re.compile(r"<task-id>([^<\s]+)</task-id>")
_AGENT = re.compile(r'<summary>Agent "([^<]+)" [^"<]*</summary>')

# The markup Claude Code writes, at the start of a record of the user's side, for what the user ran rather than wrote:
# a slash command (its name first when Claude Code carries it out, its message first when it hands Claude a skill),
# a `!` command, and what either printed. Seen on 2.1.226 to 2.1.286.
_RAN = re.compile(r"\s*<(command-name|command-message|bash-input|local-command-stdout|local-command-stderr|bash-stdout|bash-stderr)>")
# Each output tag, and how what it holds is told: what went to stderr is marked, so a command that failed is not told as
# one that printed its answer.
_OUTPUTS = (("local-command-stdout", ""), ("bash-stdout", ""), ("local-command-stderr", "stderr: "), ("bash-stderr", "stderr: "))
# A command written as the words typed: /compact ahead of the compaction it asks for, and a skill run in a fork of its own.
_TYPED = re.compile(r"(/[A-Za-z][\w:.-]*)(?:\s+(.*))?", re.DOTALL)


def _ran(ref: Ref | None, text: str, parent: Ref | None) -> Commanded | Shelled | Printed | None:
    """What the user ran, read off the markup Claude Code wrote around it; None for text that does not open with it.

    [LAW:types-are-the-program] each kind is read as itself, so neither a command's markup nor its output's
    terminal colours ever reach the narration as something the user asked.
    """
    match _RAN.match(text):
        case None:
            return None
        case found if found.group(1) in ("command-name", "command-message"):
            return Commanded(ref, _tagged(text, "command-name"), _tagged(text, "command-args"))
        case found if found.group(1) == "bash-input":
            return Shelled(ref, _tagged(text, "bash-input"))
        case _:
            return Printed(parent, "\n".join(f"{mark}{output}" for tag, mark in _OUTPUTS if (output := _tagged(text, tag))))


def _typed(record: Payload, text: str) -> Typed | None:
    """A command Claude Code wrote as the words typed; None for anything Claude was sent, which may open with a slash too.

    Claude Code marks what it sends Claude with the prompt's source (`typed`, `queued`): ` /help reply pong` carries one,
    and Claude answered it. The words a command was typed as carry none, and every prompt of a main session carries one
    from 2.1.197 to 2.1.286.
    """
    match record.fields.get("promptSource"), _TYPED.fullmatch(text):
        case None, typed if typed is not None:
            return Typed(Commanded(ref_of(record), typed.group(1), (typed.group(2) or "").strip()))
        case _:
            return None


def _tagged(text: str, tag: str) -> str:
    """What one of Claude Code's tags holds, with what a terminal is told dropped; empty where the record has none."""
    found = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return "" if found is None else ESCAPES.sub("", found.group(1)).strip()


def holds_a_tool(record: Payload) -> bool:
    """Whether this record is a tool call or a tool's result, which is what makes the record after it mid-turn."""
    return any(block.get("type") in ("tool_use", "tool_result") for block in blocks(record))


def ref_of(record: Payload, key: str = "uuid") -> Ref | None:
    """The record a record names under `key`: itself under its uuid, the record before it under its parentUuid."""
    value = record.fields.get(key)
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
