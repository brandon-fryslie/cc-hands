"""The draft and keyboard tools as the model calls them: what is read back, what is refused, what is typed, and what the audit log keeps."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import AssistantTurnStoppedMessage, LLMContextAggregatorPair, UserTurnMessageAddedMessage

from hands.core.effects import Command, Input, Key, Text, Type
from hands.core.events import Ended, Joined, StatusReported
from hands.core.session import CommandName, Membership, PromptText, SessionId
from hands.core.status import Busy, Idle, Report, Stamp, Waiting
from hands.sessions.typing import Untyped
from hands.sessions.audit import AuditLog, Record
from hands.sessions.registry import Sessions
from hands.voice.conversation import record_turns
from hands.voice.tools import Tool, audited, draft_tools, keyboard_tools, pipecat_function


def unrecorded(_: object) -> None:
    pass


async def joined(tmp: Path, record: Record = unrecorded) -> tuple[Sessions, SessionId]:
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=record)
    membership = Membership(SessionId("s1"), pid=4242, cwd=Path("/code/cc-hands"), transcript=tmp / "none.jsonl")
    await sessions.apply(Joined(membership, "startup"))
    return sessions, membership.id


async def call(tools: list[Tool], name: str, **arguments: object) -> dict[str, object]:
    [tool] = [tool for tool in tools if tool.name == name]
    return dict(await tool.body(**arguments))


async def test_a_draft_staged_amended_and_discarded_is_read_back_at_each_step(tmp_path: Path) -> None:
    sessions, id = await joined(tmp_path)
    tools = draft_tools(sessions)
    resolved = [{"heard": "auth middleware", "meant": "authMiddleware.ts"}]

    staged = await call(tools, "stage_draft", session=id, text="refactor authMiddleware.ts to use the old helper", resolutions=resolved)
    assert staged == {
        "readback": "Draft for untitled in cc-hands, reading 'auth middleware' as authMiddleware.ts: "
        "refactor authMiddleware.ts to use the old helper"
    }
    amended = await call(tools, "amend_draft", session=id, text="refactor authMiddleware.ts to use the new token helper", resolutions=resolved)
    assert amended == {"readback": "In the draft for untitled in cc-hands: 'old' is now 'new token'"}
    assert await call(tools, "discard_draft", session=id) == {"readback": "Discarded the draft for untitled in cc-hands."}
    assert await call(tools, "discard_draft", session=id) == {"readback": "There is no draft for untitled in cc-hands."}


async def test_an_ended_session_is_told_its_draft_cannot_change_and_the_draft_can_still_be_discarded(tmp_path: Path) -> None:
    sessions, id = await joined(tmp_path)
    tools = draft_tools(sessions)
    await call(tools, "stage_draft", session=id, text="run the tests", resolutions=[])
    await sessions.apply(Ended(id, "other"))
    assert await call(tools, "amend_draft", session=id, text="run the linter", resolutions=[]) == {
        "readback": "untitled in cc-hands has ended, so its draft cannot be staged, changed, or sent."
    }
    assert await call(tools, "discard_draft", session=id) == {"readback": "Discarded the draft for untitled in cc-hands."}


@pytest.mark.parametrize(
    ("text", "resolutions", "error"),
    [
        ("  \n", [], "the draft text is empty"),
        ("clear the line\x15and press enter\r", [], "control character"),
        ("fix the\tauth middleware", [], "control character"),
        ("continue the refactor \\", [], "ends with a backslash"),
        ("fine", [{"heard": "x"}], "missing field 'meant'"),
        ("fine", "none", "resolutions should be a list"),
    ],
)
async def test_arguments_that_do_not_parse_are_refused_out_loud(text: str, resolutions: object, error: str, tmp_path: Path) -> None:
    sessions, id = await joined(tmp_path)
    result = await call(draft_tools(sessions), "stage_draft", session=id, text=text, resolutions=resolutions)
    assert error in str(result["error"])


async def test_the_draft_tools_say_their_arguments_and_complete_through_a_barge_in(tmp_path: Path) -> None:
    sessions, _ = await joined(tmp_path)
    tools = draft_tools(sessions)
    assert [tool.name for tool in tools] == ["stage_draft", "amend_draft", "discard_draft", "send_draft"]
    assert tools[0].required == ("session", "text", "resolutions")
    assert tools[0].properties["resolutions"]["items"] == {"type": "object", "properties": {"heard": {"type": "string"}, "meant": {"type": "string"}}, "required": ["heard", "meant"]}
    assert all(tool.completes for tool in tools)


async def test_pipecat_is_told_a_barge_in_cancels_no_draft_tool(tmp_path: Path) -> None:
    sessions, _ = await joined(tmp_path)
    handlers = [pipecat_function(tool)._handler for tool in draft_tools(sessions)]  # pyright: ignore[reportPrivateUsage]
    assert [getattr(handler, "_pipecat_cancel_on_interruption") for handler in handlers] == [False] * 4


async def fire(aggregator: object, event: str, message: object) -> None:
    """Raise one of an aggregator's events as Pipecat does when a turn is added to the context, and wait for its handlers."""
    handler = cast(Callable[..., Awaitable[None]], getattr(aggregator, "_call_event_handler"))
    await handler(event, message)
    # Pipecat runs each async handler as a task of its own.
    await asyncio.gather(*(task for _, task in cast(set[tuple[str, asyncio.Task[None]]], getattr(aggregator, "_event_tasks"))))


async def test_a_dictation_is_traced_in_the_audit_log_from_what_the_user_said_to_the_readback(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    record = AuditLog(path, clock=lambda: datetime.now(UTC)).record
    sessions, id = await joined(tmp_path, record)
    tools = [audited(tool, record) for tool in draft_tools(sessions)]
    pair = LLMContextAggregatorPair(LLMContext())
    user, assistant = pair.user(), pair.assistant()
    record_turns(user, assistant, record)

    await fire(user, "on_user_turn_message_added", UserTurnMessageAddedMessage("tell cc-hands to run the tests", "t1"))
    await call(tools, "stage_draft", session=id, text="run the tests", resolutions=[])
    await fire(assistant, "on_assistant_turn_stopped", AssistantTurnStoppedMessage("", False, "t2"))
    await fire(assistant, "on_assistant_turn_stopped", AssistantTurnStoppedMessage("Draft for cc-hands: run the tests", False, "t2"))

    written = [json.loads(line) for line in path.read_text().splitlines()]
    trace = [(line["type"], line.get("text") or line.get("tool")) for line in written]
    assert trace == [
        ("Applied", None),
        ("Transcribed", "tell cc-hands to run the tests"),
        ("Called", "stage_draft"),
        ("Replied", "Draft for cc-hands: run the tests"),
    ]
    assert written[2]["result"] == {"readback": "Draft for untitled in cc-hands: run the tests"}
    assert [datetime.fromisoformat(line["at"]) for line in written] == sorted(datetime.fromisoformat(line["at"]) for line in written)


async def wrapped(tmp: Path, typist: Callable[[Type[Input]], None], record: Record = unrecorded) -> tuple[Sessions, SessionId]:
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=record, typist=typist)
    await sessions.apply(Joined(Membership(SessionId("s1"), 4242, Path("/code/cc-hands"), tmp / "none.jsonl", tmp / "f.sock"), "startup"))
    return sessions, SessionId("s1")


async def test_a_sent_draft_is_typed_into_its_session_once_and_is_gone(tmp_path: Path) -> None:
    typed: list[Type[Input]] = []
    sessions, id = await wrapped(tmp_path, typed.append)
    tools = draft_tools(sessions)
    await call(tools, "stage_draft", session=id, text="/compact the tests", resolutions=[])
    assert await call(tools, "send_draft", session=id) == {"readback": "Sent the draft to untitled in cc-hands."}
    assert await call(tools, "send_draft", session=id) == {"readback": "There is no draft for untitled in cc-hands."}
    [effect] = typed
    assert (effect.socket, effect.pid, effect.input) == (tmp_path / "f.sock", 4242, Text(PromptText("/compact the tests")))


async def test_a_send_fritter_could_not_type_is_said_with_the_draft_it_was(tmp_path: Path) -> None:
    def refused(_: Type[Input]) -> None:
        raise Untyped("fritter did not type into session s1: cannot write to the session")

    sessions, id = await wrapped(tmp_path, refused)
    tools = draft_tools(sessions)
    await call(tools, "stage_draft", session=id, text="run the tests", resolutions=[])
    assert await call(tools, "send_draft", session=id) == {
        "readback": "The draft for untitled in cc-hands was not sent, and is no longer staged: "
        "fritter did not type into session s1: cannot write to the session. It said: run the tests"
    }


async def test_a_session_nobody_wrapped_is_refused_by_name_and_its_draft_survives(tmp_path: Path) -> None:
    sessions, id = await joined(tmp_path)
    tools = draft_tools(sessions)
    await call(tools, "stage_draft", session=id, text="run the tests", resolutions=[])
    assert await call(tools, "send_draft", session=id) == {
        "readback": "untitled in cc-hands was not started under fritter, so hands cannot type into it. The draft is still staged."
    }
    assert await call(tools, "discard_draft", session=id) == {"readback": "Discarded the draft for untitled in cc-hands."}


async def test_a_session_waiting_at_a_permission_dialog_is_sent_nothing(tmp_path: Path) -> None:
    typed: list[Type[Input]] = []
    sessions, id = await wrapped(tmp_path, typed.append)
    tools = draft_tools(sessions)
    await call(tools, "stage_draft", session=id, text="run the tests", resolutions=[])
    # Claude Code says it waits at a dialog, whether or not hands holds the dialog's hook.
    await sessions.apply(StatusReported(id, Report(Waiting("permission prompt"), Stamp(1)), 0.0))
    assert await call(tools, "send_draft", session=id) == {
        "readback": "untitled in cc-hands is waiting at a dialog, which would take the draft as its answer. The draft is still staged."
    }
    assert typed == []


async def test_what_is_typed_is_in_the_audit_log_before_the_readback(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    record = AuditLog(path, clock=lambda: datetime.now(UTC)).record
    sessions, id = await wrapped(tmp_path, lambda _: None, record)
    tools = [audited(tool, record) for tool in draft_tools(sessions)]
    await call(tools, "stage_draft", session=id, text="run the tests", resolutions=[])
    await call(tools, "send_draft", session=id)
    written = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["type"] for line in written][-2:] == ["Typing", "Called"]
    assert written[-2]["effect"]["input"] == {"type": "Text", "prompt": "run the tests"}


async def test_a_command_with_blank_arguments_is_typed_without_them(tmp_path: Path) -> None:
    typed: list[Type[Input]] = []
    sessions, id = await wrapped(tmp_path, typed.append)
    await call(keyboard_tools(sessions), "send_command", session=id, command="compact", args="  ")
    assert [effect.input for effect in typed] == [Command(CommandName("compact"), None)]


async def test_a_command_is_typed_with_its_slash_and_read_back(tmp_path: Path) -> None:
    typed: list[Type[Input]] = []
    sessions, id = await wrapped(tmp_path, typed.append)
    result = await call(keyboard_tools(sessions), "send_command", session=id, command="/model", args="opus")
    assert result == {"readback": "Typed /model opus into untitled in cc-hands."}
    assert typed == [Type(id, tmp_path / "f.sock", 4242, Command(CommandName("model"), PromptText("opus")))]


async def test_stop_presses_escape_in_a_working_session_and_nothing_in_one_at_its_prompt(tmp_path: Path) -> None:
    typed: list[Type[Input]] = []
    sessions, id = await wrapped(tmp_path, typed.append)
    tools = keyboard_tools(sessions)
    await sessions.apply(StatusReported(id, Report(Idle(), Stamp(1)), 0.0))
    assert await call(tools, "interrupt_session", session=id) == {"readback": "untitled in cc-hands is at its prompt, so there is nothing to interrupt."}
    await sessions.apply(StatusReported(id, Report(Busy(), Stamp(2)), 1.0))
    assert await call(tools, "interrupt_session", session=id) == {"readback": "Typed Escape into untitled in cc-hands."}
    assert typed == [Type(id, tmp_path / "f.sock", 4242, Key("escape"))]


async def test_a_command_fritter_could_not_type_is_said_with_why(tmp_path: Path) -> None:
    def refused(_: Type[Input]) -> None:
        raise Untyped("fritter did not type into session s1: cannot write to the session")

    sessions, id = await wrapped(tmp_path, refused)
    assert await call(keyboard_tools(sessions), "send_command", session=id, command="compact") == {
        "readback": "/compact was not typed into untitled in cc-hands: fritter did not type into session s1: cannot write to the session."
    }


async def test_a_session_at_a_permission_dialog_is_sent_no_command(tmp_path: Path) -> None:
    typed: list[Type[Input]] = []
    sessions, id = await wrapped(tmp_path, typed.append)
    await sessions.apply(StatusReported(id, Report(Waiting("permission prompt"), Stamp(1)), 0.0))
    assert await call(keyboard_tools(sessions), "send_command", session=id, command="compact") == {
        "readback": "untitled in cc-hands is waiting at a dialog, which would take the command as its answer. Nothing was sent."
    }
    assert typed == []


@pytest.mark.parametrize(
    ("command", "args", "error"),
    [
        ("", "", "slash command's name"),
        ("compact now", "", "slash command's name"),
        ("//compact", "", "slash command's name"),
        (7, "", "command should be a string"),
        ("model", "opus\nand run the tests", "one line"),
        ("model", "opus\x1b[A", "control character"),
        ("model", "opus \\", "ends with a backslash"),
    ],
)
async def test_command_arguments_that_do_not_parse_are_refused_out_loud(command: object, args: str, error: str, tmp_path: Path) -> None:
    typed: list[Type[Input]] = []
    sessions, id = await wrapped(tmp_path, typed.append)
    result = await call(keyboard_tools(sessions), "send_command", session=id, command=command, args=args)
    assert error in str(result["error"])
    assert typed == []


async def test_the_keyboard_tools_say_their_arguments_and_complete_through_a_barge_in(tmp_path: Path) -> None:
    sessions, _ = await joined(tmp_path)
    tools = keyboard_tools(sessions)
    assert [tool.name for tool in tools] == ["send_command", "interrupt_session"]
    assert tools[0].required == ("session", "command")
    assert all(tool.completes for tool in tools)


async def test_a_command_is_in_the_audit_log_before_the_readback(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    record = AuditLog(path, clock=lambda: datetime.now(UTC)).record
    sessions, id = await wrapped(tmp_path, lambda _: None, record)
    await call([audited(tool, record) for tool in keyboard_tools(sessions)], "send_command", session=id, command="compact")
    written = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["type"] for line in written][-2:] == ["Typing", "Called"]
    assert written[-2]["effect"]["input"] == {"type": "Command", "name": "compact", "args": None}
