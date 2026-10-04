"""The draft and keyboard tools as the model calls them: what is read back, what is refused, what is typed, and what the audit log keeps."""

import asyncio
import io
import json
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from loguru import logger
from pipecat.frames.frames import Frame, FunctionCallResultProperties, TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import AssistantTurnStoppedMessage, LLMContextAggregatorPair, UserTurnMessageAddedMessage
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import FunctionCallParams

from hands.core.effects import Command, Input, Key, Text, Type
from hands.core.events import Ended, Joined, StatusReported
from hands.core.session import CommandName, Membership, PromptText, SessionId
from hands.core.status import Busy, Idle, Report, Stamp, Waiting
from hands.sessions.typing import Untyped
from hands.sessions.audit import AuditLog, Entry, Record, Typing, TypingFailed, tail
from hands.sessions.wide import unit
from hands.sessions.registry import Sessions
from hands.voice.conversation import record_turns
from hands.voice.tools import audited, cued, draft_tools, keyboard_tools, pipecat_function, Replies, Result, Tool, tool


def unrecorded(_: object) -> None:
    pass


class Lines(FrameProcessor):
    """The processor standing ahead of the speaker, keeping what hands hands it."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self.frames: list[Frame] = []

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        self.frames.append(frame)

    @property
    def said(self) -> list[tuple[str, bool]]:
        return [(frame.text, frame.append_to_context) for frame in self.frames if isinstance(frame, TTSSpeakFrame)]


async def joined(tmp: Path, record: Record = unrecorded) -> tuple[Sessions, SessionId]:
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=record)
    membership = Membership(SessionId("s1"), pid=4242, cwd=Path("/code/cc-hands"), transcript=tmp / "none.jsonl")
    await sessions.apply(Joined(membership, "startup"))
    return sessions, membership.id


async def call(tools: list[Tool], name: str, **arguments: object) -> dict[str, object]:
    """A call to the tool named, inside a unit of work, as every call the daemon makes is."""
    [tool] = [tool for tool in tools if tool.name == name]
    with unit("tool.run", unrecorded):
        return dict(await tool.body(**arguments))


def kind(line: dict[str, Any]) -> str:
    """What a line of the log is: a wide event's own name, and any other line's type."""
    return line["event"] if line["type"] == "WideEvent" else line["type"]


async def test_a_draft_staged_amended_and_discarded_is_read_back_at_each_step(tmp_path: Path) -> None:
    sessions, id = await joined(tmp_path)
    tools = draft_tools(sessions)
    resolved = [{"heard": "auth middleware", "meant": "authMiddleware.ts"}]

    staged = await call(tools, "stage_draft", session=id, text="refactor authMiddleware.ts to use the old helper", resolutions=resolved)
    assert staged == {
        "says": "Draft for cc-hands, reading 'auth middleware' as auth Middleware dot ts: refactor auth Middleware dot ts to use the old helper"
    }
    amended = await call(tools, "amend_draft", session=id, text="refactor authMiddleware.ts to use the new token helper", resolutions=resolved)
    assert amended == {"says": "In the draft for cc-hands: 'old' is now 'new token'"}
    assert await call(tools, "discard_draft", session=id) == {"readback": "Discarded the draft for cc-hands."}
    assert await call(tools, "discard_draft", session=id) == {"readback": "There is no draft for cc-hands."}


async def test_an_ended_session_is_told_its_draft_cannot_change_and_the_draft_can_still_be_discarded(tmp_path: Path) -> None:
    sessions, id = await joined(tmp_path)
    tools = draft_tools(sessions)
    await call(tools, "stage_draft", session=id, text="run the tests", resolutions=[])
    await sessions.apply(Ended(id, "other"))
    # Nothing changed, so it is the model's to answer.
    assert await call(tools, "amend_draft", session=id, text="run the linter", resolutions=[]) == {
        "error": "cc-hands has ended, so its draft cannot be staged, changed, or sent."
    }
    assert await call(tools, "discard_draft", session=id) == {"readback": "Discarded the draft for cc-hands."}


@pytest.mark.parametrize(
    ("text", "resolutions", "error"),
    [
        ("  \n", [], "the draft text is empty"),
        ("clear the line\x15and press enter\r", [], "control character"),
        ("fix the\tauth middleware", [], "control character"),
        ("continue the refactor \\", [], "ends with a backslash"),
        ("fine", [{"heard": "x"}], "resolutions[0] has no meant"),
        ("fine", "none", "resolutions should be a list"),
    ],
)
async def test_arguments_that_do_not_parse_are_refused_to_the_model(text: str, resolutions: object, error: str, tmp_path: Path) -> None:
    sessions, id = await joined(tmp_path)
    result = await call(draft_tools(sessions), "stage_draft", session=id, text=text, resolutions=resolutions)
    assert error in str(result["error"])


async def test_a_draft_for_no_session_or_with_none_staged_is_refused_to_the_model_which_can_retry(tmp_path: Path) -> None:
    sessions, id = await joined(tmp_path)
    [stage, amend, *_] = draft_tools(sessions)
    assert await call([stage], "stage_draft", session="cc-hands", text="run the tests", resolutions=[]) == {"error": "There is no session cc-hands."}
    assert await call([amend], "amend_draft", session=id, text="run the tests", resolutions=[]) == {"error": "There is no draft for cc-hands."}


async def test_the_draft_tools_say_their_arguments_and_complete_through_a_barge_in(tmp_path: Path) -> None:
    sessions, _ = await joined(tmp_path)
    tools = draft_tools(sessions)
    assert [tool.name for tool in tools] == ["stage_draft", "amend_draft", "discard_draft", "send_draft"]
    assert tools[0].required == ("session", "text", "resolutions")
    assert tools[0].properties["resolutions"]["items"] == {"type": "object", "properties": {"heard": {"type": "string"}, "meant": {"type": "string"}}, "required": ["heard", "meant"]}
    assert all(tool.completes for tool in tools)
    assert [tool.then for tool in tools] == ["silence", "silence", "reply", "reply"]


async def reply(lines: Lines, *calls: tuple[Tool, Mapping[str, object]]) -> list[bool | None]:
    """One reply's calls as Pipecat runs them, answered in the order given: whether each result asks Pipecat to run the model."""
    replies = Replies()
    replies.started([f"c{index}" for index, _ in enumerate(calls)])
    told: list[bool | None] = []

    async def result_callback(_: object, *, properties: FunctionCallResultProperties | None = None) -> None:
        told.append(None if properties is None else properties.run_llm)

    for index, (tool, arguments) in enumerate(calls):
        handler = pipecat_function(tool, lines, replies)._handler  # pyright: ignore[reportPrivateUsage]
        assert handler is not None
        await handler(cast(FunctionCallParams, SimpleNamespace(tool_call_id=f"c{index}", arguments=arguments, result_callback=result_callback)))
    return told


async def test_pipecat_says_a_staged_draft_as_written_and_runs_no_model_after_it(tmp_path: Path) -> None:
    sessions, id = await joined(tmp_path)
    [stage, *_] = draft_tools(sessions)
    lines = Lines()
    assert await reply(lines, (stage, {"session": id, "text": "run the tests", "resolutions": []})) == [False]
    # Said by hands as written, straight to the speaker: no model is asked to say it, so none rewords it.
    assert lines.said == [("Draft for cc-hands: run the tests", False)]


async def test_pipecat_runs_the_model_once_a_reply_is_in_when_any_call_in_it_was_refused_or_asks_a_reply(tmp_path: Path) -> None:
    sessions, id = await joined(tmp_path)
    [stage, _, discard, _] = draft_tools(sessions)
    lines = Lines()
    staged: tuple[Tool, Mapping[str, object]] = (stage, {"session": id, "text": "run the tests", "resolutions": []})
    refused: tuple[Tool, Mapping[str, object]] = (stage, {"session": "cc-hands", "text": "run the tests", "resolutions": []})
    # The silent call finishing last does not decide for the reply: the refusal before it is the model's to answer.
    assert await reply(lines, refused, staged) == [False, True]
    assert await reply(lines, staged, refused) == [False, True]
    assert await reply(lines, (discard, {"session": id}), staged) == [False, True]
    # Each staged draft is said by hands; the refusal is not.
    assert [text for text, _ in lines.said] == [
        "Draft for cc-hands: run the tests",
        "New draft for cc-hands, replacing the last one: run the tests",
        "Draft for cc-hands: run the tests",
    ]


async def test_pipecat_is_told_a_barge_in_cancels_no_draft_tool(tmp_path: Path) -> None:
    sessions, _ = await joined(tmp_path)
    handlers = [pipecat_function(tool, Lines(), Replies())._handler for tool in draft_tools(sessions)]  # pyright: ignore[reportPrivateUsage]
    assert [getattr(handler, "_pipecat_cancel_on_interruption") for handler in handlers] == [False] * 4


async def fire(aggregator: object, event: str, message: object) -> None:
    """Raise one of an aggregator's events as Pipecat does when a turn is added to the context, and wait for its handlers."""
    handler = cast(Callable[..., Awaitable[None]], getattr(aggregator, "_call_event_handler"))
    await handler(event, message)
    # Pipecat runs each async handler as a task of its own.
    await asyncio.gather(*(task for _, task in cast(set[tuple[str, asyncio.Task[None]]], getattr(aggregator, "_event_tasks"))))


async def test_a_dictation_is_traced_in_the_audit_log_from_what_the_user_said_to_the_readback_hands_said(tmp_path: Path) -> None:
    path = tmp_path / "audit"
    record = AuditLog(path, clock=lambda: datetime.now(UTC)).record
    sessions, id = await joined(tmp_path, record)
    tools = [audited(tool, record) for tool in draft_tools(sessions)]
    pair = LLMContextAggregatorPair(LLMContext())
    user, assistant = pair.user(), pair.assistant()
    record_turns(user, assistant, record)

    await fire(user, "on_user_turn_message_added", UserTurnMessageAddedMessage("tell cc-hands to run the tests", "t1"))
    await call(tools, "stage_draft", session=id, text="run the tests", resolutions=[])
    # The call was the whole reply: the model said nothing of its own.
    await fire(assistant, "on_assistant_turn_stopped", AssistantTurnStoppedMessage("", False, "t2"))

    written = [json.loads(line) for line in tail(path, 1000)[0]]
    trace = [(kind(line), line.get("text") or line.get("facts", {}).get("tool")) for line in written]
    assert trace == [
        ("applied", None),
        ("Transcribed", "tell cc-hands to run the tests"),
        ("tool.run", "stage_draft"),
    ]
    assert written[2]["facts"]["called"]["result"] == {"says": "Draft for cc-hands: run the tests"}
    assert [datetime.fromisoformat(line["at"]) for line in written] == sorted(datetime.fromisoformat(line["at"]) for line in written)


async def test_what_was_heard_is_shown_in_the_terminal_in_the_words_heard() -> None:
    from hands.daemon.cli import TERMINAL_LEVELS

    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(message.record["message"]), filter=TERMINAL_LEVELS)
    pair = LLMContextAggregatorPair(LLMContext())
    user, assistant = pair.user(), pair.assistant()
    record_turns(user, assistant, unrecorded)
    try:
        await fire(user, "on_user_turn_message_added", UserTurnMessageAddedMessage("Don't say \"stop\",\n  can you hear me?", "t1"))
    finally:
        logger.remove(sink)

    assert lines == ['heard: Don\'t say "stop", can you hear me?']


async def test_a_control_in_what_was_heard_reaches_the_terminal_as_its_escape_and_never_as_a_control() -> None:
    from hands.daemon.cli import to_terminal

    terminal = io.StringIO()
    sink = to_terminal(terminal)
    pair = LLMContextAggregatorPair(LLMContext())
    user, assistant = pair.user(), pair.assistant()
    record_turns(user, assistant, unrecorded)
    try:
        await fire(user, "on_user_turn_message_added", UserTurnMessageAddedMessage("\x1b[2Kgone\x07 back\x08\x08 {x}\x9b2J\x7f\u202eo", "t1"))
    finally:
        logger.remove(sink)

    written = terminal.getvalue()
    assert written.endswith(" - heard: \\u001b[2Kgone\\u0007 back\\u0008\\u0008 {x}\\u009b2J\\u007f\\u202eo\n")
    assert not {"\x1b", "\x07", "\x08", "\x9b", "\x7f", "\u202e"} & set(written)


async def wrapped(tmp: Path, typist: Callable[[Type[Input]], None], record: Record = unrecorded) -> tuple[Sessions, SessionId]:
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=record, typist=typist)
    await sessions.apply(Joined(Membership(SessionId("s1"), 4242, Path("/code/cc-hands"), tmp / "none.jsonl", tmp / "f.sock"), "startup"))
    return sessions, SessionId("s1")


async def test_a_sent_draft_is_typed_into_its_session_once_and_is_gone(tmp_path: Path) -> None:
    typed: list[Type[Input]] = []
    sessions, id = await wrapped(tmp_path, typed.append)
    tools = draft_tools(sessions)
    await call(tools, "stage_draft", session=id, text="/compact the tests", resolutions=[])
    assert await call(tools, "send_draft", session=id) == {"readback": "Sent the draft to cc-hands."}
    assert await call(tools, "send_draft", session=id) == {"readback": "There is no draft for cc-hands."}
    [effect] = typed
    assert (effect.socket, effect.pid, effect.input) == (tmp_path / "f.sock", 4242, Text(PromptText("/compact the tests")))


async def test_a_send_fritter_could_not_type_is_said_with_the_draft_it_was(tmp_path: Path) -> None:
    def refused(_: Type[Input]) -> None:
        raise Untyped("fritter did not type into session s1: cannot write to the session")

    recorded: list[Entry] = []
    sessions, id = await wrapped(tmp_path, refused, recorded.append)
    tools = draft_tools(sessions)
    await call(tools, "stage_draft", session=id, text="run the tests", resolutions=[])
    assert await call(tools, "send_draft", session=id) == {
        "readback": "The draft for cc-hands was not sent, and is no longer staged: "
        "fritter did not type into session s1: cannot write to the session. It said: run the tests"
    }
    # The log pairs the send with its failure, which is how recall tells a send that never arrived.
    [typing] = [entry for entry in recorded if isinstance(entry, Typing)]
    assert [entry for entry in recorded if isinstance(entry, TypingFailed)] == [TypingFailed(typing.effect, "fritter did not type into session s1: cannot write to the session")]


async def test_a_session_nobody_wrapped_is_refused_by_name_and_its_draft_survives(tmp_path: Path) -> None:
    sessions, id = await joined(tmp_path)
    tools = draft_tools(sessions)
    await call(tools, "stage_draft", session=id, text="run the tests", resolutions=[])
    assert await call(tools, "send_draft", session=id) == {
        "readback": "cc-hands was not started under fritter, so hands cannot type into it. The draft is still staged."
    }
    assert await call(tools, "discard_draft", session=id) == {"readback": "Discarded the draft for cc-hands."}


async def test_a_session_waiting_at_a_permission_dialog_is_sent_nothing(tmp_path: Path) -> None:
    typed: list[Type[Input]] = []
    sessions, id = await wrapped(tmp_path, typed.append)
    tools = draft_tools(sessions)
    await call(tools, "stage_draft", session=id, text="run the tests", resolutions=[])
    # Claude Code says it waits at a dialog, whether or not hands holds the dialog's hook.
    await sessions.apply(StatusReported(id, Report(Waiting("permission prompt"), Stamp(1)), 0.0))
    assert await call(tools, "send_draft", session=id) == {
        "readback": "cc-hands is waiting at a dialog, which would take the draft as its answer. The draft is still staged."
    }
    assert typed == []


async def test_what_is_typed_is_in_the_audit_log_before_the_readback(tmp_path: Path) -> None:
    path = tmp_path / "audit"
    record = AuditLog(path, clock=lambda: datetime.now(UTC)).record
    sessions, id = await wrapped(tmp_path, lambda _: None, record)
    tools = [audited(tool, record) for tool in draft_tools(sessions)]
    await call(tools, "stage_draft", session=id, text="run the tests", resolutions=[])
    await call(tools, "send_draft", session=id)
    written = [json.loads(line) for line in tail(path, 1000)[0]]
    assert [kind(line) for line in written][-2:] == ["Typing", "tool.run"]
    assert written[-2]["effect"]["input"] == {"type": "Text", "prompt": "run the tests"}
    # What was typed is joined to the call that typed it by the call's span.
    assert written[-2]["span"]["span_id"] == written[-1]["span_id"] and written[-1]["facts"]["tool"] == "send_draft"


async def test_a_command_with_blank_arguments_is_typed_without_them(tmp_path: Path) -> None:
    typed: list[Type[Input]] = []
    sessions, id = await wrapped(tmp_path, typed.append)
    await call(keyboard_tools(sessions), "send_command", session=id, command="compact", args="  ")
    assert [effect.input for effect in typed] == [Command(CommandName("compact"), None)]


async def test_a_command_is_typed_with_its_slash_and_read_back(tmp_path: Path) -> None:
    typed: list[Type[Input]] = []
    sessions, id = await wrapped(tmp_path, typed.append)
    result = await call(keyboard_tools(sessions), "send_command", session=id, command="/model", args="opus")
    assert result == {"readback": "Typed /model opus into cc-hands."}
    assert typed == [Type(id, tmp_path / "f.sock", 4242, Command(CommandName("model"), PromptText("opus")))]


async def test_stop_presses_escape_in_a_working_session_and_nothing_in_one_at_its_prompt(tmp_path: Path) -> None:
    typed: list[Type[Input]] = []
    sessions, id = await wrapped(tmp_path, typed.append)
    tools = keyboard_tools(sessions)
    await sessions.apply(StatusReported(id, Report(Idle(), Stamp(1)), 0.0))
    assert await call(tools, "interrupt_session", session=id) == {"readback": "cc-hands is at its prompt, so there is nothing to interrupt."}
    await sessions.apply(StatusReported(id, Report(Busy(), Stamp(2)), 1.0))
    assert await call(tools, "interrupt_session", session=id) == {"readback": "Typed Escape into cc-hands."}
    assert typed == [Type(id, tmp_path / "f.sock", 4242, Key("escape"))]


async def test_a_command_fritter_could_not_type_is_said_with_why(tmp_path: Path) -> None:
    def refused(_: Type[Input]) -> None:
        raise Untyped("fritter did not type into session s1: cannot write to the session")

    sessions, id = await wrapped(tmp_path, refused)
    assert await call(keyboard_tools(sessions), "send_command", session=id, command="compact") == {
        "readback": "/compact was not typed into cc-hands: fritter did not type into session s1: cannot write to the session."
    }


async def test_a_session_at_a_permission_dialog_is_sent_no_command(tmp_path: Path) -> None:
    typed: list[Type[Input]] = []
    sessions, id = await wrapped(tmp_path, typed.append)
    await sessions.apply(StatusReported(id, Report(Waiting("permission prompt"), Stamp(1)), 0.0))
    assert await call(keyboard_tools(sessions), "send_command", session=id, command="compact") == {
        "readback": "cc-hands is waiting at a dialog, which would take the command as its answer. Nothing was sent."
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
    path = tmp_path / "audit"
    record = AuditLog(path, clock=lambda: datetime.now(UTC)).record
    sessions, id = await wrapped(tmp_path, lambda _: None, record)
    await call([audited(tool, record) for tool in keyboard_tools(sessions)], "send_command", session=id, command="compact")
    written = [json.loads(line) for line in tail(path, 1000)[0]]
    assert [kind(line) for line in written][-2:] == ["Typing", "tool.run"]
    assert written[-2]["effect"]["input"] == {"type": "Command", "name": "compact", "args": None}


async def test_a_cued_tool_says_hands_is_acting_as_it_is_called_and_answers_as_before() -> None:
    happened: list[str] = []

    async def look(where: str) -> Result:
        """Look somewhere.

        Args:
            where: where to look.
        """
        happened.append(f"looked {where}")
        return {"saw": where}

    looking = cued(tool(look), lambda: happened.append("acting"))
    assert await looking.body(where="up") == {"saw": "up"}
    assert happened == ["acting", "looked up"]
    assert looking.name == "look"
