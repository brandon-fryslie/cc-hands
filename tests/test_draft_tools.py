"""The draft tools as the model calls them: what is read back, what is refused, and what the audit log keeps."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import AssistantTurnStoppedMessage, LLMContextAggregatorPair, UserTurnMessageAddedMessage
from pipecat.services.llm_service import FunctionCallParams

from hands.core.events import Ended, Joined
from hands.core.session import Membership, SessionId
from hands.sessions.audit import AuditLog, Record
from hands.sessions.registry import Sessions
from hands.voice.conversation import record_turns
from hands.voice.tools import Tool, audited, draft_tools


def unrecorded(_: object) -> None:
    pass


async def joined(tmp: Path, record: Record = unrecorded) -> tuple[Sessions, SessionId]:
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=record)
    membership = Membership(SessionId("s1"), pid=4242, cwd=Path("/code/cc-hands"), transcript=tmp / "none.jsonl")
    await sessions.apply(Joined(membership, "startup"))
    return sessions, membership.id


async def call(tools: list[Tool], name: str, **arguments: object) -> dict[str, object]:
    results: list[dict[str, object]] = []

    async def capture(result: dict[str, object], **_: object) -> None:
        results.append(result)

    [tool] = [tool for tool in tools if DirectFunctionWrapper(tool).name == name]
    params = FunctionCallParams(
        function_name=name, tool_call_id="call-1", arguments=arguments,
        llm=cast(Any, None), pipeline_worker=cast(Any, None), context=cast(Any, None), result_callback=capture,
    )
    await DirectFunctionWrapper(tool).invoke(arguments, params)
    [result] = results
    return result


async def test_a_draft_staged_amended_and_discarded_is_read_back_at_each_step(tmp_path: Path) -> None:
    sessions, id = await joined(tmp_path)
    tools = draft_tools(sessions)
    resolved = [{"heard": "auth middleware", "meant": "authMiddleware.ts"}]

    staged = await call(tools, "stage_draft", session=id, text="refactor authMiddleware.ts to use the old helper", resolutions=resolved)
    assert staged == {
        "readback": "Draft for untitled, in cc-hands, reading 'auth middleware' as authMiddleware.ts: "
        "refactor authMiddleware.ts to use the old helper"
    }
    amended = await call(tools, "amend_draft", session=id, text="refactor authMiddleware.ts to use the new token helper", resolutions=resolved)
    assert amended == {"readback": "In the draft for untitled, in cc-hands: 'old' is now 'new token'"}
    assert await call(tools, "discard_draft", session=id) == {"readback": "Discarded the draft for untitled, in cc-hands."}
    assert await call(tools, "discard_draft", session=id) == {"readback": "There is no draft for untitled, in cc-hands."}


async def test_an_ended_session_is_told_its_draft_cannot_change_and_the_draft_can_still_be_discarded(tmp_path: Path) -> None:
    sessions, id = await joined(tmp_path)
    tools = draft_tools(sessions)
    await call(tools, "stage_draft", session=id, text="run the tests", resolutions=[])
    await sessions.apply(Ended(id, "other"))
    assert await call(tools, "amend_draft", session=id, text="run the linter", resolutions=[]) == {
        "readback": "untitled, in cc-hands has ended, so its draft cannot be staged or changed."
    }
    assert await call(tools, "discard_draft", session=id) == {"readback": "Discarded the draft for untitled, in cc-hands."}


@pytest.mark.parametrize(
    ("text", "resolutions", "error"),
    [
        ("  \n", [], "the draft text is empty"),
        ("clear the line\x15and press enter\r", [], "control character"),
        ("fine", [{"heard": "x"}], "missing field 'meant'"),
        ("fine", "none", "resolutions should be a list"),
    ],
)
async def test_arguments_that_do_not_parse_are_refused_out_loud(text: str, resolutions: object, error: str, tmp_path: Path) -> None:
    sessions, id = await joined(tmp_path)
    result = await call(draft_tools(sessions), "stage_draft", session=id, text=text, resolutions=resolutions)
    assert error in str(result["error"])


async def test_the_draft_tools_are_valid_pipecat_direct_functions(tmp_path: Path) -> None:
    sessions, _ = await joined(tmp_path)
    wrappers = [DirectFunctionWrapper(tool) for tool in draft_tools(sessions)]
    assert [wrapper.name for wrapper in wrappers] == ["stage_draft", "amend_draft", "discard_draft"]
    schema = wrappers[0].to_function_schema()
    assert schema.required == ["session", "text", "resolutions"]
    assert schema.properties["resolutions"]["items"]["required"] == ["heard", "meant"]


async def test_an_interruption_does_not_cancel_a_draft_tool(tmp_path: Path) -> None:
    sessions, _ = await joined(tmp_path)
    assert [getattr(tool, "_pipecat_cancel_on_interruption") for tool in draft_tools(sessions)] == [False] * 3


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
    assert written[2]["result"] == {"readback": "Draft for untitled, in cc-hands: run the tests"}
    assert [datetime.fromisoformat(line["at"]) for line in written] == sorted(datetime.fromisoformat(line["at"]) for line in written)
