"""A permission hook against the real shim and socket: it waits, and the voice answer or the deadline decides it."""

import asyncio
import json
import os
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator, Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from loguru import logger
from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
from pipecat.frames.frames import Frame, LLMMessagesAppendFrame, TTSSpeakFrame
from pipecat.services.llm_service import FunctionCallParams

from hands.core.effects import Allow, Narrate, PermissionAsked, PermissionDeadlineNear, PermissionExpired, Speak
from hands.core.events import PermissionRequested, Tick, ToolFinished
from hands.core.reducer import EXPIRED_MESSAGE
from hands.core.session import Blocked, Permission, RequestId, SessionId, Working
from hands.sessions.home import Home
from hands.sessions.registry import Sessions
from hands.sessions.server import serve_hooks
from hands.voice.speech import frame, relay
from hands.voice.tools import DENIED_BY_VOICE, Tool, permission_tools

SID = SessionId("0f1e2d3c-aaaa-bbbb-cccc-000000000002")
COMMON = {"session_id": SID, "transcript_path": "/nowhere/t.jsonl", "cwd": "/code/cc-hands"}
START = {**COMMON, "hook_event_name": "SessionStart", "source": "startup"}
ASK: dict[str, object] = {**COMMON, "hook_event_name": "PermissionRequest", "tool_name": "Bash", "tool_input": {"command": "rm -r build"}, "permission_suggestions": []}
STOP = {**COMMON, "hook_event_name": "Stop", "stop_hook_active": False}
DEADLINE = 30.0
WAIT_SECONDS = 5.0


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def home() -> Iterator[Home]:
    # A unix socket path is capped near 104 bytes on macOS, so not under pytest's long tmp_path.
    root = Path(tempfile.mkdtemp(prefix="hands-"))
    yield Home(root)
    shutil.rmtree(root)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
async def sessions(home: Home, clock: Clock) -> AsyncIterator[Sessions]:
    registry = Sessions(permission_deadline=DEADLINE, clock=clock)
    runner = await serve_hooks(home, registry)
    yield registry
    await runner.cleanup()


class Shim:
    """One hook command, run as Claude Code runs it, its output read when it exits."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process

    @classmethod
    async def run(cls, home: Home, payload: Mapping[str, object]) -> "Shim":
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "hands.sessions.shim", str(home.root),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "TMUX_PANE": "%7"},
        )
        assert process.stdin is not None
        process.stdin.write(json.dumps(payload).encode())
        process.stdin.close()
        return cls(process)

    async def finished(self) -> tuple[int | None, str, str]:
        stdout, stderr = await asyncio.wait_for(self.process.communicate(), WAIT_SECONDS)
        return self.process.returncode, stdout.decode(), stderr.decode()


async def asked(home: Home, sessions: Sessions) -> tuple[Shim, PermissionAsked]:
    assert await (await Shim.run(home, START)).finished() == (0, "", "")
    shim = await Shim.run(home, ASK)
    heard = await asyncio.wait_for(sessions.heard(), WAIT_SECONDS)
    assert isinstance(heard, Narrate)
    return shim, heard.moment


async def call(tools: list[Tool], **arguments: object) -> dict[str, object]:
    results: list[dict[str, object]] = []

    async def capture(result: dict[str, object], **_: object) -> None:
        results.append(result)

    [tool] = tools
    await DirectFunctionWrapper(tool).invoke(arguments, cast(FunctionCallParams, SimpleNamespace(result_callback=capture)))
    [result] = results
    return result


def decision(stdout: str) -> object:
    return json.loads(stdout)["hookSpecificOutput"]["decision"]


async def test_a_voice_allow_is_what_the_waiting_hook_prints(home: Home, sessions: Sessions) -> None:
    shim, moment = await asked(home, sessions)
    assert [listing.session.state for listing in sessions.live()] == [Blocked(on=moment.permission, request=moment.request, deadline=DEADLINE, warned=False)]
    assert shim.process.returncode is None, "the hook returned before anyone answered"

    tools = permission_tools(sessions)
    assert await call(tools, request=moment.request, decision="allow") == {"readback": "Allowed Bash for untitled, in cc-hands."}
    code, stdout, _ = await shim.finished()
    assert (code, decision(stdout)) == (0, {"behavior": "allow"})
    assert [listing.session.state for listing in sessions.live()] == [Working(since=0.0)]
    assert await call(tools, request=moment.request, decision="deny") == {
        "readback": "That request is no longer waiting: it was already answered, answered at the keyboard, or denied at its deadline."
    }


@pytest.mark.parametrize(("message", "agent_reads"), [("use git clean instead", "use git clean instead"), ("", DENIED_BY_VOICE)])
async def test_a_voice_deny_carries_its_message_to_the_agent(home: Home, sessions: Sessions, message: str, agent_reads: str) -> None:
    shim, moment = await asked(home, sessions)
    assert await call(permission_tools(sessions), request=moment.request, decision="deny", message=message) == {
        "readback": "Denied Bash for untitled, in cc-hands."
    }
    code, stdout, _ = await shim.finished()
    assert (code, decision(stdout)) == (0, {"behavior": "deny", "message": agent_reads})


async def test_an_unanswered_request_is_denied_at_its_deadline_after_one_warning(home: Home, sessions: Sessions, clock: Clock) -> None:
    shim, moment = await asked(home, sessions)
    for second in range(1, int(DEADLINE) + 3):
        clock.now = float(second)
        await sessions.apply(Tick(clock.now))
    code, stdout, _ = await shim.finished()
    assert (code, decision(stdout)) == (0, {"behavior": "deny", "message": EXPIRED_MESSAGE})
    assert [await sessions.heard(), await sessions.heard()] == [
        Speak(PermissionDeadlineNear(SID, moment.permission, remaining=10.0)),
        Speak(PermissionExpired(SID, moment.permission)),
    ]
    assert await call(permission_tools(sessions), request=moment.request, decision="allow") == {
        "readback": "That request is no longer waiting: it was already answered, answered at the keyboard, or denied at its deadline."
    }


async def test_a_session_that_moves_on_lets_its_hook_return_undecided(home: Home, sessions: Sessions) -> None:
    shim, _ = await asked(home, sessions)
    assert await (await Shim.run(home, STOP)).finished() == (0, "", "")
    # Empty output decides nothing, so Claude Code's own dialog, or the keyboard answer it already had, stands.
    assert await shim.finished() == (0, "", "")


async def test_a_hook_that_went_away_is_neither_warned_about_nor_denied(home: Home, sessions: Sessions, clock: Clock) -> None:
    logged: list[str] = []
    sink = logger.add(lambda message: logged.append(message.record["message"]), level="INFO")
    try:
        shim, moment = await asked(home, sessions)
        shim.process.kill()
        await shim.process.wait()
        await asyncio.sleep(0.2)  # the daemon notices the closed connection
        clock.now = DEADLINE
        await sessions.apply(Tick(clock.now))
    finally:
        logger.remove(sink)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(sessions.heard(), 0.1)
    assert [line for line in logged if moment.request in line] == [f"the hook for session {moment.session} request {moment.request} closed"]


async def test_a_reply_decided_as_the_hook_closes_is_logged_as_never_delivered(home: Home, sessions: Sessions) -> None:
    logged: list[str] = []
    sink = logger.add(lambda message: logged.append(message.record["message"]), level="INFO")
    try:
        assert await (await Shim.run(home, START)).finished() == (0, "", "")
        request = PermissionRequested(SID, at=0.0, request=RequestId("r1"), permission=Permission("Bash", {}))
        waiting = asyncio.create_task(sessions.ask(request))
        await asyncio.wait_for(sessions.heard(), WAIT_SECONDS)
        await sessions.answer(request.request, Allow())
        waiting.cancel()  # the connection closes before the handler resumes with the reply
        with pytest.raises(asyncio.CancelledError):
            await waiting
    finally:
        logger.remove(sink)
    assert f"the hook for session {SID} request r1 closed; its reply Allow() was never delivered" in logged


async def test_tool_calls_from_a_session_that_never_joined_are_not_warned_about(sessions: Sessions) -> None:
    # Every tool call of a session started before the daemon would otherwise bury the warnings that matter.
    levels: list[str] = []
    sink = logger.add(lambda message: levels.append(message.record["level"].name), level="DEBUG")
    try:
        await sessions.apply(ToolFinished(SID, at=1.0, call=Permission("Bash", {})))
    finally:
        logger.remove(sink)
    assert levels == ["DEBUG"]


async def test_a_request_from_a_session_this_daemon_never_met_is_let_go_at_once(home: Home, sessions: Sessions) -> None:
    # A session that started before the daemon did: no SessionStart reached it, so nothing can be asked aloud.
    started = asyncio.get_running_loop().time()
    assert await (await Shim.run(home, ASK)).finished() == (0, "", "")
    assert asyncio.get_running_loop().time() - started < WAIT_SECONDS


async def test_a_voice_answer_after_the_hook_went_away_is_told_nothing_was_answered(home: Home, sessions: Sessions, clock: Clock) -> None:
    shim, moment = await asked(home, sessions)
    shim.process.kill()
    await shim.process.wait()
    await asyncio.sleep(0.2)  # the daemon notices the closed connection
    assert [listing.session.state for listing in sessions.live()] == [Working(since=0.0)]
    assert await call(permission_tools(sessions), request=moment.request, decision="allow") == {
        "readback": "That request is no longer waiting: it was already answered, answered at the keyboard, or denied at its deadline."
    }


async def test_the_tool_running_after_a_keyboard_answer_lets_the_hook_go(home: Home, sessions: Sessions) -> None:
    shim, _ = await asked(home, sessions)
    finished = {**COMMON, "hook_event_name": "PostToolUse", "tool_name": "Bash", "tool_input": {"command": "rm -r build"}, "tool_use_id": "t", "tool_response": {}}
    assert await (await Shim.run(home, finished)).finished() == (0, "", "")
    assert await shim.finished() == (0, "", "")


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"request": "r", "decision": "maybe"}, "decision should be 'allow' or 'deny', got 'maybe'"),
        ({"request": "r", "decision": "allow", "message": "sure"}, "a message goes only with deny"),
        ({"request": "", "decision": "allow"}, "request should be the request id string"),
    ],
)
async def test_an_answer_that_does_not_parse_is_refused_out_loud(sessions: Sessions, arguments: dict[str, object], error: str) -> None:
    assert error in str((await call(permission_tools(sessions), **arguments))["error"])


def test_the_tool_is_a_valid_direct_function_that_a_barge_in_cannot_cancel(sessions: Sessions) -> None:
    [tool] = permission_tools(sessions)
    schema = DirectFunctionWrapper(tool).to_function_schema()
    assert (schema.name, schema.required) == ("answer_permission", ["request", "decision"])
    assert getattr(tool, "_pipecat_cancel_on_interruption") is False


async def test_the_relay_hands_a_request_to_the_model_and_an_announcement_to_the_speaker(home: Home, sessions: Sessions, clock: Clock) -> None:
    frames: list[Frame] = []

    async def queue_frame(frame: Frame) -> None:
        frames.append(frame)

    shim, _ = await asked(home, sessions)
    relaying = asyncio.create_task(relay(sessions, queue_frame))
    await sessions.apply(Tick(DEADLINE - 10.0))
    await sessions.apply(Tick(DEADLINE))
    await asyncio.wait_for(shim.finished(), WAIT_SECONDS)
    await asyncio.sleep(0.05)
    relaying.cancel()

    # The request itself was taken off the queue by asked(); what follows it is spoken as written.
    [spoken_warning, spoken_expiry] = frames
    assert isinstance(spoken_warning, TTSSpeakFrame) and isinstance(spoken_expiry, TTSSpeakFrame)
    assert spoken_warning.text == "10 seconds left to answer untitled, in cc-hands about Bash."
    assert spoken_expiry.text == "Nobody answered untitled, in cc-hands about Bash in time, so I told it no."


def test_a_request_reaches_the_model_with_its_tool_input_and_request_id() -> None:
    moment = PermissionAsked(SID, RequestId("r-42"), Permission("Bash", {"command": "rm -r build"}))
    narrated = frame(Narrate(moment), names=lambda id: id)
    assert isinstance(narrated, LLMMessagesAppendFrame) and narrated.run_llm is True
    [message] = narrated.messages
    content = str(cast(dict[str, object], message)["content"])
    assert "session 0f1e2d3c-aaaa-bbbb-cccc-000000000002 is waiting for permission to use Bash" in content
    assert '{"command": "rm -r build"}' in content and "Request id: r-42" in content
