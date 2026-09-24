"""A permission hook against the real shim and socket: it waits, and the voice answer or the deadline decides it."""

import asyncio
import json
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

from hands.core.effects import Allow, Narrate, Withdraw, Asking, DeadlineNear, Expired, Speak
from hands.core.events import PermissionRequested, Tick, ToolFinished
from hands.core.reducer import EXPIRED_MESSAGE
from hands.core.session import AskedQuestion, AtDialog, Blocked, Option, Permission, Plan, Question, RequestId, SessionId, Working
from hands.sessions.home import Home
from hands.sessions.registry import Sessions
from hands.sessions.server import serve_hooks
from hands.voice.speech import frame, relay
from hands.voice.tools import DENIED_BY_VOICE, SENT_BACK_BY_VOICE, Tool, permission_tools

SID = SessionId("0f1e2d3c-aaaa-bbbb-cccc-000000000002")
COMMON = {"session_id": SID, "transcript_path": "/nowhere/t.jsonl", "cwd": "/code/cc-hands"}
START = {**COMMON, "hook_event_name": "SessionStart", "source": "startup"}
ASK: dict[str, object] = {**COMMON, "hook_event_name": "PermissionRequest", "tool_name": "Bash", "tool_input": {"command": "rm -r build"}, "permission_suggestions": []}
PROMPT = {**COMMON, "hook_event_name": "UserPromptSubmit", "prompt": "clean the build", "prompt_id": "p"}
STOP = {**COMMON, "hook_event_name": "Stop", "stop_hook_active": False, "prompt_id": "p"}
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
    registry = Sessions(permission_deadline=DEADLINE, clock=clock, record=lambda _: None)
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
        )
        assert process.stdin is not None
        process.stdin.write(json.dumps(payload).encode())
        process.stdin.close()
        return cls(process)

    async def finished(self) -> tuple[int | None, str, str]:
        stdout, stderr = await asyncio.wait_for(self.process.communicate(), WAIT_SECONDS)
        return self.process.returncode, stdout.decode(), stderr.decode()


QUESTIONS: dict[str, object] = {
    "questions": [
        {"question": "Which color?", "header": "Color", "options": [{"label": "red", "description": "warm"}, {"label": "green", "description": "cool"}], "multiSelect": False},
        {"question": "Which fruits?", "header": "Fruits", "options": [{"label": "pear", "description": ""}, {"label": "plum", "description": ""}], "multiSelect": True},
    ]
}
QUESTION: dict[str, object] = {**ASK, "tool_name": "AskUserQuestion", "tool_input": QUESTIONS}
QUESTIONS_ASKED = (
    AskedQuestion("Which color?", (Option("red", "warm"), Option("green", "cool")), several=False),
    AskedQuestion("Which fruits?", (Option("pear", None), Option("plum", None)), several=True),
)


PLAN_TEXT = "# Plan\n\n1. Create hello.txt.\n2. Write hi into it.\n"
PLAN: dict[str, object] = {**ASK, "tool_name": "ExitPlanMode", "tool_input": {"plan": PLAN_TEXT, "planFilePath": "/nowhere/plan.md"}}


async def asked(home: Home, sessions: Sessions, payload: Mapping[str, object] = ASK) -> tuple[Shim, Asking]:
    assert await (await Shim.run(home, START)).finished() == (0, "", "")
    # A request is asked inside the turn a prompt opened, and the Stop that ends that turn names it.
    assert await (await Shim.run(home, PROMPT)).finished() == (0, "", "")
    shim = await Shim.run(home, payload)
    heard = await asyncio.wait_for(sessions.heard(), WAIT_SECONDS)
    assert isinstance(heard, Narrate)
    return shim, heard.moment


def named(sessions: Sessions, name: str) -> Tool:
    [tool] = [tool for tool in permission_tools(sessions) if tool.__name__ == name]
    return tool


async def call(tool: Tool, **arguments: object) -> dict[str, object]:
    results: list[dict[str, object]] = []

    async def capture(result: dict[str, object], **_: object) -> None:
        results.append(result)

    params = SimpleNamespace(result_callback=capture, function_name=tool.__name__)
    await DirectFunctionWrapper(tool).invoke(arguments, cast(FunctionCallParams, params))
    [result] = results
    return result


def decision(stdout: str) -> object:
    return json.loads(stdout)["hookSpecificOutput"]["decision"]


async def test_a_voice_allow_is_what_the_waiting_hook_prints(home: Home, sessions: Sessions) -> None:
    shim, moment = await asked(home, sessions)
    assert [listing.session.state for listing in sessions.live()] == [Blocked(on=moment.on, request=moment.request, deadline=DEADLINE, warned=False)]
    assert shim.process.returncode is None, "the hook returned before anyone answered"

    tool = named(sessions, "answer_permission")
    assert await call(tool, request=moment.request, decision="allow") == {"readback": "Allowed Bash for untitled, in cc-hands."}
    code, stdout, _ = await shim.finished()
    assert (code, decision(stdout)) == (0, {"behavior": "allow"})
    assert [listing.session.state for listing in sessions.live()] == [Working(since=0.0)]
    assert await call(tool, request=moment.request, decision="deny") == {
        "readback": "That request is no longer waiting for a voice answer: it was already answered, answered at the keyboard, or its deadline passed."
    }


async def test_voice_answers_to_a_question_are_what_the_waiting_hook_prints_in_its_input(home: Home, sessions: Sessions) -> None:
    shim, moment = await asked(home, sessions, QUESTION)
    assert shim.process.returncode is None, "the hook returned before anyone answered"
    assert await call(named(sessions, "answer_question"), request=moment.request, answers=["green", "pear, plum"]) == {
        "readback": "Answered green; pear, plum for untitled, in cc-hands."
    }
    code, stdout, _ = await shim.finished()
    answers = {"Which color?": "green", "Which fruits?": "pear, plum"}
    assert (code, decision(stdout)) == (0, {"behavior": "allow", "updatedInput": {**QUESTIONS, "answers": answers}})


@pytest.mark.parametrize(
    ("tool", "arguments", "readback"),
    [
        ("answer_question", {"answers": ["green"]}, "it asked 2 questions and was given 1 answers"),
        ("answer_permission", {"decision": "allow"}, "that request is a question"),
    ],
)
async def test_an_answer_that_does_not_fit_the_question_sends_nothing_and_it_still_waits(
    home: Home, sessions: Sessions, tool: str, arguments: dict[str, object], readback: str
) -> None:
    shim, moment = await asked(home, sessions, QUESTION)
    assert readback in str((await call(named(sessions, tool), request=moment.request, **arguments))["readback"])
    await asyncio.sleep(0.2)
    assert shim.process.returncode is None, "the hook was answered with something that does not answer it"
    assert [listing.session.state for listing in sessions.live()] == [Blocked(on=moment.on, request=moment.request, deadline=DEADLINE, warned=False)]
    await call(named(sessions, "answer_question"), request=moment.request, answers=["red", "pear"])
    await shim.finished()


async def test_a_question_nobody_answers_by_its_deadline_is_left_to_its_dialog_and_said_to_be(home: Home, sessions: Sessions, clock: Clock) -> None:
    shim, moment = await asked(home, sessions, QUESTION)
    for clock.now in (DEADLINE - 10.0, DEADLINE):
        await sessions.apply(Tick(clock.now))
    code, stdout, _ = await shim.finished()
    # Printing nothing decides nothing: the dialog, where the user may be answering, stays up.
    assert (code, stdout) == (0, "")
    [warning, expiry] = [await sessions.heard(), await sessions.heard()]
    assert (warning, expiry) == (Speak(DeadlineNear(SID, moment.on, remaining=10.0)), Speak(Expired(SID, moment.on)))
    spoken = [frame(heard, names=lambda _: "quiz") for heard in (warning, expiry)]
    assert [cast(TTSSpeakFrame, said).text for said in spoken] == [
        "10 seconds left to answer quiz about its question.",
        "Nobody answered quiz about its question in time, so it is left waiting at its dialog.",
    ]
    # Still at its dialog, and said to be; a voice answer now is refused out loud rather than sent late.
    assert [listing.session.state for listing in sessions.live()] == [AtDialog(moment.on)]
    assert str((await call(named(sessions, "answer_question"), request=moment.request, answers=["red", "pear"]))["readback"]).startswith("That request is no longer waiting")
    # Answered at the keyboard after all, it comes back through PostToolUse and the session goes on.
    answered = Question(QUESTIONS_ASKED, {**QUESTIONS, "answers": {"Which color?": "red", "Which fruits?": "pear"}})
    await sessions.apply(ToolFinished(SID, at=DEADLINE + 5.0, call=answered, mode=None))
    assert [listing.session.state for listing in sessions.live()] == [Working(since=DEADLINE + 5.0)]


async def test_an_empty_answer_leaves_its_question_unanswered(home: Home, sessions: Sessions) -> None:
    shim, moment = await asked(home, sessions, QUESTION)
    assert await call(named(sessions, "answer_question"), request=moment.request, answers=["green", ""]) == {
        "readback": "Answered green; nothing for untitled, in cc-hands."
    }
    _, stdout, _ = await shim.finished()
    assert decision(stdout) == {"behavior": "allow", "updatedInput": {**QUESTIONS, "answers": {"Which color?": "green", "Which fruits?": ""}}}


@pytest.mark.parametrize(
    ("answers", "error"),
    [("green", "answers should be a list"), ([3], "each answer should be a string")],
)
async def test_answers_that_do_not_parse_are_refused_out_loud(sessions: Sessions, answers: object, error: str) -> None:
    assert error in str((await call(named(sessions, "answer_question"), request="r", answers=answers))["error"])


def test_a_question_reaches_the_model_whole_with_its_options_and_request_id() -> None:
    long = "a description long enough that the questions together run past what a tool input is shown " * 4
    asked = (
        AskedQuestion("Which color?", (Option("red", long), Option("green", None)), several=False),
        AskedQuestion("Which fruits?", (Option("pear", long), Option("plum", long)), several=True),
        AskedQuestion("Name it?", (), several=False),
    )
    narrated = frame(Narrate(Asking(SID, RequestId("q-7"), Question(asked, {}))), names=lambda id: id)
    assert isinstance(narrated, LLMMessagesAppendFrame)
    [message] = narrated.messages
    content = str(cast(dict[str, object], message)["content"])
    assert f"1. Which color? Options: red ({long}); green." in content
    assert f"2. Which fruits? Options: pear ({long}); plum ({long}). More than one may be chosen." in content
    assert "3. Name it? Answered in the user's own words." in content
    assert "Request id: q-7" in content and "answer_question" in content and "cut short" not in content


@pytest.mark.parametrize(("message", "agent_reads"), [("use git clean instead", "use git clean instead"), ("", DENIED_BY_VOICE)])
async def test_a_voice_deny_carries_its_message_to_the_agent(home: Home, sessions: Sessions, message: str, agent_reads: str) -> None:
    shim, moment = await asked(home, sessions)
    assert await call(named(sessions, "answer_permission"), request=moment.request, decision="deny", message=message) == {
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
        Speak(DeadlineNear(SID, moment.on, remaining=10.0)),
        Speak(Expired(SID, moment.on)),
    ]
    assert await call(named(sessions, "answer_permission"), request=moment.request, decision="allow") == {
        "readback": "That request is no longer waiting for a voice answer: it was already answered, answered at the keyboard, or its deadline passed."
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
        request = PermissionRequested(SID, at=0.0, request=RequestId("r1"), on=Permission("Bash", {}), mode=None)
        waiting = asyncio.create_task(sessions.ask(request))
        await asyncio.wait_for(sessions.heard(), WAIT_SECONDS)
        await sessions.answer(request.request, Allow())
        waiting.cancel()  # the connection closes before the handler resumes with the reply
        with pytest.raises(asyncio.CancelledError):
            await waiting
    finally:
        logger.remove(sink)
    assert f"the hook for session {SID} request r1 closed; its reply Allow() was never delivered" in logged


async def test_a_daemon_shutting_down_lets_a_waiting_hook_go_instead_of_waiting_out_its_deadline(home: Home, clock: Clock) -> None:
    registry = Sessions(permission_deadline=DEADLINE, clock=clock, record=lambda _: None)
    runner = await serve_hooks(home, registry)
    shim, _ = await asked(home, registry)
    await asyncio.wait_for(runner.cleanup(), WAIT_SECONDS)
    # Empty output decides nothing: Claude Code's own dialog stands.
    assert await shim.finished() == (0, "", "")


async def test_a_hook_that_asks_after_shutdown_began_is_let_go_at_once(home: Home, sessions: Sessions) -> None:
    assert await (await Shim.run(home, START)).finished() == (0, "", "")
    sessions.release_waiting()
    request = PermissionRequested(SID, at=0.0, request=RequestId("late"), on=Permission("Bash", {}), mode=None)
    assert await asyncio.wait_for(sessions.ask(request), WAIT_SECONDS) == Withdraw()


async def test_tool_calls_from_a_session_that_never_joined_are_not_warned_about(sessions: Sessions) -> None:
    # Every tool call of a session started before the daemon would otherwise bury the warnings that matter.
    levels: list[str] = []
    sink = logger.add(lambda message: levels.append(message.record["level"].name), level="DEBUG")
    try:
        await sessions.apply(ToolFinished(SID, at=1.0, call=Permission("Bash", {}), mode=None))
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
    assert await call(named(sessions, "answer_permission"), request=moment.request, decision="allow") == {
        "readback": "That request is no longer waiting for a voice answer: it was already answered, answered at the keyboard, or its deadline passed."
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
    assert error in str((await call(named(sessions, "answer_permission"), **arguments))["error"])


def test_the_tools_are_valid_direct_functions_that_a_barge_in_cannot_cancel(sessions: Sessions) -> None:
    schemas = [(DirectFunctionWrapper(tool).to_function_schema(), tool) for tool in permission_tools(sessions)]
    assert [(schema.name, schema.required) for schema, _ in schemas] == [("answer_permission", ["request", "decision"]), ("answer_question", ["request", "answers"]), ("answer_plan", ["request", "decision"])]
    assert all(getattr(tool, "_pipecat_cancel_on_interruption") is False for _, tool in schemas)


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
    moment = Asking(SID, RequestId("r-42"), Permission("Bash", {"command": "rm -r build"}))
    narrated = frame(Narrate(moment), names=lambda id: id)
    assert isinstance(narrated, LLMMessagesAppendFrame) and narrated.run_llm is True
    [message] = narrated.messages
    content = str(cast(dict[str, object], message)["content"])
    assert "session 0f1e2d3c-aaaa-bbbb-cccc-000000000002 is waiting for permission to use Bash" in content
    assert '{"command": "rm -r build"}' in content and "Request id: r-42" in content


SET_MODE = [{"type": "setMode", "mode": "acceptEdits", "destination": "session"}], [{"type": "setMode", "mode": "default", "destination": "session"}]


@pytest.mark.parametrize(
    ("choice", "permissions", "readback"),
    [
        # No mode set: ExitPlanMode goes back to the mode the session had before it planned, bypass or auto included.
        ("approve", [], "Approved its plan for untitled, in cc-hands, in the mode it had before planning."),
        ("auto-accept edits", SET_MODE[0], "Approved its plan for untitled, in cc-hands, with its edits accepted automatically."),
        ("manually approve edits", SET_MODE[1], "Approved its plan for untitled, in cc-hands, asking you about each edit."),
    ],
)
async def test_a_plan_approved_by_voice_leaves_plan_mode_for_the_mode_chosen(
    home: Home, sessions: Sessions, choice: str, permissions: list[object], readback: str
) -> None:
    shim, moment = await asked(home, sessions, PLAN)
    assert moment.on == Plan(PLAN_TEXT)
    assert shim.process.returncode is None, "the hook returned before anyone answered"
    assert await call(named(sessions, "answer_plan"), request=moment.request, decision=choice) == {"readback": readback}
    code, stdout, _ = await shim.finished()
    assert (code, decision(stdout)) == (0, {"behavior": "allow", "updatedInput": {}, "updatedPermissions": permissions})


@pytest.mark.parametrize(("message", "agent_reads"), [("split step 2 in two", "split step 2 in two"), ("", SENT_BACK_BY_VOICE)])
async def test_a_plan_sent_back_by_voice_tells_the_agent_what_to_change(home: Home, sessions: Sessions, message: str, agent_reads: str) -> None:
    shim, moment = await asked(home, sessions, PLAN)
    assert await call(named(sessions, "answer_plan"), request=moment.request, decision="keep planning", message=message) == {
        "readback": "Sent its plan back to keep planning for untitled, in cc-hands."
    }
    code, stdout, _ = await shim.finished()
    assert (code, decision(stdout)) == (0, {"behavior": "deny", "message": agent_reads})


async def test_a_plan_allowed_as_a_permission_sends_nothing_and_it_still_waits(home: Home, sessions: Sessions) -> None:
    shim, moment = await asked(home, sessions, PLAN)
    assert "that request is a plan" in str((await call(named(sessions, "answer_permission"), request=moment.request, decision="allow"))["readback"])
    await asyncio.sleep(0.2)
    assert shim.process.returncode is None, "a plain allow would leave plan mode for a mode nobody chose"
    await call(named(sessions, "answer_plan"), request=moment.request, decision="keep planning")
    await shim.finished()


async def test_feedback_for_a_plan_sent_to_a_permission_sends_nothing_and_it_still_waits(home: Home, sessions: Sessions) -> None:
    shim, moment = await asked(home, sessions)
    assert "that request is a permission" in str((await call(named(sessions, "answer_plan"), request=moment.request, decision="keep planning", message="x"))["readback"])
    await asyncio.sleep(0.2)
    assert shim.process.returncode is None, "a tool was refused with feedback meant for a plan"
    await call(named(sessions, "answer_permission"), request=moment.request, decision="deny")
    await shim.finished()


async def test_a_plan_nobody_answers_by_its_deadline_is_left_to_its_dialog(home: Home, sessions: Sessions, clock: Clock) -> None:
    shim, moment = await asked(home, sessions, PLAN)
    for clock.now in (DEADLINE - 10.0, DEADLINE):
        await sessions.apply(Tick(clock.now))
    code, stdout, _ = await shim.finished()
    assert (code, stdout) == (0, "")
    [warning, expiry] = [await sessions.heard(), await sessions.heard()]
    assert [cast(TTSSpeakFrame, frame(heard, names=lambda _: "planner")).text for heard in (warning, expiry)] == [
        "10 seconds left to answer planner about its plan.",
        "Nobody answered planner about its plan in time, so it is left waiting at its dialog.",
    ]
    assert [listing.session.state for listing in sessions.live()] == [AtDialog(moment.on)]


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"decision": "yes"}, "decision should be 'approve', 'auto-accept edits', 'manually approve edits', or 'keep planning'"),
        ({"decision": "approve", "message": "go"}, "a message goes only with keep planning"),
        ({"decision": "approve", "message": 3}, "message should be a string"),
        ({"decision": "keep planning", "message": 3}, "message should be a string"),
    ],
)
async def test_plan_answers_that_do_not_parse_are_refused_out_loud(sessions: Sessions, arguments: dict[str, object], error: str) -> None:
    assert error in str((await call(named(sessions, "answer_plan"), request="r", **arguments))["error"])


def test_a_plan_reaches_the_model_whole_with_its_request_id() -> None:
    long = "\n".join(f"{step}. A step described at length so the plan runs past what a tool input is shown." for step in range(1, 30))
    narrated = frame(Narrate(Asking(SID, RequestId("p-3"), Plan(long))), names=lambda id: id)
    assert isinstance(narrated, LLMMessagesAppendFrame)
    [message] = narrated.messages
    content = str(cast(dict[str, object], message)["content"])
    assert long in content and "Request id: p-3" in content and "answer_plan" in content
