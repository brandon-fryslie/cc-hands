"""A permission hook against the real shim and socket: it waits, and the voice answer or the deadline decides it."""

import asyncio
import inspect
import json
import os
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator, Iterator, Mapping
from pathlib import Path
from types import CoroutineType
from typing import cast

import pytest
from loguru import logger
from pipecat.frames.frames import Frame, LLMMessagesAppendFrame, TTSSpeakFrame

from hands.core.attention import Attention, Overlay
from hands.core.effects import Allow, HookReply, Narrate, Withdraw, Asking, DeadlineNear, Expired, Speak
from hands.core.events import PermissionRequested, StatusReported, Tick, ToolFinished
from hands.core.reducer import EXPIRED_MESSAGE
from hands.core.session import AskedQuestion, Held, LetGo, Option, Permission, Plan, Question, RequestId, SessionId
from hands.core.status import Busy, Idle, Report, Stamp
from hands.sessions.home import Home
from hands.sessions.registry import Sessions
from hands.sessions.names import Names
from hands.sessions.server import serve_hooks
from hands.voice.speech import Aloud, Narrated, Pushed, Told, Tailed, Unprompted, frames, relay

from test_shim import STARTED
from hands.voice.tools import DENIED_BY_VOICE, SENT_BACK_BY_VOICE, Tool, permission_tools

SID = SessionId("0f1e2d3c-aaaa-bbbb-cccc-000000000002")
COMMON = {"session_id": SID, "transcript_path": "/nowhere/t.jsonl", "cwd": "/code/cc-hands"}
START = {**COMMON, "hook_event_name": "SessionStart", "source": "startup"}
ASK: dict[str, object] = {**COMMON, "hook_event_name": "PermissionRequest", "tool_name": "Bash", "tool_input": {"command": "rm -r build"}, "permission_suggestions": []}
PROMPT = {**COMMON, "hook_event_name": "UserPromptSubmit", "prompt": "clean the build", "prompt_id": "p"}
DEADLINE = 30.0
BUSY = Report(Busy(), Stamp(1))


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
    runner = await serve_hooks(home, registry, Names(), lambda _: None)
    yield registry
    await runner.cleanup()


class Shim:
    """One hook command, run as Claude Code runs it, its output read when it exits."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process

    @classmethod
    async def run(cls, home: Home, payload: Mapping[str, object]) -> "Shim":
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "hands.sessions.shim", env={**os.environ, "HANDS_HOME": str(home.root)},
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
    assert await (await Shim.run(home, START)).finished() == (0, STARTED, "")
    # A request is asked inside the turn a prompt opened, and the Stop that ends that turn names it.
    assert await (await Shim.run(home, PROMPT)).finished() == (0, "", "")
    # Claude Code set the session busy before the prompt's hooks ran: only a running session can be held at a dialog.
    await sessions.apply(StatusReported(SID, BUSY, at=0.0))
    shim = await Shim.run(home, payload)
    heard = await asyncio.wait_for(sessions.heard(), WAIT_SECONDS)
    assert isinstance(heard, Narrate)
    return shim, heard.moment


def named(sessions: Sessions, name: str) -> Tool:
    [tool] = [tool for tool in permission_tools(sessions) if tool.name == name]
    return tool


async def call(tool: Tool, **arguments: object) -> dict[str, object]:
    return dict(await tool.body(**arguments))


def decision(stdout: str) -> object:
    return json.loads(stdout)["hookSpecificOutput"]["decision"]


async def test_a_voice_allow_is_what_the_waiting_hook_prints(home: Home, sessions: Sessions) -> None:
    shim, moment = await asked(home, sessions)
    assert [listing.session.dialog for listing in sessions.live()] == [Held(on=moment.on, request=moment.request, deadline=DEADLINE, warned=False)]
    assert shim.process.returncode is None, "the hook returned before anyone answered"

    tool = named(sessions, "answer_permission")
    assert await call(tool, request=moment.request, decision="allow") == {"readback": "Allowed Bash for cc-hands."}
    code, stdout, _ = await shim.finished()
    assert (code, decision(stdout)) == (0, {"behavior": "allow"})
    assert [listing.session.dialog for listing in sessions.live()] == [None]
    assert await call(tool, request=moment.request, decision="deny") == {
        "readback": "That request is no longer waiting for a voice answer: it was already answered, answered at the keyboard, or its deadline passed."
    }


async def test_voice_answers_to_a_question_are_what_the_waiting_hook_prints_in_its_input(home: Home, sessions: Sessions) -> None:
    shim, moment = await asked(home, sessions, QUESTION)
    assert shim.process.returncode is None, "the hook returned before anyone answered"
    assert await call(named(sessions, "answer_question"), request=moment.request, answers=["green", "pear, plum"]) == {
        "readback": "Answered green; pear, plum for cc-hands."
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
    assert [listing.session.dialog for listing in sessions.live()] == [Held(on=moment.on, request=moment.request, deadline=DEADLINE, warned=False)]
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
    assert (warning, expiry) == (Speak(DeadlineNear(SID, moment.request, moment.on, remaining=10.0)), Speak(Expired(SID, moment.on)))
    spoken = [said for heard in (warning, expiry) for said in frames(cast(Speak, heard), Pushed(), names=lambda _: "quiz")]
    assert [cast(TTSSpeakFrame, said).text for said in spoken] == [
        "10 seconds left to answer quiz about its question.",
        "Nobody answered quiz about its question in time, so it is left waiting at its dialog.",
    ]
    # Still at its dialog, and said to be; a voice answer now is refused out loud rather than sent late.
    assert [listing.session.dialog for listing in sessions.live()] == [LetGo(moment.on)]
    assert str((await call(named(sessions, "answer_question"), request=moment.request, answers=["red", "pear"]))["readback"]).startswith("That request is no longer waiting")
    # Answered at the keyboard after all, it comes back through PostToolUse and the session goes on.
    answered = Question(QUESTIONS_ASKED, {**QUESTIONS, "answers": {"Which color?": "red", "Which fruits?": "pear"}})
    await sessions.apply(ToolFinished(SID, at=DEADLINE + 5.0, call=answered, mode=None))
    assert [listing.session.dialog for listing in sessions.live()] == [None]


async def test_an_empty_answer_leaves_its_question_unanswered(home: Home, sessions: Sessions) -> None:
    shim, moment = await asked(home, sessions, QUESTION)
    assert await call(named(sessions, "answer_question"), request=moment.request, answers=["green", ""]) == {
        "readback": "Answered green; nothing for cc-hands."
    }
    _, stdout, _ = await shim.finished()
    assert decision(stdout) == {"behavior": "allow", "updatedInput": {**QUESTIONS, "answers": {"Which color?": "green", "Which fruits?": ""}}}


@pytest.mark.parametrize(
    ("answers", "error"),
    [("green", "answers should be a list"), ([3], "each answer should be a string")],
)
async def test_answers_that_do_not_parse_are_refused_out_loud(sessions: Sessions, answers: object, error: str) -> None:
    assert error in str((await call(named(sessions, "answer_question"), request="r", answers=answers))["error"])


def test_the_brain_takes_what_a_session_asks_as_a_turn_of_its_own_after_the_users() -> None:
    [narrated] = frames(Narrate(Asking(SID, RequestId("r-1"), Permission("Bash", {"command": "ls"}))), Tailed(), names=lambda id: id)
    assert isinstance(narrated, Narrated) and "is waiting for permission to use Bash" in narrated.text
    # Said as written if the brain cannot take it, so a session waiting on the user is still heard waiting.
    assert narrated.unsaid == f"{SID} is waiting on you about Bash."
    # Told the user once the brain takes it, so what they say next is taken as their answer to it.
    assert narrated.session == SID


def test_under_the_brain_an_announcement_waits_in_hands_lane_behind_the_question_it_counts_down() -> None:
    [said] = frames(Speak(DeadlineNear(SID, RequestId("r1"), Permission("Bash", {"command": "ls"}), 10.0)), Tailed(), names=lambda _: "quiz")
    assert isinstance(said, Aloud) and said.spoken.text.startswith("10 seconds left to answer quiz")


def test_a_question_reaches_the_model_whole_with_its_options_and_request_id() -> None:
    long = "a description long enough that the questions together run past what a tool input is shown " * 4
    asked = (
        AskedQuestion("Which color?", (Option("red", long), Option("green", None)), several=False),
        AskedQuestion("Which fruits?", (Option("pear", long), Option("plum", long)), several=True),
        AskedQuestion("Name it?", (), several=False),
    )
    [narrated, told] = frames(Narrate(Asking(SID, RequestId("q-7"), Question(asked, {}))), Pushed(), names=lambda id: id)
    assert isinstance(narrated, LLMMessagesAppendFrame) and isinstance(told, Told) and told.session == SID
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
        "readback": "Denied Bash for cc-hands."
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
        Speak(DeadlineNear(SID, moment.request, moment.on, remaining=10.0)),
        Speak(Expired(SID, moment.on)),
    ]
    assert await call(named(sessions, "answer_permission"), request=moment.request, decision="allow") == {
        "readback": "That request is no longer waiting for a voice answer: it was already answered, answered at the keyboard, or its deadline passed."
    }


async def test_a_session_that_moves_on_lets_its_hook_return_undecided(home: Home, sessions: Sessions) -> None:
    shim, _ = await asked(home, sessions)
    # Claude Code says it is at its prompt: the dialog was answered at the keyboard, or escaped, and the turn is over.
    await sessions.apply(StatusReported(SID, Report(Idle(), Stamp(2)), at=1.0))
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


async def asking_in_process(home: Home, sessions: Sessions) -> tuple[PermissionRequested, asyncio.Task[HookReply]]:
    """A permission request r1 asked straight of the registry, as a hook handler would, once its dialog is heard."""
    assert await (await Shim.run(home, START)).finished() == (0, STARTED, "")
    await sessions.apply(StatusReported(SID, BUSY, at=0.0))
    request = PermissionRequested(SID, at=0.0, request=RequestId("r1"), on=Permission("Bash", {}), mode=None)
    waiting = asyncio.create_task(sessions.ask(request))
    await asyncio.wait_for(sessions.heard(), WAIT_SECONDS)
    return request, waiting


async def test_a_reply_decided_as_the_hook_closes_is_logged_as_never_delivered(home: Home, sessions: Sessions) -> None:
    logged: list[str] = []
    sink = logger.add(lambda message: logged.append(message.record["message"]), level="INFO")
    try:
        request, waiting = await asking_in_process(home, sessions)
        await sessions.answer(request.request, Allow())
        waiting.cancel()  # the connection closes before the handler resumes with the reply
        with pytest.raises(asyncio.CancelledError):
            await waiting
    finally:
        logger.remove(sink)
    assert f"the hook for session {SID} request r1 closed; its reply Allow() was never delivered" in logged


async def waits_on_its_reply(handler: asyncio.Task[object]) -> None:
    """Return once the started handler is past applying its request: it awaits the future itself, not the apply coroutine."""
    coroutine = handler.get_coro()
    assert isinstance(coroutine, CoroutineType)
    while inspect.iscoroutine(coroutine.cr_await):
        await asyncio.sleep(0)


async def test_a_reply_decided_while_a_closed_hooks_cancellation_is_in_flight_is_dropped(home: Home, sessions: Sessions) -> None:
    logged: list[str] = []
    sink = logger.add(lambda message: logged.append(message.record["message"]), level="INFO")
    try:
        request, waiting = await asking_in_process(home, sessions)
        await asyncio.wait_for(waits_on_its_reply(waiting), WAIT_SECONDS)
        waiting.cancel()  # cancels the request's future at once; the handler forgets the request only when it next runs
        # A voice answer is decided, and its reply performed, before the handler has run.
        await sessions.answer(request.request, Allow())
        with pytest.raises(asyncio.CancelledError):
            await waiting
    finally:
        logger.remove(sink)
    assert f"reply Allow() for session {SID} request r1 was never delivered: its hook has closed" in logged


async def test_a_reply_decided_after_shutdown_let_its_hook_go_is_logged_as_never_delivered(home: Home, sessions: Sessions) -> None:
    logged: list[str] = []
    sink = logger.add(lambda message: logged.append(message.record["message"]), level="INFO")
    try:
        request, waiting = await asking_in_process(home, sessions)
        sessions.release_waiting()
        # A voice answer is decided, and its reply performed, before the handler has run with its Withdraw.
        await sessions.answer(request.request, Allow())
        assert await waiting == Withdraw()
    finally:
        logger.remove(sink)
    assert f"reply Allow() for session {SID} request r1 was never delivered: its hook was already given Withdraw()" in logged


async def test_a_daemon_shutting_down_lets_a_waiting_hook_go_instead_of_waiting_out_its_deadline(home: Home, clock: Clock) -> None:
    registry = Sessions(permission_deadline=DEADLINE, clock=clock, record=lambda _: None)
    runner = await serve_hooks(home, registry, Names(), lambda _: None)
    shim, _ = await asked(home, registry)
    await asyncio.wait_for(runner.cleanup(), WAIT_SECONDS)
    # Empty output decides nothing: Claude Code's own dialog stands.
    assert await shim.finished() == (0, "", "")


async def test_a_hook_that_asks_after_shutdown_began_is_let_go_at_once(home: Home, sessions: Sessions) -> None:
    assert await (await Shim.run(home, START)).finished() == (0, STARTED, "")
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


async def test_a_request_from_a_session_this_daemon_never_met_joins_it_and_is_answered_aloud(home: Home, sessions: Sessions) -> None:
    # A session running since before the plugin was: no SessionStart ever fired, so the request is where it joins.
    shim = await Shim.run(home, ASK)
    heard = await asyncio.wait_for(sessions.heard(), WAIT_SECONDS)
    assert isinstance(heard, Narrate)
    assert await call(named(sessions, "answer_permission"), request=heard.moment.request, decision="allow") == {"readback": "Allowed Bash for cc-hands."}
    code, stdout, _ = await shim.finished()
    assert (code, decision(stdout)) == (0, {"behavior": "allow"})


async def test_a_voice_answer_after_the_hook_went_away_is_told_nothing_was_answered(home: Home, sessions: Sessions, clock: Clock) -> None:
    shim, moment = await asked(home, sessions)
    shim.process.kill()
    await shim.process.wait()
    await asyncio.sleep(0.2)  # the daemon notices the closed connection
    assert [listing.session.dialog for listing in sessions.live()] == [None]
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


def test_the_tools_say_their_arguments_and_complete_through_a_barge_in(sessions: Sessions) -> None:
    tools = permission_tools(sessions)
    assert [(tool.name, tool.required) for tool in tools] == [("answer_permission", ("request", "decision")), ("answer_question", ("request", "answers")), ("answer_plan", ("request", "decision"))]
    assert all(tool.completes for tool in tools)


async def test_the_relay_hands_a_request_to_the_model_and_an_announcement_to_the_speaker(home: Home, sessions: Sessions, clock: Clock) -> None:
    queued: list[Frame] = []

    async def queue_frame(frame: Frame) -> None:
        queued.append(frame)

    shim, _ = await asked(home, sessions)
    async def unfocused(_session: SessionId) -> tuple[Attention, bool, Overlay]:
        return Attention(), False, "normal"
    relaying = asyncio.create_task(relay(sessions, queue_frame, lambda _: None, unfocused, lambda _progress, _amount: None))
    await sessions.apply(Tick(DEADLINE - 10.0))
    await sessions.apply(Tick(DEADLINE))
    await asyncio.wait_for(shim.finished(), WAIT_SECONDS)
    await asyncio.sleep(0.05)
    relaying.cancel()

    # The request itself was taken off the queue by asked(); what follows it is spoken as written.
    [spoken_warning, spoken_expiry] = [said for frame in queued for said in frames(cast(Unprompted, frame).pending, Pushed(), names=lambda _: "cc-hands")]
    assert isinstance(spoken_warning, TTSSpeakFrame) and isinstance(spoken_expiry, TTSSpeakFrame)
    assert spoken_warning.text == "10 seconds left to answer cc-hands about Bash."
    assert spoken_expiry.text == "Nobody answered cc-hands about Bash in time, so I told it no."


def test_a_request_reaches_the_model_with_its_tool_input_and_request_id() -> None:
    moment = Asking(SID, RequestId("r-42"), Permission("Bash", {"command": "rm -r build"}))
    [narrated, _] = frames(Narrate(moment), Pushed(), names=lambda id: id)
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
        ("approve", [], "Approved its plan for cc-hands, in the mode it had before planning."),
        ("auto-accept edits", SET_MODE[0], "Approved its plan for cc-hands, with its edits accepted automatically."),
        ("manually approve edits", SET_MODE[1], "Approved its plan for cc-hands, asking you about each edit."),
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
        "readback": "Sent its plan back to keep planning for cc-hands."
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
    assert [cast(TTSSpeakFrame, said).text for heard in (warning, expiry) for said in frames(cast(Speak, heard), Pushed(), names=lambda _: "planner")] == [
        "10 seconds left to answer planner about its plan.",
        "Nobody answered planner about its plan in time, so it is left waiting at its dialog.",
    ]
    assert [listing.session.dialog for listing in sessions.live()] == [LetGo(moment.on)]


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
    [narrated, _] = frames(Narrate(Asking(SID, RequestId("p-3"), Plan(long))), Pushed(), names=lambda id: id)
    assert isinstance(narrated, LLMMessagesAppendFrame)
    [message] = narrated.messages
    content = str(cast(dict[str, object], message)["content"])
    assert long in content and "Request id: p-3" in content and "answer_plan" in content
