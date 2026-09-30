"""The brain's LLM stage in a running pipeline: a turn goes to the brain, its words come off the wire as LLM text.

The brain is a stand-in whose turns end when the test says, and the wire is what the proxy would tell: each request as
it leaves, each text delta as it arrives, each exchange once it ends. So every order the real pieces can arrive in is
set up on purpose, not waited for.
"""

import asyncio
import json
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass, field

import pytest
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from hands.brain.stage import INTERRUPTED, SILENT, BrainStage
from hands.core.session import SessionId
from hands.core.wire import (
    BlockStarted,
    BlockStopped,
    Fork,
    Heard,
    Hold,
    Kind,
    MainTurn,
    Route,
    Send,
    Sent,
    Tail,
    TextDelta,
    Unknown,
)
from hands.sessions.audit import BrainAnswered, BrainInterrupted, BrainSpoke, Entry
from hands.voice.tools import Result, Tool, tool

BRAIN = SessionId("brain-session")
PATIENCE_SECS = 2.0
ANSWERED = BrainAnswered("success", False, 1, 10)
TAIL = "[hands] The Claude Code sessions running now: none of note."


async def stage_draft(session: str, text: str) -> Result:
    """Stage a draft."""
    return {"readback": f"staged for {session}: {text}"}


async def stay_silent() -> Result:
    """Say nothing."""
    return {"silent": True}


async def read_session(session: str) -> Result:
    """Read a session."""
    return {"steps": []}


TOOLS: Sequence[Tool] = (tool(stage_draft, completes=True), tool(stay_silent, then="silence"), tool(read_session))


@dataclass
class FakeBrain:
    """A brain whose every turn ends when the test ends it."""

    session: SessionId = BRAIN
    asked: list[str] = field(default_factory=list[str])
    interrupts: int = 0
    turns: list[asyncio.Future[BrainAnswered]] = field(default_factory=list[asyncio.Future[BrainAnswered]])

    async def ask(self, text: str) -> BrainAnswered:
        self.asked.append(text)
        turn = asyncio.get_running_loop().create_future()
        self.turns.append(turn)
        return await asyncio.shield(turn)

    async def interrupt(self) -> None:
        self.interrupts += 1

    def end(self, answered: BrainAnswered = ANSWERED) -> None:
        self.turns[-1].set_result(answered)


class Spoken(FrameProcessor):
    """What leaves the stage for TTS."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame | LLMFullResponseEndFrame | LLMTextFrame | TTSSpeakFrame | InterruptionFrame):
            self.frames.append(frame)
        await self.push_frame(frame, direction)

    def said(self) -> list[str]:
        return [frame.text for frame in self.frames if isinstance(frame, LLMTextFrame | TTSSpeakFrame)]

    def shape(self) -> list[str]:
        return [type(frame).__name__ for frame in self.frames]


@dataclass
class Rig:
    worker: PipelineWorker
    stage: BrainStage
    brain: FakeBrain
    out: Spoken
    recorded: list[Entry]
    errors: list[ErrorFrame]
    # How the sessions stand, as the registry would compose it when a request leaves.
    standing: list[str]
    context: LLMContext = field(default_factory=LLMContext)
    exchanges: int = 0

    async def until(self, what: Callable[[], bool]) -> None:
        async with asyncio.timeout(PATIENCE_SECS):
            while not what():
                await asyncio.sleep(0.01)

    async def say(self, *messages: dict[str, str]) -> None:
        """The aggregator adding to the context, then handing it on as a turn ends."""
        for message in messages:
            self.context.add_message({"role": message["role"], "content": message["content"]})  # pyright: ignore[reportArgumentType]
        asked = len(self.brain.asked)
        await self.worker.queue_frame(LLMContextFrame(self.context))
        await self.until(lambda: len(self.brain.asked) > asked)

    def request(self, body: object = None, kind: Kind = MainTurn(), session: SessionId | None = BRAIN) -> tuple[str, Route]:
        """A request leaving on the wire: the stage hears it and decides where it goes."""
        self.exchanges += 1
        exchange = f"x{self.exchanges}"
        sent = Sent(exchange, session, kind, body or {"messages": [{"role": "user", "content": "hi"}]})
        self.stage.hear(sent)
        return exchange, self.stage.route(sent)

    def stream(self, exchange: str, *texts: str) -> None:
        for text in texts:
            self.stage.hear(Heard(exchange, TextDelta(0, text)))

    def calls(self, exchange: str, *calls: tuple[str, str]) -> None:
        """The reply making a whole block for each call, by id and name, as the stream carries it before the reply ends."""
        for index, (call, name) in enumerate(calls):
            self.stage.hear(Heard(exchange, BlockStarted(index, {"type": "tool_use", "id": call, "name": name, "input": {}})))
            self.stage.hear(Heard(exchange, BlockStopped(index)))

    async def interrupt(self) -> None:
        await self.worker.queue_frame(InterruptionFrame())
        await self.until(lambda: "InterruptionFrame" in self.out.shape())


@pytest.fixture
async def rig() -> AsyncGenerator[Rig, None]:
    brain = FakeBrain()
    recorded: list[Entry] = []
    standing = [TAIL]
    stage = BrainStage(brain, TOOLS, lambda: standing[-1], recorded.append)
    out = Spoken()
    worker = PipelineWorker(Pipeline([stage, out]), idle_timeout_secs=None)
    started = asyncio.Event()
    errors: list[ErrorFrame] = []

    @worker.event_handler("on_pipeline_started")
    async def _started(_worker: PipelineWorker, _frame: Frame) -> None:  # pyright: ignore[reportUnusedFunction]
        started.set()

    @worker.event_handler("on_pipeline_error")
    async def _failed(_worker: PipelineWorker, error: ErrorFrame) -> None:  # pyright: ignore[reportUnusedFunction]
        errors.append(error)

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    # As the daemon runs it: a watch beside the pipeline.
    asking = asyncio.create_task(stage.ask_each())
    await asyncio.wait_for(started.wait(), PATIENCE_SECS)
    yield Rig(worker, stage, brain, out, recorded, errors, standing)
    asking.cancel()
    await worker.cancel()
    await running


def answering(name: str, result: object, call_id: str = "t1") -> dict[str, object]:
    """A request whose last message hands the result of the brain's call to `name` back to the model."""
    return {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": call_id, "name": name, "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id, "content": [{"type": "text", "text": json.dumps(result)}]}]},
            # As Claude Code 2.1.285 sends it: a message of its own after the results.
            {"role": "system", "content": [{"type": "text", "text": "<system-reminder>tokens left</system-reminder>"}]},
        ]
    }


async def test_a_turn_goes_to_the_brain_and_its_words_come_off_the_wire(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "what is running?"})
    assert rig.brain.asked == ["what is running?"]
    exchange, route = rig.request()
    assert route == Send((Tail(TAIL),))
    rig.stream(exchange, "Two sessions ", "are running.")
    await rig.until(lambda: len(rig.out.said()) == 2)
    rig.brain.end()
    await rig.until(lambda: "LLMFullResponseEndFrame" in rig.out.shape())
    assert rig.out.shape() == ["LLMFullResponseStartFrame", "LLMTextFrame", "LLMTextFrame", "LLMFullResponseEndFrame"]
    assert rig.out.said() == ["Two sessions ", "are running."]
    # The audit log ties what was spoken to the exchange on the wire it came from.
    assert BrainSpoke((exchange,), "Two sessions are running.", (), False) in rig.recorded


async def test_each_request_of_a_turn_carries_how_the_sessions_stand_as_it_leaves(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "start the tests in auth"})
    exchange, first = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__stage_draft"))
    rig.standing.append("[hands] The Claude Code sessions running now: auth, working.")
    _, second = rig.request(answering("mcp__hands__stage_draft", {"readback": "staged"}))
    # Composed as each request leaves, never kept from the one before.
    assert (first, second) == (Send((Tail(TAIL),)), Send((Tail("[hands] The Claude Code sessions running now: auth, working."),)))
    rig.brain.end()


async def test_only_the_brains_own_main_turns_are_spoken(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "hello"})
    others = [rig.request(kind=Fork()), rig.request(kind=Unknown("a messages request with no tools")), rig.request(session=SessionId("a summary"))]
    assert [route for _, route in others] == [Send()] * 3
    for exchange, _ in others:
        rig.stream(exchange, "not for the user")
    exchange, _ = rig.request()
    rig.stream(exchange, "Hello.")
    await rig.until(lambda: rig.out.said() == ["Hello."])
    rig.brain.end()
    await rig.until(lambda: "LLMFullResponseEndFrame" in rig.out.shape())
    assert rig.out.said() == ["Hello."]


async def test_the_brain_hears_what_the_context_gained_and_never_its_own_words_again(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "[hands] hands has just started."}, {"role": "user", "content": "anything new?"})
    rig.brain.end()
    await rig.until(lambda: "LLMFullResponseEndFrame" in rig.out.shape())
    await rig.say({"role": "assistant", "content": "Nothing new."}, {"role": "user", "content": "[hands] a session is waiting."})
    assert rig.brain.asked == ["[hands] hands has just started.\n\nanything new?", "[hands] a session is waiting."]


async def test_a_turn_is_written_only_once_the_one_before_it_has_ended(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "first"})
    await rig.interrupt()
    rig.context.add_message({"role": "user", "content": "second"})
    await rig.worker.queue_frame(LLMContextFrame(rig.context))
    # The stage has taken the second turn and is waiting on the first, which the brain has not ended.
    await asyncio.sleep(0.1)
    assert rig.brain.asked == ["first"]
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    assert rig.brain.asked[1] == "second"


async def test_stay_silent_holds_the_next_request_so_nothing_follows_it(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "(to a colleague) back in five"})
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__stay_silent"))
    _, route = rig.request(answering("mcp__hands__stay_silent", {"silent": True}))
    assert route == Hold(SILENT)
    rig.brain.end()
    await rig.until(lambda: "LLMFullResponseEndFrame" in rig.out.shape())
    assert rig.out.said() == []
    # The held reply closed the call in the brain's history, so the silence does not carry into the next turn.
    await rig.say({"role": "user", "content": "are you there?"})
    closed = answering("mcp__hands__stay_silent", {"silent": True})
    closed["messages"] = [*closed["messages"], {"role": "assistant", "content": [{"type": "text", "text": SILENT}]}, {"role": "user", "content": "are you there?"}]  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]
    exchange, route = rig.request(closed)
    assert route == Send((Tail(TAIL),))
    rig.stream(exchange, "I am.")
    await rig.until(lambda: rig.out.said() == ["I am."])
    rig.brain.end()


async def test_a_barge_in_mid_reply_stops_the_brain_and_nothing_more_of_the_turn_is_said(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "tell me everything"})
    exchange, _ = rig.request()
    rig.stream(exchange, "First, ")
    await rig.until(lambda: rig.out.said() == ["First, "])
    await rig.interrupt()
    assert rig.brain.interrupts == 1
    assert BrainInterrupted((), True) in rig.recorded
    # What the wire carries after the barge-in is dropped, and a request that raced the stop is not asked either.
    rig.stream(exchange, "second, ", "third.")
    _, route = rig.request()
    assert route == Hold(INTERRUPTED)
    # A second press on the same turn does not tell the brain twice.
    await rig.worker.queue_frame(InterruptionFrame())
    rig.brain.end()
    await rig.until(lambda: any(isinstance(entry, BrainSpoke) for entry in rig.recorded))
    assert rig.out.said() == ["First, "]
    assert rig.brain.interrupts == 1
    assert BrainSpoke((exchange,), "First, ", (), True) in rig.recorded


async def test_a_barge_in_while_a_draft_lands_lets_it_finish_and_speaks_its_readback(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "tell the api session to add tests"})
    exchange, _ = rig.request()
    rig.stream(exchange, "Staging it.")
    # The call is open, and so running under Claude Code, while the reply is still streaming.
    rig.calls(exchange, ("t1", "mcp__hands__stage_draft"))
    await rig.interrupt()
    # Stopped by the harness, the draft would land and be written into history as refused; it is let run instead.
    assert rig.brain.interrupts == 0
    assert BrainInterrupted(("mcp__hands__stage_draft",), False) in rig.recorded
    _, route = rig.request(answering("mcp__hands__stage_draft", {"readback": "staged for api: add tests"}))
    assert route == Hold(INTERRUPTED)
    rig.brain.end()
    # Said once the turn is over, after anything the model had begun to say.
    await rig.until(lambda: rig.out.said()[-1:] == ["staged for api: add tests"])
    await rig.until(lambda: any(isinstance(entry, BrainSpoke) for entry in rig.recorded))
    assert BrainSpoke((exchange,), "Staging it.", ("staged for api: add tests",), True) in rig.recorded


async def test_a_barge_in_while_a_reading_tool_runs_stops_the_brain_at_once(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "what did the api session do?"})
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__read_session"))
    await rig.interrupt()
    assert rig.brain.interrupts == 1
    assert BrainInterrupted(("mcp__hands__read_session",), True) in rig.recorded
    rig.brain.end()


async def test_a_barge_in_with_no_turn_in_flight_tells_the_brain_nothing(rig: Rig) -> None:
    await rig.interrupt()
    assert rig.brain.interrupts == 0
    assert not any(isinstance(entry, BrainInterrupted) for entry in rig.recorded)


async def test_frames_behind_a_turn_pass_while_the_brain_is_still_on_it(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "check every session"})
    # A session asking permission is announced while the brain works through its tools, not once it is done.
    await rig.worker.queue_frame(TTSSpeakFrame("api asks to run tests."))
    await rig.until(lambda: rig.out.said() == ["api asks to run tests."])
    rig.brain.end()


async def test_what_the_user_said_while_a_turn_ran_is_asked_once_it_ends_through_any_number_of_presses(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "first"})
    await rig.interrupt()
    rig.context.add_message({"role": "user", "content": "second"})
    await rig.worker.queue_frame(LLMContextFrame(rig.context))
    await rig.worker.queue_frame(InterruptionFrame())
    await asyncio.sleep(0.1)
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    assert rig.brain.asked == ["first", "second"]


async def test_notes_that_came_in_one_ask_are_not_asked_again_empty(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "first"})
    for note in ("[hands] api finished.", "[hands] web finished."):
        rig.context.add_message({"role": "user", "content": note})
        await rig.worker.queue_frame(LLMContextFrame(rig.context))
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    rig.brain.end()
    await asyncio.sleep(0.1)
    assert rig.brain.asked == ["first", "[hands] api finished.\n\n[hands] web finished."]


async def test_a_turn_the_brain_ended_in_error_is_reported_as_the_model_stages_error(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "hello"})
    rig.brain.end(BrainAnswered("error_during_execution", True, 1, 10))
    await rig.until(lambda: len(rig.errors) == 1)
    assert rig.errors[0].processor is rig.stage
    assert "error_during_execution" in rig.errors[0].error


async def test_a_stay_silent_answered_in_an_earlier_turn_does_not_hold_the_next(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "(to a colleague) back in five"})
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__stay_silent"))
    await rig.interrupt()
    rig.brain.end()
    # The next turn's first request carries the old call's result beside the new question: it is not this turn's call.
    await rig.say({"role": "user", "content": "are you there?"})
    body = answering("mcp__hands__stay_silent", {"silent": True})
    body["messages"] = [*body["messages"], {"role": "user", "content": "are you there?"}]  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]
    _, route = rig.request(body)
    assert route == Send((Tail(TAIL),))
    rig.brain.end()


async def test_a_draft_that_failed_under_a_held_request_says_why(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "tell the api session to add tests"})
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__stage_draft"))
    await rig.interrupt()
    _, route = rig.request(answering("mcp__hands__stage_draft", {"error": "there is no session api"}))
    assert route == Hold(INTERRUPTED)
    rig.brain.end()
    await rig.until(lambda: rig.out.said() == ["there is no session api"])


async def test_a_turn_hands_stopped_ends_in_the_error_it_asked_for_and_nothing_is_said_of_it(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "tell me everything"})
    exchange, _ = rig.request()
    rig.stream(exchange, "First, ")
    await rig.until(lambda: rig.out.said() == ["First, "])
    await rig.interrupt()
    # As Claude Code 2.1.285 ends a turn it was told to stop.
    rig.brain.end(BrainAnswered("error_during_execution", True, 1, 10))
    await rig.until(lambda: any(isinstance(entry, BrainSpoke) for entry in rig.recorded))
    await asyncio.sleep(0.1)
    assert rig.errors == []


async def test_a_barge_in_while_a_drafts_input_still_streams_stops_the_brain_before_it_runs(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "tell the api session to add tests"})
    exchange, _ = rig.request()
    # The block is open and its input still arriving: Claude Code has not run the call, and now never will.
    rig.stage.hear(Heard(exchange, BlockStarted(0, {"type": "tool_use", "id": "t1", "name": "mcp__hands__stage_draft", "input": {}})))
    await rig.interrupt()
    assert rig.brain.interrupts == 1
    assert BrainInterrupted((), True) in rig.recorded
    rig.brain.end()


async def test_a_barge_in_reaches_the_pipeline_even_when_the_brain_cannot_be_told(rig: Rig) -> None:
    async def gone() -> None:
        raise BrokenPipeError("the brain's stdin is closed")

    rig.brain.interrupt = gone  # type: ignore[method-assign]
    await rig.say({"role": "user", "content": "tell me everything"})
    exchange, _ = rig.request()
    rig.stream(exchange, "First, ")
    await rig.until(lambda: rig.out.said() == ["First, "])
    await rig.interrupt()
    # Nothing of the turn after the barge-in: no end that would have TTS say the sentence spoken over.
    assert rig.out.shape() == ["LLMFullResponseStartFrame", "LLMTextFrame", "InterruptionFrame"]
    rig.brain.end()
