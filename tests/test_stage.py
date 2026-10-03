"""The brain's LLM stage in a running pipeline: a turn goes to the brain, its words come off the wire as LLM text.

The brain is a stand-in whose turns end when the test says, and the wire is what the proxy would tell: each request as
it leaves, each text delta as it arrives, each exchange once it ends. So every order the real pieces can arrive in is
set up on purpose, not waited for.
"""

import asyncio
import json
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import pytest
from datetime import UTC, datetime
from pipecat.utils.errors import ErrorCategory
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
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from conftest import running
from hands.brain.process import Untaken
from hands.brain.stage import INTERRUPTED, SILENT, BrainStage
from hands.core.session import SessionId
from hands.core.wire import (
    Answering,
    BlockStarted,
    BlockStopped,
    Exchanged,
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
    Unreached,
    UsageLimitReached,
)
from hands.sessions.model_facts import ModelFact, ModelFailed, ModelFault, ModelReplyEmpty, ModelUnreachable
from hands.sessions.audit import BrainAnswered, BrainInterrupted, BrainSpoke, Entry
from hands.voice.speech import Aloud, Narrated
from hands.voice.tools import Result, Tool, tool

BRAIN = SessionId("brain-session")
PATIENCE_SECS = 2.0
ANSWERED = BrainAnswered("p1", None)
TAIL = "[hands] The Claude Code sessions running now: none of note."


async def stage_draft(session: str, text: str) -> Result:
    """Stage a draft."""
    return {"readback": f"staged for {session}: {text}"}


async def amend_draft(session: str, text: str) -> Result:
    """Amend a draft, which hands reads back itself."""
    return {"says": f"amended for {session}: {text}"}


async def stay_silent() -> Result:
    """Say nothing."""
    return {"silent": True}


async def read_session(session: str) -> Result:
    """Read a session."""
    return {"steps": []}


TOOLS: Sequence[Tool] = (
    tool(stage_draft, completes=True),
    tool(amend_draft, then="silence", completes=True),
    tool(stay_silent, then="silence"),
    tool(read_session),
)


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

    def interrupt(self) -> None:
        self.interrupts += 1

    def end(self, answered: BrainAnswered = ANSWERED) -> None:
        self.turns[-1].set_result(answered)

    def fail(self, error: Exception) -> None:
        self.turns[-1].set_exception(error)


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
    # The stage's clock, which the test moves.
    now: list[float]
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

    def request(self, body: object = None, kind: Kind = MainTurn(None), session: SessionId | None = BRAIN) -> tuple[str, Route]:
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
    now = [0.0]
    stage = BrainStage(brain, TOOLS, lambda: standing[-1], recorded.append, clock=lambda: now[0])
    out = Spoken()
    async with running([stage, out]) as run:
        # As the daemon runs it: a watch beside the pipeline.
        asking = asyncio.create_task(stage.ask_each())
        try:
            yield Rig(run.worker, stage, brain, out, recorded, run.errors, standing, now)
        finally:
            asking.cancel()


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


async def test_a_finished_turn_hands_narrates_reaches_the_brain_as_a_typed_turn_of_its_own(rig: Rig) -> None:
    """What hands says aloud of a session's turn is the brain's own turn, so it can answer about what the user heard."""
    await rig.worker.queue_frame(Narrated("[hands] The Claude Code session api finished a turn.", "api finished a turn, and I could not tell it."))
    await rig.until(lambda: rig.brain.asked == ["[hands] The Claude Code session api finished a turn."])
    exchange, _ = rig.request()
    rig.stream(exchange, "api opened pull request 68.")
    rig.brain.end()
    await rig.until(lambda: any(isinstance(entry, BrainSpoke) for entry in rig.recorded))
    assert rig.out.said() == ["api opened pull request 68."]
    assert BrainSpoke((exchange,), "api opened pull request 68.", (), False, "hands", 0.0, None) in rig.recorded
    # Never in Pipecat's context, where it would ride along with whatever the user says next.
    assert rig.context.get_messages() == []


async def test_the_users_turn_goes_ahead_of_a_narration_waiting_for_the_brain(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "what is running?"})
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it."))
    rig.context.add_message({"role": "user", "content": "and the backlog?"})
    await rig.worker.queue_frame(LLMContextFrame(rig.context))
    # Frames pass the stage in order, so once this is out, both before it are waiting in the stage.
    await rig.worker.queue_frame(TTSSpeakFrame("marker"))
    await rig.until(lambda: "marker" in rig.out.said())
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 3)
    assert rig.brain.asked == ["what is running?", "and the backlog?", "[hands] api finished a turn."]


async def test_a_narration_records_how_long_it_waited_behind_the_users_turn(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "what is running?"})
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it."))
    await rig.worker.queue_frame(TTSSpeakFrame("marker"))
    await rig.until(lambda: "marker" in rig.out.said())
    rig.now[0] = 2.5
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    exchange, _ = rig.request()
    rig.stream(exchange, "api finished.")
    rig.brain.end()
    await rig.until(lambda: any(isinstance(entry, BrainSpoke) and entry.asker == "hands" for entry in rig.recorded))
    assert BrainSpoke((exchange,), "api finished.", (), False, "hands", 2.5, None) in rig.recorded


async def test_what_hands_says_as_written_is_heard_after_the_narration_ahead_of_it(rig: Rig) -> None:
    """A session's end is said after its last turn, though that turn waits for the brain and the end needs no model."""
    await rig.say({"role": "user", "content": "what is running?"})
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it."))
    await rig.worker.queue_frame(Aloud(TTSSpeakFrame("The session api is gone.")))
    await rig.worker.queue_frame(TTSSpeakFrame("marker"))
    await rig.until(lambda: "marker" in rig.out.said())
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    exchange, _ = rig.request()
    rig.stream(exchange, "api opened pull request 68.")
    await rig.until(lambda: "api opened pull request 68." in rig.out.said())
    assert "The session api is gone." not in rig.out.said()
    rig.brain.end()
    await rig.until(lambda: "The session api is gone." in rig.out.said())
    assert rig.out.said() == ["marker", "api opened pull request 68.", "The session api is gone."]


async def test_a_narration_the_brain_fails_is_said_as_written_with_its_question(rig: Rig) -> None:
    unsaid = "api finished a turn, and I could not tell it. Want me to push it?"
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", unsaid))
    await rig.until(lambda: len(rig.brain.asked) == 1)
    rig.brain.end(BrainAnswered("p1", "unknown: API Error: 500 overloaded"))
    await rig.until(lambda: len(rig.errors) == 1)
    assert rig.out.said() == [unsaid]


async def test_a_users_turn_the_brain_fails_has_nothing_said_for_it(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "hello"})
    rig.brain.end(BrainAnswered("p1", "unknown: API Error: 500 overloaded"))
    await rig.until(lambda: len(rig.errors) == 1)
    assert rig.out.said() == []


async def test_a_turn_goes_to_the_brain_and_its_words_come_off_the_wire(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "what is running?"})
    assert rig.brain.asked == ["what is running?"]
    exchange, route = rig.request()
    assert route == Send((Tail(TAIL),), refusal="final")
    rig.stream(exchange, "Two sessions ", "are running.")
    await rig.until(lambda: len(rig.out.said()) == 2)
    rig.brain.end()
    await rig.until(lambda: "LLMFullResponseEndFrame" in rig.out.shape())
    assert rig.out.shape() == ["LLMFullResponseStartFrame", "LLMTextFrame", "LLMTextFrame", "LLMFullResponseEndFrame"]
    assert rig.out.said() == ["Two sessions ", "are running."]
    # The audit log ties what was spoken to the exchange on the wire it came from.
    assert BrainSpoke((exchange,), "Two sessions are running.", (), False, "user", 0.0, None) in rig.recorded


async def test_each_request_of_a_turn_carries_how_the_sessions_stand_as_it_leaves(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "start the tests in auth"})
    exchange, first = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__stage_draft"))
    rig.standing.append("[hands] The Claude Code sessions running now: auth, working.")
    _, second = rig.request(answering("mcp__hands__stage_draft", {"readback": "staged"}))
    # Composed as each request leaves, never kept from the one before.
    assert (first, second) == (Send((Tail(TAIL),), refusal="final"), Send((Tail("[hands] The Claude Code sessions running now: auth, working."),), refusal="final"))
    rig.brain.end()


async def test_only_the_brains_own_main_turns_are_spoken(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "hello"})
    others = [rig.request(kind=Fork()), rig.request(kind=Unknown("a messages request with no tools")), rig.request(session=SessionId("a summary"))]
    # Every request of the brain's is final, said or not; another session's goes as it came (hands-wire-zi2).
    assert [route for _, route in others] == [Send(refusal="final"), Send(refusal="final"), Send()]
    for exchange, _ in others:
        rig.stream(exchange, "not for the user")
    exchange, _ = rig.request()
    rig.stream(exchange, "Hello.")
    await rig.until(lambda: rig.out.said() == ["Hello."])
    rig.brain.end()
    await rig.until(lambda: "LLMFullResponseEndFrame" in rig.out.shape())
    assert rig.out.said() == ["Hello."]


async def test_a_main_turn_no_turn_asked_of_is_final_so_a_spent_limit_is_never_continued(rig: Rig) -> None:
    # Asked again at the reset, it would run the brain's tools with nobody there (hands-wire-zi2).
    _, route = rig.request()
    assert route == Send(refusal="final")


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
    # The model chose to say nothing: not a failure, so nothing is reported either.
    await asyncio.sleep(0.1)
    assert rig.errors == []
    # The held reply closed the call in the brain's history, so the silence does not carry into the next turn.
    await rig.say({"role": "user", "content": "are you there?"})
    closed = answering("mcp__hands__stay_silent", {"silent": True})
    closed["messages"] = [*closed["messages"], {"role": "assistant", "content": [{"type": "text", "text": SILENT}]}, {"role": "user", "content": "are you there?"}]  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]
    exchange, route = rig.request(closed)
    assert route == Send((Tail(TAIL),), refusal="final")
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
    assert BrainSpoke((exchange,), "First, ", (), True, "user", 0.0, None) in rig.recorded


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
    assert BrainSpoke((exchange,), "Staging it.", ("staged for api: add tests",), True, "user", 0.0, None) in rig.recorded


async def test_a_readback_a_call_hands_hands_ends_the_turn_and_is_said_by_hands_as_written(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "make it add tests too"})
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__amend_draft"))
    _, route = rig.request(answering("mcp__hands__amend_draft", {"says": "amended for api: add tests too"}))
    assert route == Hold(SILENT)
    rig.brain.end()
    await rig.until(lambda: rig.out.said() == ["amended for api: add tests too"])


async def test_a_barge_in_while_a_draft_hands_reads_back_lands_lets_it_finish_and_its_readback_is_said(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "make it add tests too"})
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__amend_draft"))
    await rig.interrupt()
    assert rig.brain.interrupts == 0
    _, route = rig.request(answering("mcp__hands__amend_draft", {"says": "amended for api: add tests too"}))
    assert route == Hold(INTERRUPTED)
    rig.brain.end()
    await rig.until(lambda: any(isinstance(entry, BrainSpoke) for entry in rig.recorded))
    # Said after the barge-in, never cut off by it: the draft changed, so the user hears how.
    assert BrainSpoke((exchange,), "", ("amended for api: add tests too",), True, "user", 0.0, None) in rig.recorded


async def test_a_refused_call_to_a_silence_tool_is_the_models_to_answer(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "make it add tests too"})
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__amend_draft"))
    refused, route = rig.request(answering("mcp__hands__amend_draft", {"error": "the draft text is empty"}))
    # Nothing was said and nothing changed: the model is asked to go on, to retry or to say what went wrong.
    assert route == Send((Tail(TAIL),), refusal="final")
    rig.stream(refused, "I couldn't change it.")
    await rig.until(lambda: rig.out.said() == ["I couldn't change it."])
    rig.brain.end()


async def test_a_reply_with_one_draft_said_and_one_refused_is_the_models_to_answer(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "make both add tests too"})
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__amend_draft"), ("t2", "mcp__hands__amend_draft"))
    results: tuple[tuple[str, dict[str, str]], ...] = (("t1", {"says": "amended for api: add tests too"}), ("t2", {"error": "There is no session web."}))
    body: dict[str, object] = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": call, "name": "mcp__hands__amend_draft", "input": {}} for call, _ in results]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call, "content": [{"type": "text", "text": json.dumps(result)}]} for call, result in results]},
        ]
    }
    _, route = rig.request(body)
    assert route == Send((Tail(TAIL),), refusal="final")
    rig.brain.end()


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


async def test_a_turn_whose_replies_carried_nothing_is_said_as_the_models_empty_reply(rig: Rig) -> None:
    unsaid = "api finished a turn, and I could not tell it."
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", unsaid))
    await rig.until(lambda: len(rig.brain.asked) == 1)
    # Whitespace is not words.
    exchange, _ = rig.request()
    rig.stream(exchange, " ")
    rig.brain.end()
    await rig.until(lambda: len(rig.errors) == 1)
    assert isinstance(error := rig.errors[0].exception, ModelFault) and error.fact == ModelReplyEmpty()
    assert rig.errors[0].processor is rig.stage
    # What hands had for the brain to tell is said as written, as for any turn the brain failed.
    await rig.until(lambda: unsaid in rig.out.said())
    # [LAW:nothing-unseen] the empty reply is on the turn's line in the audit log.
    [spoke] = [entry for entry in rig.recorded if isinstance(entry, BrainSpoke)]
    assert spoke.failed == ModelReplyEmpty()
    # An empty reply is whole and in the brain's history: its next turn is not told the API broke one off.
    await rig.say({"role": "user", "content": "hello?"})
    assert rig.brain.asked[-1] == "hello?"


async def test_a_turn_that_called_a_tool_and_then_answered_nothing_is_no_empty_reply(rig: Rig) -> None:
    """Claude often ends a turn with an empty reply to a tool's result: the call already did what the turn was for."""
    await rig.say({"role": "user", "content": "what is api doing?"})
    exchange, _ = rig.request()
    rig.stream(exchange, "Let me look.")
    rig.calls(exchange, ("t1", "mcp__hands__read_session"))
    rig.request(answering("mcp__hands__read_session", {"steps": []}))
    rig.brain.end()
    await rig.until(lambda: any(isinstance(entry, BrainSpoke) for entry in rig.recorded))
    await asyncio.sleep(0.1)
    assert rig.errors == []


async def test_a_turn_ending_on_a_call_that_lands_is_no_empty_reply(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "tell the api session to add tests"})
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__stage_draft"))
    rig.brain.end()
    await rig.until(lambda: any(isinstance(entry, BrainSpoke) for entry in rig.recorded))
    await asyncio.sleep(0.1)
    assert rig.errors == []


async def test_a_turn_the_brain_ended_in_error_is_reported_as_the_model_stages_error(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "hello"})
    rig.brain.end(BrainAnswered("p1", "unknown: API Error: 500 overloaded"))
    await rig.until(lambda: len(rig.errors) == 1)
    assert rig.errors[0].processor is rig.stage
    assert "API Error: 500 overloaded" in rig.errors[0].error


RESETS = datetime(2026, 9, 30, 18, 0, tzinfo=UTC).timestamp()


def unreached(exchange: str) -> Exchanged:
    """The proxy's record of a request it could not get to the API, told before it answers the brain 502."""
    return Exchanged(exchange, BRAIN, MainTurn(None), "POST", "/v1/messages", 2, (), 0.0, 0.0, Unreached("ClientConnectorError: no route", 0.0), True)


@pytest.mark.parametrize(
    ("answers", "failure", "fact"),
    [
        # As Claude Code 2.1.285 meets each (measured, hands-wire-6ic.gfq): a spent limit is asked once, an API the proxy
        # cannot reach eleven times, and what failed the turn is its latest answer.
        ([429], "rate_limit: You've hit your session limit · resets 12:00pm", UsageLimitReached(RESETS)),
        ([None, None, None], "server_error: API Error: 502 hands' proxy could not reach https://api.anthropic.com", ModelUnreachable()),
        ([529], "server_error: API Error: 529 Overloaded", ModelFailed(ErrorCategory.SERVER)),
        ([None, 529, 200], "server_error: API Error: Connection lost mid-response.", ModelFailed(ErrorCategory.UNKNOWN)),
        # A request whose answer never came, its handler cancelled before the head: an earlier answer is not its reason.
        ([None, "unanswered"], "server_error: API Error: Request timed out.", ModelFailed(ErrorCategory.UNKNOWN)),
    ],
)
async def test_a_failed_turn_is_said_as_the_wire_says_it_failed(rig: Rig, answers: list[int | None | Literal["unanswered"]], failure: str, fact: ModelFact) -> None:
    await rig.say({"role": "user", "content": "hello"})
    for status in answers:
        exchange, _ = rig.request()
        match status:
            case None:
                rig.stage.hear(unreached(exchange))
            case 429:
                rig.stage.hear(Answering(exchange, 429, UsageLimitReached(RESETS)))
            case int():
                rig.stage.hear(Answering(exchange, status, None))
            case "unanswered":
                pass
    rig.brain.end(BrainAnswered("p1", failure))
    await rig.until(lambda: len(rig.errors) == 1)
    assert isinstance(error := rig.errors[0].exception, ModelFault) and error.fact == fact
    # [LAW:nothing-unseen] which failure the turn was said as is on its line in the audit log.
    [spoke] = [entry for entry in rig.recorded if isinstance(entry, BrainSpoke)]
    assert spoke.failed == fact


async def test_a_reply_the_api_breaks_mid_stream_is_said_once_as_far_as_it_came_then_the_failure(rig: Rig) -> None:
    # As Claude Code 2.1.285 ends it (hands-wire-6ic.6dz, measured): no retry and no fallback request, only StopFailure.
    await rig.say({"role": "user", "content": "tell me about lighthouses"})
    exchange, _ = rig.request()
    rig.stream(exchange, "1. Lighthouses stand ", "on rocky coasts and h")
    await rig.until(lambda: len(rig.out.said()) == 2)
    rig.brain.end(BrainAnswered("p1", "server_error: API Error: Connection lost mid-response. The response above may be incomplete."))
    await rig.until(lambda: len(rig.errors) == 1)
    assert rig.out.shape() == ["LLMFullResponseStartFrame", "LLMTextFrame", "LLMTextFrame", "LLMFullResponseEndFrame"]
    assert rig.out.said() == ["1. Lighthouses stand ", "on rocky coasts and h"]
    assert rig.errors[0].processor is rig.stage
    assert "Connection lost mid-response" in rig.errors[0].error
    assert BrainSpoke((exchange,), "1. Lighthouses stand on rocky coasts and h", (), False, "user", 0.0, ModelFailed(ErrorCategory.UNKNOWN)) in rig.recorded
    # The broken reply is not in the brain's history, so its next turn tells it what the user heard, and only that one.
    await rig.say({"role": "user", "content": "what were you saying?"})
    rig.brain.end()
    await rig.say({"role": "user", "content": "thanks"})
    assert rig.brain.asked[1:] == [
        '[hands] The API broke off your last turn. The user heard you say "1. Lighthouses stand on rocky coasts and h", then that it failed. '
        "Say nothing about this unless the user asks.\n\nwhat were you saying?",
        "thanks",
    ]


async def test_what_the_user_heard_of_a_broken_turn_is_told_with_the_first_turn_the_brain_takes(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "tell me about lighthouses"})
    rig.stream(rig.request()[0], "Lighthouses stand")
    await rig.until(lambda: len(rig.out.said()) == 1)
    rig.brain.end(BrainAnswered("p1", "server_error: API Error: Connection lost mid-response."))
    await rig.until(lambda: len(rig.errors) == 1)
    await rig.say({"role": "user", "content": "what were you saying?"})
    rig.brain.fail(Untaken("the brain did not take the turn typed into it in 30s"))
    await rig.until(lambda: len(rig.errors) == 2)
    await rig.say({"role": "user", "content": "again"})
    note = '[hands] The API broke off your last turn. The user heard you say "Lighthouses stand", then that it failed. Say nothing about this unless the user asks.'
    assert rig.brain.asked[1:] == [f"{note}\n\nwhat were you saying?", f"{note}\n\nagain"]
    rig.brain.end()


async def test_a_turn_the_brain_never_took_is_reported_as_the_model_stages_error_and_the_next_is_asked(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "hello"})
    # The brain is still running, so no watch reports it: the stage does.
    rig.brain.fail(Untaken("the brain did not take the turn typed into it in 30s"))
    await rig.until(lambda: len(rig.errors) == 1)
    assert rig.errors[0].processor is rig.stage
    assert "did not take the turn" in rig.errors[0].error
    await rig.say({"role": "user", "content": "again"})
    await rig.until(lambda: rig.brain.asked == ["hello", "again"])
    rig.brain.end()


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
    assert route == Send((Tail(TAIL),), refusal="final")
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
    # A turn the API fails as it is being stopped is still a turn hands stopped, not an error to report.
    rig.brain.end(BrainAnswered("p1", "unknown: API Error: 500 overloaded"))
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


async def test_a_barge_in_before_the_brain_has_sent_the_turn_stops_nothing_and_the_turn_is_answered(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "are you listening?"})
    # Still waiting for the input: nothing of it has left, so there is nothing to stop, and the user's words follow it.
    await rig.interrupt()
    assert rig.brain.interrupts == 0
    assert not any(isinstance(entry, BrainInterrupted) for entry in rig.recorded)
    _, route = rig.request()
    assert route == Send((Tail(TAIL),), refusal="final")
    rig.brain.end()


async def test_a_barge_in_reaches_the_pipeline_even_when_the_brain_cannot_be_told(rig: Rig) -> None:
    def gone() -> None:
        raise RuntimeError("the brain cannot be told")

    rig.brain.interrupt = gone  # type: ignore[method-assign]
    await rig.say({"role": "user", "content": "tell me everything"})
    exchange, _ = rig.request()
    rig.stream(exchange, "First, ")
    await rig.until(lambda: rig.out.said() == ["First, "])
    await rig.interrupt()
    # Nothing of the turn after the barge-in: no end that would have TTS say the sentence spoken over.
    assert rig.out.shape() == ["LLMFullResponseStartFrame", "LLMTextFrame", "InterruptionFrame"]
    rig.brain.end()


async def test_each_text_block_of_a_turn_is_its_own_paragraph_so_a_closing_fence_stays_on_its_own_line(rig: Rig) -> None:
    """hands-narration-2mc.1zu: joined bare, the next block's words ran onto the closer and held the rest of the turn as code."""
    await rig.say({"role": "user", "content": "run the tests"})
    exchange, _ = rig.request()
    rig.stage.hear(Heard(exchange, BlockStarted(0, {"type": "text", "text": ""})))
    rig.stream(exchange, "Here is what I ran:\n```bash\nnpm test\n```")
    rig.calls(exchange, ("t1", "mcp__hands__stage_draft"))
    second, _ = rig.request(answering("mcp__hands__stage_draft", {"readback": "staged"}))
    rig.stage.hear(Heard(second, BlockStarted(0, {"type": "text", "text": ""})))
    rig.stream(second, "All 42 tests passed.")
    rig.brain.end()
    await rig.until(lambda: any(isinstance(entry, BrainSpoke) for entry in rig.recorded))
    assert "".join(rig.out.said()) == "Here is what I ran:\n```bash\nnpm test\n```\n\nAll 42 tests passed."
