"""The brain's LLM stage in a running pipeline: a turn goes to the brain, its words come off the wire as LLM text.

The brain is a stand-in whose turns end when the test says, and the wire is what the proxy would tell: each request as
it leaves, each text delta as it arrives, each exchange once it ends. So every order the real pieces can arrive in is
set up on purpose, not waited for.
"""

import asyncio
import json
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass, field, replace
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
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from conftest import running
from hands.brain.process import NOBODY, SPOKEN_OVER, Asked, BrainAnswered, Untaken
from hands.core.effects import Allow, Deny
from hands.core import place
from hands.core.place import Modality
from hands.core.session import Permission
from hands.brain.mcp import CallSpans
from hands.brain.stage import INTERRUPTED, SILENT, BrainStage, HandsAsked, UserAsked
from hands.core.front import FrontUnread, InFront, NoSessionInFront, SessionInFront, told
from hands.core.session import SessionId
from hands.core.trace import Span
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
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent, root
from hands.voice.player import Mark
from hands.voice.trigger import Edge
from hands.voice.turnstop import HoldDiscarded
from hands.voice.speech import Aloud, Narrated
from hands.voice.utterance import Uttered, Uttering

from test_narrator import heard as unasked
from hands.voice.tool import Result, Tool, tool

BRAIN = SessionId("brain-session")
PATIENCE_SECS = 2.0
ANSWERED = BrainAnswered("p1", None)
TAIL = "[hands] The Claude Code sessions running now: none of note."
# The span a routed request is compared with, its own kept apart in the rig's spans.
APART = Span("", "", None)
UNREAD = FrontUnread("not read in this test")
# A user's turn as the rig records it: the screen left unread, read in no time on the rig's clock.
ASKED = UserAsked("screen", "held key", UNREAD, 0.0)


def heard(words: str) -> str:
    """A user's turn as the brain is asked it at the desk, with the screen left unread."""
    return f"{words}\n\n{place.told('screen')}"


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
    # Each turn's teller of the permissions it holds.
    tellers: list[Callable[[Asked], None]] = field(default_factory=list[Callable[[Asked], None]])

    async def ask(self, text: str, asks: Callable[[Asked], None]) -> BrainAnswered:
        self.asked.append(text)
        self.tellers.append(asks)
        turn = asyncio.get_running_loop().create_future()
        self.turns.append(turn)
        return await asyncio.shield(turn)

    def permit(self, tool: str, input: dict[str, object]) -> Asked:
        """The turn in flight holding a permission its setup asks about, as the real brain does at its hook."""
        asked = Asked(Permission(tool, input), asyncio.get_running_loop().create_future())
        self.tellers[-1](asked)
        return asked

    def interrupt(self) -> None:
        self.interrupts += 1

    def end(self, answered: BrainAnswered = ANSWERED) -> None:
        self.turns[-1].set_result(answered)

    def fail(self, error: Exception) -> None:
        self.turns[-1].set_exception(error)


class Spoken(FrameProcessor):
    """What leaves the stage for TTS, played at once, as the speaker and the marks behind it would play it, unless held."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self.frames: list[Frame] = []
        # Marks not yet played, while `holding`: what is ahead of them is still being said.
        self.holding = False
        self.marks: list[Mark] = []
        # How many key releases have passed the stage, the ones it was told to throw away included.
        self.releases = 0
        # What is said of utterances: the frames that lead and close each, and the words between them, in order.
        self.uttering: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case Mark(played=played) if not self.holding:
                played()
            case Mark():
                self.marks.append(frame)
            case VADUserStoppedSpeakingFrame():
                self.releases += 1
                await self.push_frame(frame, direction)
            case LLMFullResponseStartFrame() | LLMFullResponseEndFrame() | LLMTextFrame() | TTSSpeakFrame() | InterruptionFrame():
                self.frames.append(frame)
                if isinstance(frame, LLMTextFrame | TTSSpeakFrame):
                    self.uttering.append(frame.text)
                await self.push_frame(frame, direction)
            case Uttering() | Uttered():
                # A Resumed is an Uttering, named as itself.
                self.uttering.append(type(frame).__name__)
                await self.push_frame(frame, direction)
            case _:
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
    # Each session the stage moved the focus to, with how many turns the brain had been asked as it moved.
    refocused: list[tuple[SessionId, int]]
    # What is in front on the Mac's screen as a user's turn is submitted: the last of these, read when it is asked.
    fronts: list[InFront]
    # The stage asking the brain each turn, as the daemon runs it beside the pipeline.
    asking: asyncio.Task[None]
    # Whether the user can see a screen as their turn is submitted: the last of these.
    modalities: list[Modality]
    # The edge that opened the gate's last turn: the last of these.
    edges: list[Edge]
    # Each running call's span, as hands' MCP server finds it.
    call_spans: CallSpans
    # Each pause the stage is waiting out, with the task waiting it out: its time runs only when the test says.
    pauses: list[tuple[asyncio.Future[None], asyncio.Task[object]]]
    context: LLMContext = field(default_factory=LLMContext)
    exchanges: int = 0
    # The span each request sent on carries, in the order they left.
    spans: list[Span] = field(default_factory=list[Span])

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
        route = self.stage.route(sent)
        # The span apart: where the request goes is what each test asserts, and what trace it is in is a few tests' own.
        self.spans.append(route.span)
        return exchange, replace(route, span=APART)

    def stream(self, exchange: str, *texts: str) -> None:
        for text in texts:
            self.stage.hear(Heard(exchange, TextDelta(0, text)))

    def calls(self, exchange: str, *calls: tuple[str, str]) -> None:
        """The reply making a whole block for each call, by id and name, as the stream carries it before the reply ends."""
        for index, (call, name) in enumerate(calls):
            self.stage.hear(Heard(exchange, BlockStarted(index, {"type": "tool_use", "id": call, "name": name, "input": {}})))
            self.stage.hear(Heard(exchange, BlockStopped(index)))

    async def release(self, frame: VADUserStoppedSpeakingFrame | None = None) -> None:
        """The key let go, as Whisper tells it on its way to the stage: on words to transcribe, unless told otherwise."""
        released = self.out.releases
        await self.worker.queue_frame(VADUserStoppedSpeakingFrame() if frame is None else frame)
        await self.until(lambda: self.out.releases > released)

    async def interrupt(self) -> None:
        await self.worker.queue_frame(InterruptionFrame())
        await self.until(lambda: "InterruptionFrame" in self.out.shape())

    async def elapse(self) -> None:
        """The pause the stage waits out runs its time, once it is waited out, and what waited it out runs to its end."""
        await self.until(lambda: any(not pause.done() for pause, _ in self.pauses))
        [(pause, waiting)] = [(pause, waiting) for pause, waiting in self.pauses if not pause.done()]
        pause.set_result(None)
        await asyncio.wait({waiting})


@pytest.fixture
async def rig() -> AsyncGenerator[Rig, None]:
    brain = FakeBrain()
    recorded: list[Entry] = []
    standing = [TAIL]
    now = [0.0]
    refocused: list[tuple[SessionId, int]] = []
    # A screen left unread by default: its note is nothing, so the user's words reach the brain as they were said.
    fronts: list[InFront] = [UNREAD]

    async def refocus(session: SessionId) -> None:
        refocused.append((session, len(brain.asked)))

    async def front() -> InFront:
        return fronts[-1]

    modalities: list[Modality] = ["screen"]
    edges: list[Edge] = ["held key"]
    call_spans = CallSpans()
    pauses: list[tuple[asyncio.Future[None], asyncio.Task[object]]] = []

    async def pause(_seconds: float) -> None:
        paused = asyncio.get_running_loop().create_future()
        waiting = asyncio.current_task()
        assert waiting is not None
        pauses.append((paused, waiting))
        await paused

    stage = BrainStage(brain, TOOLS, lambda: standing[-1], refocus, front, lambda: modalities[-1], lambda: edges[-1], recorded.append, call_spans, clock=lambda: now[0], pause=pause)
    out = Spoken()
    async with running([stage, out]) as run:
        # As the daemon runs it: a watch beside the pipeline.
        asking = asyncio.create_task(stage.ask_each())
        try:
            yield Rig(run.worker, stage, brain, out, recorded, run.errors, standing, now, refocused, fronts, asking, modalities, edges, call_spans, pauses)
        finally:
            asking.cancel()


def turns(recorded: Sequence[Entry]) -> list[WideEvent]:
    """The event each brain turn left, in the order they ended."""
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "voice.turn"]


def spoke(recorded: Sequence[Entry]) -> list[tuple[object, ...]]:
    """What each turn handed the speaker: its exchanges, words, readbacks, whether it was spoken over, who asked it, how
    long it waited behind the turn before it, and what it failed of."""
    facts = ("exchanges", "text", "readbacks", "interrupted", "asker", "queued_ms", "failed")
    return [tuple(turn.facts[fact] for fact in facts) for turn in turns(recorded)]


def interruptions(recorded: Sequence[Entry]) -> list[tuple[object, ...]]:
    """What each turn the user barged in on had running then, and whether the brain was told to stop at once."""
    return [(turn.facts["running"], turn.facts["stopped"]) for turn in turns(recorded) if turn.facts["interrupted"]]


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
    await rig.worker.queue_frame(Narrated("[hands] The Claude Code session api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), ()))
    await rig.until(lambda: rig.brain.asked == ["[hands] The Claude Code session api finished a turn."])
    exchange, _ = rig.request()
    rig.stream(exchange, "api opened pull request 68.")
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    assert rig.out.said() == ["api opened pull request 68."]
    assert ((exchange,), "api opened pull request 68.", (), False, HandsAsked(), 0.0, None) in spoke(rig.recorded)
    # Never in Pipecat's context, where it would ride along with whatever the user says next.
    assert rig.context.get_messages() == []


async def test_the_brain_s_turn_telling_a_finished_turn_is_part_of_its_utterance_and_sent_between_its_marks(rig: Rig) -> None:
    """The utterance is open from when hands heard the turn to when it was heard: the brain's turn telling it is a part
    of it, in its trace, and what it says is led and closed by the frames the output transport reads its fate off."""
    utterance = unasked()
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), (utterance,)))
    await rig.until(lambda: rig.brain.asked == ["[hands] api finished a turn."])
    exchange, _ = rig.request()
    rig.stream(exchange, "api opened pull request 68.")
    rig.brain.end()
    await rig.until(lambda: rig.out.uttering[-1:] == ["Uttered"])
    [turn] = turns(rig.recorded)
    assert (turn.trace_id, turn.parent_id) == (utterance.begun.span.trace_id, utterance.begun.span.span_id)
    assert rig.out.uttering == ["Uttering", "api opened pull request 68.", "Uttered"]


async def test_a_barge_in_the_telling_goes_on_through_leads_the_rest_of_it_on_again(rig: Rig) -> None:
    """None of the turn's requests had left, so the barge-in stops nothing: what the turn says is still the telling's."""
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), (unasked(),)))
    await rig.until(lambda: rig.brain.asked == ["[hands] api finished a turn."])
    await rig.interrupt()
    exchange, _ = rig.request()
    rig.stream(exchange, "api opened pull request 68.")
    rig.brain.end()
    await rig.until(lambda: rig.out.uttering[-1:] == ["Uttered"])
    assert rig.out.uttering == ["Uttering", "Resumed", "api opened pull request 68.", "Uttered"]
    assert rig.brain.interrupts == 0


async def test_a_barge_in_that_stops_the_telling_leads_nothing_more_on(rig: Rig) -> None:
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), (unasked(),)))
    await rig.until(lambda: rig.brain.asked == ["[hands] api finished a turn."])
    exchange, _ = rig.request()
    rig.stream(exchange, "api opened ")
    await rig.until(lambda: "api opened " in rig.out.uttering)
    await rig.interrupt()
    rig.brain.end()
    await rig.until(lambda: rig.out.uttering[-1:] == ["Uttered"])
    assert rig.out.uttering == ["Uttering", "api opened ", "Uttered"]


async def test_a_telling_the_brain_fails_fails_its_utterance_with_why(rig: Rig) -> None:
    utterance = unasked()
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), (utterance,)))
    await rig.until(lambda: len(rig.brain.asked) == 1)
    rig.brain.end(BrainAnswered("p1", "unknown: API Error: 500 overloaded"))
    await rig.until(lambda: rig.out.uttering[-1:] == ["Uttered"])
    [turn] = turns(rig.recorded)
    assert turn.error is not None and utterance.failure == turn.error


async def test_a_line_said_as_written_in_hands_lane_is_sent_between_its_marks(rig: Rig) -> None:
    await rig.worker.queue_frame(Aloud(TTSSpeakFrame("The session api is gone."), (unasked(),)))
    await rig.until(lambda: rig.out.uttering[-1:] == ["Uttered"])
    assert rig.out.uttering == ["Uttering", "The session api is gone.", "Uttered"]
    assert rig.out.shape() == ["TTSSpeakFrame"]


async def test_a_narration_moves_the_focus_as_the_brain_takes_it_ahead_of_what_the_user_says_meanwhile(rig: Rig) -> None:
    """The user's next words are taken as said to the session told of: so the focus moves before the brain is asked to
    tell it, and words the user speaks while it is said are asked after it."""
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), ()))
    await rig.until(lambda: rig.brain.asked == ["[hands] api finished a turn."])
    exchange, _ = rig.request()
    rig.stream(exchange, "api opened pull request 68.")
    rig.context.add_message({"role": "user", "content": "push it"})
    await rig.worker.queue_frame(LLMContextFrame(rig.context))
    await rig.until(lambda: "api opened pull request 68." in rig.out.said())
    rig.brain.end()
    await rig.until(lambda: rig.brain.asked[1:] == [heard("push it")])
    assert rig.refocused == [(SessionId("api"), 0)]


async def test_the_users_turn_goes_ahead_of_a_narration_waiting_for_the_brain(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "what is running?"})
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), ()))
    rig.context.add_message({"role": "user", "content": "and the backlog?"})
    await rig.worker.queue_frame(LLMContextFrame(rig.context))
    # Frames pass the stage in order, so once this is out, both before it are waiting in the stage.
    await rig.worker.queue_frame(TTSSpeakFrame("marker"))
    await rig.until(lambda: "marker" in rig.out.said())
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    # Words the user spoke before the telling was taken are not taken as said to the session it tells of.
    assert rig.refocused == []
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 3)
    assert rig.brain.asked == [heard("what is running?"), heard("and the backlog?"), "[hands] api finished a turn."]
    assert rig.refocused == [(SessionId("api"), 2)]


async def test_a_narration_records_how_long_it_waited_behind_the_users_turn(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "what is running?"})
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), ()))
    await rig.worker.queue_frame(TTSSpeakFrame("marker"))
    await rig.until(lambda: "marker" in rig.out.said())
    rig.now[0] = 2.5
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    exchange, _ = rig.request()
    rig.stream(exchange, "api finished.")
    rig.brain.end()
    await rig.until(lambda: any(turn.facts["asker"] == HandsAsked() for turn in turns(rig.recorded)))
    assert ((exchange,), "api finished.", (), False, HandsAsked(), 2500.0, None) in spoke(rig.recorded)


async def test_what_hands_says_as_written_is_heard_after_the_narration_ahead_of_it(rig: Rig) -> None:
    """A session's end is said after its last turn, though that turn waits for the brain and the end needs no model."""
    await rig.say({"role": "user", "content": "what is running?"})
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), ()))
    await rig.worker.queue_frame(Aloud(TTSSpeakFrame("The session api is gone."), ()))
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
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", unsaid, SessionId("api"), ()))
    await rig.until(lambda: len(rig.brain.asked) == 1)
    rig.brain.end(BrainAnswered("p1", "unknown: API Error: 500 overloaded"))
    await rig.until(lambda: len(rig.errors) == 1)
    assert rig.out.said() == [unsaid]


async def test_a_turns_event_says_what_was_typed_into_the_brain(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "what is api doing?"})
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    [turn] = turns(rig.recorded)
    assert {fact: turn.facts[fact] for fact in ("asked", "running", "stopped")} == {"asked": rig.brain.asked[0], "running": (), "stopped": False}


async def test_a_users_turn_the_brain_fails_has_nothing_said_for_it(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "hello"})
    rig.brain.end(BrainAnswered("p1", "unknown: API Error: 500 overloaded"))
    await rig.until(lambda: len(rig.errors) == 1)
    assert rig.out.said() == []


async def test_a_turn_goes_to_the_brain_and_its_words_come_off_the_wire(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "what is running?"})
    assert rig.brain.asked == [heard("what is running?")]
    exchange, route = rig.request()
    assert route == Send((Tail(TAIL),), refusal="final", span=APART)
    rig.stream(exchange, "Two sessions ", "are running.")
    await rig.until(lambda: len(rig.out.said()) == 2)
    rig.brain.end()
    await rig.until(lambda: "LLMFullResponseEndFrame" in rig.out.shape())
    assert rig.out.shape() == ["LLMFullResponseStartFrame", "LLMTextFrame", "LLMTextFrame", "LLMFullResponseEndFrame"]
    assert rig.out.said() == ["Two sessions ", "are running."]
    # The audit log ties what was spoken to the exchange on the wire it came from.
    assert ((exchange,), "Two sessions are running.", (), False, ASKED, 0.0, None) in spoke(rig.recorded)


async def test_a_users_turn_carries_what_was_in_front_as_it_was_submitted_and_its_event_says_what_was_read(rig: Rig) -> None:
    front = SessionInFront("iTerm2", SessionId("s1"), "hands, docs")
    rig.fronts.append(front)
    await rig.say({"role": "user", "content": "what am I looking at?"})
    assert rig.brain.asked == [heard(f"what am I looking at?\n\n{told(front)}")]
    exchange, _ = rig.request()
    rig.stream(exchange, "The docs session.")
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    assert ((exchange,), "The docs session.", (), False, UserAsked("screen", "held key", front, 0.0), 0.0, None) in spoke(rig.recorded)
    # Read again for the next turn, not kept: the screen has changed since.
    rig.fronts.append(NoSessionInFront("Safari"))
    await rig.say({"role": "user", "content": "and now?"})
    assert rig.brain.asked[-1] == heard(f"and now?\n\n{told(NoSessionInFront('Safari'))}")


async def test_words_that_wait_behind_a_turn_carry_what_was_in_front_as_they_arrived_and_the_wait_leaves_out_the_read(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "what is running?"})
    spoken_at = SessionInFront("iTerm2", SessionId("s1"), "hands, docs")
    rig.fronts.append(spoken_at)
    rig.context.add_message({"role": "user", "content": "what's this one doing?"})
    await rig.worker.queue_frame(LLMContextFrame(rig.context))
    await rig.worker.queue_frame(TTSSpeakFrame("marker"))
    await rig.until(lambda: "marker" in rig.out.said())
    # The user looked elsewhere while the first turn was still being answered.
    rig.fronts.append(NoSessionInFront("Safari"))
    rig.now[0] = 2.5
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    assert rig.brain.asked[-1] == heard(f"what's this one doing?\n\n{told(spoken_at)}")
    exchange, _ = rig.request()
    rig.stream(exchange, "Writing docs.")
    rig.brain.end()
    await rig.until(lambda: len(turns(rig.recorded)) == 2)
    assert ((exchange,), "Writing docs.", (), False, UserAsked("screen", "held key", spoken_at, 0.0), 2500.0, None) in spoke(rig.recorded)


async def test_a_users_turn_carries_whether_they_could_see_a_screen_as_their_words_arrived(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "what is running?"})
    rig.context.add_message({"role": "user", "content": "go audio only"})
    await rig.worker.queue_frame(LLMContextFrame(rig.context))
    await rig.worker.queue_frame(TTSSpeakFrame("marker"))
    await rig.until(lambda: "marker" in rig.out.said())
    # Switched while the words waited behind the first turn: they carry what held as they arrived.
    rig.modalities.append("audio-only")
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    assert rig.brain.asked[-1] == heard("go audio only")
    rig.brain.end()
    await rig.say({"role": "user", "content": "what now?"})
    assert rig.brain.asked[-1] == f"what now?\n\n{place.told('audio-only')}"
    exchange, _ = rig.request()
    rig.stream(exchange, "Nothing.")
    rig.brain.end()
    await rig.until(lambda: len(turns(rig.recorded)) == 3)
    assert ((exchange,), "Nothing.", (), False, UserAsked("audio-only", "held key", UNREAD, 0.0), 0.0, None) in spoke(rig.recorded)


async def test_a_users_turn_carries_the_edge_that_opened_it_though_another_opens_before_its_words_arrive(rig: Rig) -> None:
    await rig.release()
    # The phone's button pressed while the desk's words are still transcribed.
    rig.edges.append("phone button")
    await rig.say({"role": "user", "content": "what is running?"})
    exchange, _ = rig.request()
    rig.stream(exchange, "Nothing.")
    rig.brain.end()
    await rig.until(lambda: len(turns(rig.recorded)) == 1)
    assert spoke(rig.recorded)[0][4] == UserAsked("screen", "held key", UNREAD, 0.0)


async def test_a_turn_hands_narrates_is_not_read_against_the_screen(rig: Rig) -> None:
    rig.fronts.append(SessionInFront("iTerm2", SessionId("s1"), "hands, docs"))
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), ()))
    await rig.until(lambda: len(rig.brain.asked) == 1)
    assert rig.brain.asked == ["[hands] api finished a turn."]


async def test_a_turn_is_one_event_saying_how_long_the_user_waited_and_where_the_time_went(rig: Rig) -> None:
    # Let go of the key, transcribed 0.4 s on, and written to the brain at once.
    rig.now[0] = 1000.0
    await rig.release()
    rig.now[0] = 1000.4
    await rig.say({"role": "user", "content": "what is api doing?"})
    # The model asked, its answer's head heard, and a call made whole and run.
    rig.now[0] = 1000.5
    first, _ = rig.request()
    rig.now[0] = 1001.0
    rig.stage.hear(Answering(first, 200, None))
    rig.now[0] = 1001.2
    rig.calls(first, ("t1", "mcp__hands__read_session"))
    # The span hands' MCP server runs the call inside, while it runs.
    running = rig.call_spans.span("t1")
    # The call's result going back is the end of its run, and the second round trip's start.
    rig.now[0] = 1001.5
    second, _ = rig.request(answering("mcp__hands__read_session", {"steps": []}))
    rig.now[0] = 1001.9
    rig.stage.hear(Answering(second, 200, None))
    rig.now[0] = 1002.0
    rig.stream(second, "api is running its tests.")
    rig.now[0] = 1002.3
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    [turn] = turns(rig.recorded)
    events = [entry for entry in rig.recorded if isinstance(entry, WideEvent)]
    assert (turn.outcome, turn.counts) == ("ok", {"round_trips": 2, "tools": 1})
    # Two seconds from letting go to the first word: 0.4 transcribing, none waiting behind another turn, and the rest
    # the model's and its tool's, as the parts show.
    assert {fact: turn.facts[fact] for fact in ("asker", "waited_ms", "transcribed_ms", "queued_ms")} == {"asker": ASKED, "waited_ms": 2000.0, "transcribed_ms": 400.0, "queued_ms": 0.0}
    [call] = [event for event in events if event is not turn]
    assert (call.event, call.duration_ms, call.outcome, call.facts) == ("tool.call", 300.0, "ok", {"call": "t1", "tool": "mcp__hands__read_session"})
    # The call's run is the child of the very span the call is emitted as, and the turn's end forgets it.
    assert running is not None and (call.trace_id, call.span_id, call.parent_id) == (running.trace_id, running.span_id, running.parent_id)
    assert rig.call_spans.span("t1") is None
    # Each round trip is the proxy's record of its request, which carries a span of its own inside the turn's.
    assert [(span.trace_id, span.parent_id) for span in rig.spans] == [(turn.trace_id, turn.span_id)] * 2
    assert len({span.span_id for span in rig.spans}) == 2
    # One trace: each part is the turn's child.
    assert {(event.trace_id, event.parent_id) for event in events if event is not turn} == {(turn.trace_id, turn.span_id)}


async def test_a_turn_spoken_over_before_its_answer_and_its_call_came_back_counts_them_cancelled(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "what is api doing?"})
    exchange, _ = rig.request()
    rig.stream(exchange, "Let me look.")
    rig.calls(exchange, ("t1", "mcp__hands__read_session"))
    await rig.interrupt()
    # The model asked again, and the user spoke over the turn before its answer came.
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    [turn] = turns(rig.recorded)
    [call] = [entry for entry in rig.recorded if isinstance(entry, WideEvent) and entry.event == "tool.call"]
    assert call.outcome == "cancelled" and turn.facts["interrupted"] is True
    # A turn nobody asked aloud has no transcription to time.
    assert turn.facts["transcribed_ms"] is None


async def test_a_turn_stopped_mid_way_still_says_what_it_did(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "what is api doing?"})
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__read_session"))
    # hands stopping while the brain is still answering.
    rig.asking.cancel()
    await rig.until(lambda: bool(turns(rig.recorded)))
    [turn] = turns(rig.recorded)
    assert (turn.outcome, turn.counts, turn.facts["exchanges"]) == ("cancelled", {"round_trips": 1, "tools": 1}, (exchange,))
    [call] = [entry for entry in rig.recorded if isinstance(entry, WideEvent) and entry.event == "tool.call"]
    assert (call.outcome, call.parent_id) == ("cancelled", turn.span_id)


async def test_words_that_waited_behind_a_turn_together_are_timed_from_the_last_of_them(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "what is api doing?"})
    # Two holds said while the brain answers, each let go of and transcribed in its turn.
    for released, arrived, words in ((1000.0, 1000.4, "and auth?"), (1002.0, 1002.4, "and web?")):
        rig.now[0] = released
        await rig.release()
        rig.now[0] = arrived
        rig.context.add_message({"role": "user", "content": words})
        await rig.worker.queue_frame(LLMContextFrame(rig.context))
        # Frames pass the stage in order, so once this is out, the words before it are waiting in the stage.
        await rig.worker.queue_frame(TTSSpeakFrame(words))
        await rig.until(lambda: words in rig.out.said())
    rig.now[0] = 1003.0
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    exchange, _ = rig.request()
    rig.now[0] = 1004.0
    rig.stream(exchange, "Both are idle.")
    rig.brain.end()
    await rig.until(lambda: len(turns(rig.recorded)) == 2)
    assert rig.brain.asked[1] == heard("and auth?\n\nand web?")
    # Two seconds from the last let-go: 0.4 transcribing it, 0.6 behind the turn before, and 1.0 to the first word.
    timed = {fact: turns(rig.recorded)[1].facts[fact] for fact in ("waited_ms", "transcribed_ms", "queued_ms")}
    assert timed == {"waited_ms": 2000.0, "transcribed_ms": 400.0, "queued_ms": 600.0}


async def test_a_hold_thrown_away_is_no_release_to_time_a_wait_from(rig: Rig) -> None:
    rig.now[0] = 1000.0
    await rig.release()
    rig.now[0] = 1003.0
    await rig.release(HoldDiscarded())
    rig.now[0] = 1003.5
    await rig.say({"role": "user", "content": "hello"})
    exchange, _ = rig.request()
    rig.now[0] = 1004.0
    rig.stream(exchange, "Hi.")
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    [turn] = turns(rig.recorded)
    assert (turn.facts["waited_ms"], turn.facts["transcribed_ms"]) == (4000.0, 3500.0)


async def test_a_turn_the_brain_failed_is_a_failed_event_saying_why(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "hello"})
    rig.brain.end(BrainAnswered("p1", "server_error: API Error: 529 Overloaded"))
    await rig.until(lambda: bool(turns(rig.recorded)))
    [turn] = turns(rig.recorded)
    assert (turn.outcome, turn.error) == ("failed", "the brain's turn ended in error: server_error: API Error: 529 Overloaded")
    # Nothing was said, so there is no wait to a first word: absent, not zero.
    assert (turn.facts["waited_ms"], turn.counts) == (None, {"round_trips": 0, "tools": 0})


async def test_each_request_of_a_turn_carries_how_the_sessions_stand_as_it_leaves(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "start the tests in auth"})
    exchange, first = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__stage_draft"))
    rig.standing.append("[hands] The Claude Code sessions running now: auth, working.")
    _, second = rig.request(answering("mcp__hands__stage_draft", {"readback": "staged"}))
    # Composed as each request leaves, never kept from the one before.
    assert (first, second) == (Send((Tail(TAIL),), refusal="final", span=APART), Send((Tail("[hands] The Claude Code sessions running now: auth, working."),), refusal="final", span=APART))
    rig.brain.end()


async def test_only_the_brains_own_main_turns_are_spoken(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "hello"})
    others = [rig.request(kind=Fork()), rig.request(kind=Unknown("a messages request with no tools")), rig.request(session=SessionId("a summary"))]
    # Every request of the brain's is final, said or not; another session's goes as it came (hands-wire-zi2).
    assert [route for _, route in others] == [Send(refusal="final", span=APART), Send(refusal="final", span=APART), Send(span=APART)]
    # The brain's are its turn's, in one trace under one parent; another session's is the root of a trace of its own.
    fork, unknown, summary = rig.spans
    assert (fork.trace_id, fork.parent_id) == (unknown.trace_id, unknown.parent_id) and fork.parent_id is not None
    assert summary.parent_id is None and summary.trace_id != fork.trace_id
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
    assert route == Send(refusal="final", span=APART)
    # Made for no turn: the root of a trace of its own.
    assert rig.spans[0].parent_id is None


async def test_the_brain_hears_what_the_context_gained_and_never_its_own_words_again(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "[hands] hands has just started."}, {"role": "user", "content": "anything new?"})
    rig.brain.end()
    await rig.until(lambda: "LLMFullResponseEndFrame" in rig.out.shape())
    await rig.say({"role": "assistant", "content": "Nothing new."}, {"role": "user", "content": "[hands] a session is waiting."})
    assert rig.brain.asked == [heard("[hands] hands has just started.\n\nanything new?"), heard("[hands] a session is waiting.")]


async def test_a_turn_is_written_only_once_the_one_before_it_has_ended(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "first"})
    await rig.interrupt()
    rig.context.add_message({"role": "user", "content": "second"})
    await rig.worker.queue_frame(LLMContextFrame(rig.context))
    # The stage has taken the second turn and is waiting on the first, which the brain has not ended.
    await asyncio.sleep(0.1)
    assert rig.brain.asked == [heard("first")]
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    assert rig.brain.asked[1] == heard("second")


async def test_stay_silent_holds_the_next_request_so_nothing_follows_it(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "(to a colleague) back in five"})
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__stay_silent"))
    _, route = rig.request(answering("mcp__hands__stay_silent", {"silent": True}))
    assert route == Hold(SILENT, APART)
    # Held, it is still the turn's round trip, in the turn's trace beside the one before it.
    sent, held = rig.spans
    assert (held.trace_id, held.parent_id) == (sent.trace_id, sent.parent_id) and held.span_id != sent.span_id
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
    assert route == Send((Tail(TAIL),), refusal="final", span=APART)
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
    # What the wire carries after the barge-in is dropped, and a request that raced the stop is not asked either.
    rig.stream(exchange, "second, ", "third.")
    _, route = rig.request()
    assert route == Hold(INTERRUPTED, APART)
    # A second press on the same turn does not tell the brain twice.
    await rig.worker.queue_frame(InterruptionFrame())
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    assert rig.out.said() == ["First, "]
    assert rig.brain.interrupts == 1
    assert ((exchange,), "First, ", (), True, ASKED, 0.0, None) in spoke(rig.recorded)
    assert interruptions(rig.recorded) == [((), True)]


async def test_a_barge_in_while_a_draft_lands_lets_it_finish_and_speaks_its_readback(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "tell the api session to add tests"})
    exchange, _ = rig.request()
    rig.stream(exchange, "Staging it.")
    # The call is open, and so running under Claude Code, while the reply is still streaming.
    rig.calls(exchange, ("t1", "mcp__hands__stage_draft"))
    await rig.interrupt()
    # Stopped by the harness, the draft would land and be written into history as refused; it is let run instead.
    assert rig.brain.interrupts == 0
    _, route = rig.request(answering("mcp__hands__stage_draft", {"readback": "staged for api: add tests"}))
    assert route == Hold(INTERRUPTED, APART)
    rig.brain.end()
    # Said once the turn is over, after anything the model had begun to say.
    await rig.until(lambda: rig.out.said()[-1:] == ["staged for api: add tests"])
    await rig.until(lambda: bool(turns(rig.recorded)))
    assert ((exchange,), "Staging it.", ("staged for api: add tests",), True, ASKED, 0.0, None) in spoke(rig.recorded)
    assert interruptions(rig.recorded) == [(("mcp__hands__stage_draft",), False)]


async def test_a_readback_a_call_hands_hands_ends_the_turn_and_is_said_by_hands_as_written(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "make it add tests too"})
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__amend_draft"))
    _, route = rig.request(answering("mcp__hands__amend_draft", {"says": "amended for api: add tests too"}))
    assert route == Hold(SILENT, APART)
    rig.brain.end()
    await rig.until(lambda: rig.out.said() == ["amended for api: add tests too"])
    # hands-readback-ddk: said after the turn's reply has ended, where it is a turn of its own, over once it is said.
    await rig.until(lambda: rig.out.shape() == ["LLMFullResponseStartFrame", "LLMFullResponseEndFrame", "TTSSpeakFrame"])


async def test_a_barge_in_while_a_draft_hands_reads_back_lands_lets_it_finish_and_its_readback_is_said(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "make it add tests too"})
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__amend_draft"))
    await rig.interrupt()
    assert rig.brain.interrupts == 0
    _, route = rig.request(answering("mcp__hands__amend_draft", {"says": "amended for api: add tests too"}))
    assert route == Hold(INTERRUPTED, APART)
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    # Said after the barge-in, never cut off by it: the draft changed, so the user hears how.
    assert ((exchange,), "", ("amended for api: add tests too",), True, ASKED, 0.0, None) in spoke(rig.recorded)


async def test_a_refused_call_to_a_silence_tool_is_the_models_to_answer(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "make it add tests too"})
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__amend_draft"))
    refused, route = rig.request(answering("mcp__hands__amend_draft", {"error": "the draft text is empty"}))
    # Nothing was said and nothing changed: the model is asked to go on, to retry or to say what went wrong.
    assert route == Send((Tail(TAIL),), refusal="final", span=APART)
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
    assert route == Send((Tail(TAIL),), refusal="final", span=APART)
    rig.brain.end()


async def test_a_barge_in_while_a_reading_tool_runs_stops_the_brain_at_once(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "what did the api session do?"})
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__read_session"))
    await rig.interrupt()
    assert rig.brain.interrupts == 1
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    assert interruptions(rig.recorded) == [(("mcp__hands__read_session",), True)]


async def test_a_barge_in_with_no_turn_in_flight_tells_the_brain_nothing(rig: Rig) -> None:
    await rig.interrupt()
    assert rig.brain.interrupts == 0
    assert turns(rig.recorded) == []


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
    assert rig.brain.asked == [heard("first"), heard("second")]


async def test_notes_that_came_in_one_ask_are_not_asked_again_empty(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "first"})
    for note in ("[hands] api finished.", "[hands] web finished."):
        rig.context.add_message({"role": "user", "content": note})
        await rig.worker.queue_frame(LLMContextFrame(rig.context))
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    rig.brain.end()
    await asyncio.sleep(0.1)
    assert rig.brain.asked == [heard("first"), heard("[hands] api finished.\n\n[hands] web finished.")]


async def test_a_turn_whose_replies_carried_nothing_is_said_as_the_models_empty_reply(rig: Rig) -> None:
    unsaid = "api finished a turn, and I could not tell it."
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", unsaid, SessionId("api"), ()))
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
    [turn] = turns(rig.recorded)
    assert turn.facts["failed"] == ModelReplyEmpty()
    # An empty reply is whole and in the brain's history: its next turn is not told the API broke one off.
    await rig.say({"role": "user", "content": "hello?"})
    assert rig.brain.asked[-1] == heard("hello?")


async def test_a_turn_that_called_a_tool_and_then_answered_nothing_is_no_empty_reply(rig: Rig) -> None:
    """Claude often ends a turn with an empty reply to a tool's result: the call already did what the turn was for."""
    await rig.say({"role": "user", "content": "what is api doing?"})
    exchange, _ = rig.request()
    rig.stream(exchange, "Let me look.")
    rig.calls(exchange, ("t1", "mcp__hands__read_session"))
    rig.request(answering("mcp__hands__read_session", {"steps": []}))
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    await asyncio.sleep(0.1)
    assert rig.errors == []


async def test_a_turn_ending_on_a_call_that_lands_is_no_empty_reply(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "tell the api session to add tests"})
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__stage_draft"))
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
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
    return Exchanged(exchange, BRAIN, MainTurn(None), "POST", "/v1/messages", 2, (), 0.0, 0.0, Unreached("ClientConnectorError: no route", 0.0), True, root())


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
    [turn] = turns(rig.recorded)
    assert turn.facts["failed"] == fact


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
    assert ((exchange,), "1. Lighthouses stand on rocky coasts and h", (), False, ASKED, 0.0, ModelFailed(ErrorCategory.UNKNOWN)) in spoke(rig.recorded)
    # The broken reply is not in the brain's history, so its next turn tells it what the user heard, and only that one.
    await rig.say({"role": "user", "content": "what were you saying?"})
    rig.brain.end()
    await rig.say({"role": "user", "content": "thanks"})
    assert rig.brain.asked[1:] == [
        heard(
            '[hands] Your last turn was broken off. The user heard you say "1. Lighthouses stand on rocky coasts and h", then that it failed. '
            "Say nothing about this unless the user asks.\n\nwhat were you saying?"
        ),
        heard("thanks"),
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
    note = '[hands] Your last turn was broken off. The user heard you say "Lighthouses stand", then that it failed. Say nothing about this unless the user asks.'
    assert rig.brain.asked[1:] == [heard(f"{note}\n\nwhat were you saying?"), heard(f"{note}\n\nagain")]
    rig.brain.end()


async def test_a_turn_that_breaks_after_the_brain_spoke_tells_the_next_where_the_user_stopped_hearing_it(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "tell me about lighthouses"})
    rig.stream(rig.request()[0], "Lighthouses stand")
    await rig.until(lambda: len(rig.out.said()) == 1)
    # Broken in hands' own work, after the brain took the turn: its asker hears the failure, not an answer.
    rig.brain.fail(RuntimeError("no voice"))
    await rig.until(lambda: len(rig.errors) == 1)
    await rig.say({"role": "user", "content": "what were you saying?"})
    note = '[hands] Your last turn was broken off. The user heard you say "Lighthouses stand", then that it failed. Say nothing about this unless the user asks.'
    assert rig.brain.asked[1:] == [heard(f"{note}\n\nwhat were you saying?")]
    rig.brain.end()


async def test_a_turn_the_brain_never_took_is_reported_as_the_model_stages_error_and_the_next_is_asked(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "hello"})
    # The brain is still running, so no watch reports it: the stage does.
    rig.brain.fail(Untaken("the brain did not take the turn typed into it in 30s"))
    await rig.until(lambda: len(rig.errors) == 1)
    assert rig.errors[0].processor is rig.stage
    assert "did not take the turn" in rig.errors[0].error
    await rig.say({"role": "user", "content": "again"})
    await rig.until(lambda: rig.brain.asked == [heard("hello"), heard("again")])
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
    assert route == Send((Tail(TAIL),), refusal="final", span=APART)
    rig.brain.end()


async def test_a_draft_that_failed_under_a_held_request_says_why(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "tell the api session to add tests"})
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__stage_draft"))
    await rig.interrupt()
    _, route = rig.request(answering("mcp__hands__stage_draft", {"error": "there is no session api"}))
    assert route == Hold(INTERRUPTED, APART)
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
    await rig.until(lambda: bool(turns(rig.recorded)))
    await asyncio.sleep(0.1)
    assert rig.errors == []


async def test_a_barge_in_while_a_drafts_input_still_streams_stops_the_brain_before_it_runs(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "tell the api session to add tests"})
    exchange, _ = rig.request()
    # The block is open and its input still arriving: Claude Code has not run the call, and now never will.
    rig.stage.hear(Heard(exchange, BlockStarted(0, {"type": "tool_use", "id": "t1", "name": "mcp__hands__stage_draft", "input": {}})))
    await rig.interrupt()
    assert rig.brain.interrupts == 1
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    assert interruptions(rig.recorded) == [((), True)]


async def test_a_barge_in_before_the_brain_has_sent_the_turn_stops_nothing_and_the_turn_is_answered(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "are you listening?"})
    # Still waiting for the input: nothing of it has left, so there is nothing to stop, and the user's words follow it.
    await rig.interrupt()
    assert rig.brain.interrupts == 0
    # The reply goes on through the barge-in, which ended it for whatever writes it: started again, so it is written whole.
    await rig.until(lambda: rig.out.shape() == ["LLMFullResponseStartFrame", "InterruptionFrame", "LLMFullResponseStartFrame"])
    _, route = rig.request()
    assert route == Send((Tail(TAIL),), refusal="final", span=APART)
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    assert interruptions(rig.recorded) == []


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
    await rig.until(lambda: bool(turns(rig.recorded)))
    assert "".join(rig.out.said()) == "Here is what I ran:\n```bash\nnpm test\n```\n\nAll 42 tests passed."


async def test_a_permission_the_brain_holds_is_asked_after_its_words_and_the_users_yes_lets_its_turn_go_on(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "make the notes say hello"})
    exchange, _ = rig.request()
    rig.stream(exchange, "I'll write it.")
    asked = rig.brain.permit("Write", {"file_path": "/Users/bmf/notes.txt", "content": "hello"})
    await rig.until(lambda: "May I use Write on notes.txt? Say yes to allow it." in rig.out.said())
    assert rig.out.said() == ["I'll write it.", "May I use Write on notes.txt? Say yes to allow it."]
    # The user starting to answer stops what is playing, and stops nothing of the turn: it waits on their answer.
    await rig.interrupt()
    rig.context.add_message({"role": "user", "content": "Yes, go ahead."})
    await rig.worker.queue_frame(LLMContextFrame(rig.context))
    await rig.until(lambda: not asked.open)
    assert asked.decision.result() == Allow()
    assert rig.brain.interrupts == 0
    # The turn goes on, and what it says next is heard.
    after, route = rig.request()
    assert isinstance(route, Send) and route.refusal == "final"
    rig.stream(after, "Done, it says hello.")
    await rig.until(lambda: "Done, it says hello." in rig.out.said())
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    # The answer was the turn's, never a turn of its own.
    assert rig.brain.asked == [heard("make the notes say hello")]


async def test_anything_but_a_plain_yes_refuses_the_permission_and_hands_the_brain_the_users_words(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "make the notes say hello"})
    exchange, _ = rig.request()
    rig.stream(exchange, "I'll write it.")
    asked = rig.brain.permit("Edit", {"file_path": "/Users/bmf/notes.txt"})
    await rig.until(lambda: "May I use Edit on notes.txt? Say yes to allow it." in rig.out.said())
    rig.context.add_message({"role": "user", "content": "No, put it in the to-do list instead."})
    await rig.worker.queue_frame(LLMContextFrame(rig.context))
    await rig.until(lambda: not asked.open)
    match asked.decision.result():
        case Deny(message=message):
            assert '"No, put it in the to-do list instead."' in message
        case other:
            pytest.fail(f"a no allowed the edit: {other}")
    # Their next words, once nothing is asked, are a turn of their own.
    rig.brain.end()
    await rig.say({"role": "user", "content": "thanks"})
    assert rig.brain.asked == [heard("make the notes say hello"), heard("thanks")]


async def refuse_the_edit(rig: Rig) -> str:
    """A turn whose permission to edit the notes the user hears asked and refuses; the request that carries the refusal."""
    await rig.say({"role": "user", "content": "make the notes say hello"})
    exchange, _ = rig.request()
    rig.stream(exchange, "I'll write it.")
    asked = rig.brain.permit("Edit", {"file_path": "/Users/bmf/notes.txt"})
    await rig.until(lambda: "May I use Edit on notes.txt? Say yes to allow it." in rig.out.said())
    rig.context.add_message({"role": "user", "content": "no"})
    await rig.worker.queue_frame(LLMContextFrame(rig.context))
    await rig.until(lambda: not asked.open)
    after, _ = rig.request()
    return after


async def test_a_refusal_the_brain_says_nothing_after_is_said_by_hands_so_silence_never_passes_for_it_done(rig: Rig) -> None:
    after = await refuse_the_edit(rig)
    rig.stream(after, "  ")
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    assert rig.out.said()[-1] == "I did not use Edit on notes.txt, so that is not done."
    assert acknowledgements(rig) == ["I did not use Edit on notes.txt, so that is not done."]
    [turn] = turns(rig.recorded)
    assert turn.facts["refusals"] == ("I did not use Edit on notes.txt, so that is not done.",)


async def test_a_refusal_the_brain_speaks_to_is_not_said_again_by_hands(rig: Rig) -> None:
    after = await refuse_the_edit(rig)
    rig.stream(after, "Okay, I left the notes alone.")
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    assert rig.out.said()[-1] == "Okay, I left the notes alone."
    assert acknowledgements(rig) == []
    [turn] = turns(rig.recorded)
    assert turn.facts["refusals"] == ()


async def test_a_permission_held_after_the_user_spoke_over_its_turn_is_refused_and_never_asked(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "make the notes say hello"})
    exchange, _ = rig.request()
    rig.stream(exchange, "I'll write it.")
    await rig.until(lambda: "I'll write it." in rig.out.said())
    await rig.interrupt()
    assert rig.brain.interrupts == 1
    asked = rig.brain.permit("Write", {"file_path": "/Users/bmf/notes.txt"})
    assert asked.decision.result() == Deny(SPOKEN_OVER)
    assert rig.out.said() == ["I'll write it."]


async def test_words_said_before_a_permission_is_asked_are_a_turn_of_their_own_and_never_part_of_its_answer(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "make the notes say hello"})
    # Said before the turn's first request left: it waits as the next turn.
    rig.context.add_message({"role": "user", "content": "also check the logs"})
    await rig.worker.queue_frame(LLMContextFrame(rig.context))
    await asyncio.sleep(0.1)
    exchange, _ = rig.request()
    rig.stream(exchange, "I'll write it.")
    asked = rig.brain.permit("Write", {"file_path": "/Users/bmf/notes.txt"})
    await rig.until(lambda: "May I use Write on notes.txt? Say yes to allow it." in rig.out.said())
    rig.context.add_message({"role": "user", "content": "yes"})
    await rig.worker.queue_frame(LLMContextFrame(rig.context))
    await rig.until(lambda: not asked.open)
    assert asked.decision.result() == Allow()
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    assert rig.brain.asked == [heard("make the notes say hello"), heard("also check the logs")]


async def test_a_permission_to_run_a_command_is_asked_with_the_command_it_would_run(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "clean up the old branches"})
    exchange, _ = rig.request()
    rig.stream(exchange, "On it.")
    rig.brain.permit("Bash", {"command": "git branch -D old", "description": "Delete the old branch"})
    await rig.until(lambda: "May I use Bash to run git branch -D old? Say yes to allow it." in rig.out.said())
    rig.brain.end()


async def test_a_command_is_asked_whole_however_long_and_a_fetch_with_its_address(rig: Rig) -> None:
    command = "make build && " * 40 + "rm -rf ~/x"
    await rig.say({"role": "user", "content": "build it"})
    exchange, _ = rig.request()
    rig.stream(exchange, "On it.")
    run = rig.brain.permit("Bash", {"command": command})
    await rig.until(lambda: f"May I use Bash to run {command}? Say yes to allow it." in rig.out.said())
    run.settle(Deny("no"))
    rig.brain.permit("WebFetch", {"url": "https://example.com/x", "prompt": "read it"})
    await rig.until(lambda: "May I use WebFetch on https://example.com/x? Say yes to allow it." in rig.out.said())
    rig.brain.end()


async def test_a_yes_said_before_the_question_has_played_allows_nothing_and_stops_the_turn(rig: Rig) -> None:
    rig.out.holding = True
    await rig.say({"role": "user", "content": "clear out the build directory"})
    exchange, _ = rig.request()
    rig.stream(exchange, "I'll clear out the build directory.")
    rig.calls(exchange, ("c1", "Bash"))
    asked = rig.brain.permit("Bash", {"command": "rm -rf build"})
    await rig.until(lambda: "May I use Bash to run rm -rf build? Say yes to allow it." in rig.out.said())
    # Said over what is still playing: the question was never heard to its end, so nothing waits on the user's words.
    await rig.interrupt()
    assert asked.decision.result() == Deny(SPOKEN_OVER)
    assert rig.brain.interrupts == 1
    rig.context.add_message({"role": "user", "content": "yeah"})
    await rig.worker.queue_frame(LLMContextFrame(rig.context))
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    assert rig.brain.asked == [heard("clear out the build directory"), heard("yeah")]


async def test_permissions_held_at_once_are_asked_one_at_a_time_and_each_answer_goes_to_the_one_heard(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "build both"})
    exchange, _ = rig.request()
    rig.stream(exchange, "Both, then.")
    first = rig.brain.permit("Bash", {"command": "make a"})
    second = rig.brain.permit("Bash", {"command": "make b"})
    await rig.until(lambda: "May I use Bash to run make a? Say yes to allow it." in rig.out.said())
    await asyncio.sleep(0.1)
    assert "May I use Bash to run make b? Say yes to allow it." not in rig.out.said()
    rig.context.add_message({"role": "user", "content": "yes"})
    await rig.worker.queue_frame(LLMContextFrame(rig.context))
    await rig.until(lambda: "May I use Bash to run make b? Say yes to allow it." in rig.out.said())
    assert first.decision.result() == Allow() and second.open
    rig.context.add_message({"role": "user", "content": "not that one"})
    await rig.worker.queue_frame(LLMContextFrame(rig.context))
    await rig.until(lambda: not second.open)
    assert isinstance(second.decision.result(), Deny)
    rig.brain.end()


async def test_a_permission_settled_before_its_turn_came_to_be_said_is_never_asked(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "make the notes say hello"})
    exchange, _ = rig.request()
    rig.stream(exchange, "I'll write it.")
    rig.brain.permit("Write", {"file_path": "/Users/bmf/notes.txt"}).settle(Deny(SPOKEN_OVER))
    rig.stream(exchange, " Or not.")
    await rig.until(lambda: " Or not." in rig.out.said())
    assert rig.out.said() == ["I'll write it.", " Or not."]
    rig.brain.end()


async def test_a_permission_held_for_a_turn_the_stage_no_longer_asks_is_refused(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "make the notes say hello"})
    teller = rig.brain.tellers[-1]
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    asked = Asked(Permission("Write", {"file_path": "/Users/bmf/notes.txt"}), asyncio.get_running_loop().create_future())
    teller(asked)
    assert asked.decision.result() == Deny(NOBODY)


def acknowledgements(rig: Rig) -> list[str]:
    """What hands said of the turns as its own lines: never the brain's words, and kept out of the context."""
    return [frame.text for frame in rig.out.frames if isinstance(frame, TTSSpeakFrame) and not frame.append_to_context]


async def test_a_users_turn_whose_call_runs_on_with_nothing_said_is_acknowledged_before_its_answer(rig: Rig) -> None:
    rig.now[0] = 1000.0
    await rig.release()
    await rig.say({"role": "user", "content": "find out why the build broke"})
    first, _ = rig.request()
    rig.now[0] = 1001.0
    rig.calls(first, ("t1", "Bash"))
    # The call runs past the pause, and nothing of the turn has been said.
    rig.now[0] = 1003.0
    await rig.elapse()
    await rig.until(lambda: acknowledgements(rig) == ["One moment."])
    second, _ = rig.request(answering("Bash", {"output": "error"}))
    rig.stream(second, "The build broke on a missing import.")
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    # Said as a sentence of its own ahead of the answer, which is the brain's alone.
    assert rig.out.said() == ["One moment.", "The build broke on a missing import."]
    assert rig.out.shape()[:4] == ["LLMFullResponseStartFrame", "LLMFullResponseEndFrame", "TTSSpeakFrame", "LLMFullResponseStartFrame"]
    [turn] = turns(rig.recorded)
    assert (turn.facts["acknowledged"], turn.facts["acknowledged_ms"], turn.facts["text"]) == ("One moment.", 3000.0, "The build broke on a missing import.")


async def test_a_turn_whose_answer_was_said_before_its_time_is_not_acknowledged(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "what is api doing?"})
    first, _ = rig.request()
    rig.calls(first, ("t1", "mcp__hands__read_session"))
    second, _ = rig.request(answering("mcp__hands__read_session", {"steps": []}))
    rig.stream(second, "api is running its tests.")
    await rig.until(lambda: rig.out.said() == ["api is running its tests."])
    await rig.elapse()
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    assert acknowledgements(rig) == []
    [turn] = turns(rig.recorded)
    assert (turn.facts["acknowledged"], turn.facts["acknowledged_ms"]) == (None, None)


async def test_a_turn_that_said_what_it_would_do_before_its_call_is_not_acknowledged(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "find out why the build broke"})
    first, _ = rig.request()
    rig.stream(first, "Let me look at the build log.")
    rig.calls(first, ("t1", "Bash"))
    await rig.elapse()
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    assert rig.out.said() == ["Let me look at the build log."]


async def test_a_turn_hands_narrates_is_never_acknowledged_since_nobody_waits_on_it(rig: Rig) -> None:
    await rig.worker.queue_frame(Narrated("[hands] The Claude Code session api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), ()))
    await rig.until(lambda: len(rig.brain.asked) == 1)
    first, _ = rig.request()
    rig.calls(first, ("t1", "mcp__hands__read_session"))
    await rig.elapse()
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    assert acknowledgements(rig) == []


async def test_a_turn_is_acknowledged_once_however_many_calls_run_on_and_the_next_differently(rig: Rig) -> None:
    for asked in ("find out why the build broke", "and the tests?"):
        await rig.say({"role": "user", "content": asked})
        first, _ = rig.request()
        rig.calls(first, ("t1", "Bash"), ("t2", "Grep"))
        await rig.elapse()
        rig.brain.end()
        await rig.until(lambda: len(turns(rig.recorded)) == len(rig.brain.asked))
    assert acknowledgements(rig) == ["One moment.", "On it."]


async def test_a_turn_the_user_spoke_over_while_its_call_runs_is_not_acknowledged(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "find out why the build broke"})
    first, _ = rig.request()
    rig.calls(first, ("t1", "Bash"))
    await rig.interrupt()
    await rig.elapse()
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    assert acknowledgements(rig) == []


async def test_tool_work_of_short_calls_that_runs_on_with_nothing_said_is_acknowledged(rig: Rig) -> None:
    await rig.say({"role": "user", "content": "find out why the build broke"})
    first, _ = rig.request()
    rig.calls(first, ("t1", "Grep"))
    # Each call comes straight back, and the model goes on to the next: the work runs on, unheard.
    second, _ = rig.request(answering("Grep", {"matches": []}))
    rig.calls(second, ("t2", "Read"))
    await rig.elapse()
    await rig.until(lambda: acknowledgements(rig) == ["One moment."])
    rig.brain.end()


async def test_a_turn_that_ends_while_its_acknowledgement_still_plays_tells_it(rig: Rig) -> None:
    rig.now[0] = 1000.0
    rig.out.holding = True
    await rig.say({"role": "user", "content": "find out why the build broke"})
    first, _ = rig.request()
    rig.calls(first, ("t1", "Bash"))
    rig.now[0] = 1002.0
    await rig.elapse()
    second, _ = rig.request(answering("Bash", {"output": "error"}))
    rig.stream(second, "A missing import.")
    rig.brain.end()
    await rig.until(lambda: bool(turns(rig.recorded)))
    [turn] = turns(rig.recorded)
    assert (turn.facts["acknowledged"], turn.facts["acknowledged_ms"]) == ("One moment.", 2000.0)
