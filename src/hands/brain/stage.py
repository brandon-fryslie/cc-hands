"""The brain as the pipeline's LLM stage: the user's turn goes to the brain's stdin, and its words come off the wire.

Pipecat is otherwise untouched: what reaches this stage is the context the user aggregator built, and what leaves it are
the frames an LLM service pushes, which TTS already consumes. What the brain says is read from its requests on the
wire as the proxy hears them, never from its stdout [the design's rule: primary facts from the wire]: text deltas of a
main turn under the brain's own session, and of no other request.

Two things the brain's harness would do on its own are held here instead. Claude Code sends every tool result back for
another response, so a turn that called stay_silent, or one the user barged in on, has its next request answered by
hands with a line that records why: nothing more is said, and the model is not asked to go on. And a barge-in while a tool
whose effect must land is running lets that tool finish rather than having Claude Code cancel it and write it into
history as refused; what it handed back, which the model will not be asked to say, is spoken from the tool's result.
"""

import asyncio
import json
import time
from functools import partial
from itertools import cycle
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, cast

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage, LLMStandardMessage
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.errors import ErrorCategory, classify_http_status_code

from hands.brain.mcp import SERVER_NAME, CallSpans
from hands.brain.process import NOBODY, SPOKEN_OVER, Asked, BrainAnswered
from hands.core.effects import Deny
from hands.core.beside import beside
from hands.core.place import Modality
from hands.core.permissions import heard
from hands.core.front import InFront
from hands.core.session import Permission, SessionId
from hands.core.trace import Span
from hands.core.wire import (
    Answering,
    BlockStarted,
    BlockStopped,
    Exchanged,
    Heard,
    Hold,
    MainTurn,
    Observed,
    Route,
    Seconds,
    Send,
    Sent,
    Tail,
    TextDelta,
    ToolAnswer,
    Unreached,
    tool_answers,
)
from hands.sessions.model_facts import ModelFact, ModelFailed, ModelFault, ModelReplyEmpty, ModelUnreachable
from hands.sessions.audit import Record
from hands.sessions.wide import annotate, child, continuing, count, fail, here, root, unit, within
from hands.voice.player import Mark
from hands.voice.trigger import Edge
from hands.voice.turnstop import HoldDiscarded
from hands.voice.speech import Aloud, Narrated, brain_asks, brain_refused
from hands.voice.utterance import Resumed, Utterance, Uttered, Uttering, uttering
from hands.voice.tool import Result, Tool, silent, whole


class Asking(Protocol):
    """What the stage needs of the brain: the session its requests carry, a turn asked, told of each permission it holds
    for the user, and a turn told to stop."""

    @property
    def session(self) -> SessionId: ...

    async def ask(self, text: str, asks: Callable[[Asked], None]) -> BrainAnswered: ...

    def interrupt(self) -> None: ...


@dataclass(frozen=True)
class UserAsked:
    """The user's words: whether they could see a screen, the edge that opened their turn, and what was in front on the
    Mac's as they were submitted, and how long reading it took, in milliseconds."""

    modality: Modality
    opened: Edge
    front: InFront
    read_ms: float


@dataclass(frozen=True)
class HandsAsked:
    """What hands handed the brain to tell."""


# Whose turn the brain answered.
Asker = UserAsked | HandsAsked

# What the brain is recorded as saying for a request hands held. Never spoken: a held request joins no turn.
SILENT = "(stayed silent)"
INTERRUPTED = "(the user spoke over this reply, and nothing more of it was said)"

# How long the tool work of the user's turn runs, from its first call, with nothing of the turn heard yet, before hands
# acknowledges the turn: past a call that comes straight back with its answer said soon after, so only long work is.
ACKNOWLEDGE_SECONDS = 2.0
# What hands acknowledges with, one after another, so no two turns running acknowledge alike.
ACKNOWLEDGEMENTS = ("One moment.", "On it.", "Okay, working on it.", "Give me a second.")


@dataclass(frozen=True)
class _Acknowledgement:
    """Hands' own line, said ahead of a turn's words while its tools run: the user has heard the turn taken."""

    line: str


def wire_name(tool: Tool) -> str:
    """The name a hands tool has on the wire, where the brain calls it through hands' MCP server."""
    return f"mcp__{SERVER_NAME}__{tool.name}"


_UNNAMED = ModelFailed(ErrorCategory.UNKNOWN)

# What each turn's event counts: the model's round trips on the wire, and the calls its replies made.
COUNTS = ("round_trips", "tools", "refused")


@dataclass
class _Call:
    """A call one of a turn's replies made: its tool, when its block was whole and Claude Code ran it, by the stage's clock
    and the wall's, its span in the turn's trace, and when the request carrying its result left, and whether that result
    was an error."""

    tool: str
    ran: Seconds
    at: datetime
    span: Span
    answered: Seconds | None = None
    is_error: bool = False


@dataclass
class _Turn:
    """One question to the brain, from its write to stdin to its result line."""

    # Words heard on the wire, permissions the turn asks the user for, and hands' acknowledgement of it, not yet handed
    # to the speaker; None once the brain has said the turn is over.
    said: asyncio.Queue[str | Asked | _Acknowledgement | None]
    # Hands the words on to TTS until the turn is over or the user barges in.
    speaking: asyncio.Task[None] = field(init=False)
    spoken: list[str]
    # Who asked the turn: only the user is waiting on it, so only their turn is acknowledged.
    asker: Asker
    # What the turn says of the sessions, where hands asked it to tell them; none for the user's own turn.
    utterances: tuple[Utterance, ...]
    # The turn's span: each of its requests is a span inside it, which the proxy's record of that request carries.
    span: Span
    # The requests on the wire that are this turn's own, in the order they left.
    exchanges: list[str] = field(default_factory=list[str])
    # Each call the turn's replies made, by id, timed as heard.
    tools: dict[str, _Call] = field(default_factory=dict[str, _Call])
    # When the first word of the turn's reply was heard on the wire; None while none has been.
    first_word: Seconds | None = None
    # A reply of the turn's own carried words or a call: a turn none of whose replies did said nothing to the user.
    replied: bool = False
    # What a text block opens with: nothing until the turn has said a word, then a paragraph break, as Claude Code shows
    # them. Joined bare, a block that ended on a closing fence ran the next one's words onto that fence, which no longer
    # closed it, and every word to the end of the turn was held as code (hands-narration-2mc.1zu).
    between: str = ""
    # What goes ahead of the next words heard: the `between` of the text block they open.
    ahead: str = ""
    # The call blocks the reply streaming now has opened and not yet closed, by index: a call is not run until it is whole.
    opening: dict[int, tuple[str, str]] = field(default_factory=dict[int, tuple[str, str]])
    # The calls the turn's last reply made whole, by id, which run until its next request leaves.
    calls: dict[str, str] = field(default_factory=dict[str, str])
    # What hands says for the calls of a held request once the turn's own words are said, since the model is not asked to.
    readbacks: list[str] = field(default_factory=list[str])
    interrupted: bool = False
    # A reply of the turn's is under way for whatever hears it: started, and not yet ended.
    replying: bool = False
    # The calls running when the user barged in, which the turn's event names with whether the brain was told to stop.
    running: tuple[str, ...] = ()
    # The permissions the turn has put to the user, oldest first, which are asked one at a time.
    asked: list[Asked] = field(default_factory=list[Asked])
    # The permission whose question the user has heard to its end: what they say next answers it while it is open. Only a
    # question heard can be answered, so a yes said over the brain's words, or over a question cut off, allows nothing.
    heard: Asked | None = None
    # The permissions the user heard asked and that were refused with no word of the turn's said since, each once, which
    # hands says at its end: a refusal met with silence would pass for the work done.
    unacknowledged: list[Permission] = field(default_factory=list[Permission])
    # How many permissions the user heard asked were refused, whoever spoke to them after.
    refused: int = 0
    # Told to stop by hands, which Claude Code 2.1.285 ends with an error_during_execution result: asked for, not a failure.
    stopped: bool = False
    # What the turn failed of, if it fails, as its latest request's answer told it: nothing named until that answer says.
    failure: ModelFact = _UNNAMED
    # Waits out the turn's tool work to acknowledge it, which the turn's end cancels.
    acknowledging: asyncio.Task[None] = field(init=False)
    # Set at the turn's first call: its tool work has begun, and the wait to acknowledge it with it.
    working: asyncio.Event = field(default_factory=asyncio.Event)
    # What hands acknowledged the turn with, and when it was handed to the speaker, by the stage's clock, as the turn's own
    # words are told: None while it has not been.
    acknowledged: tuple[str, Seconds] | None = None

    def empty(self) -> bool:
        """The model answered the turn, the user did not speak over it, and nothing it answered said or did anything."""
        return bool(self.exchanges) and not (self.interrupted or self.replied)

    def unheard(self) -> bool:
        """The user asked the turn and has heard nothing of it: no word, no question, and they have not spoken over it."""
        return isinstance(self.asker, UserAsked) and self.first_word is None and not self.asked and not self.interrupted

    def asking(self) -> Asked | None:
        """The permission the user's next words answer: the one they heard asked, while it is still open."""
        return self.heard if self.heard is not None and self.heard.open else None

    def hear(self, asked: Asked) -> None:
        self.heard = asked


class BrainStage(FrameProcessor):
    """The LLM stage under the brain: a context in, the brain's words out as LLM text frames, and a barge-in passed on."""

    def __init__(
        self,
        brain: Asking,
        tools: Sequence[Tool],
        tail: Callable[[], str],
        refocus: Callable[[SessionId], Awaitable[None]],
        front: Callable[[], Awaitable[InFront]],
        modality: Callable[[], Modality],
        opened: Callable[[], Edge],
        record: Record,
        spans: CallSpans,
        clock: Callable[[], Seconds] = time.monotonic,
        pause: Callable[[Seconds], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._brain = brain
        # What is in front on the Mac's screen, read as the user's words arrive.
        self._front = front
        # Whether the user can see a screen, read as their words arrive.
        self._modality = modality
        # The edge that opened the gate's last turn.
        self._opened = opened
        # Moves the focus to a session whose telling the brain takes.
        self._refocus = refocus
        # What hands appends to each request of a turn, composed as that request leaves.
        self._tail = tail
        self._record = record
        # Each call's span, where hands' MCP server finds it as the call's run reaches it.
        self._spans = spans
        self._now = clock
        # Waits out a call's time before the turn is acknowledged.
        self._pause = pause
        self._acknowledgements = cycle(ACKNOWLEDGEMENTS)
        self._tools = {wire_name(tool): tool for tool in tools}
        # How many of the context's messages the brain has been handed: the rest are new to it.
        self._told = 0
        # [LAW:no-ambient-temporal-coupling] the turn is the brain's, from its write to its result line, not the pipeline's:
        # it runs beside the frames passing through, a barge-in ends what is said of it, and the next is written only
        # once the brain has ended it. What waits is in two lanes, the user's and hands', and the user's goes first.
        # Each thing waiting is kept with when it arrived, and the user's words with when they let go of the key on them,
        # if they did, and the read of the screen begun as they arrived. Hands' lane holds what it says as written beside
        # what it hands the brain, so a session's story is heard in the order it happened.
        self._contexts: deque[tuple[str, Seconds, Seconds | None, asyncio.Task[UserAsked]]] = deque()
        self._hands: deque[tuple[Narrated | Aloud, Seconds]] = deque()
        # When the user last let go of the key on words not yet handed to the brain: what their wait is timed from. One, not
        # one per hold: a hold let go of while an earlier one is still transcribed joins that hold's turn (KeyTurnStop), and
        # the turn's one context follows the last release of the holds it took in. Kept with the edge that opened that
        # hold, read from the gate as it is let go of, before any later hold can open.
        self._released: tuple[Seconds, Edge] | None = None
        self._waiting = asyncio.Event()
        self._turn: _Turn | None = None
        # What the user heard of the last turn before the API broke it off, told to the brain with its next turn: Claude
        # Code keeps the broken reply out of the brain's history (2.1.285), so without it the brain cannot answer about it.
        self._broken_off = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case LLMContextFrame(context=context) if self._turn is not None and (asked := self._turn.asking()) is not None:
                # What the user says while the brain asks them is their answer, not a turn of their own: the brain is held
                # mid-turn, and hears it as its tool's run or refusal.
                self._released = None
                if words := self._news(context):
                    asked.settle(heard(words))
            case LLMContextFrame(context=context):
                # [LAW:no-ambient-temporal-coupling] read as it arrives, so an answer given later takes only its own words,
                # never words of the user's still waiting to be asked. A context frame is a call to answer, not a message:
                # one that gained the brain nothing asks nothing.
                # [LAW:no-ambient-temporal-coupling] the screen is read from as the words arrive, not as their turn is
                # taken: the user may look elsewhere, or switch to audio-only, while it waits.
                released, self._released = self._released, None
                match released:
                    case None:
                        let_go, opened = None, self._opened()
                    case (let_go, opened):
                        pass
                if news := self._news(context):
                    self._contexts.append((news, self._now(), let_go, asyncio.ensure_future(self._read_front(self._modality(), opened))))
                    self._waiting.set()
            case HoldDiscarded():
                await self.push_frame(frame, direction)
            case VADUserStoppedSpeakingFrame():
                # The key let go on words to be transcribed: the end of the user's words, and the start of their wait.
                self._released = (self._now(), self._opened())
                await self.push_frame(frame, direction)
            case Narrated() | Aloud():
                self._hands.append((frame, self._now()))
                self._waiting.set()
            case InterruptionFrame():
                # [LAW:no-ambient-temporal-coupling] the turn stops being spoken, then the pipeline is told, then the brain:
                # what is playing stops first, and a brain that cannot be written to cannot hold the barge-in back.
                turn = self._turn
                stop = self._barge_in()
                await self.push_frame(frame, direction)
                if stop:
                    self._brain.interrupt()
                if turn is not None and not turn.interrupted:
                    if turn.replying:
                        # A barge-in ends the reply under way for everything behind this stage, and the turn goes on
                        # through it: what it says next is a reply started again, written whole at its end.
                        await self.push_frame(LLMFullResponseStartFrame())
                    # It cut off none of what the turn is still to say of the sessions.
                    await self.push_frame(Resumed(turn.utterances))
            case _:
                await self.push_frame(frame, direction)

    async def ask_each(self) -> None:
        """Asks the brain each turn the pipeline hands this stage, one at a time, for as long as it runs; returns only by
        raising what stopped it, since a stage that can no longer ask leaves every question unanswered."""
        while True:
            waiting, arrived, released = await self._upcoming()
            # Taken from its lane: how long it waited there, which reading the screen is not.
            taken = self._now()
            match waiting:
                case (str() as text, reading):
                    asker = await reading
                    await self._ask(f"{text}\n\n{beside(asker.front, asker.modality)}", asker, (), (), arrived, released, taken)
                case Narrated(text=text, unsaid=unsaid, session=session, utterances=utterances):
                    # [LAW:no-ambient-temporal-coupling] moved as the telling is taken, with the user's last turn ended and
                    # none waiting, since they go first, and before the brain is asked, so its request reads the new focus.
                    await self._refocus(session)
                    # The turn is what says them, sent with what tells what of them was heard, and a part of the first
                    # of them, in its trace.
                    await self.push_frame(Uttering(utterances))
                    with continuing(utterances[0].begun.span if utterances else None):
                        failure = await self._ask(text, HandsAsked(), (unsaid,), utterances, arrived, None, taken)
                    match failure:
                        case str():
                            for utterance in utterances:
                                # [LAW:nothing-unseen] what was heard of a telling the brain failed is that it could not be told.
                                utterance.fail(failure)
                        case None:
                            pass
                    await self.push_frame(Uttered(utterances))
                case Aloud(spoken=spoken, utterances=utterances):
                    # In hands' lane no turn of the brain's is under way.
                    for frame in uttering(utterances, (spoken,)):
                        await self.push_frame(frame)

    async def _read_front(self, modality: Modality, opened: Edge) -> UserAsked:
        began = self._now()
        front = await self._front()
        return UserAsked(modality, opened, front, (self._now() - began) * 1000)

    async def _upcoming(self) -> tuple[tuple[str, asyncio.Task[UserAsked]] | Narrated | Aloud, Seconds, Seconds | None]:
        """What is next: the user's words while any wait, since what they said goes ahead of what hands has to tell, all
        that waits of them asked as one turn, with when the last of them arrived and when the user let go of the key on it,
        since the user waits from the last words they said."""
        while not (self._contexts or self._hands):
            self._waiting.clear()
            await self._waiting.wait()
        if not self._contexts:
            told, arrived = self._hands.popleft()
            return told, arrived, None
        *earlier, (_, arrived, released, reading) = self._contexts
        # Asked as one turn, read against the screen as the last of them arrived.
        for _, _, _, superseded in earlier:
            superseded.cancel()
        news = "\n\n".join(text for text, _, _, _ in self._contexts)
        self._contexts.clear()
        return (news, reading), arrived, released

    async def _ask(
        self, text: str, asker: Asker, unsaid: Sequence[str], utterances: tuple[Utterance, ...], arrived: Seconds, released: Seconds | None, taken: Seconds
    ) -> str | None:
        """One turn of the brain's, and its one event; `unsaid` is what hands says as written if the brain cannot take it,
        and `utterances` what the turn says of the sessions. `released` is when the user let go of the key on the words
        the turn asks, None for a turn not asked aloud, and `taken` when the turn left its lane. Returns why the turn
        failed, None where it did not."""
        # [LAW:nothing-unseen] every turn passes through here, whoever asked it and however it ends.
        with unit("voice.turn", self._record, COUNTS):
            return await self._turn_of(text, asker, unsaid, utterances, arrived, released, taken)

    async def _turn_of(
        self, text: str, asker: Asker, unsaid: Sequence[str], utterances: tuple[Utterance, ...], arrived: Seconds, released: Seconds | None, taken: Seconds
    ) -> str | None:
        note, self._broken_off = self._broken_off, ""
        text = "\n\n".join(part for part in (note, text) if part)
        # What is typed into the brain for the turn, as it is typed.
        annotate(asked=text)
        said: asyncio.Queue[str | Asked | _Acknowledgement | None] = asyncio.Queue()
        spoken: list[str] = []
        turn = self._turn = _Turn(said, spoken, asker, utterances, here())
        turn.speaking = asyncio.create_task(self._speak(turn), name="the brain's words")
        turn.acknowledging = asyncio.create_task(self._acknowledge(turn), name="the turn's acknowledgement")
        # When the brain ended the turn, or it was stopped: set before anything reads it, however the turn ends.
        ended = taken
        try:
            try:
                asked = asyncio.ensure_future(self._brain.ask(text, lambda permission: self._put(turn, permission)))
                await asyncio.wait({asked})
            finally:
                self._turn = None
                ended = self._now()
                # [LAW:no-ambient-temporal-coupling] nothing acknowledges a turn once it is over.
                turn.acknowledging.cancel()
                said.put_nowait(None)
            await asyncio.wait({turn.speaking})
            # [LAW:single-enforcer] a refusal the user heard asked is never left to silence, whatever the model did after it.
            refusals = tuple(brain_refused(permission) for permission in turn.unacknowledged)
            annotate(refusals=refusals)
            await self._unsaid(refusals)
            for readback in turn.readbacks:
                # Said by hands, since the model that would have said it was not asked to go on.
                await self.push_frame(TTSSpeakFrame(readback))
        finally:
            # [LAW:nothing-unseen] what the turn did is on its event however it ended, a turn cancelled mid-way included.
            self._account(turn, asker, arrived, released, taken, ended)
        # A brain that failed the turn itself has no answer to read.
        failure = None if asked.exception() is not None else _failed(turn, asked.result().error)
        annotate(failed=None if failure is None else failure.fact)
        if (error := asked.exception()) is not None:
            failed = f"the brain failed a turn: {error}"
            # [LAW:no-silent-failure] said as the turn's failure whatever failed it: a brain that is gone also stops the run
            # from its own watch, but one that never took the turn, or could not be typed into, is still running.
            fail(failed)
            # A turn the brain never took did not tell it what the user heard: the turn after it does. One it spoke in took
            # that, and was cut where the user stopped hearing it.
            self._broken_off = _broken_off("".join(spoken)) if spoken else note
            await self._unsaid(unsaid)
            await self.push_error(failed)  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
            return failed
        elif failure is not None:
            # [LAW:no-silent-failure] said as the API services' failures are: an error from the model's stage. No category:
            # Pipecat takes an invalid request or a refused login as permanent and stops the stage, and the brain goes on.
            self._broken_off = failure.note
            fail(failure.error)
            await self._unsaid(unsaid)
            await self.push_error(failure.error, exception=ModelFault(failure.fact))  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
            return failure.error
        return None

    def _account(self, turn: _Turn, asker: Asker, arrived: Seconds, released: Seconds | None, taken: Seconds, ended: Seconds) -> None:
        """The turn's event: how long the user waited for its first word and where the time went, what it said, and each
        call its replies made, as a part of it. Its round trips to the model are the proxy's records of its requests."""
        # Timed from the end of the user's words where they spoke them, and from when hands handed it over where not.
        since = arrived if released is None else released
        annotate(
            asker=asker,
            exchanges=tuple(turn.exchanges),
            text="".join(turn.spoken),
            readbacks=tuple(turn.readbacks),
            interrupted=turn.interrupted,
            running=turn.running,
            stopped=turn.stopped,
            # The wait, then where it went: transcribing what was said, waiting behind the turn before it, and the rest,
            # from the turn leaving its lane to its first word: what was left of reading the screen (the asker's read_ms
            # is the whole read, begun as the words arrived), then the model's and its tools', as its parts show.
            waited_ms=None if turn.first_word is None else _ms(turn.first_word - since),
            # What hands acknowledged the turn with while its tools ran, and how long the user had waited for it.
            acknowledged=None if turn.acknowledged is None else turn.acknowledged[0],
            acknowledged_ms=None if turn.acknowledged is None else _ms(turn.acknowledged[1] - since),
            transcribed_ms=None if released is None else _ms(arrived - released),
            queued_ms=_ms(taken - arrived),
        )
        count(round_trips=len(turn.exchanges), tools=len(turn.tools), refused=turn.refused)
        self._spans.ended(turn.tools)
        for call, ran in turn.tools.items():
            # A call whose result never left ran until the turn ended without it.
            outcome = "cancelled" if ran.answered is None else "failed" if ran.is_error else "ok"
            child("tool.call", ran.span, ran.at, _ms((ended if ran.answered is None else ran.answered) - ran.ran), outcome, call=call, tool=ran.tool)

    async def _unsaid(self, unsaid: Sequence[str]) -> None:
        """What hands had for the brain to tell, said as written since the brain did not: a system fact, kept out of the context."""
        for line in unsaid:
            await self.push_frame(TTSSpeakFrame(line, append_to_context=False))

    def _put(self, turn: _Turn, asked: Asked) -> None:
        """Puts a permission the brain holds to the user, after what the turn has said so far."""
        if turn is not self._turn:
            # The stage stopped asking this turn, so nothing would put it to the user or hear their answer.
            asked.settle(Deny(NOBODY))
            return
        if turn.interrupted:
            # Nothing more of the turn is said, so they would never hear it asked.
            asked.settle(Deny(SPOKEN_OVER))
            return
        turn.asked.append(asked)
        turn.said.put_nowait(asked)

    async def _reply(self, turn: _Turn, under_way: bool) -> None:
        """Starts or ends the turn's reply for whatever hears it."""
        # [LAW:no-ambient-temporal-coupling] noted ahead of the frame that says so, with nothing awaited between: a
        # barge-in the turn goes on through starts again only a reply that has not ended.
        turn.replying = under_way
        await self.push_frame(LLMFullResponseStartFrame() if under_way else LLMFullResponseEndFrame())

    async def _speak(self, turn: _Turn) -> None:
        await self._reply(turn, True)
        while (words := await turn.said.get()) is not None:
            match words:
                case str():
                    turn.spoken.append(words)
                    # The brain's own words after a refusal are its acknowledgement of it.
                    if words.strip():
                        turn.unacknowledged.clear()
                    await self.push_frame(LLMTextFrame(words))
                case Asked(permission=permission) as asked if asked.open:
                    await self._aside(turn, TTSSpeakFrame(brain_asks(permission)))
                    # [LAW:no-ambient-temporal-coupling] answerable once the speaker has played it to its end, which the
                    # mark is told of, and never if a barge-in cut it off.
                    await self.push_frame(Mark(partial(turn.hear, asked)))
                    # One question at a time, so what the user answers is the question they heard last.
                    await asyncio.wait({asked.decision})
                    # Owed only for a question the user heard to its end: one cut off or never reached they cannot take
                    # for the work done. Said once, however often the brain asked it again in silence.
                    if turn.heard is asked and isinstance(asked.decision.result(), Deny):
                        turn.refused += 1
                        if permission not in turn.unacknowledged:
                            turn.unacknowledged.append(permission)
                    await self._reply(turn, True)
                case Asked():
                    # Settled before its turn came to be said, by the deadline, the turn's end, or an answer: nothing asks.
                    pass
                case _Acknowledgement(line=line):
                    turn.acknowledged = (line, self._now())
                    # Kept out of the context: the brain never said it.
                    await self._aside(turn, TTSSpeakFrame(line, append_to_context=False))
                    await self._reply(turn, True)
        # Not on a barge-in, which cancels this: an end would have TTS say the sentence the user spoke over.
        await self._reply(turn, False)

    async def _aside(self, turn: _Turn, line: TTSSpeakFrame) -> None:
        """Says a line of hands' own in the turn, as a sentence of its own once the brain's words before it are: the response
        so far ends first, so what TTS holds of it is said ahead of the line."""
        await self._reply(turn, False)
        await self.push_frame(line)

    def _news(self, context: LLMContext) -> str:
        """What the context gained since the brain last heard it, as one message: the user's words and hands' notes."""
        messages = context.get_messages()
        fresh = messages[self._told :]
        self._told = len(messages)
        # [LAW:one-source-of-truth] the brain keeps its own history, so what it said itself, which the assistant
        # aggregator writes back into this context, is never handed to it again.
        return "\n\n".join(_user_text(message) for message in fresh if not isinstance(message, LLMSpecificMessage) and message.get("role") == "user")

    def _barge_in(self) -> bool:
        """Stops the turn in flight being spoken; True when the brain is to be told to stop it too."""
        turn = self._turn
        # A turn none of whose requests has left is not being answered yet: the user's new words follow it as the next turn.
        # One that asks the user a permission is waiting on what they say: their words answer it, and stop nothing.
        if turn is None or turn.interrupted or not turn.exchanges or turn.asking() is not None:
            return False
        turn.interrupted = True
        # Cancelled before anything else runs, so no word of the turn follows the barge-in down the pipeline.
        turn.speaking.cancel()
        # None of them was heard to its end, or the user's words would have answered it: now none can be.
        for asked in turn.asked:
            asked.settle(Deny(SPOKEN_OVER))
        # The user is speaking: what they say is the next turn, which the brain answers knowing what was refused.
        turn.unacknowledged.clear()
        # A tool whose effect must land runs to its end; stopped by the harness, it would land and be written in
        # history as refused. Its turn's next request is held instead, so the model is not asked to go on either way.
        turn.running = tuple(turn.calls.values())
        turn.stopped = not any(self._completes(name) for name in turn.running)
        return turn.stopped

    def route(self, sent: Sent) -> Route:
        """Where a request on the wire goes: each of a turn's own requests with hands' tail on it, and the next one after
        stay_silent or a barge-in held."""
        # Another session's request is made for no unit of the stage's: the root of a trace of its own.
        if sent.session != self._brain.session:
            return Send(span=root())
        # [LAW:single-enforcer] every request of the brain's is final, whatever its kind and whether a turn asked it: asked
        # again, Claude Code would keep the user waiting minutes on its retries, and a spent limit it would wait out to
        # continue the task on its own at the reset, hours on, with nobody asking (hands-wire-zi2).
        turn = self._turn
        # A request the brain makes while a turn is asked of it, a subagent's or a fork's among them, is that turn's part.
        if not isinstance(sent.kind, MainTurn):
            return Send(refusal="final", span=root() if turn is None else within(turn.span))
        if turn is None:
            # [LAW:no-silent-failure] the brain asked the model something with no turn written to it: heard, never spoken.
            logger.warning(f"the brain sent a main turn (exchange {sent.exchange}) with no turn asked of it; nothing it says will be spoken")
            return Send(refusal="final", span=root())
        now = self._now()
        carried = tool_answers(sent.body)
        # A call's result leaving is the end of its run, whether the request carrying it goes on or is held.
        for answer in carried:
            if (ran := turn.tools.get(answer.call)) is not None and ran.answered is None:
                ran.answered, ran.is_error = now, answer.is_error
        # Only the calls this turn's last reply opened: a request carries every result of the brain's history.
        answers = [(self._tools.get(turn.calls[answer.call]), answer.text, _result(answer)) for answer in carried if answer.call in turn.calls]
        # Said by hands once the brain's own words are, whatever the model does next: what a call hands hands to say.
        turn.readbacks.extend(says for _, _, result in answers if result is not None and isinstance(says := result.get("says"), str))
        if not (turn.interrupted or whole([tool is not None and result is not None and silent(tool, result) for tool, _, result in answers])):
            turn.exchanges.append(sent.exchange)
            turn.opening, turn.calls, turn.failure = {}, {}, _UNNAMED
            # Refused once is the turn's failure, said at once as the API variants say theirs, who ask once. The proxy's
            # record of the request is the turn's round trip to the model: a span inside the turn's.
            return Send((Tail(self._tail()),), refusal="final", span=within(turn.span))
        turn.readbacks.extend(said for tool, text, result in answers if tool is not None and tool.completes and (said := _owed(text, result)) is not None)
        return Hold(INTERRUPTED if turn.interrupted else SILENT, within(turn.span))

    async def _acknowledge(self, turn: _Turn) -> None:
        """Acknowledges the turn once its tool work has run its time, however many calls it took, if the user has heard
        nothing of it: they hear it was taken, rather than silence until its answer."""
        await turn.working.wait()
        await self._pause(ACKNOWLEDGE_SECONDS)
        if turn.unheard():
            turn.said.put_nowait(_Acknowledgement(next(self._acknowledgements)))


    def _completes(self, name: str) -> bool:
        """Whether a barge-in lets the call finish, as hands' tools say: a call to a tool not hands' never does."""
        tool = self._tools.get(name)
        return tool is not None and tool.completes

    def hear(self, observed: Observed) -> None:
        turn = self._turn
        if turn is None:
            return
        now = self._now()
        match observed:
            # Said as it arrives, and never twice: a reply the API breaks mid-stream is not asked for again, streamed or
            # not; the turn ends in StopFailure, with the broken reply kept out of the brain's history (2.1.285, hands-wire-6ic.6dz).
            case Heard(exchange=exchange, event=BlockStarted(block={"type": "text"})) if exchange in turn.exchanges:
                turn.ahead = turn.between
            case Heard(exchange=exchange, event=TextDelta(text=text)) if exchange in turn.exchanges:
                turn.said.put_nowait(turn.ahead + text)
                turn.ahead, turn.between = "", "\n\n"
                turn.replied = turn.replied or bool(text.strip())
                if text.strip() and turn.first_word is None:
                    turn.first_word = now
            case Heard(exchange=exchange, event=BlockStarted(index=index, block={"type": "tool_use", "id": str() as call, "name": str() as name})) if exchange in turn.exchanges:
                turn.opening[index] = (call, name)
                turn.replied = True
            # Claude Code runs a call once its block is whole, before the reply's last byte, and hands hears each byte
            # before Claude Code does: the call is running from here on.
            case Heard(exchange=exchange, event=BlockStopped(index=index)) if exchange in turn.exchanges and index in turn.opening:
                call, name = turn.opening.pop(index)
                turn.calls[call] = name
                ran = turn.tools[call] = _Call(name, now, datetime.now(UTC), within(turn.span))
                self._spans.opened(call, ran.span)
                turn.working.set()
            # [LAW:one-source-of-truth] why a turn failed is the wire's, as the API variants read it off their own calls: the
            # head of its latest answer, heard before Claude Code reads any of it, or an API the proxy could not reach,
            # told before its 502. Its own requests are held final, so Claude Code asks once; a 401 it asks again after
            # refreshing its login: the latest request's is the turn's.
            case Answering(exchange=exchange, status=status, limit=limit) if exchange in turn.exchanges:
                turn.failure = limit or ModelFailed(classify_http_status_code(status))
            case Exchanged(exchange=exchange, reply=Unreached()) if exchange in turn.exchanges:
                turn.failure = ModelUnreachable()
            case _:
                pass


def _ms(seconds: Seconds) -> float:
    return round(seconds * 1000, 3)


def _user_text(message: LLMStandardMessage) -> str:
    content = message.get("content")
    if not isinstance(content, str):
        # [LAW:no-silent-failure] hands' aggregator and notes write plain text; anything else is a change to hear about.
        raise TypeError(f"a user message in the context is not plain text: {message!r}")
    return content


@dataclass(frozen=True)
class _Failure:
    """What a turn the brain finished failed of: the stage's error, the fact it is, and the note the brain's next turn carries."""

    error: str
    fact: ModelFact
    note: str


def _failed(turn: _Turn, error: str | None) -> _Failure | None:
    """What a turn the brain finished failed of; None for one that did not."""
    if turn.stopped:
        # Told to stop by hands is asked for, not a failure, whatever the turn ended in.
        return None
    if error is not None:
        return _Failure(f"the brain's turn ended in error: {error}", turn.failure, _broken_off("".join(turn.spoken)))
    if turn.empty():
        # An empty reply is a whole one, kept in the brain's history: there is nothing broken off to tell it of.
        return _Failure("the brain's reply was empty", ModelReplyEmpty(), "")
    return None


def _broken_off(spoken: str) -> str:
    """The note that tells the brain what the user heard of a turn broken off before its end; none when nothing of it was said."""
    return f'[hands] Your last turn was broken off. The user heard you say "{spoken}", then that it failed. Say nothing about this unless the user asks.' if spoken else ""


def _result(answer: ToolAnswer) -> Result | None:
    """What the tool handed back, or None for the MCP server's own failure: a line of text naming the tool and what went wrong."""
    try:
        result: object = json.loads(answer.text)
    except ValueError:
        return None
    return cast(Result, result) if isinstance(result, dict) else None


def _owed(text: str, result: Result | None) -> str | None:
    """What a call that must land handed back for the user that the model, not asked to go on, would have said: its
    readback, or why it failed. None when hands says it already, and the MCP server's own failure as it wrote it."""
    match result:
        case {"says": str()}:
            return None
        case {"readback": str() as said} | {"error": str() as said}:
            return said
        case _:
            return text
