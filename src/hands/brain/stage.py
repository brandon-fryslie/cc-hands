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
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol, cast

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

from hands.brain.mcp import SERVER_NAME
from hands.brain.process import NOBODY, SPOKEN_OVER, Asked
from hands.core.effects import Deny
from hands.core.permissions import heard
from hands.core.session import SessionId
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
from hands.sessions.audit import BrainAnswered, BrainInterrupted, Record
from hands.sessions.wide import annotate, child, count, fail, unit
from hands.voice.player import Mark
from hands.voice.turnstop import HoldDiscarded
from hands.voice.speech import Aloud, Narrated, brain_asks
from hands.voice.tools import Result, Tool, silent, whole


class Asking(Protocol):
    """What the stage needs of the brain: the session its requests carry, a turn asked, told of each permission it holds
    for the user, and a turn told to stop."""

    @property
    def session(self) -> SessionId: ...

    async def ask(self, text: str, asks: Callable[[Asked], None]) -> BrainAnswered: ...

    def interrupt(self) -> None: ...


# Whose turn the brain answered: the user's words, or what hands handed it to tell.
Asker = Literal["user", "hands"]

# What the brain is recorded as saying for a request hands held. Never spoken: a held request joins no turn.
SILENT = "(stayed silent)"
INTERRUPTED = "(the user spoke over this reply, and nothing more of it was said)"


def wire_name(tool: Tool) -> str:
    """The name a hands tool has on the wire, where the brain calls it through hands' MCP server."""
    return f"mcp__{SERVER_NAME}__{tool.name}"


_UNNAMED = ModelFailed(ErrorCategory.UNKNOWN)

# What each turn's event counts: the model's round trips on the wire, and the calls its replies made.
COUNTS = ("round_trips", "tools")


@dataclass
class _RoundTrip:
    """One of a turn's requests to the model, as the stage heard it: when it left, its answer's status, when that answer's
    head and its latest event were heard, each before Claude Code reads it; `unreached` is why the API was never heard from."""

    left: Seconds
    status: int | None = None
    first: Seconds | None = None
    last: Seconds | None = None
    unreached: str | None = None


@dataclass
class _Call:
    """A call one of a turn's replies made: its tool, when its block was whole and Claude Code ran it, and when the request
    carrying its result left, and whether that result was an error."""

    tool: str
    ran: Seconds
    answered: Seconds | None = None
    is_error: bool = False


@dataclass
class _Turn:
    """One question to the brain, from its write to stdin to its result line."""

    # Words heard on the wire, and permissions the turn asks the user for, not yet handed to the speaker; None once the
    # brain has said the turn is over.
    said: asyncio.Queue[str | Asked | None]
    # Hands the words on to TTS until the turn is over or the user barges in.
    speaking: asyncio.Task[None] = field(init=False)
    spoken: list[str]
    # The requests on the wire that are this turn's own, in the order they left.
    exchanges: list[str] = field(default_factory=list[str])
    # Each of those requests' round trip, by exchange, and each call its replies made, by id, timed as heard.
    round_trips: dict[str, _RoundTrip] = field(default_factory=dict[str, _RoundTrip])
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
    # The permissions the turn has put to the user, oldest first, which are asked one at a time.
    asked: list[Asked] = field(default_factory=list[Asked])
    # The permission whose question the user has heard to its end: what they say next answers it while it is open. Only a
    # question heard can be answered, so a yes said over the brain's words, or over a question cut off, allows nothing.
    heard: Asked | None = None
    # Told to stop by hands, which Claude Code 2.1.285 ends with an error_during_execution result: asked for, not a failure.
    stopped: bool = False
    # What the turn failed of, if it fails, as its latest request's answer told it: nothing named until that answer says.
    failure: ModelFact = _UNNAMED

    def empty(self) -> bool:
        """The model answered the turn, the user did not speak over it, and nothing it answered said or did anything."""
        return bool(self.exchanges) and not (self.interrupted or self.replied)

    def asking(self) -> Asked | None:
        """The permission the user's next words answer: the one they heard asked, while it is still open."""
        return self.heard if self.heard is not None and self.heard.open else None

    def hear(self, asked: Asked) -> None:
        self.heard = asked


class BrainStage(FrameProcessor):
    """The LLM stage under the brain: a context in, the brain's words out as LLM text frames, and a barge-in passed on."""

    def __init__(
        self, brain: Asking, tools: Sequence[Tool], tail: Callable[[], str], refocus: Callable[[SessionId], Awaitable[None]], record: Record, clock: Callable[[], Seconds] = time.time
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._brain = brain
        # Moves the focus to a session whose telling the brain takes.
        self._refocus = refocus
        # What hands appends to each request of a turn, composed as that request leaves.
        self._tail = tail
        self._record = record
        # Wall-clock, as the wire's times are, so what the stage times and what the proxy times read on one line.
        self._now = clock
        self._tools = {wire_name(tool): tool for tool in tools}
        # How many of the context's messages the brain has been handed: the rest are new to it.
        self._told = 0
        # [LAW:no-ambient-temporal-coupling] the turn is the brain's, from its write to its result line, not the pipeline's:
        # it runs beside the frames passing through, a barge-in ends what is said of it, and the next is written only
        # once the brain has ended it. What waits is in two lanes, the user's and hands', and the user's goes first.
        # Each thing waiting is kept with when it arrived, and the user's words with when they let go of the key on them,
        # if they did. Hands' lane holds what it says as written beside what it hands the brain, so a session's story is
        # heard in the order it happened.
        self._contexts: deque[tuple[str, Seconds, Seconds | None]] = deque()
        self._hands: deque[tuple[Narrated | Aloud, Seconds]] = deque()
        # When the user last let go of the key on words not yet handed to the brain: what their wait is timed from.
        self._released: Seconds | None = None
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
                # never words of the user's still waiting to be asked.
                self._contexts.append((self._news(context), self._now(), self._released))
                self._released = None
                self._waiting.set()
            case HoldDiscarded():
                await self.push_frame(frame, direction)
            case VADUserStoppedSpeakingFrame():
                # The key let go on words to be transcribed: the end of the user's words, and the start of their wait.
                self._released = self._now()
                await self.push_frame(frame, direction)
            case Narrated() | Aloud():
                self._hands.append((frame, self._now()))
                self._waiting.set()
            case InterruptionFrame():
                # [LAW:no-ambient-temporal-coupling] the turn stops being spoken, then the pipeline is told, then the brain:
                # what is playing stops first, and a brain that cannot be written to cannot hold the barge-in back.
                stop = self._barge_in()
                await self.push_frame(frame, direction)
                if stop:
                    self._brain.interrupt()
            case _:
                await self.push_frame(frame, direction)

    async def ask_each(self) -> None:
        """Asks the brain each turn the pipeline hands this stage, one at a time, for as long as it runs; returns only by
        raising what stopped it, since a stage that can no longer ask leaves every question unanswered."""
        while True:
            waiting, arrived, released = await self._upcoming()
            match waiting:
                case str() as text:
                    # A context frame is a call to answer, not a message: one that gained the brain nothing asks nothing.
                    if text:
                        await self._ask(text, "user", (), arrived, released)
                case Narrated(text=text, unsaid=unsaid, session=session):
                    # [LAW:no-ambient-temporal-coupling] moved as the telling is taken, with the user's last turn ended and
                    # none waiting, since they go first, and before the brain is asked, so its request reads the new focus.
                    await self._refocus(session)
                    await self._ask(text, "hands", (unsaid,), arrived, None)
                case Aloud(spoken=spoken):
                    await self.push_frame(spoken)

    async def _upcoming(self) -> tuple[str | Narrated | Aloud, Seconds, Seconds | None]:
        """What is next: the user's words while any wait, since what they said goes ahead of what hands has to tell, all
        that waits of them asked as one turn, with when the first of them arrived and when the user last let go of the key."""
        while not (self._contexts or self._hands):
            self._waiting.clear()
            await self._waiting.wait()
        if not self._contexts:
            told, arrived = self._hands.popleft()
            return told, arrived, None
        arrived = self._contexts[0][1]
        released = max((released for _, _, released in self._contexts if released is not None), default=None)
        news = "\n\n".join(text for text, _, _ in self._contexts if text)
        self._contexts.clear()
        return news, arrived, released

    async def _ask(self, text: str, asker: Asker, unsaid: Sequence[str], arrived: Seconds, released: Seconds | None) -> None:
        """One turn of the brain's, and its one event; `unsaid` is what hands says as written if the brain cannot take it.
        `released` is when the user let go of the key on the words the turn asks, None for a turn not asked aloud."""
        # [LAW:nothing-unseen] every turn passes through here, whoever asked it and however it ends.
        with unit("voice.turn", self._record, COUNTS):
            await self._turn_of(text, asker, unsaid, arrived, released)

    async def _turn_of(self, text: str, asker: Asker, unsaid: Sequence[str], arrived: Seconds, released: Seconds | None) -> None:
        written = self._now()
        note, self._broken_off = self._broken_off, ""
        text = "\n\n".join(part for part in (note, text) if part)
        said: asyncio.Queue[str | Asked | None] = asyncio.Queue()
        spoken: list[str] = []
        turn = self._turn = _Turn(said, spoken)
        turn.speaking = asyncio.create_task(self._speak(turn), name="the brain's words")
        try:
            asked = asyncio.ensure_future(self._brain.ask(text, lambda permission: self._put(turn, permission)))
            await asyncio.wait({asked})
        finally:
            self._turn = None
            said.put_nowait(None)
        await asyncio.wait({turn.speaking})
        for readback in turn.readbacks:
            # Said by hands, since the model that would have said it was not asked to go on.
            await self.push_frame(TTSSpeakFrame(readback))
        # A brain that failed the turn itself has no answer to read.
        failure = None if asked.exception() is not None else _failed(turn, asked.result().error)
        self._account(turn, asker, arrived, released, written, failure)
        if (error := asked.exception()) is not None:
            fail(f"the brain failed a turn: {error}")
            # [LAW:no-silent-failure] said as the turn's failure whatever failed it: a brain that is gone also stops the run
            # from its own watch, but one that never took the turn, or could not be typed into, is still running.
            logger.opt(exception=error).error("the brain failed a turn")
            # A turn the brain never took did not tell it what the user heard: the turn after it does.
            self._broken_off = note
            await self._unsaid(unsaid)
            await self.push_error(f"the brain failed a turn: {error}")  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        elif failure is not None:
            # [LAW:no-silent-failure] said as the API services' failures are: an error from the model's stage. No category:
            # Pipecat takes an invalid request or a refused login as permanent and stops the stage, and the brain goes on.
            self._broken_off = failure.note
            fail(failure.error)
            await self._unsaid(unsaid)
            await self.push_error(failure.error, exception=ModelFault(failure.fact))  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)

    def _account(self, turn: _Turn, asker: Asker, arrived: Seconds, released: Seconds | None, written: Seconds, failure: "_Failure | None") -> None:
        """The turn's event: how long the user waited for its first word and where the time went, what it said, and each
        round trip to the model and each call its replies made, as parts of it."""
        ended = self._now()
        # Timed from the end of the user's words where they spoke them, and from when hands handed it over where not.
        since = arrived if released is None else released
        annotate(
            asker=asker,
            exchanges=tuple(turn.exchanges),
            text="".join(turn.spoken),
            readbacks=tuple(turn.readbacks),
            interrupted=turn.interrupted,
            failed=None if failure is None else failure.fact,
            # The wait, then where it went: transcribing what was said, waiting behind the turn before it, and the rest,
            # from the turn being written to its first word, the model's and its tools', as its parts show.
            waited_ms=None if turn.first_word is None else _ms(turn.first_word - since),
            transcribed_ms=None if released is None else _ms(arrived - released),
            queued_ms=_ms(written - arrived),
        )
        count(round_trips=len(turn.round_trips), tools=len(turn.tools))
        for exchange, trip in turn.round_trips.items():
            match trip:
                case _RoundTrip(unreached=str() as unreached):
                    outcome, error = "failed", unreached
                case _RoundTrip(status=int() as status) if status >= 400:
                    outcome, error = "failed", f"the API answered {status}"
                case _RoundTrip(status=None):
                    # No answer heard before the turn ended: the user spoke over it, or hands stopped it.
                    outcome, error = "cancelled", None
                case _:
                    outcome, error = "ok", None
            back = ended if trip.last is None else trip.last
            first_byte_ms = None if trip.first is None else _ms(trip.first - trip.left)
            child("model.round_trip", _at(trip.left), _ms(back - trip.left), outcome, error, exchange=exchange, status=trip.status, first_byte_ms=first_byte_ms)
        for call, ran in turn.tools.items():
            # A call whose result never left ran until the turn ended without it.
            outcome = "cancelled" if ran.answered is None else "failed" if ran.is_error else "ok"
            child("tool.call", _at(ran.ran), _ms((ended if ran.answered is None else ran.answered) - ran.ran), outcome, call=call, tool=ran.tool)

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

    async def _speak(self, turn: _Turn) -> None:
        await self.push_frame(LLMFullResponseStartFrame())
        while (words := await turn.said.get()) is not None:
            match words:
                case str():
                    turn.spoken.append(words)
                    await self.push_frame(LLMTextFrame(words))
                case Asked(permission=permission) as asked if asked.open:
                    # Said by hands, as its own sentence once the brain's words before it are: the response so far ends
                    # first, so what TTS holds of it is said ahead of the question.
                    await self.push_frame(LLMFullResponseEndFrame())
                    await self.push_frame(TTSSpeakFrame(brain_asks(permission)))
                    # [LAW:no-ambient-temporal-coupling] answerable once the speaker has played it to its end, which the
                    # mark is told of, and never if a barge-in cut it off.
                    await self.push_frame(Mark(partial(turn.hear, asked)))
                    # One question at a time, so what the user answers is the question they heard last.
                    await asyncio.wait({asked.decision})
                    await self.push_frame(LLMFullResponseStartFrame())
                case Asked():
                    # Settled before its turn came to be said, by the deadline, the turn's end, or an answer: nothing asks.
                    pass
        # Not on a barge-in, which cancels this: an end would have TTS say the sentence the user spoke over.
        await self.push_frame(LLMFullResponseEndFrame())

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
        # A tool whose effect must land runs to its end; stopped by the harness, it would land and be written in
        # history as refused. Its turn's next request is held instead, so the model is not asked to go on either way.
        running = tuple(turn.calls.values())
        turn.stopped = not any(self._completes(name) for name in running)
        self._record(BrainInterrupted(running, turn.stopped))
        return turn.stopped

    def route(self, sent: Sent) -> Route:
        """Where a request on the wire goes: each of a turn's own requests with hands' tail on it, and the next one after
        stay_silent or a barge-in held."""
        if sent.session != self._brain.session:
            return Send()
        # [LAW:single-enforcer] every request of the brain's is final, whatever its kind and whether a turn asked it: asked
        # again, Claude Code would keep the user waiting minutes on its retries, and a spent limit it would wait out to
        # continue the task on its own at the reset, hours on, with nobody asking (hands-wire-zi2).
        if not isinstance(sent.kind, MainTurn):
            return Send(refusal="final")
        turn = self._turn
        if turn is None:
            # [LAW:no-silent-failure] the brain asked the model something with no turn written to it: heard, never spoken.
            logger.warning(f"the brain sent a main turn (exchange {sent.exchange}) with no turn asked of it; nothing it says will be spoken")
            return Send(refusal="final")
        now = self._now()
        # A call's result leaving is the end of its run, whether the request carrying it goes on or is held.
        for answer in tool_answers(sent.body):
            if (ran := turn.tools.get(answer.call)) is not None and ran.answered is None:
                ran.answered, ran.is_error = now, answer.is_error
        # Only the calls this turn's last reply opened: a request carries every result of the brain's history.
        answers = [(self._tools.get(turn.calls[answer.call]), answer.text, _result(answer)) for answer in tool_answers(sent.body) if answer.call in turn.calls]
        # Said by hands once the brain's own words are, whatever the model does next: what a call hands hands to say.
        turn.readbacks.extend(says for _, _, result in answers if result is not None and isinstance(says := result.get("says"), str))
        if not (turn.interrupted or whole([tool is not None and result is not None and silent(tool, result) for tool, _, result in answers])):
            turn.exchanges.append(sent.exchange)
            turn.round_trips[sent.exchange] = _RoundTrip(now)
            turn.opening, turn.calls, turn.failure = {}, {}, _UNNAMED
            # Refused once is the turn's failure, said at once as the API variants say theirs, who ask once.
            return Send((Tail(self._tail()),), refusal="final")
        turn.readbacks.extend(said for tool, text, result in answers if tool is not None and tool.completes and (said := _owed(text, result)) is not None)
        return Hold(INTERRUPTED if turn.interrupted else SILENT)

    def _completes(self, name: str) -> bool:
        """Whether a barge-in lets the call finish, as hands' tools say: a call to a tool not hands' never does."""
        tool = self._tools.get(name)
        return tool is not None and tool.completes

    def hear(self, observed: Observed) -> None:
        turn = self._turn
        if turn is None:
            return
        now = self._now()
        # A reply's head and each of its events are heard before Claude Code reads them, so its round trip is timed by
        # them before the brain can end the turn on it.
        match observed:
            case Answering(exchange=exchange) if (trip := turn.round_trips.get(exchange)) is not None:
                trip.first = trip.last = now
            case Heard(exchange=exchange) | Exchanged(exchange=exchange, reply=Unreached()) if (trip := turn.round_trips.get(exchange)) is not None:
                trip.last = now
            case _:
                pass
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
                turn.tools[call] = _Call(name, now)
            # [LAW:one-source-of-truth] why a turn failed is the wire's, as the API variants read it off their own calls: the
            # head of its latest answer, heard before Claude Code reads any of it, or an API the proxy could not reach,
            # told before its 502. Its own requests are held final, so Claude Code asks once; a 401 it asks again after
            # refreshing its login: the latest request's is the turn's.
            case Answering(exchange=exchange, status=status, limit=limit) if exchange in turn.exchanges:
                turn.round_trips[exchange].status = status
                turn.failure = limit or ModelFailed(classify_http_status_code(status))
            case Exchanged(exchange=exchange, reply=Unreached(error=error)) if exchange in turn.exchanges:
                turn.round_trips[exchange].unreached = error
                turn.failure = ModelUnreachable()
            case _:
                pass


def _ms(seconds: Seconds) -> float:
    return round(seconds * 1000, 3)


def _at(seconds: Seconds) -> datetime:
    return datetime.fromtimestamp(seconds, UTC)


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
    """The note that tells the brain what the user heard of a turn the API broke off; none when nothing of it was said."""
    return f'[hands] The API broke off your last turn. The user heard you say "{spoken}", then that it failed. Say nothing about this unless the user asks.' if spoken else ""


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
