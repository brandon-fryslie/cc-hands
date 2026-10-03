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
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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
)
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage, LLMStandardMessage
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.errors import ErrorCategory, classify_http_status_code

from hands.brain.mcp import SERVER_NAME
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
    Send,
    Sent,
    Tail,
    TextDelta,
    ToolAnswer,
    Unreached,
    tool_answers,
)
from hands.sessions.model_facts import ModelFact, ModelFailed, ModelFault, ModelReplyEmpty, ModelUnreachable
from hands.sessions.audit import Asker, BrainAnswered, BrainInterrupted, BrainSpoke, Record
from hands.voice.speech import Aloud, Narrated
from hands.voice.tools import Result, Tool, silent


class Asking(Protocol):
    """What the stage needs of the brain: the session its requests carry, a turn asked, and a turn told to stop."""

    @property
    def session(self) -> SessionId: ...

    async def ask(self, text: str) -> BrainAnswered: ...

    def interrupt(self) -> None: ...


# What the brain is recorded as saying for a request hands held. Never spoken: a held request joins no turn.
SILENT = "(stayed silent)"
INTERRUPTED = "(the user spoke over this reply, and nothing more of it was said)"


def wire_name(tool: Tool) -> str:
    """The name a hands tool has on the wire, where the brain calls it through hands' MCP server."""
    return f"mcp__{SERVER_NAME}__{tool.name}"


_UNNAMED = ModelFailed(ErrorCategory.UNKNOWN)


@dataclass
class _Turn:
    """One question to the brain, from its write to stdin to its result line."""

    # Words heard on the wire and not yet handed to the speaker; None once the brain has said the turn is over.
    said: asyncio.Queue[str | None]
    # Hands the words on to TTS until the turn is over or the user barges in.
    speaking: asyncio.Task[None]
    spoken: list[str]
    # The requests on the wire that are this turn's own, in the order they left.
    exchanges: list[str] = field(default_factory=list[str])
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
    # Told to stop by hands, which Claude Code 2.1.285 ends with an error_during_execution result: asked for, not a failure.
    stopped: bool = False
    # What the turn failed of, if it fails, as its latest request's answer told it: nothing named until that answer says.
    failure: ModelFact = _UNNAMED

    def empty(self) -> bool:
        """The model answered the turn, the user did not speak over it, and nothing it answered said or did anything."""
        return bool(self.exchanges) and not (self.interrupted or self.replied)


class BrainStage(FrameProcessor):
    """The LLM stage under the brain: a context in, the brain's words out as LLM text frames, and a barge-in passed on."""

    def __init__(self, brain: Asking, tools: Sequence[Tool], tail: Callable[[], str], record: Record, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._brain = brain
        # What hands appends to each request of a turn, composed as that request leaves.
        self._tail = tail
        self._record = record
        self._now = clock
        self._completes = frozenset(wire_name(tool) for tool in tools if tool.completes)
        self._tools = {wire_name(tool): tool for tool in tools}
        # How many of the context's messages the brain has been handed: the rest are new to it.
        self._told = 0
        # [LAW:no-ambient-temporal-coupling] the turn is the brain's, from its write to its result line, not the pipeline's:
        # it runs beside the frames passing through, a barge-in ends what is said of it, and the next is written only
        # once the brain has ended it. What waits is in two lanes, the user's and hands', and the user's goes first.
        # Each thing waiting is kept with when it arrived. Hands' lane holds what it says as written beside what it
        # hands the brain, so a session's story is heard in the order it happened.
        self._contexts: deque[tuple[LLMContext, float]] = deque()
        self._hands: deque[tuple[Narrated | Aloud, float]] = deque()
        self._waiting = asyncio.Event()
        self._turn: _Turn | None = None
        # What the user heard of the last turn before the API broke it off, told to the brain with its next turn: Claude
        # Code keeps the broken reply out of the brain's history (2.1.285), so without it the brain cannot answer about it.
        self._broken_off = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case LLMContextFrame(context=context):
                self._contexts.append((context, self._now()))
                self._waiting.set()
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
            waiting, arrived = await self._upcoming()
            match waiting:
                case LLMContext() as context:
                    # A context frame is a call to answer, not a message: one whose messages an earlier turn already took asks nothing.
                    if text := self._news(context):
                        await self._ask(text, "user", (), arrived)
                case Narrated(text=text, unsaid=unsaid):
                    await self._ask(text, "hands", (unsaid,), arrived)
                case Aloud(spoken=spoken):
                    await self.push_frame(spoken)

    async def _upcoming(self) -> tuple[LLMContext | Narrated | Aloud, float]:
        """What is next: the user's words while any wait, since what they said goes ahead of what hands has to tell."""
        while not (self._contexts or self._hands):
            self._waiting.clear()
            await self._waiting.wait()
        return self._contexts.popleft() if self._contexts else self._hands.popleft()

    async def _ask(self, text: str, asker: Asker, unsaid: Sequence[str], arrived: float) -> None:
        """One turn of the brain's; `unsaid` is what hands says as written if the brain cannot take it."""
        waited = self._now() - arrived
        note, self._broken_off = self._broken_off, ""
        text = "\n\n".join(part for part in (note, text) if part)
        said: asyncio.Queue[str | None] = asyncio.Queue()
        spoken: list[str] = []
        turn = self._turn = _Turn(said, asyncio.create_task(self._speak(said, spoken), name="the brain's words"), spoken)
        try:
            asked = asyncio.ensure_future(self._brain.ask(text))
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
        self._record(BrainSpoke(tuple(turn.exchanges), "".join(turn.spoken), tuple(turn.readbacks), turn.interrupted, asker, waited, None if failure is None else failure.fact))
        if (error := asked.exception()) is not None:
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
            await self._unsaid(unsaid)
            await self.push_error(failure.error, exception=ModelFault(failure.fact))  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)

    async def _unsaid(self, unsaid: Sequence[str]) -> None:
        """What hands had for the brain to tell, said as written since the brain did not: a system fact, kept out of the context."""
        for line in unsaid:
            await self.push_frame(TTSSpeakFrame(line, append_to_context=False))

    async def _speak(self, said: asyncio.Queue[str | None], spoken: list[str]) -> None:
        await self.push_frame(LLMFullResponseStartFrame())
        while (words := await said.get()) is not None:
            spoken.append(words)
            await self.push_frame(LLMTextFrame(words))
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
        if turn is None or turn.interrupted or not turn.exchanges:
            return False
        turn.interrupted = True
        # Cancelled before anything else runs, so no word of the turn follows the barge-in down the pipeline.
        turn.speaking.cancel()
        # A tool whose effect must land runs to its end; stopped by the harness, it would land and be written in
        # history as refused. Its turn's next request is held instead, so the model is not asked to go on either way.
        running = tuple(turn.calls.values())
        turn.stopped = not any(name in self._completes for name in running)
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
        # Only the calls this turn's last reply opened: a request carries every result of the brain's history.
        answers = [(turn.calls[answer.call], answer) for answer in tool_answers(sent.body) if answer.call in turn.calls]
        if not (turn.interrupted or any(self._silent(name, answer) for name, answer in answers)):
            turn.exchanges.append(sent.exchange)
            turn.opening, turn.calls, turn.failure = {}, {}, _UNNAMED
            # Refused once is the turn's failure, said at once as the API variants say theirs, who ask once.
            return Send((Tail(self._tail()),), refusal="final")
        turn.readbacks.extend(said for name, answer in answers if name in self._completes and (said := _owed(answer)) is not None)
        return Hold(INTERRUPTED if turn.interrupted else SILENT)

    def _silent(self, name: str, answer: ToolAnswer) -> bool:
        """Whether the call was the whole reply, as hands' tools say: a call to a tool not hands' never is."""
        tool, result = self._tools.get(name), _result(answer)
        return tool is not None and result is not None and silent(tool, result)

    def hear(self, observed: Observed) -> None:
        turn = self._turn
        if turn is None:
            return
        match observed:
            # Said as it arrives, and never twice: a reply the API breaks mid-stream is not asked for again, streamed or
            # not; the turn ends in StopFailure, with the broken reply kept out of the brain's history (2.1.285, hands-wire-6ic.6dz).
            case Heard(exchange=exchange, event=BlockStarted(block={"type": "text"})) if exchange in turn.exchanges:
                turn.ahead = turn.between
            case Heard(exchange=exchange, event=TextDelta(text=text)) if exchange in turn.exchanges:
                turn.said.put_nowait(turn.ahead + text)
                turn.ahead, turn.between = "", "\n\n"
                turn.replied = turn.replied or bool(text.strip())
            case Heard(exchange=exchange, event=BlockStarted(index=index, block={"type": "tool_use", "id": str() as call, "name": str() as name})) if exchange in turn.exchanges:
                turn.opening[index] = (call, name)
                turn.replied = True
            # Claude Code runs a call once its block is whole, before the reply's last byte, and hands hears each byte
            # before Claude Code does: the call is running from here on.
            case Heard(exchange=exchange, event=BlockStopped(index=index)) if exchange in turn.exchanges and index in turn.opening:
                call, name = turn.opening.pop(index)
                turn.calls[call] = name
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


def _owed(answer: ToolAnswer) -> str | None:
    """What a call that must land handed back for the user and nobody has said: its readback, or why it failed, which
    the model would have said. None when hands said it as the call ran."""
    match _result(answer):
        case {"said": str()}:
            return None
        case {"readback": str() as said} | {"error": str() as said}:
            return said
        case _:
            return answer.text
