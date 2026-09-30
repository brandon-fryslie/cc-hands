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
from typing import Protocol

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

from hands.brain.mcp import SERVER_NAME
from hands.core.session import SessionId
from hands.core.wire import (
    BlockStarted,
    BlockStopped,
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
    tool_answers,
)
from hands.sessions.audit import Asker, BrainAnswered, BrainInterrupted, BrainSpoke, Record
from hands.voice.speech import Aloud, Narrated
from hands.voice.tools import Tool


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
    # The call blocks the reply streaming now has opened and not yet closed, by index: a call is not run until it is whole.
    opening: dict[int, tuple[str, str]] = field(default_factory=dict[int, tuple[str, str]])
    # The calls the turn's last reply made whole, by id, which run until its next request leaves.
    calls: dict[str, str] = field(default_factory=dict[str, str])
    # What hands says for the calls of a held request once the turn's own words are said, since the model is not asked to.
    readbacks: list[str] = field(default_factory=list[str])
    interrupted: bool = False
    # Told to stop by hands, which Claude Code 2.1.285 ends with an error_during_execution result: asked for, not a failure.
    stopped: bool = False


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
        self._silences = frozenset(wire_name(tool) for tool in tools if tool.then == "silence")
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
        self._record(BrainSpoke(tuple(turn.exchanges), "".join(turn.spoken), tuple(turn.readbacks), turn.interrupted, asker, waited))
        if (error := asked.exception()) is not None:
            # [LAW:no-silent-failure] said as the turn's failure whatever failed it: a brain that is gone also stops the run
            # from its own watch, but one that never took the turn, or could not be typed into, is still running.
            logger.opt(exception=error).error("the brain failed a turn")
            # A turn the brain never took did not tell it what the user heard: the turn after it does.
            self._broken_off = note
            await self._unsaid(unsaid)
            await self.push_error(f"the brain failed a turn: {error}")  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        elif (failed := asked.result().error) is not None and not turn.stopped:
            # [LAW:no-silent-failure] said as the API services' failures are: an error from the model's stage.
            self._broken_off = _broken_off("".join(turn.spoken))
            await self._unsaid(unsaid)
            await self.push_error(f"the brain's turn ended in error: {failed}")  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)

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
        if sent.session != self._brain.session or not isinstance(sent.kind, MainTurn):
            return Send()
        turn = self._turn
        if turn is None:
            # [LAW:no-silent-failure] the brain asked the model something with no turn written to it: heard, never spoken.
            logger.warning(f"the brain sent a main turn (exchange {sent.exchange}) with no turn asked of it; nothing it says will be spoken")
            return Send()
        # Only the calls this turn's last reply opened: a request carries every result of the brain's history.
        answers = [(turn.calls[answer.call], answer) for answer in tool_answers(sent.body) if answer.call in turn.calls]
        if not (turn.interrupted or any(name in self._silences for name, _ in answers)):
            turn.exchanges.append(sent.exchange)
            turn.opening, turn.calls = {}, {}
            return Send((Tail(self._tail()),))
        turn.readbacks.extend(_said(answer) for name, answer in answers if name in self._completes)
        return Hold(INTERRUPTED if turn.interrupted else SILENT)

    def hear(self, observed: Observed) -> None:
        turn = self._turn
        if turn is None:
            return
        match observed:
            # Said as it arrives, and never twice: a reply the API breaks mid-stream is not asked for again, streamed or
            # not; the turn ends in StopFailure, with the broken reply kept out of the brain's history (2.1.285, hands-wire-6ic.6dz).
            case Heard(exchange=exchange, event=TextDelta(text=text)) if exchange in turn.exchanges:
                turn.said.put_nowait(text)
            case Heard(exchange=exchange, event=BlockStarted(index=index, block={"type": "tool_use", "id": str() as call, "name": str() as name})) if exchange in turn.exchanges:
                turn.opening[index] = (call, name)
            # Claude Code runs a call once its block is whole, before the reply's last byte, and hands hears each byte
            # before Claude Code does: the call is running from here on.
            case Heard(exchange=exchange, event=BlockStopped(index=index)) if exchange in turn.exchanges and index in turn.opening:
                call, name = turn.opening.pop(index)
                turn.calls[call] = name
            case _:
                pass


def _user_text(message: LLMStandardMessage) -> str:
    content = message.get("content")
    if not isinstance(content, str):
        # [LAW:no-silent-failure] hands' aggregator and notes write plain text; anything else is a change to hear about.
        raise TypeError(f"a user message in the context is not plain text: {message!r}")
    return content


def _broken_off(spoken: str) -> str:
    """The note that tells the brain what the user heard of a turn the API broke off; none when nothing of it was said."""
    return f'[hands] The API broke off your last turn. The user heard you say "{spoken}", then that it failed. Say nothing about this unless the user asks.' if spoken else ""


def _said(answer: ToolAnswer) -> str:
    """What a call that must land handed back for the user: its readback, or why it failed, which the model would have said."""
    try:
        result: object = json.loads(answer.text)
    except ValueError:
        # The MCP server's own failure: a line of text naming the tool and what went wrong.
        return answer.text
    match result:
        case {"readback": str() as said} | {"error": str() as said}:
            return said
        case _:
            return answer.text
