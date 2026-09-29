"""The brain as the pipeline's LLM stage: the user's turn goes to the brain's stdin, and its words come off the wire.

Pipecat is otherwise untouched: what reaches this stage is the context the user aggregator built, and what leaves it are
the frames an LLM service pushes, which TTS already consumes. What the brain says is read from its requests on the
wire as the proxy hears them, never from its stdout [the design's rule: primary facts from the wire]: text deltas of a
main turn under the brain's own session, and of no other request.

Two things the brain's harness would do on its own are held here instead. Claude Code sends every tool result back for
another response, so a turn that called stay_silent, or one the user barged in on, has its next request answered by
hands with a line that records why: nothing more is said, and the model is not asked to go on. And a barge-in while a tool
whose effect must land is running lets that tool finish rather than having Claude Code cancel it and write it into
history as refused; its readback, which the model will not be asked to say, is spoken from the tool's result.
"""

import asyncio
import json
from collections.abc import Sequence
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
    Exchanged,
    Forward,
    Heard,
    Hold,
    MainTurn,
    Observed,
    Reached,
    Route,
    Sent,
    Streamed,
    TextDelta,
    ToolAnswer,
    ToolUse,
    tool_answers,
)
from hands.sessions.audit import BrainAnswered, BrainInterrupted, BrainSpoke, Record
from hands.voice.tools import Tool


class Asking(Protocol):
    """What the stage needs of the brain: the session its requests carry, a turn asked, and a turn told to stop."""

    @property
    def session(self) -> SessionId: ...

    async def ask(self, text: str) -> BrainAnswered: ...

    async def interrupt(self) -> None: ...


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
    said: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue[str | None])
    # The requests on the wire that are this turn's own, in the order they left.
    exchanges: list[str] = field(default_factory=list[str])
    # The tools the turn's last reply called, which run until its next request leaves.
    running: tuple[str, ...] = ()
    interrupted: bool = False
    spoken: list[str] = field(default_factory=list[str])


class BrainStage(FrameProcessor):
    """The LLM stage under the brain: a context in, the brain's words out as LLM text frames, and a barge-in passed on."""

    def __init__(self, brain: Asking, tools: Sequence[Tool], record: Record) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._brain = brain
        self._record = record
        self._completes = frozenset(wire_name(tool) for tool in tools if tool.completes)
        self._silences = frozenset(wire_name(tool) for tool in tools if tool.then == "silence")
        # How many of the context's messages the brain has been handed: the rest are new to it.
        self._told = 0
        # [LAW:no-ambient-temporal-coupling] the turn is the brain's, from its write to its result line, not the pipeline's:
        # a barge-in ends what is said of it, and the next turn is written only once the brain has ended it.
        self._turn: _Turn | None = None
        self._asked: asyncio.Future[BrainAnswered] | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case LLMContextFrame(context=context):
                await self._answer(context)
            case InterruptionFrame():
                await self._barge_in()
                await self.push_frame(frame, direction)
            case _:
                await self.push_frame(frame, direction)

    async def _answer(self, context: LLMContext) -> None:
        text = self._news(context)
        if self._asked is not None:
            await asyncio.wait({self._asked})
        turn = self._turn = _Turn()
        asked = self._asked = asyncio.ensure_future(self._brain.ask(text))
        asked.add_done_callback(lambda done: self._over(turn, done))
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            # A barge-in cancels this loop with Pipecat's own interruption, and the turn stops being spoken.
            while (words := await turn.said.get()) is not None:
                turn.spoken.append(words)
                await self.push_frame(LLMTextFrame(words))
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

    def _news(self, context: LLMContext) -> str:
        """What the context gained since the brain last heard it, as one message: the user's words and hands' notes."""
        messages = context.get_messages()
        fresh = messages[self._told :]
        self._told = len(messages)
        # [LAW:one-source-of-truth] the brain keeps its own history, so what it said itself, which the assistant
        # aggregator writes back into this context, is never handed to it again.
        return "\n\n".join(_user_text(message) for message in fresh if not isinstance(message, LLMSpecificMessage) and message.get("role") == "user")

    def _over(self, turn: _Turn, asked: asyncio.Future[BrainAnswered]) -> None:
        if self._turn is turn:
            self._turn = None
        turn.said.put_nowait(None)
        self._record(BrainSpoke(tuple(turn.exchanges), "".join(turn.spoken), turn.interrupted))
        if not asked.cancelled() and (error := asked.exception()) is not None:
            # [LAW:no-silent-failure] a brain that is gone stops the run from its own watch; this says which turn it took.
            logger.opt(exception=error).error("the brain failed a turn")

    async def _barge_in(self) -> None:
        turn = self._turn
        if turn is None or turn.interrupted:
            return
        turn.interrupted = True
        # A tool whose effect must land runs to its end; stopped by the harness, it would land and be written in
        # history as refused. Its turn's next request is held instead, so the model is not asked to go on either way.
        stopped = not any(name in self._completes for name in turn.running)
        self._record(BrainInterrupted(turn.running, stopped))
        if stopped:
            await self._brain.interrupt()

    def route(self, sent: Sent) -> Route:
        """Where a request on the wire goes: the brain's own next request after stay_silent or a barge-in is held."""
        if sent.session != self._brain.session or not isinstance(sent.kind, MainTurn):
            return Forward()
        turn = self._turn
        if turn is None:
            # [LAW:no-silent-failure] the brain asked the model something with no turn written to it: heard, never spoken.
            logger.warning(f"the brain sent a main turn (exchange {sent.exchange}) with no turn asked of it; nothing it says will be spoken")
            return Forward()
        answers = tool_answers(sent.body)
        silenced = any(answer.name in self._silences for answer in answers)
        if not (turn.interrupted or silenced):
            turn.exchanges.append(sent.exchange)
            turn.running = ()
            return Forward()
        for readback in _readbacks(answers, self._completes):
            # Said by hands, since the model that would have said it is not asked to go on.
            self.create_task(self.push_frame(TTSSpeakFrame(readback)), "readback")  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        return Hold(INTERRUPTED if turn.interrupted else SILENT)

    def hear(self, observed: Observed) -> None:
        turn = self._turn
        if turn is None:
            return
        match observed:
            case Heard(exchange=exchange, event=TextDelta(text=text)) if exchange in turn.exchanges:
                turn.said.put_nowait(text)
            case Exchanged(exchange=exchange, reply=Reached(body=Streamed(message=message))) if exchange in turn.exchanges:
                turn.running = tuple(block.name for block in message.content if isinstance(block, ToolUse))
            case _:
                pass


def _user_text(message: LLMStandardMessage) -> str:
    content = message.get("content")
    if not isinstance(content, str):
        # [LAW:no-silent-failure] hands' aggregator and notes write plain text; anything else is a change to hear about.
        raise TypeError(f"a user message in the context is not plain text: {message!r}")
    return content


def _readbacks(answers: Sequence[ToolAnswer], completes: frozenset[str]) -> list[str]:
    """The readback each tool that must land handed back, as its body wrote it."""
    spoken: list[str] = []
    for answer in (answer for answer in answers if answer.name in completes):
        try:
            readback = json.loads(answer.text)["readback"]
        except (ValueError, KeyError, TypeError):
            # [LAW:no-silent-failure] a failed call hands back its error, not a readback, and nobody is asked to say it.
            logger.warning(f"{answer.name} answered without a readback to speak (is_error {answer.is_error}): {answer.text[:200]!r}")
            continue
        spoken.append(str(readback))
    return spoken
