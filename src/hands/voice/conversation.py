"""The conversation with the intermediary, written to the audit log turn by turn as it enters the model's context."""

from collections.abc import Callable

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    FunctionCallCancelFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMAssistantPushAggregationFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMAssistantAggregator,
    LLMUserAggregator,
    LLMUserAggregatorParams,
    UserTurnMessageAddedMessage,
)
from pipecat.processors.frame_processor import FrameDirection

from hands.sessions.audit import Record, Replied, Transcribed
from hands.voice.turnstart import CutWritten


class AssistantTurns(LLMAssistantAggregator):
    """Pipecat's assistant aggregator, which also ends the turn of a line hands says as written and keeps in the
    context: written once it is said, as heard whole, or with the model's turn where one is open.

    [LAW:single-enforcer] the one place such a line's turn is ended, under either model, so whoever says a line sends
    the line alone. Pipecat's TTS service ends it only while it takes no reply as under way, and one that pushes its own
    text frames, as pocket-tts does, takes a reply as under way from its start frame to the next barge-in (1.10.0,
    `tts_service.py:858`): the line stayed an open turn, written at the next key press as cut off. After a barge-in it
    takes none as under way whatever the model's turn, so the end it sends is not taken inside one.

    [LAW:no-ambient-temporal-coupling] the extent of the model's turn is held here, where the context is written: open
    from a reply's start to its end, while a call it made is unanswered, and from a request for a reply, made or owed,
    until that reply starts. A line written inside it would follow the call's result in the context, and the
    request that answers the call would end on an assistant message, which Claude refuses as a prefill. So a line said
    inside the model's turn is written with it: by the end of the reply that answers the call, or, where the model is
    not run again, as the last of its calls is answered or cancelled.
    """

    _replying = False
    _answering = False
    # The calls the model made that are neither answered nor cancelled, read off the frames that say so: Pipecat's own
    # count keeps a call cancelled before its in-progress frame arrived, a frame the audio ahead of it holds back (1.10.0).
    _calls: frozenset[str] = frozenset()
    # How much of what is said and unwritten was said before the reply under way started: the lines it answers behind.
    _ahead = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        match frame:
            case LLMAssistantPushAggregationFrame() if self._open:
                # The TTS service's end of a line's turn, sent after a barge-in whatever the model's turn (1.10.0).
                return
            case LLMFullResponseStartFrame():
                self._replying, self._answering, self._ahead = True, False, len(self._aggregation)
            case LLMFullResponseEndFrame():
                self._replying = False
            case FunctionCallsStartedFrame(function_calls=calls):
                self._calls |= {call.tool_call_id for call in calls}
            case FunctionCallResultFrame(tool_call_id=call, properties=None | FunctionCallResultProperties(is_final=True)) | FunctionCallCancelFrame(tool_call_id=call):
                self._calls -= {call}
            case InterruptionFrame():
                # A barge-in drops the reply under way and the request for the one that would have answered a call. A
                # line waiting on either was heard whole, its text arriving behind its audio: written as that, ahead of
                # Pipecat writing a reply as cut off once a sentence of its own has been heard.
                if self._aggregation and not (self._replying and len(self._aggregation) > self._ahead):
                    await self._end_turn()
                self._replying = self._answering = False
            case _:
                pass
        await super().process_frame(frame, direction)
        match frame:
            case InterruptionFrame():
                # Written as cut off: the user's turn that made the cut is taken in behind it (`hands.voice.turnstart`).
                await self.push_frame(CutWritten(), FrameDirection.UPSTREAM)
            case _:
                pass
        # Whatever the frame, so whichever one closes the model's turn writes what was said inside it: a line itself,
        # which the output transport passes on once its audio has played, where no turn is open.
        if self._aggregation and not self._open:
            await self._end_turn()

    @property
    def _open(self) -> bool:
        # A result the model is to be run on once the speaker stops is a request owed (Pipecat's own note of it, 1.10.0).
        return self._replying or self._answering or self._push_context_on_bot_stopped_speaking or bool(self._calls)

    async def _end_turn(self) -> None:
        # Ended as Pipecat's own service ends a line's turn, by the frame it sends for one.
        await super().process_frame(LLMAssistantPushAggregationFrame(), FrameDirection.DOWNSTREAM)

    async def push_context_frame(self, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        # [LAW:one-source-of-truth] the context sent up to the model is the request for a reply, whatever asked for it:
        # read where it is made, never worked out again from what led to it.
        self._answering |= direction is FrameDirection.UPSTREAM
        await super().push_context_frame(direction)


def turns(context: LLMContext, user_params: LLMUserAggregatorParams) -> tuple[LLMUserAggregator, AssistantTurns]:
    """The two sides of the conversation over one context, wired as Pipecat's `LLMContextAggregatorPair` wires its own
    (1.10.0), which builds no assistant side but Pipecat's."""
    user = LLMUserAggregator(context, params=user_params)
    return user, AssistantTurns(context, _paired_user_aggregator=user)


def record_turns(user_turns: LLMUserAggregator, assistant_turns: LLMAssistantAggregator, record: Record) -> None:
    """Write each user transcript and each reply as its turn is added to the context."""

    # Added to the context, not merely stopped: the user text is final here, and it is what the model read.
    @user_turns.event_handler("on_user_turn_message_added")
    async def heard(_aggregator: LLMUserAggregator, message: UserTurnMessageAddedMessage) -> None:  # pyright: ignore[reportUnusedFunction]
        record(Transcribed(message.content))
        # The words themselves, in the terminal as well as the audit log: the latency line says only that a
        # transcript came, and whoever is talking wants to see what was heard: the plain words, on one line.
        logger.info(f"heard: {' '.join(message.content.split())}")

    @assistant_turns.event_handler("on_assistant_turn_stopped")
    async def replied(_aggregator: LLMAssistantAggregator, message: AssistantTurnStoppedMessage) -> None:  # pyright: ignore[reportUnusedFunction]
        match message:
            case AssistantTurnStoppedMessage(content="", interrupted=False):
                # A turn that only called a tool said nothing; its tool's event is the record of it.
                pass
            case _:
                record(Replied(message.content, message.interrupted))


def cue_receipt(user_turns: LLMUserAggregator, received: Callable[[], None]) -> None:
    """Tell `received` of each user turn as its words are written to the context: a turn that heard nothing writes none."""

    @user_turns.event_handler("on_user_turn_message_added")
    async def written(_aggregator: LLMUserAggregator, _message: UserTurnMessageAddedMessage) -> None:  # pyright: ignore[reportUnusedFunction]
        received()
