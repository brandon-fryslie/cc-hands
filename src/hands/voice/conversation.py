"""The conversation with the intermediary, written to the audit log turn by turn as it enters the model's context."""

from collections.abc import Callable

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMAssistantPushAggregationFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
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


class AssistantTurns(LLMAssistantAggregator):
    """Pipecat's assistant aggregator, which also ends the turn of a line hands says as written and keeps in the
    context: written once it is said, as heard whole, or with the model's turn where one is open.

    [LAW:single-enforcer] the one place such a line's turn is ended, under either model, so whoever says a line sends
    the line alone. Pipecat's TTS service ends it only while it takes no reply as under way, and one that pushes its own
    text frames, as pocket-tts does, takes a reply as under way from its start frame to the next barge-in (1.10.0,
    `tts_service.py:858`): the line stayed an open turn, written at the next key press as cut off.

    [LAW:no-ambient-temporal-coupling] the extent of the model's turn is held here, where the context is written: open
    from a reply's start to its end, while a call it made is in progress, and from a result the model is run on until
    the reply that answers it starts. A line written inside it would follow the call's result in the context, and the
    request that answers the call would end on an assistant message, which Claude refuses as a prefill. So a line said
    inside the model's turn is written with it: by the end of the reply that answers the call, or, where the model is
    not run again, as the last of its calls is answered.
    """

    _replying = False
    _answering = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        match frame:
            case LLMFullResponseStartFrame():
                self._replying, self._answering = True, False
            case LLMFullResponseEndFrame():
                self._replying = False
            case InterruptionFrame():
                # A barge-in drops the reply under way and the request for the one that would have answered a call.
                self._replying = self._answering = False
            case _:
                pass
        await super().process_frame(frame, direction)
        match frame:
            # The two arrivals that can leave something said and unwritten with no turn of the model's open: the line
            # itself, which the output transport passes on once its audio has played, and the result that ends a turn.
            case TextFrame() | FunctionCallResultFrame() if self.aggregation_string() and not self._open:
                # Ended as Pipecat's own service ends a line's turn, by the frame it sends for one.
                await super().process_frame(LLMAssistantPushAggregationFrame(), direction)
            case _:
                pass

    @property
    def _open(self) -> bool:
        return self._replying or self._answering or self.has_function_calls_in_progress

    async def _maybe_push_context_after_function_result(self) -> None:
        # [LAW:one-source-of-truth] Pipecat's aggregator decides here, and nowhere else, that the model is run on a
        # call's result, at once or once the speaker stops (1.10.0): read where it is decided, never worked out again.
        self._answering = True
        await super()._maybe_push_context_after_function_result()


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
