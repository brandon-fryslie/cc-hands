"""The conversation with the brain, written to the audit log turn by turn as each side's aggregator writes it."""

from collections.abc import Callable

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMAssistantPushAggregationFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage
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
    """Pipecat's assistant aggregator, which also ends the turn of a line hands says as written and records as a reply:
    written once it is said, as heard whole, or with the brain's reply where one is open.

    [LAW:single-enforcer] the one place such a line's turn is ended, so whoever says a line sends the line alone.
    Pipecat's TTS service ends it only while it takes no reply as under way, and one that pushes its own text frames, as
    pocket-tts does, takes a reply as under way from its start frame to the next barge-in (1.10.0, `tts_service.py:858`):
    the line stayed an open turn, written at the next key press as cut off. After a barge-in it takes none as under way
    whatever the brain's reply, so the end it sends is not taken inside one.

    [LAW:no-ambient-temporal-coupling] the extent of the brain's reply is held here, where the turn is written: open
    from its start frame to its end frame. A line said inside it is written with it, as the reply ends.
    """

    _replying = False
    # How much of what is said and unwritten was said before the reply under way started: the lines it answers behind.
    _ahead = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        match frame:
            case LLMAssistantPushAggregationFrame() if self._replying:
                # The TTS service's end of a line's turn, sent after a barge-in whatever the brain's reply (1.10.0).
                return
            case LLMFullResponseStartFrame():
                self._replying, self._ahead = True, len(self._aggregation)
            case LLMFullResponseEndFrame():
                self._replying = False
            case InterruptionFrame():
                # A barge-in drops the reply under way. A line waiting on it was heard whole, its text arriving behind its
                # audio: written as that, ahead of Pipecat writing a reply as cut off once a sentence of its own has been heard.
                if self._aggregation and not (self._replying and len(self._aggregation) > self._ahead):
                    await self._end_turn()
                self._replying = False
            case _:
                pass
        await super().process_frame(frame, direction)
        match frame:
            case InterruptionFrame():
                # Written as cut off: the user's turn that made the cut is taken in behind it (`hands.voice.turnstart`).
                await self.push_frame(CutWritten(), FrameDirection.UPSTREAM)
            case _:
                pass
        # Whatever the frame, so whichever one closes the brain's reply writes what was said inside it: a line itself,
        # which the output transport passes on once its audio has played, where no reply is open.
        if self._aggregation and not self._replying:
            await self._end_turn()

    async def _end_turn(self) -> None:
        # Ended as Pipecat's own service ends a line's turn, by the frame it sends for one.
        await super().process_frame(LLMAssistantPushAggregationFrame(), FrameDirection.DOWNSTREAM)


class _Unkept(LLMContext):
    """The assistant side's context, which keeps none of what is said: the brain keeps its own history and is never handed
    a reply back, and the audit log's record of each one is the event the aggregator raises, built from what it heard."""

    def add_message(self, message: LLMContextMessage) -> None:
        pass


def turns(context: LLMContext, user_params: LLMUserAggregatorParams) -> tuple[LLMUserAggregator, AssistantTurns]:
    """The two sides of the conversation, wired as Pipecat's `LLMContextAggregatorPair` wires its own (1.10.0), which builds
    no assistant side but Pipecat's: the user's words go into `context`, and what is said goes into none."""
    user = LLMUserAggregator(context, params=user_params)
    return user, AssistantTurns(_Unkept(), _paired_user_aggregator=user)


def record_turns(user_turns: LLMUserAggregator, assistant_turns: LLMAssistantAggregator, record: Record) -> None:
    """Write each user transcript and each reply as its turn is written."""

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
