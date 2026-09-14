"""The conversation with the intermediary, written to the audit log turn by turn as it enters the model's context."""

from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMAssistantAggregator,
    LLMUserAggregator,
    UserTurnMessageAddedMessage,
)

from hands.sessions.audit import Record, Replied, Transcribed


def record_turns(user_turns: LLMUserAggregator, assistant_turns: LLMAssistantAggregator, record: Record) -> None:
    """Write each user transcript and each reply as its turn is added to the context."""

    # Added to the context, not merely stopped: the user text is final here, and it is what the model read.
    @user_turns.event_handler("on_user_turn_message_added")
    async def heard(_aggregator: LLMUserAggregator, message: UserTurnMessageAddedMessage) -> None:  # pyright: ignore[reportUnusedFunction]
        record(Transcribed(message.content))

    @assistant_turns.event_handler("on_assistant_turn_stopped")
    async def replied(_aggregator: LLMAssistantAggregator, message: AssistantTurnStoppedMessage) -> None:  # pyright: ignore[reportUnusedFunction]
        match message:
            case AssistantTurnStoppedMessage(content="", interrupted=False):
                # A turn that only called a tool said nothing; its Called line is the record of it.
                pass
            case _:
                record(Replied(message.content, message.interrupted))
