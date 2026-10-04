"""The user's turns as an API model's context takes them: each with hands' notes beside it, where the brain's stage
types the same notes beside the words it hands the brain."""

from collections.abc import Awaitable, Callable

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator, LLMUserAggregatorParams

from hands.core.beside import beside
from hands.core.front import InFront
from hands.core.place import Modality
from hands.sessions.audit import Record
from hands.sessions.wide import annotate, unit

# What writes the user's turns into the context the pipeline is built on.
UserTurns = Callable[[LLMContext, LLMUserAggregatorParams], LLMUserAggregator]


def unnoted(context: LLMContext, params: LLMUserAggregatorParams) -> LLMUserAggregator:
    """The user's turns under the brain, whose stage notes each one itself as it types it."""
    return LLMUserAggregator(context, params=params)


class NotedTurns(LLMUserAggregator):
    """The user's turns under an API model: behind each one written to the context goes a note of what was in front on
    the Mac's screen and whether the user could see one, and the model is asked with both. What hands tells of the
    sessions enters the context by another door, so no note follows it."""

    def __init__(self, context: LLMContext, params: LLMUserAggregatorParams, front: Callable[[], Awaitable[InFront]], modality: Callable[[], Modality], record: Record) -> None:
        super().__init__(context, params=params)  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        self._front = front
        self._modality = modality
        self._record = record

    async def _push_aggregation(self, *, run_llm: bool = True) -> str:
        # [LAW:no-ambient-temporal-coupling] Pipecat writes the turn and asks the model in one step (1.10.0). Written
        # here and not yet asked, so the note is in the context before any request is made of it. A turn that heard
        # nothing writes nothing, and there is nothing to note.
        said = await super()._push_aggregation(run_llm=False)
        if said:
            self.add_messages([{"role": "user", "content": await self._noted()}])  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
            if run_llm:
                await self.push_context_frame()
        return said

    async def _noted(self) -> str:
        # [LAW:nothing-unseen] one event a turn noted: what was read, or why it was not, and how long the read took.
        with unit("voice.asked", self._record):
            # Read as the words are written, as the brain's stage reads both as they arrive.
            modality = self._modality()
            front = await self._front()
            annotate(front=front, modality=modality)
            return beside(front, modality)
