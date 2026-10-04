"""The user's turns as an API model is asked them: each with hands' notes beside it, where the brain's stage types the
same notes beside the words it hands the brain."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from hands.core.beside import beside
from hands.core.front import InFront
from hands.core.place import Modality
from hands.sessions.audit import Record
from hands.sessions.wide import annotate, unit


@dataclass
class TurnAsked(LLMContextFrame):
    """The model is asked a turn of the user's: the context, their words just written to it."""


class UserTurns(LLMUserAggregator):
    """The user's turns, each asking the model by a frame that says it is theirs: what hands tells of the sessions asks
    by Pipecat's own, so what follows can tell the two apart."""

    async def _push_aggregation(self, *, run_llm: bool = True) -> str:
        # Pipecat writes the turn and asks the model in one step, by the frame every ask is made by (1.10.0). Written
        # by Pipecat and asked here; a turn that heard nothing writes nothing, and asks nothing.
        said = await super()._push_aggregation(run_llm=False)
        if said and run_llm:
            await self.push_frame(TurnAsked(context=self.context))
        return said


class Noting(FrameProcessor):
    """Ahead of an API model's stage: each turn of the user's reaches the model with a note behind it of what was in
    front on the Mac's screen and whether the user could see one. What hands tells of the sessions passes un-noted."""

    def __init__(self, front: Callable[[], Awaitable[InFront]], modality: Callable[[], Modality], record: Record) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._front = front
        self._modality = modality
        self._record = record

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case TurnAsked(context=context):
                # [LAW:no-ambient-temporal-coupling] the screen is read where the model's request is made, behind the
                # user's turns and not within them: a turn of theirs that opens meanwhile drops this one's ask as it
                # would drop its request, and their words are asked again with that turn, noted as it arrives.
                context.add_message({"role": "user", "content": await self._noted()})
            case _:
                pass
        await self.push_frame(frame, direction)

    async def _noted(self) -> str:
        # [LAW:nothing-unseen] one event a turn noted: what was read, or why it was not, and how long the read took.
        with unit("voice.noted", self._record):
            # Read as the words arrive, as the brain's stage reads both.
            modality = self._modality()
            front = await self._front()
            annotate(front=front, modality=modality)
            return beside(front, modality)
