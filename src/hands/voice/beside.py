"""The user's words as an API model's context takes them: each hold's with hands' notes ahead of it, where the brain's
stage types the same notes beside the words it hands the brain."""

from collections.abc import Awaitable, Callable

from pipecat.frames.frames import Frame, LLMMessagesAppendFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from hands.core.beside import beside
from hands.core.front import InFront
from hands.core.place import Modality
from hands.sessions.audit import Record
from hands.sessions.wide import annotate, unit


class Noting(FrameProcessor):
    """Ahead of an API model's user aggregator: as a hold's words arrive, a note of what was in front on the Mac's
    screen and whether the user could see one goes into the context ahead of them."""

    def __init__(self, front: Callable[[], Awaitable[InFront]], modality: Callable[[], Modality], record: Record) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._front = front
        self._modality = modality
        self._record = record

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case TranscriptionFrame():
                # [LAW:no-ambient-temporal-coupling] the screen is read with the words still on their way to the turn
                # that writes them: that the hold is done comes behind them, so the turn stays open through the read,
                # a key pressed meanwhile joins it, and the floor holds what hands tells until it has ended. The note
                # is in the context before the user's words, whatever the read took.
                await self.push_frame(LLMMessagesAppendFrame([{"role": "user", "content": await self._noted()}], run_llm=False))
            case _:
                pass
        await self.push_frame(frame, direction)

    async def _noted(self) -> str:
        # [LAW:nothing-unseen] one event a hold noted: what was read, or why it was not, and how long the read took.
        with unit("voice.noted", self._record):
            # Read as the words arrive, as the brain's stage reads both.
            modality = self._modality()
            front = await self._front()
            annotate(front=front, modality=modality)
            return beside(front, modality)
