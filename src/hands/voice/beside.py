"""The user's words as an API model's context takes them: with hands' notes ahead of them, where the brain's stage types
the same notes beside the words it hands the brain."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pipecat.frames.frames import Frame, LLMMessagesAppendFrame, TranscriptionFrame, UninterruptibleFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from hands.core.beside import beside
from hands.core.front import FrontUnread, InFront
from hands.core.place import Modality
from hands.sessions.audit import Record
from hands.sessions.wide import annotate, unit
from hands.voice.turnstop import HoldDiscarded

_NOT_YET_READ = FrontUnread("the screen was still being read as the user's words arrived")


@dataclass(frozen=True)
class _Read:
    """What was read as the user let go of the key: whether they could see a screen, and the read of the Mac's, under way."""

    modality: Modality
    front: asyncio.Task[InFront]


class Note(LLMMessagesAppendFrame, UninterruptibleFrame):
    """The note beside the user's words: theirs as the words are, so the interruption that lands behind the words of a
    turn the voice opened drops neither."""


class Noting(FrameProcessor):
    """Ahead of an API model's user aggregator: a note of what was in front on the Mac's screen and whether the user
    could see one goes into the context ahead of the words they said."""

    def __init__(self, front: Callable[[], Awaitable[InFront]], modality: Callable[[], Modality], record: Record) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._front = front
        self._modality = modality
        self._record = record
        # [LAW:no-ambient-temporal-coupling] the screen is read beside the pipeline, from the key let go to the words
        # Whisper makes of the hold, and no frame waits on it: words that arrive first are noted without it. The read of
        # the last hold let go and not yet noted, so holds let go before the first one's words arrive are noted once.
        self._read: _Read | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        match frame, self._read:
            case HoldDiscarded(), _:
                pass
            case VADUserStoppedSpeakingFrame(), _:
                self._let_go()
                modality = self._modality()
                self._read = _Read(modality, asyncio.ensure_future(self._reading(modality)))
            case TranscriptionFrame(), _Read(modality=modality, front=reading):
                front = reading.result() if reading.done() else _NOT_YET_READ
                self._let_go()
                # Behind the words and ahead of the hold's end, so it is in the context before the turn that writes them.
                await self.push_frame(Note([{"role": "user", "content": beside(front, modality)}], run_llm=False))
            case _:
                pass

    async def cleanup(self) -> None:
        await super().cleanup()
        self._let_go()

    def _let_go(self) -> None:
        """Ends the read kept, which is superseded, noted, or no longer wanted; one already done is left as it ended."""
        read, self._read = self._read, None
        match read:
            case _Read(front=reading):
                reading.cancel()
            case None:
                pass

    async def _reading(self, modality: Modality) -> InFront:
        # [LAW:nothing-unseen] one event a hold let go: what was read, or why it was not, and how long the read took;
        # cancelled where the words arrived first, or a later hold's read took its place.
        with unit("front.read", self._record):
            annotate(modality=modality)
            front = await self._front()
            annotate(front=front)
            return front
