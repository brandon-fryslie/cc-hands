"""The floor: while the user's turn is open, what hands says unprompted waits, and follows the turn in the order it came.

The user's turn opens on the press, when the user aggregator says the user started speaking, and closes once every hold
it took in is transcribed and sent, when it says they stopped. Both are system frames it sends ahead of anything queued,
so the floor is taken the moment the key goes down and given back just ahead of the user's own words reaching the
model. A session that stops while the user talks is announced after they let go, never over them; nothing waiting is
dropped, since none of it had started to play.

This is the one queue unprompted speech waits in before the model's stage, under either telling: the brain's stage keeps
its own lanes behind it, and an API model takes what passes here in order with the user's turn.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from pipecat.frames.frames import Frame, LLMMessagesAppendFrame, TTSSpeakFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from hands.sessions.audit import Record, Yielded
from hands.voice.speech import Aloud, Narrated

# [LAW:types-are-the-program] what hands puts into the pipeline unasked, under both tellings: `as_written` and `handed`
# make the first three, and notes and the opening briefing are the fourth. Nothing the user's turn makes is one of these.
Unprompted = Aloud | Narrated | TTSSpeakFrame | LLMMessagesAppendFrame


@dataclass
class _Taken:
    """The user has the floor: since when, and what hands had to say meanwhile, in the order it came."""

    since: float
    held: list[Frame] = field(default_factory=list[Frame])


class Floor(FrameProcessor):
    """Holds what hands says unprompted while the user's turn is open, and passes it on, in order, as the turn closes."""

    def __init__(self, record: Record, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._record = record
        self._now = clock
        self._taken: _Taken | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame, self._taken:
            case UserStartedSpeakingFrame(), None:
                self._taken = _Taken(self._now())
                await self.push_frame(frame, direction)
            case UserStoppedSpeakingFrame(), _Taken(since=since, held=held):
                self._taken = None
                await self.push_frame(frame, direction)
                for each in held:
                    await self.push_frame(each)
                if held:
                    # [LAW:nothing-unseen] what waited for the user, and how long their turn held the floor.
                    self._record(Yielded(tuple(type(each).__name__ for each in held), self._now() - since))
            case _, _Taken(held=held) if isinstance(frame, Unprompted) and direction == FrameDirection.DOWNSTREAM:
                held.append(frame)
            case _:
                await self.push_frame(frame, direction)
