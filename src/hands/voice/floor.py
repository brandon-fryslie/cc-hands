"""The floor: while the user's turn is open, what hands tells of the sessions waits, and follows the turn in the order it came.

The user's turn opens on the press, when the user aggregator says the user started speaking, and closes once every hold
it took in is transcribed and sent, when it says they stopped. It says both upstream as well as down, as system frames
ahead of anything queued, so the floor, which sits ahead of it, is taken the moment the key goes down. What waited is
given back into the aggregator's queue behind the frame that closed the turn, so the user's words reach the model first
and what hands had to say follows them. A session that stops while the user talks is announced after they let go, never
over them; nothing waiting is dropped as the turn closes, since none of it had started to play.

This is the one queue what hands tells of the sessions waits in before the model's stage, under either telling: the
brain's stage keeps its own lanes behind it, and an API model's notes join the context here, behind the user's turn. The
system voice is not in it: it reports the model's own failures, so it is queued past the model, at the TTS.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from pipecat.frames.frames import DataFrame, Frame, LLMMessagesAppendFrame, TTSSpeakFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from hands.sessions.audit import Record, Yielded
from hands.voice.speech import Aloud, Narrated

# [LAW:types-are-the-program] what hands puts into the pipeline unasked, under both tellings: `as_written` and `handed`
# make the first three, and notes and the opening briefing are the fourth. Nothing the user's turn makes is one of these.
Unprompted = Aloud | Narrated | TTSSpeakFrame | LLMMessagesAppendFrame


@dataclass
class _Given(DataFrame):
    """The turn closed: queued behind every frame that reached the floor before the close, so what waited is passed on
    in the order it came, and nothing arriving after it passes ahead of what is still held."""


@dataclass
class _Taken:
    """The user has the floor: since when, whether the turn is still open, and what hands had to say meanwhile, in order."""

    since: float
    open: bool = True
    held: list[Frame] = field(default_factory=list[Frame])


class Floor(FrameProcessor):
    """Holds what hands tells of the sessions while the user's turn is open, and passes it on, in order, as the turn closes."""

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
            case UserStartedSpeakingFrame(), _Taken() as taken:
                # A press before what the last turn held was given back: it waits out this turn as well.
                taken.open = True
                await self.push_frame(frame, direction)
            case UserStoppedSpeakingFrame(), _Taken() as taken:
                # [LAW:no-ambient-temporal-coupling] system frames are handled on their own task beside the data frames;
                # the release goes through the data queue, so the one task that holds is the one that gives back.
                taken.open = False
                await self.push_frame(frame, direction)
                await self.queue_frame(_Given())
            case _Given(), _Taken(open=False, since=since, held=held):
                self._taken = None
                waited = self._now() - since
                for each in held:
                    await self.push_frame(each)
                if held:
                    # [LAW:nothing-unseen] what waited for the user, and how long their turn held the floor.
                    self._record(Yielded(tuple(type(each).__name__ for each in held), waited))
            case _Given(), _:
                # The close of a turn a later press reopened: that turn's own close gives everything back.
                pass
            case _, _Taken(held=held) if isinstance(frame, Unprompted) and direction == FrameDirection.DOWNSTREAM:
                held.append(frame)
            case _:
                await self.push_frame(frame, direction)
