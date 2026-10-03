"""The floor: while the user's turn is open, what hands tells of the sessions waits, and follows the turn soonest first.

The user's turn opens on the press, when the user aggregator says the user started speaking, and closes once every hold
it took in is transcribed and sent, when it says they stopped. It says both upstream as well as down, as system frames
ahead of anything queued, so the floor, which sits ahead of it, is taken the moment the key goes down. What waited is
given back into the aggregator's queue behind the frame that closed the turn, so the user's words reach the model first
and what hands had to say follows them. A session that stops while the user talks is announced after they let go, never
over them. As the turn closes, `coalesce` orders what waited: what a session waits on the user for before what a
session did, each session's finished turns folded into one telling, and what no longer waits on the user, a request
answered at the keyboard meanwhile, dropped. Nothing else is dropped, since none of it had started to play.

This is the one queue what hands tells of the sessions waits in before the model's stage, under either telling: the
brain's stage keeps its own lanes behind it, and an API model's notes join the context here, behind the user's turn. The
system voice is not in it: it reports the model's own failures, so it is queued past the model, at the TTS.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from pipecat.frames.frames import DataFrame, Frame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from hands.core.pending import Pending, coalesce
from hands.core.session import Held, SessionId
from hands.sessions.audit import Record, Yielded
from hands.voice.speech import Names, Telling, Unprompted, frames


@dataclass
class _Given(DataFrame):
    """The turn closed: queued behind every frame that reached the floor before the close, so what waited is passed on
    after the user's words, and nothing arriving after it passes ahead of what is still held."""


@dataclass
class _Taken:
    """The user has the floor: since when, whether the turn is still open, and what hands had to say meanwhile, in order."""

    since: float
    open: bool = True
    held: list[Pending] = field(default_factory=list[Pending])


class Floor(FrameProcessor):
    """Holds what hands tells of the sessions while the user's turn is open, and lets it go, soonest first, as the turn
    closes; what arrives with no turn open is let go at once, the same way."""

    def __init__(
        self,
        record: Record,
        telling: Telling,
        names: Names,
        held: Callable[[], Mapping[SessionId, Held]],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._record = record
        self._telling = telling
        self._names = names
        self._held = held
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
                await self._let_go(held, self._now() - since)
            case _Given(), _:
                # The close of a turn a later press reopened: that turn's own close gives everything back.
                pass
            case Unprompted(pending=pending), _Taken(held=held):
                held.append(pending)
            case Unprompted(pending=pending), None:
                await self._let_go([pending], 0.0)
            case _:
                await self.push_frame(frame, direction)

    async def _let_go(self, pending: list[Pending], waited: float) -> None:
        """Tell what was pending, as `coalesce` orders and folds it, with what waits on the user read as it is let go."""
        told = coalesce(pending, self._held())
        for each in told:
            for spoken in frames(each, self._telling, self._names):
                await self.push_frame(spoken)
        # [LAW:nothing-unseen] what came, what was told of it, and how long the user's turn held it.
        self._record(Yielded(_kinds(pending), _kinds(told), waited))


def _kinds(pending: Sequence[Pending]) -> tuple[str, ...]:
    return tuple(type(each).__name__ for each in pending)
