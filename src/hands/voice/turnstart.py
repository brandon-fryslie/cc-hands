"""The start of a user turn, and the cut: the moment it cuts off what hands is saying, which the edge that opened it decides.

Every turn starts as the hold that opens it opens, so its words gather from the first hold and the floor is the user's
from the moment they speak (`hands.voice.floor`). Starting is not cutting. The cut is the user taking over: Pipecat's
start of a turn, which the user aggregator broadcasts as `UserStartedSpeakingFrame` and then the interruption that stops
the speaker, cancels the reply still streaming from the model and its calls in flight, and settles a line waiting to be
heard as cut off. A turn the user's hand opened cuts as it opens; one the voice opened cuts once Whisper pushes words for
it, Pipecat's transcription start. A turn that ends uncut has taken nothing over: nothing is cut off, so nothing has to
be undone, and the reply it found on its way goes on as if no turn had opened, a tool's result included. The cut can
land while the turn's own words and their note are on their way, so those are uninterruptible
(`hands.voice.turnstop.Words`, `hands.voice.beside.Note`): an interruption stops hands, never the user.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

from pipecat.frames.frames import Frame, UserStartedSpeakingFrame
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import BaseUserTurnStartStrategy

from hands.sessions.audit import Record, TurnStart, UserTurn
from hands.voice.trigger import turn_start
from hands.voice.turnstop import TurnOpened, Words


@dataclass
class _Open:
    """The turn open: when the edge that opened it has it cut, when it opened, and how long after that it cut, None until
    it has."""

    start: TurnStart
    opened: float
    cut: float | None = None


class EdgeTurnStart(BaseUserTurnStartStrategy):
    """Starts the user turn as its first hold opens, and cuts hands off when the edge that opened each hold says
    (`hands.voice.trigger.turn_start`).

    [LAW:single-enforcer] the one place a turn's start and its cut are decided: the pipeline runs no other start strategy,
    and Pipecat says nothing of its own as the turn starts, neither that the user started speaking nor an interruption.
    """

    def __init__(self, record: Record, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(enable_interruptions=False)  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        self._record = record
        self._now = clock
        # Between turns, None: Pipecat ends a turn, and this is told so (`handle_user_turn_stopped`).
        self._turn: _Open | None = None
        # Each cut, for the user aggregator to broadcast (`interrupting`).
        self._register_event_handler("on_cut", sync=True)

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        match frame, self._turn:
            case TurnOpened(hold=hold), None:
                start = turn_start(hold.edge)
                self._turn = _Open(start, self._now())
                await self.trigger_user_turn_started(enable_user_speaking_frames=False)
                await self._cut_where(start, self._turn)
            case TurnOpened(hold=hold), _Open() as turn:
                # A hold pressed while Whisper is still on the last joins its turn, and cuts as its own edge says.
                await self._cut_where(turn_start(hold.edge), turn)
            case Words(), _Open() as turn:
                # Words, in whichever hold, are someone speaking.
                await self._cut(turn)
            case Words(), None:
                # [LAW:no-silent-failure] a turn ends only once Whisper is done with every hold, behind each one's words.
                raise RuntimeError("Whisper heard words outside any user turn")
            case _:
                pass
        return ProcessFrameResult.CONTINUE

    async def handle_user_turn_stopped(self) -> None:
        match self._turn:
            case _Open(start=start, cut=cut):
                # [LAW:nothing-unseen] every turn is a line: how its edge had it cut, and when, or that it cut nothing.
                self._record(UserTurn(start, cut))
                self._turn = None
            case None:
                # [LAW:no-silent-failure] Pipecat stops only a turn it started, and nothing but this starts one.
                raise RuntimeError("Pipecat stopped a user turn this never started")

    async def _cut_where(self, start: TurnStart, turn: _Open) -> None:
        match start:
            case "on the hold":
                await self._cut(turn)
            case "on words":
                pass

    async def _cut(self, turn: _Open) -> None:
        """Cut hands off, once a turn."""
        match turn.cut:
            case None:
                turn.cut = self._now() - turn.opened
                await self._call_event_handler("on_cut")  # pyright: ignore[reportUnknownMemberType]  (Pipecat's *args is untyped)
            case _:
                pass


def interrupting(start: EdgeTurnStart, turns: LLMUserAggregator) -> None:
    """Have `turns`, the user aggregator `start` is the start strategy of, broadcast each cut `start` makes.

    The aggregator broadcasts it as it broadcasts the start of a turn Pipecat starts, in the same order: that the user
    started speaking, which the assistant side reads as the user having taken over, so it runs the model on no tool's
    result until the turn is sent; then the interruption. Both go inline, from within the frame being processed, so they
    are pushed ahead of anything behind that frame. Queued as frames of their own, the turn's context could leave for the
    model ahead of them, and the interruption would then cancel the reply to the very words that made it.
    """

    @start.event_handler("on_cut")
    async def cut(_start: EdgeTurnStart) -> None:  # pyright: ignore[reportUnusedFunction]
        await turns.broadcast_frame(UserStartedSpeakingFrame)  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        await turns.broadcast_interruption()
