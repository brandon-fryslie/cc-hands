"""The start of a user turn: the moment it cuts off what hands is saying, which the edge that opened its hold decides.

Pipecat names both starts this chooses between: a VAD start (`VADUserTurnStartStrategy`), as the hold opens, and a
transcription start (`TranscriptionUserTurnStartStrategy`), once Whisper pushes words. Starting the turn is what
interrupts: the user aggregator broadcasts the interruption that stops the speaker, cancels the reply still streaming
from the model and its calls in flight, and settles a line waiting to be heard as cut off. A hold whose start waits on
words and that Whisper hears none in starts nothing, so none of that happens, and nothing has to be undone.
"""

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import BaseUserTurnStartStrategy

from hands.sessions.audit import FalseBargeIn, Record
from hands.voice.trigger import turn_start
from hands.voice.turnstop import TurnOpened, TurnResolved


class EdgeTurnStart(BaseUserTurnStartStrategy):
    """Starts the user turn as the edge that opened its hold says (`hands.voice.trigger.turn_start`).

    [LAW:single-enforcer] the one place a turn's start is decided: the pipeline runs no other start strategy.
    """

    def __init__(self, record: Record) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        self._record = record

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        match frame:
            case TurnOpened(hold=hold):
                match turn_start(hold.edge):
                    case "on the hold":
                        await self.trigger_user_turn_started()
                    case "on words":
                        pass
            case TranscriptionFrame():
                # Words, in whichever hold, are someone speaking. Pipecat starts an open turn no second time.
                await self.trigger_user_turn_started()
            case TurnResolved(hold=hold, transcribed=False):
                match turn_start(hold.edge):
                    case "on words":
                        # [LAW:nothing-unseen] a hold that cut nothing off is a line, as every barge-in is.
                        self._record(FalseBargeIn(hold.number))
                    case "on the hold":
                        pass
            case _:
                pass
        return ProcessFrameResult.CONTINUE
