"""The end of a user turn that has no text: one the talk key dropped, or one Whisper heard nothing in.

Pipecat's speech-timeout strategy ends a turn only once it holds a transcript, so a turn that will never have one stays
open until the 5 s stop timeout; meanwhile a new press opens no turn of its own and interrupts nothing it should. Whisper
says, with `TurnUnheard`, that the turn it just closed has no text, and `KeyTurnStop` ends the turn on it.
"""

from dataclasses import dataclass

from pipecat.frames.frames import DataFrame, Frame
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy


@dataclass
class TurnUnheard(DataFrame):
    """The user turn Whisper just closed has no text, and never will."""


class KeyTurnStop(SpeechTimeoutUserTurnStopStrategy):
    """The speech-timeout strategy, which also ends a turn Whisper said has no text, once the VAD has stopped too.

    Whisper cuts turns by the key each frame was captured under, and the VAD by the key when the aggregator reads a
    frame, so either may come first; Pipecat will not end a turn while the VAD says the user is speaking.
    """

    def __init__(self, *, user_speech_timeout: float) -> None:
        super().__init__(user_speech_timeout=user_speech_timeout)  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        self._unheard = False

    async def handle_user_turn_started(self) -> None:
        await super().handle_user_turn_started()
        self._unheard = False

    async def handle_user_turn_stopped(self) -> None:
        await super().handle_user_turn_stopped()
        self._unheard = False

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        result = await super().process_frame(frame)
        match frame:
            case TurnUnheard():
                self._unheard = True
            case _:
                pass
        # The aggregator ends the turn with nothing aggregated, so nothing is sent to the model.
        if self._unheard and not self._vad_user_speaking:
            self._unheard = False
            await self.trigger_user_turn_stopped()
        return result
