"""Whisper on MLX, cutting turns where the key cut them: it transcribes a turn the key sent, throws away one the key
dropped, and says of every turn that ends with no text that it has none."""

from collections.abc import AsyncGenerator

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.whisper.stt import WhisperSTTServiceMLX

from hands.voice.ptt import Key, KeyedAudio
from hands.voice.turnstop import TurnUnheard

NOTHING_TRANSCRIBED = "on_nothing_transcribed"


class Whisper(WhisperSTTServiceMLX):
    def __init__(self, *, settings: WhisperSTTServiceMLX.Settings) -> None:
        super().__init__(settings=settings)  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        # The key the last frame of microphone audio was captured under.
        self._captured: Key = "up"
        # Sync, so the handler runs as the turn's transcription ends rather than after the stop timeout gives up on it.
        self._register_event_handler(NOTHING_TRANSCRIBED, sync=True)

    # [LAW:one-source-of-truth] a turn's audio begins and ends where the frames say the key moved, not where the VAD
    # says, which reads the key only when a frame reaches the aggregator, downstream of here.
    async def _handle_user_started_speaking(self, frame: VADUserStartedSpeakingFrame) -> None:
        pass

    async def _handle_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame) -> None:
        pass

    async def process_audio_frame(self, frame: InputAudioRawFrame, direction: FrameDirection) -> None:
        # [LAW:parse-dont-validate] the microphone makes every frame this sees, and it tags each one.
        if not isinstance(frame, KeyedAudio):
            raise TypeError(f"{type(frame).__name__} carries no key; the keyed microphone makes every frame Whisper hears")
        match self._captured, frame.key:
            case "down", "up":
                # The key was let go: the turn's audio is transcribed and sent.
                await super()._handle_user_stopped_speaking(VADUserStoppedSpeakingFrame())
            case "down", "dropped":
                # Another key was pressed: the turn was typing, not speech. What it recorded is thrown away, so nothing
                # is transcribed, sent, or reported as empty; the turn is said to have no text, so it ends at once.
                self._user_speaking = False
                self._audio_buffer.clear()
                await self.push_frame(TurnUnheard())
            case "up" | "dropped", "down":
                await super()._handle_user_started_speaking(VADUserStartedSpeakingFrame())
            case _:
                pass
        self._captured = frame.key
        await super().process_audio_frame(frame, direction)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        produced: list[Frame] = []
        async for frame in super().run_stt(audio):
            produced.append(frame)
            yield frame
        if not produced:
            # A failed transcription yields an ErrorFrame, so only a turn Whisper heard nothing in reaches here.
            await self._call_event_handler(NOTHING_TRANSCRIBED)  # pyright: ignore[reportUnknownMemberType]  (its *args are untyped)
        # Heard nothing or failed, the turn has no text either way.
        if not any(isinstance(frame, TranscriptionFrame) for frame in produced):
            yield TurnUnheard()
