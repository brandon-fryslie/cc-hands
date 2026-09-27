"""Whisper on MLX that reports a turn it transcribed to nothing, which Pipecat otherwise leaves as silence, and never
transcribes a turn the talk key dropped."""

from collections.abc import AsyncGenerator

from pipecat.frames.frames import Frame, VADUserStoppedSpeakingFrame
from pipecat.services.whisper.stt import WhisperSTTServiceMLX

from hands.voice.ptt import PushToTalk

NOTHING_TRANSCRIBED = "on_nothing_transcribed"


class Whisper(WhisperSTTServiceMLX):
    def __init__(self, *, settings: WhisperSTTServiceMLX.Settings, key: PushToTalk) -> None:
        super().__init__(settings=settings)  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        self._key = key
        # Sync, so the handler runs as the turn's transcription ends rather than after the stop timeout gives up on it.
        self._register_event_handler(NOTHING_TRANSCRIBED, sync=True)

    async def _handle_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame) -> None:
        # A turn another key was pressed in is typing, not speech: what it recorded is thrown away here, where the
        # turn's audio becomes a segment, so nothing is transcribed, sent, or reported as empty. The key stays
        # dropped until the next turn opens, which the hold allows only after HOLD_SECONDS of a fresh press, far
        # longer than this frame takes to arrive from the aggregator.
        if self._key.gate.key == "dropped":
            self._user_speaking = False
            self._audio_buffer.clear()
            return
        await super()._handle_user_stopped_speaking(frame)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        produced = False
        async for frame in super().run_stt(audio):
            produced = True
            yield frame
        if not produced:
            # A failed transcription yields an ErrorFrame, so only a turn Whisper heard nothing in reaches here.
            await self._call_event_handler(NOTHING_TRANSCRIBED)  # pyright: ignore[reportUnknownMemberType]  (its *args are untyped)
