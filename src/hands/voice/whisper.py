"""Whisper on MLX that reports a turn it transcribed to nothing, which Pipecat otherwise leaves as silence."""

from collections.abc import AsyncGenerator

from pipecat.frames.frames import Frame
from pipecat.services.whisper.stt import WhisperSTTServiceMLX

NOTHING_TRANSCRIBED = "on_nothing_transcribed"


class Whisper(WhisperSTTServiceMLX):
    def __init__(self, *, settings: WhisperSTTServiceMLX.Settings) -> None:
        super().__init__(settings=settings)  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        # Sync, so the handler runs as the turn's transcription ends rather than after the stop timeout gives up on it.
        self._register_event_handler(NOTHING_TRANSCRIBED, sync=True)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        produced = False
        async for frame in super().run_stt(audio):
            produced = True
            yield frame
        if not produced:
            # A failed transcription yields an ErrorFrame, so only a turn Whisper heard nothing in reaches here.
            await self._call_event_handler(NOTHING_TRANSCRIBED)  # pyright: ignore[reportUnknownMemberType]  (its *args are untyped)
