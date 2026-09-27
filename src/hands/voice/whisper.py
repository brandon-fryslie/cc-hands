"""Whisper on MLX, cutting holds where the key cut them: it says where the user started and stopped speaking, numbering
each hold, transcribes a hold the key sent, throws away one the key dropped, and says when it is done with each."""

from collections import deque
from collections.abc import AsyncGenerator

from pipecat.frames.frames import Frame, InputAudioRawFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.whisper.stt import WhisperSTTServiceMLX

from hands.voice.ptt import Key, KeyedAudio
from hands.voice.turnstop import TurnOpened, TurnResolved

NOTHING_TRANSCRIBED = "on_nothing_transcribed"


class Whisper(WhisperSTTServiceMLX):
    """The pipeline's voice activity detector as well as its transcriber.

    [LAW:one-source-of-truth] a hold begins and ends where the frames say the key moved, in capture order, and nothing
    else says so: Whisper segments its audio there and pushes the VAD frames the turn strategies act on, so the two
    never disagree about where a hold is.
    """

    def __init__(self, *, settings: WhisperSTTServiceMLX.Settings) -> None:
        super().__init__(settings=settings)  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        # The key the last frame of microphone audio was captured under.
        self._captured: Key = "up"
        # The number of the last hold the key opened.
        self._opened = 0
        # The holds whose audio is queued for transcription, oldest first. Pipecat transcribes its queue one segment at
        # a time, in order, so each transcription is of the oldest.
        self._transcribing: deque[int] = deque()
        # Sync, so the handler runs as the turn's transcription ends rather than after the stop timeout gives up on it.
        self._register_event_handler(NOTHING_TRANSCRIBED, sync=True)

    async def process_audio_frame(self, frame: InputAudioRawFrame, direction: FrameDirection) -> None:
        # [LAW:parse-dont-validate] the microphone makes every frame this sees, and it tags each one.
        if not isinstance(frame, KeyedAudio):
            raise TypeError(f"{type(frame).__name__} carries no key; the keyed microphone makes every frame Whisper hears")
        match self._captured, frame.key:
            case "up" | "dropped", "down":
                self._opened += 1
                opened = TurnOpened(hold=self._opened)
                await self._handle_user_started_speaking(opened)
                await self.push_frame(opened)
            case "down", "up" if self.is_usable:
                # The key was let go: the hold's audio is queued, to be transcribed and sent.
                stopped = VADUserStoppedSpeakingFrame()
                self._transcribing.append(self._opened)
                await self._handle_user_stopped_speaking(stopped)
                await self.push_frame(stopped)
            case "down", "up" | "dropped":
                # Another key was pressed, so the hold was typing, not speech; or the key was let go of a Whisper that
                # can no longer transcribe, which Pipecat would give nothing to. What the hold recorded is thrown
                # away, so nothing is transcribed, sent, or reported as empty, and Whisper is done with it at once.
                self._user_speaking = False
                self._audio_buffer.clear()
                await self.push_frame(VADUserStoppedSpeakingFrame())
                await self.push_frame(TurnResolved(hold=self._opened))
            case _:
                pass
        self._captured = frame.key
        await super().process_audio_frame(frame, direction)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        hold = self._transcribing.popleft()
        produced: list[Frame] = []
        async for frame in super().run_stt(audio):
            produced.append(frame)
            yield frame
        if not produced:
            # A failed transcription yields an ErrorFrame, so only a hold Whisper heard nothing in reaches here.
            await self._call_event_handler(NOTHING_TRANSCRIBED)  # pyright: ignore[reportUnknownMemberType]  (its *args are untyped)
        # [LAW:dataflow-not-control-flow] heard, heard nothing, or failed, Whisper is done with the hold.
        yield TurnResolved(hold=hold)
