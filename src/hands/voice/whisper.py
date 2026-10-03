"""Whisper on MLX, cutting holds where the key cut them: it says where the user started and stopped speaking, numbering
each hold, transcribes a hold the key sent, throws away one the key dropped, and says when it is done with each."""

import asyncio
import time
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, cast

import mlx_whisper
import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.whisper.stt import WhisperSTTServiceMLX
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.types import assert_given, require_given

from hands.voice.ptt import Key, KeyedAudio
from hands.voice.turnstop import HoldDiscarded, TurnOpened, TurnResolved

# What the model is loaded by: a second of silence at the 16 kHz Whisper hears at.
_SILENCE = np.zeros(16_000, dtype=np.float32)

# The compression ratio Pipecat drops a segment for, as a hallucination.
_HALLUCINATED = 0.5555555555555556

# The average log probability below which a segment is Whisper guessing. Primed, the silence of a hold with nothing said
# comes back as "." (no word, dropped for that) or as "The End" and "and" at -2.1 and -2.7, which no other filter drops.
# Fourteen short commands said by `say`, clean and quiet under noise, primed and not, came back no lower than -0.97
# where heard right and -1.38 where misheard; Whisper's own logprob_threshold, -1.0, would drop "okay" said
# quietly (2026-10-03).
_GUESSED = -1.5


class Whisper(WhisperSTTServiceMLX):
    """The pipeline's voice activity detector as well as its transcriber.

    [LAW:one-source-of-truth] a hold begins and ends where the frames say the key moved, in capture order, and nothing
    else says so: Whisper segments its audio there and pushes the VAD frames the turn strategies act on, so the two
    never disagree about where a hold is.
    """

    def __init__(self, *, settings: WhisperSTTServiceMLX.Settings, prompt: Callable[[], Awaitable[str | None]]) -> None:
        super().__init__(settings=settings)  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        # The initial prompt each hold is transcribed with, read as it is: see hands.voice.vocabulary.
        self._prompt = prompt
        self._warm()
        # The key the last frame of microphone audio was captured under.
        self._captured: Key = "up"
        # The number of the last hold the key opened.
        self._opened = 0
        # The holds whose audio is queued for transcription, oldest first. Pipecat transcribes its queue one segment at
        # a time, in order, so each transcription is of the oldest.
        self._transcribing: deque[int] = deque()

    def _warm(self) -> None:
        """Fetch the model, load it, and run it once now, while hands is starting, rather than in the first turn.

        MLX Whisper loads its model inside the first transcription it is asked for, downloading it first if it was
        never fetched: the first turn of the first run waited 28 s on a 1.6 GB download (2026-09-26). Once fetched,
        loading takes 0.5 s and the first run after it 1.3 s against 0.3 s for every later one, so loading alone
        would leave the first turn a second slower than the rest. The model is kept for the process, keyed on the
        name it was asked for, so silence transcribed here through the call every hold makes leaves every turn finding it
        loaded and run [LAW:no-ambient-temporal-coupling].
        """
        began = time.monotonic()
        self._transcribe(_SILENCE, None)
        logger.info(f"Whisper loaded {self._settings.model} in {time.monotonic() - began:.1f} s")

    def _transcribe(self, audio: np.ndarray, prompt: str | None) -> list[dict[str, Any]]:
        """The segments MLX Whisper heard in `audio`, primed with `prompt`; what both the load and every hold run."""
        transcribed: dict[str, Any] = mlx_whisper.transcribe(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  (untyped in mlx_whisper)
            audio,
            path_or_hf_repo=require_given(self._settings.model, "Whisper model"),
            temperature=assert_given(self._settings.temperature),
            language=assert_given(self._settings.language),
            initial_prompt=prompt,
        )
        return cast("list[dict[str, Any]]", transcribed.get("segments", []))

    # Only the key cuts holds: a VAD frame from anywhere else, which Pipecat's segmenting would act on, moves nothing.
    async def _handle_user_started_speaking(self, frame: VADUserStartedSpeakingFrame) -> None:
        pass

    async def _handle_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame) -> None:
        pass

    async def process_audio_frame(self, frame: InputAudioRawFrame, direction: FrameDirection) -> None:
        # [LAW:parse-dont-validate] the microphone makes every frame this sees, and it tags each one.
        if not isinstance(frame, KeyedAudio):
            raise TypeError(f"{type(frame).__name__} carries no key; the keyed microphone makes every frame Whisper hears")
        match self._captured, frame.key:
            case "up" | "dropped", "arming":
                # A hold's audio begins at its press: nothing heard before it is any part of it.
                self._audio_buffer.clear()
            case "arming", "up" | "dropped":
                # The press was Shift after all: what it heard is no part of any turn.
                self._user_speaking = False
                self._audio_buffer.clear()
            case "up" | "arming" | "dropped", "down":
                self._opened += 1
                opened = TurnOpened(hold=self._opened)
                await super()._handle_user_started_speaking(opened)
                await self.push_frame(opened)
            case "down", "up" if self.is_usable:
                # The key was let go: the hold's audio is queued, to be transcribed and sent.
                stopped = VADUserStoppedSpeakingFrame()
                self._transcribing.append(self._opened)
                await super()._handle_user_stopped_speaking(stopped)
                await self.push_frame(stopped)
            case "down", "up" | "arming" | "dropped":
                # Another key was pressed, so the hold was typing, not speech (and the key may already be pressed
                # again); or the key was let go of a Whisper that can no longer transcribe, which Pipecat would give
                # nothing to. What the hold recorded is thrown away, so nothing is transcribed or sent, and Whisper is
                # done with it at once.
                self._user_speaking = False
                self._audio_buffer.clear()
                await self.push_frame(HoldDiscarded())
                await self.push_frame(TurnResolved(hold=self._opened))
            case _:
                pass
        self._captured = frame.key
        if frame.key == "arming":
            # [LAW:no-ambient-temporal-coupling] Pipecat keeps only the last second of audio nobody is speaking in, so a
            # press is heard as speech from the start: nothing it hears is trimmed while HOLD_SECONDS runs, however long.
            self._user_speaking = True
        await super().process_audio_frame(frame, direction)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        hold = self._transcribing.popleft()
        try:
            text = await self._heard(audio)
        except Exception as error:
            # [LAW:no-silent-failure] a failed transcription is heard: the pipeline says an ErrorFrame from Whisper aloud.
            yield ErrorFrame(error=f"Whisper could not transcribe hold {hold}: {type(error).__name__}: {error}", exception=error)
        else:
            match text:
                case None:
                    # Not said: Brandon does not need to hear it (2026-09-27). Logged, so "I spoke and nothing happened"
                    # can still be looked into.
                    logger.info(f"Whisper heard nothing in hold {hold}")
                case said:
                    language = cast("Language | None", assert_given(self._settings.language))
                    # Pipecat's span for a transcription, which its own run_stt opens.
                    await self._handle_transcription(said, True, language)  # pyright: ignore[reportUnknownMemberType]  (Pipecat's tracing decorator is untyped)
                    yield TranscriptionFrame(said, self._user_id, time_now_iso8601(), language)
        # [LAW:dataflow-not-control-flow] heard, heard nothing, or failed, Whisper is done with the hold.
        yield TurnResolved(hold=hold)

    async def _heard(self, audio: bytes) -> str | None:
        """What was said in a hold's 16-bit samples, primed with the vocabulary as it is now; None where nothing was.

        Pipecat's own run_stt takes no prompt, so this is its transcription with one, its filters kept: a segment
        that is likely no speech is dropped, and so is one with the compression ratio Pipecat found Whisper's
        hallucinations to have. A segment with no word in it, or one Whisper only guessed at, is dropped too: that is
        what a primed Whisper makes of silence.
        """
        prompt = await self._prompt()
        await self.start_processing_metrics()
        segments = await asyncio.to_thread(self._transcribe, np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0, prompt)
        await self.stop_processing_metrics()
        threshold = assert_given(self._settings.no_speech_prob)
        heard: list[str] = []
        for segment in segments:
            text: str = segment["text"].strip()
            worded = any(character.isalnum() for character in text)
            if worded and segment["no_speech_prob"] < threshold and segment["compression_ratio"] != _HALLUCINATED and segment["avg_logprob"] >= _GUESSED:
                heard.append(text)
            else:
                logger.info(
                    f"Whisper dropped {text!r} as not said: no_speech_prob {segment['no_speech_prob']:.2f},"
                    f" compression_ratio {segment['compression_ratio']:.2f}, avg_logprob {segment['avg_logprob']:.2f}"
                )
        return " ".join(heard).strip() or None
