"""Whisper, cutting holds where the key cut them: it says where the user started and stopped speaking, numbering each
hold, transcribes a hold the key sent (hands.voice.transcription), throws away one the key dropped, and says when it is
done with each."""

import asyncio
import io
import math
import threading
import wave
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable

import numpy as np
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt  # pyright: ignore[reportUnknownVariableType]  (untyped in Pipecat)

from hands.sessions.audit import HoldHeard, Levels, Record, Unsaid
from hands.sessions.wide import annotate, unit
from hands.core.place import Place
from hands.threads import SerialThread
from hands.voice import transcription
from hands.voice.ptt import Key, KeyedAudio
from hands.voice.turnstop import HoldDiscarded, TurnOpened, TurnResolved

# What every hold is said in, as Pipecat names it.
LANGUAGE = Language(transcription.LANGUAGE)

# Whisper's own compression_ratio_threshold: above it, the decoder was repeating itself, as it does over noise primed
# with a vocabulary, sure of every word ("and turnstop, and turnstop, ..." read 17.1 to 23.8; speech no more than 0.92,
# hands-dictation-2bs.1xq).
_REPEATING = 2.4

# The average log probability below which a segment is Whisper guessing. Primed noise came back as "and slow-talking."
# at -2.8, which the compression ratio (0.68) does not catch; "okay" said quietly at -25 dBFS read
# -0.89 to -1.11 (hands-dictation-d7i).
_GUESSED = -1.5

# The longest a hold's transcription may take before it fails. The model takes about a third of a second a hold, and the
# longest hold the key keeps open (TURN_LIMIT_SECONDS) is a few seconds' work, so a minute is a transcription that is
# not coming back.
TRANSCRIBING_SECONDS = 60.0


class Whisper(SegmentedSTTService):
    """The pipeline's voice activity detector as well as its transcriber.

    [LAW:one-source-of-truth] a hold begins and ends where the frames say the key moved, in capture order, and nothing
    else says so: Whisper segments its audio there and pushes the VAD frames the turn strategies act on, so the two
    never disagree about where a hold is.
    """

    def __init__(self, *, prompt: Callable[[], Awaitable[str | None]], record: Record) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        # [LAW:nothing-unseen] the load is a unit of work of its own: how long the start waited on it, and on what model.
        with unit("whisper.loaded", record):
            transcription.load()
            annotate(model=transcription.MODEL)
        # The vocabulary each hold is transcribed with, read as it is: see hands.voice.vocabulary.
        self._prompt = prompt
        self._record = record
        # The one thread the model transcribes on: a transcription given up on (TRANSCRIBING_SECONDS) is still running,
        # and the next must not run beside it.
        self._model = SerialThread("Whisper")
        # The key the last frame of microphone audio was captured under.
        self._captured: Key = "up"
        # Whose microphone the last frame came from, and how many turns the gate had sent and thrown away by it.
        self._heard_at: Place = "desk"
        self._sent = 0
        self._dropped = 0
        # The number of the last hold the key opened.
        self._opened = 0
        # The holds whose audio is queued for transcription, oldest first, each with how loud it was. Pipecat transcribes
        # its queue one segment at a time, in order, so each transcription is of the oldest.
        self._transcribing: deque[tuple[int, Levels]] = deque()
        # [LAW:nothing-unseen] the microphone's audio before the echo canceller, for the very frames Pipecat's
        # `_audio_buffer` holds, so a hold is measured on both sides of the canceller over the audio it is transcribed from.
        self._uncancelled = bytearray()

    # Only the key cuts holds: a VAD frame from anywhere else, which Pipecat's segmenting would act on, moves nothing.
    async def _handle_user_started_speaking(self, frame: VADUserStartedSpeakingFrame) -> None:
        pass

    async def _handle_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame) -> None:
        pass

    async def process_audio_frame(self, frame: InputAudioRawFrame, direction: FrameDirection) -> None:
        # [LAW:parse-dont-validate] the microphone makes every frame this sees, and it tags each one.
        if not isinstance(frame, KeyedAudio):
            raise TypeError(f"{type(frame).__name__} carries no key; the keyed microphone makes every frame Whisper hears")
        if frame.place != self._heard_at:
            # A turn is heard at one place: what the other's microphone heard before it is no part of it.
            self._audio_buffer.clear()
            self._uncancelled.clear()
            self._heard_at = frame.place
        # [LAW:no-ambient-temporal-coupling] a turn ends as the gate's counts move, not as its key is next seen at rest: a
        # turn can end and the next arm between two frames, with no frame captured at rest between them.
        match self._captured, frame.sent != self._sent, frame.dropped != self._dropped:
            case "down", True, _ if self.is_usable:
                # Sent: the hold's audio is queued, to be transcribed and sent.
                stopped = VADUserStoppedSpeakingFrame()
                # Measured as the hold is cut, before Pipecat pads it with silence for Whisper.
                self._transcribing.append((self._opened, Levels(_dbfs(self._uncancelled), _dbfs(self._audio_buffer))))
                await super()._handle_user_stopped_speaking(stopped)
                await self.push_frame(stopped)
                self._captured = "up"
            case ("down", True, _) | ("down", _, True):
                # Thrown away, as typing, too long open, or a call came or went; or sent to a Whisper that can no longer
                # transcribe, which Pipecat would give nothing to. Nothing is transcribed or sent, and Whisper is done
                # with the hold at once.
                self._user_speaking = False
                self._audio_buffer.clear()
                await self.push_frame(HoldDiscarded())
                await self.push_frame(TurnResolved(hold=self._opened))
                self._captured = "up"
            case _:
                pass
        self._sent, self._dropped = frame.sent, frame.dropped
        match self._captured, frame.key:
            case "up", "arming":
                # A hold's audio begins at its press: nothing heard before it is any part of it.
                self._audio_buffer.clear()
            case "arming", "up":
                # The press was Shift after all: what it heard is no part of any turn.
                self._user_speaking = False
                self._audio_buffer.clear()
            case "arming", "listening":
                # A start that was only a noise: the desk listens on, and its last second is kept as Pipecat keeps it.
                self._user_speaking = False
            case "up" | "listening" | "arming", "down":
                self._opened += 1
                opened = TurnOpened(hold=self._opened)
                await super()._handle_user_started_speaking(opened)
                await self.push_frame(opened)
            case "down", "up" | "listening" | "arming":
                # [LAW:no-silent-failure] the gate counts every end of a turn, so a key leaving down uncounted is a gate
                # that lies.
                raise RuntimeError(f"the key went from down to {frame.key} with no turn sent or thrown away")
            case _:
                # From listening to arming, the second the desk heard before it is kept: the turn's first words.
                pass
        self._captured = frame.key
        if frame.key == "arming":
            # [LAW:no-ambient-temporal-coupling] Pipecat keeps only the last second of audio nobody is speaking in, so a
            # press is heard as speech from the start: nothing it hears is trimmed while HOLD_SECONDS runs, however long.
            self._user_speaking = True
        await super().process_audio_frame(frame, direction)
        # [LAW:one-source-of-truth] Pipecat's buffer alone says which audio a hold is made of: it is cut only from its
        # front, here and wherever it is cleared above, and this is cut to its length, frame for frame.
        self._uncancelled += frame.captured
        del self._uncancelled[: len(self._uncancelled) - len(self._audio_buffer)]

    @traced_stt
    async def _handle_transcription(self, transcript: str, is_final: bool, language: Language | None = None) -> None:
        """Pipecat's span for a transcription, which its tracing decorator opens around this."""

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        hold, levels = self._transcribing.popleft()
        # [LAW:no-silent-failure] a transcription that never returns would hold its turn open for ever, since nothing but
        # Whisper resolving its holds ends one: it fails instead. A model that is running cannot be stopped, so the next
        # hold's transcription waits behind it, on the one thread, inside its own bound.
        bound = asyncio.timeout(TRANSCRIBING_SECONDS)
        try:
            async with bound:
                heard = await self._heard(hold, levels, audio)
            # Recorded, so "I spoke and nothing happened" can be looked into.
            self._record(heard)
            match heard.said:
                case None:
                    # Not said: Brandon does not need to hear it (2026-09-27).
                    pass
                case said:
                    await self._handle_transcription(said, True, LANGUAGE)  # pyright: ignore[reportUnknownMemberType]  (Pipecat's tracing decorator is untyped)
                    yield TranscriptionFrame(said, self._user_id, time_now_iso8601(), LANGUAGE)
        except Exception as error:
            # [LAW:no-silent-failure] a failed transcription is heard: the pipeline says an ErrorFrame from Whisper aloud.
            why = f"nothing after {TRANSCRIBING_SECONDS:g} s" if bound.expired() else f"{type(error).__name__}: {error}"
            yield ErrorFrame(error=f"Whisper could not transcribe hold {hold}: {why}", exception=error)
        # [LAW:dataflow-not-control-flow] heard, heard nothing, or failed, Whisper is done with the hold.
        yield TurnResolved(hold=hold)

    async def _heard(self, hold: int, levels: Levels, audio: bytes) -> HoldHeard:
        """What was said in a hold's WAV, primed with the vocabulary as it is now.

        A segment with no word in it is dropped, as are one the decoder repeated itself in and one Whisper only guessed
        at: that is what a primed Whisper makes of noise. A hold with nothing said in it comes back with no segment.
        """
        prompt = await self._prompt()
        samples = _samples(audio)
        # Set once nobody waits for this hold's transcription any more: the model is not run on it if it has yet to be.
        given_up = threading.Event()
        await self.start_processing_metrics()
        try:
            # Off the loop: the model runs for a third of a second a hold, and the speaker and the key go on meanwhile.
            scored = await self._model.run(lambda: [] if given_up.is_set() else transcription.segments(samples, prompt))
        finally:
            given_up.set()
            await self.stop_processing_metrics()
        said: list[str] = []
        dropped: list[Unsaid] = []
        for segment in scored:
            worded = any(character.isalnum() for character in segment.text)
            if worded and segment.compression_ratio <= _REPEATING and segment.avg_logprob >= _GUESSED:
                said.append(segment.text)
            else:
                dropped.append(segment)
        return HoldHeard(hold, " ".join(said).strip() or None, tuple(dropped), levels)


def _samples(wav: bytes) -> bytes:
    """The samples of a hold's WAV, as Pipecat wraps them at the pipeline's input rate; a WAV in any form but the one
    Whisper hears is refused, loudly, rather than heard sped up or slowed down [LAW:parse-dont-validate]."""
    with wave.open(io.BytesIO(wav)) as read:
        heard = (read.getnchannels(), read.getsampwidth(), read.getframerate())
        if heard != (1, 2, transcription.RATE):
            raise ValueError(f"a hold came as {heard[0]} channel(s) of {8 * heard[1]}-bit audio at {heard[2]} Hz; Whisper hears mono 16-bit at {transcription.RATE} Hz")
        return read.readframes(read.getnframes())


def _dbfs(audio: bytes | bytearray) -> float | None:
    """The mean power of 16-bit audio in dB below full scale, to a tenth; None where no sample of it sounds."""
    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float64) / 32768
    power = float(np.dot(samples, samples))
    # [LAW:types-are-the-program] digital silence has no level in dB: it is said as an absence, not as a floor.
    return round(10 * math.log10(power / samples.size), 1) if power else None
