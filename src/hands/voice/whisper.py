"""Whisper as LowTalker serves it, cutting holds where the key cut them: it says where the user started and stopped
speaking, numbering each hold, transcribes a hold the key sent, throws away one the key dropped, and says when it is done
with each.

LowTalker (~/code/low-talker) keeps its Whisper resident on the Neural Engine for its own dictation and serves it at
OpenAI's POST /audio/transcriptions, so hands uploads each hold there rather than running a second Whisper of its own.
"""

import json
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import cast

import aiohttp
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

from hands.sessions.audit import HoldHeard, Record, Unsaid
from hands.core.place import Place
from hands.voice.ptt import Key, KeyedAudio
from hands.voice.turnstop import HoldDiscarded, TurnOpened, TurnResolved

# What every hold is said in.
LANGUAGE = Language.EN

# The model the upload names. OpenAI's contract requires one; LowTalker transcribes with the one it holds, whatever is named.
_MODEL = "whisper-1"

# Whisper's own compression_ratio_threshold: above it, the decoder was repeating itself, as it does over noise primed
# with a vocabulary, sure of every word ("and turnstop, and turnstop, ..." read 17.1 to 23.8; speech no more than 0.92,
# hands-dictation-2bs.1xq).
_REPEATING = 2.4

# The average log probability below which a segment is Whisper guessing. Primed noise came back from LowTalker as "and
# slow-talking." at -2.8, which the compression ratio (0.68) does not catch; "okay" said quietly at -25 dBFS read
# -0.89 to -1.11 (hands-dictation-d7i).
_GUESSED = -1.5

# How long a hold may wait on its answer before it is a failure: holds are transcribed one at a time, so a server that
# never answers would hold up every hold after it. A 4 s hold came back in 0.93 s (hands-dictation-2bs.tsm), and the
# longest hold, TURN_LIMIT_SECONDS, is four of Whisper's 30 s windows.
ANSWER_SECONDS = 30.0


class TranscriptionFailed(Exception):
    """The server answered a hold with something other than its transcription."""


class Whisper(SegmentedSTTService):
    """The pipeline's voice activity detector as well as its transcriber.

    [LAW:one-source-of-truth] a hold begins and ends where the frames say the key moved, in capture order, and nothing
    else says so: Whisper segments its audio there and pushes the VAD frames the turn strategies act on, so the two
    never disagree about where a hold is.
    """

    def __init__(self, *, url: str, prompt: Callable[[], Awaitable[str | None]], record: Record) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        # The transcription server's base, which /audio/transcriptions is appended to.
        self._url = url
        # The vocabulary each hold is transcribed with, read as it is: see hands.voice.vocabulary.
        self._prompt = prompt
        self._record = record
        # The key the last frame of microphone audio was captured under.
        self._captured: Key = "up"
        # Whose microphone the last frame came from, and how many turns the gate had sent and thrown away by it.
        self._heard_at: Place = "desk"
        self._sent = 0
        self._dropped = 0
        # The number of the last hold the key opened.
        self._opened = 0
        # The holds whose audio is queued for transcription, oldest first. Pipecat transcribes its queue one segment at
        # a time, in order, so each transcription is of the oldest.
        self._transcribing: deque[int] = deque()

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
            self._heard_at = frame.place
        # [LAW:no-ambient-temporal-coupling] a turn ends as the gate's counts move, not as its key is next seen at rest: a
        # turn can end and the next arm between two frames, with no frame captured at rest between them.
        match self._captured, frame.sent != self._sent, frame.dropped != self._dropped:
            case "down", True, _ if self.is_usable:
                # Sent: the hold's audio is queued, to be transcribed and sent.
                stopped = VADUserStoppedSpeakingFrame()
                self._transcribing.append(self._opened)
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

    @traced_stt
    async def _handle_transcription(self, transcript: str, is_final: bool, language: Language | None = None) -> None:
        """Pipecat's span for a transcription, which its tracing decorator opens around this."""

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        hold = self._transcribing.popleft()
        try:
            heard = await self._heard(hold, audio)
        except Exception as error:
            # [LAW:no-silent-failure] a failed transcription is heard: the pipeline says an ErrorFrame from Whisper aloud.
            yield ErrorFrame(error=f"Whisper could not transcribe hold {hold}: {type(error).__name__}: {error}", exception=error)
        else:
            # Recorded, so "I spoke and nothing happened" can be looked into.
            self._record(heard)
            match heard.said:
                case None:
                    # Not said: Brandon does not need to hear it (2026-09-27).
                    pass
                case said:
                    await self._handle_transcription(said, True, LANGUAGE)  # pyright: ignore[reportUnknownMemberType]  (Pipecat's tracing decorator is untyped)
                    yield TranscriptionFrame(said, self._user_id, time_now_iso8601(), LANGUAGE)
        # [LAW:dataflow-not-control-flow] heard, heard nothing, or failed, Whisper is done with the hold.
        yield TurnResolved(hold=hold)

    async def _heard(self, hold: int, audio: bytes) -> HoldHeard:
        """What was said in a hold's WAV, primed with the vocabulary as it is now.

        A segment with no word in it is dropped, as are one the decoder repeated itself in and one Whisper only guessed
        at: that is what a primed Whisper makes of noise. A hold with nothing said in it comes back with no segment.
        """
        prompt = await self._prompt()
        form = aiohttp.FormData()
        form.add_field("file", audio, filename=f"hold-{hold}.wav", content_type="audio/wav")
        form.add_field("model", _MODEL)
        form.add_field("language", LANGUAGE.value)
        form.add_field("response_format", "verbose_json")
        # [LAW:dataflow-not-control-flow] no prompt is no field, which is Whisper unprimed.
        for vocabulary in filter(None, (prompt,)):
            form.add_field("prompt", vocabulary)
        await self.start_processing_metrics()
        try:
            # [LAW:no-ambient-temporal-coupling] a session per hold, so nothing has to be opened before the first hold or
            # closed after the last; on loopback the connection it opens costs nothing a turn would notice.
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=ANSWER_SECONDS)) as session, session.post(f"{self._url}/audio/transcriptions", data=form) as response:
                body = await response.read()
                if response.status != 200:
                    raise TranscriptionFailed(f"{self._url} answered {response.status}: {_refusal(body)}")
        except TimeoutError as error:
            raise TranscriptionFailed(f"{self._url} did not answer within {ANSWER_SECONDS:.0f} s") from error
        finally:
            await self.stop_processing_metrics()
        said: list[str] = []
        dropped: list[Unsaid] = []
        for scored in _segments(body):
            worded = any(character.isalnum() for character in scored.text)
            if worded and scored.compression_ratio <= _REPEATING and scored.avg_logprob >= _GUESSED:
                said.append(scored.text)
            else:
                dropped.append(scored)
        return HoldHeard(hold, " ".join(said).strip() or None, tuple(dropped))


def _segments(body: bytes) -> list[Unsaid]:
    """[LAW:parse-dont-validate] the segments of a verbose_json answer, each with its text and both scores; raises
    TranscriptionFailed showing the answer where it is not one."""
    try:
        answer: object = json.loads(body)
    except ValueError as error:
        raise TranscriptionFailed(f"the answer is not JSON: {body[:200]!r}") from error
    match answer:
        case {"segments": list()}:
            return [_segment(segment) for segment in cast("dict[str, list[object]]", answer)["segments"]]
        case _:
            raise TranscriptionFailed(f"the answer is not verbose_json, having no segments: {body[:200]!r}")


def _segment(segment: object) -> Unsaid:
    match segment:
        case {"text": str() as text, "compression_ratio": int() | float() as ratio, "avg_logprob": int() | float() as logprob}:
            return Unsaid(text.strip(), float(ratio), float(logprob))
        case _:
            raise TranscriptionFailed(f"the answer is not verbose_json, a segment lacking its text or a score: {segment!r:.200}")


def _refusal(body: bytes) -> str:
    """What an error answer says: an OpenAI-shaped error's code and message, or the body itself where it is not one."""
    try:
        answer: object = json.loads(body)
    except ValueError:
        # Not JSON, as a proxy's or another server's error page is: the body is the refusal.
        return f"{body[:200]!r}"
    match answer:
        case {"error": {"message": str() as message, "code": str() as code}}:
            return f"{code}: {message}"
        case {"error": {"message": str() as message}}:
            return message
        case _:
            return f"{body[:200]!r}"
