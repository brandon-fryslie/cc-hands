"""Whisper as LowTalker serves it, cutting holds where the key cut them: it says where the user started and stopped
speaking, numbering each hold, transcribes a hold the key sent by uploading it to LowTalker (hands.voice.transcription),
throws away one the key dropped, and says when it is done with each. Words the user typed are a hold of their own, numbered with the key's and heard as they came.

A hold whose cut waits on words (`hands.voice.trigger.turn_start`) is also heard while it is still open, every
`OVERHEARING_SECONDS` of its audio, until Whisper hears words in it: those are pushed as `InterimWords`, so hands is cut
off while the user speaks over it rather than once they stop, which in engaged conversation is only after end-of-turn
detection closes the hold. The same filters judge what counts as words, so hands' echo or a cough still cuts nothing.
Each hearing hears the audio since the one before it was queued, two hops of it: a word straddling one hearing's end is
whole in the next, and a hearing costs what LowTalker takes to hear a hop, however long the hold has been open."""

import asyncio
import io
import math
import wave
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pipecat.audio.utils import pcm_to_wav
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt  # pyright: ignore[reportUnknownVariableType]  (untyped in Pipecat)

from hands.sessions.audit import HoldHeard, Levels, Record, Speaker, Unsaid, Untellable, Voiced
from hands.sessions.wide import annotate, fail, unit
from hands.core.place import Place
from hands.threads import SerialThread
from hands.voice import transcription
from hands.voice.ptt import Key, KeyedAudio
from hands.voice.speakers import seconds, spoken_as
from hands.voice.trigger import turn_start
from hands.voice.turnstop import Hold, HoldDiscarded, InterimWords, TurnOpened, TurnResolved, Typed, Words

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

# The longest a hold's transcription, its speaker told with it, may take before it fails: nothing but Whisper resolving a
# hold ends its turn, so one that never comes back would hold the turn open for ever.
TRANSCRIBING_SECONDS = 60.0

# How long one upload may wait on LowTalker's answer before it is a fault of its own (transcription.Unanswered): holds are
# transcribed one at a time, so a server that never answers would hold up every hold after it. A 4 s hold came back in
# 0.93 s (hands-dictation-2bs.tsm), and the longest hold, TURN_LIMIT_SECONDS, is four of Whisper's 30 s windows.
ANSWER_SECONDS = 30.0

# How much more audio a hold still open gathers before Whisper hears it again, where its cut waits on words: a word or two
# said, and about a third of a second for the model to hear it, so hands is cut off within a second of the user speaking
# over it. The audit log had engaged turns cut 0.5 to 12.5 s after they opened, each just after the hold closed, all
# that time talked over (hands-voice-o6o).
OVERHEARING_SECONDS = 0.5

# What a typed or thrown-away hold puts in Pipecat's queue of segments, which skips an empty one: it has no audio to hear,
# so all it needs there is its place in line.
_IN_LINE = b"in line"


@dataclass(frozen=True)
class _Recorded:
    """A hold of the key's, its audio in Pipecat's queue, and how loud it was."""

    hold: Hold
    levels: Levels


@dataclass(frozen=True)
class _Written:
    """A typed hold, and its words."""

    hold: Hold
    text: str


@dataclass(frozen=True)
class _Discarded:
    """A hold the key threw away, resolved in line behind every hearing of it still waiting."""

    hold: Hold


# Where hearing a hold while it is open stands: between hearings; one queued and not yet heard; or over, since words were
# heard, the hold closed, a hearing failed, or its cut never waited on words.
Overhearing = Literal["listening", "queued", "over"]


@dataclass(frozen=True)
class _Unread:
    """The vocabulary, not yet read for the hearings of a hold still open."""


@dataclass
class _Open:
    """The hold the key last opened, as Whisper hears it while it is open: the bytes of its audio in the hop before its
    last hearing was queued and gathered since, where hearing it stands, and the vocabulary every hearing of it is primed
    with, read for the first."""

    hold: Hold
    hop: int
    gathered: int
    overhearing: Overhearing
    prompt: str | None | _Unread = _Unread()


@dataclass(frozen=True)
class _Partial:
    """A hearing of a hold still open: the hold, the seconds of its audio heard, and how loud they were."""

    opened: _Open
    seconds: float
    levels: Levels


def _overhearing(hold: Hold) -> Overhearing:
    """Where hearing `hold` while it is open starts: listening where its cut waits on words, over where it cut as it opened."""
    match turn_start(hold.opener):
        case "on words":
            return "listening"
        case "on the hold":
            return "over"


class Whisper(SegmentedSTTService):
    """The pipeline's voice activity detector as well as its transcriber.

    [LAW:one-source-of-truth] a hold begins and ends where the frames say the key moved, in capture order, and nothing
    else says so: Whisper segments its audio there and pushes the VAD frames the turn strategies act on, so the two
    never disagree about where a hold is.
    """

    def __init__(self, *, url: str, prompt: Callable[[], Awaitable[str | None]], told: Callable[[bytes, Hold], Speaker], record: Record) -> None:
        # Pipecat checks at start that the settings say every field; these are what each hold is uploaded with.
        super().__init__(settings=STTSettings(model=transcription.MODEL, language=LANGUAGE))  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        # LowTalker's transcription server, the base /audio/transcriptions is appended to.
        self._url = url
        # The vocabulary each hold is transcribed with, read as it is: see hands.voice.vocabulary.
        self._prompt = prompt
        # Whose voice a hold's samples are in (hands.voice.speakers), told on the speaker model's thread.
        self._told = told
        self._record = record
        # The one thread the speaker model runs on: a telling given up on (TRANSCRIBING_SECONDS) is still running, and the
        # next must not run beside it.
        self._speakers = SerialThread("Speakers")
        # The key the last frame of microphone audio was captured under.
        self._captured: Key = "up"
        # Whose microphone the last frame came from, and how many turns the gate had sent and thrown away by it.
        self._heard_at: Place = "desk"
        self._sent = 0
        self._dropped = 0
        # How many holds have opened, the key's and those typed; and the last the key opened, hold 0 until one has.
        self._holds = 0
        self._opened = _Open(Hold(0, "held key", 0), 0, 0, "over")
        # The holds queued to be heard, oldest first, the key's, those typed and those thrown away, and hearings of the hold
        # still open.
        # Pipecat takes its queue one segment at a time, in order, so each it takes is the oldest, and a hold typed is
        # heard after every hold queued ahead of it.
        self._transcribing: deque[_Recorded | _Written | _Discarded | _Partial] = deque()
        # [LAW:nothing-unseen] the microphone's audio before the echo canceller, for the very frames Pipecat's
        # `_audio_buffer` holds, so a hold is measured on both sides of the canceller over the audio it is transcribed from.
        self._uncancelled = bytearray()

    # Only the key cuts holds: a VAD frame from anywhere else, which Pipecat's segmenting would act on, moves nothing.
    async def _handle_user_started_speaking(self, frame: VADUserStartedSpeakingFrame) -> None:
        pass

    async def _handle_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame) -> None:
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        match frame:
            case Typed(text=text):
                # Taken here, not passed on: the hold it is goes on in its place.
                await self._typed(text)
            case _:
                await super().process_frame(frame, direction)

    async def _typed(self, text: str) -> None:
        """A hold that opens and ends at once, and is heard in line behind the holds queued ahead of it, as one the key
        sent would be: it joins a turn a hold of the key's has open, and leaves the key's hold and its audio as they were."""
        self._holds += 1
        hold = Hold(self._holds, "typed", self._opened.hold.conversation)
        await self.push_frame(TurnOpened(hold=hold))
        await self.push_frame(VADUserStoppedSpeakingFrame())
        self._transcribing.append(_Written(hold, text))
        await self._segment_queue.put(_IN_LINE)

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
                self._transcribing.append(_Recorded(self._opened.hold, Levels(_dbfs(self._uncancelled), _dbfs(self._audio_buffer))))
                # Heard whole next: a hearing of it still waiting is moot.
                self._opened.overhearing = "over"
                await super()._handle_user_stopped_speaking(stopped)
                await self.push_frame(stopped)
                self._captured = "up"
            case ("down", True, _) | ("down", _, True):
                # Thrown away, as typing, too long open, or a call came or went; or sent to a Whisper that can no longer
                # transcribe, which Pipecat would give nothing to. Nothing is transcribed or sent, and Whisper is done
                # with the hold as soon as it is done with every hearing of it while it was open.
                self._user_speaking = False
                self._audio_buffer.clear()
                # What a hearing of it still waiting hears is no one's.
                self._opened.overhearing = "over"
                await self.push_frame(HoldDiscarded())
                # [LAW:no-ambient-temporal-coupling] resolved by the one task that pushes what Whisper heard, in line, so
                # nothing heard in the hold can land behind its resolution.
                self._transcribing.append(_Discarded(self._opened.hold))
                await self._segment_queue.put(_IN_LINE)
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
                self._holds += 1
                hold = Hold(self._holds, frame.opened, frame.conversation)
                # The second the desk heard before it is its first hop.
                self._opened = _Open(hold, len(self._audio_buffer), 0, _overhearing(hold))
                opened = TurnOpened(hold=hold)
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
        self._opened.gathered += len(frame.audio)
        await self._overhear()

    async def _overhear(self) -> None:
        """Queue the hold open to be heard, its last two hops, where its cut waits on words, none was heard in it yet, no
        hearing of it is waiting, and it has gathered `OVERHEARING_SECONDS` of audio since it opened or was last queued."""
        opened = self._opened
        match opened.overhearing:
            case "listening" if self.is_usable and opened.gathered >= OVERHEARING_SECONDS * self._audio_buffer_size_1s:
                heard = opened.hop + opened.gathered
                opened.hop, opened.gathered, opened.overhearing = opened.gathered, 0, "queued"
                audio = bytes(self._audio_buffer[-heard:])
                levels = Levels(_dbfs(self._uncancelled[-heard:]), _dbfs(audio))
                self._transcribing.append(_Partial(opened, len(audio) / self._audio_buffer_size_1s, levels))
                await self._segment_queue.put(pcm_to_wav(audio + self._trailing_silence(), self.sample_rate))
            case _:
                pass

    async def fault(self) -> transcription.Fault | None:
        """The fault a hold sent now would meet, or None where LowTalker transcribes."""
        # [LAW:nothing-unseen] the fault whole, its reason and the server's refusal with it, where what is said of it
        # keeps only what to do.
        with unit("transcription.probed", self._record):
            fault = await transcription.probe(self._url)
            annotate(url=self._url, fault=fault)
        return fault

    @traced_stt
    async def _handle_transcription(self, transcript: str, is_final: bool, language: Language | None = None) -> None:
        """Pipecat's span for a transcription, which its tracing decorator opens around this."""

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        queued = self._transcribing.popleft()
        match queued:
            case _Partial() as partial:
                # A hold still open: Whisper is not done with it.
                for frame in await self._overheard(partial, audio):
                    yield frame
            case _Written(text=text):
                yield Words(text, self._user_id, time_now_iso8601(), LANGUAGE)
                yield TurnResolved(hold=queued.hold)
            case _Discarded(hold=hold):
                yield TurnResolved(hold=hold)
            case _Recorded(hold=hold, levels=levels):
                async for frame in self._transcribed(hold, levels, audio):
                    yield frame
                # [LAW:dataflow-not-control-flow] heard, heard nothing, or failed, Whisper is done with the hold.
                yield TurnResolved(hold=hold)

    async def _overheard(self, partial: _Partial, audio: bytes) -> tuple[InterimWords, ...]:
        """The words heard in a hold still open: none where nothing was said in it lately, or where it closed before they
        were heard, since its own hearing, whole, then says what it holds."""
        # [LAW:nothing-unseen] each hearing of a hold still open is a unit of work: which hold, how much of it, what was
        # heard in it, and what came of that.
        with unit("whisper.overheard", self._record):
            annotate(hold=partial.opened.hold.number, seconds=round(partial.seconds, 3))
            match partial.opened.overhearing:
                case "over":
                    # Closed or thrown away before the model took it: not heard at all.
                    annotate(heard="moot")
                    return ()
                case "listening" | "queued":
                    pass
            # Where hearing it stands is read again once it is heard: the hold can close, or be thrown away, meanwhile.
            opened = partial.opened
            try:
                async with asyncio.timeout(TRANSCRIBING_SECONDS):
                    heard = await self._heard(opened.hold.number, partial.levels, audio, await self._primed(opened))
            except Exception as error:
                # [LAW:no-silent-failure] failed on its event; the hold's own hearing as it closes says it aloud if it
                # fails too.
                opened.overhearing = "over"
                fail(f"Whisper could not hear hold {opened.hold.number} while it was open: {type(error).__name__}: {error}")
                return ()
            annotate(hearing=heard)
            match opened.overhearing, heard.said:
                case "over", _:
                    annotate(heard="moot")
                    return ()
                case _, None:
                    # [LAW:single-enforcer] what counts as words is what `_heard` keeps: noise, a cough, or hands' echo
                    # left over by the canceller is none, and cuts nothing.
                    annotate(heard="nothing")
                    opened.overhearing = "listening"
                    return ()
                case _, said:
                    annotate(heard="words")
                    opened.overhearing = "over"
                    return (InterimWords(said, self._user_id, time_now_iso8601(), LANGUAGE),)

    async def _primed(self, opened: _Open) -> str | None:
        """The vocabulary the hearings of the hold open are primed with: read for its first hearing, kept for the rest."""
        match opened.prompt:
            case _Unread():
                prompt = await self._prompt()
                opened.prompt = prompt
                return prompt
            case prompt:
                return prompt

    async def _transcribed(self, hold: Hold, levels: Levels, audio: bytes) -> AsyncGenerator[Frame, None]:
        """The words said in a hold of the key's, none, or why it could not be transcribed."""
        # [LAW:no-silent-failure] a transcription that never returns would hold its turn open for ever, since nothing but
        # Whisper resolving its holds ends one: it fails instead.
        bound = asyncio.timeout(TRANSCRIBING_SECONDS)
        try:
            async with bound:
                heard = await self._heard(hold.number, levels, audio, await self._prompt())
                # Recorded, so "I spoke and nothing happened" can be looked into.
                self._record(heard)
                match heard.said:
                    case None:
                        given = None
                    case said:
                        # Told inside the bound too: a speaker model that never returns holds the turn open as surely.
                        given = spoken_as(said, await self._speaker(hold, audio))
            match heard.said, given:
                case str() as said, str() as given:
                    await self._handle_transcription(said, True, LANGUAGE)  # pyright: ignore[reportUnknownMemberType]  (Pipecat's tracing decorator is untyped)
                    yield Words(given, self._user_id, time_now_iso8601(), LANGUAGE)
                case _:
                    # Not said: Brandon does not need to hear it (2026-09-27).
                    pass
        except Exception as error:
            # [LAW:no-silent-failure] a failed transcription is heard: the pipeline says an ErrorFrame from Whisper aloud.
            why = f"nothing after {TRANSCRIBING_SECONDS:g} s" if bound.expired() else f"{type(error).__name__}: {error}"
            yield ErrorFrame(error=f"Whisper could not transcribe hold {hold.number}: {why}", exception=error)

    async def _speaker(self, hold: Hold, audio: bytes) -> Speaker:
        """Whose voice the hold in `audio` is in, recorded; told on the voice alone, without the silence Pipecat pads a
        hold's WAV with."""
        samples = _samples(audio)[: -len(self._trailing_silence()) or None]
        try:
            speaker = await self._speakers.run(lambda: self._told(samples, hold))
        except Exception as error:
            # [LAW:no-silent-failure] the failure is an error line; the words Whisper heard still reach the brain.
            speaker = Untellable(f"{type(error).__name__}: {error}")
        self._record(Voiced(hold.number, speaker, round(seconds(samples), 3)))
        return speaker

    async def _heard(self, hold: int, levels: Levels, audio: bytes, prompt: str | None) -> HoldHeard:
        """What was said in a hold's WAV, primed with `prompt`, the vocabulary.

        A segment with no word in it is dropped, as are one the decoder repeated itself in and one Whisper only guessed
        at: that is what a primed Whisper makes of noise. A hold with nothing said in it comes back with no segment.
        """
        # [LAW:parse-dont-validate] a WAV in any form but the one Whisper hears is refused here, not uploaded.
        _samples(audio)
        await self.start_processing_metrics()
        try:
            scored = await transcription.segments(self._url, audio, f"hold-{hold}.wav", prompt, ANSWER_SECONDS)
        finally:
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
