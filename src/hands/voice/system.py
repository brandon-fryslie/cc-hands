"""The system channel: what hands says about itself, from a template, with no model in the way.

A failure of the model is spoken without the model, and a failure of speech is posted to the
screen instead. Every fact is logged as well, which is the path that needs neither.
"""

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import ClassVar

import anthropic
import openai
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSSpeakFrame
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.utils.errors import ErrorCategory

from hands.sessions.audit import Announced, Record
from hands.voice.microphone import Devices
from hands.voice.pipeline import Voice
from hands.voice.whisper import NOTHING_TRANSCRIBED, Whisper


@dataclass(frozen=True)
class Started:
    """The pipeline is running on these devices; after_crash when the run before this one ended without being stopped."""

    after_crash: bool
    devices: Devices


@dataclass(frozen=True)
class ModelUnreachable:
    pass


@dataclass(frozen=True)
class ModelFailed:
    category: ErrorCategory


@dataclass(frozen=True)
class TranscriptionFailed:
    pass


@dataclass(frozen=True)
class NothingTranscribed:
    pass


@dataclass(frozen=True)
class AudioMoved:
    """The system's default devices changed, and the transport was reopened on these."""

    devices: Devices


SystemFact = Started | ModelUnreachable | ModelFailed | TranscriptionFailed | NothingTranscribed | AudioMoved


def system_text(fact: SystemFact) -> str:
    match fact:
        case Started(after_crash=after_crash, devices=Devices(input=input_)):
            up = "hands is back after a crash" if after_crash else "hands is up"
            return f"{up}." if input_ is not None else f"{up}, but there is no microphone, so it cannot hear you."
        case ModelUnreachable():
            return "The language model is unreachable."
        case ModelFailed(category=category):
            return f"The language model failed: {category.value.replace('_', ' ')}."
        case TranscriptionFailed():
            return "Speech recognition failed for that turn."
        case NothingTranscribed():
            return "Whisper returned nothing for that turn."
        case AudioMoved(devices=Devices(input=None, output=output)):
            # [LAW:no-silent-failure] said on whatever speaker is left, since nothing will be heard until a microphone is back.
            return f"No microphone: hands cannot hear you. Speaking on {output}."
        case AudioMoved(devices=Devices(input=input_, output=output)):
            return f"Audio moved: listening on {input_}, speaking on {output}."


@dataclass(frozen=True)
class Say:
    fact: SystemFact


@dataclass(frozen=True)
class Post:
    """For the screen: speech is what failed.

    `fault` is what recurs and `error` is what varies. Pipecat's own text for a silent utterance carries a fresh
    context id every time — `TTS context {uuid} completed with no audio`, pushed without costing the service its
    usability — so the sentence is never twice the same and cannot be what a burst is measured by. The fault it
    reports can, and this whole type reports one fault.
    """

    error: str
    fault: ClassVar[str] = "hands cannot speak"

    @property
    def text(self) -> str:
        return f"{self.fault}: {self.error}"


@dataclass(frozen=True)
class Unrouted:
    """An error from a processor the channel has no sentence for; it is logged."""

    source: str
    error: str


Alarm = Say | Post | Unrouted

# An SDK's connection error carries no status code and is no ConnectionError, so Pipecat files it as UNKNOWN.
_UNREACHABLE = (openai.APIConnectionError, anthropic.APIConnectionError, ConnectionError, TimeoutError)


def alarm(error: ErrorFrame, *, stt: FrameProcessor, llm: FrameProcessor, tts: FrameProcessor) -> Alarm:
    """What the user is told about a pipeline error, decided by the processor that raised it."""
    match error.processor:
        case processor if processor is tts:
            return Post(error.error)
        case processor if processor is llm:
            return Say(model_fact(error))
        case processor if processor is stt:
            return Say(TranscriptionFailed())
        case processor:
            return Unrouted(str(processor), error.error)


def model_fact(error: ErrorFrame) -> ModelUnreachable | ModelFailed:
    if isinstance(error.exception, _UNREACHABLE) or error.category is ErrorCategory.CONNECTIVITY:
        return ModelUnreachable()
    return ModelFailed(error.category or ErrorCategory.UNKNOWN)


# Posts to the screen. True when the screen took it, so nothing is recorded as given that was not given.
Notify = Callable[[str], Awaitable[bool]]

BURST_SECONDS = 10.0
"""How long a fault stays said, counted from the last time it was said and not from its last occurrence.

A fault that recurs recurs in bursts: a key held down on 2026-09-22 queued hundreds of empty turns whose reports
went out every 0.43 s for as long as they drained, which is not a loud failure but a jammed one, because nothing
else could have been heard while it ran. Ten seconds is twenty-odd times quieter than that cadence and still
answers a user who pressed the key again. It is a span of quiet rather than "until something else was said"
because in the case this exists for — a muted microphone, a model that is down — nothing else ever is.
"""


class SystemChannel:
    """Says each fact through text-to-speech alone, and posts it to the screen when speech cannot.

    It says a burst once: a sentence is not said again until `BURST_SECONDS` have passed since it last was, because
    is the one channel that reports the daemon's own faults and a fault that recurs recurs in bursts. What a burst
    costs is the saying and never the knowing — every occurrence is still a log line — and `Announced` is written
    where the sentence was taken: handed to a working TTS, or accepted by the screen, never where it was refused.
    """

    def __init__(
        self,
        tts: FrameProcessor,
        notify: Notify,
        record: Record,
        clock: Callable[[], float] = time.monotonic,  # monotonic seconds
        burst: float = BURST_SECONDS,
    ) -> None:
        self._tts = tts
        self._notify = notify
        self._record = record
        self._clock = clock
        self._burst = burst
        # When each sentence last reached the user, so the rest of its burst is logged and not said.
        self._said: dict[str, float] = {}

    async def say(self, fact: SystemFact) -> None:
        text = system_text(fact)
        logger.info(f"system: {text}")
        if not self._claim(text):
            # [LAW:no-silent-failure] the log line above is the knowing, which a burst never costs.
            return
        if self._tts.is_usable:
            # [LAW:effects-at-boundaries] queued at the TTS, past the model, because this channel reports the model's own failures;
            # kept out of the model's context too, where it would read as a reply the model gave.
            await self._tts.queue_frame(TTSSpeakFrame(text, append_to_context=False))
            self._record(Announced(text, "speech"))
        elif await self._notify(f"hands cannot speak, so: {text}"):
            self._record(Announced(text, "screen"))

    def _claim(self, fault: str) -> bool:
        """True when this fault may be said now, taking its window as it answers.

        The fault is what recurs, and it comes from a closed set by construction — a sentence `system_text` built
        from the `SystemFact` union, or `Post.fault` — never text from outside. Keyed on a sentence carrying a
        Pipecat error verbatim, this window would be taken afresh by every occurrence and would never once close.

        How often hands tries to reach the user and what the user was given are two different facts, kept apart:
        the window is taken here at the decision, with no await between the asking and the taking, while
        `Announced` is written below by whichever delivery actually took the sentence. Both halves matter.
        Pipecat dispatches each pipeline error on its own task, so a dead TTS raising once per queued frame puts
        hundreds of these in flight together and a window taken on the way out is one that none of them sees; and
        a screen with no GUI session to post into refuses every time, which must still cost only one attempt.
        """
        now = self._clock()
        # A fault older than its window can no longer keep anything quiet, so the map holds only those inside one.
        self._said = {said: at for said, at in self._said.items() if now - at < self._burst}
        if fault in self._said:
            return False
        self._said[fault] = now
        return True

    async def sound(self, alarm: Alarm) -> None:
        match alarm:
            case Say(fact=fact):
                await self.say(fact)
            case Post() as post:
                logger.error(post.text)
                # [LAW:single-enforcer] the same window `say` takes: a TTS that fails fails once per frame it
                # was handed, and hundreds of notifications fill the screen exactly as hundreds filled the ear.
                if self._claim(post.fault) and await self._notify(post.text):
                    self._record(Announced(post.text, "screen"))
            case Unrouted(source=source, error=error):
                # [LAW:no-silent-failure] no sentence fits, so the log carries it.
                logger.error(f"{source} failed: {error}")


def listen(voice: Voice, channel: SystemChannel, after_crash: bool) -> None:
    """Connect the pipeline's own reports to the channel: its start, its errors, and a turn with nothing in it."""

    @voice.worker.event_handler("on_pipeline_started")
    async def announce(_worker: PipelineWorker, _frame: Frame) -> None:  # pyright: ignore[reportUnusedFunction]
        # The devices are read once the pipeline has opened its streams on them.
        await channel.say(Started(after_crash, voice.audio.devices))

    @voice.worker.event_handler("on_pipeline_error")
    async def failed(_worker: PipelineWorker, error: ErrorFrame) -> None:  # pyright: ignore[reportUnusedFunction]
        await channel.sound(alarm(error, stt=voice.stt, llm=voice.llm, tts=voice.tts))

    @voice.stt.event_handler(NOTHING_TRANSCRIBED)
    async def empty(_stt: Whisper) -> None:  # pyright: ignore[reportUnusedFunction]
        await channel.say(NothingTranscribed())
