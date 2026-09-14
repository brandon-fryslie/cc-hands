"""The system channel: what hands says about itself, from a template, with no model in the way.

A failure of the model is spoken without the model, and a failure of speech is posted to the
screen instead. Every fact is logged as well, which is the path that needs neither.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import anthropic
import openai
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSSpeakFrame
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.utils.errors import ErrorCategory

from hands.voice.pipeline import Voice
from hands.voice.whisper import NOTHING_TRANSCRIBED, Whisper


@dataclass(frozen=True)
class Started:
    """The pipeline is running; after_crash when the run before this one ended without being stopped."""

    after_crash: bool


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


SystemFact = Started | ModelUnreachable | ModelFailed | TranscriptionFailed | NothingTranscribed


def system_text(fact: SystemFact) -> str:
    match fact:
        case Started(after_crash=after_crash):
            return "hands is back after a crash." if after_crash else "hands is up."
        case ModelUnreachable():
            return "The language model is unreachable."
        case ModelFailed(category=category):
            return f"The language model failed: {category.value.replace('_', ' ')}."
        case TranscriptionFailed():
            return "Speech recognition failed for that turn."
        case NothingTranscribed():
            return "Whisper returned nothing for that turn."


@dataclass(frozen=True)
class Say:
    fact: SystemFact


@dataclass(frozen=True)
class Post:
    """For the screen: speech is what failed."""

    text: str


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
            return Post(f"hands cannot speak: {error.error}")
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


Notify = Callable[[str], Awaitable[None]]


class SystemChannel:
    """Says each fact through text-to-speech alone, and posts it to the screen when speech cannot."""

    def __init__(self, tts: FrameProcessor, notify: Notify) -> None:
        self._tts = tts
        self._notify = notify

    async def say(self, fact: SystemFact) -> None:
        text = system_text(fact)
        logger.info(f"system: {text}")
        if self._tts.is_usable:
            # [LAW:effects-at-boundaries] queued at the TTS, past the model, because this channel reports the model's own failures;
            # kept out of the model's context too, where it would read as a reply the model gave.
            await self._tts.queue_frame(TTSSpeakFrame(text, append_to_context=False))
        else:
            await self._notify(f"hands cannot speak, so: {text}")

    async def sound(self, alarm: Alarm) -> None:
        match alarm:
            case Say(fact=fact):
                await self.say(fact)
            case Post(text=text):
                logger.error(text)
                await self._notify(text)
            case Unrouted(source=source, error=error):
                # [LAW:no-silent-failure] no sentence fits, so the log carries it.
                logger.error(f"{source} failed: {error}")


def listen(voice: Voice, channel: SystemChannel, started: Started) -> None:
    """Connect the pipeline's own reports to the channel: its start, its errors, and a turn with nothing in it."""

    @voice.worker.event_handler("on_pipeline_started")
    async def announce(_worker: PipelineWorker, _frame: Frame) -> None:  # pyright: ignore[reportUnusedFunction]
        await channel.say(started)

    @voice.worker.event_handler("on_pipeline_error")
    async def failed(_worker: PipelineWorker, error: ErrorFrame) -> None:  # pyright: ignore[reportUnusedFunction]
        await channel.sound(alarm(error, stt=voice.stt, llm=voice.llm, tts=voice.tts))

    @voice.stt.event_handler(NOTHING_TRANSCRIBED)
    async def empty(_stt: Whisper) -> None:  # pyright: ignore[reportUnusedFunction]
        await channel.say(NothingTranscribed())
