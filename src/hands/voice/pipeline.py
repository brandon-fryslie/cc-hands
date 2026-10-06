"""Assemble the voice pipeline from a configuration.

Building it loads the two local models, Whisper's and pocket-tts's, so a
built voice answers its first turn as fast as its tenth; nothing here opens a
microphone or calls an API until the returned worker is run. `VoiceConfig` is
the whole variability of the pipeline as data.
"""

import asyncio
import math
from itertools import takewhile
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
    LLMContextFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.pocket_tts.tts import PocketTTSService
from pipecat.transports.local.audio import LocalAudioTransportParams
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from hands.sessions.model_facts import ModelFault, ModelReplyEmpty
from hands.sessions.wide import Begun, Fact, begun, ended
from hands.voice.transcript import TranscriptObserver
from hands.voice.wakeword import Pretrained, Word
from hands.voice.floor import Floor
from hands.voice.refocus import Refocus, Refocusing
from hands.voice.latency import LatencyObserver
from hands.voice.utterance import Audible
from hands.voice.microphone import KeyedAudioTransport
from hands.voice.phone import Phone
from hands.sessions.audit import Record
from hands.voice.player import Marks, Player
from hands.voice.ptt import PushToTalk
from hands.voice.spoken import FenceAggregator, SpokenForm
from hands.voice.tool import Tool
from hands.voice.tools import RunsReplies, context_tools
from hands.voice.turnstart import EdgeTurnStart, interrupting
from hands.voice.turnstop import KeyTurnStop
from hands.voice import conversation, voices
from hands.voice.whisper import Whisper
from hands.voice.backends import AnthropicBackend, LLMBackend, OpenAICompatibleBackend
from hands.voice.token_usage import INPUT_USAGE, USAGE, reported

@dataclass(frozen=True)
class VoiceConfig:
    """Everything that varies between two runs of the pipeline."""

    llm: LLMBackend
    voice: voices.Voice
    # How the model comes across, in the user's words; None is hands' own.
    personality: str | None = None
    # What the wake word trigger listens for.
    wake: Word = Pretrained()
    max_reply_tokens: int = 300


class EmptyReplyFails(RunsReplies):
    """An LLM service whose reply that carries no words, no call, and no error of its own is reported as the model's
    failure: in a voice turn, a reply with nothing in it is heard as hands not having heard the user at all."""

    # The reply streaming now has been opened and nothing in it has been heard yet.
    _empty = False
    # The context being answered holds a call's result after the model's last reply: its call already did what the turn was for.
    _answers_call = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, LLMContextFrame):
            self._answers_call = _answers_call(frame.context)
        await super().process_frame(frame, direction)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        # [LAW:single-enforcer] the reply is read off the frames the service itself pushes, which both services push alike.
        match frame:
            # Claude often answers a call's result with nothing, as the brain's does: a turn that called a tool is no empty reply.
            case LLMFullResponseStartFrame():
                self._empty = not self._answers_call
            # A call is the model choosing what happens, stay_silent included; an error of the service's own is already said.
            case LLMTextFrame(text=text) if text.strip():
                self._empty = False
            case FunctionCallsStartedFrame():
                self._empty = False
            case ErrorFrame(processor=processor) if processor is self:
                self._empty = False
            case LLMFullResponseEndFrame():
                # A reply whose task is being cancelled is abandoned, not empty: the user spoke over it or the pipeline is
                # stopping, and Pipecat closes it with the same end frame as a finished one.
                if self._empty and not _cancelling():
                    # [LAW:no-silent-failure] said by the system channel as the model's failure, ahead of the reply's end.
                    await self.push_error("the language model's reply was empty", exception=ModelFault(ModelReplyEmpty()))  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
                self._empty = False
            case _:
                pass
        await super().push_frame(frame, direction)


def _answers_call(context: LLMContext) -> bool:
    """A call's result came in since the model last replied. Not the last message alone: Pipecat writes a result into the
    tool message its call opened, and a note hands added while the call ran sits after it."""
    since = takewhile(lambda message: not (isinstance(message, dict) and message.get("role") == "assistant"), reversed(context.get_messages()))
    return any(isinstance(message, dict) and message.get("role") == "tool" for message in since)


def _cancelling() -> bool:
    """The task pushing a frame is being cancelled, as Pipecat cancels a reply's task to stop it."""
    return _this_task().cancelling() > 0


def _this_task() -> asyncio.Task[Any]:
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("an LLM frame was pushed outside a task, so the reply it belongs to cannot be known")
    return task


@dataclass
class _Request:
    """A request a service is answering: when it began, the task answering it, the usage its API reported, and what the
    service said failed it, the exception where there was one."""

    began: Begun
    task: asyncio.Task[Any]
    usage: dict[str, Fact] = field(default_factory=dict[str, Fact])
    raised: BaseException | None = None
    failure: str | None = None


class RequestEvents(EmptyReplyFails):
    """An LLM service whose empty reply fails, and each of whose requests is one wide event, `model.request`, saying how
    it ended and the usage its API reported."""

    # The request being answered, in the task its context frame came in on: a service answers one at a time.
    _request: _Request | None = None

    def __init__(self, *args: Any, record: Record, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._record = record

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if not isinstance(frame, LLMContextFrame):
            await super().process_frame(frame, direction)
            return
        # [LAW:nothing-unseen] every request is asked by a context frame and answered before its processing returns.
        # Emitted by `ended`, not run as a `unit`: a unit is open in the context each task the request starts copies,
        # and a tool call it starts is work of its own that outlives it.
        request = self._request = _Request(begun(), _this_task())
        raised: BaseException | None = None
        try:
            await super().process_frame(frame, direction)
        except BaseException as error:
            raised = error
            raise
        finally:
            self._request = None
            ended("model.request", self._record, request.began, raised or request.raised, request.failure, **request.usage)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        request = self._request
        # [LAW:no-silent-failure] Pipecat catches what the API raised and says it as this frame alone, so the request
        # fails by it. Only by one the answering task pushes: a tool call says its own failure from its own task.
        if isinstance(frame, ErrorFrame) and frame.processor is self and request is not None and request.task is _this_task():
            request.raised, request.failure = frame.exception, frame.error
        await super().push_frame(frame, direction)

    def _answering(self) -> _Request:
        if self._request is None:
            raise LookupError("a reply's usage was read with no request being answered")
        return self._request


class AnthropicService(RequestEvents, AnthropicLLMService):
    """Claude through Pipecat's Anthropic service, whose request that timed out is said as the OpenAI-compatible one's is."""

    async def _create_message_stream(self, api_call: Any, params: Any) -> AsyncIterator[Any]:
        stream = await super()._create_message_stream(api_call, params)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  (untyped in Pipecat)
        return _usage_read(cast(AsyncIterator[Any], stream), self._answering().usage)

    async def _call_event_handler(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        if event_name == "on_completion_timeout":
            # [LAW:no-silent-failure] Pipecat's Anthropic service drops a request that timed out with this event alone
            # (1.10.0), where the OpenAI one pushes an error too. A timeout is the model out of reach, not an empty reply.
            await self.push_error("LLM completion timeout", exception=TimeoutError("LLM completion timeout"))  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        await super()._call_event_handler(event_name, *args, **kwargs)  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)


async def _usage_read(stream: AsyncIterator[Any], usage: dict[str, Fact]) -> AsyncIterator[Any]:
    """The stream as it came, with the usage its events report written into `usage` as each arrives.

    Read here, not from Pipecat's figures: Pipecat adds message_delta's usage to message_start's (1.10.0), and the API
    reports message_delta's cumulatively, so its input count is twice what was read. Each later report replaces an
    earlier one, field by field; a field an event leaves out is not one it reported."""
    async for event in stream:
        match event.type:
            case "message_start":
                usage.update(reported(event.message.usage, INPUT_USAGE))
            case "message_delta":
                usage.update(reported(event.usage, USAGE))
            case _:
                pass
        yield event


class FailFastOpenAILLMService(RequestEvents, OpenAILLMService):
    """An OpenAI-compatible model whose failed request is reported at once, not retried with backoff."""

    def create_client(self, *args: Any, **kwargs: Any) -> AsyncOpenAI:
        # [LAW:no-silent-failure] the SDK's two retries turn a refused connection into 4.6 s of silence (measured
        # against inferno); in a voice turn that sounds like thinking, so the failure is spoken instead.
        client: AsyncOpenAI = super().create_client(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        return client.with_options(max_retries=0)

    async def start_llm_usage_metrics(self, tokens: LLMTokenUsage) -> None:
        # Pipecat holds the last usage the stream reported and gives it here once, in the answering task, cut off or not.
        # Its prompt_tokens is gross of the cache, as these APIs count it, where Anthropic's input_tokens is net.
        self._answering().usage.update(tokens.model_dump(exclude_none=True))
        await super().start_llm_usage_metrics(tokens)


def build_llm(
    backend: AnthropicBackend | OpenAICompatibleBackend, *, instruction: str, max_tokens: int, record: Record
) -> AnthropicLLMService | OpenAILLMService:
    """The one place an API backend's variant is inspected."""
    # [LAW:one-type-per-behavior] both services speak the same frame protocol
    # to the rest of the pipeline; only their construction differs.
    match backend:
        case AnthropicBackend(base_url=base_url, api_key=api_key, model=model):
            return AnthropicService(
                api_key=api_key,
                # Reported at once, as for the OpenAI-compatible model: a retry with backoff is silence in a voice turn.
                client=AsyncAnthropic(base_url=base_url, api_key=api_key, max_retries=0),
                record=record,
                settings=AnthropicService.Settings(
                    # Pipecat marks the last two user messages, and the cache is a prefix: the tools and the instruction
                    # ahead of them are read from it, at a tenth of the input price, by a request within five minutes of the last.
                    model=model, system_instruction=instruction, max_tokens=max_tokens, enable_prompt_caching=True
                ),
            )
        case OpenAICompatibleBackend(base_url=base_url, api_key=api_key, model=model):
            return FailFastOpenAILLMService(
                base_url=base_url,
                api_key=api_key,
                record=record,
                settings=OpenAILLMService.Settings(
                    model=model, system_instruction=instruction, max_tokens=max_tokens
                ),
            )


@dataclass(frozen=True)
class Voice:
    """The assembled pipeline plus the handles its edges need: the key, the audio devices, the phone, the three services that report failures, and the two sides of the conversation."""

    worker: PipelineWorker
    key: PushToTalk
    audio: KeyedAudioTransport
    phone: Phone
    stt: Whisper
    llm: FrameProcessor
    tts: PocketTTSService
    user_turns: LLMUserAggregator
    assistant_turns: LLMAssistantAggregator


def build_voice(
    config: VoiceConfig, tools: Sequence[Tool], llm: FrameProcessor, noting: Sequence[FrameProcessor], key: PushToTalk, player: Player, floor: Floor, refocus: Refocus, prompt: Callable[[], Awaitable[str | None]], record: Record
) -> Voice:
    """Wire mic, push-to-talk, Whisper on MLX, the model's stage, pocket-tts, speakers, and the phone beside the mic and speakers."""
    # [LAW:one-source-of-truth] the key is the only voice activity signal:
    # it mutes the microphone at the transport, and Whisper reads it off each
    # frame to push the VAD frames the turn strategies act on, so the user
    # aggregator runs no VAD of its own. The turn opens on the press, and cuts
    # hands off there or, where the voice pressed it, on its first words
    # (`hands.voice.turnstart`); it closes on the release, which is final, so
    # there is no wait for the user to "say more".
    params = PipelineParams(enable_metrics=True)
    # [LAW:one-source-of-truth] the phone's audio is at the pipeline's own rates, as the desk's devices are opened at.
    phone = Phone(key, heard_rate=params.audio_in_sample_rate, played_rate=params.audio_out_sample_rate, record=record)
    transport = KeyedAudioTransport(LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True), key, phone, record)
    stt = Whisper(prompt=prompt, record=record)
    # [LAW:single-enforcer] every utterance is filtered here, whichever of them sent it: Pipecat applies a
    # TTS service's filters to the text of a TTSSpeakFrame and to each aggregated sentence of the model's
    # own reply alike, so this is the one place all of them meet before they are heard.
    # It also decides what the intermediary remembers, because the frame appended to the assistant context
    # is built from what a filter returned. Intended: the context is kept so the model can answer about
    # what the user heard, and before this filter it held what was sent to the speaker,
    # which was never the same string. See voice/spoken.py, which also records what it costs.
    tts = PocketTTSService(settings=PocketTTSService.Settings(voice=config.voice, language=voices.LANGUAGE), text_filters=[SpokenForm()])
    # The reply is broken into the pieces that filter sees here, ahead of the service, so a fenced block reaches it
    # whole; Pipecat flushes this aggregator at the end of each reply and resets it on a barge-in.
    pieces = LLMTextProcessor(text_aggregator=FenceAggregator())

    start = EdgeTurnStart(record)
    turns = UserTurnStrategies(
        start=[start],
        stop=[KeyTurnStop()],
    )
    context = LLMContext(tools=context_tools(tools, player.lines, llm))
    user_aggregator, assistant_aggregator = conversation.turns(
        context,
        # [LAW:single-enforcer] the key's holds alone end a user turn. Pipecat's aggregator also ends one itself after
        # this long with no speech and no transcript, which a slow or queued transcription outlasts: the turn would end
        # empty and the hold's words go out with the next one. Whisper resolves every hold it opens, heard or not or
        # failed, and fails one it cannot transcribe within TRANSCRIBING_SECONDS, so every turn ends.
        LLMUserAggregatorParams(user_turn_strategies=turns, user_turn_stop_timeout=math.inf),
    )
    interrupting(start, user_aggregator)

    # The floor ahead of the user aggregator, so what hands tells of the sessions waits out the user's turn before either the
    # context or the model's stage takes it, and follows the user's words when given back.
    # `noting` between the floor and the user aggregator, so a note of the user's words is in the context before the
    # turn that writes them ends, and before anything the floor held.
    # Right behind the model's stage, so the focus moves to a session told of in order with the user's words to the model.
    # What the player says again enters just ahead of the speaker, and it reads what is played off the speaker's pushes.
    # Right behind the output transport, which passes a mark on only once what was said ahead of it has played.
    output = transport.output()
    pipeline = Pipeline([transport.input(), stt, floor, *noting, user_aggregator, llm, Refocusing(refocus), pieces, player.lines, tts, output, Marks(), assistant_aggregator])
    worker = PipelineWorker(
        pipeline,
        params=params,
        # The phone is told each mark of every turn, as the latency log is, and each line of the transcript, and passes
        # them to the page of the call that is up.
        observers=[LatencyObserver(phone.tell), TranscriptObserver(stt, output, phone.tell), player.watching(tts, output), Audible(output)],
        idle_timeout_secs=None,
    )
    return Voice(
        worker=worker, key=key, audio=transport, phone=phone, stt=stt, llm=llm, tts=tts, user_turns=user_aggregator, assistant_turns=assistant_aggregator
    )
