"""Assemble the voice pipeline from a configuration.

Pure with respect to the world: nothing here opens a microphone, loads a
model, or calls an API until the returned worker is run. The daemon's edges do
that. `VoiceConfig` is the whole variability of the pipeline as data.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMContextAggregatorPair,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.pocket_tts.tts import PocketTTSService
from pipecat.services.whisper.stt import WhisperSTTServiceMLX
from pipecat.transports.local.audio import LocalAudioTransportParams
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from hands.voice.latency import LatencyObserver
from hands.voice.microphone import KeyedAudioTransport, Speaker
from hands.voice.ptt import KeyVAD, PushToTalk
from hands.voice.tools import Tool
from hands.voice.whisper import Whisper

# Replies are spoken, so the instruction is about speech, not personality.
# The real intermediary prompt is its own deliverable; this is the spike's.
SPOKEN_REPLY_INSTRUCTION = (
    "You are the voice intermediary for a developer's Claude Code sessions. "
    "Everything you say is spoken aloud: answer in one or two short sentences "
    "with no formatting, no lists, and no code. Call list_sessions when asked "
    "what is running. When the user dictates something for a session, stage it "
    "with stage_draft and say the readback; send it only when they say send. "
    "When a session asks permission, explain what it wants and ask; call "
    "answer_permission only with the decision the user gave."
)


# [LAW:types-are-the-program] the two ways to reach a model differ in what
# they need, not in what they do, so each is a variant with exactly its own
# fields; there is no bag of optional keys and URLs to guard downstream.
@dataclass(frozen=True)
class AnthropicBackend:
    """Claude over the Anthropic API."""

    api_key: str
    model: str


@dataclass(frozen=True)
class OpenAICompatibleBackend:
    """Any OpenAI-compatible chat completions server, such as mlx_lm.server."""

    base_url: str
    model: str


LLMBackend = AnthropicBackend | OpenAICompatibleBackend


@dataclass(frozen=True)
class VoiceConfig:
    """Everything that varies between two runs of the pipeline."""

    llm: LLMBackend
    whisper_model: str
    voice: str
    max_reply_tokens: int = 300


class FailFastOpenAILLMService(OpenAILLMService):
    """An OpenAI-compatible model whose failed request is reported at once, not retried with backoff."""

    def create_client(self, *args: Any, **kwargs: Any) -> AsyncOpenAI:
        # [LAW:no-silent-failure] the SDK's two retries turn a refused connection into 4.6 s of silence (measured
        # against inferno); in a voice turn that sounds like thinking, so the failure is spoken instead.
        client: AsyncOpenAI = super().create_client(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        return client.with_options(max_retries=0)


def build_llm(
    backend: LLMBackend, *, instruction: str, max_tokens: int
) -> AnthropicLLMService | OpenAILLMService:
    """The one place the backend variant is inspected."""
    # [LAW:one-type-per-behavior] both services speak the same frame protocol
    # to the rest of the pipeline; only their construction differs.
    match backend:
        case AnthropicBackend(api_key=api_key, model=model):
            return AnthropicLLMService(
                api_key=api_key,
                # Reported at once, as for the OpenAI-compatible model: a retry with backoff is silence in a voice turn.
                client=AsyncAnthropic(api_key=api_key, max_retries=0),
                settings=AnthropicLLMService.Settings(
                    model=model, system_instruction=instruction, max_tokens=max_tokens
                ),
            )
        case OpenAICompatibleBackend(base_url=base_url, model=model):
            return FailFastOpenAILLMService(
                base_url=base_url,
                api_key="unused",
                settings=OpenAILLMService.Settings(
                    model=model, system_instruction=instruction, max_tokens=max_tokens
                ),
            )


@dataclass(frozen=True)
class Voice:
    """The assembled pipeline plus the handles its edges need: the key, the speaker, the three services that report failures, and the two sides of the conversation."""

    worker: PipelineWorker
    key: PushToTalk
    speaker: Speaker
    stt: Whisper
    llm: AnthropicLLMService | OpenAILLMService
    tts: PocketTTSService
    user_turns: LLMUserAggregator
    assistant_turns: LLMAssistantAggregator


def build_voice(config: VoiceConfig, tools: Sequence[Tool]) -> Voice:
    """Wire mic, push-to-talk, Whisper on MLX, Claude, pocket-tts, speakers."""
    # [LAW:one-source-of-truth] the key is the only voice activity signal:
    # it mutes the microphone at the transport and it is the VAD the turn
    # strategies read. The turn opens on the press and closes on the release;
    # the release is final, so there is no wait for the user to "say more".
    key = PushToTalk()
    transport = KeyedAudioTransport(LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True), key)
    stt = Whisper(settings=WhisperSTTServiceMLX.Settings(model=config.whisper_model))
    llm = build_llm(
        config.llm, instruction=SPOKEN_REPLY_INSTRUCTION, max_tokens=config.max_reply_tokens
    )
    tts = PocketTTSService(settings=PocketTTSService.Settings(voice=config.voice))

    turns = UserTurnStrategies(
        start=[VADUserTurnStartStrategy()],
        stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0)],
    )
    context = LLMContext(tools=list(tools))
    pair = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=KeyVAD(key), user_turn_strategies=turns),
    )
    user_aggregator, assistant_aggregator = pair.user(), pair.assistant()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        observers=[LatencyObserver()],
        idle_timeout_secs=None,
    )
    return Voice(
        worker=worker, key=key, speaker=transport.output(), stt=stt, llm=llm, tts=tts, user_turns=user_aggregator, assistant_turns=assistant_aggregator
    )
