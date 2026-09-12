"""Assemble the voice pipeline from a configuration.

Pure with respect to the world: nothing here opens a microphone, loads a
model, or calls an API until the returned worker is run. The daemon's edges do
that. `VoiceConfig` is the whole variability of the pipeline as data.
"""

from dataclasses import dataclass

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.pocket_tts.tts import PocketTTSService
from pipecat.services.whisper.stt import WhisperSTTServiceMLX
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from hands.voice.latency import LatencyObserver
from hands.voice.ptt import KeyMute, KeyVAD, PushToTalk
from hands.voice.tools import list_sessions

# Replies are spoken, so the instruction is about speech, not personality.
# The real intermediary prompt is its own deliverable; this is the spike's.
SPOKEN_REPLY_INSTRUCTION = (
    "You are the voice intermediary for a developer's Claude Code sessions. "
    "Everything you say is spoken aloud: answer in one or two short sentences "
    "with no formatting, no lists, and no code. Call list_sessions when asked "
    "what is running."
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
                settings=AnthropicLLMService.Settings(
                    model=model, system_instruction=instruction, max_tokens=max_tokens
                ),
            )
        case OpenAICompatibleBackend(base_url=base_url, model=model):
            return OpenAILLMService(
                base_url=base_url,
                api_key="unused",
                settings=OpenAILLMService.Settings(
                    model=model, system_instruction=instruction, max_tokens=max_tokens
                ),
            )


@dataclass(frozen=True)
class Voice:
    """The assembled pipeline plus the handle the keyboard edge needs."""

    worker: PipelineWorker
    key: PushToTalk


def build_voice(config: VoiceConfig) -> Voice:
    """Wire mic, push-to-talk, Whisper on MLX, Claude, pocket-tts, speakers."""
    # [LAW:one-source-of-truth] the key is the only voice activity signal:
    # it mutes the microphone at the transport and it is the VAD the turn
    # strategies read. The turn opens on the press and closes on the release;
    # the release is final, so there is no wait for the user to "say more".
    key = PushToTalk()
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True, audio_out_enabled=True, audio_in_filter=KeyMute(key)
        )
    )
    stt = WhisperSTTServiceMLX(settings=WhisperSTTServiceMLX.Settings(model=config.whisper_model))
    llm = build_llm(
        config.llm, instruction=SPOKEN_REPLY_INSTRUCTION, max_tokens=config.max_reply_tokens
    )
    tts = PocketTTSService(settings=PocketTTSService.Settings(voice=config.voice))

    turns = UserTurnStrategies(
        start=[VADUserTurnStartStrategy()],
        stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0)],
    )
    context = LLMContext(tools=[list_sessions])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=KeyVAD(key), user_turn_strategies=turns),
    )

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
    return Voice(worker=worker, key=key)
