"""Assemble the voice pipeline from a configuration.

Building it loads the two local models, Whisper's and pocket-tts's, so a
built voice answers its first turn as fast as its tenth; nothing here opens a
microphone or calls an API until the returned worker is run. `VoiceConfig` is
the whole variability of the pipeline as data.
"""

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.services.pocket_tts.tts import PocketTTSService
from pipecat.transports.local.audio import LocalAudioTransportParams
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from hands.voice.transcript import TranscriptObserver
from hands.voice.wakeword import Pretrained, Word
from hands.voice.floor import Floor
from hands.voice.latency import LatencyObserver
from hands.voice.utterance import Audible
from hands.voice.microphone import KeyedAudioTransport
from hands.voice.phone import Phone
from hands.sessions.audit import Record, Speaker
from hands.voice.player import Marks, Player
from hands.voice.ptt import PushToTalk
from hands.voice.spoken import FenceAggregator, SpokenForm
from hands.voice.trigger import Opener
from hands.voice.turnstart import EdgeTurnStart, interrupting
from hands.voice.turnstop import KeyTurnStop
from hands.voice import conversation, voices
from hands.voice.whisper import Whisper
from hands.voice.backends import ClaudeCodeBackend

@dataclass(frozen=True)
class VoiceConfig:
    """Everything that varies between two runs of the pipeline."""

    llm: ClaudeCodeBackend
    voice: voices.Voice
    # How the model comes across, in the user's words; None is hands' own.
    personality: str | None = None
    # What the wake word trigger listens for.
    wake: Word = Pretrained()


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
    config: VoiceConfig,
    llm: FrameProcessor,
    key: PushToTalk,
    player: Player,
    floor: Floor,
    prompt: Callable[[], Awaitable[str | None]],
    told: Callable[[bytes, Opener], Speaker],
    record: Record,
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
    stt = Whisper(prompt=prompt, told=told, record=record)
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
    # The brain's stage decides each reply itself, its tools answered by hands' MCP server, so the context carries none.
    context = LLMContext()
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
    # What the player says again enters just ahead of the speaker, and it reads what is played off the speaker's pushes.
    # Right behind the output transport, which passes a mark on only once what was said ahead of it has played.
    output = transport.output()
    pipeline = Pipeline([transport.input(), stt, floor, user_aggregator, llm, pieces, player.lines, tts, output, Marks(), assistant_aggregator])
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
