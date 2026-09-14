"""The Mac's own microphone and speakers, deciding what the pipeline hears at the moment sound is captured.

Two sounds must not reach the pipeline as the user's voice. The first is anything
captured while the key is up. Pipecat's input filter decides that when the event
loop gets to a frame, tens of milliseconds after capture, so a press passes audio
recorded before it; the key is read in the capture callback instead. The second is
the pipeline's own speech. An interruption stops new audio reaching the speaker
within a few milliseconds, but what was already written still leaves the output
buffer and crosses the room: measured 2026-09-14 on MacBook Pro speakers and
microphone, the reply stayed above the room's floor for about 185 ms after the
last write, and Whisper made a word of it ("Wow.", "Well.", "What?") with nobody
speaking. So the speaker records when the sound it was given will have died away
at the microphone, and the microphone is silent until then.
"""

import time
from collections.abc import Callable
from typing import Protocol, cast

import pyaudio
from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.transports.local.audio import (
    LocalAudioInputTransport,
    LocalAudioOutputTransport,
    LocalAudioTransport,
    LocalAudioTransportParams,
)

from hands.voice.ptt import Gate, PushToTalk

Instant = float  # seconds on the monotonic clock

# From the end of a chunk given to the speaker to its sound having died away at the microphone, beyond
# the output stream's own latency: the room and the microphone's buffer. The measured total above was
# 185 ms after the last blocking write returned, against a reported output latency of 76 ms; this leaves
# about 40 ms of margin.
ECHO_PATH_SECS = 0.15


def heard(audio: bytes, gate: Gate, captured: Instant, speaker_quiet_at: Instant) -> bytes:
    """The microphone bytes as the pipeline hears them: intact only while the key is down and the speaker's sound is gone."""
    # [LAW:dataflow-not-control-flow] a frame of the same length always goes out; only its content is decided.
    return gate.audible(audio) if captured >= speaker_quiet_at else bytes(len(audio))


class Speaker(LocalAudioOutputTransport):
    """The local speaker, which records when the sound it has been given will have died away at the microphone."""

    def __init__(self, py_audio: pyaudio.PyAudio, params: LocalAudioTransportParams, clock: Callable[[], Instant]) -> None:
        super().__init__(py_audio, params)
        self._clock = clock
        self._fade = ECHO_PATH_SECS
        # Written by the output task, read by the microphone's capture thread: one float, whole either way.
        self.quiet_at: Instant = 0.0

    async def setup(self, setup: FrameProcessorSetup) -> None:
        await super().setup(setup)
        # [LAW:one-source-of-truth] the output buffer's share of the fade is read from the open stream,
        # so a speaker with a longer buffer keeps the microphone shut for longer.
        self._fade = _output_stream(self).get_output_latency() + ECHO_PATH_SECS

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        if frame.audio.count(0) != len(frame.audio):
            # [LAW:no-ambient-temporal-coupling] recorded before the write is awaited: an interruption cancels
            # the await, but the chunk already handed to PortAudio's thread plays out all the same.
            # Silence padding makes no sound, so it holds nothing shut.
            self.quiet_at = self._clock() + _duration(frame) + self._fade
        return await super().write_audio_frame(frame)


def _duration(frame: OutputAudioRawFrame) -> float:
    # 16-bit samples, as Pipecat's local transport opens its stream.
    return len(frame.audio) / (2 * frame.num_channels * frame.sample_rate)


class KeyedMicrophone(LocalAudioInputTransport):
    """The local microphone, silent while the key is up and while the speaker's sound is still in the room."""

    def __init__(self, py_audio: pyaudio.PyAudio, params: LocalAudioTransportParams, key: PushToTalk, speaker: Speaker, clock: Callable[[], Instant]) -> None:
        super().__init__(py_audio, params)
        self._key = key
        self._speaker = speaker
        self._clock = clock

    def _audio_in_callback(self, in_data: bytes, frame_count: int, time_info: object, status: int) -> tuple[None, int]:
        # [LAW:no-ambient-temporal-coupling] decided on the audio thread, not when the event loop gets to the
        # frame, and dated by when the sound was recorded, not when the callback ran: a callback held up behind
        # other work delivers audio from well before it, which may still be the speaker's.
        captured = self._clock() - buffer_age(time_info)
        audio = heard(in_data, self._key.gate, captured=captured, speaker_quiet_at=self._speaker.quiet_at)
        return _deliver(self, audio, frame_count, time_info, status)


def buffer_age(time_info: object) -> float:
    """How long before its callback a microphone buffer's first sample was recorded, from PortAudio's stream times."""
    # [LAW:parse-dont-validate] PortAudio reports zero for a time the host API does not know, and then the
    # callback's own moment is the best there is; measured on CoreAudio, a buffer is about 27 ms old.
    match time_info:
        case {"current_time": float(now), "input_buffer_adc_time": float(recorded)} if recorded > 0.0:
            return max(0.0, now - recorded)
        case _:
            return 0.0


# Pipecat's callback is untyped; this names what it takes and returns.
_deliver = cast(
    Callable[[LocalAudioInputTransport, bytes, int, object, int], tuple[None, int]],
    LocalAudioInputTransport._audio_in_callback,  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
)


class KeyedAudioTransport(LocalAudioTransport):
    """`LocalAudioTransport` with the keyed microphone and the speaker it listens past."""

    def __init__(self, params: LocalAudioTransportParams, key: PushToTalk, clock: Callable[[], Instant] = time.monotonic) -> None:
        super().__init__(params)
        # [LAW:effects-at-boundaries] the speaker's writes and the microphone's captures are stamped from one clock.
        self._speaker = Speaker(self._pyaudio, params, clock)
        self._microphone = KeyedMicrophone(self._pyaudio, params, key, self._speaker, clock)

    def input(self) -> KeyedMicrophone:
        return self._microphone

    def output(self) -> Speaker:
        return self._speaker


class _OutputStream(Protocol):
    def get_output_latency(self) -> float: ...


def _output_stream(speaker: LocalAudioOutputTransport) -> _OutputStream:
    # PyAudio is untyped; this names the one thing read from the open stream.
    stream = cast(_OutputStream | None, speaker._out_stream)  # pyright: ignore[reportPrivateUsage]
    assert stream is not None, "the speaker stream opens in setup"
    return stream
