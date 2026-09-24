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

Both stay on whatever devices the system calls its defaults. When a device goes, a
headset unplugged, PortAudio says nothing: measured 2026-09-24 against an aggregate
device destroyed mid-stream, the microphone's callbacks simply stop and a write to
the speaker blocks for good. So the transport is reopened whole when the defaults
change, which macOS does the moment the default device disappears.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol, cast

import pyaudio
from loguru import logger
from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.local.audio import (
    LocalAudioInputTransport,
    LocalAudioOutputTransport,
    LocalAudioTransport,
    LocalAudioTransportParams,
)

from hands.voice.coreaudio import DefaultDevices, default_devices
from hands.voice.ptt import Gate, PushToTalk
from hands.voice.threads import SerialThread, off_loop

Instant = float  # seconds on the monotonic clock

# From the end of a chunk given to the speaker to its sound having died away at the microphone, beyond
# the output stream's own latency: the room and the microphone's buffer. The measured total above was
# 185 ms after the last blocking write returned, against a reported output latency of 76 ms; this leaves
# about 40 ms of margin.
ECHO_PATH_SECS = 0.15

# How long a reopen may take before the run gives up on reopening in place. Closing a microphone whose device is
# gone was measured at 2.7 to 3.7 s; a reopen that never finishes fails the run, and launchd's restart opens a
# fresh process on the new defaults.
REOPEN_DEADLINE = timedelta(seconds=10)


class Stream(Protocol):
    """What is used of a PyAudio stream, which is untyped."""

    def start_stream(self) -> None: ...
    def stop_stream(self) -> None: ...
    def close(self) -> None: ...


class Playback(Stream, Protocol):
    """A stream the speaker writes to."""

    def write(self, audio: bytes) -> None: ...


class NoInput:
    """The microphone's stream when the system has no input device: nothing to start, stop, or close.

    [LAW:dataflow-not-control-flow] the transport opens, starts, stops, and closes the microphone the same way with
    or without one; only this value differs.
    """

    def start_stream(self) -> None:
        pass

    def stop_stream(self) -> None:
        pass

    def close(self) -> None:
        pass


class PortAudio(Protocol):
    """What is used of PyAudio itself, which is untyped."""

    def open(self, **settings: object) -> Playback: ...
    def get_format_from_width(self, width: int) -> int: ...
    def get_default_input_device_info(self) -> object: ...
    def get_default_output_device_info(self) -> object: ...
    def terminate(self) -> None: ...


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
        # When sound was last handed to the speaker, for the heartbeat; None before the first.
        self.sounded_at: Instant | None = None
        # [LAW:single-enforcer] every write to the stream goes through this one thread, in place of Pipecat's executor,
        # whose thread the interpreter waits for at exit: a write blocked on a lost device would hold the process open.
        self._writes = SerialThread("speaker writes")
        # Set while a stream is attached; a write waits on it through a reopen.
        self._attached = asyncio.Event()

    async def setup(self, setup: FrameProcessorSetup) -> None:
        # [LAW:one-source-of-truth] Pipecat's local setup is only the base's and an open; the open is open_stream's, so
        # the stream is opened one way at setup and at every reopen.
        await BaseOutputTransport.setup(self, setup)
        py_audio = cast(PortAudio, self._py_audio)
        self.attach(py_audio, self.open_stream(py_audio))

    def open_stream(self, py_audio: PortAudio) -> Playback:
        """A new stream on the default output, not yet the speaker's. Blocking: PortAudio talks to the device."""
        return py_audio.open(
            format=py_audio.get_format_from_width(2),
            channels=self._params.audio_out_channels,
            rate=self.sample_rate,
            output=True,
            output_device_index=self._params.output_device_index,
        )

    def attach(self, py_audio: PortAudio, stream: Playback) -> None:
        self._py_audio = cast(pyaudio.PyAudio, py_audio)
        self._out_stream = stream
        # [LAW:one-source-of-truth] the output buffer's share of the fade is read from the open stream,
        # so a speaker with a longer buffer keeps the microphone shut for longer.
        self._fade = _output_stream(self).get_output_latency() + ECHO_PATH_SECS
        self._attached.set()

    def detach(self) -> Playback:
        """Take the stream away: a write from here on waits for the next one to be attached."""
        stream: Playback | None = self._out_stream
        assert stream is not None, "the speaker stream opens in setup"
        self._attached.clear()
        self._out_stream = None
        return stream

    async def let_go(self, stream: Playback) -> None:
        """Stop a detached stream, wait out the writes already given to it, then close it."""
        # [LAW:no-ambient-temporal-coupling] closing frees the stream, so it waits until no write is inside it.
        # Stopping is what releases a write blocked on a lost device, and every write handed over before the
        # detach has returned once the writer thread reaches the empty work queued behind them.
        await off_loop(lambda: _letting_go(stream.stop_stream), "stopping the speaker")
        await self._writes.run(lambda: None)
        await off_loop(stream.close, "closing the speaker")

    async def cleanup(self) -> None:
        # Pipecat's local cleanup stops and closes the stream on the loop, under a write that may still be running;
        # the stream is let go of here the one way a reopen does it, off the loop.
        await BaseOutputTransport.cleanup(self)
        await _let_go_at_cleanup(self._out_stream, self.let_go, self._attached.clear)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        # [LAW:dataflow-not-control-flow] every frame is written; one that comes while the transport reopens waits for
        # the new stream, so a reply carries on over the move instead of losing its middle.
        await self._attached.wait()
        stream = cast(Playback, self._out_stream)
        if frame.audio.count(0) != len(frame.audio):
            # [LAW:no-ambient-temporal-coupling] recorded before the write is awaited: an interruption cancels
            # the await, but the chunk already handed to PortAudio's thread plays out all the same.
            # Silence padding makes no sound, so it holds nothing shut.
            self.sounded_at = self._clock()
            self.quiet_at = self.sounded_at + _duration(frame) + self._fade
        await self._writes.run(lambda: stream.write(frame.audio))
        return True


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

    async def setup(self, setup: FrameProcessorSetup) -> None:
        # As for the speaker: the base's setup, then the one way the stream is opened.
        await BaseInputTransport.setup(self, setup)
        py_audio = cast(PortAudio, self._py_audio)
        self.attach(py_audio, self.open_stream(py_audio))

    def open_stream(self, py_audio: PortAudio) -> Stream:
        """A new stream on the default input, not yet the microphone's. Blocking: PortAudio talks to the device."""
        match default_input(py_audio):
            case None:
                return NoInput()
            case _:
                return self._open(py_audio)

    def _open(self, py_audio: PortAudio) -> Stream:
        return py_audio.open(
            format=py_audio.get_format_from_width(2),
            channels=self._params.audio_in_channels,
            rate=self._sample_rate,
            frames_per_buffer=int(self._sample_rate / 100) * 2,  # 20 ms, as Pipecat opens it
            stream_callback=self._audio_in_callback,
            input=True,
            input_device_index=self._params.input_device_index,
        )

    def attach(self, py_audio: PortAudio, stream: Stream) -> None:
        self._py_audio = cast(pyaudio.PyAudio, py_audio)
        self._in_stream = stream

    def detach(self) -> Stream:
        stream: Stream | None = self._in_stream
        assert stream is not None, "the microphone stream opens in setup"
        self._in_stream = None
        return stream

    async def let_go(self, stream: Stream) -> None:
        """Stop and close a detached stream. Measured at 2.7 to 3.7 s when its device is gone."""

        def stop_and_close() -> None:
            _letting_go(stream.stop_stream)
            stream.close()

        await off_loop(stop_and_close, "closing the microphone")

    async def cleanup(self) -> None:
        # As for the speaker: off the loop, where closing a microphone whose device is gone takes seconds.
        await BaseInputTransport.cleanup(self)
        await _let_go_at_cleanup(self._in_stream, self.let_go, lambda: None)

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


@dataclass(frozen=True)
class Devices:
    """The devices the transport is on, by the names the system gives them."""

    input: str | None  # None when the system has no input device at all, so hands cannot hear
    output: str


class KeyedAudioTransport(LocalAudioTransport):
    """`LocalAudioTransport` with the keyed microphone and the speaker it listens past."""

    def __init__(
        self,
        params: LocalAudioTransportParams,
        key: PushToTalk,
        clock: Callable[[], Instant] = time.monotonic,
        portaudio: Callable[[], PortAudio] = lambda: cast(PortAudio, pyaudio.PyAudio()),
        defaults: Callable[[], DefaultDevices] = default_devices,
    ) -> None:
        # [LAW:one-source-of-truth] which defaults the streams are on is read where PortAudio lists them, just before
        # it does, here and at every reopen; a change after the read is one the follower sees. Read any later, it
        # would miss an unplug during the model load between this and the pipeline's start.
        self.opened_on = defaults()
        # [LAW:single-enforcer] PortAudio is started by the one factory, here and at every reopen. LocalAudioTransport's
        # own init would start an instance of its own, and while two are alive, ending one lists no devices anew.
        BaseTransport.__init__(self)
        self._params = params
        self._pyaudio = cast(pyaudio.PyAudio, portaudio())
        # [LAW:effects-at-boundaries] the speaker's writes and the microphone's captures are stamped from one clock.
        self._speaker = Speaker(self._pyaudio, params, clock)
        self._microphone = KeyedMicrophone(self._pyaudio, params, key, self._speaker, clock)
        self._portaudio = portaudio
        self._defaults = defaults

    def input(self) -> KeyedMicrophone:
        return self._microphone

    def output(self) -> Speaker:
        return self._speaker

    async def reopen(self) -> Devices:
        """Close both streams and PortAudio itself, and open them again on the system's default devices as they are now."""
        # [LAW:no-ambient-temporal-coupling] the order is the mechanism. Detached first, so a write that comes
        # meanwhile finds no stream rather than the lost one. The speaker is let go of before the microphone,
        # because stopping it is what releases a write blocked on a lost device. PortAudio is ended before it is
        # started again, because it lists the devices only when it starts, and while one instance lives a new one
        # still sees the device that is gone. Every step that talks to a device runs off the loop, inside the
        # deadline, so a device that hangs stops neither the heartbeat nor the run's own stop.
        speaker, microphone, ending = self._speaker.detach(), self._microphone.detach(), self._audio_now()
        async with asyncio.timeout(REOPEN_DEADLINE.total_seconds()):
            await self._speaker.let_go(speaker)
            await self._microphone.let_go(microphone)
            started = await off_loop(lambda: self._start_again(ending), "starting PortAudio again")
        self.opened_on = started.defaults
        # [LAW:one-source-of-truth] one PortAudio handle, shared by the transport and both sides; the ended one is dropped everywhere.
        self._pyaudio = cast(pyaudio.PyAudio, started.portaudio)
        self._speaker.attach(started.portaudio, started.speaker)
        self._microphone.attach(started.portaudio, started.microphone)
        # A write that timed out on the lost device cost the speaker its usability; the new device has it back.
        await self._speaker.set_usable(True)
        return self.devices

    @property
    def devices(self) -> Devices:
        """The devices the streams are open on: the defaults as PortAudio listed them when it started."""
        return Devices(default_input(self._audio_now()), _name(self._audio_now().get_default_output_device_info()))

    def _audio_now(self) -> PortAudio:
        return cast(PortAudio, self._pyaudio)

    def _start_again(self, ending: PortAudio) -> "_Started":
        ending.terminate()
        defaults = self._defaults()
        portaudio = self._portaudio()
        speaker, microphone = self._speaker.open_stream(portaudio), self._microphone.open_stream(portaudio)
        speaker.start_stream()
        microphone.start_stream()
        return _Started(defaults, portaudio, speaker, microphone)


@dataclass(frozen=True)
class _Started:
    """PortAudio started again, with both streams open and running on it, not yet the sides'."""

    defaults: DefaultDevices
    portaudio: PortAudio
    speaker: Playback
    microphone: Stream


def _letting_go(step: Callable[[], None]) -> None:
    try:
        step()
    except OSError as error:
        # A stream on a device that is gone refuses to stop cleanly; it is let go of all the same, and said.
        logger.warning(f"letting go of an audio stream whose device is gone: {error}")


async def _let_go_at_cleanup[S: Stream](held: S | None, let_go: Callable[[S], Awaitable[None]], detached: Callable[[], None]) -> None:
    match held:
        case None:
            # Nothing is held: a reopen that failed part way through let go of the old stream and attached no new one.
            pass
        case stream:
            detached()
            async with asyncio.timeout(REOPEN_DEADLINE.total_seconds()):
                await let_go(stream)


def default_input(py_audio: PortAudio) -> str | None:
    """The default input device's name, or None when the system has none: a Mac with no built-in microphone, its headset unplugged."""
    # [LAW:parse-dont-validate] PortAudio says there is no default input by raising; here that becomes a typed absence.
    try:
        info = py_audio.get_default_input_device_info()
    except OSError:
        return None
    return _name(info)


def _name(info: object) -> str:
    # PyAudio's device info is an untyped mapping; the name is the one field read from it.
    match info:
        case {"name": str(name)}:
            return name
        case _:
            raise AssertionError(f"PortAudio described a device without a name: {info!r}")


class _OutputStream(Protocol):
    def get_output_latency(self) -> float: ...


def _output_stream(speaker: LocalAudioOutputTransport) -> _OutputStream:
    # PyAudio is untyped; this names the one thing read from the open stream.
    stream = cast(_OutputStream | None, speaker._out_stream)  # pyright: ignore[reportPrivateUsage]
    assert stream is not None, "the speaker stream opens in setup"
    return stream
