"""The Mac's own microphone and speakers, deciding what the pipeline hears at the moment sound is captured.

Two sounds must not reach the pipeline as the user's voice. The first is anything
captured while the key is up. Pipecat's input filter decides that when the event
loop gets to a frame, tens of milliseconds after capture, so a press passes audio
recorded before it; the key is read in the capture callback instead. The second is
the pipeline's own speech, which crosses the room from the speakers: an interruption
stops new audio reaching the speaker within a few milliseconds, but what was already
written still plays for about 185 ms (measured 2026-09-14, MacBook Pro speakers and
microphone), and Whisper made a word of it with nobody speaking. So everything the
speaker plays is given to the echo canceller (`hands.voice.echo`), and every buffer
the microphone captures is heard through it: the microphone stays open while the
speaker plays, and a press mid-reply keeps its first word.

Both stay on whatever devices the system calls its defaults. When a device goes, a
headset unplugged, PortAudio says nothing: measured 2026-09-24 against an aggregate
device destroyed mid-stream, the microphone's callbacks simply stop and a write to
the speaker blocks for good. So the transport is reopened whole when the defaults
change, which macOS does the moment the default device disappears.

The phone is the other place hands can be (`hands.voice.phone`). The gate says which: the pipeline hears only that
place's microphone, so Whisper is handed one stream of frames, and the speaker plays to that place, so a reply goes
where the user is.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from functools import partial
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol, cast

import pyaudio
from loguru import logger
from pipecat.frames.frames import EndWorkerFrame, OutputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
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
from hands.voice.cues import Cue, sound
from hands.sessions.audit import Record
from hands.sessions.wide import annotate, count, unit
from hands.voice.echo import COUNTS, Echo, EchoCanceller
from hands.voice.phone import Phone
from hands.voice.ptt import KeyedAudio, PushToTalk
from hands.threads import SerialThread, off_loop

Instant = float  # seconds on the monotonic clock

# The speaker stream's period, which sizes the buffer PortAudio keeps ahead of the device. Left to PortAudio, it is the
# device's low-latency default: 7 ms on BlackHole, 76 ms on a MacBook's speakers. Every chunk reaches the device through
# the event loop and the writer thread, so a stall longer than the buffer is a gap in the sound, and a reply broken by
# one every chunk is crunchy. 20 ms opens 105 ms on BlackHole and 138 ms on the speakers (measured, 24 kHz mono).
SPEAKER_PERIOD_SECS = 0.02

# How long a reopen may take before the run gives up on reopening in place. Closing a microphone whose device is
# gone was measured at 2.7 to 3.7 s; a reopen that never finishes fails the run, which reads as down, and the next
# `hands run` opens on the new defaults.
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


@dataclass(frozen=True)
class Output:
    """A speaker stream as opened, with the name of the device it was opened on and the canceller told what it plays."""

    stream: Playback
    device: str
    echo: Echo


@dataclass(frozen=True)
class Input:
    """A microphone stream as opened, with the name of its device, and the canceller its every buffer is heard through;
    no name, and a NoInput stream, when there is none."""

    stream: Stream
    device: str | None
    echo: Echo


class PortAudio(Protocol):
    """What is used of PyAudio itself, which is untyped."""

    def open(self, **settings: object) -> Playback: ...
    def get_format_from_width(self, width: int) -> int: ...
    def get_default_input_device_info(self) -> object: ...
    def get_default_output_device_info(self) -> object: ...
    def terminate(self) -> None: ...


class Speaker(LocalAudioOutputTransport):
    """The local speaker, which gives the echo canceller of the stream attached everything it plays, and which hands the
    sound to the phone instead while hands is there."""

    def __init__(self, py_audio: pyaudio.PyAudio, params: LocalAudioTransportParams, key: PushToTalk, phone: Phone, echo: Echo, clock: Callable[[], Instant]) -> None:
        super().__init__(py_audio, params)
        self._key = key
        self._phone = phone
        # The canceller of the attached stream; the first stream is opened with this one.
        self._echo = echo
        self._clock = clock
        # When sound was last handed to the speaker, for the heartbeat; None before the first.
        self.sounded_at: Instant | None = None
        # [LAW:single-enforcer] every write to the stream goes through this one thread, in place of Pipecat's executor,
        # whose thread the interpreter waits for at exit: a write blocked on a lost device would hold the process open.
        self._writes = SerialThread("speaker writes")
        # What the attached stream was opened on; None until setup opens the first.
        self.opened: Output | None = None
        # Set while a stream is attached; a write waits on it through a reopen.
        self._attached = asyncio.Event()

    async def setup(self, setup: FrameProcessorSetup) -> None:
        # [LAW:one-source-of-truth] Pipecat's local setup is only the base's and an open; the open is open_stream's, so
        # the stream is opened one way at setup and at every reopen.
        await BaseOutputTransport.setup(self, setup)
        py_audio = cast(PortAudio, self._py_audio)
        self.attach(py_audio, self.open_stream(py_audio, self._echo))

    def open_stream(self, py_audio: PortAudio, echo: Echo) -> Output:
        """A new stream on the default output, not yet the speaker's. Blocking: PortAudio talks to the device."""
        stream = py_audio.open(
            format=py_audio.get_format_from_width(2),
            channels=self._params.audio_out_channels,
            rate=self.sample_rate,
            frames_per_buffer=round(self.sample_rate * SPEAKER_PERIOD_SECS),
            output=True,
            output_device_index=self._params.output_device_index,
        )
        # [LAW:one-source-of-truth] the name is read from the same live PortAudio the stream was opened on, as it opens;
        # asked later, an instance that has since been ended answers "no device" whatever is plugged in.
        return Output(stream, _name(py_audio.get_default_output_device_info()), echo)

    def attach(self, py_audio: PortAudio, opened: Output) -> None:
        self._py_audio = cast(pyaudio.PyAudio, py_audio)
        self.opened = opened
        stream = opened.stream
        self._out_stream = stream
        self._echo = opened.echo
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
        # [LAW:one-source-of-truth] the gate says where hands is, read as each frame is written: a reply carries on at
        # the place the user has moved to. The phone's earbuds keep its microphone clear, so the canceller is not told.
        match self._key.gate.place:
            case "phone":
                if frame.audio.count(0) != len(frame.audio):
                    self.sounded_at = self._clock()
                await self._phone.play(frame.audio)
                return True
            case "desk":
                return await self._write_here(frame)

    async def _write_here(self, frame: OutputAudioRawFrame) -> bool:
        # [LAW:dataflow-not-control-flow] every frame is written; one that comes while the transport reopens waits for
        # the new stream, so a reply carries on over the move instead of losing its middle.
        await self._attached.wait()
        stream, echo = cast(Playback, self._out_stream), self._echo

        def write() -> None:
            # [LAW:no-ambient-temporal-coupling] given to the canceller on the writer thread as the chunk goes to the
            # device, not when it was handed over: a turn's cue queued ahead of it plays first, and the canceller's
            # reference stays in the order the room hears it. An interruption cancels the await, but the chunk
            # already handed to that thread plays out, and is cancelled, all the same.
            if frame.audio.count(0) != len(frame.audio):
                self.sounded_at = self._clock()
            echo.played(frame.audio, frame.sample_rate, frame.num_channels)
            stream.write(frame.audio)

        await self._writes.run(write)
        return True

    def cue(self, cue: Cue) -> None:
        """Hand a turn's cue to the stream attached now, ahead of the pipeline's next chunk, and not wait for it to play.

        [LAW:no-ambient-temporal-coupling] the talk key's edge calls this, and the key's next move never waits on the
        speaker, which a reopen holds for seconds. A cue belongs to the moment of its edge: with no stream attached
        there is nothing to play it on, and a tone that fails to play loses only the tone.

        It moves `sounded_at`, since it is sound given to the speaker, and the canceller hears it as it plays, so a
        word said over the tone that opens a turn reaches Whisper without the tone.
        """
        match self._key.gate.place, self._attached.is_set():
            case "phone", _:
                self.sounded_at = self._clock()
                # Not waited for, as a cue given to the desk's writer thread is not.
                self._phone.play(sound(cue, self.sample_rate, 1))
            case "desk", False:
                logger.warning(f"no speaker is attached; the tone for {cue.line!r} is not played")
            case "desk", True:
                stream, echo = cast(Playback, self._out_stream), self._echo
                audio = sound(cue, self.sample_rate, self._params.audio_out_channels)

                def write() -> None:
                    # A stream a reopen has stopped refuses the write; the writer thread says so and carries on.
                    self.sounded_at = self._clock()
                    echo.played(audio, self.sample_rate, self._params.audio_out_channels)
                    stream.write(audio)

                self._writes.give(write)


class KeyedMicrophone(LocalAudioInputTransport):
    """The local microphone, heard through the echo canceller and silent while the key is up; and the phone's, which
    the pipeline hears in its place while hands is at the phone."""

    def __init__(self, py_audio: pyaudio.PyAudio, params: LocalAudioTransportParams, key: PushToTalk, phone: Phone, echo: Echo, record: Record) -> None:
        super().__init__(py_audio, params)
        self._key = key
        self._phone = phone
        # The canceller the first stream is opened with; each stream after it is opened with its own.
        self._echo = echo
        self._record = record
        self._listening: asyncio.Task[None] | None = None
        # What the attached stream was opened on; None until setup opens the first.
        self.opened: Input | None = None

    async def setup(self, setup: FrameProcessorSetup) -> None:
        # As for the speaker: the base's setup, then the one way the stream is opened.
        await BaseInputTransport.setup(self, setup)
        py_audio = cast(PortAudio, self._py_audio)
        self.attach(py_audio, self.open_stream(py_audio, self._echo))

    def open_stream(self, py_audio: PortAudio, echo: Echo) -> Input:
        """A new stream on the default input, not yet the microphone's. Blocking: PortAudio talks to the device."""
        match default_input(py_audio):
            case None:
                return Input(NoInput(), None, echo)
            case device:
                return Input(self._open(py_audio, echo), device, echo)

    def _open(self, py_audio: PortAudio, echo: Echo) -> Stream:
        return py_audio.open(
            format=py_audio.get_format_from_width(2),
            channels=self._params.audio_in_channels,
            rate=self._sample_rate,
            frames_per_buffer=int(self._sample_rate / 100) * 2,  # 20 ms, as Pipecat opens it
            # [LAW:single-enforcer] a stream's buffers are heard through its own canceller alone: one still calling
            # back after a reopen has attached the next never touches the new one from a second thread.
            stream_callback=partial(self._captured, echo),
            input=True,
            input_device_index=self._params.input_device_index,
        )

    def attach(self, py_audio: PortAudio, opened: Input) -> None:
        self._py_audio = cast(pyaudio.PyAudio, py_audio)
        self.opened = opened
        self._in_stream = opened.stream

    def detach(self) -> Input:
        opened = self._held()
        assert opened is not None, "the microphone stream opens in setup"
        self._in_stream = None
        return opened

    def _held(self) -> Input | None:
        # Pipecat's start reads `_in_stream`, so it is what says a stream is attached; `opened` outlives a detach.
        return None if self._in_stream is None else self.opened

    async def let_go(self, opened: Input) -> None:
        """Stop and close a detached stream, then its canceller. Measured at 2.7 to 3.7 s when its device is gone."""

        def stop_and_close() -> None:
            _letting_go(opened.stream.stop_stream)
            opened.stream.close()
            # [LAW:no-ambient-temporal-coupling] on the thread that closed the stream, once it can call back no more:
            # a stream that never finishes closing keeps its canceller.
            opened.echo.close()

        # [LAW:nothing-unseen] one event for each stream let go of, carrying what its canceller did over its life.
        with unit("microphone.let_go", self._record, COUNTS):
            annotate(device=opened.device)
            try:
                await off_loop(stop_and_close, "closing the microphone")
            finally:
                count(**opened.echo.counts())

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        self._listening = self.create_task(self._hear_the_phone(), "the phone's microphone")  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)

    async def _hear_the_phone(self) -> None:
        while True:
            # Keyed by the phone as each arrived, in order with its button.
            await self.push_audio_frame(await self._phone.heard.get())

    async def cleanup(self) -> None:
        if self._listening is not None:
            await self.cancel_task(self._listening)  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        # As for the speaker: off the loop, where closing a microphone whose device is gone takes seconds.
        await BaseInputTransport.cleanup(self)
        await _let_go_at_cleanup(self._held(), self.let_go, lambda: None)

    def _captured(self, echo: Echo, in_data: bytes, frame_count: int, time_info: object, status: int) -> tuple[None, int]:
        # [LAW:no-ambient-temporal-coupling] decided on the audio thread, not when the event loop gets to the frame.
        # [LAW:dataflow-not-control-flow] every buffer goes through the canceller, key up or down and at either place:
        # it keeps learning the room, and takes the speaker's sound in step with it.
        try:
            cleaned = echo.heard(in_data, self._sample_rate, self._params.audio_in_channels)
        except Exception as error:
            # [LAW:no-silent-failure] PyAudio would abort the stream on a raise and say so only on stderr, leaving
            # hands running deaf; the pipeline is ended instead, and the run with it.
            asyncio.run_coroutine_threadsafe(self._fail(error), self.get_event_loop())
            return None, pyaudio.paAbort
        # One read of the gate: what the frame holds and the key it says it was captured under always agree.
        gate = self._key.gate
        # [LAW:single-enforcer] the gate alone says which place's microphone reaches Whisper; while hands is at the
        # phone, the desk's frames are not the pipeline's at all, so its stream is the phone's unbroken.
        if not gate.hears("desk"):
            return None, pyaudio.paContinue
        frame = KeyedAudio(
            audio=gate.audible(cleaned),
            sample_rate=self._sample_rate,
            num_channels=self._params.audio_in_channels,
            key=gate.key,
        )
        asyncio.run_coroutine_threadsafe(self.push_audio_frame(frame), self.get_event_loop())
        return None, pyaudio.paContinue

    async def _fail(self, error: Exception) -> None:
        await self.push_error("the microphone stopped: its capture failed", exception=error, force_treat_as_permanent=True)  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        await self.push_frame(EndWorkerFrame(reason="the microphone's capture failed"), FrameDirection.UPSTREAM)


@dataclass(frozen=True)
class Devices:
    """The devices the transport is on, by the names the system gives them."""

    input: str | None  # None when the system has no input device at all, so hands cannot hear
    output: str


class KeyedAudioTransport(LocalAudioTransport):
    """`LocalAudioTransport` with the keyed microphone and the speaker, joined by the echo canceller of the streams open."""

    def __init__(
        self,
        params: LocalAudioTransportParams,
        key: PushToTalk,
        phone: Phone,
        record: Record,
        clock: Callable[[], Instant] = time.monotonic,
        portaudio: Callable[[], PortAudio] = lambda: cast(PortAudio, pyaudio.PyAudio()),
        defaults: Callable[[], DefaultDevices] = default_devices,
        echo: Callable[[], Echo] = EchoCanceller,
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
        # [LAW:one-source-of-truth] one canceller for the streams opened together: what the speaker plays is what it
        # takes out of what the microphone hears. A reopen opens the next pair with a new one, which learns the new
        # devices' room from nothing and holds none of the old one's sound.
        cancelling = echo()
        self._echo = echo
        self._speaker = Speaker(self._pyaudio, params, key, phone, cancelling, clock)
        self._microphone = KeyedMicrophone(self._pyaudio, params, key, phone, cancelling, record)
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
        """The devices the streams are open on, as each was named when it was opened."""
        match (self._microphone.opened, self._speaker.opened):
            case (Input(device=heard), Output(device=played)):
                return Devices(heard, played)
            case _:
                raise AssertionError("the devices are known once the pipeline has set up its streams")

    @property
    def deaf(self) -> bool:
        """Whether the microphone is open on no device, so a press to talk hears nothing; never before setup opens it."""
        # [LAW:one-source-of-truth] the device the spoken warning names, read where `devices` reads it.
        match self._microphone.opened:
            case Input(device=None):
                return True
            case _:
                return False

    def _audio_now(self) -> PortAudio:
        return cast(PortAudio, self._pyaudio)

    def _start_again(self, ending: PortAudio) -> "_Started":
        ending.terminate()
        defaults = self._defaults()
        portaudio = self._portaudio()
        cancelling = self._echo()
        try:
            speaker, microphone = self._speaker.open_stream(portaudio, cancelling), self._microphone.open_stream(portaudio, cancelling)
            speaker.stream.start_stream()
            microphone.stream.start_stream()
        except BaseException:
            # No stream was started on it, so nothing calls back into it, and no let_go will reach it: it goes here.
            cancelling.close()
            raise
        return _Started(defaults, portaudio, speaker, microphone)


@dataclass(frozen=True)
class _Started:
    """PortAudio started again, with both streams open and running on it, not yet the sides'."""

    defaults: DefaultDevices
    portaudio: PortAudio
    speaker: Output
    microphone: Input


def _letting_go(step: Callable[[], None]) -> None:
    try:
        step()
    except OSError as error:
        # A stream on a device that is gone refuses to stop cleanly; it is let go of all the same, and said.
        logger.warning(f"letting go of an audio stream whose device is gone: {error}")


async def _let_go_at_cleanup[S](held: S | None, let_go: Callable[[S], Awaitable[None]], detached: Callable[[], None]) -> None:
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
    except OSError as error:
        # Asked only of a PortAudio that is running: an ended one says the same words whatever is plugged in.
        if error.args != (_NO_DEFAULT_INPUT,):
            raise
        return None
    return _name(info)


# PyAudio's words when PortAudio lists no default input, raised with no errno to tell it by.
_NO_DEFAULT_INPUT = "No Default Input Device Available"


def _name(info: object) -> str:
    # PyAudio's device info is an untyped mapping; the name is the one field read from it.
    match info:
        case {"name": str(name)}:
            return name
        case _:
            raise AssertionError(f"PortAudio described a device without a name: {info!r}")

