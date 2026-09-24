"""What the keyed microphone lets the pipeline hear, decided at capture, with no audio device."""

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any, cast

import pyaudio
import pytest
from pipecat.clocks.system_clock import SystemClock
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager

from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
from pipecat.transports.local.audio import LocalAudioInputTransport, LocalAudioOutputTransport, LocalAudioTransportParams

from hands.voice.coreaudio import DefaultDevices
from hands.voice.microphone import ECHO_PATH_SECS, Devices, KeyedAudioTransport, NoInput, PortAudio, buffer_age, heard
from hands.voice.ptt import Gate, PushToTalk

LOUD = b"\x7f\x7f" * 320
QUIET = bytes(len(LOUD))
DOWN = Gate(key="down")
CHUNK = len(LOUD) / (2 * 16000)  # how long LOUD plays at 16 kHz mono


def test_only_a_held_key_after_the_speaker_has_gone_quiet_is_heard() -> None:
    assert heard(LOUD, DOWN, captured=2.0, speaker_quiet_at=1.0) == LOUD
    assert heard(LOUD, DOWN, captured=0.9, speaker_quiet_at=1.0) == QUIET
    assert heard(LOUD, Gate(), captured=2.0, speaker_quiet_at=1.0) == QUIET


class Rig:
    """The transport with its PyAudio calls replaced: the clock is set by hand and pushed audio is kept."""

    def __init__(self) -> None:
        self.now = 0.0
        self.key = PushToTalk()
        self.pushed: list[bytes] = []
        params = LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
        self.transport = KeyedAudioTransport(params, self.key, clock=lambda: self.now)
        self.speaker = self.transport.output()
        self.microphone = self.transport.input()
        self.microphone._sample_rate = 16000  # pyright: ignore[reportPrivateUsage]
        self.stream = SimpleStream()
        self.speaker.attach(cast(PortAudio, SimpleNamespace()), self.stream)  # as setup attaches the stream it opened

        async def push_audio_frame(frame: InputAudioRawFrame) -> None:
            self.pushed.append(frame.audio)

        self.microphone.push_audio_frame = push_audio_frame
        self.microphone.get_event_loop = asyncio.get_running_loop
        self.speaker.get_event_loop = asyncio.get_running_loop

    async def play(self, audio: bytes, at: float) -> None:
        self.now = at
        await self.speaker.write_audio_frame(OutputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1))

    async def capture(self, at: float, age: float = 0.0) -> None:
        self.now = at
        times = {"current_time": 100.0, "input_buffer_adc_time": 100.0 - age}
        self.microphone._audio_in_callback(LOUD, 320, times, 0)  # pyright: ignore[reportPrivateUsage]
        await asyncio.sleep(0.01)


class SimpleStream:
    def __init__(self) -> None:
        self.blocking = False
        self.written: list[bytes] = []

    def write(self, audio: bytes) -> None:
        # A blocking write returns when the device has taken the chunk, about when it has played.
        while self.blocking:
            time.sleep(0.001)
        self.written.append(audio)

    def get_output_latency(self) -> float:
        return 0.0

    def start_stream(self) -> None: ...
    def stop_stream(self) -> None: ...
    def close(self) -> None: ...


async def test_the_reply_already_given_to_the_speaker_is_not_heard_after_a_press() -> None:
    devices = Rig()
    await devices.play(LOUD, at=1.0)
    await devices.capture(at=1.0)  # key up
    devices.key.move_key("down")
    await devices.capture(at=1.0 + CHUNK + ECHO_PATH_SECS - 0.01)  # the reply still in the room
    await devices.capture(at=1.0 + CHUNK + ECHO_PATH_SECS)  # gone
    assert devices.pushed == [QUIET, QUIET, LOUD]


async def test_silence_written_to_the_speaker_keeps_nothing_shut() -> None:
    devices = Rig()
    devices.key.move_key("down")
    await devices.play(QUIET, at=1.0)
    await devices.capture(at=1.01)
    assert devices.pushed == [LOUD]


async def test_a_late_callback_is_judged_by_when_its_sound_was_recorded() -> None:
    devices = Rig()
    devices.key.move_key("down")
    await devices.play(LOUD, at=1.0)
    # Delivered after the speaker went quiet, but recorded 50 ms before: the reply's tail.
    await devices.capture(at=1.0 + CHUNK + ECHO_PATH_SECS + 0.01, age=0.05)
    assert devices.pushed == [QUIET]


def test_a_buffer_the_host_cannot_date_is_as_old_as_its_callback() -> None:
    assert buffer_age({"current_time": 10.0, "input_buffer_adc_time": 9.973}) == 10.0 - 9.973
    assert buffer_age({"current_time": 10.0, "input_buffer_adc_time": 0.0}) == 0.0
    assert buffer_age(None) == 0.0


async def test_a_chunk_whose_write_an_interruption_cancels_still_holds_the_microphone_shut() -> None:
    devices = Rig()
    devices.key.move_key("down")
    devices.stream.blocking = True
    devices.now = 1.0
    writing = asyncio.create_task(devices.speaker.write_audio_frame(OutputAudioRawFrame(audio=LOUD, sample_rate=16000, num_channels=1)))
    await asyncio.sleep(0.01)
    writing.cancel()  # the barge-in; PortAudio's thread plays the chunk out regardless
    devices.stream.blocking = False
    await devices.capture(at=1.0 + CHUNK + ECHO_PATH_SECS - 0.01)
    assert devices.pushed == [QUIET]


class LostStream:
    """A stream on a device that is gone, as PortAudio leaves it: a write blocks until the stream is stopped, and stopping fails."""

    def __init__(self, log: list[str], name: str) -> None:
        self.log = log
        self.name = name
        self.stopped = threading.Event()

    def write(self, audio: bytes) -> None:
        self.stopped.wait()
        self.log.append(f"write returned from {self.name}")
        raise OSError(-9986, "Internal PortAudio error")

    def start_stream(self) -> None:
        self.log.append(f"start {self.name}")

    def get_output_latency(self) -> float:
        return 0.2  # the new speaker's buffer, longer than the one it replaced

    def stop_stream(self) -> None:
        self.log.append(f"stop {self.name}")
        self.stopped.set()
        raise OSError(-9986, "Internal PortAudio error")

    def close(self) -> None:
        self.log.append(f"close {self.name}")


class FreshPortAudio:
    """PortAudio started again: it opens streams on the defaults it lists now."""

    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.opened: list[dict[str, object]] = []
        log.append("start portaudio")

    def open(self, **settings: object) -> LostStream:
        side = "microphone" if settings.get("input") else "speaker"
        self.log.append(f"open {side}")
        self.opened.append(settings)
        return LostStream(self.log, f"new {side}")

    def get_format_from_width(self, width: int) -> int:
        return 8

    def get_default_input_device_info(self) -> object:
        return {"name": "MacBook Pro Microphone"}

    def get_default_output_device_info(self) -> object:
        return {"name": "MacBook Pro Speakers"}

    def terminate(self) -> None:
        self.log.append("end portaudio")


def lost_transport(log: list[str]) -> KeyedAudioTransport:
    params = LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    defaults = iter([DefaultDevices(input=1, output=1), DefaultDevices(input=2, output=2)])
    transport = KeyedAudioTransport(params, PushToTalk(), portaudio=lambda: FreshPortAudio(log), defaults=lambda: next(defaults))
    speaker, microphone = transport.output(), transport.input()
    speaker.get_event_loop = asyncio.get_running_loop
    microphone._sample_rate = 16000  # pyright: ignore[reportPrivateUsage]
    speaker._sample_rate = 24000  # pyright: ignore[reportPrivateUsage]
    speaker.attach(cast(PortAudio, SimpleNamespace()), LostStream(log, "old speaker"))
    microphone.attach(cast(PortAudio, SimpleNamespace()), LostStream(log, "old microphone"))
    log.clear()  # the first PortAudio, started by the same factory as every later one
    return transport


async def test_reopening_lets_go_of_the_lost_devices_and_opens_on_the_defaults_as_they_are_now() -> None:
    log: list[str] = []
    transport = lost_transport(log)
    speaker = transport.output()
    frame = OutputAudioRawFrame(audio=LOUD, sample_rate=24000, num_channels=1)
    stuck = asyncio.create_task(speaker.write_audio_frame(frame))  # the reply playing when the headset went
    await asyncio.sleep(0.01)
    await speaker.set_usable(False)  # as Pipecat's write timeout leaves it

    devices = await transport.reopen()

    assert devices == Devices(input="MacBook Pro Microphone", output="MacBook Pro Speakers")
    assert log == [
        "stop old speaker", "write returned from old speaker", "close old speaker",  # closed only once no write is inside it
        "stop old microphone", "close old microphone", "end portaudio",
        "start portaudio", "open speaker", "open microphone", "start new speaker", "start new microphone",
    ]  # fmt: skip
    assert transport.opened_on == DefaultDevices(input=2, output=2)  # read again, as PortAudio listed them anew
    with pytest.raises(OSError):
        await stuck
    assert speaker.is_usable
    assert speaker._fade == 0.2 + ECHO_PATH_SECS  # pyright: ignore[reportPrivateUsage]  (read from the new stream)


async def test_a_frame_given_while_the_transport_reopens_waits_and_plays_on_the_new_stream() -> None:
    rig = Rig()
    rig.speaker.detach()
    rig.now = 1.0
    writing = asyncio.create_task(rig.speaker.write_audio_frame(OutputAudioRawFrame(audio=LOUD, sample_rate=16000, num_channels=1)))
    await asyncio.sleep(0.01)
    assert not writing.done()
    assert (rig.speaker.quiet_at, rig.speaker.sounded_at) == (0.0, None)  # nothing has sounded yet
    reopened = SimpleStream()
    rig.speaker.attach(cast(PortAudio, SimpleNamespace()), reopened)
    assert await writing is True
    assert reopened.written == [LOUD]
    assert rig.speaker.sounded_at == 1.0


async def test_the_streams_are_opened_as_pipecat_opens_them() -> None:
    """The sides open their own streams so they can open them again; this is what says a Pipecat upgrade changed how."""
    ours, pipecats = FreshPortAudio([]), FreshPortAudio([])
    params = LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    setup = FrameProcessorSetup(clock=SystemClock(), task_manager=TaskManager(), pipeline_worker=cast(Any, None), audio_in_sample_rate=16000, audio_out_sample_rate=24000)
    for portaudio, output, input_ in (
        (ours, KeyedAudioTransport(params, PushToTalk()).output(), KeyedAudioTransport(params, PushToTalk()).input()),
        (pipecats, LocalAudioOutputTransport(cast(pyaudio.PyAudio, pipecats), params), LocalAudioInputTransport(cast(pyaudio.PyAudio, pipecats), params)),
    ):
        setattr(output, "_py_audio", portaudio)
        setattr(input_, "_py_audio", portaudio)
        await output.setup(setup)
        await input_.setup(setup)
    assert len(ours.opened) == 2
    assert [{k: v for k, v in opened.items() if k != "stream_callback"} for opened in ours.opened] == [
        {k: v for k, v in opened.items() if k != "stream_callback"} for opened in pipecats.opened
    ]


class DeafPortAudio(FreshPortAudio):
    """PortAudio on a Mac whose last input device is gone: it lists no default input."""

    def get_default_input_device_info(self) -> object:
        raise OSError(-9996, "No Default Input Device Available")


async def test_with_no_microphone_left_the_transport_reopens_to_speak_and_says_it_cannot_hear() -> None:
    log: list[str] = []
    transport = lost_transport(log)
    setattr(transport, "_portaudio", lambda: DeafPortAudio(log))

    devices = await transport.reopen()

    assert devices == Devices(input=None, output="MacBook Pro Speakers")
    assert "open microphone" not in log  # nothing to open: the microphone holds a stream of nothing
    assert "start new speaker" in log
    assert isinstance(transport.input()._in_stream, NoInput)  # pyright: ignore[reportPrivateUsage]

    # A microphone plugged in again is opened by the next reopen, the same way as any other.
    setattr(transport, "_portaudio", lambda: FreshPortAudio(log))
    setattr(transport, "_defaults", lambda: DefaultDevices(input=3, output=2))
    log.clear()
    assert (await transport.reopen()).input == "MacBook Pro Microphone"
    assert "open microphone" in log


async def test_a_daemon_started_with_no_microphone_runs_rather_than_failing_its_setup() -> None:
    params = LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    transport = KeyedAudioTransport(params, PushToTalk(), portaudio=lambda: DeafPortAudio([]), defaults=lambda: DefaultDevices(0, 1))
    setup = FrameProcessorSetup(clock=SystemClock(), task_manager=TaskManager(), pipeline_worker=cast(Any, None), audio_in_sample_rate=16000, audio_out_sample_rate=24000)
    await transport.input().setup(setup)
    assert isinstance(transport.input()._in_stream, NoInput)  # pyright: ignore[reportPrivateUsage]
    assert transport.devices == Devices(input=None, output="MacBook Pro Speakers")
    await transport.input().cleanup()  # and lets go of nothing, without complaint
