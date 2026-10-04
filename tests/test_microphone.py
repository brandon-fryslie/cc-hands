"""What the keyed microphone lets the pipeline hear, decided at capture, with no audio device."""

import asyncio
import itertools
import threading
import time
from types import SimpleNamespace
from typing import Any, cast

import pyaudio
import pytest
from pipecat.clocks.system_clock import SystemClock
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager

from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, EndWorkerFrame, Frame, InputAudioRawFrame, OutputAudioRawFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.local.audio import LocalAudioInputTransport, LocalAudioOutputTransport, LocalAudioTransportParams

from hands.sessions.audit import Entry, Record
from hands.sessions.wide import WideEvent
from hands.voice.cues import OPENED, sound
from hands.voice.coreaudio import DefaultDevices
from hands.voice.microphone import Devices, Input, KeyedAudioTransport, NoInput, Output, PortAudio, default_input
from hands.voice.phone import Phone
from hands.voice.ptt import PushToTalk

LOUD = b"\x7f\x7f" * 320
QUIET = bytes(len(LOUD))
CLEANED = b"\x01\x00" * 320  # what the room's canceller makes of anything the microphone hears


def _phone() -> Phone:
    return Phone(PushToTalk(lambda _: None), heard_rate=16000, played_rate=16000, record=lambda _: None)


class Room:
    """An echo canceller that keeps what the speaker played and cleans every microphone buffer to CLEANED."""

    def __init__(self, log: list[str] | None = None, name: str = "canceller") -> None:
        self.plays: list[tuple[bytes, int, int]] = []
        self.hears: list[bytes] = []
        self.log = [] if log is None else log
        self.name = name

    def played(self, audio: bytes, sample_rate: int, channels: int) -> None:
        self.plays.append((audio, sample_rate, channels))

    def heard(self, audio: bytes, sample_rate: int, channels: int) -> bytes:
        self.hears.append(audio)
        return CLEANED

    def counts(self) -> dict[str, int]:
        return {"heard": len(self.hears), "unplayed": 0, "dropped": 0}

    def close(self) -> None:
        self.log.append(f"close {self.name}")


class Rig:
    """The transport with its PyAudio calls replaced: the clock is set by hand and pushed audio is kept."""

    def __init__(self) -> None:
        self.now = 0.0
        self.key = PushToTalk(lambda _: None)
        self.phone = Phone(self.key, heard_rate=16000, played_rate=16000, record=lambda _: None)
        self.pushed: list[bytes] = []
        self.room = Room()
        params = LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
        self.transport = KeyedAudioTransport(params, self.key, self.phone, lambda _: None, clock=lambda: self.now, echo=lambda: self.room)
        self.speaker = self.transport.output()
        self.microphone = self.transport.input()
        self.microphone._sample_rate = 16000  # pyright: ignore[reportPrivateUsage]
        self.stream = SimpleStream()
        self.speaker.attach(cast(PortAudio, SimpleNamespace()), Output(self.stream, "MacBook Pro Speakers", self.room))  # as setup attaches the stream it opened

        async def push_audio_frame(frame: InputAudioRawFrame) -> None:
            self.pushed.append(frame.audio)

        self.microphone.push_audio_frame = push_audio_frame
        self.microphone.get_event_loop = asyncio.get_running_loop
        self.speaker.get_event_loop = asyncio.get_running_loop

    async def play(self, audio: bytes, at: float) -> None:
        self.now = at
        await self.speaker.write_audio_frame(OutputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1))

    async def capture(self, at: float) -> None:
        self.now = at
        self.microphone._captured(self.room, LOUD, 320, {}, 0)  # pyright: ignore[reportPrivateUsage]
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


async def test_the_microphone_is_heard_through_the_canceller_and_only_while_the_key_is_down() -> None:
    devices = Rig()
    await devices.capture(at=1.0)  # key up
    devices.key.move("start", "held key")
    await devices.capture(at=1.02)
    assert devices.pushed == [QUIET, CLEANED]
    assert devices.room.hears == [LOUD, LOUD]  # key up too: the canceller learns the room from every buffer


async def test_the_canceller_hears_the_desk_microphone_while_hands_is_at_the_phone() -> None:
    devices = Rig()
    devices.key.go("phone")
    await devices.capture(at=1.0)
    assert devices.pushed == []  # the phone's microphone is the pipeline's
    assert devices.room.hears == [LOUD]  # and the canceller takes the speaker's sound in step all the same


class Refusing(Room):
    def heard(self, audio: bytes, sample_rate: int, channels: int) -> bytes:
        raise RuntimeError("the audio processing module refused the frame")


async def test_a_capture_the_canceller_fails_ends_the_pipeline_rather_than_leaving_hands_deaf() -> None:
    devices = Rig()
    errors: list[str] = []
    pushed: list[tuple[Frame, FrameDirection]] = []

    async def push_error(error_msg: str, exception: Exception | None = None, fatal: bool = False, category: object = None, force_treat_as_permanent: bool = False) -> None:
        errors.append(f"{error_msg}: {exception}")

    async def push_frame(frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        pushed.append((frame, direction))

    setattr(devices.microphone, "push_error", push_error)
    setattr(devices.microphone, "push_frame", push_frame)
    assert devices.microphone._captured(Refusing(), LOUD, 320, {}, 0) == (None, pyaudio.paAbort)  # pyright: ignore[reportPrivateUsage]
    await asyncio.sleep(0.01)
    assert errors == ["the microphone stopped: its capture failed: the audio processing module refused the frame"]
    assert [(type(frame), direction) for frame, direction in pushed] == [(EndWorkerFrame, FrameDirection.UPSTREAM)]
    assert devices.pushed == []


async def test_a_press_while_the_reply_plays_is_heard_at_once() -> None:
    devices = Rig()
    await devices.play(LOUD, at=1.0)
    devices.key.move("start", "held key")
    await devices.capture(at=1.0)  # the reply still in the room, and a word over it
    assert devices.pushed == [CLEANED]
    assert devices.room.plays == [(LOUD, 16000, 1)]


async def test_silence_given_to_the_speaker_is_the_cancellers_reference_too() -> None:
    devices = Rig()
    await devices.play(QUIET, at=1.0)
    assert devices.room.plays == [(QUIET, 16000, 1)]
    assert devices.speaker.sounded_at is None  # silence padding is no sound for the heartbeat


async def test_a_chunk_whose_write_an_interruption_cancels_is_still_given_to_the_canceller() -> None:
    devices = Rig()
    devices.stream.blocking = True
    devices.now = 1.0
    writing = asyncio.create_task(devices.speaker.write_audio_frame(OutputAudioRawFrame(audio=LOUD, sample_rate=16000, num_channels=1)))
    await asyncio.sleep(0.01)
    writing.cancel()  # the barge-in; PortAudio's thread plays the chunk out regardless
    devices.stream.blocking = False
    await devices.speaker._writes.run(lambda: None)  # pyright: ignore[reportPrivateUsage]
    assert devices.room.plays == [(LOUD, 16000, 1)]
    assert devices.stream.written == [LOUD]


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


def lost_transport(log: list[str], events: list[Entry] | None = None) -> KeyedAudioTransport:
    params = LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    defaults = iter([DefaultDevices(input=1, output=1), DefaultDevices(input=2, output=2)])
    cancellers = iter(Room(log, f"canceller {n}") for n in itertools.count(1))
    record: Record = (lambda _: None) if events is None else events.append
    transport = KeyedAudioTransport(
        params, PushToTalk(lambda _: None), _phone(), record, portaudio=lambda: FreshPortAudio(log), defaults=lambda: next(defaults), echo=lambda: next(cancellers)
    )
    speaker, microphone = transport.output(), transport.input()
    speaker.get_event_loop = asyncio.get_running_loop
    microphone._sample_rate = 16000  # pyright: ignore[reportPrivateUsage]
    speaker._sample_rate = 24000  # pyright: ignore[reportPrivateUsage]
    old = Room(log, "old canceller")
    speaker.attach(cast(PortAudio, SimpleNamespace()), Output(LostStream(log, "old speaker"), "headset", old))
    microphone.attach(cast(PortAudio, SimpleNamespace()), Input(LostStream(log, "old microphone"), "headset", old))
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
        "stop old microphone", "close old microphone", "close old canceller", "end portaudio",
        "start portaudio", "open speaker", "open microphone", "start new speaker", "start new microphone",
    ]  # fmt: skip
    assert transport.opened_on == DefaultDevices(input=2, output=2)  # read again, as PortAudio listed them anew
    with pytest.raises(OSError):
        await stuck
    assert speaker.is_usable


async def test_a_reopen_opens_both_streams_on_one_new_canceller_and_reports_the_old_ones_life() -> None:
    log: list[str] = []
    events: list[Entry] = []
    transport = lost_transport(log, events)
    old = transport.input().opened
    assert old is not None
    old.echo.heard(LOUD, 16000, 1)  # a buffer heard on the old devices

    await transport.reopen()

    speaker, microphone = transport.output().opened, transport.input().opened
    assert speaker is not None and microphone is not None
    assert speaker.echo is microphone.echo is not old.echo
    assert "close canceller 2" not in log
    [event] = events
    assert isinstance(event, WideEvent)
    assert (event.event, event.outcome, event.facts, dict(event.counts)) == ("microphone.let_go", "ok", {"device": "headset"}, {"heard": 1, "unplayed": 0, "dropped": 0})


class RefusingPortAudio(FreshPortAudio):
    """PortAudio started again on a device that refuses to open."""

    def open(self, **settings: object) -> LostStream:
        raise OSError(-9996, "Invalid input device")


async def test_a_reopen_whose_streams_fail_to_open_lets_go_of_the_canceller_it_made_for_them() -> None:
    log: list[str] = []
    transport = lost_transport(log)
    setattr(transport, "_portaudio", lambda: RefusingPortAudio(log))
    with pytest.raises(OSError, match="Invalid input device"):
        await transport.reopen()
    assert "close canceller 2" in log  # the run fails, and exits without LiveKit's assertion over a canceller left open


async def test_a_frame_given_while_the_transport_reopens_waits_and_plays_on_the_new_stream() -> None:
    rig = Rig()
    rig.speaker.detach()
    rig.now = 1.0
    writing = asyncio.create_task(rig.speaker.write_audio_frame(OutputAudioRawFrame(audio=LOUD, sample_rate=16000, num_channels=1)))
    await asyncio.sleep(0.01)
    assert not writing.done()
    assert rig.speaker.sounded_at is None  # nothing has sounded yet
    reopened = SimpleStream()
    rig.speaker.attach(cast(PortAudio, SimpleNamespace()), Output(reopened, "AirPods", rig.room))
    assert await writing is True
    assert reopened.written == [LOUD]
    assert rig.speaker.sounded_at == 1.0


async def test_the_streams_are_opened_as_pipecat_opens_them() -> None:
    """The sides open their own streams so they can open them again; this is what says a Pipecat upgrade changed how."""
    ours, pipecats = FreshPortAudio([]), FreshPortAudio([])
    params = LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    setup = FrameProcessorSetup(clock=SystemClock(), task_manager=TaskManager(), pipeline_worker=cast(Any, None), audio_in_sample_rate=16000, audio_out_sample_rate=24000)
    for portaudio, output, input_ in (
        (ours, KeyedAudioTransport(params, PushToTalk(lambda _: None), _phone(), lambda _: None).output(), KeyedAudioTransport(params, PushToTalk(lambda _: None), _phone(), lambda _: None).input()),
        (pipecats, LocalAudioOutputTransport(cast(pyaudio.PyAudio, pipecats), params), LocalAudioInputTransport(cast(pyaudio.PyAudio, pipecats), params)),
    ):
        setattr(output, "_py_audio", portaudio)
        setattr(input_, "_py_audio", portaudio)
        await output.setup(setup)
        await input_.setup(setup)
    assert len(ours.opened) == 2
    # The one difference is chosen: the speaker's period, which Pipecat leaves to the device's low-latency default.
    speaker, microphone = ours.opened
    assert speaker.pop("frames_per_buffer") == 480
    assert [{k: v for k, v in opened.items() if k != "stream_callback"} for opened in (speaker, microphone)] == [
        {k: v for k, v in opened.items() if k != "stream_callback"} for opened in pipecats.opened
    ]


class DeafPortAudio(FreshPortAudio):
    """PortAudio on a Mac whose last input device is gone: it lists no default input."""

    def get_default_input_device_info(self) -> object:
        raise OSError("No Default Input Device Available")  # as PyAudio raises it: the words alone, no errno


async def test_with_no_microphone_left_the_transport_reopens_to_speak_and_says_it_cannot_hear() -> None:
    log: list[str] = []
    transport = lost_transport(log)
    setattr(transport, "_portaudio", lambda: DeafPortAudio(log))
    assert not transport.deaf

    devices = await transport.reopen()

    assert devices == Devices(input=None, output="MacBook Pro Speakers")
    assert transport.deaf  # what the heartbeat carries to the menu bar
    assert "open microphone" not in log  # nothing to open: the microphone holds a stream of nothing
    assert "start new speaker" in log
    assert isinstance(transport.input()._in_stream, NoInput)  # pyright: ignore[reportPrivateUsage]

    # A microphone plugged in again is opened by the next reopen, the same way as any other.
    setattr(transport, "_portaudio", lambda: FreshPortAudio(log))
    setattr(transport, "_defaults", lambda: DefaultDevices(input=3, output=2))
    log.clear()
    assert (await transport.reopen()).input == "MacBook Pro Microphone"
    assert "open microphone" in log
    assert not transport.deaf


async def test_a_daemon_started_with_no_microphone_runs_rather_than_failing_its_setup() -> None:
    params = LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    transport = KeyedAudioTransport(params, PushToTalk(lambda _: None), _phone(), lambda _: None, portaudio=lambda: DeafPortAudio([]), defaults=lambda: DefaultDevices(0, 1), echo=Room)
    assert not transport.deaf  # nothing is said of hearing before setup opens the microphone
    setup = FrameProcessorSetup(clock=SystemClock(), task_manager=TaskManager(), pipeline_worker=cast(Any, None), audio_in_sample_rate=16000, audio_out_sample_rate=24000)
    await transport.input().setup(setup)
    assert isinstance(transport.input()._in_stream, NoInput)  # pyright: ignore[reportPrivateUsage]
    opened = transport.input().opened
    assert opened is not None and opened.device is None and isinstance(opened.stream, NoInput)
    assert transport.deaf
    await transport.input().cleanup()  # and lets go of nothing, without complaint



def test_only_portaudios_own_no_default_input_reads_as_no_microphone() -> None:
    assert default_input(cast(PortAudio, DeafPortAudio([]))) is None
    refusing = SimpleNamespace(get_default_input_device_info=lambda: (_ for _ in ()).throw(OSError(-9999, "Unanticipated host error")))
    with pytest.raises(OSError, match="host error"):
        default_input(cast(PortAudio, refusing))


async def test_a_turns_cue_is_played_at_once_and_the_canceller_hears_it() -> None:
    devices = Rig()
    devices.speaker._sample_rate = 16000  # pyright: ignore[reportPrivateUsage]  # as setup sets it
    devices.key.move("start", "held key")
    devices.now = 1.0
    devices.speaker.cue(OPENED)
    await devices.capture(at=1.0)  # a word said over the cue
    await devices.speaker._writes.run(lambda: None)  # pyright: ignore[reportPrivateUsage]
    assert devices.stream.written == [sound(OPENED, 16000, 1)]
    assert devices.room.plays == [(sound(OPENED, 16000, 1), 16000, 1)]
    assert devices.pushed == [CLEANED]  # the word said over it, with the tone taken out
    assert devices.speaker.sounded_at == 1.0


async def test_a_cue_never_holds_the_talk_key_on_the_speaker() -> None:
    devices = Rig()
    devices.speaker._sample_rate = 16000  # pyright: ignore[reportPrivateUsage]
    devices.stream.blocking = True  # a write stuck on a device that is going
    assert devices.speaker.cue(OPENED) == "desk"
    devices.speaker.detach()  # a reopen under way
    assert devices.speaker.cue(OPENED) == "unattached"  # said, so a cue for silence is never recorded as heard
    devices.stream.blocking = False
    await devices.speaker._writes.run(lambda: None)  # pyright: ignore[reportPrivateUsage]
    assert devices.stream.written == [sound(OPENED, 16000, 1)]


async def test_a_tone_that_fails_to_play_loses_only_the_tone() -> None:
    devices = Rig()
    devices.speaker._sample_rate = 16000  # pyright: ignore[reportPrivateUsage]
    written = devices.stream.write

    def stopped(audio: bytes) -> None:
        raise OSError("Stream is stopped")

    devices.stream.write = stopped
    devices.speaker.cue(OPENED)
    await devices.speaker._writes.run(lambda: None)  # pyright: ignore[reportPrivateUsage]
    devices.stream.write = written
    await devices.play(LOUD, at=1.0)
    assert devices.stream.written == [LOUD]


async def test_the_canceller_hears_a_chunk_queued_behind_a_cue_after_the_cue() -> None:
    devices = Rig()
    devices.speaker._sample_rate = 16000  # pyright: ignore[reportPrivateUsage]
    devices.stream.blocking = True
    devices.speaker.cue(OPENED)  # the barge-in's tone, still going to the device
    writing = asyncio.create_task(devices.speaker.write_audio_frame(OutputAudioRawFrame(audio=LOUD, sample_rate=16000, num_channels=1)))
    await asyncio.sleep(0.01)
    devices.stream.blocking = False
    await writing
    assert [audio for audio, _, _ in devices.room.plays] == [sound(OPENED, 16000, 1), LOUD]  # in the order the room hears them


async def test_the_speaker_is_quiet_while_neither_hands_nor_the_user_is_speaking(monkeypatch: pytest.MonkeyPatch) -> None:
    devices = Rig()
    pushed: list[Frame] = []

    async def passed(_self: object, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        pushed.append(frame)

    # Pipecat's sender pushes each edge of hands' speech through the transport, downstream and up.
    monkeypatch.setattr(LocalAudioOutputTransport, "push_frame", passed)
    assert devices.speaker.quiet.is_set()
    await devices.speaker.push_frame(BotStartedSpeakingFrame())
    assert not devices.speaker.quiet.is_set()
    await devices.speaker.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
    await devices.speaker.push_frame(BotStoppedSpeakingFrame())
    assert devices.speaker.quiet.is_set()
    # The user's turn is speech too: quiet waits for both sides, whichever stops last.
    await devices.speaker.push_frame(UserStartedSpeakingFrame())
    await devices.speaker.push_frame(BotStartedSpeakingFrame())
    await devices.speaker.push_frame(BotStoppedSpeakingFrame())
    assert not devices.speaker.quiet.is_set()
    await devices.speaker.push_frame(UserStoppedSpeakingFrame())
    assert devices.speaker.quiet.is_set()
    assert len(pushed) == 7  # each passed on
