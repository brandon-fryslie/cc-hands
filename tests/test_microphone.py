"""What the keyed microphone lets the pipeline hear, decided at capture, with no audio device."""

import asyncio

from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
from pipecat.transports.local.audio import LocalAudioTransportParams

from hands.voice.microphone import ECHO_PATH_SECS, KeyedAudioTransport, buffer_age, heard
from hands.voice.ptt import Gate, PushToTalk

LOUD = b"\x7f\x7f" * 320
QUIET = bytes(len(LOUD))
DOWN = Gate(key="down")


def test_only_a_held_key_after_the_speaker_has_gone_quiet_is_heard() -> None:
    assert heard(LOUD, DOWN, captured=2.0, speaker_quiet_at=1.0) == LOUD
    assert heard(LOUD, DOWN, captured=0.9, speaker_quiet_at=1.0) == QUIET
    assert heard(LOUD, Gate(), captured=2.0, speaker_quiet_at=1.0) == QUIET


class Devices:
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
        setattr(self.speaker, "_out_stream", SimpleStream())  # PyAudio's stream, which the speaker only writes to

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
    def write(self, audio: bytes) -> None:
        pass


async def test_the_reply_already_given_to_the_speaker_is_not_heard_after_a_press() -> None:
    devices = Devices()
    await devices.play(LOUD, at=1.0)
    await devices.capture(at=1.0)  # key up
    devices.key.move_key("down")
    await devices.capture(at=1.0 + ECHO_PATH_SECS - 0.01)  # the reply still in the room
    await devices.capture(at=1.0 + ECHO_PATH_SECS)  # gone
    assert devices.pushed == [QUIET, QUIET, LOUD]


async def test_silence_written_to_the_speaker_keeps_nothing_shut() -> None:
    devices = Devices()
    devices.key.move_key("down")
    await devices.play(QUIET, at=1.0)
    await devices.capture(at=1.01)
    assert devices.pushed == [LOUD]


async def test_a_late_callback_is_judged_by_when_its_sound_was_recorded() -> None:
    devices = Devices()
    devices.key.move_key("down")
    await devices.play(LOUD, at=1.0)
    # Delivered after the speaker went quiet, but recorded 50 ms before: the reply's tail.
    await devices.capture(at=1.0 + ECHO_PATH_SECS + 0.01, age=0.05)
    assert devices.pushed == [QUIET]


def test_a_buffer_the_host_cannot_date_is_as_old_as_its_callback() -> None:
    assert buffer_age({"current_time": 10.0, "input_buffer_adc_time": 9.973}) == 10.0 - 9.973
    assert buffer_age({"current_time": 10.0, "input_buffer_adc_time": 0.0}) == 0.0
    assert buffer_age(None) == 0.0
