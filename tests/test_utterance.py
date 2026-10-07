"""Something a session gives hands to say unasked is one wide event, from heard to its fate, read off the output
transport by the frames sent with what says it."""

import asyncio
import threading
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pytest
from pipecat.frames.frames import Frame, InterruptionFrame, OutputAudioRawFrame, TTSAudioRawFrame, TTSSpeakFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from hands.core.effects import Expired, SessionGone, Speak
from hands.core.session import Permission, SessionId
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.voice.speech import Known, sent
from hands.voice.utterance import Audible, Resumed, Utterance, Utterances, Uttered, Uttering, uttering

from conftest import running
from hands.voice.microphone import DefaultDevices, KeyedAudioTransport, Pull
from hands.voice.ptt import PushToTalk
from pipecat.transports.local.audio import LocalAudioTransportParams
from test_microphone import FreshPortAudio, Room, _phone  # pyright: ignore[reportPrivateUsage]

API = SessionId("api")
EXPIRED = Speak(Expired(API, Permission("Bash", {"command": "ls"})))
# What the output transport pushes of one utterance: the frames that lead and close it, and what plays between.
Step = Literal["lead", "audio", "barge-in", "resume", "close"]
AUDIO = OutputAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=16_000, num_channels=1)


@dataclass
class Rig:
    utterances: Utterances
    recorded: list[Entry]
    audible: Audible
    output: FrameProcessor
    # The output transport's clock, which the test moves.
    now: list[float]

    async def played(self, frames: Sequence[Frame]) -> None:
        """Each frame pushed by the output transport, in order, as it plays it."""
        for frame in frames:
            await self.audible.on_push_frame(FramePushed(self.output, FrameProcessor(), frame, FrameDirection.DOWNSTREAM, 0))

    async def event(self) -> WideEvent:
        async with asyncio.timeout(5):
            while not (events := [entry for entry in self.recorded if isinstance(entry, WideEvent) and entry.event == "utterance"]):
                await asyncio.sleep(0.01)
        [event] = events
        return event


@pytest.fixture
async def rig() -> AsyncGenerator[Rig, None]:
    recorded: list[Entry] = []
    utterances = Utterances(recorded.append)
    output, now = FrameProcessor(), [0.0]
    keeping = asyncio.create_task(utterances.keep())
    try:
        yield Rig(utterances, recorded, Audible(output, lambda: now[0]), output, now)
    finally:
        keeping.cancel()
        await asyncio.gather(keeping, return_exceptions=True)


def said(utterance: Utterance) -> tuple[Frame, ...]:
    """What the brain's stage sends to say a line as written, with what reads its fate."""
    return uttering((utterance,), (TTSSpeakFrame("Nobody answered api about Bash in time, so I told it no."),))


async def test_an_announcement_spoken_is_one_event_with_its_first_audio_timed_from_when_it_was_heard(rig: Rig) -> None:
    utterance = rig.utterances.heard(API, EXPIRED)
    lead, line, close = said(utterance)
    rig.now[0] = utterance.begun.began + 0.25
    await rig.played([lead, AUDIO])
    rig.now[0] += 1.0
    await rig.played([AUDIO, close])
    event = await rig.event()
    assert (event.outcome, event.facts["fate"], event.facts["first_audio_ms"], event.facts["session"], event.facts["heard"]) == ("ok", "played", 250.0, API, EXPIRED)
    assert [type(frame) for frame in (lead, line, close)] == [Uttering, TTSSpeakFrame, Uttered]


@pytest.mark.parametrize(
    ("steps", "fate"),
    [
        pytest.param(("lead", "close"), "silent", id="handed on and nothing of it played"),
        pytest.param(("lead", "audio", "barge-in", "close"), "cut", id="a barge-in over it"),
        pytest.param(("lead", "barge-in", "close"), "cut", id="a barge-in before its first audio"),
        pytest.param(("close",), "cut", id="a barge-in that dropped it before it reached the speaker"),
        pytest.param(("lead", "audio", "barge-in", "resume", "audio", "close"), "played", id="a barge-in the turn saying it went on through"),
        pytest.param(("barge-in", "resume", "audio", "close"), "played", id="a barge-in the turn went on through, that dropped its lead"),
    ],
)
async def test_what_of_an_utterance_was_heard_is_read_off_the_output_transport(rig: Rig, steps: Sequence[Step], fate: str) -> None:
    utterance = rig.utterances.heard(API, EXPIRED)
    lead, _, close = said(utterance)
    pushed: dict[Step, Frame] = {"lead": lead, "audio": AUDIO, "barge-in": InterruptionFrame(), "resume": Resumed((utterance,)), "close": close}
    await rig.played([pushed[step] for step in steps])
    assert (await rig.event()).facts["fate"] == fate


async def test_what_the_model_knows_unsaid_sends_nothing_and_is_silent_as_it_is_sent(rig: Rig) -> None:
    """Never said, so never read off the speaker, where a brain turn speaking beside it would lend it its audio."""
    utterance = rig.utterances.heard(API, EXPIRED)
    assert tuple(sent(Known(), (utterance,))) == ()
    assert (await rig.event()).facts["fate"] == "silent"


async def test_audio_that_played_after_a_barge_in_cut_an_utterance_is_not_its_first(rig: Rig) -> None:
    utterance = rig.utterances.heard(API, EXPIRED)
    lead, _, close = said(utterance)
    await rig.played([lead, InterruptionFrame(), AUDIO, close])
    event = await rig.event()
    assert event.facts["fate"] == "cut" and "first_audio_ms" not in event.facts


async def test_audio_that_played_before_an_utterance_was_led_on_is_not_its_first(rig: Rig) -> None:
    utterance = rig.utterances.heard(API, EXPIRED)
    lead, _, close = said(utterance)
    await rig.played([AUDIO, Uttered(()), lead, close])
    event = await rig.event()
    assert event.facts["fate"] == "silent" and "first_audio_ms" not in event.facts


async def test_an_utterance_open_as_hands_stops_ends_cancelled() -> None:
    recorded: list[Entry] = []
    utterances = Utterances(recorded.append)
    keeping = asyncio.create_task(utterances.keep())
    utterances.heard(API, SessionGone(API))
    await asyncio.sleep(0.05)
    keeping.cancel()
    await asyncio.gather(keeping, return_exceptions=True)
    [event] = recorded
    assert isinstance(event, WideEvent) and (event.event, event.outcome, event.facts["session"]) == ("utterance", "cancelled", API)


async def test_an_utterance_failed_on_its_way_says_why_and_its_fate_all_the_same(rig: Rig) -> None:
    utterance = rig.utterances.heard(API, EXPIRED)
    utterance.fail("the text could not be summarised: SummaryFailed: nothing came back")
    utterance.settle("noted")
    event = await rig.event()
    assert (event.outcome, event.error, event.facts["fate"]) == ("failed", "the text could not be summarised: SummaryFailed: nothing came back", "noted")


async def test_an_utterance_settled_twice_is_refused(rig: Rig) -> None:
    utterance = rig.utterances.heard(API, EXPIRED)
    utterance.settle("dropped")
    with pytest.raises(asyncio.InvalidStateError):
        utterance.settle("played")
    assert (await rig.event()).facts["fate"] == "dropped"


class Playing:
    """A speaker device driven as PortAudio drives a callback stream: it takes 20 ms of sound every millisecond, as fast
    as it is given it, unless it is holding."""

    def __init__(self) -> None:
        self.pull: Pull | None = None
        self.holding = False
        self.sounded = 0  # samples taken that were not silence
        self._stopped = threading.Event()

    def start_stream(self) -> None:
        threading.Thread(target=self._play, daemon=True).start()

    def stop_stream(self) -> None:
        self._stopped.set()

    def close(self) -> None: ...

    def _play(self) -> None:
        assert self.pull is not None, "opened before it is started"
        while not self._stopped.wait(0.001):
            if not self.holding:
                taken = self.pull(None, 320, {}, 0)[0]
                self.sounded += int(np.count_nonzero(np.frombuffer(taken, np.int16)))


class Speakers(FreshPortAudio):
    """PortAudio whose speaker is a device that plays."""

    def __init__(self) -> None:
        super().__init__([])
        self.speaker = Playing()

    def open(self, **settings: object) -> Playing:  # pyright: ignore[reportIncompatibleMethodOverride]
        self.speaker.pull = cast(Pull, settings["stream_callback"])
        return self.speaker


async def test_the_output_transport_passes_what_reads_an_utterance_in_order_with_the_audio_it_writes() -> None:
    """The output transport writes audio to the device and passes every other frame on only once the audio ahead of it is
    written: the fate read off its pushes is what the speaker played, with hands' own transport on a fake device."""
    speakers = Speakers()
    params = LocalAudioTransportParams(audio_out_enabled=True, audio_out_sample_rate=16_000)
    speaker = KeyedAudioTransport(params, PushToTalk(lambda _: None), _phone(), lambda _: None, portaudio=lambda: speakers, defaults=lambda: DefaultDevices(input=1, output=1), echo=lambda: Room()).output()
    recorded: list[Entry] = []
    utterances = Utterances(recorded.append)
    keeping = asyncio.create_task(utterances.keep())
    utterance = utterances.heard(API, EXPIRED)
    try:
        async with running([speaker], observers=[Audible(speaker)]) as run:
            await run.worker.queue_frames([Uttering((utterance,)), TTSAudioRawFrame(b"\x01\x00" * 1600, 16_000, 1), Uttered((utterance,))])
            async with asyncio.timeout(5):
                while not recorded:
                    await asyncio.sleep(0.01)
    finally:
        keeping.cancel()
        await asyncio.gather(keeping, return_exceptions=True)
    [event] = recorded
    assert isinstance(event, WideEvent) and event.facts["fate"] == "played" and isinstance(event.facts["first_audio_ms"], float)
    assert speakers.speaker.sounded  # the line reached the device


async def test_a_barge_in_while_the_output_transport_still_holds_an_utterance_s_audio_cuts_it_and_its_close_still_arrives() -> None:
    speakers = Speakers()
    params = LocalAudioTransportParams(audio_out_enabled=True, audio_out_sample_rate=16_000)
    speaker = KeyedAudioTransport(params, PushToTalk(lambda _: None), _phone(), lambda _: None, portaudio=lambda: speakers, defaults=lambda: DefaultDevices(input=1, output=1), echo=lambda: Room()).output()
    recorded: list[Entry] = []
    utterances = Utterances(recorded.append)
    keeping = asyncio.create_task(utterances.keep())
    utterance = utterances.heard(API, EXPIRED)
    try:
        async with running([speaker], observers=[Audible(speaker)]) as run:
            # The device takes its first chunk and holds it: the rest of the line waits in the output transport.
            speakers.speaker.holding = True
            await run.worker.queue_frames([Uttering((utterance,)), *(TTSAudioRawFrame(b"\x01\x00" * 1600, 16_000, 1) for _ in range(20)), Uttered((utterance,))])
            await asyncio.sleep(0.1)
            await run.worker.queue_frame(InterruptionFrame())
            await asyncio.sleep(0.1)
            speakers.speaker.holding = False
            async with asyncio.timeout(5):
                while not recorded:
                    await asyncio.sleep(0.01)
    finally:
        speakers.speaker.holding = False
        keeping.cancel()
        await asyncio.gather(keeping, return_exceptions=True)
    [event] = recorded
    assert isinstance(event, WideEvent) and event.facts["fate"] == "cut"
    assert speakers.speaker.sounded < 20 * 1600


async def test_what_a_turn_says_after_a_barge_in_it_went_on_through_is_still_its_utterance_s() -> None:
    """Led on again right behind the barge-in, through the output transport's flush of what it held."""
    speakers = Speakers()
    params = LocalAudioTransportParams(audio_out_enabled=True, audio_out_sample_rate=16_000)
    speaker = KeyedAudioTransport(params, PushToTalk(lambda _: None), _phone(), lambda _: None, portaudio=lambda: speakers, defaults=lambda: DefaultDevices(input=1, output=1), echo=lambda: Room()).output()
    recorded: list[Entry] = []
    utterances = Utterances(recorded.append)
    keeping = asyncio.create_task(utterances.keep())
    utterance = utterances.heard(API, EXPIRED)
    try:
        async with running([speaker], observers=[Audible(speaker)]) as run:
            speakers.speaker.holding = True
            await run.worker.queue_frames([Uttering((utterance,)), *(TTSAudioRawFrame(b"\x01\x00" * 1600, 16_000, 1) for _ in range(20))])
            await asyncio.sleep(0.1)
            await run.worker.queue_frames([InterruptionFrame(), Resumed((utterance,))])
            await asyncio.sleep(0.1)
            speakers.speaker.holding = False
            await run.worker.queue_frames([TTSAudioRawFrame(b"\x01\x00" * 1600, 16_000, 1), Uttered((utterance,))])
            async with asyncio.timeout(5):
                while not recorded:
                    await asyncio.sleep(0.01)
    finally:
        speakers.speaker.holding = False
        keeping.cancel()
        await asyncio.gather(keeping, return_exceptions=True)
    [event] = recorded
    assert isinstance(event, WideEvent) and event.facts["fate"] == "played"
