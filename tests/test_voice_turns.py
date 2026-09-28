"""The user's turns through Pipecat's real turn machinery: key-tagged microphone frames in, what the model is sent out.

No keyboard and no microphone, so nothing anyone does at this computer can change a result. Each hold is a run of frames
tagged with the key they were captured under, as the keyed microphone makes them, and each transcription waits until the
test hands Whisper its text, so a hold that lands while another is still being transcribed is set up on purpose rather
than hoped for. Whisper, the user aggregator, and the stop strategy are the ones `build_voice` wires.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass, field

import mlx_whisper
import pytest
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    LLMContextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMSpecificMessage
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.whisper.stt import WhisperSTTServiceMLX
from pipecat.workers.runner import WorkerRunner

from hands.voice import pipeline as built
from hands.voice.ptt import Key, KeyedAudio
from hands.voice.turnstop import TurnOpened, TurnResolved
from hands.voice.whisper import Whisper

CLOSING = "that is all"

# Well inside the 5 s after which Pipecat ends a turn on its own, so a turn left open fails here instead of ending late.
PATIENCE_SECS = 2.0


class Recorded(FrameProcessor):
    """What leaves the user aggregator for the model, the turns it ended, and the holds Whisper opened."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self.sent: list[str] = []
        self.started = 0
        self.stopped = 0
        self.holds: list[int] = []
        self.resolved: list[int] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case LLMContextFrame(context=context):
                said = context.get_messages()[-1]
                # The aggregator writes the user's turn as a plain message, never a provider's own.
                assert not isinstance(said, LLMSpecificMessage)
                self.sent.append(str(said.get("content")))
            case UserStartedSpeakingFrame():
                self.started += 1
            case UserStoppedSpeakingFrame():
                self.stopped += 1
            case TurnResolved(hold=hold):
                self.resolved.append(hold)
            case TurnOpened(hold=hold):
                self.holds.append(hold)
            case _:
                pass
        await self.push_frame(frame, direction)


class NoSpeech(FrameProcessor):
    """Stands in for pocket-tts, which loads its weights when it is made."""

    Settings = built.PocketTTSService.Settings

    def __init__(self, **_: object) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)


@dataclass
class Rig:
    worker: PipelineWorker
    stt: Whisper
    out: Recorded
    # What each transcription, oldest first, will find; it waits for the test to say.
    texts: asyncio.Queue[str] = field(default_factory=asyncio.Queue[str])
    # The audio each transcription was given.
    heard: list[bytes] = field(default_factory=list[bytes])

    async def hold(self, keys: Sequence[Key], sound: bytes = b"\x00\x00" * 320) -> None:
        await self.worker.queue_frames([KeyedAudio(audio=sound, sample_rate=16000, num_channels=1, key=key) for key in keys])

    async def until(self, what: Callable[[], bool]) -> None:
        async with asyncio.timeout(PATIENCE_SECS):
            while not what():
                await asyncio.sleep(0.01)

    async def everything_sent(self, holds: int) -> list[str]:
        """What the model was sent once the test's `holds` holds are all resolved, closed by one more spoken hold.

        Whether a hold joins the turn the one before it is still open in depends on whether its opening, a system
        frame, overtakes the last one's resolution, a data frame, so what is waited for is every hold resolved and
        every turn that started ended, not a number of turns. The aggregator says a turn stopped, a system frame,
        ahead of what it sends the model, a data frame, which keeps its order among the data; so once the closing
        hold's words are in, so is everything sent before them.
        """
        out = self.out
        await self.until(lambda: len(out.holds) == len(out.resolved) == holds and out.started == out.stopped)
        await self.hold(["down", "up"])
        await self.texts.put(CLOSING)
        await self.until(lambda: CLOSING in self.out.sent)
        return self.out.sent[:-1]


@pytest.fixture
async def rig(monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[Rig, None]:
    monkeypatch.setattr(built, "PocketTTSService", NoSpeech)
    voice = built.build_voice(
        built.VoiceConfig(llm=built.AnthropicBackend(api_key="unused", model="unused"), whisper_model="unused", voice="unused"),
        tools=[],
    )
    out = Recorded()
    worker = PipelineWorker(Pipeline([voice.stt, voice.user_turns, out]), idle_timeout_secs=None)
    made = Rig(worker, voice.stt, out)

    async def transcribe(_self: WhisperSTTServiceMLX, audio: bytes) -> AsyncGenerator[Frame, None]:
        made.heard.append(audio)
        text = await made.texts.get()
        for heard in [text] if text else []:
            yield TranscriptionFrame(heard, "user", "now")

    monkeypatch.setattr(WhisperSTTServiceMLX, "run_stt", transcribe)
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def _started(_worker: PipelineWorker, _frame: Frame) -> None:  # pyright: ignore[reportUnusedFunction]
        started.set()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    await asyncio.wait_for(started.wait(), PATIENCE_SECS)
    yield made
    await worker.cancel()
    await running


async def test_a_spoken_hold_is_sent(rig: Rig) -> None:
    await rig.hold(["down", "down", "up"])
    await rig.texts.put("what time is it")
    assert await rig.everything_sent(holds=1) == ["what time is it"]


async def test_a_hold_hears_from_its_press_and_nothing_before_it(rig: Rig) -> None:
    shift, early, held = (bytes([n, n]) * 320 for n in (1, 2, 3))
    # A press that was Shift after all, then one said into before it meant talk.
    await rig.hold(["arming"], sound=shift)
    await rig.hold(["up"])
    await rig.hold(["arming", "arming"], sound=early)
    await rig.hold(["down", "up"], sound=held)
    await rig.texts.put("what time is it")
    assert await rig.everything_sent(holds=1) == ["what time is it"]
    assert rig.heard[0].startswith(early + early + held)
    assert shift not in rig.heard[0]


async def test_a_hold_keeps_everything_from_its_press_however_long_it_takes_to_mean_talk(rig: Rig) -> None:
    # Pipecat keeps the last second of audio nobody is speaking in; 60 frames of 20 ms arm for 1.2 s.
    early, held = (bytes([n, n]) * 320 for n in (2, 3))
    arming: list[Key] = ["arming"] * 60
    await rig.hold(arming, sound=early)
    await rig.hold(["down", "up"], sound=held)
    await rig.texts.put("what time is it")
    assert await rig.everything_sent(holds=1) == ["what time is it"]
    assert rig.heard[0].startswith(early * 60 + held)


async def test_a_press_that_was_shift_leaves_nothing_behind_for_the_next_hold(rig: Rig) -> None:
    shift, held = (bytes([n, n]) * 320 for n in (1, 3))
    arming: list[Key] = ["arming"] * 60
    up: list[Key] = ["up"] * 60
    await rig.hold(arming, sound=shift)
    await rig.hold(up)
    await rig.hold(["arming", "down", "up"], sound=held)
    await rig.texts.put("what time is it")
    assert await rig.everything_sent(holds=1) == ["what time is it"]
    assert shift not in rig.heard[0]
    assert rig.heard[0].startswith(held * 2)


async def test_a_dropped_hold_ends_its_turn_and_sends_nothing(rig: Rig) -> None:
    await rig.hold(["down", "down", "dropped"])
    assert await rig.everything_sent(holds=1) == []


async def test_a_hold_that_heard_nothing_ends_its_turn_and_sends_nothing(rig: Rig) -> None:
    await rig.hold(["down", "up"])
    await rig.texts.put("")
    assert await rig.everything_sent(holds=1) == []


async def test_a_hold_released_to_a_whisper_that_cannot_transcribe_ends_its_turn_and_sends_nothing(rig: Rig) -> None:
    await rig.stt.set_usable(False)
    await rig.hold(["down", "up"])
    await rig.until(lambda: rig.out.stopped == 1)
    await rig.stt.set_usable(True)
    assert await rig.everything_sent(holds=1) == []


async def test_a_hold_dropped_by_a_press_is_thrown_away_and_the_press_is_a_hold_of_its_own(rig: Rig) -> None:
    # A press that finds the key down drops the hold and arms again between two frames, so no frame sees it dropped.
    await rig.hold(["down", "arming", "down", "up"])
    await rig.texts.put("what time is it")
    assert await rig.everything_sent(holds=2) == ["what time is it"]


async def test_a_hold_dropped_while_the_last_is_transcribed_leaves_the_last_to_be_sent(rig: Rig) -> None:
    # The second press joins the turn the first is still open in; its drop must not end that turn before the first
    # hold's words are in it.
    await rig.hold(["down", "down", "up", "down", "dropped"])
    await rig.until(lambda: rig.out.holds == [1, 2])
    await rig.texts.put("what time is it")
    assert await rig.everything_sent(holds=2) == ["what time is it"]


async def test_a_hold_that_heard_nothing_does_not_end_the_hold_pressed_after_it(rig: Rig) -> None:
    await rig.hold(["down", "up", "down", "down"])
    await rig.until(lambda: rig.out.holds == [1, 2])
    # The first hold's empty transcription arrives while the second is still held.
    await rig.texts.put("")
    await rig.hold(["up"])
    await rig.texts.put("how many sessions are running")
    assert await rig.everything_sent(holds=2) == ["how many sessions are running"]


async def test_whisper_has_loaded_the_model_its_turns_transcribe_with_once_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """MLX Whisper keeps the model it loaded for the process, keyed on what it was asked for, so the first turn pays
    for no load only if the one done at construction asked for exactly what a turn's transcription asks for."""
    asked: list[dict[str, object]] = []

    def transcribe(_audio: object, **options: object) -> dict[str, object]:
        asked.append(options)
        return {"segments": []}

    monkeypatch.setattr(mlx_whisper, "transcribe", transcribe)
    whisper = Whisper(settings=WhisperSTTServiceMLX.Settings(model="mlx-community/whisper-tiny"))
    assert [options["path_or_hf_repo"] for options in asked] == ["mlx-community/whisper-tiny"]
    whisper._transcribing.append(1)  # pyright: ignore[reportPrivateUsage]  (the hold a release queues)
    [frame async for frame in whisper.run_stt(b"\x00\x00" * 16_000)]
    assert len(asked) == 2 and asked[1] == asked[0]


async def test_whisper_hears_only_the_keyed_microphone() -> None:
    whisper = Whisper(settings=WhisperSTTServiceMLX.Settings(model="unused"))
    with pytest.raises(TypeError, match="carries no key"):
        await whisper.process_audio_frame(InputAudioRawFrame(b"\x00\x00", 16000, 1), FrameDirection.DOWNSTREAM)
