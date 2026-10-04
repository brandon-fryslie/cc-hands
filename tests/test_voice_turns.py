"""The user's turns through Pipecat's real turn machinery: key-tagged microphone frames in, what the model is sent out.

No keyboard and no microphone, so nothing anyone does at this computer can change a result. Each hold is a run of frames
tagged with the key they were captured under, as the keyed microphone makes them, and each transcription waits until the
test hands Whisper its text, so a hold that lands while another is still being transcribed is set up on purpose rather
than hoped for. Whisper, the user aggregator, and the stop strategy are the ones `build_voice` wires.
"""

import asyncio
from pathlib import Path
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass, field

import mlx_whisper
import pytest
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    LLMContextFrame,
    TextFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMSpecificMessage
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.whisper.stt import WhisperSTTServiceMLX

from conftest import running, unprimed
from hands.sessions.audit import Entry, HoldHeard, Yielded
from hands.voice import pipeline as built
from hands.sessions.home import Home
from hands.sessions.registry import Sessions
from hands.voice.floor import Floor
from hands.voice.refocus import Refocus
from hands.voice.player import Player
from hands.voice.ptt import Key, KeyedAudio, PushToTalk
from hands.core.effects import Asking, Narrate, SessionGone
from hands.core.pending import Finished, News
from hands.core.session import Held, Membership, Permission, RequestId, Running, Session, SessionId
from hands.core.status import Busy, Stamp
from hands.voice.speech import Pushed, Unprompted
from hands.voice.turnstop import TurnOpened, TurnResolved
from hands.voice.whisper import Whisper
from hands.voice import voices

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
        # The user's turn and what hands said around it, in the order the model's stage would take them.
        self.order: list[str] = []
        # Every message in the model's context as the last frame of it passed: appends that land together share one.
        self.context: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case LLMContextFrame(context=context):
                self.context = [str(message.get("content")) for message in context.get_messages() if not isinstance(message, LLMSpecificMessage)]
                said = context.get_messages()[-1]
                # The aggregator writes the user's turn as a plain message, never a provider's own.
                assert not isinstance(said, LLMSpecificMessage)
                self.sent.append(str(said.get("content")))
                self.order.append(f"sent: {said.get('content')}")
            case UserStartedSpeakingFrame():
                self.started += 1
                self.order.append("started")
            case UserStoppedSpeakingFrame():
                self.stopped += 1
                self.order.append("stopped")
            case TTSSpeakFrame(text=text):
                self.order.append(f"said: {text}")
            case TextFrame(text=text):
                self.order.append(text)
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
class Clock:
    """The floor's clock, read where the test sets it."""

    now: float = 0.0

    def __call__(self) -> float:
        return self.now


@dataclass
class Rig:
    worker: PipelineWorker
    stt: Whisper
    out: Recorded
    recorded: list[Entry]
    clock: Clock
    # Each live session, as the floor reads it when it lets go.
    live: dict[SessionId, Session]
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
async def rig(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AsyncGenerator[Rig, None]:
    monkeypatch.setattr(built, "PocketTTSService", NoSpeech)
    recorded: list[Entry] = []
    voice = built.build_voice(
        built.VoiceConfig(llm=built.AnthropicBackend(base_url="unused", api_key="unused", model="unused"), whisper_model="unused", voice=voices.DEFAULT),
        tools=[],
        llm=FrameProcessor(),
        key=PushToTalk(recorded.append),
        player=Player(recorded.append),
        floor=Floor(lambda _: None, Pushed(), lambda id: id, dict),
        refocus=Refocus(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=recorded.append), Home(tmp_path), recorded.append),
        prompt=unprimed,
        record=recorded.append,
    )
    out, clock, live = Recorded(), Clock(), dict[SessionId, Session]()
    texts: asyncio.Queue[str] = asyncio.Queue()
    heard: list[bytes] = []

    async def transcribe(_self: Whisper, hold: int, audio: bytes) -> HoldHeard:
        heard.append(audio)
        return HoldHeard(hold, await texts.get() or None, ())

    monkeypatch.setattr(Whisper, "_heard", transcribe)
    # The floor sits where build_voice puts it, between Whisper and the user aggregator.
    async with running([voice.stt, Floor(recorded.append, Pushed(), lambda id: id, lambda: live, clock), voice.user_turns, out]) as run:
        yield Rig(run.worker, voice.stt, out, recorded, clock, live, texts, heard)


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


API, WEB = SessionId("api"), SessionId("web")
WAITING = Unprompted(SessionGone(API))


def asking(session: SessionId, request: str) -> Unprompted:
    return Unprompted(Narrate(Asking(session, RequestId(request), Permission("Bash", {"command": "ls"}))))


def finished(session: SessionId, reply: str) -> Unprompted:
    return Unprompted(Finished(session, (News(None, reply, "", "", (), frozenset()),), "full"))


async def test_a_session_waiting_while_the_key_is_held_is_said_after_the_users_turn_is_sent_and_not_before(rig: Rig) -> None:
    rig.clock.now = 10.0
    await rig.hold(["down", "down"])
    await rig.until(lambda: rig.out.started == 1)
    await rig.worker.queue_frames([WAITING, TextFrame("marker")])
    # Frames keep their order, so the marker past the floor with the announcement not is the announcement held.
    await rig.until(lambda: "marker" in rig.out.order)
    rig.clock.now = 13.5
    await rig.hold(["up"])
    await rig.texts.put("what time is it")
    await rig.until(lambda: "said: The session api is gone." in rig.out.order)
    assert rig.out.order == ["started", "marker", "stopped", "sent: what time is it", "said: The session api is gone."]
    assert [(entry.held, entry.told, entry.waited) for entry in rig.recorded if isinstance(entry, Yielded)] == [(("SessionGone",), ("SessionGone",), 3.5)]


def waiting_on(session: SessionId, request: str) -> Session:
    """The session live, its dialog waiting on the user to answer `request`."""
    member = Membership(session, pid=4242, cwd=Path("/code/a"), transcript=Path("/code/a/t.jsonl"))
    dialog = Held(Permission("Bash", {"command": "ls"}), RequestId(request), deadline=60.0, warned=False)
    return Session(member, Running(Busy(), Stamp(1000), None), mode=None, dialog=dialog)


async def test_a_request_for_an_api_model_while_the_key_is_held_joins_the_context_after_the_users_turn(rig: Rig) -> None:
    rig.live[API] = waiting_on(API, "r1")
    await rig.hold(["down", "down"])
    await rig.until(lambda: rig.out.started == 1)
    await rig.worker.queue_frames([asking(API, "r1"), TextFrame("marker")])
    await rig.until(lambda: "marker" in rig.out.order)
    await rig.hold(["up"])
    await rig.texts.put("what time is it")
    await rig.until(lambda: len(rig.out.order) == 5)
    assert rig.out.order[:4] == ["started", "marker", "stopped", "sent: what time is it"]
    assert rig.out.order[4].startswith("sent: [hands] The Claude Code session api is waiting for permission to use Bash")


async def test_two_sessions_finishing_during_one_held_key_are_told_asks_first_and_one_telling_a_session(rig: Rig) -> None:
    """What a session waits on the user for is told before what a session did; a session's turns that finished while
    the user talked are one telling; and a request answered at the keyboard meanwhile is not told at all."""
    rig.live[WEB] = waiting_on(WEB, "w2")
    await rig.hold(["down", "down"])
    await rig.until(lambda: rig.out.started == 1)
    # web's first request was answered at the keyboard while the key was down; its second still waits.
    await rig.worker.queue_frames([finished(API, "Fixed the parser."), asking(WEB, "w1"), finished(API, "Pushed it."), asking(WEB, "w2"), TextFrame("marker")])
    await rig.until(lambda: "marker" in rig.out.order)
    await rig.hold(["up"])
    await rig.texts.put("what time is it")
    await rig.until(lambda: len(rig.out.context) == 3)
    said, asked, told = rig.out.context
    assert said == "what time is it"
    assert asked.startswith("[hands] The Claude Code session web is waiting for permission") and "Request id: w2." in asked
    assert told.startswith("[hands] The Claude Code session api (id api) finished 2 turns. The last thing it said was:\n\nFixed the parser.\n\nThen, in the turn after that: The last thing it said was:\n\nPushed it.")
    [yielded] = [entry for entry in rig.recorded if isinstance(entry, Yielded)]
    assert (yielded.held, yielded.told) == (("Finished", "Narrate", "Finished", "Narrate"), ("Narrate", "Finished"))


async def test_a_hold_that_was_shift_after_all_gives_back_what_waited_for_it(rig: Rig) -> None:
    await rig.hold(["down"])
    await rig.until(lambda: rig.out.started == 1)
    await rig.worker.queue_frame(WAITING)
    await rig.hold(["dropped"])
    await rig.until(lambda: "said: The session api is gone." in rig.out.order)
    assert rig.out.order == ["started", "stopped", "said: The session api is gone."]


async def test_what_hands_says_with_no_turn_open_passes_at_once_and_records_no_wait(rig: Rig) -> None:
    await rig.worker.queue_frame(WAITING)
    await rig.until(lambda: rig.out.order == ["said: The session api is gone."])
    assert [(entry.held, entry.told, entry.waited) for entry in rig.recorded if isinstance(entry, Yielded)] == [(("SessionGone",), ("SessionGone",), 0.0)]


async def test_a_request_answered_before_it_reaches_the_floor_is_not_told(rig: Rig) -> None:
    await rig.worker.queue_frames([asking(API, "gone"), TextFrame("marker")])
    await rig.until(lambda: "marker" in rig.out.order)
    assert rig.out.order == ["marker"]
    assert [(entry.held, entry.told) for entry in rig.recorded if isinstance(entry, Yielded)] == [(("Narrate",), ())]


async def test_whisper_has_loaded_the_model_its_turns_transcribe_with_once_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """MLX Whisper keeps the model it loaded for the process, keyed on what it was asked for, so the first turn pays
    for no load only if the one done at construction asked for exactly what a turn's transcription asks for."""
    asked: list[dict[str, object]] = []

    def transcribe(_audio: object, **options: object) -> dict[str, object]:
        asked.append(options)
        return {"segments": []}

    monkeypatch.setattr(mlx_whisper, "transcribe", transcribe)
    whisper = Whisper(settings=WhisperSTTServiceMLX.Settings(model="mlx-community/whisper-tiny"), prompt=unprimed, record=lambda _: None)
    assert [options["path_or_hf_repo"] for options in asked] == ["mlx-community/whisper-tiny"]
    whisper._transcribing.append(1)  # pyright: ignore[reportPrivateUsage]  (the hold a release queues)
    [frame async for frame in whisper.run_stt(b"\x00\x00" * 16_000)]
    assert len(asked) == 2 and asked[1] == asked[0]


async def test_whisper_hears_only_the_keyed_microphone() -> None:
    whisper = Whisper(settings=WhisperSTTServiceMLX.Settings(model="unused"), prompt=unprimed, record=lambda _: None)
    with pytest.raises(TypeError, match="carries no key"):
        await whisper.process_audio_frame(InputAudioRawFrame(b"\x00\x00", 16000, 1), FrameDirection.DOWNSTREAM)
