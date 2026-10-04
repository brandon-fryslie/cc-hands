"""The user's turns through Pipecat's real turn machinery: key-tagged microphone frames in, what the model is sent out.

No keyboard and no microphone, so nothing anyone does at this computer can change a result. Each hold is a run of frames
tagged with the key they were captured under, as the keyed microphone makes them, and each transcription waits until the
test hands Whisper its text, so a hold that lands while another is still being transcribed is set up on purpose rather
than hoped for. Whisper, the user aggregator, and the stop strategy are the ones `build_voice` wires.
"""

import asyncio
import struct
import wave
from pathlib import Path
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from io import BytesIO
from dataclasses import dataclass, field, replace
from typing import Literal, cast

import pytest
from aiohttp import web
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

from conftest import Endpoint, ServeApi, events, running, unprimed
from hands.core.front import FrontUnread, InFront, SessionInFront
from hands.core.place import Place
from hands.voice.backends import AnthropicBackend, OpenAICompatibleBackend
from hands.voice.beside import Noting
from hands.sessions.audit import Entry, HoldHeard, Levels
from hands.sessions.wide import Fact
from hands.voice import pipeline as built
from hands.voice.conversation import cue_receipt
from hands.sessions.home import Home
from hands.sessions.registry import Sessions
from hands.voice.floor import Floor
from hands.voice.refocus import Refocus
from hands.voice.player import Player
from hands.voice.ptt import Gate, Key, KeyedAudio, PushToTalk
from hands.core.effects import Asking, Narrate, SessionGone
from hands.core.pending import Finished, News
from hands.core.session import Held, Membership, Permission, RequestId, Running, Session, SessionId
from hands.core.status import Busy, Stamp
from hands.voice.speech import Pushed, Unprompted
from hands.voice.turnstop import TurnOpened, TurnResolved
from hands.voice.whisper import Whisper
from test_llm import Shape, anthropic_stream, openai_stream
from test_narrator import heard
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


# A key position, or "dropped": the turn open thrown away, and the key at rest.
Captured = Key | Literal["dropped"]


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
    # Each turn hands said it received, as its words were written to the context.
    received: list[None] = field(default_factory=list[None])

    # The gate the last frame was captured under.
    gate: Gate = field(default_factory=Gate)

    async def hold(self, keys: Sequence[Captured], sound: bytes = b"\x00\x00" * 320, at: Place = "desk", captured: bytes | None = None) -> None:
        """A frame captured under each key, counted as the gate counts them: the key leaving down for a rest sends the
        turn, and for a press (a key pressed while it was held) or "dropped" throws it away. `captured` is the sound
        before the echo canceller, where the canceller changed it."""
        frames: list[KeyedAudio] = []
        for held in keys:
            match self.gate.key, held:
                case "down", "up" | "listening":
                    self.gate = replace(self.gate, sent=self.gate.sent + 1)
                case "down", "arming" | "dropped":
                    self.gate = replace(self.gate, dropped=self.gate.dropped + 1)
                case _:
                    pass
            self.gate = replace(self.gate, key="up" if held == "dropped" else held)
            frames.append(self.gate.framed(sound, sound if captured is None else captured, 16000, 1, at))
        await self.worker.queue_frames(frames)

    async def capture(self, gate: Gate, sound: bytes) -> None:
        """A frame captured under `gate`, as it stands after whatever moves made it."""
        self.gate = gate
        await self.worker.queue_frame(gate.framed(sound, sound, 16000, 1, gate.place))

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
    async with rigged(monkeypatch, tmp_path, FrameProcessor(), [], []) as made:
        yield made


@asynccontextmanager
async def rigged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, llm: FrameProcessor, noting: list[FrameProcessor], behind: list[FrameProcessor]) -> AsyncGenerator[Rig]:
    """The rig, the user's turns passed through `noting` on their way to what records them, with `behind` run behind that."""
    monkeypatch.setattr(built, "PocketTTSService", NoSpeech)
    recorded: list[Entry] = []
    voice = built.build_voice(
        built.VoiceConfig(llm=built.AnthropicBackend(base_url="unused", api_key="unused", model="unused"), transcription="http://unused/v1", voice=voices.DEFAULT),
        tools=[],
        llm=llm,
        noting=noting,
        key=PushToTalk(recorded.append),
        player=Player(recorded.append),
        floor=Floor(Pushed(), lambda id: id, dict),
        refocus=Refocus(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=recorded.append), Home(tmp_path), recorded.append),
        prompt=unprimed,
        record=recorded.append,
    )
    out, clock, live = Recorded(), Clock(), dict[SessionId, Session]()
    texts: asyncio.Queue[str] = asyncio.Queue()
    heard: list[bytes] = []

    async def transcribe(_self: Whisper, hold: int, levels: Levels, audio: bytes) -> HoldHeard:
        # The hold as uploaded, a WAV: its samples are what the microphone heard.
        with wave.open(BytesIO(audio)) as uploaded:
            heard.append(uploaded.readframes(uploaded.getnframes()))
        return HoldHeard(hold, await texts.get() or None, (), levels)

    monkeypatch.setattr(Whisper, "_heard", transcribe)
    # The floor sits where build_voice puts it, between Whisper and the user aggregator.
    received: list[None] = []
    cue_receipt(voice.user_turns, lambda: received.append(None))
    async with running([voice.stt, Floor(Pushed(), lambda id: id, lambda: live, clock), voice.user_turns, *noting, out, *behind]) as run:
        yield Rig(run.worker, voice.stt, out, recorded, clock, live, texts, heard, received)


async def test_a_spoken_hold_is_sent(rig: Rig) -> None:
    await rig.hold(["down", "down", "up"])
    await rig.texts.put("what time is it")
    assert await rig.everything_sent(holds=1) == ["what time is it"]


def holds_heard(rig: Rig) -> list[HoldHeard]:
    return [entry for entry in rig.recorded if isinstance(entry, HoldHeard)]


async def test_a_holds_record_says_how_loud_it_was_before_and_after_the_echo_canceller(rig: Rig) -> None:
    # Half of full scale captured, a tenth of that left by the canceller: -6.0 and -26.0 dBFS, 20 dB of echo taken out.
    loud, residual = struct.pack("<h", 16384) * 320, struct.pack("<h", 1638) * 320
    # What the key was up over is no part of the hold, however loud: the levels are of the audio transcribed alone.
    await rig.hold(["up", "arming", "down", "down", "up"], sound=residual, captured=loud)
    await rig.texts.put("run")
    await rig.until(lambda: len(holds_heard(rig)) == 1)
    assert holds_heard(rig) == [HoldHeard(1, "run", (), Levels(captured_dbfs=-6.0, heard_dbfs=-26.0))]
    # The record is of the audio Whisper was given: the three frames from the press on, then Pipecat's silence.
    assert rig.heard[0].rstrip(b"\x00") == residual * 3


async def test_a_hold_of_digital_silence_has_no_level_and_a_hold_at_the_phone_has_one_level(rig: Rig) -> None:
    await rig.hold(["down", "up"])
    await rig.texts.put("")
    await rig.until(lambda: len(holds_heard(rig)) == 1)
    quiet = struct.pack("<h", 3277) * 320  # a tenth of full scale
    await rig.hold(["down", "up"], sound=quiet, at="phone")
    await rig.texts.put("hello")
    await rig.until(lambda: len(holds_heard(rig)) == 2)
    assert [hold.levels for hold in holds_heard(rig)] == [Levels(None, None), Levels(-20.0, -20.0)]


async def test_a_turn_is_received_once_its_words_are_written_and_a_turn_with_none_is_not(rig: Rig) -> None:
    await rig.hold(["down", "up"])
    await rig.texts.put("what time is it")
    await rig.until(lambda: rig.out.sent == ["what time is it"])
    await rig.until(lambda: len(rig.received) == 1)
    # Heard nothing, then dropped: the key ended both, and neither reached the model.
    await rig.hold(["down", "up"])
    await rig.texts.put("")
    await rig.hold(["down", "down", "dropped"])
    assert await rig.everything_sent(holds=3) == ["what time is it"]
    await rig.until(lambda: len(rig.received) == 2)  # the closing hold's
    await asyncio.sleep(0.05)
    assert len(rig.received) == 2


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


async def test_an_engaged_turn_keeps_the_second_the_desk_heard_before_it_and_listens_on_after(rig: Rig) -> None:
    # Pipecat keeps the last second of audio nobody is speaking in: 50 frames of 20 ms. The desk listens for 1.6 s
    # while the room is quiet, then the user's first word comes as the detector makes sure of it.
    room, onset, said, after = (bytes([n, n]) * 320 for n in (1, 2, 3, 4))
    listening: list[Key] = ["listening"]
    await rig.hold(listening * 70, sound=room)
    await rig.hold(listening * 5, sound=onset)
    await rig.hold(["arming", "down", "down"], sound=said)
    await rig.hold(["listening"], sound=after)
    await rig.texts.put("what time is it")
    assert await rig.everything_sent(holds=1) == ["what time is it"]
    assert rig.heard[0].startswith(room * 45 + onset * 5 + said * 3)
    assert after not in rig.heard[0]


async def test_an_engaged_start_that_was_only_noise_opens_no_turn_and_the_desk_listens_on(rig: Rig) -> None:
    noise, said = (bytes([n, n]) * 320 for n in (1, 3))
    listening: list[Key] = ["listening"]
    await rig.hold(["listening", "arming", "arming", "listening"], sound=noise)
    await rig.hold(listening * 60)
    await rig.hold(["arming", "down", "listening"], sound=said)
    await rig.texts.put("what time is it")
    assert await rig.everything_sent(holds=1) == ["what time is it"]
    assert noise not in rig.heard[0]


async def test_a_phone_turn_opened_while_the_desk_listens_hears_nothing_the_desk_heard(rig: Rig) -> None:
    room, said = (bytes([n, n]) * 320 for n in (1, 3))
    listening: list[Key] = ["listening"]
    await rig.hold(listening * 20, sound=room)
    await rig.hold(["down", "down", "up"], sound=said, at="phone")
    await rig.texts.put("what time is it")
    assert await rig.everything_sent(holds=1) == ["what time is it"]
    assert room not in rig.heard[0]


async def test_a_turn_ended_and_the_next_armed_between_two_frames_is_sent(rig: Rig) -> None:
    # The verdict ends an engaged turn and the voice arms the next before the microphone captures a frame at rest.
    said, more = (bytes([n, n]) * 320 for n in (3, 5))
    turn = Gate().after("listen", "desk").after("arm", "desk").after("start", "desk")
    for _ in range(3):
        await rig.capture(turn, said)
    await rig.capture(turn.after("stop", "desk").after("arm", "desk"), more)
    await rig.texts.put("what time is it")
    assert await rig.everything_sent(holds=1) == ["what time is it"]
    assert rig.heard[0].startswith(said * 3) and more not in rig.heard[0]


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
def waiting() -> Unprompted:
    return Unprompted(SessionGone(API), (heard(),))


def asking(session: SessionId, request: str) -> Unprompted:
    return Unprompted(Narrate(Asking(session, RequestId(request), Permission("Bash", {"command": "ls"}))), (heard(),))


def finished(session: SessionId, reply: str) -> Unprompted:
    return Unprompted(Finished(session, (News(None, reply, "", "", (), frozenset()),), "full"), (heard(),))


def let_go(unprompted: Unprompted) -> dict[str, Fact]:
    """What the floor found of the one thing hands heard that `unprompted` tells, and its fate where the floor settled it."""
    [utterance] = unprompted.utterances
    settled: dict[str, Fact] = {"fate": utterance.settled.result()} if utterance.settled.done() else {}
    return {key: value for key, value in utterance.facts.items() if key in ("held_ms", "told", "folded")} | settled


async def test_a_session_waiting_while_the_key_is_held_is_said_after_the_users_turn_is_sent_and_not_before(rig: Rig) -> None:
    rig.clock.now = 10.0
    await rig.hold(["down", "down"])
    await rig.until(lambda: rig.out.started == 1)
    gone = waiting()
    await rig.worker.queue_frames([gone, TextFrame("marker")])
    # Frames keep their order, so the marker past the floor with the announcement not is the announcement held.
    await rig.until(lambda: "marker" in rig.out.order)
    rig.clock.now = 13.5
    await rig.hold(["up"])
    await rig.texts.put("what time is it")
    await rig.until(lambda: "said: The session api is gone." in rig.out.order)
    assert rig.out.order == ["started", "marker", "stopped", "sent: what time is it", "said: The session api is gone."]
    assert let_go(gone) == {"held_ms": 3500.0, "told": "SessionGone", "folded": 1}


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


def user_texts(request: dict[str, object]) -> list[str]:
    """Each text a request's user messages hold, in order, in either API's shape: a string, or blocks of text."""
    texts: list[str] = []
    for message in cast(list[dict[str, str | list[dict[str, str]]]], request["messages"]):
        match message:
            case {"role": "user", "content": str() as text}:
                texts.append(text)
            case {"role": "user", "content": list() as blocks}:
                texts.extend(block["text"] for block in blocks)
            case _:
                pass
    return texts


AUDIO_ONLY = "[hands] The user is audio-only: they cannot see a screen."
IN_FRONT = '[hands] As the user said this, iTerm2 was in front on the Mac\'s screen, showing the session "api" (id api). Say nothing about this unless it bears on what they said.'


async def api_in_front() -> InFront:
    return SessionInFront("iTerm2", API, "api")


@dataclass
class Served:
    """The rig asking a served model, each turn of the user's noted on its way: every request the model was made, and
    each note's event."""

    rig: Rig
    requests: list[dict[str, object]]
    noted: list[Entry]


@asynccontextmanager
async def served(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api_server: ServeApi, shape: Shape, front: Callable[[], Awaitable[InFront]]) -> AsyncGenerator[Served]:
    requests: list[dict[str, object]] = []

    def answering(stream: list[str]) -> Endpoint:
        async def answer(request: web.Request) -> web.StreamResponse:
            requests.append(await request.json())
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for event in stream:
                await response.write(event.encode())
            return response

        return answer

    api = await api_server(answering(openai_stream("words")), answering(anthropic_stream("words")))
    backend = OpenAICompatibleBackend(base_url=api.url, api_key="k", model="m") if shape == "openai" else AnthropicBackend(base_url=api.anthropic_url, api_key="k", model="m")
    llm = built.build_llm(backend, instruction="Speak.", max_tokens=50)
    noted: list[Entry] = []
    async with rigged(monkeypatch, tmp_path, llm, [Noting(front, lambda: "audio-only", noted.append)], [llm]) as rig:
        yield Served(rig, requests, noted)


@pytest.mark.parametrize("shape", ["openai", "anthropic"])
async def test_an_api_models_request_carries_hands_notes_beside_the_users_words_and_beside_nothing_hands_tells(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api_server: ServeApi, shape: Shape
) -> None:
    async with served(monkeypatch, tmp_path, api_server, shape, api_in_front) as asked:
        rig, requests = asked.rig, asked.requests
        await rig.hold(["down", "up"])
        await rig.texts.put("what time is it")
        await rig.until(lambda: len(requests) == 1)
        # What hands tells of a session is no turn of the user's: it is asked with no note of its own.
        rig.live[API] = waiting_on(API, "r1")
        await rig.worker.queue_frame(asking(API, "r1"))
        await rig.until(lambda: len(requests) == 2)
    note = f"{IN_FRONT}\n\n{AUDIO_ONLY}"
    assert user_texts(requests[0]) == ["what time is it", note]
    words, noted_once, told = user_texts(requests[1])
    assert (words, noted_once) == ("what time is it", note) and told.startswith("[hands] The Claude Code session api is waiting for permission")
    # [LAW:nothing-unseen] the one turn noted is one event: what was read, and where the user was.
    [noted] = events(asked.noted, "voice.noted")
    assert noted.outcome == "ok" and noted.facts == {"front": SessionInFront("iTerm2", API, "api"), "modality": "audio-only"}


async def test_a_turn_whose_screen_could_not_be_read_is_asked_with_where_the_user_is_alone_and_its_event_says_why(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api_server: ServeApi
) -> None:
    unread = FrontUnread("the screen was not read in this test")

    async def front() -> InFront:
        return unread

    async with served(monkeypatch, tmp_path, api_server, "anthropic", front) as asked:
        await asked.rig.hold(["down", "up"])
        await asked.rig.texts.put("what time is it")
        await asked.rig.until(lambda: len(asked.requests) == 1)
    assert user_texts(asked.requests[0]) == ["what time is it", AUDIO_ONLY]
    [noted] = events(asked.noted, "voice.noted")
    assert noted.outcome == "ok" and noted.facts == {"front": unread, "modality": "audio-only"}


async def test_a_key_pressed_while_the_screen_is_read_drops_that_ask_and_the_next_turn_asks_both_with_one_note(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api_server: ServeApi
) -> None:
    reading = asyncio.Event()

    async def front() -> InFront:
        # The first read never returns: the user's next press is what ends it.
        if not reading.is_set():
            reading.set()
            await asyncio.Event().wait()
        return await api_in_front()

    async with served(monkeypatch, tmp_path, api_server, "anthropic", front) as asked:
        rig = asked.rig
        await rig.hold(["down", "up"])
        await rig.texts.put("what time is it")
        await reading.wait()
        await rig.hold(["down"])
        await rig.until(lambda: len(events(asked.noted, "voice.noted")) == 1)
        await rig.hold(["up"])
        await rig.texts.put("in Lisbon")
        await rig.until(lambda: len(events(asked.noted, "voice.noted")) == 2)
        await rig.until(lambda: len(asked.requests) == 1)
    # The model was never asked the first turn alone: its words are asked with the next, noted once, as that one arrived.
    assert [user_texts(request) for request in asked.requests] == [["what time is it", "in Lisbon", f"{IN_FRONT}\n\n{AUDIO_ONLY}"]]
    assert [noted.outcome for noted in events(asked.noted, "voice.noted")] == ["cancelled", "ok"]


async def test_two_sessions_finishing_during_one_held_key_are_told_asks_first_and_one_telling_a_session(rig: Rig) -> None:
    """What a session waits on the user for is told before what a session did; a session's turns that finished while
    the user talked are one telling; and a request answered at the keyboard meanwhile is not told at all."""
    rig.live[WEB] = waiting_on(WEB, "w2")
    await rig.hold(["down", "down"])
    await rig.until(lambda: rig.out.started == 1)
    # web's first request was answered at the keyboard while the key was down; its second still waits.
    held = [finished(API, "Fixed the parser."), asking(WEB, "w1"), finished(API, "Pushed it."), asking(WEB, "w2")]
    rig.clock.now = 2.0
    await rig.worker.queue_frames([*held, TextFrame("marker")])
    await rig.until(lambda: "marker" in rig.out.order)
    # Each is held from when it came, not from the press.
    rig.clock.now = 5.0
    await rig.hold(["up"])
    await rig.texts.put("what time is it")
    await rig.until(lambda: len(rig.out.context) == 3)
    said, asked, told = rig.out.context
    assert said == "what time is it"
    assert asked.startswith("[hands] The Claude Code session web is waiting for permission") and "Request id: w2." in asked
    assert told.startswith("[hands] The Claude Code session api (id api) finished 2 turns. The last thing it said was:\n\nFixed the parser.\n\nThen, in the turn after that: The last thing it said was:\n\nPushed it.")
    # Each thing heard is followed to its fate: both turns told in one telling, and the request answered meanwhile dropped.
    assert [let_go(each) for each in held] == [
        {"held_ms": 3000.0, "told": "Finished", "folded": 2},
        {"held_ms": 3000.0, "fate": "dropped"},
        {"held_ms": 3000.0, "told": "Finished", "folded": 2},
        {"held_ms": 3000.0, "told": "Narrate", "folded": 1},
    ]


async def test_a_hold_that_was_shift_after_all_gives_back_what_waited_for_it(rig: Rig) -> None:
    await rig.hold(["down"])
    await rig.until(lambda: rig.out.started == 1)
    await rig.worker.queue_frame(waiting())
    await rig.hold(["dropped"])
    await rig.until(lambda: "said: The session api is gone." in rig.out.order)
    assert rig.out.order == ["started", "stopped", "said: The session api is gone."]


async def test_what_hands_says_with_no_turn_open_passes_at_once_and_records_no_wait(rig: Rig) -> None:
    gone = waiting()
    await rig.worker.queue_frame(gone)
    await rig.until(lambda: rig.out.order == ["said: The session api is gone."])
    assert let_go(gone) == {"held_ms": 0.0, "told": "SessionGone", "folded": 1}


async def test_a_request_answered_before_it_reaches_the_floor_is_not_told(rig: Rig) -> None:
    answered = asking(API, "gone")
    await rig.worker.queue_frames([answered, TextFrame("marker")])
    await rig.until(lambda: "marker" in rig.out.order)
    assert rig.out.order == ["marker"]
    assert let_go(answered) == {"held_ms": 0.0, "fate": "dropped"}


async def test_whisper_hears_only_the_keyed_microphone() -> None:
    whisper = Whisper(url="http://unused/v1", prompt=unprimed, record=lambda _: None)
    with pytest.raises(TypeError, match="carries no key"):
        await whisper.process_audio_frame(InputAudioRawFrame(b"\x00\x00", 16000, 1), FrameDirection.DOWNSTREAM)
