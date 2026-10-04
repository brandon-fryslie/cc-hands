"""The wake word: saying it opens a turn with no key, end-of-turn detection closes it, and while hands speaks the
detector hears silence."""

import asyncio
import subprocess
import wave
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.metrics.metrics import TurnMetricsData

from hands.sessions.wide import WideEvent, unit
from hands.voice.engaged import Act, Begun, Engagement, Event, SpeechStarted, SpeechStarting, SpeechStopped, TurnEnded, TurnTooLong, Woken, drive, untapped
from hands.voice.hold import Move, Pressed, Released, Ripe
from hands.voice.engaged import loaded as ears_loaded
from hands.voice.wake import EMBEDDING, MELSPECTROGRAM, SAMPLE_RATE, WAKE, WORD, WakeWord, fetched, listening, loaded, step


def acts(events: Sequence[Event]) -> list[Act]:
    engagement = Engagement()
    made: list[Act] = []
    for event in (Begun(), *events):
        engagement, now = step(engagement, event)
        made.extend(now)
    return made


@pytest.mark.parametrize(
    ("events", "made"),
    [
        # The desk listens from the start; speech alone opens nothing.
        ([SpeechStarting(), SpeechStarted(2.0), SpeechStopped(), TurnEnded("verdict")], ["listen"]),
        # The wake word opens the turn; the pause after it ends nothing; what is asked starts, stops, and is judged, and
        # the verdict sends it.
        ([Woken(2.0), SpeechStopped(), SpeechStarting(), SpeechStarted(2.5), SpeechStopped(), TurnEnded("verdict")], ["listen", "arm", "start", "judge", "stop"]),
        # Before what is asked has started, a stop or silence ends nothing: it is the end of the wake word.
        ([Woken(2.0), SpeechStopped(), TurnEnded("silence")], ["listen", "arm", "start"]),
        # The wake word said again inside a turn is part of the turn.
        ([Woken(2.0), SpeechStarted(2.5), Woken(3.0), SpeechStopped(), TurnEnded("verdict")], ["listen", "arm", "start", "judge", "stop"]),
        # Turns follow one another, each opened by the wake word.
        ([Woken(2.0), SpeechStarted(2.5), TurnEnded("silence"), SpeechStarted(4.0), Woken(5.0), SpeechStarted(5.5), TurnEnded("silence")], ["listen", "arm", "start", "stop", "arm", "start", "stop"]),
        # A turn open past the limit is thrown away, whether or not what is asked has started; a limit set for an
        # earlier turn ends nothing.
        ([Woken(2.0), TurnTooLong(2.0)], ["listen", "arm", "start", "expire"]),
        ([Woken(2.0), SpeechStarted(2.5), TurnTooLong(2.0)], ["listen", "arm", "start", "expire"]),
        ([Woken(2.0), SpeechStarted(2.5), TurnEnded("silence"), Woken(4.0), TurnTooLong(2.0)], ["listen", "arm", "start", "stop", "arm", "start"]),
        # The talk key does nothing.
        ([Pressed(1.0), Ripe(1.0), Released(), Woken(2.0)], ["listen", "arm", "start"]),
    ],
)
def test_what_the_edge_does(events: list[Event], made: list[Act]) -> None:
    assert acts(events) == made


class ScriptedEars:
    """Silero and Smart Turn as a script: each buffer names the detector's state, and every verdict is complete."""

    async def detect(self, audio: bytes) -> VADState:
        return {b"q": VADState.QUIET, b"S": VADState.SPEAKING, b"w": VADState.SPEAKING}[audio]

    def heard(self, audio: bytes, speech: bool) -> bool:
        return False

    async def judge(self) -> tuple[bool, TurnMetricsData | None]:
        return True, TurnMetricsData(processor="test", is_complete=True, probability=0.9, e2e_processing_time_ms=40.0)

    def clear(self) -> None:
        pass

    def afresh(self) -> None:
        pass


async def test_the_wake_word_opens_turns_with_no_key_and_a_switch_away_stops_the_desk_listening() -> None:
    audio: asyncio.Queue[bytes] = asyncio.Queue()
    made: list[Move] = []
    events: list[WideEvent] = []

    @asynccontextmanager
    async def overheard() -> AsyncGenerator[AsyncIterator[bytes]]:
        async def buffers() -> AsyncIterator[bytes]:
            while True:
                yield await audio.get()

        yield buffers()

    async def woken(heard: bytes) -> bool:
        return heard == b"w"

    async def on_move(move: Move) -> None:
        made.append(move)

    driving = asyncio.create_task(drive(WAKE, untapped, overheard, ScriptedEars(), woken, on_move, events.append))
    # Talk in the room opens nothing; the wake word opens a turn, the pause after it ends nothing, and the verdict on
    # the stop after what was asked sends it; and again.
    for each in (b"S", b"q", b"w", b"q", b"S", b"q", b"S", b"q", b"w", b"q", b"S", b"q"):
        audio.put_nowait(each)
    async with asyncio.timeout(1.0):
        while len(made) < 7:
            await asyncio.sleep(0.001)
    driving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await driving
    assert made == ["listen", "arm", "start", "stop", "arm", "start", "stop", "deafen"]
    awake = [event for event in events if event.event == "trigger.awake"]
    assert [(event.outcome, dict(event.counts)["start"], dict(event.counts)["stop"], dict(event.counts)["deafen"]) for event in awake] == [("cancelled", 2, 2, 1)]


class FakeWord(WakeWord):
    """The wake word's model as a script: it is sure of the wake word in any buffer that is not silence."""

    def __init__(self) -> None:
        self.heard: list[bytes] = []
        self.resets = 0

    def score(self, audio: bytes) -> float:
        self.heard.append(audio)
        return 0.0 if audio.count(0) == len(audio) else 0.9

    def reset(self) -> None:
        self.resets += 1


def test_while_hands_speaks_the_detector_hears_silence_and_each_waking_is_an_event() -> None:
    word = FakeWord()
    speaking = [True]
    events: list[WideEvent] = []
    woken = listening(word, lambda: speaking[0], events.append)

    async def heard() -> list[bool]:
        with unit("trigger.awake", events.append, counts=("woken", "muted")):
            deaf = await woken(b"\x01\x02" * 160)
            speaking[0] = False
            return [deaf, await woken(b"\x01\x02" * 160)]

    assert asyncio.run(heard()) == [False, True]
    assert word.heard == [bytes(320), b"\x01\x02" * 160]
    assert word.resets == 1
    assert [(event.event, event.facts.get("score")) for event in events] == [("trigger.woken", 0.9), ("trigger.awake", None)]
    assert dict(events[1].counts) == {"woken": 1, "muted": 1}
    assert events[0].parent_id == events[1].span_id


def said(text: str, at: Path) -> bytes:
    """`text` spoken by macOS's own voice, as 16 kHz mono 16-bit audio with a second of silence either side."""
    subprocess.run(["say", "-v", "Samantha", "-o", str(at), f"--data-format=LEI16@{SAMPLE_RATE}", text], check=True)
    with wave.open(str(at)) as clip:
        audio = clip.readframes(clip.getnframes())
    return bytes(SAMPLE_RATE * 2) + audio + bytes(SAMPLE_RATE * 2)


@pytest.fixture
async def models(pytestconfig: pytest.Config) -> Path:
    """The wake word's models, fetched from openWakeWord's release once and kept in pytest's cache."""
    directory = pytestconfig.cache.mkdir("wake-word") if pytestconfig.cache else pytest.fail("the cache provider is off")
    await fetched(directory)
    return directory


async def test_the_model_wakes_on_hey_jarvis_and_not_on_jarvis_named_in_passing(tmp_path: Path, models: Path) -> None:
    events: list[WideEvent] = []
    async with loaded(SAMPLE_RATE, models, events.append) as word:
        # The desk's microphone hands over 20 ms at a time.
        def wakes(audio: bytes) -> bool:
            word.reset()
            return any(word.score(audio[start : start + 640]) >= 0.5 for start in range(0, len(audio), 640))

        assert wakes(said("Hey Jarvis. What is the status of the build?", tmp_path / "wake.wav"))
        assert not wakes(said("I was talking with Jarvis earlier about the build, and it was fine.", tmp_path / "passing.wav"))
    assert [(event.event, event.outcome) for event in events] == [("trigger.wake_word_loaded", "ok")]


async def test_a_microphone_at_another_rate_is_refused_out_loud() -> None:
    events: list[WideEvent] = []
    with pytest.raises(ValueError, match="16000 Hz"):
        async with loaded(48_000, Path("unread"), events.append):
            pass
    assert [(event.event, event.outcome) for event in events] == [("trigger.wake_word_loaded", "failed")]


@pytest.mark.parametrize(
    "text",
    [
        "Hey Jarvis what is the status of the build",
        "Hey Jarvis, what is the status of the build?",
        "Hey Jarvis. [[slnc 1500]] What is the status of the build?",
        "Hey Jarvis stop",
    ],
)
async def test_what_is_asked_after_the_wake_word_is_judged_whether_asked_in_one_breath_or_after_a_pause(text: str, tmp_path: Path, models: Path) -> None:
    """Silero, Smart Turn, and the wake word's model, real, on macOS's own voice: Smart Turn's verdict on what was asked
    sends the turn, not the silence after it, and not the pause after the wake word."""
    audio = said(text, tmp_path / "asked.wav") + bytes(SAMPLE_RATE * 2 * 4)
    made: list[Move] = []
    events: list[WideEvent] = []

    @asynccontextmanager
    async def overheard() -> AsyncGenerator[AsyncIterator[bytes]]:
        async def buffers() -> AsyncIterator[bytes]:
            for start in range(0, len(audio), 640):
                yield audio[start : start + 640]
            await asyncio.Event().wait()

        yield buffers()

    async def on_move(move: Move) -> None:
        made.append(move)

    async with ears_loaded(SAMPLE_RATE, events.append) as ears, loaded(SAMPLE_RATE, models, events.append) as word:
        driving = asyncio.create_task(drive(WAKE, untapped, overheard, ears, listening(word, lambda: False, events.append), on_move, events.append))
        async with asyncio.timeout(60.0):
            while "stop" not in made:
                await asyncio.sleep(0.01)
        driving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await driving
    assert made == ["listen", "arm", "start", "stop", "deafen"]
    [awake] = [event for event in events if event.event == "trigger.awake"]
    assert dict(awake.counts)["ended_on_silence"] == 0
    assert [event.facts["complete"] for event in events if event.event == "trigger.judged"] == [True]


async def test_the_models_are_fetched_whole_or_not_at_all_and_never_twice(tmp_path: Path) -> None:
    served: list[str] = []

    async def model(request: web.Request) -> web.StreamResponse:
        name = request.match_info["name"]
        served.append(name)
        match name:
            case "embedding_model.onnx":
                raise web.HTTPNotFound()
            case "hey_jarvis_v0.1.onnx":
                # The connection drops with the body half sent.
                response = web.StreamResponse(headers={"Content-Length": "100"})
                await response.prepare(request)
                await response.write(b"half")
                request.transport.close() if request.transport else None
                return response
            case _:
                return web.Response(body=b"features")

    app = web.Application()
    app.router.add_get("/{name}", model)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        with pytest.raises(aiohttp.ClientResponseError):
            await fetched(tmp_path, f"http://127.0.0.1:{port}")
        assert sorted(path.name for path in tmp_path.iterdir()) == [MELSPECTROGRAM]
        (tmp_path / EMBEDDING).write_bytes(b"features")
        with pytest.raises(aiohttp.ClientPayloadError):
            await fetched(tmp_path, f"http://127.0.0.1:{port}")
        assert not (tmp_path / WORD).exists()
        (tmp_path / WORD).write_bytes(b"word")
        assert await fetched(tmp_path, f"http://127.0.0.1:{port}") == ()
        assert served == [MELSPECTROGRAM, EMBEDDING, WORD]
    finally:
        await runner.cleanup()
