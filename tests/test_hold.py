"""The talk key: Right Shift held alone opens a turn, its release sends it, and any other key is typing."""

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from pipecat.frames.frames import Frame, InputAudioRawFrame, TextFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.whisper.stt import WhisperSTTServiceMLX

from hands.daemon import cli
from hands.voice import keys, talkkey
from hands.voice.hold import Hold, Idle, KeyEvent, Move, Pressed, Released, Ripe, Typed, step
from hands.voice.ptt import Key, KeyedAudio
from hands.voice.turnstop import KeyTurnStop, TurnUnheard
from hands.voice.whisper import Whisper


def moves(events: Sequence[KeyEvent]) -> list[Move]:
    hold: Hold = Idle()
    made: list[Move] = []
    for event in events:
        hold, now = step(hold, event)
        made.extend(now)
    return made


@pytest.mark.parametrize(
    ("events", "made"),
    [
        # Held alone past the hold, then released: a whole turn, sent.
        ([Pressed(1.0), Ripe(1.0), Released()], ["start", "stop"]),
        # A quick tap is nothing.
        ([Pressed(1.0), Released(), Ripe(1.0)], []),
        # A capital typed with Right Shift never starts a turn, however long the key is then held.
        ([Pressed(1.0), Typed(), Ripe(1.0), Released()], []),
        # A key pressed during a started turn drops it, and the release after sends nothing.
        ([Pressed(1.0), Ripe(1.0), Typed(), Released()], ["start", "drop"]),
        # A Ripe left over from an earlier press does not open the press after it; its own Ripe does.
        ([Pressed(1.0), Released(), Pressed(1.1), Ripe(1.0), Released()], []),
        ([Pressed(1.0), Released(), Pressed(1.1), Ripe(1.0), Ripe(1.1), Released()], ["start", "stop"]),
        # Typing with Right Shift up is nothing to the turn.
        ([Typed(), Typed()], []),
        # A press that finds the key already down proves a release went by unseen: an open turn is dropped, and the
        # press starts afresh, from Typing too.
        ([Pressed(1.0), Ripe(1.0), Pressed(2.0), Ripe(2.0), Released()], ["start", "drop", "start", "stop"]),
        ([Pressed(1.0), Typed(), Pressed(2.0), Ripe(2.0), Released()], ["start", "stop"]),
        # After a dropped turn the key works again from the next press.
        ([Pressed(1.0), Ripe(1.0), Typed(), Released(), Pressed(2.0), Ripe(2.0), Released()], ["start", "drop", "start", "stop"]),
    ],
)
def test_the_hold_moves_the_turn(events: list[KeyEvent], made: list[Move]) -> None:
    assert moves(events) == made


LEFT_SHIFT_ONLY = talkkey.SHIFT | 0x2  # NX_DEVICELSHIFTKEYMASK


@pytest.mark.parametrize(
    ("kind", "keycode", "flags", "event"),
    [
        (talkkey.FLAGS_CHANGED, talkkey.RIGHT_SHIFT, talkkey.SHIFT | talkkey.RIGHT_SHIFT_DOWN, Pressed(5.0)),
        # Right Shift let go while Left Shift is still held: the Shift flag stays, Right Shift's own bit goes.
        (talkkey.FLAGS_CHANGED, talkkey.RIGHT_SHIFT, LEFT_SHIFT_ONLY, Released()),
        (talkkey.FLAGS_CHANGED, talkkey.RIGHT_SHIFT, 0, Released()),
        (talkkey.FLAGS_CHANGED, 56, LEFT_SHIFT_ONLY, Typed()),  # Left Shift
        (talkkey.KEY_DOWN, 0, talkkey.RIGHT_SHIFT_DOWN, Typed()),  # A, as Shift+A
        # Right Shift pressed inside a chord, Cmd already down (Cmd+Shift+4), is Shift, not talk.
        (talkkey.FLAGS_CHANGED, talkkey.RIGHT_SHIFT, talkkey.SHIFT | talkkey.RIGHT_SHIFT_DOWN | 0x100000, Typed()),
        (talkkey.FLAGS_CHANGED, talkkey.RIGHT_SHIFT, talkkey.SHIFT | talkkey.RIGHT_SHIFT_DOWN | 0x2, Typed()),
        # A shift-click and a shift-scroll are Shift too.
        *((kind, 0, talkkey.SHIFT | talkkey.RIGHT_SHIFT_DOWN, Typed()) for kind in talkkey.WATCHED_KINDS[2:]),
    ],
)
def test_each_tapped_event_is_what_it_means_to_the_hold(kind: int, keycode: int, flags: int, event: KeyEvent) -> None:
    assert talkkey.event_of(kind, keycode, flags, 5.0) == event


async def test_the_talk_key_opens_a_turn_once_held_and_sends_it_on_release(monkeypatch: pytest.MonkeyPatch) -> None:
    taps: list[Callable[[KeyEvent], None]] = []
    stopped: list[None] = []

    def tap(heard: Callable[[KeyEvent], None]) -> Callable[[], None]:
        taps.append(heard)
        return lambda: stopped.append(None)

    monkeypatch.setattr(talkkey, "tap", tap)
    monkeypatch.setattr(keys, "HOLD_SECONDS", 0.05)
    made: list[Move] = []
    started = asyncio.Event()

    async def on_move(move: Move) -> None:
        made.append(move)
        started.set()

    driving = asyncio.create_task(keys.drive_talk_key(on_move))
    while not taps:
        await asyncio.sleep(0)
    taps[0](Pressed(1.0))
    await asyncio.wait_for(started.wait(), 1.0)
    taps[0](Released())
    while len(made) < 2:
        await asyncio.sleep(0.01)
    driving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await driving
    assert made == ["start", "stop"]
    assert stopped == [None]


def keyed(key: Key) -> KeyedAudio:
    return KeyedAudio(audio=b"\x7f\x7f" * 320, sample_rate=16000, num_channels=1, key=key)


@pytest.mark.parametrize(("ended", "transcribed", "unheard"), [("up", 1, 0), ("dropped", 0, 1)])
async def test_whisper_cuts_turns_where_the_captured_key_moved(
    ended: Key, transcribed: int, unheard: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    whisper = Whisper(settings=WhisperSTTServiceMLX.Settings(model="unused"))
    pushed: list[Frame] = []

    async def push(frame: Frame, _direction: object = None) -> None:
        pushed.append(frame)

    monkeypatch.setattr(whisper, "push_frame", push)
    for key in ("up", "down", "down", ended):
        await whisper.process_audio_frame(keyed(key), FrameDirection.DOWNSTREAM)
    # The VAD's own word, which reads the key later than the frames were captured, moves nothing.
    await whisper.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    # A sent turn is a segment for transcription; a dropped one is none, and is said at once to have no text.
    assert whisper._segment_queue.qsize() == transcribed  # pyright: ignore[reportPrivateUsage]
    assert sum(isinstance(frame, TurnUnheard) for frame in pushed) == unheard


async def test_whisper_hears_only_the_keyed_microphone() -> None:
    whisper = Whisper(settings=WhisperSTTServiceMLX.Settings(model="unused"))
    with pytest.raises(TypeError, match="carries no key"):
        await whisper.process_audio_frame(InputAudioRawFrame(b"\x00\x00", 16000, 1), FrameDirection.DOWNSTREAM)


@pytest.mark.parametrize("vad_first", [True, False])
async def test_a_turn_with_no_text_ends_once_whisper_and_the_vad_both_say_so(vad_first: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    stop = KeyTurnStop(user_speech_timeout=0.0)
    ended: list[None] = []

    async def trigger(**_: object) -> None:
        ended.append(None)

    monkeypatch.setattr(stop, "trigger_user_turn_stopped", trigger)
    await stop.process_frame(TextFrame("any other frame"))
    assert ended == []
    # Whisper cuts by the key each frame was captured under and the VAD by the key when it reads a frame, so either
    # may come first; the turn ends on the second, and only once.
    stop._vad_user_speaking = not vad_first  # pyright: ignore[reportPrivateUsage]
    await stop.process_frame(TurnUnheard())
    assert ended == ([None] if vad_first else [])
    stop._vad_user_speaking = False  # pyright: ignore[reportPrivateUsage]
    await stop.process_frame(TextFrame("the next frame after the VAD stopped"))
    assert ended == [None]


def test_a_run_without_the_input_monitoring_grant_is_refused_at_the_door(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    asked: list[None] = []
    monkeypatch.setattr(talkkey, "granted", lambda: False)
    monkeypatch.setattr(talkkey, "ask", lambda: asked.append(None))
    assert cli.main(["--home", str(tmp_path), "run"]) == 1
    assert "Input Monitoring is not granted" in capsys.readouterr().err
    assert asked == [None]
    # Refused before the first heartbeat: nothing says a run is starting, so nothing later reads as a crash.
    assert not (tmp_path / "status.json").exists()
