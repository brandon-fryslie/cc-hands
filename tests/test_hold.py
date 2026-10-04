"""The talk key: Right Shift held alone opens a turn, its release sends it, and any other key is typing."""

import asyncio
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from hands.daemon import cli
from hands.sessions import audit
from hands.sessions.home import Home
from hands.voice import keys, talkkey
from hands.voice.hold import Hold, Idle, KeyEvent, Move, Overlong, Pressed, Released, Ripe, Typed, step


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
        ([Pressed(1.0), Ripe(1.0), Released()], ["arm", "start", "stop"]),
        # A quick tap opens the microphone and closes it again: no turn.
        ([Pressed(1.0), Released(), Ripe(1.0)], ["arm", "disarm"]),
        # A capital typed with Right Shift never starts a turn, however long the key is then held.
        ([Pressed(1.0), Typed(), Ripe(1.0), Released()], ["arm", "disarm"]),
        # A key pressed during a started turn drops it, and the release after sends nothing.
        ([Pressed(1.0), Ripe(1.0), Typed(), Released()], ["arm", "start", "drop"]),
        # A Ripe left over from an earlier press does not open the press after it; its own Ripe does.
        ([Pressed(1.0), Released(), Pressed(1.1), Ripe(1.0), Released()], ["arm", "disarm", "arm", "disarm"]),
        ([Pressed(1.0), Released(), Pressed(1.1), Ripe(1.0), Ripe(1.1), Released()], ["arm", "disarm", "arm", "start", "stop"]),
        # Typing with Right Shift up is nothing to the turn.
        ([Typed(), Typed()], []),
        # A press that finds the key already down proves a release went by unseen: an open turn is dropped, and the
        # press starts afresh, from Typing too.
        ([Pressed(1.0), Ripe(1.0), Pressed(2.0), Ripe(2.0), Released()], ["arm", "start", "drop", "arm", "start", "stop"]),
        ([Pressed(1.0), Typed(), Pressed(2.0), Ripe(2.0), Released()], ["arm", "disarm", "arm", "start", "stop"]),
        # A turn open past the limit is thrown away, and the release after it, whenever it comes, sends nothing.
        ([Pressed(1.0), Ripe(1.0), Overlong(1.0), Released()], ["arm", "start", "expire"]),
        # The limit counts only for the press it was scheduled for, and a turn already sent has nothing left to expire.
        ([Pressed(1.0), Released(), Pressed(2.0), Ripe(2.0), Overlong(1.0), Released()], ["arm", "disarm", "arm", "start", "stop"]),
        ([Pressed(1.0), Ripe(1.0), Released(), Overlong(1.0)], ["arm", "start", "stop"]),
        # After a dropped turn the key works again from the next press.
        ([Pressed(1.0), Ripe(1.0), Typed(), Released(), Pressed(2.0), Ripe(2.0), Released()], ["arm", "start", "drop", "arm", "start", "stop"]),
    ],
)
def test_the_hold_moves_the_turn(events: list[KeyEvent], made: list[Move]) -> None:
    assert moves(events) == made


KEY_DOWN = talkkey.WATCHED_KINDS[0]
SHIFT = 0x20000  # kCGEventFlagMaskShift, set by either Shift
LEFT_SHIFT_ONLY = SHIFT | 0x2  # NX_DEVICELSHIFTKEYMASK


@pytest.mark.parametrize(
    ("kind", "keycode", "flags", "event"),
    [
        (talkkey.FLAGS_CHANGED, talkkey.RIGHT_SHIFT, SHIFT | talkkey.RIGHT_SHIFT_DOWN, Pressed(5.0)),
        # Right Shift let go while Left Shift is still held: the Shift flag stays, Right Shift's own bit goes.
        (talkkey.FLAGS_CHANGED, talkkey.RIGHT_SHIFT, LEFT_SHIFT_ONLY, Released()),
        (talkkey.FLAGS_CHANGED, talkkey.RIGHT_SHIFT, 0, Released()),
        (talkkey.FLAGS_CHANGED, 56, LEFT_SHIFT_ONLY, Typed()),  # Left Shift
        (KEY_DOWN, 0, talkkey.RIGHT_SHIFT_DOWN, Typed()),  # A, as Shift+A
        # Right Shift pressed inside a chord, Cmd already down (Cmd+Shift+4), is Shift, not talk.
        (talkkey.FLAGS_CHANGED, talkkey.RIGHT_SHIFT, SHIFT | talkkey.RIGHT_SHIFT_DOWN | 0x100000, Typed()),
        (talkkey.FLAGS_CHANGED, talkkey.RIGHT_SHIFT, SHIFT | talkkey.RIGHT_SHIFT_DOWN | 0x2, Typed()),
        # A shift-click and a shift-scroll are Shift too.
        *((kind, 0, SHIFT | talkkey.RIGHT_SHIFT_DOWN, Typed()) for kind in talkkey.WATCHED_KINDS[2:]),
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
        if move == "start":
            started.set()

    driving = asyncio.create_task(keys.drive_talk_key(on_move))
    while not taps:
        await asyncio.sleep(0)
    taps[0](Pressed(asyncio.get_running_loop().time()))  # dated as the tap dates it, on the loop's clock
    await asyncio.wait_for(started.wait(), 1.0)
    taps[0](Released())
    while len(made) < 3:
        await asyncio.sleep(0.01)
    driving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await driving
    assert made == ["arm", "start", "stop"]
    assert stopped == [None]


async def test_a_turn_held_past_the_limit_is_thrown_away_without_a_release(monkeypatch: pytest.MonkeyPatch) -> None:
    taps: list[Callable[[KeyEvent], None]] = []

    def tap(heard: Callable[[KeyEvent], None]) -> Callable[[], None]:
        taps.append(heard)
        return lambda: None

    monkeypatch.setattr(talkkey, "tap", tap)
    monkeypatch.setattr(keys, "HOLD_SECONDS", 0.05)
    monkeypatch.setattr(keys, "TURN_LIMIT_SECONDS", 0.1)
    made: list[Move] = []

    async def on_move(move: Move) -> None:
        made.append(move)

    driving = asyncio.create_task(keys.drive_talk_key(on_move))
    while not taps:
        await asyncio.sleep(0)
    taps[0](Pressed(asyncio.get_running_loop().time()))  # and never released: a key stuck down
    async with asyncio.timeout(1.0):
        while len(made) < 3:
            await asyncio.sleep(0.01)
    driving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await driving
    assert made == ["arm", "start", "expire"]


def test_a_run_without_the_input_monitoring_grant_is_refused_at_the_door(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    asked: list[None] = []
    monkeypatch.setattr(talkkey, "granted", lambda: False)
    monkeypatch.setattr(talkkey, "ask", lambda: asked.append(None))
    assert cli.main(["--home", str(tmp_path), "run"]) == 1
    assert "has no Input Monitoring grant" in capsys.readouterr().err
    assert asked == [None]
    # Refused before its first heartbeat: the one there is left to whatever wrote it, a running hands or a crash, and
    # the reason is in the audit log.
    assert not (tmp_path / "status.json").exists()
    [refused] = [json.loads(line) for line in audit.tail(Home(tmp_path).audit, 10)[0]]
    assert (refused["event"], refused["outcome"]) == ("hands.start", "failed") and "has no Input Monitoring grant" in refused["error"]


def test_a_unit_of_work_that_failed_is_said_on_the_terminal_and_is_one_line_in_the_log() -> None:
    import io
    from datetime import UTC, datetime

    from loguru import logger

    from hands.sessions import wide

    recorded: list[audit.Entry] = []
    record = cli.said_failed(recorded.append)
    terminal = io.StringIO()
    shown = cli.to_terminal(terminal)
    failures = logger.add(audit.failures_to(record), level="ERROR", filter="hands")
    try:
        with wide.unit("naming.pass", record):
            wide.fail("the naming model timed out\x1b[2J")
        with wide.unit("summary.pass", record):
            pass
        with wide.unit("brain.turn", record):
            wide.child("tool.call", wide.within(wide.here()), datetime.now(UTC), 1.0, "failed")
        # Raised through two units, it is said by what catches it, never by each unit it passed.
        with pytest.raises(OSError), wide.unit("hook", record), wide.unit("applied", record):
            raise OSError("the session is gone")
    finally:
        logger.remove(shown)
        logger.remove(failures)
    # The failed ones are said by name and error, controls made visible. The one that ended ok is not said, and
    # neither are the ones that raised.
    said = terminal.getvalue().splitlines()
    assert [line.split(" - ", 1)[1] for line in said] == ["naming.pass failed: the naming model timed out\\u001b[2J", "tool.call failed"]
    assert all("| ERROR    |" in line for line in said)
    # Every event is written, and no Failure line beside the ones it says.
    assert [(entry.event, entry.outcome) for entry in recorded if isinstance(entry, wide.WideEvent)] == [
        ("naming.pass", "failed"), ("summary.pass", "ok"), ("tool.call", "failed"), ("brain.turn", "ok"), ("applied", "failed"), ("hook", "failed"),
    ]
    assert len(recorded) == 6


def test_the_terminal_shows_hands_from_info_and_everything_else_from_warning() -> None:
    from loguru import logger

    from hands.daemon.cli import TERMINAL_LEVELS

    shown: list[str] = []
    sink = logger.add(lambda message: shown.append(message.record["message"]), filter=TERMINAL_LEVELS)
    try:
        for module in ("hands.sessions.tail", "pipecat.services.anthropic.llm"):
            patched = logger.patch(lambda record, module=module: record.update(name=module))
            for level in ("DEBUG", "INFO", "WARNING"):
                patched.log(level, f"{module} {level}")
    finally:
        logger.remove(sink)
    assert shown == [
        "hands.sessions.tail INFO",
        "hands.sessions.tail WARNING",
        "pipecat.services.anthropic.llm WARNING",
    ]
