"""The talk key: Right Shift held alone opens a turn, its release sends it, and any other key is typing."""

import asyncio
import json
import subprocess
import sys
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


def holds(home: Path, said: str) -> str:
    """Python that holds `home` as a running daemon does, then prints `said`."""
    return f"from pathlib import Path\nfrom hands.daemon import cli\nfrom hands.sessions.home import Home\ncli.hold(Home(Path({str(home)!r})))\nprint({said!r}, flush=True)\n"


def holding(home: Path, then: str) -> subprocess.Popen[str]:
    """A process that holds `home` as a running daemon does, then runs the Python `then`; returned once it holds it."""
    holder = subprocess.Popen([sys.executable, "-c", holds(home, "held") + then], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert holder.stdout is not None and holder.stdout.readline() == "held\n"
    return holder


@pytest.mark.parametrize("run", [["run"], ["run", "--restarted", "0"]])
def test_a_second_run_on_a_home_a_daemon_holds_is_refused_and_leaves_its_heartbeat_and_sockets(run: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import shutil
    import socket
    import tempfile
    from datetime import UTC, datetime, timedelta

    from hands.sessions import heartbeat

    # A unix socket path is capped near 104 bytes on macOS, so not under pytest's long tmp_path.
    root = Path(tempfile.mkdtemp(prefix="hands-"))
    home = Home(root)
    holder = holding(root, "import sys; sys.stdin.read()")
    listening = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        heartbeat.Heart(home.status, holder.pid, datetime.now(UTC), timedelta(seconds=2)).beat("running", None, 1, listening=True, deaf=False)
        live = home.status.read_bytes()
        listening.bind(str(home.socket))
        listening.listen()
        # The ticket's repro: the second run is in a terminal without the grant, and is told the daemon runs, not
        # asked for a grant it does not need.
        asked: list[None] = []
        monkeypatch.setattr(talkkey, "granted", lambda: False)
        monkeypatch.setattr(talkkey, "ask", lambda: asked.append(None))
        assert cli.main(["--home", str(root), *run]) == 1
        assert f"hands is already running on {root}, as pid {holder.pid}" in capsys.readouterr().err
        assert asked == []
        assert home.status.read_bytes() == live
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as shim:
            shim.connect(str(home.socket))
        [refused] = [json.loads(line) for line in audit.tail(home.audit, 10)[0]]
        assert (refused["event"], refused["outcome"]) == ("hands.start", "failed") and "already running" in refused["error"]
    finally:
        listening.close()
        holder.kill()
        holder.wait()
        shutil.rmtree(root)


def test_a_restart_execd_in_the_running_daemons_process_keeps_its_home(tmp_path: Path) -> None:
    # Held, the process execs into hands again, as a restart's run does, and holds the home there too, through the one
    # descriptor it already had on the lock.
    lock = Home(tmp_path).lock
    counted = f"import os\ntarget = os.stat({str(lock)!r})\nopen_on = 0\nfor name in os.listdir('/dev/fd'):\n    try:\n        found = os.fstat(int(name))\n    except OSError:\n        continue\n    open_on += (found.st_dev, found.st_ino) == (target.st_dev, target.st_ino)\nprint(open_on, flush=True)\n"
    restarted = holds(tmp_path, "again") + counted + "import sys; sys.stdin.read()"
    holder = holding(tmp_path, f"import sys\nfrom hands.daemon.starting import again\nagain([sys.executable, '-c', {restarted!r}])")
    try:
        assert holder.stdout is not None and holder.stdout.readline() == "again\n"
        assert holder.stdout.readline() == "1\n"
        # And still holds it against every other process.
        other = subprocess.run([sys.executable, "-c", holds(tmp_path, "held")], capture_output=True, text=True)
        assert other.returncode != 0 and f"hands is already running on {tmp_path}, as pid {holder.pid}" in other.stderr
    finally:
        holder.kill()
        holder.wait()


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
        for sink in (*shown, failures):
            logger.remove(sink)
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


def test_an_exception_reaches_the_terminal_with_its_controls_as_escapes_and_its_diagnosis_kept() -> None:
    import io

    from loguru import logger

    class Terminal(io.StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.writes: list[str] = []

        def write(self, text: str) -> int:
            self.writes.append(text)
            return super().write(text)

    terminal = Terminal()
    shown = cli.to_terminal(terminal)
    try:
        try:
            said = "gone\x1b[2K\x07\x08"
            raise ValueError(said)
        except ValueError:
            logger.exception("the turn could not be read")
        logger.warning("next")
        logger.opt(raw=True).warning("raw\n")
    finally:
        for sink in shown:
            logger.remove(sink)
    written = terminal.getvalue()
    assert not {"\x1b", "\x07", "\x08"} & set(written)
    assert "ValueError: gone\\u001b[2K\\u0007\\u0008" in written
    # The backtrace marks the frame that caught it, and the diagnosis names a variable's value under the line using it.
    assert "> File " in written and "'gone\\x1b[2K\\x07\\x08'" in written
    # Each record is one write, by one sink: the trace whole, then the line after it, then the raw one once.
    [trace, line, raw] = terminal.writes
    assert " - the turn could not be read\n" in trace and "ValueError" in trace
    assert line.endswith(" - next\n")
    assert raw == "raw\n"


def test_the_terminal_shows_hands_from_info_and_everything_else_from_warning() -> None:
    from loguru import logger

    from hands.daemon.cli import on_terminal

    shown: list[str] = []
    sink = logger.add(lambda message: shown.append(message.record["message"]), filter=on_terminal)
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
