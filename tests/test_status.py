"""The heartbeat file and what `hands status` concludes from it, with no daemon running."""

import asyncio
import os
import plistlib
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hands.daemon import indicator, launchd, status
from hands.daemon.cli import main
from hands.daemon.run import keep_beating
from hands.voice.threads import off_loop
from hands.sessions.home import Home
from hands.sessions.payload import Rejected

NOW = datetime(2026, 9, 14, 12, 0, 0, tzinfo=UTC)
BEAT = timedelta(seconds=2)


def beat(**changes: object) -> status.Status:
    fields: dict[str, object] = {
        "pid": 4242,
        "started_at": NOW - timedelta(minutes=5, seconds=3),
        "written_at": NOW - timedelta(seconds=1),
        "heartbeat": BEAT,
        "pipeline": "running",
        "last_audio_out": NOW - timedelta(seconds=12),
        "live_sessions": 2,
        **changes,
    }
    return status.Status(**fields)  # pyright: ignore[reportArgumentType]


def test_a_heartbeat_reads_back_as_it_was_written(tmp_path: Path) -> None:
    written = beat(last_audio_out=None)
    status.write(tmp_path / "status.json", written)
    assert status.read(tmp_path / "status.json") == written
    assert [path.name for path in tmp_path.iterdir()] == ["status.json"]  # nothing left of the replacement


def test_every_heartbeat_of_a_run_repeats_what_the_heart_fixed(tmp_path: Path) -> None:
    heart = status.Heart(tmp_path / "status.json", pid=4242, started_at=NOW, period=BEAT)
    heart.beat("starting", None, 0)
    first = status.read(heart.path)
    heart.beat("running", NOW, 2)
    second = status.read(heart.path)
    assert first is not None and second is not None
    assert (first.pid, first.started_at, first.heartbeat, first.pipeline, first.live_sessions) == (4242, NOW, BEAT, "starting", 0)
    assert (second.pid, second.started_at, second.heartbeat, second.pipeline, second.last_audio_out) == (4242, NOW, BEAT, "running", NOW)
    assert second.written_at >= first.written_at


def test_no_file_is_a_daemon_that_never_ran(tmp_path: Path) -> None:
    assert status.read(tmp_path / "status.json") is None


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        (b"{", "not JSON"),
        (b'{"pid": 1}', "started_at"),
        (status.encode(beat()).replace("running", "dancing").encode(), "pipeline should be"),
        (status.encode(beat()).replace("+00:00", "").encode(), "carries its zone"),
        (status.encode(beat(pid=2**63)).encode(), "not a process id"),
        (status.encode(beat(pid=2**31)).encode(), "not a process id"),
        (status.encode(beat(pid=100_000)).encode(), "not a process id"),  # past macOS's PID_MAX, which ps refuses
        (status.encode(beat(pid=0)).encode(), "not a process id"),  # kill(0, 0) asks after our own process group
        (status.encode(beat(pid=-1)).encode(), "not a process id"),
        (status.encode(beat()).replace('"heartbeat_ms": 2000', '"heartbeat_ms": 100000000000000000000').encode(), "not a heartbeat period"),
        (status.encode(beat(heartbeat=timedelta(0))).encode(), "not a heartbeat period"),
    ],
)
def test_a_heartbeat_that_does_not_parse_is_refused(raw: bytes, error: str) -> None:
    with pytest.raises(Rejected, match=error):
        status.parse(raw)


def test_the_verdict_follows_the_pid_and_the_heartbeat_age(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    assert status.judge(path, None, NOW, alive=False) == status.NeverRan(path)
    assert status.judge(path, beat(), NOW, alive=True) == status.Up(beat())
    assert status.judge(path, beat(), NOW, alive=False) == status.Down(beat())
    late = beat(written_at=NOW - BEAT * status.MISSED_BEATS - timedelta(seconds=1))
    assert status.judge(path, late, NOW, alive=True) == status.Unresponsive(late)


@pytest.mark.parametrize("alive", [False, True])
def test_a_daemon_whose_last_heartbeat_said_stopped_is_stopped_whoever_holds_its_pid_now(alive: bool, tmp_path: Path) -> None:
    # Alive is a daemon still cleaning up, or another process that got the pid; neither is a hang.
    stopped = beat(pipeline="stopped", written_at=NOW - timedelta(hours=1))
    assert status.judge(tmp_path, stopped, NOW, alive=alive) == status.Stopped(stopped)


def test_each_verdict_is_said_plainly(tmp_path: Path) -> None:
    assert status.describe(status.Up(beat()), NOW) == (
        "hands is up: pid 4242, up 5m 3s, pipeline running, last audio out 12s ago, 2 live sessions"
    )
    assert status.describe(status.Up(beat(last_audio_out=None, live_sessions=1, started_at=NOW - timedelta(hours=2))), NOW) == (
        "hands is up: pid 4242, up 2h 0m 0s, pipeline running, last audio out never, 1 live session"
    )
    assert status.describe(status.Down(beat()), NOW) == "hands is down: its process, pid 4242, is gone; its last heartbeat was 1s ago"
    assert status.describe(status.Unresponsive(beat(written_at=NOW - timedelta(minutes=3))), NOW) == (
        "hands is not responding: pid 4242 is running, pipeline running, but its last heartbeat was 3m 0s ago"
    )
    assert status.describe(status.Stopped(beat(pipeline="stopped")), NOW) == "hands is stopped: pid 4242 finished its pipeline 1s ago"
    assert status.describe(status.NeverRan(tmp_path), NOW) == f"hands has not run: there is no heartbeat at {tmp_path}"
    assert status.describe(status.Unreadable(tmp_path, "not JSON"), NOW) == (
        f"hands is unknown: its heartbeat at {tmp_path} cannot be read: not JSON"
    )


def test_looking_at_the_heartbeat_judges_it_against_the_process_table(dead_pid: Callable[[], int], tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    now = datetime.now(UTC)
    assert status.look(path, now) == status.NeverRan(path)
    status.write(path, beat(pid=os.getpid(), started_at=now, written_at=now))
    assert isinstance(status.look(path, now), status.Up)
    status.write(path, beat(pid=dead_pid(), started_at=now, written_at=now))
    assert isinstance(status.look(path, now), status.Down)


@pytest.mark.parametrize(
    "raw", [b"{", status.encode(beat(pid=2**63)).encode(), status.encode(beat()).replace('"heartbeat_ms": 2000', '"heartbeat_ms": 1e400').encode()]
)
def test_a_heartbeat_that_cannot_be_read_is_its_own_verdict_and_never_raises(raw: bytes, tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    path.write_bytes(raw)
    assert isinstance(status.look(path, NOW), status.Unreadable)
    path.unlink()
    path.mkdir()  # a read that fails in the OS, not in the parse
    assert isinstance(status.look(path, NOW), status.Unreadable)


def test_hands_status_exits_zero_only_when_the_daemon_is_up(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path)
    assert main(["--home", str(tmp_path), "status"]) == 1
    assert "has not run" in capsys.readouterr().out
    status.write(home.status, beat(pid=os.getpid(), started_at=datetime.now(UTC), written_at=datetime.now(UTC)))
    assert main(["--home", str(tmp_path), "status"]) == 0
    assert capsys.readouterr().out.startswith(f"hands is up: pid {os.getpid()}")
    home.status.write_text("{")
    assert main(["--home", str(tmp_path), "status"]) == 2
    assert "cannot be read" in capsys.readouterr().err
    status.write(home.status, beat(pid=2**63))
    assert main(["--home", str(tmp_path), "status"]) == 2
    assert "not a process id" in capsys.readouterr().err


def test_the_daemon_is_running_only_if_its_pid_is_held_by_the_process_that_started_then(dead_pid: Callable[[], int]) -> None:
    now = datetime.now(UTC)
    assert status.running(beat(pid=os.getpid(), started_at=now))
    assert status.running(beat(pid=1, started_at=now))  # launchd, root's: seen, and it started before now
    assert not status.running(beat(pid=dead_pid(), started_at=now))
    # This process holds the pid, but it started long after the heartbeat's daemon did: the number was reused.
    assert not status.running(beat(pid=os.getpid(), started_at=now - timedelta(days=3)))


def test_a_heartbeat_whose_pid_went_to_a_later_process_reads_as_down(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    now = datetime.now(UTC)
    # After a reboot: the old heartbeat, still on disk, names a pid some new process now holds.
    status.write(path, beat(pid=os.getpid(), started_at=now - timedelta(days=3), written_at=now - timedelta(days=3)))
    assert isinstance(status.look(path, now), status.Down)


async def test_the_heartbeat_is_rewritten_every_period() -> None:
    beats: list[None] = []
    beating = asyncio.create_task(keep_beating(lambda: beats.append(None), 0.01))
    await asyncio.sleep(0.055)
    beating.cancel()
    assert 3 <= len(beats) <= 6


async def test_work_off_the_loop_returns_its_result_or_raises_its_error() -> None:
    assert await off_loop(lambda: 42, "answer") == 42
    with pytest.raises(ValueError, match="no models"):
        await off_loop(lambda: (_ for _ in ()).throw(ValueError("no models")), "failure")


def test_a_process_exits_without_waiting_for_work_left_running_off_the_loop(tmp_path: Path) -> None:
    script = (
        "import asyncio, time\n"
        "from hands.voice.threads import off_loop\n"
        "async def main():\n"
        "    work = asyncio.create_task(off_loop(lambda: time.sleep(30), 'slow'))\n"
        "    await asyncio.sleep(0.1)\n"
        "    work.cancel()\n"
        "asyncio.run(main())\n"
    )
    started = time.monotonic()
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)
    assert time.monotonic() - started < 5


def test_the_launch_agent_keeps_the_daemon_up_and_logs_where_status_can_point(tmp_path: Path) -> None:
    home = Home(tmp_path)
    agent = plistlib.loads(launchd.agent(launchd.DAEMON, Path("/venv/bin/python"), home))
    assert agent == {
        "Label": "hands.daemon",
        "ProgramArguments": ["/venv/bin/python", "-m", "hands.daemon", "--home", str(tmp_path), "run"],
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Interactive",
        "StandardOutPath": str(home.daemon_log),
        "StandardErrorPath": str(home.daemon_log),
    }


def test_the_indicator_has_a_launch_agent_of_its_own(tmp_path: Path, capsysbinary: pytest.CaptureFixture[bytes]) -> None:
    home = Home(tmp_path)
    assert main(["--home", str(tmp_path), "launchd", "indicator"]) == 0
    agent = plistlib.loads(capsysbinary.readouterr().out)
    assert agent["Label"] == "hands.indicator"
    assert agent["ProgramArguments"][-3:] == ["--home", str(tmp_path), "indicator"]
    assert agent["KeepAlive"] is True
    assert agent["StandardErrorPath"] == str(home.indicator_log)


def test_each_verdict_has_its_own_light_and_the_broken_ones_warn(tmp_path: Path) -> None:
    verdicts: list[status.Verdict] = [
        status.Up(beat()),
        status.Unresponsive(beat()),
        status.Down(beat()),
        status.NeverRan(tmp_path),
        status.Unreadable(tmp_path, "not JSON"),
    ]
    shown = [indicator.show(None, verdict, NOW) for verdict in verdicts]
    assert [seen.light for seen in shown] == ["up", "not responding", "down", "off", "unreadable"]
    assert len({seen.title for seen in shown}) == len(shown)
    assert [seen.title.startswith("⚠︎") for seen in shown] == [False, True, True, False, True]
    assert indicator.show(None, status.Stopped(beat(pipeline="stopped")), NOW).light == "off"
    assert [seen.text for seen in shown] == [status.describe(verdict, NOW) for verdict in verdicts]


def shown_over(looks: Sequence[tuple[status.Verdict, datetime]]) -> list[tuple[str, ...]]:
    before: indicator.Shown | None = None
    posted: list[tuple[str, ...]] = []
    for verdict, at in looks:
        before = indicator.show(before, verdict, at)
        posted.append(before.notices)
    return posted


def test_a_notification_is_posted_when_the_verdict_leaves_up_and_only_then(tmp_path: Path) -> None:
    down, up = status.Down(beat()), status.Up(beat())
    looks: list[status.Verdict] = [down, up, up, down, down, status.Unreadable(tmp_path, "x")]
    assert shown_over([(verdict, NOW) for verdict in looks]) == [(), (), (), (status.describe(down, NOW),), (), ()]


def test_a_daemon_that_keeps_crashing_is_announced_once_a_quiet_window(tmp_path: Path) -> None:
    down, up = status.Down(beat()), status.Up(beat())
    # launchd restarts a daemon that crashes on every start about every ten seconds.
    looks = [(verdict, NOW + timedelta(seconds=10 * cycle)) for cycle in range(8) for verdict in (up, down)]
    posted = [at for (_, at), notices in zip(looks, shown_over(looks)) if notices]
    assert posted == [NOW, NOW + timedelta(seconds=60)]


def test_a_death_inside_the_quiet_window_is_announced_when_the_window_closes_if_hands_is_still_down() -> None:
    down, up = status.Down(beat()), status.Up(beat())
    at = [NOW + timedelta(seconds=seconds) for seconds in (0, 1, 10, 40, 50, 61, 70)]
    looks = list(zip([up, down, up, down, down, down, down], at))
    assert [bool(notices) for notices in shown_over(looks)] == [False, True, False, False, False, True, False]


def test_a_death_owed_inside_the_quiet_window_is_dropped_if_hands_comes_back_before_it_closes() -> None:
    down, up = status.Down(beat()), status.Up(beat())
    at = [NOW + timedelta(seconds=seconds) for seconds in (0, 1, 10, 40, 55, 70)]
    looks = list(zip([up, down, up, down, up, up], at))
    assert not any(shown_over(looks)[2:])
