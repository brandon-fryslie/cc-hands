"""The heartbeat file and what `hands status` concludes from it, with no daemon running."""

import asyncio
import os
import plistlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hands.daemon import launchd, status
from hands.daemon.cli import main, pid_alive
from hands.daemon.run import keep_beating
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


def test_no_file_is_a_daemon_that_never_ran(tmp_path: Path) -> None:
    assert status.read(tmp_path / "status.json") is None


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        (b"{", "not JSON"),
        (b'{"pid": 1}', "started_at"),
        (status.encode(beat()).replace("running", "dancing").encode(), "pipeline should be"),
        (status.encode(beat()).replace("+00:00", "").encode(), "carries its zone"),
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


def test_each_verdict_is_said_plainly(tmp_path: Path) -> None:
    assert status.describe(status.Up(beat()), NOW) == (
        "hands is up: pid 4242, up 5m 3s, pipeline running, last audio out 12s ago, 2 live sessions"
    )
    assert status.describe(status.Up(beat(last_audio_out=None, live_sessions=1, started_at=NOW - timedelta(hours=2))), NOW) == (
        "hands is up: pid 4242, up 2h 0m 0s, pipeline running, last audio out never, 1 live session"
    )
    assert status.describe(status.Down(beat()), NOW) == "hands is down: pid 4242 is not running; its last heartbeat was 1s ago"
    assert status.describe(status.Unresponsive(beat(written_at=NOW - timedelta(minutes=3))), NOW) == (
        "hands is not responding: pid 4242 is running, but its last heartbeat was 3m 0s ago"
    )
    assert status.describe(status.NeverRan(tmp_path), NOW) == f"hands has not run: there is no heartbeat at {tmp_path}"


def test_hands_status_exits_zero_only_when_the_daemon_is_up(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path)
    assert main(["--home", str(tmp_path), "status"]) == 1
    assert "has not run" in capsys.readouterr().out
    status.write(home.status, beat(pid=os.getpid(), written_at=datetime.now(UTC)))
    assert main(["--home", str(tmp_path), "status"]) == 0
    assert capsys.readouterr().out.startswith(f"hands is up: pid {os.getpid()}")
    home.status.write_text("{")
    assert main(["--home", str(tmp_path), "status"]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_liveness_comes_from_the_os() -> None:
    assert pid_alive(os.getpid())
    assert pid_alive(1)  # launchd: alive, and not ours to signal
    assert not pid_alive(2**22 + 12345)


async def test_the_heartbeat_is_rewritten_every_period() -> None:
    beats: list[None] = []
    beating = asyncio.create_task(keep_beating(lambda: beats.append(None), 0.01))
    await asyncio.sleep(0.055)
    beating.cancel()
    assert 3 <= len(beats) <= 6


def test_the_launch_agent_keeps_the_daemon_up_and_logs_where_status_can_point(tmp_path: Path) -> None:
    home = Home(tmp_path)
    agent = plistlib.loads(launchd.agent(Path("/venv/bin/python"), home, path="/opt/homebrew/bin:/usr/bin"))
    assert agent == {
        "Label": "hands.daemon",
        "ProgramArguments": ["/venv/bin/python", "-m", "hands.daemon", "--home", str(tmp_path), "run"],
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Interactive",
        "StandardOutPath": str(home.daemon_log),
        "StandardErrorPath": str(home.daemon_log),
        "EnvironmentVariables": {"PATH": "/opt/homebrew/bin:/usr/bin"},
    }
