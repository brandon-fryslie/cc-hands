"""The plugin's restart skill: a running daemon is started again in the same process, with the sessions it listed."""

import asyncio
import json
import os
import select
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from conftest import NO_PYTHON, unedited
from hands.core.session import Membership, SessionId
from hands.daemon import cli
from hands.daemon.cli import launch, retire
from hands.daemon.restart import RESTART_SIGNAL
from hands.daemon.starting import Ended, Start
from hands.sessions import audit, heartbeat
from hands.sessions.hookconfig import LAUNCHER
from hands.sessions.home import Home
from hands.sessions.membership import write_membership
from hands.sessions.wide import WideEvent

STANDIN = Path(__file__).resolve().parent / "fixtures" / "standin_daemon.py"
NOW = datetime(2026, 10, 3, 12, 0, 0, tzinfo=UTC)
EDITED = audit.SettingsEdited(path="/home/config.toml", refused=None)


def started(home: Home) -> list[dict[str, Any]]:
    """Each start's event in the home's audit log, oldest first."""
    return [line for line in map(json.loads, audit.tail(home.audit, 10_000)[0]) if line.get("event") == "hands.start"]


def start_outcomes(recorded: list[audit.Entry]) -> list[str]:
    return [entry.outcome for entry in recorded if isinstance(entry, WideEvent) and entry.event == "hands.start"]


def restart(plugin: Path, home: Home, cwd: Path) -> subprocess.CompletedProcess[str]:
    """`/hands:restart` as the skill runs it: the plugin's launcher, in the session's directory."""
    environment = {"HANDS_HOME": str(home.root), "PATH": NO_PYTHON, "HOME": str(cwd)}
    return subprocess.run([plugin / LAUNCHER, "-m", "hands.daemon", "restart"], env=environment, cwd=cwd, capture_output=True, text=True, timeout=60)


def running(home: Home, deadline: float = 10.0) -> heartbeat.Status:
    """The heartbeat once it says the pipeline is running."""
    until = time.monotonic() + deadline
    while time.monotonic() < until:
        status = heartbeat.read(home.status)
        if status is not None and status.pipeline == "running":
            return status
        time.sleep(0.02)
    raise AssertionError(f"the daemon did not say running within {deadline}s: {heartbeat.read(home.status)}")


@pytest.fixture
def session(tmp_path: Path) -> Iterator[Membership]:
    """A session that is running: a process, started before its membership file was written."""
    process = subprocess.Popen(["sleep", "60"])
    try:
        yield Membership(SessionId("the-session"), pid=process.pid, cwd=tmp_path, transcript=tmp_path / "the-session.jsonl")
    finally:
        process.kill()
        process.wait()


def test_a_restart_asked_through_the_plugin_brings_the_daemon_back_with_its_sessions(tmp_path: Path, plugin: Path, session: Membership) -> None:
    home = Home(tmp_path / "home")
    write_membership(home, session)
    daemon = subprocess.Popen([sys.executable, STANDIN, str(home.root)], stdin=subprocess.DEVNULL)
    try:
        before = running(home)
        assert before.live_sessions == 1

        done = restart(plugin, home, tmp_path)

        assert done.returncode == 0, done.stderr
        after = heartbeat.read(home.status)
        assert after is not None
        # The same process, started again: what the skill said is what the heartbeat says.
        assert daemon.poll() is None
        assert (after.pid, after.pipeline, after.live_sessions) == (daemon.pid, "running", 1)
        assert after.started_at > before.started_at
        assert done.stdout.startswith(f"hands restarted: pid {daemon.pid} is running again after ")
        assert done.stdout.endswith(", with 1 live session.\n")
        # Each start is one event: the first, and the restart's, in the same process, ready.
        starts = started(home)
        assert [(line["outcome"], line["facts"]["pid"], line["facts"]["restarted"]) for line in starts] == [("ok", daemon.pid, False), ("ok", daemon.pid, True)]
        # [LAW:nothing-unseen] and the restart asked is one command's event, which says what it saw the daemon come back as, and
        # how long that took.
        [command] = [line for line in map(json.loads, audit.tail(home.audit, 10_000)[0]) if line.get("event") == "hands.command"]
        outcome = command["facts"]["outcome"]
        assert (command["outcome"], command["facts"]["command"], command["facts"]["exit_code"], outcome["type"]) == ("ok", "restart", 0, "Restarted")
        assert (outcome["status"]["pid"], outcome["status"]["started_at"]) == (daemon.pid, after.started_at.isoformat(timespec="milliseconds"))
        assert outcome["took"] > 0
    finally:
        daemon.terminate()
        daemon.wait(timeout=10)
    stopped = heartbeat.read(home.status)
    assert stopped is not None and stopped.pipeline == "stopped"


def test_an_edit_to_the_settings_brings_the_daemon_back_on_them_with_its_sessions(tmp_path: Path, session: Membership) -> None:
    home = Home(tmp_path / "home")
    write_membership(home, session)
    daemon = subprocess.Popen([sys.executable, STANDIN, str(home.root)], stdin=subprocess.DEVNULL)
    try:
        before = running(home)
        home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
        after = running(home)
        until = time.monotonic() + 10
        while after.started_at == before.started_at and time.monotonic() < until:
            time.sleep(0.02)
            after = running(home)
        # Started again, in the same process, with nothing asked of it but the edit.
        assert daemon.poll() is None
        assert (after.pid, after.live_sessions) == (daemon.pid, 1) and after.started_at > before.started_at
        said = [line for line in map(json.loads, audit.tail(home.audit, 10_000)[0]) if line["type"] == "SettingsEdited" or line.get("event") == "hands.start"]
        assert [(line["type"], line.get("refused"), line.get("facts", {}).get("restarted")) for line in said] == [("WideEvent", None, False), ("SettingsEdited", None, None), ("WideEvent", None, True)]
    finally:
        daemon.terminate()
        daemon.wait(timeout=10)


async def test_settings_edited_end_a_run_as_a_restart(tmp_path: Path) -> None:
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=NOW, period=timedelta(milliseconds=10))
    edit = asyncio.Event()

    async def run(quit_event: asyncio.Event) -> Ended:
        edit.set()
        await quit_event.wait()
        return Ended(None, 0)

    async def edited() -> audit.SettingsEdited:
        await edit.wait()
        return EDITED

    recorded: list[audit.Entry] = []
    assert await launch(lambda: run, heart, lambda: (), edited, recorded.append, Start(restarted=False)) == "restart"
    # Told to stop before it was ready, the start ends cancelled.
    assert recorded[0] == EDITED and start_outcomes(recorded) == ["cancelled"] and len(recorded) == 2


async def test_settings_edited_as_a_run_ends_are_taken_up_by_the_next_start_and_not_said(tmp_path: Path) -> None:
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=NOW, period=timedelta(milliseconds=10))
    edit = asyncio.Event()

    async def run(quit_event: asyncio.Event) -> Ended:
        # As the q key does: the event set alone, then the run winds down while the file is saved.
        quit_event.set()
        edit.set()
        await asyncio.sleep(0.05)
        return Ended(None, 0)

    async def edited() -> audit.SettingsEdited:
        await edit.wait()
        return EDITED

    recorded: list[audit.Entry] = []
    assert await launch(lambda: run, heart, lambda: (), edited, recorded.append, Start(restarted=False)) == "quit"
    assert start_outcomes(recorded) == ["cancelled"] and len(recorded) == 1


async def test_settings_that_cannot_be_watched_stop_the_run_saying_so(tmp_path: Path) -> None:
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=NOW, period=timedelta(milliseconds=10))

    async def run(quit_event: asyncio.Event) -> Ended:
        await quit_event.wait()
        return Ended(None, 0)

    async def edited() -> audit.SettingsEdited:
        raise PermissionError("config.toml")

    recorded: list[audit.Entry] = []
    with pytest.raises(PermissionError, match="config.toml"):
        await launch(lambda: run, heart, lambda: (), edited, recorded.append, Start(restarted=False))
    # A start that ends raising is failed, with what it raised.
    assert [(entry.outcome, entry.error) for entry in recorded if isinstance(entry, WideEvent)] == [("failed", "PermissionError: config.toml")]
    # As a run whose background task failed: its last heartbeat does not read as stopped.
    status = heartbeat.read(heart.path)
    assert status is None or status.pipeline != "stopped"


def test_a_daemon_that_is_not_running_is_not_asked_and_the_skill_says_why(tmp_path: Path, plugin: Path) -> None:
    home = Home(tmp_path / "home")
    done = restart(plugin, home, tmp_path)
    assert (done.returncode, done.stdout) == (1, "")
    assert done.stderr == f"hands was not restarted: hands has not run: there is no heartbeat at {home.status}.\n"


def test_a_daemon_still_starting_is_not_asked(tmp_path: Path, plugin: Path) -> None:
    # Its pid is this test's, which a restart signal would end: being refused, it is never sent.
    home = Home(tmp_path / "home")
    heartbeat.Heart(home.status, os.getpid(), datetime.now(UTC), heartbeat.HEARTBEAT).beat("starting", None, 0, listening=False, degraded=())
    done = restart(plugin, home, tmp_path)
    assert done.returncode == 1
    assert done.stderr.startswith(f"hands was not restarted: hands is up: pid {os.getpid()}, ")
    assert "pipeline starting" in done.stderr


async def test_the_restart_signal_ends_a_run_as_a_restart_whose_last_heartbeat_says_starting(tmp_path: Path) -> None:
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=NOW, period=timedelta(milliseconds=10))
    told: list[asyncio.Event] = []

    async def run(quit_event: asyncio.Event) -> Ended:
        told.append(quit_event)
        await quit_event.wait()
        return Ended(NOW, 3)

    launched = asyncio.create_task(launch(lambda: run, heart, lambda: (), unedited, lambda _entry: None, Start(restarted=False)))
    while not told:
        await asyncio.sleep(0.005)
    os.kill(os.getpid(), RESTART_SIGNAL)
    assert await launched == "restart"
    last = heartbeat.read(heart.path)
    # Not stopped: the indicator reads the moment before the next run as starting, and posts nothing.
    assert last is not None and (last.pipeline, last.last_audio_out, last.live_sessions) == ("starting", NOW, 3)


async def test_a_restart_asked_while_a_quit_winds_the_run_down_does_not_start_it_again(tmp_path: Path) -> None:
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=NOW, period=timedelta(milliseconds=10))
    winding_down = asyncio.Event()

    async def run(quit_event: asyncio.Event) -> Ended:
        # As the q key does: the event is set with no signal, and the run takes a while to close.
        quit_event.set()
        winding_down.set()
        await asyncio.sleep(0.05)
        return Ended(None, 0)

    launched = asyncio.create_task(launch(lambda: run, heart, lambda: (), unedited, lambda _entry: None, Start(restarted=False)))
    await winding_down.wait()
    os.kill(os.getpid(), RESTART_SIGNAL)
    assert await launched == "quit"
    last = heartbeat.read(heart.path)
    assert last is not None and last.pipeline == "stopped"


def spawned(*argv: str) -> int:
    """A child of this process, as the indicator is of the run that starts it and of every run exec'd after."""
    # A session of its own, so it leads a process group, as `start_indicator` makes the indicator.
    return os.posix_spawnp(argv[0], list(argv), os.environ, setsid=True)


def test_a_restart_ends_the_indicator_the_run_before_showed() -> None:
    shown = spawned("sleep", "60")
    assert retire(shown) == "ended"
    # Reaped as it was ended: no zombie is left under the run.
    with pytest.raises(ChildProcessError):
        os.waitpid(shown, os.WNOHANG)


def test_an_indicator_that_does_not_end_when_asked_is_killed_with_its_group() -> None:
    ready, said = os.pipe()
    argv = ["sh", "-c", 'trap "" TERM; sleep 60 & echo; wait']
    stubborn = os.posix_spawnp("sh", argv, os.environ, file_actions=[(os.POSIX_SPAWN_DUP2, said, 1)], setsid=True)
    os.close(said)
    with os.fdopen(ready) as lines:
        # Its line comes once SIGTERM is ignored, so the ask below cannot land before it is.
        lines.readline()
        assert retire(stubborn, grace=0.2) == "killed"
        with pytest.raises(ChildProcessError):
            os.waitpid(stubborn, os.WNOHANG)
        # The sleep it started holds the pipe too, so the pipe ends only once it went with the group.
        assert select.select([lines], [], [], 5)[0] == [lines]
        assert lines.read() == ""


def test_an_indicator_that_exited_before_the_restart_is_reaped_and_said() -> None:
    exited = spawned("false")
    # Exited and not reaped, as an indicator that exits during the exec is: no reap thread outlives it. A zombie in a
    # terminal's foreground group says Z+.
    while not subprocess.run(["ps", "-o", "stat=", "-p", str(exited)], capture_output=True, text=True).stdout.startswith("Z"):
        time.sleep(0.01)
    logged: list[str] = []
    sink = logger.add(lambda message: logged.append(message.record["message"]), level="ERROR")
    try:
        assert retire(exited) == "exited"
    finally:
        logger.remove(sink)
    assert logged == ["the menu-bar indicator exited (1) while hands runs; hands is not shown in the menu bar"]
    # Reaped by the run before, as its reap thread does with an indicator that exits, and said there.
    assert retire(exited) == "reaped"


def test_a_restarted_run_shows_itself_through_an_indicator_it_started(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The run before's indicator judged the heartbeat by the code that run had; this run's is started on the code it runs.
    from hands.voice import talkkey

    home = Home(tmp_path)
    previous = spawned("sleep", "60")
    # [LAW:behavior-not-structure] the order is the contract: the one before is gone before this run's is shown.
    steps: list[str] = []

    def retired(indicator: int) -> cli.Retired:
        steps.append("retired")
        return retire(indicator)

    def start_indicator(started: Home) -> int:
        assert started == home
        steps.append("started")
        return 0

    def kept(*_: object) -> None:
        pass

    def loaded(*_: object) -> cli.Run:
        async def quits(_quit_event: asyncio.Event) -> Ended:
            return Ended(None, 0)

        return quits

    monkeypatch.setattr(talkkey, "granted", lambda: True)
    monkeypatch.setattr(cli, "retire", retired)
    monkeypatch.setattr(cli, "start_indicator", start_indicator)
    monkeypatch.setattr(cli, "reap", kept)
    monkeypatch.setattr(cli, "to_terminal", kept)
    monkeypatch.setattr(cli.logger, "remove", kept)
    monkeypatch.setattr(cli, "loaded", loaded)
    assert cli.main(["--home", str(home.root), "run", "--restarted", str(previous)]) == 0
    assert steps == ["retired", "started"]
    with pytest.raises(ChildProcessError):
        os.waitpid(previous, os.WNOHANG)
    [start] = started(home)
    assert (start["facts"]["restarted"], start["facts"]["previous_indicator"]) == (True, "ended")
