"""The plugin's restart skill: a running daemon is started again in the same process, with the sessions it listed."""

import asyncio
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from loguru import logger

from conftest import unedited
from hands.core.session import Membership, SessionId
from hands.daemon.cli import launch, still_shown
from hands.daemon.restart import RESTART_SIGNAL
from hands.daemon.starting import Ended
from hands.sessions import audit, heartbeat
from hands.sessions.hookconfig import LAUNCHER, PLUGIN_DIR
from hands.sessions.home import Home
from hands.sessions.membership import write_membership

PLUGIN_ROOT = Path(__file__).resolve().parent.parent / PLUGIN_DIR
STANDIN = Path(__file__).resolve().parent / "fixtures" / "standin_daemon.py"
NOW = datetime(2026, 10, 3, 12, 0, 0, tzinfo=UTC)


def restart(home: Home, cwd: Path, path: str) -> subprocess.CompletedProcess[str]:
    """`/hands:restart` as the skill runs it: the plugin's launcher, in the session's directory, with no venv."""
    environment = {"HANDS_HOME": str(home.root), "PATH": path, "HOME": str(cwd)}
    return subprocess.run([PLUGIN_ROOT / LAUNCHER, "-m", "hands.daemon.restart"], env=environment, cwd=cwd, capture_output=True, text=True, timeout=60)


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


def test_a_restart_asked_through_the_plugin_brings_the_daemon_back_with_its_sessions(tmp_path: Path, python312: str, session: Membership) -> None:
    home = Home(tmp_path / "home")
    write_membership(home, session)
    daemon = subprocess.Popen([sys.executable, STANDIN, str(home.root)], stdin=subprocess.DEVNULL)
    try:
        before = running(home)
        assert before.live_sessions == 1

        done = restart(home, tmp_path, python312)

        assert done.returncode == 0, done.stderr
        after = heartbeat.read(home.status)
        assert after is not None
        # The same process, started again: what the skill said is what the heartbeat says.
        assert daemon.poll() is None
        assert (after.pid, after.pipeline, after.live_sessions) == (daemon.pid, "running", 1)
        assert after.started_at > before.started_at
        assert done.stdout.startswith(f"hands restarted: pid {daemon.pid} is running again after ")
        assert done.stdout.endswith(", with 1 live session.\n")
        restarts = [line for line in map(json.loads, audit.tail(home.audit, 10_000)[0]) if line["type"] == "Restarting"]
        assert restarts == [{"at": restarts[0]["at"], "level": "info", "type": "Restarting", "pid": daemon.pid}]
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
        home.config.write_text('[llm]\nbackend = "claude"\n')
        after = running(home)
        until = time.monotonic() + 10
        while after.started_at == before.started_at and time.monotonic() < until:
            time.sleep(0.02)
            after = running(home)
        # Started again, in the same process, with nothing asked of it but the edit.
        assert daemon.poll() is None
        assert (after.pid, after.live_sessions) == (daemon.pid, 1) and after.started_at > before.started_at
        said = [line for line in map(json.loads, audit.tail(home.audit, 10_000)[0]) if line["type"] in ("SettingsEdited", "Restarting")]
        assert [(line["type"], line.get("refused")) for line in said] == [("SettingsEdited", None), ("Restarting", None)]
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

    async def edited() -> None:
        await edit.wait()

    assert await launch(lambda: run, heart, edited) == "restart"


async def test_settings_that_cannot_be_watched_stop_the_run_saying_so(tmp_path: Path) -> None:
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=NOW, period=timedelta(milliseconds=10))

    async def run(quit_event: asyncio.Event) -> Ended:
        await quit_event.wait()
        return Ended(None, 0)

    async def edited() -> None:
        raise PermissionError("config.toml")

    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(str(message)), level="ERROR")
    try:
        assert await launch(lambda: run, heart, edited) == "quit"
    finally:
        logger.remove(sink)
    assert any("could not watch its settings file" in error and "PermissionError" in error for error in errors)


def test_a_daemon_that_is_not_running_is_not_asked_and_the_skill_says_why(tmp_path: Path, python312: str) -> None:
    home = Home(tmp_path / "home")
    done = restart(home, tmp_path, python312)
    assert (done.returncode, done.stdout) == (1, "")
    assert done.stderr == f"hands was not restarted: hands has not run: there is no heartbeat at {home.status}.\n"


def test_a_daemon_still_starting_is_not_asked(tmp_path: Path, python312: str) -> None:
    # Its pid is this test's, which a restart signal would end: being refused, it is never sent.
    home = Home(tmp_path / "home")
    heartbeat.Heart(home.status, os.getpid(), datetime.now(UTC), heartbeat.HEARTBEAT).beat("starting", None, 0, listening=False, deaf=False)
    done = restart(home, tmp_path, python312)
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

    launched = asyncio.create_task(launch(lambda: run, heart, unedited))
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

    launched = asyncio.create_task(launch(lambda: run, heart, unedited))
    await winding_down.wait()
    os.kill(os.getpid(), RESTART_SIGNAL)
    assert await launched == "quit"
    last = heartbeat.read(heart.path)
    assert last is not None and last.pipeline == "stopped"


def test_a_handed_on_indicator_is_kept_only_while_it_runs() -> None:
    shown = subprocess.Popen(["sleep", "60"])
    try:
        assert still_shown(shown.pid) == shown.pid
    finally:
        shown.kill()
        shown.wait()
    # Reaped by the run before, as its reap thread does with an indicator that exits.
    assert still_shown(shown.pid) is None
    exited = subprocess.Popen(["true"])
    while still_shown(exited.pid) is not None:
        time.sleep(0.01)
