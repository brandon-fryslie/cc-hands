"""The shim against a real hook socket: it records membership and posts; with no daemon it is silent when hands is
off and loud when hands is broken."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import socket
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hands.core.session import Idle, Membership, PromptId, Session, SessionId, Submitted
from hands.sessions import heartbeat
from hands.sessions.hookconfig import LAUNCHER, PLUGIN_DIR, SHIM_MODULE
from hands.sessions.home import Home
from hands.sessions.membership import read_membership
from hands.sessions.registry import Sessions
from hands.sessions.server import serve_hooks

SID = SessionId("0f1e2d3c-aaaa-bbbb-cccc-000000000001")
COMMON = {"session_id": SID, "transcript_path": "/nowhere/t.jsonl", "cwd": "/code/a"}
START = {**COMMON, "hook_event_name": "SessionStart", "source": "startup"}
PROMPT = {**COMMON, "hook_event_name": "UserPromptSubmit", "prompt": "hi", "prompt_id": "p1"}
END = {**COMMON, "hook_event_name": "SessionEnd", "reason": "other"}
ASK = {**COMMON, "hook_event_name": "PermissionRequest", "tool_name": "Bash", "tool_input": {"command": "ls"}}
PLUGIN_ROOT = Path(__file__).resolve().parent.parent / PLUGIN_DIR


@pytest.fixture
def home() -> Iterator[Home]:
    # A unix socket path is capped near 104 bytes on macOS, so not under pytest's long tmp_path.
    root = Path(tempfile.mkdtemp(prefix="hands-"))
    yield Home(root)
    shutil.rmtree(root)


@pytest.fixture
async def sessions(home: Home) -> AsyncIterator[Sessions]:
    registry = Sessions(permission_deadline=60.0, clock=lambda: 10.0, record=lambda _: None)
    runner = await serve_hooks(home, registry)
    yield registry
    await runner.cleanup()


async def shim(home: Home, payload: Mapping[str, object], fritter: str | None = None) -> tuple[int | None, str, str]:
    """The shim's exit, stdout, and stderr for one hook, run as the plugin runs it: with only HANDS_HOME to go on."""
    # The shim reads FRITTER_SOCKET from its environment, so the tests say what is in it
    # rather than inheriting whatever the run happens to have. Without this, running the
    # suite from inside a wrapped session would change what the shim records.
    environment = {key: value for key, value in os.environ.items() if key != "FRITTER_SOCKET"}
    if fritter is not None:
        environment["FRITTER_SOCKET"] = fritter
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        SHIM_MODULE,
        env={**environment, "HANDS_HOME": str(home.root)},
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate(json.dumps(payload).encode())
    return process.returncode, stdout.decode(), stderr.decode()


def beat(home: Home, pid: int, written_ago: timedelta, pipeline: heartbeat.PipelineState = "running") -> None:
    now = datetime.now(UTC)
    heartbeat.write(home.status, heartbeat.Status(pid, now, now - written_ago, heartbeat.HEARTBEAT, pipeline, None, 0))


@pytest.mark.parametrize("payload", [START, PROMPT, ASK], ids=["start", "prompt", "permission"])
async def test_a_hands_that_never_ran_costs_the_session_nothing(home: Home, payload: Mapping[str, object]) -> None:
    # Nothing on stdout, so a permission request falls through to Claude Code's own dialog.
    assert await shim(home, payload) == (0, "", "")


async def test_a_hands_that_was_stopped_costs_the_session_nothing(home: Home) -> None:
    beat(home, os.getpid(), timedelta(minutes=5), pipeline="stopped")
    assert await shim(home, ASK) == (0, "", "")


async def test_a_session_started_while_hands_is_off_is_still_recorded_for_when_it_starts(home: Home) -> None:
    await shim(home, START)
    assert read_membership(home, SID).pid == os.getpid()


async def test_a_hands_that_died_is_reported_with_the_socket_and_the_heartbeat(home: Home, dead_pid: Callable[[], int]) -> None:
    beat(home, dead_pid(), timedelta(seconds=1))
    code, stdout, stderr = await shim(home, PROMPT)
    assert (code, stdout) == (1, "")
    assert f"cannot reach the hands daemon at {home.socket}" in stderr
    assert "hands is down" in stderr


async def test_a_hands_that_hung_is_reported(home: Home) -> None:
    beat(home, os.getpid(), timedelta(minutes=1))
    code, _, stderr = await shim(home, ASK)
    assert code == 1
    assert "hands is not responding" in stderr


async def test_a_heartbeat_nothing_can_read_is_reported(home: Home) -> None:
    home.status.write_text("not json")
    code, _, stderr = await shim(home, PROMPT)
    assert code == 1
    assert "hands is unknown" in stderr


async def test_a_hands_still_starting_has_not_served_its_socket_yet_and_costs_the_session_nothing(home: Home) -> None:
    beat(home, os.getpid(), timedelta(seconds=0), pipeline="starting")
    assert await shim(home, ASK) == (0, "", "")


async def test_a_relative_home_is_refused_rather_than_made_in_every_project(home: Home, tmp_path: Path) -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", SHIM_MODULE, cwd=tmp_path, env={**os.environ, "HANDS_HOME": "relhome"},
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate(json.dumps(START).encode())
    assert process.returncode == 1
    assert "HANDS_HOME must be an absolute path, got 'relhome'" in stderr.decode()
    assert list(tmp_path.iterdir()) == []


async def test_a_hands_that_says_it_is_up_but_does_not_answer_is_reported(home: Home) -> None:
    beat(home, os.getpid(), timedelta(seconds=0))
    code, _, stderr = await shim(home, PROMPT)
    assert code == 1
    assert "cannot reach the hands daemon" in stderr and "hands is up" in stderr


def launch(home: Home, cwd: Path, path: str) -> subprocess.CompletedProcess[bytes]:
    """The plugin's hook as Claude Code spawns it: its launcher, run directly in the session's directory, with no venv."""
    environment = {"HANDS_HOME": str(home.root), "PATH": path, "HOME": str(cwd)}
    return subprocess.run([PLUGIN_ROOT / LAUNCHER, "-m", SHIM_MODULE], input=json.dumps(START).encode(), env=environment, cwd=cwd, capture_output=True)


@pytest.fixture
def python312(tmp_path: Path) -> str:
    """A PATH whose only Python new enough is a python3.12 outside any venv; /usr/bin's python3 on macOS is 3.9."""
    interpreters = tmp_path / "bin"
    interpreters.mkdir()
    (interpreters / "python3.12").symlink_to(Path(getattr(sys, "_base_executable", sys.executable)).resolve())
    return f"/usr/bin:/bin:{interpreters}"


def test_the_plugin_launcher_runs_the_shim_from_the_plugin_as_the_process_claude_code_spawned(home: Home, tmp_path: Path, python312: str) -> None:
    ran = launch(home, tmp_path, python312)
    assert (ran.returncode, ran.stdout, ran.stderr) == (0, b"", b"")
    # Spawned directly, as Claude Code spawns an exec-form hook, so the pid recorded must be this process's.
    assert read_membership(home, SID).pid == os.getpid()


def test_a_project_s_own_modules_cannot_stand_in_for_the_shim_s(home: Home, tmp_path: Path, python312: str) -> None:
    project = tmp_path / "project"
    (project / "hands").mkdir(parents=True)
    (project / "json.py").write_text("raise SystemExit('shadowed json')\n")
    (project / "hands" / "__init__.py").write_text("raise SystemExit('shadowed hands')\n")
    ran = launch(home, project, python312)
    assert (ran.returncode, ran.stdout, ran.stderr) == (0, b"", b"")


def test_with_no_python_new_enough_the_launcher_says_so(home: Home, tmp_path: Path) -> None:
    ran = launch(home, tmp_path, "/usr/bin:/bin")
    assert ran.returncode == 1
    assert ran.stderr.startswith(b"hands: no Python 3.12 or newer on PATH")


async def test_a_start_records_membership_and_joins_the_registry(home: Home, sessions: Sessions) -> None:
    assert await shim(home, START) == (0, "", "")
    assert await shim(home, PROMPT) == (0, "", "")
    membership = Membership(SID, pid=os.getpid(), cwd=Path("/code/a"), transcript=Path("/nowhere/t.jsonl"))
    assert [listing.session for listing in sessions.live()] == [Session(membership, Submitted(since=10.0), mode=None, turn=PromptId("p1"))]
    assert home.membership(SID).exists()


async def test_an_end_removes_membership_and_leaves_the_listing(home: Home, sessions: Sessions) -> None:
    await shim(home, START)
    assert await shim(home, END) == (0, "", "")
    assert not home.membership(SID).exists()
    assert sessions.live() == []


async def test_a_hook_the_daemon_refuses_exits_nonzero_with_its_reason(home: Home, sessions: Sessions) -> None:
    code, _, stderr = await shim(home, {**COMMON, "hook_event_name": "PreCompact"})
    assert code == 1
    assert "refused this hook (400): hook event 'PreCompact' is not one hands handles" in stderr
    assert sessions.live() == []


async def test_a_second_daemon_will_not_take_a_live_socket(home: Home, sessions: Sessions) -> None:
    with pytest.raises(RuntimeError, match="already listening"):
        await serve_hooks(home, Sessions(60.0, clock=lambda: 0.0, record=lambda _: None))
    await shim(home, START)
    assert [listing.session.state for listing in sessions.live()] == [Idle()]


async def test_a_socket_left_by_a_dead_daemon_is_reclaimed(home: Home) -> None:
    # A daemon that died without cleaning up leaves a socket file nothing listens on.
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(home.socket))
    dead.close()
    assert home.socket.exists()
    registry = Sessions(60.0, clock=lambda: 0.0, record=lambda _: None)
    runner = await serve_hooks(home, registry)
    try:
        assert await shim(home, START) == (0, "", "")
        assert len(registry.live()) == 1
    finally:
        await runner.cleanup()


async def test_a_session_started_under_fritter_records_where_to_type_into_it(home: Home, sessions: Sessions) -> None:
    # fritter publishes its socket in the environment of the process it wrapped, and this
    # hook is a child of that process, so the address arrives without either side naming
    # a path the other has to guess.
    assert await shim(home, START, fritter="/tmp/fritter-abc.sock") == (0, "", "")
    [listing] = sessions.live()
    assert listing.session.membership.fritter == Path("/tmp/fritter-abc.sock")


async def test_a_session_started_outside_fritter_records_no_way_to_type_into_it(home: Home, sessions: Sessions) -> None:
    assert await shim(home, START) == (0, "", "")
    [listing] = sessions.live()
    assert listing.session.membership.fritter is None


async def test_an_empty_fritter_address_is_no_address(home: Home, sessions: Sessions) -> None:
    # An exported-but-empty variable is how a shell hands on a value it does not have.
    assert await shim(home, START, fritter="") == (0, "", "")
    [listing] = sessions.live()
    assert listing.session.membership.fritter is None
