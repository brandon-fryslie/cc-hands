"""The shim against a real hook socket: it records membership, posts, and fails loudly."""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import socket
from collections.abc import AsyncIterator, Iterator, Mapping
from pathlib import Path

import pytest

from hands.core.session import Idle, Membership, Session, SessionId, TmuxPane, Working
from hands.sessions.home import Home
from hands.sessions.registry import Sessions
from hands.sessions.server import serve_hooks

SID = SessionId("0f1e2d3c-aaaa-bbbb-cccc-000000000001")
COMMON = {"session_id": SID, "transcript_path": "/nowhere/t.jsonl", "cwd": "/code/a"}
START = {**COMMON, "hook_event_name": "SessionStart", "source": "startup"}
PROMPT = {**COMMON, "hook_event_name": "UserPromptSubmit", "prompt": "hi"}
END = {**COMMON, "hook_event_name": "SessionEnd", "reason": "other"}


@pytest.fixture
def home() -> Iterator[Home]:
    # A unix socket path is capped near 104 bytes on macOS, so not under pytest's long tmp_path.
    root = Path(tempfile.mkdtemp(prefix="hands-"))
    yield Home(root)
    shutil.rmtree(root)


@pytest.fixture
async def sessions(home: Home) -> AsyncIterator[Sessions]:
    registry = Sessions(permission_deadline=60.0, clock=lambda: 10.0)
    runner = await serve_hooks(home, registry)
    yield registry
    await runner.cleanup()


async def shim(home: Home, payload: Mapping[str, object]) -> tuple[int | None, str]:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "hands.sessions.shim",
        str(home.root),
        stdin=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "TMUX_PANE": "%7"},
    )
    _, stderr = await process.communicate(json.dumps(payload).encode())
    return process.returncode, stderr.decode()


async def test_with_no_daemon_the_shim_exits_nonzero_naming_the_socket(home: Home) -> None:
    code, stderr = await shim(home, PROMPT)
    assert code == 1
    assert f"cannot reach the hands daemon at {home.socket}" in stderr


async def test_a_start_records_membership_and_joins_the_registry(home: Home, sessions: Sessions) -> None:
    assert await shim(home, START) == (0, "")
    assert await shim(home, PROMPT) == (0, "")
    membership = Membership(SID, pid=os.getpid(), pane=TmuxPane("%7"), cwd=Path("/code/a"), transcript=Path("/nowhere/t.jsonl"))
    assert [listing.session for listing in sessions.live()] == [Session(membership, Working(since=10.0))]
    assert home.membership(SID).exists()


async def test_an_end_removes_membership_and_leaves_the_listing(home: Home, sessions: Sessions) -> None:
    await shim(home, START)
    assert await shim(home, END) == (0, "")
    assert not home.membership(SID).exists()
    assert sessions.live() == []


async def test_a_hook_the_daemon_refuses_exits_nonzero_with_its_reason(home: Home, sessions: Sessions) -> None:
    code, stderr = await shim(home, {**COMMON, "hook_event_name": "PreCompact"})
    assert code == 1
    assert "refused this hook (400): hook event 'PreCompact' is not one hands handles" in stderr
    assert sessions.live() == []


async def test_a_second_daemon_will_not_take_a_live_socket(home: Home, sessions: Sessions) -> None:
    with pytest.raises(RuntimeError, match="already listening"):
        await serve_hooks(home, Sessions(60.0, clock=lambda: 0.0))
    await shim(home, START)
    assert [listing.session.state for listing in sessions.live()] == [Idle()]


async def test_a_socket_left_by_a_dead_daemon_is_reclaimed(home: Home) -> None:
    # A daemon that died without cleaning up leaves a socket file nothing listens on.
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(home.socket))
    dead.close()
    assert home.socket.exists()
    registry = Sessions(60.0, clock=lambda: 0.0)
    runner = await serve_hooks(home, registry)
    try:
        assert await shim(home, START) == (0, "")
        assert len(registry.live()) == 1
    finally:
        await runner.cleanup()
