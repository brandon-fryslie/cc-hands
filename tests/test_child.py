"""A child run to its end inside a time limit, and never left running by a caller that stops waiting on it."""

import asyncio
import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from loguru import logger

from hands.sessions import child
from hands.sessions.child import Ran, run


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list[subprocess.Popen[bytes]]:
    """Every child `run` starts, so a test can ask after it once `run` has let it go."""
    seen: list[subprocess.Popen[bytes]] = []

    class Seen(subprocess.Popen[bytes]):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            seen.append(self)

    monkeypatch.setattr(child.subprocess, "Popen", Seen)
    return seen


@pytest.fixture
def logged() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    yield messages
    logger.remove(sink)


def reaped(process: subprocess.Popen[bytes]) -> bool:
    return process.returncode is not None


async def test_a_child_s_output_and_exit_are_what_it_ran_to() -> None:
    assert await run("sh", "-c", "echo out; echo err >&2; exit 3", timeout=5) == Ran(3, b"out\n", b"err\n")


async def test_a_child_that_runs_past_its_time_is_killed_and_reaped(started: list[subprocess.Popen[bytes]], logged: list[str]) -> None:
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        await run("sleep", "30", timeout=0.2)
    assert time.monotonic() - start < 2
    assert reaped(started[0])
    assert any(f"killed sleep ({started[0].pid})" in line and "ran past 0.2s" in line for line in logged)


async def test_what_a_child_started_is_killed_with_it(tmp_path: Path) -> None:
    pid = tmp_path / "pid"
    with pytest.raises(TimeoutError):
        await run("sh", "-c", f"sleep 30 & echo $! > {pid}; wait", timeout=0.5)
    grandchild = int(pid.read_text())
    # Killed, it is reparented to launchd and reaped there; give that a moment before asking after it.
    for _ in range(50):
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"the sleep {grandchild} its timed-out parent started is still running")


async def test_a_child_whose_caller_stops_waiting_is_killed_and_reaped(started: list[subprocess.Popen[bytes]], logged: list[str]) -> None:
    running = asyncio.create_task(run("sleep", "30", timeout=30))
    await asyncio.sleep(0.1)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert reaped(started[0])
    assert any(f"killed sleep ({started[0].pid})" in line and "its caller stopped waiting" in line for line in logged)


async def test_a_child_that_ended_as_it_was_killed_still_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    # What macOS says to a kill of a group whose processes have all exited but are not yet reaped.
    def exited(_pgid: int, _signal: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(child.os, "killpg", exited)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await run("sleep", "0.4", timeout=0.1)
    assert time.monotonic() - started >= 0.3, "it left before the child it could not kill was reaped"


async def test_a_child_that_cannot_start_says_so() -> None:
    with pytest.raises(OSError):
        await run("/nonexistent/hands-no-such-program", timeout=5)
