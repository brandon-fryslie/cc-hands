"""A child run to its end inside a time limit, and never left running by a caller that stops waiting on it."""

import asyncio
import os
import subprocess
import time

import pytest

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


def reaped(process: subprocess.Popen[bytes]) -> bool:
    # A child killed and not yet reaped still answers signal 0, as a zombie; only a reaped one is not there at all.
    try:
        os.kill(process.pid, 0)
    except ProcessLookupError:
        return True
    return False


async def test_a_child_s_output_and_exit_are_what_it_ran_to() -> None:
    assert await run("sh", "-c", "echo out; echo err >&2; exit 3", timeout=5) == Ran(3, b"out\n", b"err\n")


async def test_a_child_that_runs_past_its_time_is_killed_and_reaped(started: list[subprocess.Popen[bytes]]) -> None:
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        await run("sleep", "30", timeout=0.2)
    assert time.monotonic() - start < 2
    assert reaped(started[0])


async def test_a_child_whose_caller_stops_waiting_is_killed_and_reaped(started: list[subprocess.Popen[bytes]]) -> None:
    running = asyncio.create_task(run("sleep", "30", timeout=30))
    await asyncio.sleep(0.1)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert reaped(started[0])


async def test_a_child_that_cannot_start_says_so() -> None:
    with pytest.raises(OSError):
        await run("/nonexistent/hands-no-such-program", timeout=5)
