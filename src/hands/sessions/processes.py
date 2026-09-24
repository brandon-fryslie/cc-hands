"""When running processes started, from the process table, and whether a pid still names the process it named then.

A pid is a number the OS hands out again. A running process under a remembered pid is the remembered process only
if it was already running when it was remembered; one that started later took the number of one that is dead. The
session sweep and the daemon's heartbeat both ask that, and both answer it here.
"""

import asyncio
import subprocess
import time
from collections.abc import Collection, Mapping

# ps reports elapsed time in whole seconds, and it is read a moment after the clock is.
START_SLACK_SECONDS = 2.0


def still_running(pid: int, seen_at: float, starts: Mapping[int, float]) -> bool:
    """Whether the process running under pid now is the one that was running under it at seen_at (wall-clock seconds)."""
    start = starts.get(pid)
    # [LAW:single-enforcer] the one test for a reused pid: a process that started after seen_at took the number of
    # the one that was seen, which is dead.
    return start is not None and start <= seen_at + START_SLACK_SECONDS


async def process_starts(pids: Collection[int]) -> dict[int, float]:
    """When each running pid started, in wall-clock seconds; a pid that is not running is absent."""
    if not pids:
        return {}
    ps = await asyncio.create_subprocess_exec(*_ps(pids), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await ps.communicate()
    return _starts(out, err, ps.returncode, time.time())


def process_starts_now(pids: Collection[int]) -> dict[int, float]:
    """As process_starts, for a caller with no event loop: `hands status` and the menu-bar indicator."""
    if not pids:
        return {}
    ps = subprocess.run(_ps(pids), capture_output=True, check=False)
    return _starts(ps.stdout, ps.stderr, ps.returncode, time.time())


def _ps(pids: Collection[int]) -> list[str]:
    return ["ps", "-o", "pid=,etime=", "-p", ",".join(str(pid) for pid in sorted(pids))]


def _starts(out: bytes, err: bytes, returncode: int | None, now: float) -> dict[int, float]:
    # ps exits 1, printing nothing, when none of the pids is running.
    if err or returncode not in (0, 1):
        raise RuntimeError(f"ps exited {returncode} asking which processes are running: {err.decode(errors='replace').strip()}")
    starts: dict[int, float] = {}
    for line in out.decode().splitlines():
        pid, etime = line.split()
        starts[int(pid)] = now - elapsed_seconds(etime)
    return starts


def elapsed_seconds(etime: str) -> int:
    """ps's elapsed time, [[dd-]hh:]mm:ss, in seconds."""
    days, _, clock = etime.rpartition("-")
    seconds = 0
    for part in clock.split(":"):
        seconds = seconds * 60 + int(part)
    return (int(days) if days else 0) * 86400 + seconds
