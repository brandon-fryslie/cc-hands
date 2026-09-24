"""When hands' processes started, from the kernel, and whether a pid still names the process it named before.

A pid is a number the OS hands out again. A running process under a remembered pid is the remembered process only
if it was already running when it was remembered; one that started later took the number of one that is dead. The
session sweep and the daemon's heartbeat both ask that, and both answer it here.
"""

import ctypes
import ctypes.util
import os
from collections.abc import Collection, Mapping

from hands.sessions.payload import Rejected

# The largest pid macOS hands out.
PID_MAX = 99999

# A process starts before it can write down that it is running, so its start is never later than that record. The
# slack covers a wall clock stepped back between the two, not pid reuse, which on macOS does not come round in a second.
START_SLACK_SECONDS = 1.0


def parse_pid(number: int) -> int:
    """A number read from a file, as a pid some process could have."""
    # [LAW:parse-dont-validate] refused here, so nothing ever asks the kernel about a number no process can have.
    if not 0 < number <= PID_MAX:
        raise Rejected(f"pid {number} is not a process id")
    return number


def still_running(pid: int, seen_at: float, starts: Mapping[int, float]) -> bool:
    """Whether the process running under pid now is the one that was running under it at seen_at (wall-clock seconds)."""
    start = starts.get(pid)
    # [LAW:single-enforcer] the one test for a reused pid: a process that started after seen_at took the number of
    # the one that was seen, which is dead.
    return start is not None and start <= seen_at + START_SLACK_SECONDS


def process_starts(pids: Collection[int]) -> dict[int, float]:
    """When each pid's process started, in wall-clock seconds; a pid no process has is absent.

    One sysctl a pid, about ten microseconds, so it is asked on the event loop and once a second from the menu bar alike.
    """
    return {pid: start for pid in pids if (start := _process_start(pid)) is not None}


# kern.proc.pid.<pid>: the kernel's `struct kinfo_proc`, the record ps itself reads. It opens with the process's start,
# a `struct timeval`: 8 bytes of seconds, then 4 of microseconds.
_KERN_PROC_PID = (1, 14, 1)  # CTL_KERN, KERN_PROC, KERN_PROC_PID
_KINFO_PROC_ROOM = 1024  # more than the 648 bytes the record takes, so a larger one in a later macOS still fits
_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


def _process_start(pid: int) -> float | None:
    mib = (ctypes.c_int * 4)(*_KERN_PROC_PID, pid)
    record = ctypes.create_string_buffer(_KINFO_PROC_ROOM)
    size = ctypes.c_size_t(_KINFO_PROC_ROOM)
    if _libc.sysctl(mib, len(mib), record, ctypes.byref(size), None, 0) != 0:
        failure = ctypes.get_errno()
        # [LAW:no-silent-failure] the kernel refusing a question it always answers is raised, never taken for "gone".
        raise OSError(failure, f"sysctl could not say when pid {pid} started: {os.strerror(failure)}")
    if size.value == 0:
        return None  # no process has the pid; the kernel answers with an empty record
    return ctypes.c_int64.from_buffer(record, 0).value + ctypes.c_int32.from_buffer(record, 8).value / 1_000_000
