"""When hands' processes started, from the kernel, and whether a pid still names the process it named before.

A pid is a number the OS hands out again. A running process under a remembered pid is the remembered process only
if it was already running when it was remembered; one that started later took the number of one that is dead. The
session sweep and the daemon's heartbeat both ask that, and both answer it here.
"""

import ctypes
import ctypes.util
import errno
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
    """When each pid's process started, in wall-clock seconds; a pid with no process of this user's is absent.

    One syscall a pid, about a microsecond, so it is asked on the event loop and once a second from the menu bar alike.
    """
    return {pid: start for pid in pids if (start := _process_start(pid)) is not None}


class _BSDInfo(ctypes.Structure):
    """`struct proc_bsdinfo`, from <sys/proc_info.h>: read for the start time at its end."""

    _fields_ = [  # pyright: ignore[reportUnannotatedClassAttribute]
        *((name, ctypes.c_uint32) for name in ("flags", "status", "xstatus", "pid", "ppid", "uid", "gid", "ruid", "rgid", "svuid", "svgid", "rfu_1")),
        ("comm", ctypes.c_char * 16),
        ("name", ctypes.c_char * 32),
        *((name, ctypes.c_uint32) for name in ("nfiles", "pgid", "pjobc", "e_tdev", "e_tpgid")),
        ("nice", ctypes.c_int32),
        ("start_tvsec", ctypes.c_uint64),
        ("start_tvusec", ctypes.c_uint64),
    ]


_PROC_PIDTBSDINFO = 3
_libproc = ctypes.CDLL(ctypes.util.find_library("proc"), use_errno=True)


def _process_start(pid: int) -> float | None:
    info = _BSDInfo()
    if _libproc.proc_pidinfo(pid, _PROC_PIDTBSDINFO, 0, ctypes.byref(info), ctypes.sizeof(info)) == ctypes.sizeof(info):
        return float(info.start_tvsec) + float(info.start_tvusec) / 1_000_000
    match ctypes.get_errno():
        case errno.ESRCH:
            return None  # no process has the pid
        case errno.EPERM:
            return None  # another user's process has it; hands' own processes all run as this user
        case other:
            # [LAW:no-silent-failure] anything else is the kernel refusing a question it always answers.
            raise OSError(other, f"proc_pidinfo could not say when pid {pid} started: {os.strerror(other)}")
