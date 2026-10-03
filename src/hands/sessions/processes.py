"""When hands' processes started, from the kernel, and whether a pid still names the process it named before; and
which of this user's processes run at a terminal.

A pid is a number the OS hands out again. A running process under a remembered pid is the remembered process only
if it was already running when it was remembered; one that started later took the number of one that is dead. The
session sweep and the daemon's heartbeat both ask that, and both answer it here.
"""

import ctypes
import ctypes.util
import errno
import os
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

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

    One sysctl a pid, about ten microseconds, so it is asked on the event loop and five times a second from the menu bar alike.
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


@dataclass(frozen=True)
class Terminal:
    """A process of this user's with a controlling terminal: what it runs, and where."""

    pid: int
    executable: Path
    cwd: Path


def terminal_processes() -> list[Terminal]:
    """Every process of this user's that has a controlling terminal: interactive programs, never daemons or apps."""
    return [process for pid in _own_pids() if (process := _terminal(pid)) is not None]


# libproc, the library ps and lsof read: struct proc_bsdinfo (136 bytes, flags first) and struct proc_vnodepathinfo
# (two vnode_info_path of 1176 bytes, the cwd's first, its path after a 152-byte vnode_info).
_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
_PROC_UID_ONLY = 4
_PROC_PIDTBSDINFO, _BSDINFO_SIZE = 3, 136
_PROC_PIDVNODEPATHINFO, _VNODEPATHINFO_SIZE, _CWD_PATH_AT, _MAXPATHLEN = 9, 2352, 152, 1024
_PROC_FLAG_CONTROLT = 0x80
_PATH_ROOM = 4096  # PROC_PIDPATHINFO_MAXSIZE


def _own_pids() -> list[int]:
    # Asked once for the room needed, then with room to spare for processes started between the two calls.
    room = _listpids(None, 0)
    pids = (ctypes.c_int * (room // ctypes.sizeof(ctypes.c_int) + 256))()
    filled = _listpids(pids, ctypes.sizeof(pids))
    # A pid of 0 is an unused slot.
    return [pid for pid in pids[: filled // ctypes.sizeof(ctypes.c_int)] if pid]


def _listpids(into: ctypes.Array[ctypes.c_int] | None, size: int) -> int:
    used = _libproc.proc_listpids(_PROC_UID_ONLY, os.getuid(), into, size)
    if used <= 0:
        failure = ctypes.get_errno()
        raise OSError(failure, f"proc_listpids could not list this user's processes: {os.strerror(failure)}")
    return used


def _terminal(pid: int) -> Terminal | None:
    """The process under pid, if it has a controlling terminal; None if it has none, or has exited since it was listed."""
    try:
        bsd = _pidinfo(pid, _PROC_PIDTBSDINFO, _BSDINFO_SIZE)
        if not ctypes.c_uint32.from_buffer(bsd, 0).value & _PROC_FLAG_CONTROLT:
            return None
        cwd = _pidinfo(pid, _PROC_PIDVNODEPATHINFO, _VNODEPATHINFO_SIZE).raw[_CWD_PATH_AT : _CWD_PATH_AT + _MAXPATHLEN]
        executable = ctypes.create_string_buffer(_PATH_ROOM)
        if _libproc.proc_pidpath(pid, executable, _PATH_ROOM) <= 0:
            _raise_unless_exited(pid, "proc_pidpath")
    except _Exited:
        return None
    return Terminal(
        pid,
        Path(os.fsdecode(executable.value)),
        Path(os.fsdecode(cwd.split(b"\0", 1)[0])),
    )


class _Exited(Exception):
    pass


def _pidinfo(pid: int, flavor: int, size: int) -> ctypes.Array[ctypes.c_char]:
    record = ctypes.create_string_buffer(size)
    filled = _libproc.proc_pidinfo(pid, flavor, 0, record, size)
    if filled <= 0:
        _raise_unless_exited(pid, f"proc_pidinfo flavor {flavor}")
    if filled != size:
        # A record of another size is a kernel whose struct is not the one laid out above; no errno says so.
        raise OSError(f"proc_pidinfo flavor {flavor} gave {filled} bytes for pid {pid}, where this kernel's struct was taken to be {size}")
    return record


def _raise_unless_exited(pid: int, call: str) -> None:
    failure = ctypes.get_errno()
    # A process that exited, or is a zombie waiting on its parent, has no info left: it is no longer running.
    if failure == errno.ESRCH:
        raise _Exited
    # [LAW:no-silent-failure] anything else is the kernel refusing a question about one of this user's own processes.
    raise OSError(failure, f"{call} could not look at pid {pid}: {os.strerror(failure)}")
