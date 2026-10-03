"""Which of this user's processes run at a terminal: what each was started as, under what environment, and where.

Read from the kernel through libproc and sysctl, the sources ps and lsof read.
"""

import ctypes
import ctypes.util
import errno
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Terminal:
    """A process of this user's with a controlling terminal."""

    pid: int
    parent: int
    # The program it was started as, with links resolved as they are now: a file deleted since, as an updater prunes
    # old versions, is still named.
    executable: Path
    cwd: Path
    environment: Mapping[str, str]


def terminal_processes() -> list[Terminal]:
    """Every process of this user's that has a controlling terminal: interactive programs, never daemons or apps."""
    return [process for pid in _own_pids() if (process := _terminal(pid)) is not None]


# libproc: struct proc_bsdinfo (136 bytes: flags, status, exit status, pid, then the parent's pid) and struct
# proc_vnodepathinfo (two vnode_info_path of 1176 bytes, the cwd's first, its path after a 152-byte vnode_info).
_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
_PROC_UID_ONLY = 4
_PROC_PIDTBSDINFO, _BSDINFO_SIZE, _PARENT_AT = 3, 136, 16
_PROC_PIDVNODEPATHINFO, _VNODEPATHINFO_SIZE, _CWD_PATH_AT, _MAXPATHLEN = 9, 2352, 152, 1024
_PROC_FLAG_CONTROLT = 0x80
# kern.procargs2.<pid>: argc, the path the process was exec'd by as execve was given it, padding, its arguments, and
# its environment, each string ending in a NUL.
_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_KERN_PROCARGS2 = (1, 49)  # CTL_KERN, KERN_PROCARGS2


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
        cwd = Path(os.fsdecode(_string(_pidinfo(pid, _PROC_PIDVNODEPATHINFO, _VNODEPATHINFO_SIZE).raw[_CWD_PATH_AT : _CWD_PATH_AT + _MAXPATHLEN])))
        executable, environment = _started_as(pid)
    except _Exited:
        return None
    # A path exec'd relative to the directory the process was started in; a session keeps that directory.
    return Terminal(pid, ctypes.c_uint32.from_buffer(bsd, _PARENT_AT).value, (cwd / executable).resolve(), cwd, environment)


def _started_as(pid: int) -> tuple[Path, dict[str, str]]:
    """The path pid was exec'd by, and its environment."""
    mib = (ctypes.c_int * 3)(*_KERN_PROCARGS2, pid)
    size = ctypes.c_size_t(0)
    if _libc.sysctl(mib, len(mib), None, ctypes.byref(size), None, 0) != 0:
        _raise_unless_exited(pid, "kern.procargs2")
    record = ctypes.create_string_buffer(size.value)
    if _libc.sysctl(mib, len(mib), record, ctypes.byref(size), None, 0) != 0:
        _raise_unless_exited(pid, "kern.procargs2")
    raw = record.raw[: size.value]
    argc = ctypes.c_int.from_buffer_copy(raw, 0).value
    path, rest = raw[ctypes.sizeof(ctypes.c_int) :].split(b"\0", 1)
    # The environment runs from after the arguments to the first empty string.
    after = rest.lstrip(b"\0").split(b"\0")[argc:]
    entries = after[: after.index(b"")] if b"" in after else after
    return Path(os.fsdecode(path)), {name: value for name, _, value in (os.fsdecode(entry).partition("=") for entry in entries)}


def _string(raw: bytes) -> bytes:
    return raw.split(b"\0", 1)[0]


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
    """Raised after `call` failed for pid: _Exited when the process is gone, or a zombie, and OSError when it runs."""
    failure = ctypes.get_errno()
    # [LAW:single-enforcer] the one test for an exit, whichever call failed: libproc has nothing on a process that
    # exited, or is a zombie waiting on its parent.
    probe = ctypes.create_string_buffer(_BSDINFO_SIZE)
    if _libproc.proc_pidinfo(pid, _PROC_PIDTBSDINFO, 0, probe, _BSDINFO_SIZE) <= 0 and ctypes.get_errno() == errno.ESRCH:
        raise _Exited
    # [LAW:no-silent-failure] anything else is the kernel refusing a question about one of this user's own processes.
    raise OSError(failure, f"{call} could not look at pid {pid}: {os.strerror(failure)}")
