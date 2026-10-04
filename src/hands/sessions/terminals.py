"""Which of this user's processes run at a terminal: what each was started as, with what arguments, under what
environment, where, and on which terminal; and whether one reads and writes its terminal.

Read from the kernel through libproc and sysctl, the sources ps and lsof read.
"""

import ctypes
import ctypes.util
import errno
import os
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn


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
    # Its arguments after the program, as it was started with them.
    arguments: tuple[str, ...]
    # The device of its controlling terminal, as Process.tty.
    tty: int


def terminal_processes() -> list[Terminal]:
    """Every process of this user's that has a controlling terminal: interactive programs, never daemons or apps."""
    own = os.geteuid()
    return [terminal for process in process_table().values() if process.uid == own and process.tty is not None and (terminal := _terminal(process, process.tty)) is not None]


def attended(process: Terminal) -> bool:
    """Whether its stdin and stdout are both its terminal, as `[ -t 0 ] && [ -t 1 ]` finds them at a terminal: its
    controlling terminal's device, or /dev/tty, which is that terminal under the name each process has for its own.
    False for one that has exited since it was listed, which reads nothing."""
    terminal = {process.tty, os.stat("/dev/tty").st_rdev}
    try:
        return all(_device(process.pid, fd) in terminal for fd in (0, 1))
    except _Exited:
        return False


@dataclass(frozen=True)
class Process:
    """A process of any user's: its parent, its effective user, and the device number of the terminal it is controlled
    by, which is `st_rdev` of that terminal's /dev path; None for a process with no controlling terminal, as an app or a
    daemon has none."""

    pid: int
    parent: int
    uid: int
    tty: int | None


def process_table() -> dict[int, Process]:
    """Every process on the Mac but the kernel's, by pid, as it stands now: root's too, as the /usr/bin/login a terminal app starts each
    tab through, so a line of parents runs unbroken from a session up to its app."""
    mib = (ctypes.c_int * 3)(*_KERN_PROC_ALL)
    size = ctypes.c_size_t(0)
    if _libc.sysctl(mib, len(mib), None, ctypes.byref(size), None, 0) != 0:
        _raise_errno("kern.proc.all")
    # With room to spare for processes started between the two calls.
    size.value += 256 * _KINFO_PROC_SIZE
    table = ctypes.create_string_buffer(size.value)
    if _libc.sysctl(mib, len(mib), table, ctypes.byref(size), None, 0) != 0:
        _raise_errno("kern.proc.all")
    if size.value % _KINFO_PROC_SIZE:
        # A record of another size is a kernel whose struct is not the one laid out above; no errno says so.
        raise OSError(f"kern.proc.all gave {size.value} bytes, not a whole number of this kernel's {_KINFO_PROC_SIZE}-byte kinfo_proc")
    processes = (_process(flag, pid, uid, parent, tty) for flag, pid, uid, parent, tty in _KINFO_PROC.iter_unpack(table.raw[: size.value]))
    # Not kernel_task, pid 0, which is its own parent: without it every line of parents ends, at launchd.
    return {process.pid: process for process in processes if process.pid != 0}


def _process(flag: int, pid: int, uid: int, parent: int, tty: int) -> Process:
    return Process(pid, parent, uid, tty if flag & _P_CONTROLT else None)


# sysctl kern.proc.all: one struct kinfo_proc of 648 bytes per process; its extern_proc's p_flag and p_pid, then its
# eproc's effective uid (e_ucred.cr_uid), parent's pid, and controlling terminal's device, at these offsets.
_KERN_PROC_ALL = (1, 14, 0)  # CTL_KERN, KERN_PROC, KERN_PROC_ALL
_KINFO_PROC = struct.Struct("=32xi4xi376xI136xi8xi72x")
_KINFO_PROC_SIZE = _KINFO_PROC.size
_P_CONTROLT = 0x2
# libproc: struct proc_bsdinfo (136 bytes), asked only whether a process still exists, struct proc_vnodepathinfo
# (two vnode_info_path of 1176 bytes, the cwd's first, its path after a 152-byte vnode_info), and struct
# vnode_fdinfowithpath (a 24-byte proc_fileinfo, then a vnode_info_path), whose vnode's vinfo_stat has st_rdev at 116, a dev_t as e_tdev is.
_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
_PROC_PIDTBSDINFO, _BSDINFO_SIZE = 3, 136
_PROC_PIDVNODEPATHINFO, _VNODEPATHINFO_SIZE, _CWD_PATH_AT, _MAXPATHLEN = 9, 2352, 152, 1024
_PROC_PIDFDVNODEPATHINFO, _FDVNODEPATHINFO_SIZE = 2, 1200
_FD_RDEV = struct.Struct("=140xi1056x")
# kern.procargs2.<pid>: argc, the path the process was exec'd by as execve was given it, padding, its arguments, and
# its environment, each string ending in a NUL.
_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_KERN_PROCARGS2 = (1, 49)  # CTL_KERN, KERN_PROCARGS2


def _raise_errno(what: str) -> NoReturn:
    failure = ctypes.get_errno()
    raise OSError(failure, f"{what} could not be read: {os.strerror(failure)}")


def _terminal(process: Process, tty: int) -> Terminal | None:
    """The process at a terminal, with where it runs and what it was started as; None if it has exited since it was listed."""
    try:
        cwd = Path(os.fsdecode(_string(_pidinfo(process.pid, _PROC_PIDVNODEPATHINFO, _VNODEPATHINFO_SIZE).raw[_CWD_PATH_AT : _CWD_PATH_AT + _MAXPATHLEN])))
        executable, arguments, environment = _started_as(process.pid)
    except _Exited:
        return None
    # A path exec'd relative to the directory the process was started in, which a session leaves for a worktree and
    # the kernel keeps no record of. [LAW:one-source-of-truth] The PWD its shell started it with names that directory,
    # in the one record the path itself is read from; a program started with none, or one that is no absolute path, is
    # taken to be where it started.
    return Terminal(process.pid, process.parent, (cwd / environment.get("PWD", "") / executable).resolve(), cwd, environment, arguments, tty)


def _device(pid: int, fd: int) -> int | None:
    """The device a file pid has open as fd is, as `st_rdev`; None for a pipe, a socket, or an fd it has not open."""
    record = ctypes.create_string_buffer(_FDVNODEPATHINFO_SIZE)
    filled = _libproc.proc_pidfdinfo(pid, fd, _PROC_PIDFDVNODEPATHINFO, record, _FDVNODEPATHINFO_SIZE)
    if filled <= 0:
        failure = ctypes.get_errno()
        # EBADF is an fd that is no file, or no fd at all: nothing a terminal is.
        if failure == errno.EBADF:
            return None
        _raise_unless_exited(pid, f"proc_pidfdinfo of fd {fd}")
    if filled != _FDVNODEPATHINFO_SIZE:
        raise OSError(f"proc_pidfdinfo gave {filled} bytes for fd {fd} of pid {pid}, where this kernel's struct was taken to be {_FDVNODEPATHINFO_SIZE}")
    # A regular file's st_rdev is 0, never a terminal's device.
    (device,) = _FD_RDEV.unpack(record.raw)
    return device


def _started_as(pid: int) -> tuple[Path, tuple[str, ...], dict[str, str]]:
    """The path pid was exec'd by, its arguments after the program, and its environment."""
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
    strings = rest.lstrip(b"\0").split(b"\0")
    # The environment runs from after the arguments to the first empty string.
    after = strings[argc:]
    entries = after[: after.index(b"")] if b"" in after else after
    arguments = tuple(os.fsdecode(argument) for argument in strings[1:argc])
    return Path(os.fsdecode(path)), arguments, {name: value for name, _, value in (os.fsdecode(entry).partition("=") for entry in entries)}


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


def _raise_unless_exited(pid: int, call: str) -> NoReturn:
    """Raised after `call` failed for pid: _Exited when the process is gone, or a zombie, and OSError when it runs."""
    failure = ctypes.get_errno()
    # [LAW:single-enforcer] the one test for an exit, whichever call failed: libproc has nothing on a process that
    # exited, or is a zombie waiting on its parent.
    probe = ctypes.create_string_buffer(_BSDINFO_SIZE)
    if _libproc.proc_pidinfo(pid, _PROC_PIDTBSDINFO, 0, probe, _BSDINFO_SIZE) <= 0 and ctypes.get_errno() == errno.ESRCH:
        raise _Exited
    # [LAW:no-silent-failure] anything else is the kernel refusing a question about one of this user's own processes.
    raise OSError(failure, f"{call} could not look at pid {pid}: {os.strerror(failure)}")
