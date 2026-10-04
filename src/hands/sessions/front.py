# pyright: basic, reportAttributeAccessIssue=false
"""Reads what is in front on the Mac's screen: the window list for the app, the app itself for its front tab, tmux for
the pane each of its clients shows, and the kernel for the terminals each session runs under.

hands is tied to no one terminal. An app that can say which of its tabs is in front by AppleScript is asked; any other
shows every terminal running under it, which names the session in front whenever it holds only one.
"""

import os
import shutil
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import Quartz
from AppKit import NSRunningApplication
from Foundation import NSAppleScript

from hands.core.front import Candidate, FrontUnread, InFront, Screen, in_front
from hands.core.session import SessionId
from hands.sessions.terminals import Process, process_table

# Each app that says by AppleScript which terminal its front tab shows, by bundle id. Only the app in front is asked:
# telling an app anything launches it.
_FRONT_TAB: Mapping[str, str] = {
    "com.googlecode.iterm2": 'tell application id "com.googlecode.iterm2" to tty of current session of current window',
    "com.apple.Terminal": 'tell application id "com.apple.Terminal" to tty of selected tab of front window',
}
# Compiled once, on the first ask: the first costs ~170 ms, the ones after it ~5 ms (measured with iTerm2 3.6).
_scripts: dict[str, NSAppleScript] = {}

# What tmux says, on stderr, for a socket no server listens on any more: a server that exited leaves its socket behind.
_NO_SERVER = ("no server running on", "error connecting to")


class _Unread(Exception):
    pass


def read_front(sessions: Mapping[SessionId, tuple[int, str]], environment: Mapping[str, str]) -> InFront:
    """What is in front now, among `sessions`, each a session's pid and its name as it is spoken; `environment` is the
    run's, which says where tmux keeps its sockets and where tmux is."""
    try:
        processes = process_table()
        screen = _screen(processes)
        panes = dict(_panes(environment))
    except _Unread as unread:
        return FrontUnread(str(unread))
    except (OSError, subprocess.TimeoutExpired) as error:
        # [LAW:no-silent-failure] the turn goes without the fact, and the turn's record says why it was left out.
        return FrontUnread(f"{type(error).__name__}: {error}")
    candidates = [Candidate(session, name, frozenset(_terminals(pid, processes))) for session, (pid, name) in sessions.items()]
    return in_front(screen, panes, candidates)


def _screen(processes: Mapping[int, Process]) -> Screen:
    windows = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements, Quartz.kCGNullWindowID)
    # Front to back; layer 0 is apps' own windows, above it the menu bar, the Dock, and panels that float.
    front = next((window for window in windows or () if window[Quartz.kCGWindowLayer] == 0), None)
    if front is None:
        raise _Unread("no window is on screen")
    pid = int(front[Quartz.kCGWindowOwnerPID])
    app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
    name = str(front[Quartz.kCGWindowOwnerName])
    match _FRONT_TAB.get(str(app.bundleIdentifier())) if app is not None else None:
        case str() as source:
            return Screen(name, frozenset({_device(_asked(name, source))}))
        case None:
            return Screen(name, frozenset(process.tty for process in _under(pid, processes) if process.tty is not None))


def _asked(app: str, source: str) -> str:
    script = _scripts.setdefault(source, NSAppleScript.alloc().initWithSource_(source))
    answer, error = script.executeAndReturnError_(None)
    if answer is None:
        raise _Unread(f"{app} did not say which tab is in front: {error.get('NSAppleScriptErrorMessage', error) if error else 'no answer'}")
    return str(answer.stringValue())


def _under(pid: int, processes: Mapping[int, Process]) -> Iterator[Process]:
    """Every process descended from pid."""
    children: dict[int, list[Process]] = {}
    for process in processes.values():
        children.setdefault(process.parent, []).append(process)
    stack = list(children.get(pid, ()))
    while stack:
        process = stack.pop()
        yield process
        stack.extend(children.get(process.pid, ()))


def _terminals(pid: int, processes: Mapping[int, Process]) -> Iterator[int]:
    """Every terminal on pid's line of ancestors, its own first."""
    process = processes.get(pid)
    while process is not None:
        if process.tty is not None:
            yield process.tty
        process = processes.get(process.parent)


def _panes(environment: Mapping[str, str]) -> Iterator[tuple[int, int]]:
    """Each tmux client's terminal, and the terminal of the pane it shows, from every server of this user's."""
    # Where tmux puts a server's socket unless -S names one: $TMUX_TMPDIR, else /tmp, in tmux-<uid>.
    directory = Path(environment.get("TMUX_TMPDIR", "/tmp")) / f"tmux-{os.getuid()}"
    sockets = [path for path in directory.glob("*") if path.is_socket()]
    if not sockets:
        return
    tmux = shutil.which("tmux", path=environment.get("PATH"))
    if tmux is None:
        raise _Unread(f"tmux sockets are in {directory}, and no tmux is on the PATH to ask them which pane each client shows")
    for socket in sockets:
        for line in _clients(tmux, socket):
            client, pane = line.split("\t")
            yield _device(client), _device(pane)


def _clients(tmux: str, socket: Path) -> Sequence[str]:
    asked = subprocess.run([tmux, "-S", str(socket), "list-clients", "-F", "#{client_tty}\t#{pane_tty}"], capture_output=True, text=True, timeout=2)
    if asked.returncode != 0:
        if asked.stderr.startswith(_NO_SERVER):
            return ()
        raise _Unread(f"tmux at {socket} did not list its clients: {asked.stderr.strip()}")
    return asked.stdout.splitlines()


def _device(path: str) -> int:
    try:
        return os.stat(path).st_rdev
    except OSError as error:
        raise _Unread(f"the terminal {path} could not be read: {error}") from error
