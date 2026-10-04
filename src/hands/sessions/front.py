# pyright: basic, reportAttributeAccessIssue=false
"""Reads what is in front on the Mac's screen: LaunchServices for the app that is, the app itself for its front tab, tmux
for the pane each of its clients shows, and the kernel for the terminals each session runs under.

hands is tied to no one terminal. An app that can say which of its tabs is in front by AppleScript is asked; any other
shows every terminal running under it, which names the session in front whenever it holds only one.
"""

import os
import re
import shutil
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from Foundation import NSAppleScript

from hands.core.front import Candidate, FrontUnread, InFront, Screen, in_front
from hands.core.session import SessionId
from hands.sessions.child import run
from hands.sessions.terminals import Process, process_table
from hands.threads import SerialThread

# Each app that says by AppleScript which terminal its front tab shows, by bundle id. Only the app in front is asked:
# telling an app anything launches it. The time limit is the Apple event's own, which otherwise waits two minutes on an
# app that does not answer, or on the user deciding whether hands may ask it at all.
_FRONT_TAB: Mapping[str, str] = {
    bundle: f'with timeout of 2 seconds\ntell application id "{bundle}" to {asked}\nend timeout'
    for bundle, asked in (("com.googlecode.iterm2", "tty of current session of current window"), ("com.apple.Terminal", "tty of selected tab of front window"))
}
# [LAW:no-shared-mutable-globals] NSAppleScript is not thread-safe: every script is compiled and run on this one thread,
# which owns the compiled scripts, and which a stopping daemon never waits on. Compiled once, on the first ask: the first
# costs ~170 ms, the ones after it ~5 ms (measured with iTerm2 3.6).
_applescript = SerialThread("AppleScript")
_scripts: dict[str, NSAppleScript] = {}

_LSAPPINFO = "/usr/bin/lsappinfo"
# lsappinfo info -only names one field a line: "pid"=93900, "CFBundleIdentifier"="com.google.Chrome".
_FIELD = re.compile(r'^"(\w+)"=(?:"(.*)"|(\d+))$', re.MULTILINE)

# What tmux says, on stderr, for a socket no server listens on any more: a server that exited leaves its socket behind.
_NO_SERVER = (b"no server running on", b"error connecting to")


class _Unread(Exception):
    pass


@dataclass(frozen=True)
class _App:
    pid: int
    bundle: str | None
    name: str


async def read_front(sessions: Mapping[SessionId, tuple[int, str]], environment: Mapping[str, str]) -> InFront:
    """What is in front now, among `sessions`, each a session's pid and its name as it is spoken; `environment` is the
    run's, which says where tmux keeps its sockets and where tmux is."""
    try:
        app = await _front_app()
        processes = process_table()
        screen = await _screen(app, processes)
        panes = dict(await _panes(environment))
    except _Unread as unread:
        return FrontUnread(str(unread))
    except (OSError, TimeoutError) as error:
        # [LAW:no-silent-failure] the turn goes without the fact, and the turn's record says why it was left out.
        return FrontUnread(f"{type(error).__name__}: {error}")
    candidates = [Candidate(session, name, frozenset(_terminals(pid, processes))) for session, (pid, name) in sessions.items()]
    return in_front(screen, panes, candidates)


async def _front_app() -> _App:
    """The app LaunchServices says is in front: the one the user is using, whether or not a window of its is on screen."""
    serial = (await _ran(_LSAPPINFO, "front")).strip()
    if not serial.startswith("ASN:"):
        raise _Unread(f"no app is in front: lsappinfo said {serial!r}")
    info = await _ran(_LSAPPINFO, "info", "-only", "pid", "-only", "bundleid", "-only", "name", serial)
    fields = {match[1]: match[2] if match[3] is None else match[3] for match in _FIELD.finditer(info)}
    match fields:
        case {"pid": pid, "LSDisplayName": name}:
            return _App(int(pid), fields.get("CFBundleIdentifier"), name)
        case _:
            raise _Unread(f"lsappinfo did not say which app {serial} is: {info!r}")


async def _ran(*argv: str) -> str:
    ran = await run(*argv, timeout=2)
    if ran.returncode != 0:
        raise _Unread(f"{' '.join(argv)} exited {ran.returncode}: {ran.err.decode(errors='replace').strip()}")
    return ran.out.decode()


async def _screen(app: _App, processes: Mapping[int, Process]) -> Screen:
    match _FRONT_TAB.get(app.bundle or ""):
        case str() as source:
            return Screen(app.name, frozenset({_device(await _applescript.run(lambda: _asked(app.name, source)))}))
        case None:
            return Screen(app.name, frozenset(process.tty for process in _under(app.pid, processes) if process.tty is not None))


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


async def _panes(environment: Mapping[str, str]) -> list[tuple[int, int]]:
    """Each tmux client's terminal, and the terminal of the pane it shows, from every server of this user's."""
    # Where tmux puts a server's socket unless -S names one: $TMUX_TMPDIR, else /tmp, in tmux-<uid>.
    directory = Path(environment.get("TMUX_TMPDIR", "/tmp")) / f"tmux-{os.getuid()}"
    sockets = [path for path in directory.glob("*") if path.is_socket()]
    if not sockets:
        return []
    tmux = shutil.which("tmux", path=environment.get("PATH"))
    if tmux is None:
        raise _Unread(f"tmux sockets are in {directory}, and no tmux is on the PATH to ask them which pane each client shows")
    panes: list[tuple[int, int]] = []
    for socket in sockets:
        for line in await _clients(tmux, socket):
            client, pane = line.split("\t")
            panes.append((_device(client), _device(pane)))
    return panes


async def _clients(tmux: str, socket: Path) -> Sequence[str]:
    asked = await run(tmux, "-S", str(socket), "list-clients", "-F", "#{client_tty}\t#{pane_tty}", timeout=2)
    if asked.returncode != 0:
        if asked.err.startswith(_NO_SERVER):
            return ()
        raise _Unread(f"tmux at {socket} did not list its clients: {asked.err.decode(errors='replace').strip()}")
    return asked.out.decode().splitlines()


def _device(path: str) -> int:
    try:
        return os.stat(path).st_rdev
    except OSError as error:
        raise _Unread(f"the terminal {path} could not be read: {error}") from error
