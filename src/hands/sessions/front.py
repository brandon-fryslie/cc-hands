# pyright: basic, reportAttributeAccessIssue=false
"""Reads what is in front on the Mac's screen: LaunchServices for the app that is, the app itself for its front tab, tmux
for the pane each of its clients shows, and the kernel for the terminals each session runs under.

hands is tied to no one terminal. An app that can say which of its tabs is in front by AppleScript is asked; any other
shows every terminal running under it, which names the session in front whenever it holds only one.
"""

import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from Foundation import NSAppleScript
from loguru import logger

from hands.core.front import Candidate, FrontUnread, InFront, Screen, in_front
from hands.core.session import SessionId
from hands.core.tmux import Unanswered
from hands.sessions.child import run
from hands.sessions import tmux
from hands.sessions.terminals import Process, ancestor_terminals, process_table
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
        # A tmux client is on screen only in a terminal the app in front shows.
        panes = dict(await _panes(environment, [pid for pid, _ in sessions.values()], processes)) if screen.shown else {}
    except _Unread as unread:
        return FrontUnread(str(unread))
    except (OSError, TimeoutError) as error:
        # [LAW:no-silent-failure] the turn goes without the fact, and the turn's record says why it was left out.
        return FrontUnread(f"{type(error).__name__}: {error}")
    except Exception as error:
        # [LAW:no-silent-failure] a read broken by a fault of its own is logged with where, and recorded the same: the
        # screen is a fact the turn can go without, never one that ends the brain's turns.
        logger.opt(exception=error).error("reading what is in front broke")
        return FrontUnread(f"{type(error).__name__}: {error}")
    candidates = [Candidate(session, name, frozenset(ancestor_terminals(pid, processes))) for session, (pid, name) in sessions.items()]
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


async def _panes(environment: Mapping[str, str], pids: Sequence[int], processes: Mapping[int, Process]) -> list[tuple[int, int]]:
    """Each tmux client's terminal, and the terminal of the pane it shows, from each server that may hold a pane of one
    of `pids`."""
    answers = await tmux.asked(environment, pids, processes, "list-clients", "-F", "#{client_tty}\t#{pane_tty}")
    if unanswered := [answer.reason for answer in answers if isinstance(answer, Unanswered)]:
        raise _Unread("; ".join(unanswered))
    # A client with no terminal, as one in control mode over a pipe, is on no screen; nor is one gone since it was listed.
    pairs = [(tmux.device(client), tmux.device(pane)) for answer in answers if isinstance(answer, tmux.Answered) for client, pane in (line.split("\t") for line in answer.lines) if client]
    return [(client, pane) for client, pane in pairs if client is not None and pane is not None]


def _device(path: str) -> int:
    try:
        return os.stat(path).st_rdev
    except OSError as error:
        raise _Unread(f"the terminal {path} could not be read: {error}") from error
