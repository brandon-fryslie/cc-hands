"""The user's tmux servers: where the sockets of those that may hold a session's pane are, what each says when asked,
the panes each holds, which pane a process runs in, what a pane shows, and typing into one."""

import asyncio
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from loguru import logger

from hands.core.effects import Command, Input, Key, Text
from hands.core.session import Keystroke
from hands.core.tmux import InPane, Keyboard, Listed, Pane, PaneUnread, Server, Unanswered, keyboard_of, pane_of
from hands.sessions.child import Ran, run
from hands.sessions.terminals import Process, Undescribed, ancestor_terminals, ancestors, directory_of, environment_of, front_terminal, process_table

# What tmux says, on stderr, for a socket no server listens on any more, as a server that exited leaves behind, and
# for one that is gone; never "error connecting to" for any other reason, as a path too long for a socket.
_NO_SERVER = re.compile(rb"no server running on |error connecting to .* \(No such file or directory\)$")
# $TMUX as tmux sets it in a pane: its server's socket, the server's pid, and the session's index.
_TMUX = re.compile(r"(?P<socket>.+),(?P<server>\d+),\d+")
# How long a server has to answer: one that does not is wedged.
ANSWER_SECONDS = 2.0


@dataclass(frozen=True)
class Answered:
    """The lines a tmux server answered with; none from a socket no server listens on."""

    socket: Path
    lines: Sequence[str]


Answer = Answered | Unanswered


async def asked(environment: Mapping[str, str], pids: Sequence[int], processes: Mapping[int, Process], *arguments: str) -> list[Answer]:
    """What each tmux server that may hold a pane of one of `pids`, as `sockets` finds them, answers `arguments` with,
    and why any could not be found; `environment` says where tmux keeps its sockets and where tmux is."""
    at = await asyncio.to_thread(sockets, environment, pids, processes)
    found = [socket for socket in at if isinstance(socket, Path)]
    unfound: list[Answer] = [socket for socket in at if isinstance(socket, Unanswered)]
    tmux = shutil.which("tmux", path=environment.get("PATH"))
    match (found, tmux):
        case ([], _):
            return unfound
        case (_, None):
            return [*unfound, Unanswered(f"tmux sockets are at {', '.join(map(str, found))}, and no tmux is on the PATH to ask them")]
        case (_, str(tmux)):
            return [*unfound, *await asyncio.gather(*(_answer(tmux, socket, arguments) for socket in found))]


def sockets(environment: Mapping[str, str], pids: Sequence[int], processes: Mapping[int, Process]) -> list[Path | Unanswered]:
    """The socket of every tmux server that may hold a pane of one of `pids`: each in the directory `environment` names,
    and each named by $TMUX on a line of ancestors of `pids`, which is the only word of a server started with `-S`
    elsewhere, or under another $TMUX_TMPDIR; or why one process on such a line could not be read.

    [LAW:one-source-of-truth] tmux sets $TMUX to its socket, then its pid and the session's index; it resolves the
    directory's path (/tmp is /private/tmp), so one server found both ways is known by its real path and asked once, by
    its path in the directory.
    """
    own = os.geteuid()
    # A pane's $TMUX is on the processes at its terminal; the tmux server, and the app and launchd above it, are at none.
    lines = {process.pid: process for pid in pids for process in ancestors(pid, processes) if process.uid == own and process.tty is not None}
    found = [*(path for path in socket_directory(environment).glob("*") if path.is_socket()), *(socket for process in lines.values() for socket in _named(process, processes))]
    # Reversed, so the first path found for a server is the one kept.
    return list({os.path.realpath(socket) if isinstance(socket, Path) else socket: socket for socket in reversed(found)}.values())


def _named(process: Process, processes: Mapping[int, Process]) -> list[Path | Unanswered]:
    """The socket $TMUX names in the environment `process` was started with: none outside tmux, or once it has exited."""
    match environment_of(process):
        case {"TMUX": str(named)} if tmux := _TMUX.fullmatch(named):
            return _socket(Path(tmux["socket"]), processes.get(int(tmux["server"])))
        case Undescribed() as refused:
            # [LAW:no-silent-failure] a server its environment would name may hold a pane: never read as no pane.
            return [_unread(f"which tmux server process {process.pid} runs under", refused)]
        case _:
            return []


def _socket(path: Path, server: Process | None) -> list[Path | Unanswered]:
    """Where the socket $TMUX names as `path` is: tmux keeps a relative -S path as it was given, which is relative to the
    directory `server` was started in and never leaves."""
    if path.is_absolute():
        return [path]
    match None if server is None else directory_of(server):
        case Path() as directory:
            return [directory / path]
        case Undescribed() as refused:
            return [_unread(f"where the tmux server at {path} runs", refused)]
        case None:
            # A server that has exited holds no pane.
            return []


def _unread(what: str, refused: Undescribed) -> Unanswered:
    return Unanswered(f"{what} could not be read: {refused.call} refused with errno {refused.errno}")


def socket_directory(environment: Mapping[str, str]) -> Path:
    """Where tmux puts a server's socket unless -S names one: in tmux-<uid> under $TMUX_TMPDIR, or /tmp when that is
    unset or empty, as tmux reads it."""
    return Path(environment.get("TMUX_TMPDIR") or "/tmp") / f"tmux-{os.getuid()}"


async def _answer(tmux: str, socket: Path, arguments: Sequence[str]) -> Answer:
    match await ran_at(tmux, socket, arguments):
        case Ran(returncode=0, out=out):
            return Answered(socket, out.decode().splitlines())
        case Ran(err=err) if _NO_SERVER.match(err):
            return Answered(socket, ())
        case Ran(err=err):
            return Unanswered(f"tmux at {socket} did not answer {arguments[0]}: {err.decode(errors='replace').strip()}")
        case Unanswered() as unanswered:
            return unanswered


async def shown(environment: Mapping[str, str], socket: Path, pane: str) -> str | Unanswered:
    """The text tmux pane `pane` of the server at `socket` shows now, down to its last line drawn on, the blank rows below
    left off, or why it could not be read; `environment` says where tmux is."""
    tmux = shutil.which("tmux", path=environment.get("PATH"))
    if tmux is None:
        return Unanswered(f"no tmux is on the PATH to read pane {pane} of tmux at {socket}")
    # [LAW:no-silent-failure] a server gone since the pane was named is said, never read as a blank screen.
    match await ran_at(tmux, socket, ("capture-pane", "-p", "-t", pane)):
        case Ran(returncode=0, out=out):
            return out.decode(errors="replace").rstrip()
        case Ran(err=err):
            return Unanswered(f"tmux at {socket} did not show pane {pane}: {err.decode(errors='replace').strip()}")
        case Unanswered() as unanswered:
            return unanswered


async def typed(environment: Mapping[str, str], pane: Pane, input: Input) -> Unanswered | None:
    """Type `input` into `pane` as someone at its keyboard would, or say why it was not; `environment` says where tmux is.

    [LAW:no-ambient-temporal-coupling] one tmux command list, so what is typed and the Return that sends it happen
    together or not at all: no half-typed prompt is left in the input for a retry to double. And an Escape is answered
    only once it has been read alone, so whatever is typed next cannot be read with it as one chord.
    """
    tmux = shutil.which("tmux", path=environment.get("PATH"))
    if tmux is None:
        return Unanswered(f"no tmux is on the PATH to type into pane {pane.id} of tmux at {pane.socket}")
    # A buffer of its own per paste, so two sends cannot paste each other's text; loaded from a file, since a command
    # tmux is handed is far shorter than a prompt can be.
    buffer = f"hands-{uuid4().hex}"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix="hands-paste-", delete_on_close=False) as pasted:
        pasted.write(_pasted(input))
        pasted.close()
        match await ran_at(tmux, pane.socket, _typing(pane.id, input, buffer, pasted.name)):
            case Ran(returncode=0):
                await asyncio.sleep(_quiet(input))
                return None
            case Ran(err=err):
                return Unanswered(f"tmux at {pane.socket} did not type into pane {pane.id}: {err.decode(errors='replace').strip()}")
            case Unanswered() as unanswered:
                return unanswered


def _pasted(input: Input) -> str:
    """What of `input` is pasted rather than pressed: a prompt, a command's arguments behind their space, nothing of a key."""
    match input:
        case Text() as text:
            return text.typed
        case Command(args=None) | Key():
            return ""
        case Command(args=args):
            return f" {args}"


def _quiet(input: Input) -> float:
    """How long the keys of `input` keep the session's input to themselves: an Escape until it is read alone."""
    match input:
        case Key(key="escape"):
            return LONE_ESCAPE
        case Text() | Command() | Key():
            return 0.0


def _typing(pane: str, input: Input, buffer: str, pasted: str) -> list[str]:
    """The tmux command list that types `input` into `pane`, the file `pasted` holding what of it is pasted.

    Something is typed into the pane before anything is loaded - a command's name, or nothing ahead of a prompt - so a
    pane gone since it was read stops the list before the prompt sits in a buffer that only a paste would delete. A
    paste is bracketed (-p), so its newlines stay in the prompt rather than send it; pasted as they are rather than as
    carriage returns (-r); and its buffer is deleted once pasted (-d). A command's name is pressed as keys and its
    arguments pasted behind it, so a long paste Claude Code folds into a placeholder cannot fold the command in with it.
    """
    paste = ["load-buffer", "-b", buffer, pasted, ";", "paste-buffer", "-p", "-r", "-d", "-b", buffer, "-t", pane, ";"]
    enter = ["send-keys", "-t", pane, "Enter"]
    match input:
        case Text():
            return ["send-keys", "-t", pane, "-l", "", ";", *paste, *enter]
        case Command(args=None) as command:
            return ["send-keys", "-t", pane, "-l", command.word, ";", *enter]
        case Command() as command:
            return ["send-keys", "-t", pane, "-l", command.word, ";", *paste, *enter]
        case Key(key=key):
            return ["send-keys", "-t", pane, KEYS[key]]


# How long an Escape keeps the session's input to itself: fritter's loneEscape, measured on Claude Code 2.1.285
# (fritter/control.go).
LONE_ESCAPE = 0.1

# [LAW:types-are-the-program] each chord hands speaks, as tmux names its key: every Keystroke has one.
KEYS: Mapping[Keystroke, str] = {
    "escape": "Escape",
    "enter": "Enter",
    "ctrl_c": "C-c",
    "ctrl_u": "C-u",
    "up": "Up",
    "down": "Down",
    "tab": "Tab",
    "shift_tab": "BTab",
}


async def ran_at(tmux: str, socket: Path, arguments: Sequence[str]) -> Ran | Unanswered:
    """How the tmux server at `socket` exited from `arguments`, or why it gave no answer: wedged, or `tmux` not run."""
    try:
        return await run(tmux, "-S", str(socket), *arguments, timeout=ANSWER_SECONDS)
    except TimeoutError:
        return Unanswered(f"tmux at {socket} did not answer {arguments[0]} in {ANSWER_SECONDS:.0f} seconds")
    except OSError as error:
        return Unanswered(f"{tmux} could not be run to ask tmux at {socket}: {error}")


async def panes(pids: Sequence[int], environment: Mapping[str, str]) -> list[InPane]:
    """The tmux pane each of `pids` runs in, from one read of the processes and of each tmux server that may hold it."""
    return await _read(pids, environment, lambda pid, processes, read: pane_of(ancestor_terminals(pid, processes), read))


async def keyboards(pids: Sequence[int], environment: Mapping[str, str]) -> list[Keyboard]:
    """The tmux pane whose keys reach each of `pids`, from one read of the processes and of each tmux server that may hold
    it."""
    return await _read(pids, environment, lambda pid, processes, read: keyboard_of(front_terminal(pid, processes), ancestor_terminals(pid, processes), read))


async def _read[P](pids: Sequence[int], environment: Mapping[str, str], of: Callable[[int, Mapping[int, Process], list[Server]], P]) -> list[P | PaneUnread]:
    try:
        processes = await asyncio.to_thread(process_table)
        read = await servers(environment, pids, processes)
    except Exception as error:
        # [LAW:no-silent-failure] the pane is a fact its readers can go without: a read that broke is logged with
        # where, and each process says why its pane is missing.
        logger.opt(exception=error).error("reading which tmux pane each session runs in broke")
        return [PaneUnread(f"{type(error).__name__}: {error}")] * len(pids)
    return [of(pid, processes, read) for pid in pids]


async def servers(environment: Mapping[str, str], pids: Sequence[int], processes: Mapping[int, Process]) -> list[Server]:
    """Each tmux server that may hold a pane of one of `pids`, with the live panes it holds or why it did not say."""
    # A dead pane, kept by remain-on-exit, runs nothing and keeps the name of a terminal that is gone or reused.
    # The session's name last: it is the one field that may hold a tab.
    answers = await asked(environment, pids, processes, "list-panes", "-a", "-f", "#{?pane_dead,0,1}", "-F", "#{pane_tty}\t#{pane_id}\t#{window_index}\t#{session_name}")
    return [listed(answer) if isinstance(answer, Answered) else answer for answer in answers]


def listed(answered: Answered) -> Server:
    """The panes a server's list-panes answer names, by the device number of each one's terminal."""
    panes = [(tty, Pane(answered.socket, id, session, int(window))) for tty, id, window, session in (line.split("\t", 3) for line in answered.lines)]
    try:
        terminals = [(device(tty), pane) for tty, pane in panes]
    except OSError as error:
        return Unanswered(f"a terminal of a pane of tmux at {answered.socket} could not be read: {error}")
    return Listed({device: pane for device, pane in terminals if device is not None})


def device(tty: str) -> int | None:
    """The device number of a terminal tmux named, of a pane or a client; none for one closed since it was named, which
    is gone and shows or runs nothing."""
    try:
        return os.stat(tty).st_rdev
    except FileNotFoundError:
        return None
