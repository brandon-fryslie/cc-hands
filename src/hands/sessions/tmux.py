"""The user's tmux servers: where their sockets are, what each says when asked, the panes each holds, and what a pane shows."""

import asyncio
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hands.core.tmux import Listed, Pane, Server, Unanswered
from hands.sessions.child import Ran, run

# What tmux says, on stderr, for a socket no server listens on any more: a server that exited leaves its socket behind.
_NO_SERVER = (b"no server running on", b"error connecting to")
# How long a server has to answer: one that does not is wedged.
ANSWER_SECONDS = 2.0


@dataclass(frozen=True)
class Answered:
    """The lines a tmux server answered with; none from a socket no server listens on."""

    socket: Path
    lines: Sequence[str]


Answer = Answered | Unanswered


async def asked(environment: Mapping[str, str], *arguments: str) -> list[Answer]:
    """What every tmux server of the user's answers `arguments` with; `environment` says where tmux keeps its sockets
    and where tmux is."""
    directory = socket_directory(environment)
    sockets = [path for path in directory.glob("*") if path.is_socket()]
    tmux = shutil.which("tmux", path=environment.get("PATH"))
    match (sockets, tmux):
        case ([], _):
            return []
        case (_, None):
            return [Unanswered(f"tmux sockets are in {directory}, and no tmux is on the PATH to ask them")]
        case (_, str(tmux)):
            return list(await asyncio.gather(*(_answer(tmux, socket, arguments) for socket in sockets)))


def socket_directory(environment: Mapping[str, str]) -> Path:
    """Where tmux puts a server's socket unless -S names one: in tmux-<uid> under $TMUX_TMPDIR, or /tmp when that is
    unset or empty, as tmux reads it."""
    return Path(environment.get("TMUX_TMPDIR") or "/tmp") / f"tmux-{os.getuid()}"


async def _answer(tmux: str, socket: Path, arguments: Sequence[str]) -> Answer:
    match await ran_at(tmux, socket, arguments):
        case Ran(returncode=0, out=out):
            return Answered(socket, out.decode().splitlines())
        case Ran(err=err) if err.startswith(_NO_SERVER):
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


async def ran_at(tmux: str, socket: Path, arguments: Sequence[str]) -> Ran | Unanswered:
    """How the tmux server at `socket` exited from `arguments`, or why it gave no answer: wedged, or `tmux` not run."""
    try:
        return await run(tmux, "-S", str(socket), *arguments, timeout=ANSWER_SECONDS)
    except TimeoutError:
        return Unanswered(f"tmux at {socket} did not answer {arguments[0]} in {ANSWER_SECONDS:.0f} seconds")
    except OSError as error:
        return Unanswered(f"{tmux} could not be run to ask tmux at {socket}: {error}")


async def servers(environment: Mapping[str, str]) -> list[Server]:
    """Every tmux server of the user's, each with the live panes it holds or why it did not say."""
    # A dead pane, kept by remain-on-exit, runs nothing and keeps the name of a terminal that is gone or reused.
    # The session's name last: it is the one field that may hold a tab.
    answers = await asked(environment, "list-panes", "-a", "-f", "#{?pane_dead,0,1}", "-F", "#{pane_tty}\t#{pane_id}\t#{window_index}\t#{session_name}")
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
