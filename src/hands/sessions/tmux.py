"""The user's tmux servers: where their sockets are, what each says when asked, and the panes each holds."""

import asyncio
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from hands.core.tmux import Listed, Pane, Server, Unanswered
from hands.sessions.child import run

# What tmux says, on stderr, for a socket no server listens on any more: a server that exited leaves its socket behind.
_NO_SERVER = (b"no server running on", b"error connecting to")
# How long a server has to answer: one that does not is wedged.
ANSWER_SECONDS = 2.0


class NotAnswered(Exception):
    """A tmux server that answered with an error, or not in time."""


def sockets(environment: Mapping[str, str]) -> tuple[Path, list[Path]]:
    """Where tmux puts a server's socket unless -S names one, $TMUX_TMPDIR else /tmp in tmux-<uid>, and the sockets there."""
    directory = Path(environment.get("TMUX_TMPDIR", "/tmp")) / f"tmux-{os.getuid()}"
    return directory, [path for path in directory.glob("*") if path.is_socket()]


async def asked(tmux: str, socket: Path, *arguments: str) -> Sequence[str]:
    """The lines the server at `socket` answers `arguments` with; none from a socket no server listens on."""
    try:
        ran = await run(tmux, "-S", str(socket), *arguments, timeout=ANSWER_SECONDS)
    except TimeoutError as error:
        raise NotAnswered(f"tmux at {socket} did not answer {arguments[0]} in {ANSWER_SECONDS:.0f} seconds") from error
    if ran.returncode != 0:
        if ran.err.startswith(_NO_SERVER):
            return ()
        raise NotAnswered(f"tmux at {socket} did not answer {arguments[0]}: {ran.err.decode(errors='replace').strip()}")
    return ran.out.decode().splitlines()


async def servers(environment: Mapping[str, str]) -> list[Server]:
    """Every tmux server of the user's, each with the panes it holds or why it did not say; `environment` says where tmux
    keeps its sockets and where tmux is."""
    directory, found = sockets(environment)
    if not found:
        return []
    tmux = shutil.which("tmux", path=environment.get("PATH"))
    if tmux is None:
        return [Unanswered(f"tmux sockets are in {directory}, and no tmux is on the PATH to ask them which panes they hold")]
    return list(await asyncio.gather(*(_server(tmux, socket) for socket in found)))


async def _server(tmux: str, socket: Path) -> Server:
    try:
        # The session's name last: it is the one field that may hold a tab.
        lines = await asked(tmux, socket, "list-panes", "-a", "-F", "#{pane_tty}\t#{pane_id}\t#{window_index}\t#{session_name}")
        return Listed({os.stat(tty).st_rdev: Pane(socket, id, session, int(window)) for tty, id, window, session in (line.split("\t", 3) for line in lines)})
    except NotAnswered as error:
        return Unanswered(str(error))
    except OSError as error:
        # A pane closed between its listing and the look at its terminal.
        return Unanswered(f"a terminal of a pane of tmux at {socket} could not be read: {error}")
