"""The user's tmux servers: where their sockets are, what each says when asked, and the panes each holds."""

import asyncio
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hands.core.tmux import Listed, Pane, Server, Unanswered
from hands.sessions.child import run

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
    # Where tmux puts a server's socket unless -S names one: $TMUX_TMPDIR, else /tmp, in tmux-<uid>.
    directory = Path(environment.get("TMUX_TMPDIR", "/tmp")) / f"tmux-{os.getuid()}"
    sockets = [path for path in directory.glob("*") if path.is_socket()]
    tmux = shutil.which("tmux", path=environment.get("PATH"))
    match (sockets, tmux):
        case ([], _):
            return []
        case (_, None):
            return [Unanswered(f"tmux sockets are in {directory}, and no tmux is on the PATH to ask them")]
        case (_, str(tmux)):
            return list(await asyncio.gather(*(_answer(tmux, socket, arguments) for socket in sockets)))


async def _answer(tmux: str, socket: Path, arguments: Sequence[str]) -> Answer:
    try:
        ran = await run(tmux, "-S", str(socket), *arguments, timeout=ANSWER_SECONDS)
    except TimeoutError:
        return Unanswered(f"tmux at {socket} did not answer {arguments[0]} in {ANSWER_SECONDS:.0f} seconds")
    if ran.returncode != 0:
        if ran.err.startswith(_NO_SERVER):
            return Answered(socket, ())
        return Unanswered(f"tmux at {socket} did not answer {arguments[0]}: {ran.err.decode(errors='replace').strip()}")
    return Answered(socket, ran.out.decode().splitlines())


async def servers(environment: Mapping[str, str]) -> list[Server]:
    """Every tmux server of the user's, each with the live panes it holds or why it did not say."""
    # A dead pane, kept by remain-on-exit, runs nothing and keeps the name of a terminal that is gone or reused.
    # The session's name last: it is the one field that may hold a tab.
    answers = await asked(environment, "list-panes", "-a", "-f", "#{?pane_dead,0,1}", "-F", "#{pane_tty}\t#{pane_id}\t#{window_index}\t#{session_name}")
    return [_listed(answer) if isinstance(answer, Answered) else answer for answer in answers]


def _listed(answered: Answered) -> Server:
    panes = [(tty, Pane(answered.socket, id, session, int(window))) for tty, id, window, session in (line.split("\t", 3) for line in answered.lines)]
    try:
        return Listed({os.stat(tty).st_rdev: pane for tty, pane in panes})
    except OSError as error:
        # A pane closed between its listing and the look at its terminal.
        return Unanswered(f"a terminal of a pane of tmux at {answered.socket} could not be read: {error}")
