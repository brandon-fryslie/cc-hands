"""Which tmux pane a session runs in: the pane, among every pane of every tmux server of the user's, whose terminal is on
the session's line of ancestor terminals.

Read when asked, never kept: a pane moves between windows and sessions (`break-pane`, `join-pane`), and a pane kept would
go stale without a word.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Pane:
    """A tmux pane: the socket of the server holding it, its id (`%12`), and the tmux session and window it is in now."""

    socket: Path
    id: str
    session: str
    window: int


@dataclass(frozen=True)
class NotInTmux:
    """No pane of any tmux server of the user's runs the session; with no server running, none does."""


@dataclass(frozen=True)
class PaneUnread:
    """Why which pane runs the session could not be read."""

    reason: str


InPane = Pane | NotInTmux | PaneUnread


@dataclass(frozen=True)
class Behind:
    """The pane a session runs in, where some other program has the keyboard: the session is stopped, or runs inside a
    program with a terminal of its own (an editor's, screen's, ssh's), or was started from inside another session. Keys
    typed there reach that program, not the session."""

    pane: Pane


# Where keys typed for a session go: the pane in front of it, or why there is none.
Keyboard = Pane | Behind | NotInTmux | PaneUnread


@dataclass(frozen=True)
class Listed:
    """A tmux server's panes, by the device number of each one's terminal."""

    panes: Mapping[int, Pane]


@dataclass(frozen=True)
class Unanswered:
    """A tmux server that did not answer what it was asked, and why."""

    reason: str


Server = Listed | Unanswered


def pane_of(terminals: Iterable[int], servers: Iterable[Server]) -> InPane:
    """The pane running the session whose line of ancestor terminals is `terminals`, its own first, among `servers`."""
    servers = list(servers)
    panes = {terminal: pane for server in servers if isinstance(server, Listed) for terminal, pane in server.panes.items()}
    match (next((panes[terminal] for terminal in terminals if terminal in panes), None), [server.reason for server in servers if isinstance(server, Unanswered)]):
        case (Pane() as pane, _):
            return pane
        case (None, []):
            return NotInTmux()
        case (None, unanswered):
            # The pane may be one a server that did not answer holds.
            return PaneUnread("; ".join(unanswered))


def keyboard_of(front: Iterable[int], terminals: Iterable[int], servers: Iterable[Server]) -> Keyboard:
    """The pane whose keys reach the session: the pane that is `front`, the session's own terminal while the session is
    in front of it, if any is; else the pane on its line of ancestor `terminals`, its own first, that it runs behind.

    [LAW:single-enforcer] fritter types only into the process it wrapped; this is the same rule for a pane, which
    otherwise types into whatever program has its keyboard at the time.
    """
    servers = list(servers)
    match (pane_of(front, servers), pane_of(terminals, servers)):
        case (Pane() as pane, _):
            return pane
        case (_, Pane() as pane):
            return Behind(pane)
        case (_, NotInTmux() | PaneUnread() as missing):
            return missing
