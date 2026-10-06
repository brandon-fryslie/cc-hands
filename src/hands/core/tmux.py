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
class Listed:
    """A tmux server's panes, by the device number of each one's terminal."""

    panes: Mapping[int, Pane]


@dataclass(frozen=True)
class Unanswered:
    """A tmux server that did not say which panes it holds, and why."""

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
