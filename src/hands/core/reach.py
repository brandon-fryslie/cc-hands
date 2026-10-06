"""What types into a session, or why a request to type into it did not reach it, whatever the request was."""

from dataclasses import dataclass
from pathlib import Path

from hands.core.effects import Fritter, Writer
from hands.core.session import Membership, SessionId
from hands.core.tmux import Behind, Keyboard, NotInTmux, Pane, PaneUnread


@dataclass(frozen=True)
class UnknownSession:
    session: SessionId


@dataclass(frozen=True)
class SessionEnded:
    session: SessionId


@dataclass(frozen=True)
class Unwrapped:
    """The session was not started under fritter, and has no tmux pane in front of it, so there is nothing to type into:
    `pane` says whether it runs behind another program in one, is in none, or why which one could not be read."""

    session: SessionId
    pane: Behind | NotInTmux | PaneUnread


@dataclass(frozen=True)
class AtItsDialog:
    """The session is waiting at a dialog, which would take what is typed as its answer."""

    session: SessionId


Unreached = UnknownSession | SessionEnded | Unwrapped | AtItsDialog


def writer(membership: Membership, pane: Keyboard) -> Writer | Unwrapped:
    """What types into the session: the fritter that wrapped it whenever one did, else tmux into the pane in front of it.

    [LAW:single-enforcer] the one place a session's writer is chosen, for a draft, a command, and a key alike.
    """
    match (membership.fritter, pane):
        case (Path() as socket, _):
            return Fritter(socket, membership.pid)
        case (None, Pane()):
            return pane
        case (None, Behind() | NotInTmux() | PaneUnread()):
            return Unwrapped(membership.id, pane)
