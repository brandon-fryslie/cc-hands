"""What types into a session, or why a request to type into it did not reach it, whatever the request was."""

from dataclasses import dataclass

from hands.core.effects import Fritter, Writer
from hands.core.session import Membership, Running, Session, SessionId
from hands.core.status import Waiting
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


def through[F](wrapper: F | None, pane: Keyboard) -> F | Pane | Behind | NotInTmux | PaneUnread:
    """What keys typed for a process go through: the fritter that wrapped it whenever one did, else the tmux pane in
    front of it, else why there is none.

    [LAW:single-enforcer] the one rule for which way a process is typed into, for a session hands knows and for one it
    has no record of alike; `wrapper` is the fritter as the caller knows it.
    """
    match (wrapper, pane):
        case (None, Pane() | Behind() | NotInTmux() | PaneUnread()):
            return pane
        case (wrapped, _):
            return wrapped


def writer(membership: Membership, pane: Keyboard) -> Writer | Unwrapped:
    """What types into the session: the fritter that wrapped it whenever one did, else tmux into the pane in front of it.

    [LAW:single-enforcer] the one place a session's writer is chosen, for a draft, a command, and a key alike.
    """
    fritter = None if membership.fritter is None else Fritter(membership.fritter, membership.pid)
    match through(fritter, pane):
        case Fritter() | Pane() as typed:
            return typed
        case Behind() | NotInTmux() | PaneUnread() as missing:
            return Unwrapped(membership.id, missing)


def prompter(session: Session, pane: Keyboard) -> Writer | Unwrapped | AtItsDialog:
    """What types a prompt into the session, or why nothing may: a session waiting at a dialog would take it as the answer
    to what it asked, which is the user's to answer.

    [LAW:single-enforcer] the one place it is decided whether a prompt or a command may be typed into a session now.
    """
    match (session.state, writer(session.membership, pane)):
        case (_, Unwrapped() as unwrapped):
            return unwrapped
        case (Running(status=Waiting()), _):
            return AtItsDialog(session.membership.id)
        case (_, Fritter() | Pane() as by):
            return by
