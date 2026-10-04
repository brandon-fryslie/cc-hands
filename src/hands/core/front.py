"""What was in front on the Mac's screen as the user spoke: the app, and the session it showed, if any.

Read once as a turn is submitted, never watched between turns. "No session" is a fact the brain is told; a screen hands
could not read is a fact left out, and the turn goes without it.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass

from hands.core.session import SessionId


@dataclass(frozen=True)
class Screen:
    """The app in front, and the terminals its front window shows, by device number: the front tab's, or every terminal
    under the app where it cannot be asked which tab is in front."""

    app: str
    shown: frozenset[int]


@dataclass(frozen=True)
class Candidate:
    """A running session, as it is spoken, and every terminal on its line of ancestors: its own, its wrapper's, and the
    one the tab or the tmux pane holding it shows."""

    session: SessionId
    name: str
    terminals: frozenset[int]


@dataclass(frozen=True)
class SessionInFront:
    app: str
    session: SessionId
    name: str


@dataclass(frozen=True)
class NoSessionInFront:
    app: str


@dataclass(frozen=True)
class FrontUnread:
    """Why what was in front could not be read."""

    reason: str


InFront = SessionInFront | NoSessionInFront | FrontUnread


def in_front(screen: Screen, panes: Mapping[int, int], candidates: Collection[Candidate]) -> InFront:
    """The session the screen shows. `panes` is each tmux client's terminal to the terminal of the pane it shows: a tab
    running tmux shows that pane, not the client."""
    visible = {panes.get(terminal, terminal) for terminal in screen.shown}
    match [candidate for candidate in candidates if candidate.terminals & visible]:
        case []:
            return NoSessionInFront(screen.app)
        case [candidate]:
            return SessionInFront(screen.app, candidate.session, candidate.name)
        case several:
            # Two sessions on one terminal, one started from the other or suspended under it, or an app holding several
            # that cannot say which of its tabs is in front.
            return FrontUnread(f"{len(several)} sessions run under the terminals {screen.app} shows")


def told(front: InFront) -> str:
    """The note the brain reads beside the user's words; nothing for a screen that could not be read."""
    match front:
        case SessionInFront(app=app, session=session, name=name):
            shown = f'the session "{name}" (id {session})'
        case NoSessionInFront(app=app):
            shown = "no session"
        case FrontUnread():
            return ""
    return f"[hands] As the user said this, {app} was in front on the Mac's screen, showing {shown}. Say nothing about this unless it bears on what they said."
