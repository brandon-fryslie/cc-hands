"""Refocus: once hands has told the user of a session's turn or its question, that session is the focus, so what the
user says next, naming no session, is taken first as said to it: an answer to its question, or its next prompt.

It moves the one focus the user moves by voice [LAW:one-source-of-truth], so it holds as theirs does, across restarts and
until something moves it again, and the brain reads it where it reads theirs. A default target, never a lock: a session
the user names still gets what they say to it.
"""

import asyncio

from loguru import logger

from hands.core.session import SessionId
from hands.sessions.audit import Record, Refocused
from hands.sessions.focus import set_focus
from hands.sessions.home import Home
from hands.sessions.registry import Sessions


class NotRunning(Exception):
    """The session to focus has ended, or never joined."""


async def move_focus(sessions: Sessions, home: Home, to: SessionId | None) -> None:
    """Focus `to`, or no session: raises NotRunning for a session the registry does not hold, OSError for a focus that
    cannot be written."""
    # [LAW:single-enforcer] the one place hands moves the focus, so only a running session is ever moved to, whoever moves it.
    if to is not None and sessions.live_session(to) is None:
        raise NotRunning(f"no running session has the id {to!r}; take one from list_sessions")
    await asyncio.to_thread(set_focus, home, to)


class Refocus:
    """Moves the focus to a session just told of, as the model takes the telling, and records each move."""

    def __init__(self, sessions: Sessions, home: Home, record: Record) -> None:
        self._sessions = sessions
        self._home = home
        self._record = record

    async def __call__(self, session: SessionId) -> None:
        try:
            await move_focus(self._sessions, self._home, session)
        except NotRunning:
            # A session whose last turn is told as it ends: the focus stays where it was, on a session still there to take it.
            self._record(Refocused(session, "ended", None))
        except OSError as error:
            # [LAW:no-silent-failure] logged, and the Refocused line is an error: the telling stands, and the focus is as it was.
            logger.error(f"cannot move the focus to session {session}, which was just told: {error}")
            self._record(Refocused(session, "failed", str(error)))
        else:
            self._record(Refocused(session, "moved", None))
