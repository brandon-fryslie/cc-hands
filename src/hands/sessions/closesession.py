"""Ending a Claude Code session the user is finished with: the `close_session` tool.

The session's `claude` is sent SIGTERM, which Claude Code ends on as it ends on a closed terminal: it fires SessionEnd and
exits (2.1.289, within two seconds). What ran it goes with it as it would at a terminal: a tmux window hands opened ran
nothing but that `claude`, so tmux closes it, and a shell the user started it from takes their prompt again. The close
returns once the registry no longer lists the session, or says why it did not.
"""

import asyncio
import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from hands.core import status
from hands.core.events import Attached, Died, MovedOn
from hands.core.session import Idle, Session, SessionId, Told, Untold
from hands.sessions import audit, liveness, wide
from hands.sessions.home import Home

# How long a closed session has to leave the registry: Claude Code's exit, and the SessionEnd that says so, or the sweep
# that finds its process gone.
END_SECONDS = 10.0
# How often the registry is looked at while it ends.
LOOK_SECONDS = 0.1

# Why a session is closed. The user named it, so it ends whatever it is doing; or they asked for the sessions that are
# done, so it ends only where it is done (`done`), and is otherwise left running.
Asked = Literal["named", "done"]


class NotClosed(Exception):
    """The session was not ended, or has not left the registry. The message says why."""


@dataclass(frozen=True)
class Closed:
    pass


@dataclass(frozen=True)
class LeftRunning:
    """A session asked about as done that is not: it is working, at a dialog, or running a shell in the background."""

    # The session as it was found not done.
    session: Session


Outcome = Closed | LeftRunning


def done(session: Session) -> bool:
    """Whether a session is done: at its prompt, with no dialog up, no turn opened, and no shell running in the background."""
    # [LAW:one-source-of-truth] at its prompt is Claude Code's status, busy while a background subagent works (see
    # status.Busy); a background shell still running is work it was set to and has not finished, which ending the
    # session would kill. A turn opens on its prompt's hook, before the
    # status that says it is busy is read, so an opened turn runs whatever the status still says.
    match session:
        case Session(state=Idle(status=status.Idle()), dialog=None, turn=Untold() | Told()):
            return True
        case _:
            return False


async def close(home: Home, record: audit.Record, live: Callable[[SessionId], Session | None], session: SessionId, asked: Asked) -> Outcome:
    """End `session` as `asked`, and wait until `live` no longer has it; or leave it running, where it was asked about as
    done and is not."""
    # [LAW:nothing-unseen] a close is one unit of work: which session, why, and whether it was left running or signalled.
    with wide.unit("session.close", record):
        wide.annotate(session=session, asked=asked)
        found = live(session)
        if found is None:
            raise NotClosed(f"no session {session} is running")
        is_done = done(found)
        wide.annotate(done=is_done)
        if asked == "done" and not is_done:
            return LeftRunning(found)
        signalled = _signalled(home, found)
        wide.annotate(signalled=signalled)
        await _ended(live, session)
        return Closed()


def _signalled(home: Home, session: Session) -> bool:
    """Send the session's `claude` SIGTERM, where its process still holds it: whether it was sent."""
    pid = session.membership.pid
    match liveness.observed(home, session.membership):
        case Attached():
            pass
        case Died() | MovedOn():
            # Its process ended, or holds another session now, which the user did not ask to close: this session has
            # ended, and the sweep says so to the registry.
            return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        # It ended since it was judged running.
        return False
    except OSError as error:
        raise NotClosed(f"session {session.membership.id}'s claude, pid {pid}, could not be signalled to end: {error}") from error
    return True


async def _ended(live: Callable[[SessionId], Session | None], session: SessionId) -> None:
    deadline = time.monotonic() + END_SECONDS
    while live(session) is not None:
        if time.monotonic() > deadline:
            raise NotClosed(f"session {session} was told to end and is still running {END_SECONDS:g} seconds later")
        await asyncio.sleep(LOOK_SECONDS)
