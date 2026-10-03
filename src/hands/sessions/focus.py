"""The focus: the session the user is talking to when they name none, or none. Kept in the home, so it outlives the
daemon, and read at each use, never copied [LAW:one-source-of-truth], so a change holds from the next thing said.

It is a default target and never a lock: a call that names a session reaches that session, focused or not.
"""

import re

from hands.core.session import SessionId
from hands.sessions.files import replace_whole
from hands.sessions.home import Home
from hands.sessions.payload import Rejected

# What a session id is as Claude Code writes one: a uuid. Anything else in the file was not written by the daemon.
_ID = re.compile(rb"[0-9A-Za-z-]+")


def focused(home: Home) -> SessionId | None:
    """The focused session; None where the user focused none."""
    try:
        # Bytes, not text: a file edited by hand can hold anything, and whatever is not one session id is refused alike.
        written = home.focus.read_bytes().strip()
    except FileNotFoundError:
        return None
    if _ID.fullmatch(written) is None:
        raise Rejected(f"{home.focus} says {written.decode(errors='replace')!r}, which is no session id; focus a session again, or remove the file to focus none")
    return SessionId(written.decode())


def set_focus(home: Home, to: SessionId | None) -> None:
    match to:
        case None:
            home.focus.unlink(missing_ok=True)
        case session:
            # [LAW:no-ambient-temporal-coupling] replaced whole, so a reader never sees half an id.
            replace_whole(home.focus, f"{session}\n", 0o644)
