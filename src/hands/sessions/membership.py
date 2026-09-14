"""The membership file: written by the shim at SessionStart, read by the daemon.

The file's name is the session id, so the id is not repeated inside it.
"""

import json
from pathlib import Path

from hands.core.session import Membership, SessionId, TmuxPane
from hands.sessions.home import Home
from hands.sessions.payload import Payload, Rejected


def write_membership(home: Home, membership: Membership) -> None:
    path = home.membership(membership.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
        {
            "pid": membership.pid,
            "pane": membership.pane,
            "cwd": str(membership.cwd),
            "transcript_path": str(membership.transcript),
        }
    )
    # [LAW:no-ambient-temporal-coupling] written beside and renamed into place,
    # so the daemon reading it never sees half a file.
    staging = path.with_suffix(".tmp")
    staging.write_text(body)
    staging.replace(path)


def remove_membership(home: Home, session: SessionId) -> None:
    # A session that started before the hooks were installed never had a file to remove.
    home.membership(session).unlink(missing_ok=True)


def read_membership(home: Home, session: SessionId) -> Membership:
    path = home.membership(session)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise Rejected(f"no membership file for session {session} at {path}") from None
    record = Payload.parse(raw)
    pane = record.optional_text("pane")
    return Membership(
        id=session,
        pid=record.integer("pid"),
        pane=None if pane is None else TmuxPane(pane),
        cwd=Path(record.text("cwd")),
        transcript=Path(record.text("transcript_path")),
    )
