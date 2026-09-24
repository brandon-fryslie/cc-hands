"""The membership file: written by the shim at SessionStart, read by the daemon.

The file's name is the session id, so the id is not repeated inside it.
"""

import json
from pathlib import Path

from hands.core.session import Membership, SessionId
from hands.sessions.home import Home
from hands.sessions.payload import Payload, Rejected
from hands.sessions.processes import parse_pid


def write_membership(home: Home, membership: Membership) -> None:
    path = home.membership(membership.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
        {
            "pid": membership.pid,
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


def remove_ended_membership(home: Home, ended: Membership) -> None:
    """Remove the file of a session the sweep found over, unless a new process has since started the session again."""
    try:
        current = read_membership(home, ended.id)
    except Rejected:
        # Already removed by the session's end, or unreadable, which the next sweep reports.
        return
    if current.pid == ended.pid:
        home.membership(ended.id).unlink(missing_ok=True)


def read_membership(home: Home, session: SessionId) -> Membership:
    path = home.membership(session)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise Rejected(f"no membership file for session {session} at {path}") from None
    return parse_membership(session, raw)


def parse_membership(session: SessionId, raw: bytes) -> Membership:
    record = Payload.parse(raw)
    return Membership(
        id=session,
        pid=parse_pid(record.integer("pid")),
        cwd=Path(record.text("cwd")),
        transcript=Path(record.text("transcript_path")),
    )
