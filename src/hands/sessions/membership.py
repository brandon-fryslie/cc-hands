"""The membership file: written by the shim at SessionStart, or at the first hook of a session that never fired one; read by the daemon.

The file's name is the session id, so the id is not repeated inside it.
"""

import json
import os
from pathlib import Path

from hands.core.session import Membership, SessionId
from hands.sessions.home import Home
from hands.sessions.payload import Payload, Rejected
from hands.sessions.processes import parse_pid, process_starts, still_running


def write_membership(home: Home, membership: Membership) -> None:
    path = home.membership(membership.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
        {
            "pid": membership.pid,
            "cwd": str(membership.cwd),
            "transcript_path": str(membership.transcript),
            # Written as null rather than left out, so a file from a session nobody
            # wrapped and a file written before this field existed read the same way.
            "fritter_socket": None if membership.fritter is None else str(membership.fritter),
        }
    )
    # [LAW:no-ambient-temporal-coupling] written beside and renamed into place,
    # so the daemon reading it never sees half a file. Named for the writing shim:
    # a session's first hooks can run at once, parallel tool calls each firing one.
    staging = path.with_suffix(f".{os.getpid()}.tmp")
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


def held(home: Home, pid: int) -> bool:
    """Whether any membership file names this process: one naming its pid, written while it was running."""
    starts = process_starts({pid})
    for path in home.memberships.glob("*.json"):
        try:
            # A file written before the process under its pid started names a dead process whose pid it took.
            if Payload.parse(path.read_bytes()).integer("pid") == pid and still_running(pid, path.stat().st_mtime, starts):
                return True
        # Ended since the listing, or unreadable, which the daemon's sweep reports and removes: either way it names
        # no process.
        except (FileNotFoundError, Rejected):
            continue
    return False


def recorded_membership(home: Home, session: SessionId) -> Membership | None:
    """The session's membership, or None when it has no file: it ended, or its process holds another session now."""
    try:
        raw = home.membership(session).read_bytes()
    except FileNotFoundError:
        return None
    return parse_membership(session, raw)


def read_membership(home: Home, session: SessionId) -> Membership:
    match recorded_membership(home, session):
        case None:
            raise Rejected(f"no membership file for session {session} at {home.membership(session)}")
        case membership:
            return membership


def parse_membership(session: SessionId, raw: bytes) -> Membership:
    record = Payload.parse(raw)
    # [LAW:parse-dont-validate] Missing and empty both mean this session was not wrapped.
    # Path("") is PosixPath("."), and a Typist made from that would dial a directory.
    address = record.optional_text("fritter_socket")
    return Membership(
        id=session,
        pid=parse_pid(record.integer("pid")),
        cwd=Path(record.text("cwd")),
        transcript=Path(record.text("transcript_path")),
        fritter=Path(address) if address else None,
    )
