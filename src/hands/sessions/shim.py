"""The hook command: record membership, post the payload to the daemon, return.

    HANDS_HOME=<hands home> <python 3.12+> -m hands.sessions.shim     # HANDS_HOME defaults to ~/.hands

It runs in Claude Code's critical path for every subscribed hook, so it imports
only the standard library and hands' data modules, and it never waits on
anything but the post. Every post returns at once but a PermissionRequest's,
which waits for the answer and prints it for Claude Code to read.

The plugin installs the hooks whether or not hands is running, so a daemon the
shim cannot reach is judged by its heartbeat: one that was stopped or never ran
costs the session nothing, and one that died, hung, or left a heartbeat nothing
can read is reported.
"""

import http.client
import os
import socket
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from hands.core.session import Membership
from hands.sessions import heartbeat
from hands.sessions.home import Home, default_home
from hands.sessions.hookconfig import post_timeout
from hands.sessions.membership import remove_membership, write_membership
from hands.sessions.payload import Payload, Rejected

# [LAW:no-silent-failure] Claude Code shows a hook's stderr for exit 1 and carries
# on. Exit 2 would instead block the prompt or the stop and hand the message to
# the model, turning a dead daemon into a stuck session.
FAILED = 1
USAGE = 64


class Unreached(Exception):
    """Nothing answered on the socket: the daemon is off, or something is wrong with it, as its heartbeat says."""


class Refused(Exception):
    """The daemon answered, and said no."""


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: Path, timeout: float) -> None:
        super().__init__("hands", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.timeout)
        connection.connect(str(self._path))
        self.sock = connection


def record(home: Home, payload: Payload) -> None:
    match payload.text("hook_event_name"):
        case "SessionStart":
            membership = Membership(
                id=payload.session_id(),
                # The hook's shell execs a single simple command, so this is the claude
                # process. A compound hook command (`a; b`) would make it that shell.
                pid=os.getppid(),
                cwd=Path(payload.text("cwd")),
                transcript=Path(payload.text("transcript_path")),
            )
            write_membership(home, membership)
        case "SessionEnd":
            remove_membership(home, payload.session_id())
        case _:
            pass


def post(home: Home, body: bytes, timeout: float) -> str:
    """The daemon's reply: empty for every hook but a decided permission request."""
    connection = _UnixConnection(home.socket, timeout)
    try:
        connection.request("POST", "/hook", body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        detail = response.read().decode(errors="replace")
    except OSError as error:
        raise Unreached(f"cannot reach the hands daemon at {home.socket}: {error}") from error
    finally:
        connection.close()
    if response.status >= 300:
        raise Refused(f"the hands daemon refused this hook ({response.status}): {detail}")
    return detail


def unreached(home: Home, error: Unreached) -> int:
    """The exit for a hook no daemon answered: silent when hands is off, loud when it is broken."""
    now = datetime.now(UTC)
    verdict = heartbeat.look(home.status, now)
    match verdict:
        # Off is a state the user chose, not a failure. Nothing is printed, so a permission request falls through
        # to Claude Code's own dialog.
        case heartbeat.NeverRan() | heartbeat.Stopped():
            return 0
        # [LAW:no-silent-failure] a daemon that died, hung, or cannot be judged, and one that says it is up but
        # does not answer, are each reported with what the heartbeat says of it.
        case heartbeat.Down() | heartbeat.Unresponsive() | heartbeat.Unreadable() | heartbeat.Up():
            print(f"hands: {error}; {heartbeat.describe(verdict, now)}", file=sys.stderr)
            return FAILED


def main(argv: Sequence[str]) -> int:
    match argv:
        case [_]:
            home = default_home()
        case _:
            print("usage: python -m hands.sessions.shim  (the home is HANDS_HOME, or ~/.hands)", file=sys.stderr)
            return USAGE
    body = sys.stdin.buffer.read()
    try:
        payload = Payload.parse(body)
        record(home, payload)
        reply = post(home, body, post_timeout(payload.text("hook_event_name")))
    except Unreached as error:
        return unreached(home, error)
    except (Rejected, Refused) as error:
        print(f"hands: {error}", file=sys.stderr)
        return FAILED
    # Claude Code reads a permission decision from the hook's stdout.
    sys.stdout.write(reply)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
