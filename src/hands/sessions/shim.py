"""The hook command: record membership, post the payload to the daemon, return.

    <hands python> -m hands.sessions.shim <hands home>

It runs in Claude Code's critical path for every subscribed hook, so it imports
only the standard library and hands' data modules, and it never waits on
anything but the post.
"""

import http.client
import os
import socket
import sys
from collections.abc import Sequence
from pathlib import Path

from hands.core.session import Membership, TmuxPane
from hands.sessions.home import Home
from hands.sessions.membership import remove_membership, write_membership
from hands.sessions.payload import Payload, Rejected

POST_TIMEOUT_SECONDS = 2.0

# [LAW:no-silent-failure] Claude Code shows a hook's stderr for exit 1 and carries
# on. Exit 2 would instead block the prompt or the stop and hand the message to
# the model, turning a dead daemon into a stuck session.
FAILED = 1
USAGE = 64


class Undelivered(Exception):
    pass


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: Path) -> None:
        super().__init__("hands", timeout=POST_TIMEOUT_SECONDS)
        self._path = path

    def connect(self) -> None:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.timeout)
        connection.connect(str(self._path))
        self.sock = connection


def record(home: Home, payload: Payload) -> None:
    match payload.text("hook_event_name"):
        case "SessionStart":
            pane = os.environ.get("TMUX_PANE")
            membership = Membership(
                id=payload.session_id(),
                # Claude Code runs the hook command as its own child, with no shell between.
                pid=os.getppid(),
                pane=None if pane is None else TmuxPane(pane),
                cwd=Path(payload.text("cwd")),
                transcript=Path(payload.text("transcript_path")),
            )
            write_membership(home, membership)
        case "SessionEnd":
            remove_membership(home, payload.session_id())
        case _:
            pass


def post(home: Home, body: bytes) -> None:
    connection = _UnixConnection(home.socket)
    try:
        connection.request("POST", "/hook", body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        detail = response.read().decode(errors="replace")
    except OSError as error:
        raise Undelivered(f"cannot reach the hands daemon at {home.socket}: {error}") from error
    finally:
        connection.close()
    if response.status >= 300:
        raise Undelivered(f"the hands daemon refused this hook ({response.status}): {detail}")


def main(argv: Sequence[str]) -> int:
    match argv:
        case [_, root]:
            home = Home(Path(root))
        case _:
            print("usage: python -m hands.sessions.shim <hands home>", file=sys.stderr)
            return USAGE
    body = sys.stdin.buffer.read()
    try:
        record(home, Payload.parse(body))
        post(home, body)
    except (Rejected, Undelivered) as error:
        print(f"hands: {error}", file=sys.stderr)
        return FAILED
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
