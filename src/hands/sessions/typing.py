"""Typing into a session, through the fritter that wrapped it.

fritter runs a session's `claude` on a pseudo-terminal and listens on a unix socket
beside it; anything asked for there is typed into that session's input. This is the
client side of that socket, and the whole of hands' dependence on fritter.

It decides nothing about *what* to type. Whether a draft is ready, whether a session's
state allows a send, and what a leading slash means are all settled before anything
reaches here `[LAW:single-enforcer]`.
"""

import json
import socket
from dataclasses import dataclass
from pathlib import Path

from hands.core.session import Keystroke, Membership, PromptText, SessionId

# How long to wait for fritter to answer. It answers as soon as it has written to the
# pty, so this bounds a fritter that is wedged rather than one that is busy; long enough
# to be certain of that, short enough that the daemon is not held by it.
ANSWER_TIMEOUT = 5.0


class Untyped(Exception):
    """The text did not reach the session. The message names why."""


@dataclass(frozen=True)
class Typist:
    """A session that can be typed into, which is to say one fritter wrapped.

    [LAW:parse-dont-validate] Made only by `of`, so holding one is the proof that an
    address exists. Nothing below re-checks whether the session is wrapped, because a
    session that is not cannot be represented here.
    """

    session: SessionId
    socket: Path

    @classmethod
    def of(cls, membership: Membership) -> "Typist":
        if membership.fritter is None:
            # [LAW:no-silent-failure] Doing nothing here would leave a draft that was
            # never sent looking exactly like one that was.
            raise Untyped(
                f"session {membership.id} was not started under fritter, so there is nowhere to type into it"
            )
        return cls(session=membership.id, socket=membership.fritter)

    def type(self, text: PromptText, submit: bool) -> None:
        """Put text into the session's input, and press Enter after it when asked to.

        A newline inside the text stays a newline in the message: fritter brackets the
        paste when the session accepts bracketing, so only `submit` submits.
        """
        self._ask({"kind": "text", "text": str(text), "submit": submit})

    def press(self, key: Keystroke) -> None:
        """Send one named chord, which is not text and is never escaped as text."""
        self._ask({"kind": "key", "key": key})

    def _ask(self, request: dict[str, object]) -> None:
        body = json.dumps(request).encode() + b"\n"
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(ANSWER_TIMEOUT)
                connection.connect(str(self.socket))
                connection.sendall(body)
                answer = _read_line(connection)
        except OSError as error:
            # A session whose process is gone leaves a socket nobody is listening on, and
            # that is the common case here rather than an exotic one.
            raise Untyped(f"cannot reach the fritter for session {self.session} at {self.socket}: {error}") from error
        _raise_if_refused(self.session, answer)


def _read_line(connection: socket.socket) -> bytes:
    chunks: list[bytes] = []
    while b"\n" not in b"".join(chunks):
        chunk = connection.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _raise_if_refused(session: SessionId, answer: bytes) -> None:
    try:
        decoded = json.loads(answer)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise Untyped(f"fritter for session {session} answered something that is not JSON: {error}") from error
    match decoded:
        case {"ok": True}:
            return
        case {"ok": False, "reason": str() as reason}:
            raise Untyped(f"fritter refused to type into session {session}: {reason}")
        case other:
            raise Untyped(f"fritter for session {session} answered {other!r}, which says neither yes nor no")
