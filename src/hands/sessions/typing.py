"""Typing into a session, through the fritter that wrapped it.

fritter runs a session's `claude` on a pseudo-terminal and listens on a unix socket
beside it; text asked for there is typed into that session's input and sent with Return,
as someone at the keyboard would. This is the client side of that socket. The other half
of hands' dependence on fritter is the name `FRITTER_SOCKET`, which
`hands.sessions.shim` reads out of a wrapped session's environment.

It decides nothing about *what* to type. Whether a draft is ready, whether a session can
be sent to, and whether a leading slash is escaped are all settled in the core before anything
reaches here `[LAW:single-enforcer]`.
"""

import json
import socket
from dataclasses import dataclass
from pathlib import Path

from hands.core.effects import Command, Input, Key, Text, Type
from hands.core.session import Keystroke, PromptText, SessionId

# How long the exchange may take. fritter answers as soon as its one write is done.
ANSWER_TIMEOUT = 5.0


class Untyped(Exception):
    """fritter could not be reached, or refused, or could not write. The message says which."""


@dataclass(frozen=True)
class Typist:
    """A session fritter wrapped: where its socket is, and the process fritter wrapped."""

    session: SessionId
    socket: Path
    # fritter types only into the process it wrapped, and its address is inherited, so a
    # session started from inside a wrapped one carries its parent's. Every request names this.
    pid: int

    def type(self, text: PromptText) -> None:
        """Type text into the session's input and press Return."""
        self._ask({"pid": self.pid, "kind": "text", "text": str(text)})

    def press(self, key: Keystroke) -> None:
        """Press one named chord."""
        self._ask({"pid": self.pid, "kind": "key", "key": key})

    def _ask(self, request: dict[str, object]) -> None:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(ANSWER_TIMEOUT)
                connection.connect(str(self.socket))
                connection.sendall(json.dumps(request).encode() + b"\n")
                answer = json.loads(connection.makefile("rb").readline() or b"null")
        except (OSError, ValueError) as error:
            raise Untyped(f"cannot talk to the fritter for session {self.session} at {self.socket}: {error}") from error
        match answer:
            case {"ok": True}:
                return
            case {"ok": False, "reason": str() as reason}:
                raise Untyped(f"fritter did not type into session {self.session}: {reason}")
            case other:
                raise Untyped(f"the fritter for session {self.session} answered {other!r}")


def type_into(effect: Type[Input]) -> None:
    """Perform a Type: text or a command typed into the session and sent with Return, or a key pressed."""
    typist = Typist(effect.session, effect.socket, effect.pid)
    match effect.input:
        case Text() | Command() as typed:
            typist.type(typed.typed)
        case Key(key=key):
            typist.press(key)
