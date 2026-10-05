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

    def command(self, command: Command) -> None:
        """Type a slash command into the session's input and press Return: the command as keys, and its arguments
        pasted behind it, so a long paste Claude Code folds into a placeholder cannot fold the command in with it."""
        self._ask({"pid": self.pid, "kind": "command", "command": command.word, "text": "" if command.args is None else str(command.args)})

    def press(self, key: Keystroke) -> None:
        """Press one named chord."""
        self._ask({"pid": self.pid, "kind": "key", "key": key})

    def pasting(self) -> None:
        """Returns if the session has turned bracketed paste on, which Claude Code does once its input is up, and raises
        Untyped saying why if it has not. Types nothing."""
        self._ask({"pid": self.pid, "kind": "pasting"})

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
            # A session keeps the fritter it started under for its whole life, so one started before hands was updated
            # refuses a kind added since in exactly fritter's words (control.go, `no request kind named %q`).
            case {"ok": False, "reason": str() as reason} if reason == f"no request kind named {json.dumps(request['kind'])}":
                raise Untyped(
                    f"session {self.session} runs a fritter older than this hands, one with no {request['kind']} request: "
                    "restart that session so it starts under the fritter hands installed"
                )
            case {"ok": False, "reason": str() as reason}:
                raise Untyped(f"fritter did not type into session {self.session}: {reason}")
            case other:
                raise Untyped(f"the fritter for session {self.session} answered {other!r}")


def type_into(effect: Type[Input]) -> None:
    """Perform a Type: text or a command typed into the session and sent with Return, or a key pressed."""
    typist = Typist(effect.session, effect.socket, effect.pid)
    match effect.input:
        case Text() as typed:
            typist.type(typed.typed)
        case Command() as command:
            typist.command(command)
        case Key(key=key):
            typist.press(key)
