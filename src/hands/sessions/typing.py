"""Typing into a session, through the fritter that wrapped it.

fritter runs a session's `claude` on a pseudo-terminal and listens on a unix socket
beside it; anything asked for there is typed into that session's input. This is the
client side of that socket. The other half of hands' dependence on fritter is the name
`FRITTER_SOCKET`, which `hands.sessions.shim` reads out of a wrapped session's
environment; between them they are all of it.

It decides nothing about *what* to type. Whether a draft is ready, whether a session's
state allows a send, and what a leading slash means are all settled before anything
reaches here `[LAW:single-enforcer]`.
"""

import json
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from hands.core.session import Keystroke, Membership, PromptText, SessionId

# How long the whole exchange may take, and the most an answer may run to.
#
# This must stay longer than fritter's own bounds added up, and they are: one second to
# read the request, half a second waiting for any other write to finish, one for each
# write into the session (at most two), and one more for the reply, which fritter grants
# after the typing is over so that answering is always affordable. Four and a half against
# this five. Shorten this below fritter's sum and hands hears its
# own timer instead of what fritter had to say. Reaching it at all means
# fritter itself is stuck, which is why a deadline reached after the request went out says
# something different from one reached before: see _ask.
ANSWER_TIMEOUT = 5.0
ANSWER_LIMIT = 64 * 1024

# The most a request may run to. fritter reads this much and no more, and then answers and
# closes - so a larger request is refused here rather than sent, because the close arrives
# while sendall is still writing and the caller sees a broken pipe instead of the reason.
REQUEST_LIMIT = 64 * 1024


class Untyped(Exception):
    """The text was not typed, or it is not known whether it was. The message says which.

    Not "the text did not reach the session": that is true of most of these and false of
    some. fritter writes to the pty before it answers, so a failure after the request went
    out leaves a message that may be sitting in the input box, and a write that had to be
    given up on leaves one that is partly there. The message is where that distinction
    lives, because the only reader of it is a model reading English.
    """


@dataclass(frozen=True)
class Typist:
    """A session that can be typed into, which is to say one fritter wrapped.

    [LAW:parse-dont-validate] Made only by `of`, so holding one is the proof that an
    address exists. Nothing below re-checks whether the session is wrapped, because a
    session that is not cannot be represented here.
    """

    session: SessionId
    socket: Path
    # The process the session is. fritter types only into the process it wrapped, and the
    # address alone does not say which that is: it is inherited, so a session started from
    # inside a wrapped one carries its parent's. Every request names this, and fritter
    # refuses one that names a process it did not wrap.
    pid: int

    @classmethod
    def of(cls, membership: Membership) -> "Typist":
        if membership.fritter is None:
            # [LAW:no-silent-failure] Doing nothing here would leave a draft that was
            # never sent looking exactly like one that was.
            raise Untyped(
                f"session {membership.id} was not started under fritter, so there is nowhere to type into it"
            )
        return cls(session=membership.id, socket=membership.fritter, pid=membership.pid)

    def type(self, text: PromptText, submit: bool) -> None:
        """Put text into the session's input, and press Enter after it when asked to.

        A newline inside the text stays a newline in the message: fritter brackets the
        paste when the session accepts bracketing, so only `submit` submits.
        """
        self._ask({"pid": self.pid, "kind": "text", "text": str(text), "submit": submit})

    def press(self, key: Keystroke) -> None:
        """Send one named chord, which is not text and is never escaped as text."""
        self._ask({"pid": self.pid, "kind": "key", "key": key})

    def _ask(self, request: dict[str, object]) -> None:
        body = json.dumps(request).encode() + b"\n"
        # [LAW:parse-dont-validate] Refused here rather than sent: fritter stops reading at
        # its own limit and closes, which arrives as a broken pipe partway through sendall
        # and is reported as a fritter that cannot be reached - inviting a retry of a
        # request that can never succeed.
        if len(body) > REQUEST_LIMIT:
            raise Untyped(
                f"this request is {len(body)} bytes and fritter takes at most {REQUEST_LIMIT}; nothing was typed"
            )
        # One deadline covers connecting, sending and reading. Per-operation timeouts
        # bound each call and not the exchange, so a fritter dribbling a byte every four
        # seconds would hold the daemon forever without once timing out.
        deadline = time.monotonic() + ANSWER_TIMEOUT
        # [LAW:no-silent-failure] fritter writes to the pty before it answers, so once the
        # request has gone out a failure here no longer means the text did not land - it
        # means nobody knows. A caller told "cannot reach it" retypes, and retyping a
        # draft that did land sends it twice.
        delivered = False
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(_left(deadline))
                connection.connect(str(self.socket))
                connection.settimeout(_left(deadline))
                connection.sendall(body)
                delivered = True
                answer = _read_line(connection, deadline)
        except OSError as error:
            if delivered:
                raise self._nobody_knows(f"it never answered ({error})") from error
            # A session whose process is gone leaves a socket nobody is listening on, and
            # that is the common case here rather than an exotic one.
            raise Untyped(f"cannot reach the fritter for session {self.session} at {self.socket}: {error}") from error
        _raise_if_refused(self.session, answer, self._nobody_knows)

    def _nobody_knows(self, what: str) -> Untyped:
        """The failure to report when the request went out and no answer came back.

        [LAW:one-source-of-truth] Every failure past the point the request was delivered is
        worded here, because they all mean the same thing and one of them saying less than
        the others is how a message gets typed twice. fritter types into the pty before it
        answers, so "no answer" never means "nothing happened" - not when the connection
        closed silently, not when what came back was not JSON, and not when it was JSON
        that says neither yes nor no.
        """
        return Untyped(
            f"the request reached the fritter for session {self.session} at {self.socket} but {what},"
            " so the text may already be in the input box - do not send it again"
        )


def _left(deadline: float) -> float:
    """What is left of the exchange's deadline, or a timeout if it is spent."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"no answer within {ANSWER_TIMEOUT} seconds")
    return remaining


def _read_line(connection: socket.socket, deadline: float) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        connection.settimeout(_left(deadline))
        chunk = connection.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
        # [LAW:no-silent-failure] An answer is one small JSON object. Anything still
        # arriving past this is not one, and reading it to the end would be the daemon
        # taking dictation from a wedged session.
        if size > ANSWER_LIMIT:
            raise ConnectionError(f"the answer ran past {ANSWER_LIMIT} bytes without ending")
        if b"\n" in chunk:
            break
    return b"".join(chunks)


def _raise_if_refused(session: SessionId, answer: bytes, nobody_knows: Callable[[str], Untyped]) -> None:
    """Read the answer. An answer that says nothing is not the same as one that says no.

    [LAW:parse-dont-validate] Three outcomes, and only the middle one is fritter speaking:
    it said yes, it said no and why, or nothing usable came back. The last is every other
    shape an answer can take, and all of them happen after fritter has already typed.
    """
    if not answer:
        # A closed connection and a malformed reply both reach json.loads, and the second
        # message would send someone looking for a garbled answer that was never sent.
        raise nobody_knows("it closed the connection without answering")
    try:
        decoded = json.loads(answer)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise nobody_knows(f"it answered something that is not JSON ({error})") from error
    match decoded:
        case {"ok": True}:
            return
        case {"ok": False, "reason": str() as reason}:
            # The one answer fritter actually gave. Its reason says whether anything
            # reached the box, so this is the only failure that does not need the warning.
            raise Untyped(f"fritter refused to type into session {session}: {reason}")
        case other:
            raise nobody_knows(f"it answered {other!r}, which says neither yes nor no")
