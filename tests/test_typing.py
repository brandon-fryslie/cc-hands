"""Typing into a session through fritter's socket: what is sent, and how refusals arrive."""

import json
import re
import shutil
import socket
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from hands.core.effects import Text, Type
from hands.core.session import PromptText, SessionId
from hands.sessions import typing
from hands.sessions.typing import Typist, Untyped, type_into

SID = SessionId("0f1e2d3c-aaaa-bbbb-cccc-000000000001")


@pytest.fixture
def short_dir() -> Iterator[Path]:
    # A unix socket path is capped near 104 bytes on macOS, so not under pytest's tmp_path.
    root = Path(tempfile.mkdtemp(prefix="fritter-"))
    yield root
    shutil.rmtree(root, ignore_errors=True)


class FakeFritter:
    """A socket that answers one request the way fritter would, and remembers what it was asked."""

    def __init__(self, path: Path, answer: bytes) -> None:
        self.asked: bytes | None = None
        self._answer = answer
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(str(path))
        self._listener.listen(1)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            connection, _ = self._listener.accept()
        except OSError:
            return
        with connection:
            self.asked = connection.recv(65536)
            connection.sendall(self._answer)

    def close(self) -> None:
        self._listener.close()
        self._thread.join(timeout=2)


def typist(path: Path) -> Typist:
    return Typist(SID, path, pid=4242)


def asked(path: Path, answer: bytes, send: Callable[[Typist], None]) -> dict[str, object]:
    """What a fake fritter answering `answer` was asked when `send` ran against it."""
    fritter = FakeFritter(path, answer)
    try:
        send(typist(path))
    finally:
        fritter.close()
    assert fritter.asked is not None
    return json.loads(fritter.asked)


def test_text_reaches_fritter_as_text_for_the_process_it_wrapped(short_dir: Path) -> None:
    # A newline is text: fritter pastes it, so only the Return fritter adds sends the message.
    sent = asked(short_dir / "f.sock", b'{"ok":true}\n', lambda t: t.type(PromptText("first\nsecond")))
    assert sent == {"pid": 4242, "kind": "text", "text": "first\nsecond"}


def test_a_type_effect_is_typed_behind_a_space(short_dir: Path) -> None:
    path = short_dir / "f.sock"
    fritter = FakeFritter(path, b'{"ok":true}\n')
    try:
        type_into(Type(SID, path, pid=4242, input=Text(PromptText("/compact now"))))
    finally:
        fritter.close()
    assert fritter.asked is not None
    assert json.loads(fritter.asked)["text"] == " /compact now"


def test_a_named_key_is_sent_as_a_key_and_never_as_text(short_dir: Path) -> None:
    assert asked(short_dir / "f.sock", b'{"ok":true}\n', lambda t: t.press("escape")) == {"pid": 4242, "kind": "key", "key": "escape"}


@pytest.mark.parametrize(
    ("answer", "says"),
    [
        (b'{"ok":false,"reason":"no key named f13"}\n', "no key named f13"),
        (b'{"ok":"sure"}\n', "answered {'ok': 'sure'}"),
        (b"not json\n", "cannot talk to the fritter"),
        (b"", "answered None"),
    ],
)
def test_anything_but_a_yes_is_raised_with_what_fritter_said(short_dir: Path, answer: bytes, says: str) -> None:
    with pytest.raises(Untyped, match=re.escape(says)):
        asked(short_dir / "f.sock", answer, lambda t: t.type(PromptText("hello")))


def test_a_socket_nobody_is_listening_on_says_so(short_dir: Path) -> None:
    # The common case: the session's process ended and took its fritter with it.
    path = short_dir / "gone.sock"
    with pytest.raises(Untyped, match=re.escape(str(path))):
        typist(path).type(PromptText("hello"))


class MuteFritter:
    """A fritter that takes the request and then says nothing."""

    def __init__(self, path: Path, hold: float) -> None:
        self._hold = hold
        self.asked: bytes | None = None
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(str(path))
        self._listener.listen(1)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            connection, _ = self._listener.accept()
        except OSError:
            return
        with connection:
            self.asked = connection.recv(65536)
            time.sleep(self._hold)

    def close(self) -> None:
        self._listener.close()
        self._thread.join(timeout=8)


def test_a_fritter_that_never_answers_does_not_hold_the_daemon(short_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(typing, "ANSWER_TIMEOUT", 0.3)
    path = short_dir / "mute.sock"
    fritter = MuteFritter(path, hold=3.0)
    try:
        with pytest.raises(Untyped, match="timed out"):
            typist(path).type(PromptText("hello"))
    finally:
        fritter.close()
