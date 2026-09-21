"""Typing into a session through fritter's socket: what is sent, and how refusals arrive."""

import json
import shutil
import socket
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from hands.core.session import Membership, PromptText, SessionId
from hands.sessions.typing import Typist, Untyped

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


def wrapped(path: Path) -> Membership:
    return Membership(SID, pid=4242, cwd=Path("/code/a"), transcript=Path("/nowhere/t.jsonl"), fritter=path)


def test_a_session_nobody_wrapped_cannot_be_typed_into() -> None:
    plain = Membership(SID, pid=4242, cwd=Path("/code/a"), transcript=Path("/nowhere/t.jsonl"))
    with pytest.raises(Untyped) as refused:
        Typist.of(plain)
    assert SID in str(refused.value)
    assert "not started under fritter" in str(refused.value)


def test_text_and_whether_to_submit_reach_fritter_as_asked(short_dir: Path) -> None:
    path = short_dir / "f.sock"
    fritter = FakeFritter(path, b'{"ok":true}\n')
    try:
        Typist.of(wrapped(path)).type(PromptText("fix the auth middleware"), submit=True)
    finally:
        fritter.close()
    assert fritter.asked is not None
    assert json.loads(fritter.asked) == {"kind": "text", "text": "fix the auth middleware", "submit": True}


def test_a_newline_is_sent_as_text_because_bracketing_is_fritters_job(short_dir: Path) -> None:
    # PromptText allows a newline and forbids a carriage return, so a multi-line draft is
    # ordinary text here. Only `submit` submits it.
    path = short_dir / "f.sock"
    fritter = FakeFritter(path, b'{"ok":true}\n')
    try:
        Typist.of(wrapped(path)).type(PromptText("first\nsecond"), submit=False)
    finally:
        fritter.close()
    assert fritter.asked is not None
    assert json.loads(fritter.asked) == {"kind": "text", "text": "first\nsecond", "submit": False}


def test_a_named_key_is_sent_as_a_key_and_never_as_text(short_dir: Path) -> None:
    path = short_dir / "f.sock"
    fritter = FakeFritter(path, b'{"ok":true}\n')
    try:
        Typist.of(wrapped(path)).press("escape")
    finally:
        fritter.close()
    assert fritter.asked is not None
    assert json.loads(fritter.asked) == {"kind": "key", "key": "escape"}


def test_a_refusal_carries_fritters_reason(short_dir: Path) -> None:
    path = short_dir / "f.sock"
    fritter = FakeFritter(path, b'{"ok":false,"reason":"the user has unsent text in this session\'s input"}\n')
    try:
        with pytest.raises(Untyped) as refused:
            Typist.of(wrapped(path)).type(PromptText("hello"), submit=True)
    finally:
        fritter.close()
    assert "the user has unsent text" in str(refused.value)


def test_a_socket_nobody_is_listening_on_says_so(short_dir: Path) -> None:
    # The common case: the session's process ended and took its fritter with it.
    path = short_dir / "gone.sock"
    with pytest.raises(Untyped) as refused:
        Typist.of(wrapped(path)).type(PromptText("hello"), submit=True)
    assert str(path) in str(refused.value)


def test_an_answer_that_says_neither_yes_nor_no_is_not_taken_for_yes(short_dir: Path) -> None:
    # [LAW:no-silent-failure] Anything but a clear yes leaves the draft unsent, and says so.
    for answer in (b'{"maybe":1}\n', b"not json at all\n"):
        path = short_dir / f"f{len(answer)}.sock"
        fritter = FakeFritter(path, answer)
        try:
            with pytest.raises(Untyped):
                Typist.of(wrapped(path)).type(PromptText("hello"), submit=True)
        finally:
            fritter.close()
