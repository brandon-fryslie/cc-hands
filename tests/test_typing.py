"""Typing into a session through fritter's socket: what is sent, and how refusals arrive."""

import json
import shutil
import socket
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from hands.core.session import Membership, PromptText, SessionId
from hands.sessions import typing
from hands.sessions.typing import Elsewhere, NotListening, Typist, Untyped

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
    assert json.loads(fritter.asked) == {"pid": 4242, "kind": "text", "text": "fix the auth middleware", "submit": True}


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
    assert json.loads(fritter.asked) == {"pid": 4242, "kind": "text", "text": "first\nsecond", "submit": False}


def test_a_named_key_is_sent_as_a_key_and_never_as_text(short_dir: Path) -> None:
    path = short_dir / "f.sock"
    fritter = FakeFritter(path, b'{"ok":true}\n')
    try:
        Typist.of(wrapped(path)).press("escape")
    finally:
        fritter.close()
    assert fritter.asked is not None
    assert json.loads(fritter.asked) == {"pid": 4242, "kind": "key", "key": "escape"}


def test_a_turn_starting_is_sent_as_its_own_kind(short_dir: Path) -> None:
    path = short_dir / "f.sock"
    fritter = FakeFritter(path, b'{"ok":true}\n')
    try:
        Typist.of(wrapped(path)).working()
    finally:
        fritter.close()
    assert fritter.asked is not None
    assert json.loads(fritter.asked) == {"pid": 4242, "kind": "working"}


def test_an_unanswered_turn_start_does_not_warn_about_text_nobody_sent(short_dir: Path) -> None:
    path = short_dir / "f.sock"
    fritter = FakeFritter(path, b"")
    try:
        with pytest.raises(Untyped) as unanswered:
            Typist.of(wrapped(path)).working()
    finally:
        fritter.close()
    assert "whether it heard is not known" in str(unanswered.value)
    assert "input box" not in str(unanswered.value)


def test_an_address_another_session_owns_is_told_apart(short_dir: Path) -> None:
    path = short_dir / "f.sock"
    fritter = FakeFritter(path, b'{"ok":false,"elsewhere":true,"reason":"this socket types into process 1"}\n')
    try:
        with pytest.raises(Elsewhere):
            Typist.of(wrapped(path)).type(PromptText("hello"), submit=True)
    finally:
        fritter.close()


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
    with pytest.raises(NotListening) as refused:
        Typist.of(wrapped(path)).type(PromptText("hello"), submit=True)
    assert str(path) in str(refused.value)


def test_an_answer_that_says_neither_yes_nor_no_is_not_taken_for_yes(short_dir: Path) -> None:
    # [LAW:no-silent-failure] Anything but a clear yes leaves the draft unsent, and says so.
    #
    # And says the same thing a silence says. All of these arrive after fritter took the
    # request, and fritter types before it answers, so none of them is evidence that
    # nothing happened. A caller told only "that was not JSON" reads it as "nothing was
    # sent" and retypes a draft that is already in the box.
    for answer in (b'{"maybe":1}\n', b"not json at all\n", b'{"ok":"sure"}\n'):
        path = short_dir / f"f{len(answer)}.sock"
        fritter = FakeFritter(path, answer)
        try:
            with pytest.raises(Untyped) as refused:
                Typist.of(wrapped(path)).type(PromptText("hello"), submit=True)
        finally:
            fritter.close()
        assert "may already be in the input box" in str(refused.value), answer
        assert "do not send it again" in str(refused.value), answer


def test_a_refusal_fritter_actually_gave_is_not_dressed_up_as_a_silence(short_dir: Path) -> None:
    # The one answer that does not get the warning: fritter said no and said why, and the
    # reason itself says whether anything reached the box. Adding "it may already be in
    # the input box" to a refusal that begins "the user has unsent text" would tell a
    # caller not to resend a message that was never sent at all.
    path = short_dir / "refused.sock"
    fritter = FakeFritter(path, b'{"ok":false,"reason":"the user has unsent text in this session\'s input"}\n')
    try:
        with pytest.raises(Untyped) as refused:
            Typist.of(wrapped(path)).type(PromptText("hello"), submit=True)
    finally:
        fritter.close()
    assert "the user has unsent text" in str(refused.value)
    assert "may already be in the input box" not in str(refused.value)


def test_a_request_too_big_for_fritter_is_refused_rather_than_sent(short_dir: Path) -> None:
    # fritter reads its limit and then answers and closes. A larger request has the close
    # land partway through sendall, so the caller sees a broken pipe and is told the
    # fritter cannot be reached - which invites retrying a request that can never work.
    path = short_dir / "big.sock"
    fritter = FakeFritter(path, b'{"ok":true}\n')
    try:
        with pytest.raises(Untyped) as refused:
            Typist.of(wrapped(path)).type(PromptText("x" * (typing.REQUEST_LIMIT + 1)), submit=True)
    finally:
        fritter.close()
    assert fritter.asked is None, "an oversize request was sent rather than refused"
    assert "takes at most" in str(refused.value)
    assert "nothing was typed" in str(refused.value)


class EndlessFritter:
    """A fritter that answers and answers and never finishes the line."""

    def __init__(self, path: Path, chunk: bytes, gap: float, stop_after: float) -> None:
        self._chunk, self._gap, self._stop_after = chunk, gap, stop_after
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
        # It gives up on its own so that a client which does not bound the exchange fails
        # the test slowly rather than hanging it.
        until = time.monotonic() + self._stop_after
        with connection:
            connection.recv(65536)
            while time.monotonic() < until:
                try:
                    connection.sendall(self._chunk)
                except OSError:
                    return
                time.sleep(self._gap)

    def close(self) -> None:
        self._listener.close()
        self._thread.join(timeout=8)


def test_a_fritter_that_answers_forever_does_not_hold_the_daemon_forever(
    short_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # settimeout bounds each recv and not the exchange, so a byte every 20ms resets the
    # clock indefinitely and a wedged session would own a daemon task for as long as it
    # kept dribbling. One deadline is computed before the first read instead.
    monkeypatch.setattr(typing, "ANSWER_TIMEOUT", 0.3)
    path = short_dir / "slow.sock"
    fritter = EndlessFritter(path, chunk=b"x", gap=0.02, stop_after=5.0)
    try:
        started = time.monotonic()
        with pytest.raises(Untyped):
            Typist.of(wrapped(path)).type(PromptText("hello"), submit=True)
        waited = time.monotonic() - started
    finally:
        fritter.close()
    assert waited < 2.0, f"the deadline is {typing.ANSWER_TIMEOUT}s and the exchange took {waited:.1f}s"


def test_an_answer_that_runs_past_its_size_is_cut_off(short_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # [LAW:no-silent-failure] An answer is one small JSON object. Reading anything larger
    # to its end is the daemon taking dictation from a session that has lost the plot.
    monkeypatch.setattr(typing, "ANSWER_LIMIT", 1024)
    path = short_dir / "loud.sock"
    fritter = EndlessFritter(path, chunk=b"x" * 4096, gap=0.0, stop_after=5.0)
    try:
        started = time.monotonic()
        with pytest.raises(Untyped):
            Typist.of(wrapped(path)).type(PromptText("hello"), submit=True)
        waited = time.monotonic() - started
    finally:
        fritter.close()
    # Cut off at the cap rather than read to the end of whatever the session felt like
    # sending, which is what the elapsed time is here to distinguish.
    assert waited < 2.0, f"the cap is {typing.ANSWER_LIMIT} bytes and the read took {waited:.1f}s"


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


def test_a_request_that_landed_but_went_unanswered_says_not_to_send_it_again(
    short_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # fritter writes to the pty before it answers, so a deadline reached after the request
    # went out does not mean the text did not land - it means nobody knows. Told "cannot
    # reach it", a caller retypes, and retyping a draft that did land sends it twice.
    monkeypatch.setattr(typing, "ANSWER_TIMEOUT", 0.3)
    path = short_dir / "mute.sock"
    fritter = MuteFritter(path, hold=3.0)
    try:
        with pytest.raises(Untyped) as refused:
            Typist.of(wrapped(path)).type(PromptText("fix the auth middleware"), submit=True)
    finally:
        fritter.close()
    assert fritter.asked is not None, "the request never reached the fake fritter, so this tests nothing"
    assert "may already be in the input box" in str(refused.value)
    assert "do not send it again" in str(refused.value)


def test_a_fritter_that_hangs_up_without_answering_says_that_and_not_that_it_answered_badly(short_dir: Path) -> None:
    # A closed connection and a garbled reply both arrive at json.loads as empty bytes.
    # Calling that "not JSON" sends someone looking for an answer that was never sent.
    path = short_dir / "hangup.sock"
    fritter = MuteFritter(path, hold=0.0)
    try:
        with pytest.raises(Untyped) as refused:
            Typist.of(wrapped(path)).type(PromptText("hello"), submit=True)
    finally:
        fritter.close()
    assert "closed the connection without answering" in str(refused.value)
    assert "not JSON" not in str(refused.value)
    # MuteFritter reads the request before it hangs up, so this is a delivered request
    # whose answer never came - the text may be in the box already.
    assert fritter.asked, "the request never reached the fake fritter, so this tests nothing"
    assert "may already be in the input box" in str(refused.value)
