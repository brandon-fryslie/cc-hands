import fcntl
import os
import subprocess
import sys
import termios
import time
from collections.abc import Iterator

import pytest

from hands.core.front import Candidate, FrontUnread, NoSessionInFront, Screen, SessionInFront, in_front, told
from hands.core.session import SessionId
from hands.sessions.front import _terminals, _under  # pyright: ignore[reportPrivateUsage]
from hands.sessions.terminals import process_table

# Terminals by device number: two tabs, a tmux client in a third, and the panes behind it.
TAB, OTHER_TAB, CLIENT, PANE, OTHER_PANE, FRITTER = 1, 2, 3, 10, 11, 20
DOCS = Candidate(SessionId("s1"), "hands, docs", frozenset({FRITTER, TAB}))
TESTS = Candidate(SessionId("s2"), "hands, tests", frozenset({21, PANE}))
LAWS = Candidate(SessionId("s3"), "laws", frozenset({22, OTHER_PANE}))


def test_the_session_whose_terminal_the_front_tab_shows_is_in_front() -> None:
    assert in_front(Screen("iTerm2", frozenset({TAB})), {}, [DOCS, TESTS]) == SessionInFront("iTerm2", SessionId("s1"), "hands, docs")


def test_a_tab_running_tmux_shows_the_pane_its_client_is_on_not_the_others_of_the_server() -> None:
    panes = {CLIENT: PANE, 4: OTHER_PANE}
    assert in_front(Screen("iTerm2", frozenset({CLIENT})), panes, [DOCS, TESTS, LAWS]) == SessionInFront("iTerm2", SessionId("s2"), "hands, tests")


def test_an_app_showing_no_session_terminal_is_no_session_in_front_and_says_which_app() -> None:
    assert in_front(Screen("iTerm2", frozenset({OTHER_TAB})), {}, [DOCS]) == NoSessionInFront("iTerm2")
    assert in_front(Screen("Safari", frozenset()), {CLIENT: PANE}, [DOCS, TESTS]) == NoSessionInFront("Safari")


def test_an_app_that_cannot_say_its_front_tab_and_holds_several_sessions_is_left_unread() -> None:
    match in_front(Screen("Ghostty", frozenset({TAB, CLIENT})), {CLIENT: PANE}, [DOCS, TESTS, LAWS]):
        case FrontUnread(reason=reason):
            assert reason == "2 sessions run under the terminals Ghostty shows"
        case other:
            pytest.fail(f"read as {other}")


def test_the_brain_is_told_the_session_in_front_or_that_none_is_and_nothing_of_a_screen_left_unread() -> None:
    assert told(SessionInFront("iTerm2", SessionId("s1"), "hands, docs")) == (
        '[hands] As the user said this, iTerm2 was in front on the Mac\'s screen, showing the session "hands, docs" (id s1). '
        "Say nothing about this unless it bears on what they said."
    )
    assert told(NoSessionInFront("Safari")) == (
        "[hands] As the user said this, Safari was in front on the Mac's screen, showing no session. Say nothing about this unless it bears on what they said."
    )
    assert told(FrontUnread("no window is on screen")) == ""


@pytest.fixture
def at_a_terminal() -> Iterator[tuple[subprocess.Popen[bytes], int]]:
    """A process whose controlling terminal is a pty of the test's own, with a child of its own under it."""
    controller, terminal = os.openpty()

    def control() -> None:
        os.setsid()
        fcntl.ioctl(terminal, termios.TIOCSCTTY, 0)

    child = "import subprocess, sys; subprocess.run([sys.executable, '-c', 'import time; time.sleep(30)'])"
    process = subprocess.Popen([sys.executable, "-c", child], stdin=terminal, stdout=terminal, stderr=terminal, preexec_fn=control)
    try:
        yield process, os.fstat(terminal).st_rdev
    finally:
        process.kill()
        process.wait()
        os.close(controller)
        os.close(terminal)


def test_the_kernel_says_each_process_terminal_and_the_terminals_a_session_runs_under(at_a_terminal: tuple[subprocess.Popen[bytes], int]) -> None:
    process, device = at_a_terminal
    # Until the child has started: polled, since nothing of the child's says when it has.
    deadline = time.monotonic() + 10
    processes = process_table()
    while not any(child.parent == process.pid for child in processes.values()):
        assert time.monotonic() < deadline, "the child never started"
        time.sleep(0.01)
        processes = process_table()
    [grandchild] = [child for child in processes.values() if child.parent == process.pid]
    assert processes[process.pid].tty == device
    # The child inherits the terminal; the test itself, started by pytest, sits above it with whatever it has.
    assert next(_terminals(grandchild.pid, processes)) == device
    assert device not in set(_terminals(os.getpid(), processes))
    assert [found.pid for found in _under(process.pid, processes)] == [grandchild.pid]


def test_the_process_table_holds_roots_processes_too_so_every_line_of_parents_ends_at_launchd() -> None:
    processes = process_table()
    # A terminal app starts each tab's shell through /usr/bin/login, which runs as root: a table of this user's
    # processes alone breaks the line between a session and its app there.
    assert processes[1].uid == 0
    for process in processes.values():
        line = process
        while line.pid != 1:
            line = processes[line.parent]
    # Its own parent, the kernel, is not in the table, so a walk up ends there.
    assert processes[1].parent not in processes
