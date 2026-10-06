"""Which tmux pane a session runs in: read over a process table and every tmux server's panes, never kept."""

import asyncio
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Generator
from pathlib import Path

import pytest

from hands.core.events import Joined
from hands.core.session import Membership, SessionId
from hands.core.tmux import Listed, NotInTmux, Pane, PaneUnread, Unanswered, pane_of
from hands.sessions import tmux
from hands.sessions.registry import Sessions
from hands.sessions.tmux import Answered
from hands.sessions.terminals import Process, ancestor_terminals, process_table
from hands.voice.tool import Result
from hands.voice.tools import read_screen_tool

TMUX = shutil.which("tmux")
needs_tmux = pytest.mark.skipif(TMUX is None, reason="no tmux to run panes in")

# Terminals by device number: a pane of each of two servers, the pseudoterminal fritter runs a session on, and a tab.
PANE, OTHER_PANE, FRITTER, TAB = 10, 11, 20, 1
WORK, PLAY = Path("/tmp/tmux-501/default"), Path("/tmp/tmux-501/play")
HANDS = Pane(WORK, "%3", "cc-hands", 2)
LAWS = Pane(PLAY, "%0", "laws", 0)
# A session under fritter in a pane of the first server: claude on fritter's pseudoterminal, fritter on the pane's.
# Another, in a pane of the second server; and one in a plain tab.
TABLE = {
    process.pid: process
    for process in (
        Process(100, 1, 501, PANE),
        Process(101, 100, 501, PANE),
        Process(102, 101, 501, FRITTER),
        Process(200, 1, 501, OTHER_PANE),
        Process(300, 1, 501, TAB),
    )
}
SERVERS = [Listed({PANE: HANDS}), Listed({OTHER_PANE: LAWS})]


def test_a_session_under_fritter_in_a_pane_is_in_that_pane_and_one_in_another_server_names_that_servers_socket() -> None:
    assert pane_of(ancestor_terminals(102, TABLE), SERVERS) == HANDS
    assert pane_of(ancestor_terminals(200, TABLE), SERVERS) == LAWS


def test_a_session_in_a_plain_tab_is_not_in_tmux_and_with_no_server_running_none_is() -> None:
    assert pane_of(ancestor_terminals(300, TABLE), SERVERS) == NotInTmux()
    assert pane_of(ancestor_terminals(102, TABLE), []) == NotInTmux()


def test_a_server_that_did_not_answer_leaves_unread_only_the_sessions_no_other_server_holds() -> None:
    servers = [Listed({PANE: HANDS}), Unanswered("tmux at /tmp/tmux-501/play did not answer list-panes in 2 seconds")]
    assert pane_of(ancestor_terminals(102, TABLE), servers) == HANDS
    assert pane_of(ancestor_terminals(300, TABLE), servers) == PaneUnread("tmux at /tmp/tmux-501/play did not answer list-panes in 2 seconds")


@pytest.fixture
def sockets(monkeypatch: pytest.MonkeyPatch) -> Generator[Path]:
    """A socket directory of the test's own, for servers that read no tmux.conf of the user's; its tmux-<uid>."""
    # A tmux socket's path has to fit a sockaddr_un, which a pytest tmp_path on macOS does not.
    directory = Path(tempfile.mkdtemp(prefix="hands-tmux-", dir="/tmp"))
    monkeypatch.setenv("TMUX_TMPDIR", str(directory))
    monkeypatch.delenv("TMUX", raising=False)
    try:
        yield directory / f"tmux-{os.getuid()}"
    finally:
        for socket in (directory / f"tmux-{os.getuid()}").glob("*"):
            subprocess.run([str(TMUX), "-S", str(socket), "kill-server"], capture_output=True, check=False)
        shutil.rmtree(directory)


def run_in(socket: str, session: str) -> tuple[str, int]:
    """A new tmux session of the server at `socket` running a process; its pane and the process's pid."""
    started = subprocess.run([str(TMUX), "-f", "/dev/null", "-L", socket, "new-session", "-d", "-P", "-F", "#{pane_id} #{pane_pid}", "-s", session, "sleep", "120"], capture_output=True, text=True, check=True)
    pane, pid = started.stdout.split()
    return pane, int(pid)


def in_pane(pid: int) -> object:
    return pane_of(ancestor_terminals(pid, process_table()), asyncio.run(tmux.servers(os.environ)))


@needs_tmux
def test_a_process_in_a_pane_of_each_of_two_servers_is_read_in_its_own_and_follows_it_across_break_pane(sockets: Path) -> None:
    work, worker = run_in("default", "work")
    play, player = run_in("play", "play")
    assert in_pane(worker) == Pane(sockets / "default", work, "work", 0)
    assert in_pane(player) == Pane(sockets / "play", play, "play", 0)
    assert in_pane(os.getpid()) == NotInTmux()

    # The pane split beside another, then broken out into a window of its own: the same pane, in its new window.
    subprocess.run([str(TMUX), "-L", "default", "split-window", "-d", "-t", work, "sleep", "120"], check=True)
    subprocess.run([str(TMUX), "-L", "default", "break-pane", "-d", "-s", work], check=True)
    assert in_pane(worker) == Pane(sockets / "default", work, "work", 1)


@needs_tmux
def test_a_socket_no_server_listens_on_holds_no_pane_and_sockets_with_no_tmux_to_ask_are_unread(sockets: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, worker = run_in("default", "work")
    # A server killed leaves its socket behind.
    server = int(subprocess.run([str(TMUX), "-L", "default", "display-message", "-p", "#{pid}"], capture_output=True, text=True, check=True).stdout)
    os.kill(server, 9)
    deadline = time.monotonic() + 5
    while subprocess.run([str(TMUX), "-L", "default", "has-session"], capture_output=True, check=False).returncode == 0:
        assert time.monotonic() < deadline, "the killed tmux server still answers"
        time.sleep(0.05)
    assert (sockets / "default").is_socket()
    assert in_pane(worker) == NotInTmux()

    run_in("default", "work")
    monkeypatch.setenv("PATH", "/nonexistent")
    match in_pane(worker):
        case PaneUnread(reason=reason):
            assert reason == f"tmux sockets are in {sockets}, and no tmux is on the PATH to ask them"
        case other:
            pytest.fail(f"read as {other}")


@needs_tmux
def test_a_dead_pane_kept_by_remain_on_exit_leaves_the_live_panes_of_its_server_read(sockets: Path) -> None:
    pane, worker = run_in("default", "work")
    subprocess.run([str(TMUX), "-L", "default", "set", "-g", "remain-on-exit", "on", ";", "new-window", "-t", "work", "true"], check=True)
    deadline = time.monotonic() + 5
    while subprocess.run([str(TMUX), "-L", "default", "list-panes", "-a", "-f", "#{pane_dead}", "-F", "#{pane_id}"], capture_output=True, text=True, check=True).stdout == "":
        assert time.monotonic() < deadline, "the window's command never exited"
        time.sleep(0.05)
    assert in_pane(worker) == Pane(sockets / "default", pane, "work", 0)


def test_sockets_are_looked_for_where_tmux_puts_them_with_an_empty_tmux_tmpdir_read_as_unset() -> None:
    assert tmux.socket_directory({"TMUX_TMPDIR": "/var/run/mine"}) == Path(f"/var/run/mine/tmux-{os.getuid()}")
    assert tmux.socket_directory({"TMUX_TMPDIR": ""}) == tmux.socket_directory({}) == Path(f"/tmp/tmux-{os.getuid()}")


def test_a_pane_closed_between_its_listing_and_the_look_at_its_terminal_leaves_the_rest_of_its_server_read() -> None:
    lines = ["/dev/null\t%1\t0\twork", "/dev/hands-no-such-terminal\t%2\t1\twork"]
    assert tmux.listed(Answered(WORK, lines)) == Listed({os.stat("/dev/null").st_rdev: Pane(WORK, "%1", "work", 0)})


@needs_tmux
def test_a_pane_is_shown_as_it_reads_now_and_one_that_cannot_be_read_says_why_never_as_a_blank_screen(sockets: Path) -> None:
    pane, _ = run_in("default", "work")
    socket = sockets / "default"
    subprocess.run([str(TMUX), "-L", "default", "send-keys", "-t", pane, "-l", "Allow this edit?"], check=True)
    deadline = time.monotonic() + 5
    while asyncio.run(tmux.shown(os.environ, socket, pane)) != "Allow this edit?":
        assert time.monotonic() < deadline, "what was typed never showed"
        time.sleep(0.05)

    match asyncio.run(tmux.shown(os.environ, socket, "%999")):
        case Unanswered(reason=reason):
            assert reason.startswith(f"tmux at {socket} did not show pane %999: ")
        case other:
            pytest.fail(f"read as {other!r}")
    subprocess.run([str(TMUX), "-L", "default", "kill-server"], check=True)
    match asyncio.run(tmux.shown(os.environ, socket, pane)):
        case Unanswered(reason=reason):
            assert reason.startswith(f"tmux at {socket} did not show pane {pane}: ")
        case other:
            pytest.fail(f"read as {other!r}")
    assert asyncio.run(tmux.shown({"PATH": "/nonexistent"}, socket, pane)) == Unanswered(f"no tmux is on the PATH to read pane {pane} of tmux at {socket}")


@needs_tmux
def test_a_tmux_that_cannot_be_run_leaves_its_servers_unread_and_says_why(sockets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, worker = run_in("default", "work")
    # An executable with no interpreter line and no machine code: exec refuses it.
    (tmp_path / "tmux").write_bytes(b"\x00\x01")
    (tmp_path / "tmux").chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    match in_pane(worker):
        case PaneUnread(reason=reason):
            assert reason.startswith(f"{tmp_path / 'tmux'} could not be run to ask tmux at {sockets / 'default'}: ")
        case other:
            pytest.fail(f"read as {other}")


def screen_of(sessions: Sessions, session: str) -> object:
    async def called() -> Result:
        return await read_screen_tool(sessions, os.environ).body(session=session)

    return asyncio.run(called())


def joined(*members: Membership) -> Sessions:
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    for member in members:
        asyncio.run(sessions.apply(Joined(member, "startup")))
    return sessions


@needs_tmux
def test_read_screen_says_why_for_a_session_in_no_pane_one_not_running_and_a_pane_tmux_did_not_show(sockets: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pane, worker = run_in("default", "work")
    # A session whose process is in no pane of the test's servers: this test's own.
    plain = Membership(SessionId("plain"), os.getpid(), Path("/code/plain"), Path("/nonexistent"))
    framed = Membership(SessionId("framed"), worker, Path("/code/framed"), Path("/nonexistent"))
    sessions = joined(plain, framed)
    assert screen_of(sessions, "plain") == {"error": "plain runs in no tmux pane, and hands reads a screen only off one"}
    assert screen_of(sessions, "gone") == {"error": "no session gone is running: list_sessions names the ones that are"}

    async def unshown(_environment: object, socket: Path, _pane: str) -> Unanswered:
        return Unanswered(f"tmux at {socket} did not answer capture-pane in 2 seconds")

    monkeypatch.setattr(tmux, "shown", unshown)
    assert screen_of(sessions, "framed") == {
        "error": f"the screen of framed could not be read: tmux at {sockets / 'default'} did not answer capture-pane in 2 seconds",
        "tmux": {"socket": str(sockets / "default"), "pane": pane, "session": "work", "window": 0},
    }
