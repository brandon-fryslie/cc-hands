"""Which tmux pane a session runs in: read over a process table and every tmux server's panes, never kept."""

import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Generator, Sequence
from functools import partial
from pathlib import Path

import pytest

from hands.core.drafts import SendDraft, StageDraft
from hands.core.effects import Command, Input, Key, Text, Type, Typed
from hands.core.events import Joined
from hands.core.session import CommandName, Membership, PromptText, SessionId, Staged
from hands.core.tmux import Behind, Listed, NotInTmux, Pane, PaneUnread, Server, Unanswered, keyboard_of, pane_of
from hands.sessions import tmux
from hands.sessions.registry import Sessions
from hands.sessions.tmux import Answered
from hands.sessions.terminals import Process, ancestor_terminals, front_terminal, process_table
from hands.sessions.typing import Untyped, type_into
from hands.sessions.wide import unit
from hands.voice.tool import Result
from hands.voice.tools import read_screen_tool

TMUX = shutil.which("tmux")
needs_tmux = pytest.mark.skipif(TMUX is None, reason="no tmux to run panes in")

# Terminals by device number: a pane of each of two servers, the pseudoterminal fritter runs a session on, and a tab.
PANE, OTHER_PANE, FRITTER, TAB = 10, 11, 20, 1
WORK, PLAY = Path("/tmp/tmux-501/default"), Path("/tmp/tmux-501/play")
HANDS = Pane(WORK, "%3", "cc-hands", 2)
LAWS = Pane(PLAY, "%0", "laws", 0)
# A session under fritter in a pane of the first server: claude on fritter's pseudoterminal, fritter on the pane's in
# front of its shell. Another, in a pane of the second server, and one stopped there, behind it; and one in a plain tab.
TABLE = {
    process.pid: process
    for process in (
        Process(100, 1, 501, PANE, None),
        Process(101, 100, 501, PANE, PANE),
        Process(102, 101, 501, FRITTER, FRITTER),
        Process(200, 1, 501, OTHER_PANE, OTHER_PANE),
        Process(201, 1, 501, OTHER_PANE, None),
        Process(300, 1, 501, TAB, TAB),
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


def keyboard(pid: int, servers: Sequence[Server] = SERVERS) -> object:
    return keyboard_of(front_terminal(pid, TABLE), ancestor_terminals(pid, TABLE), servers)


def test_keys_reach_a_session_through_its_pane_only_while_it_is_in_front_of_that_pane_itself() -> None:
    assert keyboard(200) == LAWS
    # Stopped, or a background job: its shell has the pane's keys.
    assert keyboard(201) == Behind(LAWS)
    # On a terminal of its own inside the pane - fritter's here, an editor's or ssh's elsewhere - which has the pane's keys.
    assert keyboard(102) == Behind(HANDS)
    # Gone since its pane was read: nothing of it is in front of anything.
    assert keyboard(999) == NotInTmux()
    assert keyboard(300) == NotInTmux()
    assert keyboard(300, [Listed({PANE: HANDS}), Unanswered("tmux at /tmp/tmux-501/play did not answer list-panes in 2 seconds")]) == PaneUnread("tmux at /tmp/tmux-501/play did not answer list-panes in 2 seconds")


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


def recording(sockets: Path, session: str, into: Path) -> tuple[Pane, int]:
    """A new tmux session of the default server whose process turns bracketed paste on, as Claude Code does, and keeps
    every byte typed into its pane, raw, in `into`; its pane and the process's pid, once it is reading."""
    keep = f"printf '\\033[?2004h'; stty raw -echo; exec cat > {into}"
    started = subprocess.run([str(TMUX), "-f", "/dev/null", "-L", "default", "new-session", "-d", "-P", "-F", "#{pane_id} #{pane_pid}", "-s", session, "sh", "-c", keep], capture_output=True, text=True, check=True)
    pane, pid = started.stdout.split()
    deadline = time.monotonic() + 5
    while not into.exists():
        assert time.monotonic() < deadline, "the pane's process never began reading"
        time.sleep(0.05)
    return Pane(sockets / "default", pane, session, 0), int(pid)


def kept(into: Path, expected: bytes) -> bytes:
    """What the pane's process kept, once it has kept as much as `expected`, or after five seconds."""
    deadline = time.monotonic() + 5
    while len(into.read_bytes()) < len(expected) and time.monotonic() < deadline:
        time.sleep(0.05)
    return into.read_bytes()


@needs_tmux
@pytest.mark.parametrize(
    ("input", "received"),
    [
        # Pasted bracketed behind its space, its newline kept a newline, and sent by the Return behind the paste.
        (Text(PromptText("/fix the tests\nand report")), b"\x1b[200~ /fix the tests\nand report\x1b[201~\r"),
        (Command(CommandName("compact"), None), b"/compact\r"),
        # The command pressed as keys, and its arguments pasted behind it.
        (Command(CommandName("model"), PromptText("opus")), b"/model\x1b[200~ opus\x1b[201~\r"),
        (Key("escape"), b"\x1b"),
        (Key("shift_tab"), b"\x1b[Z"),
    ],
)
def test_what_is_typed_into_a_pane_reaches_its_process_as_someone_at_the_keyboard_would_type_it(input: Input, received: bytes, sockets: Path, tmp_path: Path) -> None:
    into = tmp_path / "typed"
    pane, _ = recording(sockets, "work", into)
    asyncio.run(type_into(os.environ, Type(SessionId("s1"), pane, input)))
    assert kept(into, received) == received
    # Each paste's buffer is gone once pasted.
    assert subprocess.run([str(TMUX), "-L", "default", "list-buffers"], capture_output=True, text=True, check=True).stdout == ""


@needs_tmux
def test_a_pane_tmux_could_not_type_into_is_said_with_why(sockets: Path, tmp_path: Path) -> None:
    pane, _ = recording(sockets, "work", tmp_path / "typed")
    with pytest.raises(Untyped, match=re.escape(f"tmux did not type into session s1: no tmux is on the PATH to type into pane {pane.id} of tmux at {pane.socket}")):
        asyncio.run(type_into({"PATH": "/nonexistent"}, Type(SessionId("s1"), pane, Key("escape"))))
    subprocess.run([str(TMUX), "-L", "default", "kill-server"], check=True)
    with pytest.raises(Untyped, match=re.escape(f"tmux did not type into session s1: tmux at {pane.socket} did not type into pane {pane.id}: ")):
        asyncio.run(type_into(os.environ, Type(SessionId("s1"), pane, Text(PromptText("hello")))))


@needs_tmux
def test_a_pane_gone_since_it_was_read_is_said_with_why_and_leaves_no_prompt_in_a_buffer(sockets: Path, tmp_path: Path) -> None:
    pane, _ = recording(sockets, "work", tmp_path / "typed")
    gone = Pane(pane.socket, "%99", "work", 0)
    with pytest.raises(Untyped, match=re.escape(f"tmux did not type into session s1: tmux at {pane.socket} did not type into pane %99: can't find pane: %99")):
        asyncio.run(type_into(os.environ, Type(SessionId("s1"), gone, Text(PromptText("the secret plan")))))
    assert subprocess.run([str(TMUX), "-L", "default", "list-buffers"], capture_output=True, text=True, check=True).stdout == ""


@needs_tmux
def test_an_escape_is_answered_only_once_it_has_been_read_alone(sockets: Path, tmp_path: Path) -> None:
    pane, _ = recording(sockets, "work", tmp_path / "typed")
    started = time.monotonic()
    asyncio.run(type_into(os.environ, Type(SessionId("s1"), pane, Key("escape"))))
    assert time.monotonic() - started >= tmux.LONE_ESCAPE


@needs_tmux
def test_keys_reach_a_process_in_front_of_its_pane_and_not_a_job_behind_it(sockets: Path, tmp_path: Path) -> None:
    pane, _ = run_in("default", "work")
    # A shell with job control, as a terminal gives one, and a job it put in the background.
    subprocess.run([str(TMUX), "-L", "default", "respawn-pane", "-k", "-t", pane, "zsh", "-f"], check=True)
    job = tmp_path / "job"
    subprocess.run([str(TMUX), "-L", "default", "send-keys", "-t", pane, f"sleep 120 & echo $! > {job}", "Enter"], check=True)
    deadline = time.monotonic() + 5
    while not job.exists() or not job.read_text().strip():
        assert time.monotonic() < deadline, "the shell never started its job"
        time.sleep(0.05)
    shell = int(subprocess.run([str(TMUX), "-L", "default", "display-message", "-p", "-t", pane, "#{pane_pid}"], capture_output=True, text=True, check=True).stdout)
    at = Pane(sockets / "default", pane, "work", 0)
    assert asyncio.run(tmux.keyboards([shell, int(job.read_text()), os.getpid()], os.environ)) == [at, Behind(at), NotInTmux()]


@needs_tmux
def test_a_session_nobody_wrapped_is_sent_a_draft_through_the_tmux_pane_it_runs_in(sockets: Path, tmp_path: Path) -> None:
    into = tmp_path / "typed"
    _, pid = recording(sockets, "work", into)
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None, typist=partial(type_into, os.environ), keyboards=partial(tmux.keyboards, environment=os.environ))
    asyncio.run(sessions.apply(Joined(Membership(SessionId("s1"), pid, Path("/code/cc-hands"), Path("/nonexistent")), "startup")))

    async def sent() -> object:
        # Inside a unit of work, as every tool call the daemon makes is.
        with unit("tool.run", lambda _: None):
            await sessions.draft(StageDraft(SessionId("s1"), Staged(PromptText("run the tests"), ())))
            return await sessions.draft(SendDraft(SessionId("s1")))

    assert asyncio.run(sent()) == Typed(SessionId("s1"), Text(PromptText("run the tests")))
    assert kept(into, b"\x1b[200~ run the tests\x1b[201~\r") == b"\x1b[200~ run the tests\x1b[201~\r"
