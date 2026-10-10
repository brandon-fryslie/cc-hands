"""`hands check`: each piece hands needs, found or named as missing, through the same edges a user's machine has."""

import contextlib
import ctypes
import errno
import http.server
import threading
import fcntl
import json
import os
import pty
import shutil
import socket
import subprocess
import tempfile
import termios
import sys
from collections import Counter
from collections.abc import Callable, Generator, Iterator, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import cast

import pytest

from hands.core.session import Membership, SessionId
from hands.core.tmux import Behind, Keyboard, NotInTmux, Pane, PaneUnread
from hands.daemon import readiness
from hands.daemon.cli import main
from hands.daemon.readiness import Missing, Ready, Unknown
from hands.sessions import audit, heartbeat, liveness, terminals, wrapper
from hands.sessions.hookconfig import PLUGIN_ID
from hands.sessions.home import Home
from hands.sessions.membership import write_membership
from hands.sessions.payload import Rejected
from hands.sessions.terminals import Terminal, Terminals, Undescribed, attended, terminal_processes
from hands.sessions.wrapper import shim_script
from hands.voice import transcription as transcribing
from hands.voice.backends import Account, ClaudeCodeBackend


@pytest.fixture
def root() -> Iterator[Path]:
    # Short on purpose: a test session's fritter socket lives under here, and macOS caps a socket path near 104 bytes.
    root = Path(tempfile.mkdtemp(prefix="ready-", dir="/tmp")).resolve()
    yield root
    shutil.rmtree(root)


@pytest.fixture(autouse=True)
def only_this_tests_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    """A check scans this machine's real processes, and any other at a terminal, a shell in another pane among them,
    may be caught mid-exec while it does: only a refusal of a process a test started is the test's."""
    scan = readiness.terminal_processes

    def own() -> Terminals:
        found = scan()
        return Terminals(found.found, [each for each in found.unread if each.parent == os.getpid()])

    monkeypatch.setattr(readiness, "terminal_processes", own)


def executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def claude_listing(root: Path, plugins: object) -> str:
    """A PATH whose `claude plugin list --json` prints plugins, and whose `claude auth status` says it is logged in unless
    the file logged-out is beside it."""
    executable(
        root / "real" / "claude",
        f"#!/bin/sh\nif [ \"$1 $2\" = 'auth status' ]; then\n"
        f"  if [ -e \"$(dirname \"$0\")/logged-out\" ]; then echo '{{\"loggedIn\": false}}'; exit 1; fi\n"
        f"  echo '{{\"loggedIn\": true}}'; exit 0\nfi\ncat <<'EOF'\n{json.dumps(plugins)}\nEOF\n",
    )
    return f"{root / 'real'}:/usr/bin:/bin"


def listed(id: str, enabled: bool, scope: str = "user") -> dict[str, object]:
    return {"id": id, "version": "f24447053e9c", "scope": scope, "enabled": enabled}


# The plugin


def test_an_enabled_plugin_is_ready(root: Path) -> None:
    found = readiness.plugin(claude_listing(root, [listed("other@elsewhere", False), listed(PLUGIN_ID, True)]))
    assert isinstance(found, Ready) and "/reload-plugins" in found.said


def test_a_plugin_enabled_at_any_scope_is_ready() -> None:
    assert isinstance(readiness.plugin_listed(json.dumps([listed(PLUGIN_ID, False), listed(PLUGIN_ID, True)])), Ready)


def test_a_disabled_plugin_is_missing_and_says_how_to_install_it() -> None:
    found = readiness.plugin_listed(json.dumps([listed(PLUGIN_ID, False)]))
    assert isinstance(found, Missing) and "`hands install-plugin`" in found.said


def test_a_plugin_not_installed_is_missing_and_says_how_to_install_it(root: Path) -> None:
    found = readiness.plugin(claude_listing(root, [listed("other@elsewhere", True)]))
    assert isinstance(found, Missing) and "`hands install-plugin`" in found.said


def test_an_install_for_one_project_is_not_one_for_every_session() -> None:
    project = {**listed(PLUGIN_ID, True, "local"), "projectPath": "/code/cc-hands"}
    found = readiness.plugin_listed(json.dumps([project]))
    assert isinstance(found, Missing) and "`hands install-plugin`" in found.said


def test_another_plugin_listed_unreadably_says_nothing_of_hands() -> None:
    assert isinstance(readiness.plugin_listed(json.dumps([{"broken": True}, listed(PLUGIN_ID, True)])), Ready)


@pytest.mark.parametrize(
    "printed", ["not json", '{"plugins": []}', '[{"id": "hands@cc-hands", "scope": "user"}]'], ids=["text", "object", "no-enabled"]
)
def test_a_listing_hands_cannot_read_is_unknown_not_missing(printed: str) -> None:
    assert isinstance(readiness.plugin_listed(printed), Unknown)


def test_a_claude_that_fails_to_list_is_unknown_and_says_why(root: Path) -> None:
    executable(root / "real" / "claude", "#!/bin/sh\necho 'not logged in' >&2\nexit 3\n")
    found = readiness.plugin(f"{root / 'real'}:/usr/bin:/bin")
    assert isinstance(found, Unknown) and "(3)" in found.said and "not logged in" in found.said


def test_a_listing_that_is_not_text_is_unknown(root: Path) -> None:
    executable(root / "real" / "claude", "#!/bin/sh\nprintf '\\377\\376'\n")
    assert isinstance(readiness.plugin(f"{root / 'real'}:/usr/bin:/bin"), Unknown)


def test_no_claude_to_ask_is_unknown(root: Path) -> None:
    assert isinstance(readiness.plugin(f"{root / 'empty'}:/usr/bin:/bin"), Unknown)


def test_the_plugin_is_asked_of_the_shim_as_a_session_would_ask_it(root: Path) -> None:
    # Off a terminal the shim runs the real claude, so the answer is the real one, not fritter's.
    executable(root / "bin" / "fritter", "#!/bin/sh\necho fritter ran >&2\nexit 9\n")
    executable(root / "bin" / "claude", shim_script(root / "bin" / "fritter", root / "wire.sock"))
    path = claude_listing(root, [listed(PLUGIN_ID, True)])
    assert isinstance(readiness.plugin(f"{root / 'bin'}:{path}"), Ready)


# The shim


def test_the_shim_first_on_path_is_ready(root: Path, fritter: Path) -> None:
    home = Home(root / "home")
    home.bin.mkdir(parents=True)
    shutil.copy2(fritter, home.bin / "fritter")
    executable(home.shim, shim_script(home.bin / "fritter", home.wire))
    executable(root / "real" / "claude", "#!/bin/sh\n")
    assert isinstance(readiness.shim(home, f"{home.bin}:{root / 'real'}"), Ready)


def test_an_installed_shim_behind_the_real_claude_wants_only_the_path(root: Path) -> None:
    home = Home(root / "home")
    executable(home.shim, shim_script(home.bin / "fritter", home.wire))
    executable(root / "real" / "claude", "#!/bin/sh\n")
    found = readiness.shim(home, f"{root / 'real'}:{home.bin}")
    assert isinstance(found, Missing)
    assert f"`claude` on this PATH is {root / 'real' / 'claude'}" in found.said
    assert f'export PATH="{home.bin}:$PATH"' in found.said and "install-fritter" not in found.said


def test_a_shim_whose_fritter_is_gone_is_missing(root: Path) -> None:
    home = Home(root / "home")
    executable(home.shim, shim_script(home.bin / "fritter", home.wire))
    found = readiness.shim(home, f"{home.bin}")
    assert isinstance(found, Missing) and f"its fritter {home.bin / 'fritter'} is not there to run" in found.said


def test_a_shim_whose_fritter_is_not_the_one_hands_carries_says_to_install_it_again(root: Path) -> None:
    # As after hands is upgraded: the home's copy is the old hands' fritter.
    home = Home(root / "home")
    executable(home.bin / "fritter", "#!/bin/sh\n")
    executable(home.shim, shim_script(home.bin / "fritter", home.wire))
    found = readiness.shim(home, f"{home.bin}")
    assert isinstance(found, Missing)
    assert f"its fritter {home.bin / 'fritter'} is not the one this hands carries, {wrapper.PACKAGED}: run `hands install-fritter`" in found.said


@pytest.mark.parametrize("installed", [True, False], ids=["shim", "none"])
def test_a_hands_built_without_its_fritter_names_the_rebuild_not_install_fritter(root: Path, monkeypatch: pytest.MonkeyPatch, installed: bool) -> None:
    home = Home(root / "home")
    if installed:
        executable(home.bin / "fritter", "#!/bin/sh\n")
        executable(home.shim, shim_script(home.bin / "fritter", home.wire))
    monkeypatch.setattr(wrapper, "PACKAGED", root / "package" / "bin" / "fritter")
    found = readiness.shim(home, f"{home.bin}")
    assert isinstance(found, Missing) and found.said.startswith(f"hands' package carries no fritter at {root / 'package' / 'bin' / 'fritter'}: install hands again")


def runs_subcommands_as_sessions(text: str) -> str:
    return text.replace("session=no ;;", ";;")


def before_the_wire(text: str) -> str:
    return "\n".join(text.splitlines()[:3])


@pytest.mark.parametrize("older", [runs_subcommands_as_sessions, before_the_wire])
def test_a_shim_an_older_hands_wrote_says_to_install_it_again(root: Path, fritter: Path, older: Callable[[str], str]) -> None:
    # As after hands is upgraded: the shim is not the script this hands writes, so it may not run what `run` says it does;
    # the second is one written before the wire, with no wire line.
    home = Home(root / "home")
    home.bin.mkdir(parents=True)
    shutil.copy2(fritter, home.bin / "fritter")
    executable(home.shim, older(shim_script(home.bin / "fritter", home.wire)))
    found = readiness.shim(home, f"{home.bin}")
    assert found == Missing(f"`claude` on this PATH is hands' shim, {home.shim}, but not the one this hands writes: run `hands install-fritter`")


def test_another_homes_current_shim_is_ready_by_the_wire_it_names(root: Path, fritter: Path) -> None:
    # As a probe's temporary home checked while ~/.hands/bin is first on PATH.
    home, other = Home(root / "home"), Home(root / "other")
    other.bin.mkdir(parents=True)
    shutil.copy2(fritter, other.bin / "fritter")
    executable(other.shim, shim_script(other.bin / "fritter", other.wire))
    assert isinstance(readiness.shim(home, f"{other.bin}"), Ready)


def test_no_shim_installed_says_to_install_it(root: Path) -> None:
    found = readiness.shim(Home(root / "home"), f"{root / 'empty'}")
    assert isinstance(found, Missing)
    assert "`claude` on this PATH is nothing" in found.said and "hands install-fritter" in found.said


# The grant


def test_the_grant_is_ready_when_given_and_missing_with_where_to_give_it_when_not() -> None:
    assert isinstance(readiness.grant(True), Ready)
    missing = readiness.grant(False)
    assert isinstance(missing, Missing) and "Privacy & Security > Input Monitoring" in missing.said


def test_a_running_hands_has_the_grant_of_the_app_it_runs_in_though_this_one_has_none() -> None:
    # hands.app holds the grant; the terminal a check runs in does not.
    held = readiness.hears(False, Ready("hands is running: pid 7"))
    assert isinstance(held, Ready) and "hands is running, so it started with the Input Monitoring grant of the app it runs in" in held.said


# The running sessions


def untmuxed(pids: Sequence[int]) -> list[Keyboard]:
    """No pane in front of any process: the check never reads the tmux the tests may run inside."""
    return [NotInTmux()] * len(pids)


PANE = Pane(Path("/tmp/tmux-501/default"), "%12", "work", 3)


def joined(home: Home, name: str, pid: int, fritter: Path | None) -> Membership:
    membership = Membership(SessionId(f"0f1e2d3c-aaaa-bbbb-cccc-{name:0>12}"), pid, Path(f"/code/{name}"), Path("/nowhere/t.jsonl"), fritter)
    write_membership(home, membership)
    return membership


def installed(root: Path) -> str:
    """A PATH whose real `claude` is a link into an install directory of versions, as Claude Code installs itself."""
    version = root / "install" / "9.9.9"
    if not version.exists():
        version.parent.mkdir(parents=True)
        shutil.copy("/bin/sleep", version)
        (root / "bin").mkdir()
        (root / "bin" / "claude").symlink_to(version)
    return f"{root / 'bin'}:/usr/bin:/bin"


@contextlib.contextmanager
def at_a_terminal(executable: Path, cwd: Path, arguments: Sequence[str] = ("30",), piped: bool = False, pass_fds: Sequence[int] = ()) -> Generator[int]:
    """A process started as executable in cwd with a terminal of its own, as a session runs, its stdin a pipe if piped;
    its pid, once it runs executable."""
    controller, terminal = pty.openpty()
    # Popen returns only once the child has exec'd, so the process is executable from the first look at it.
    process = subprocess.Popen(
        [executable, *arguments],
        cwd=cwd,
        # As a shell starts a program: with the directory it starts in as its PWD.
        env={**os.environ, "PWD": str(cwd)},
        pass_fds=pass_fds,
        stdin=subprocess.PIPE if piped else terminal,
        stdout=terminal,
        stderr=terminal,
        start_new_session=True,
        preexec_fn=lambda: fcntl.ioctl(1, termios.TIOCSCTTY, 0),
    )
    try:
        yield process.pid
    finally:
        process.kill()
        process.wait()
        if process.stdin is not None:
            process.stdin.close()
        os.close(controller)
        os.close(terminal)


def test_no_running_session_is_said_as_none_not_left_out(root: Path) -> None:
    assert readiness.sessions(Home(root / "home"), installed(root), untmuxed) == Ready("running sessions hands knows of: 0, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0")


def test_each_running_session_that_cannot_be_typed_into_is_named_with_why(root: Path) -> None:
    home = Home(root / "home")
    listening = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listening.bind(str(root / "live.sock"))
    listening.listen()
    # Running processes of their own, since one process holds one session.
    sleepers = [subprocess.Popen(["sleep", "30"]) for _ in range(3)]
    try:
        joined(home, "wrapped", sleepers[0].pid, root / "live.sock")
        joined(home, "unwrapped", sleepers[1].pid, None)
        joined(home, "orphaned", sleepers[2].pid, root / "gone.sock")
        found = readiness.sessions(home, installed(root), untmuxed)
    finally:
        listening.close()
        for sleeper in sleepers:
            sleeper.kill()
            sleeper.wait()
    assert isinstance(found, Missing)
    assert found.said.startswith("running sessions hands knows of: 3, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0\n  hands cannot reach these:")
    assert f"/code/orphaned (pid {sleepers[2].pid}) has lost its fritter, whose socket {root / 'gone.sock'} is gone" in found.said
    assert f"/code/unwrapped (pid {sleepers[1].pid}) was started outside fritter and runs in no tmux pane, so it cannot be typed into: restart it" in found.said
    assert found.said.endswith(f"\n  these are typed into:\n    /code/wrapped (pid {sleepers[0].pid}) is typed into through its fritter")


def test_a_session_started_outside_fritter_in_front_of_its_tmux_pane_is_typed_into_through_it(root: Path) -> None:
    home = Home(root / "home")
    sleeper = subprocess.Popen(["sleep", "30"])
    asked: list[Sequence[int]] = []

    def in_front(pids: Sequence[int]) -> list[Keyboard]:
        asked.append(pids)
        return [PANE] * len(pids)

    try:
        joined(home, "unwrapped", sleeper.pid, None)
        found = readiness.sessions(home, installed(root), in_front)
    finally:
        sleeper.kill()
        sleeper.wait()
    assert asked == [[sleeper.pid]]
    assert found == Ready(f"running sessions hands knows of: 1, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0\n  these are typed into:\n    /code/unwrapped (pid {sleeper.pid}) is typed into through its tmux pane %12")


def unwrapped(pane: Keyboard) -> readiness.Finding:
    member = Membership(SessionId("0f1e2d3c-aaaa-bbbb-cccc-000000000007"), 7, Path("/code/bare"), Path("/nowhere/t.jsonl"))
    return readiness.sessions_found([member], set(), [], readiness.Unrecorded([], Counter(), []), {7: pane})


def test_a_session_started_outside_fritter_behind_another_program_in_its_pane_is_missing_with_the_way_back() -> None:
    assert unwrapped(Behind(PANE)) == Missing(
        "running sessions hands knows of: 1, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0\n  hands cannot reach these:\n"
        "    /code/bare (pid 7) was started outside fritter, and another program has the keyboard of the tmux pane it runs in, %12 (window 3 of work), "
        "so it cannot be typed into: bring it back to the front of its pane, or restart it from a PATH whose `claude` is hands' shim"
    )


def test_a_session_started_outside_fritter_whose_pane_could_not_be_read_is_unknown_and_says_why() -> None:
    assert unwrapped(PaneUnread("tmux at /tmp/tmux-501/default did not answer list-panes in 5 seconds")) == Unknown(
        "running sessions hands knows of: 1, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0\n  whether hands can reach these is unknown:\n"
        "    /code/bare (pid 7) was started outside fritter, and which tmux pane it runs in could not be read, so whether it can be typed into is unknown: "
        "tmux at /tmp/tmux-501/default did not answer list-panes in 5 seconds"
    )


def test_a_session_whose_reach_is_unknown_is_not_said_among_those_hands_cannot_reach() -> None:
    lost = Membership(SessionId("0f1e2d3c-aaaa-bbbb-cccc-000000000008"), 8, Path("/code/lost"), Path("/nowhere/t.jsonl"), Path("/gone.sock"))
    bare = Membership(SessionId("0f1e2d3c-aaaa-bbbb-cccc-000000000007"), 7, Path("/code/bare"), Path("/nowhere/t.jsonl"))
    found = readiness.sessions_found([lost, bare], set(), [], readiness.Unrecorded([], Counter(), []), {7: PaneUnread("tmux broke"), 8: NotInTmux()})
    assert found == Missing(
        "running sessions hands knows of: 2, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0\n"
        "  hands cannot reach these:\n"
        "    /code/lost (pid 8) has lost its fritter, whose socket /gone.sock is gone, so it cannot be typed into: "
        "restart it from a PATH whose `claude` is hands' shim\n"
        "  whether hands can reach these is unknown:\n"
        "    /code/bare (pid 7) was started outside fritter, and which tmux pane it runs in could not be read, so whether it can be typed into is unknown: tmux broke"
    )


def test_a_session_whose_process_has_ended_is_not_running(root: Path) -> None:
    home = Home(root / "home")
    ended = subprocess.Popen(["true"])
    ended.wait()
    joined(home, "ended", ended.pid, None)
    assert readiness.sessions(home, installed(root), untmuxed) == Ready("running sessions hands knows of: 0, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0")


def test_an_unreadable_membership_file_is_named_and_left_where_it_is(root: Path) -> None:
    home = Home(root / "home")
    home.memberships.mkdir(parents=True)
    bad = home.membership(SessionId("bad"))
    bad.write_text("{not json")
    found = readiness.sessions(home, installed(root), untmuxed)
    assert isinstance(found, Missing) and f"{bad} names no session hands can read" in found.said
    assert bad.exists()


def test_a_session_that_never_ran_a_hook_is_named_by_cwd_and_pid_with_the_fix(root: Path) -> None:
    home = Home(root / "home")
    path = installed(root)
    project = root / "project"
    project.mkdir()
    with at_a_terminal(root / "install" / "9.9.9", project) as pid:
        found = readiness.sessions(home, path, untmuxed)
    assert isinstance(found, Missing) and f"{project} (pid {pid}) is a session hands has no record of" in found.said


def test_a_claude_printing_or_piped_into_at_a_terminal_is_no_session(root: Path) -> None:
    home = Home(root / "home")
    path = installed(root)
    shell = root / "install" / "9.9.7"
    shutil.copy("/bin/zsh", shell)
    # The shell waits on its sleep, so it is still the process its -p was given to.
    with at_a_terminal(shell, root, ["-c", "sleep 30; :", "-p", "hello"]), at_a_terminal(root / "install" / "9.9.9", root, piped=True):
        found = readiness.sessions(home, path, untmuxed)
    assert found == Ready("running sessions hands knows of: 0, runs of claude at a terminal that are none: piped 1, subcommand 0, print 1")


def test_a_check_run_from_a_removed_directory_says_its_own_config_cannot_be_told(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gone = root / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    found = readiness.unrecorded(Home(root / "home"), installed(root), set())
    assert isinstance(found, readiness.Unfindable) and found.said.startswith("cannot tell which Claude Code config this check runs under")


def test_the_kernel_says_what_a_process_at_a_terminal_was_started_with_and_whether_it_reads_and_writes_it(root: Path) -> None:
    with at_a_terminal(Path("/bin/sleep"), root) as reading, at_a_terminal(Path("/bin/sleep"), root, ["31"], piped=True) as piped:
        found = {process.pid: process for process in terminal_processes().found}
        said = {pid: (found[pid].arguments, attended(found[pid])) for pid in (reading, piped)}
    assert said == {reading: (("30",), True), piped: (("31",), False)}


def test_a_process_reading_its_terminal_as_dev_tty_reads_its_terminal(root: Path) -> None:
    controller, terminal = pty.openpty()

    def reopened() -> None:
        # As `xargs -o` and `< /dev/tty` give a run its stdin: the terminal, opened by the name each process has for its own.
        fcntl.ioctl(1, termios.TIOCSCTTY, 0)
        os.dup2(os.open("/dev/tty", os.O_RDWR), 0)

    process = subprocess.Popen(["/bin/sleep", "30"], cwd=root, stdin=subprocess.DEVNULL, stdout=terminal, stderr=terminal, start_new_session=True, preexec_fn=reopened)
    try:
        found = next(found for found in terminal_processes().found if found.pid == process.pid)
        assert attended(found)
    finally:
        process.kill()
        process.wait()
        os.close(controller)
        os.close(terminal)


def test_a_process_the_kernel_will_not_describe_is_named_and_hides_no_other(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started_as = terminals._started_as  # pyright: ignore[reportPrivateUsage]

    # As kern.procargs2 answers a process caught mid-exec, or on its way out.
    def refusing(pid: int) -> tuple[Path, tuple[str, ...], dict[str, str]]:
        if pid == refused:
            ctypes.set_errno(errno.EIO)
            terminals._raise_unless_exited(pid, "kern.procargs2")  # pyright: ignore[reportPrivateUsage]
        return started_as(pid)

    monkeypatch.setattr(terminals, "_started_as", refusing)
    with at_a_terminal(Path("/bin/sleep"), root) as refused, at_a_terminal(Path("/bin/sleep"), root) as described:
        scan = terminal_processes()
        # Only this test's own: any other process at a terminal on this machine may be caught mid-exec too.
        unread = [each for each in scan.unread if each.pid in (refused, described)]
    assert (unread, described in {process.pid for process in scan.found}) == ([Undescribed(refused, os.getpid(), "kern.procargs2", errno.EIO)], True)


def test_a_process_that_has_ended_is_no_process_the_kernel_would_not_describe() -> None:
    process = subprocess.Popen(["/usr/bin/true"])
    process.wait()
    ctypes.set_errno(errno.EIO)
    with pytest.raises(terminals._Exited):  # pyright: ignore[reportPrivateUsage]
        terminals._raise_unless_exited(process.pid, "kern.procargs2")  # pyright: ignore[reportPrivateUsage]


def test_processes_that_could_not_be_looked_at_leave_unknown_whether_any_is_a_session() -> None:
    unread = [Undescribed(29648, 1, "kern.procargs2", errno.EIO), Undescribed(29649, 1, "proc_pidfdinfo of fd 0", errno.EIO)]
    assert readiness.sessions_found([], set(), [], readiness.Unrecorded([], Counter(), unread), {}) == Unknown(
        "running sessions hands knows of: 0, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0\n  whether hands can reach these is unknown:\n"
        "    the kernel would not describe these processes at a terminal, so whether any is a session hands has no record of is unknown: "
        "pid 29648 (kern.procargs2: Input/output error), pid 29649 (proc_pidfdinfo of fd 0: Input/output error)"
    )


def test_a_process_not_described_is_unknown_unless_hands_knows_it_or_it_is_a_runs_helper() -> None:
    run = terminal(1, "/v/2.1.288")
    stranger, member, helper = (Undescribed(pid, parent, "kern.procargs2", errno.EIO) for pid, parent in ((7, 0), (8, 0), (9, 1)))
    found = readiness.unjoined(Home(Path("/h")), Path("/v/2.1.288"), CHECKED, Terminals([run], [stranger, member, helper]), {1, 8}, reads_its_terminal)
    assert found.unread == [stranger]


def test_a_run_of_claude_whose_terminal_the_kernel_will_not_say_is_unknown_not_a_failed_scan() -> None:
    told, piped = terminal(1, "/v/2.1.288"), terminal(2, "/v/2.1.288", tty=PIPED)
    refusal = Undescribed(1, 0, "proc_pidfdinfo of fd 0", errno.EIO)

    def refusing(process: Terminal) -> bool | Undescribed:
        return refusal if process is told else reads_its_terminal(process)

    found = readiness.unjoined(Home(Path("/h")), Path("/v/2.1.288"), CHECKED, Terminals([told, piped], []), set(), refusing)
    assert (found.sessions, found.others, found.unread) == ([], {"piped": 1}, [refusal])


def test_a_reused_pid_of_an_ended_session_hides_no_session_hands_has_no_record_of(root: Path) -> None:
    home = Home(root / "home")
    path = installed(root)
    with at_a_terminal(root / "install" / "9.9.9", root) as pid:
        # Written before the process under its pid started, as an ended session's file left behind is.
        os.utime(home.membership(joined(home, "ended", pid, None).id), (0, 0))
        found = readiness.sessions(home, path, untmuxed)
    assert isinstance(found, Missing) and f"(pid {pid}) is a session hands has no record of" in found.said


def test_a_session_on_a_version_pruned_since_it_started_is_still_found(root: Path) -> None:
    home = Home(root / "home")
    path = installed(root)
    pruned = root / "install" / "9.9.8"
    shutil.copy("/bin/sleep", pruned)
    with at_a_terminal(pruned, root) as pid:
        pruned.unlink()
        found = readiness.sessions(home, path, untmuxed)
    assert isinstance(found, Missing) and f"(pid {pid}) is a session hands has no record of" in found.said


# Moves to the directory it is given, says so on the descriptor it is given, and stays.
MOVES = 'cd "$1" && printf . >&"$2" && read -r _'


def test_a_program_started_by_a_relative_path_is_named_where_it_is_after_it_changes_directory(root: Path) -> None:
    read, write = os.pipe()
    with open(read, "rb", buffering=0) as moved, open(write, "wb", buffering=0) as said:
        # As the shim starts the real `claude` under a relative PATH entry, and as a session then moves into a worktree.
        with at_a_terminal(Path("bin/bash"), Path("/"), ("-c", MOVES, "bash", str(root), str(write)), pass_fds=(write,)) as pid:
            # Only the program can say so now: one that ends without moving is read as nothing, not waited on.
            said.close()
            assert moved.read(1) == b"."
            [process] = [process for process in terminal_processes().found if process.pid == pid]
    assert (process.executable, process.cwd) == (Path("/bin/bash"), root)


def test_a_session_hands_knows_of_is_not_named_as_unknown(root: Path) -> None:
    home = Home(root / "home")
    path = installed(root)
    listening = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listening.bind(str(root / "live.sock"))
    listening.listen()
    try:
        with at_a_terminal(root / "install" / "9.9.9", root) as pid:
            joined(home, "known", pid, root / "live.sock")
            found = readiness.sessions(home, path, untmuxed)
    finally:
        listening.close()
    assert found == Ready(f"running sessions hands knows of: 1, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0\n  these are typed into:\n    /code/known (pid {pid}) is typed into through its fritter")


CONFIG = Path("/home/.claude")
CHECKED = readiness.config_dir({"CLAUDE_CONFIG_DIR": str(CONFIG)}, Path("/"))
# The terminal a process made by `terminal` reads and writes, unless it is made with another, as a piped one is.
ATTENDED, PIPED = 1, 2


def terminal(
    pid: int, executable: str, cwd: str = "/code", parent: int = 0, config: Path = CONFIG, arguments: tuple[str, ...] = (), tty: int = ATTENDED
) -> Terminal:
    return Terminal(pid, parent, Path(executable), Path(cwd), {"CLAUDE_CONFIG_DIR": str(config)}, arguments, tty)



def reads_its_terminal(process: Terminal) -> bool:
    return process.tty == ATTENDED


def unjoined(claude: str, terminals: list[Terminal], members: set[int] | None = None, home: Home = Home(Path("/h"))) -> list[Terminal]:
    return [session.process for session in readiness.unjoined(home, Path(claude), CHECKED, Terminals(terminals, []), members or set(), reads_its_terminal).sessions]


def test_a_session_is_a_terminal_process_of_any_version_of_the_real_claudes_install() -> None:
    terminals = [
        terminal(1, "/v/2.1.286", "/code/old"),
        terminal(2, "/v/2.1.288", "/code/joined"),
        terminal(3, "/bin/zsh", "/code/shell"),
        terminal(4, "/elsewhere/v/2.1.288", "/code/other-install"),
    ]
    assert unjoined("/v/2.1.288", terminals, {2}) == [terminals[0]]


def test_a_cask_keeps_each_version_in_a_directory_named_for_it() -> None:
    old = terminal(1, "/opt/homebrew/Caskroom/claude-code/2.1.286/claude")
    assert unjoined("/opt/homebrew/Caskroom/claude-code/2.1.288/claude", [old]) == [old]


def test_a_pre_release_is_a_version_of_the_same_install() -> None:
    beta = terminal(1, "/v/2.1.300-beta.1")
    assert unjoined("/v/2.1.299", [beta]) == [beta]


def test_a_plain_claude_is_itself_and_not_its_directorys_other_programs() -> None:
    terminals = [terminal(1, "/usr/local/bin/claude"), terminal(2, "/usr/local/bin/nvim")]
    assert unjoined("/usr/local/bin/claude", terminals) == [terminals[0]]


def test_a_session_under_another_config_as_the_brain_is_has_other_plugins_and_is_not_named() -> None:
    brain = terminal(1, "/v/2.1.288", config=Path("/h/brain"))
    assert unjoined("/v/2.1.288", [brain]) == []


@pytest.mark.parametrize("checked, started", [(".claude", "dotfiles/claude"), ("dotfiles/claude", ".claude")])
def test_a_session_under_the_same_config_by_way_of_a_link_is_named(tmp_path: Path, checked: str, started: str) -> None:
    (tmp_path / "dotfiles" / "claude").mkdir(parents=True)
    (tmp_path / ".claude").symlink_to(tmp_path / "dotfiles" / "claude")
    session = terminal(1, "/v/2.1.288", config=tmp_path / started)
    config = readiness.config_dir({"CLAUDE_CONFIG_DIR": str(tmp_path / checked)}, Path("/"))
    found = readiness.unjoined(Home(Path("/h")), Path("/v/2.1.288"), config, Terminals([session], []), set(), reads_its_terminal)
    assert [joined.process for joined in found.sessions] == [session]


def test_a_relative_config_is_read_from_the_working_directory_of_the_process_that_names_it(tmp_path: Path) -> None:
    assert readiness.config_dir({"CLAUDE_CONFIG_DIR": "alt"}, tmp_path) == readiness.config_dir({"CLAUDE_CONFIG_DIR": str(tmp_path / "alt")}, Path("/"))


def test_with_no_config_named_a_process_runs_under_the_claude_in_its_own_home(tmp_path: Path) -> None:
    assert readiness.config_dir({"HOME": str(tmp_path)}, Path("/")) == readiness.config_dir({"CLAUDE_CONFIG_DIR": str(tmp_path / ".claude")}, Path("/"))


def test_a_config_named_through_a_link_loop_is_a_config_no_session_runs_under(tmp_path: Path) -> None:
    (tmp_path / "loop").symlink_to(tmp_path / "loop")
    assert readiness.config_dir({"CLAUDE_CONFIG_DIR": str(tmp_path / "loop")}, Path("/")) != readiness.config_dir({"HOME": str(tmp_path)}, Path("/"))


def test_a_sessions_own_helper_run_from_its_executable_is_not_a_session() -> None:
    session, helper = terminal(10, "/v/2.1.288"), terminal(11, "/v/2.1.288", parent=10)
    assert unjoined("/v/2.1.288", [session, helper], {10}) == []


def test_a_claude_the_shim_would_not_have_run_as_a_session_is_not_named() -> None:
    printing, piped = terminal(1, "/v/2.1.288", arguments=("-p", "hello")), terminal(2, "/v/2.1.288", tty=PIPED)
    prompted = terminal(3, "/v/2.1.288", arguments=("--", "-p"))
    controlling = terminal(4, "/v/2.1.288", arguments=("remote-control", "--spawn", "worktree"))
    asked = terminal(5, "/v/2.1.288", arguments=("fix the readme",))
    found = readiness.unjoined(
        Home(Path("/h")), Path("/v/2.1.288"), CHECKED, Terminals([printing, piped, prompted, controlling, asked], []), set(), reads_its_terminal
    )
    assert [session.process for session in found.sessions] == [prompted, asked]
    assert found.others == {"print": 1, "piped": 1, "subcommand": 1}


def test_the_helper_of_a_claude_that_is_no_session_is_no_session_either() -> None:
    printing, helper = terminal(10, "/v/2.1.288", arguments=("-p", "hello")), terminal(11, "/v/2.1.288", parent=10)
    assert unjoined("/v/2.1.288", [printing, helper]) == []


def test_a_session_hands_has_no_record_of_is_told_to_reload_where_it_can_be_typed_into_and_to_restart_where_not(root: Path) -> None:
    home = Home(root / "home")
    fritter = terminal(5, str(home.fritter))
    inside, outside = terminal(10, "/v/2.1.288", "/code/in", parent=5), terminal(11, "/v/2.1.288", "/code/out", parent=6)
    paned, behind = terminal(12, "/v/2.1.288", "/code/paned", parent=6), terminal(13, "/v/2.1.288", "/code/behind", parent=6)
    unread = terminal(14, "/v/2.1.288", "/code/unread", parent=6)
    sessions = readiness.unjoined(home, Path("/v/2.1.288"), CHECKED, Terminals([fritter, inside, outside, paned, behind, unread], []), set(), reads_its_terminal)
    # The fritter that wrapped a session types into it whatever its pane is.
    panes: dict[int, Keyboard] = {10: PaneUnread("no tmux"), 11: NotInTmux(), 12: PANE, 13: Behind(PANE), 14: PaneUnread("tmux did not answer")}
    found = readiness.sessions_found([], set(), [], sessions, panes)
    assert isinstance(found, Missing)
    assert "/code/in (pid 10) is a session hands has no record of, so it cannot be reached: /reload-plugins in it" in found.said
    assert "/code/out (pid 11) is a session hands has no record of, started outside fritter and in no tmux pane, so it cannot be typed into: restart it" in found.said
    assert "/code/paned (pid 12) is a session hands has no record of, so it cannot be reached: /reload-plugins in it" in found.said
    assert (
        "/code/behind (pid 13) is a session hands has no record of, started outside fritter, and another program has the keyboard of the tmux pane it runs in, "
        "%12 (window 3 of work), so it cannot be typed into: bring it back to the front of its pane, then /reload-plugins in it"
    ) in found.said
    assert "/code/unread (pid 14) is a session hands has no record of, started outside fritter, and which tmux pane it runs in could not be read (tmux did not answer)" in found.said


def test_with_no_real_claude_a_session_hands_has_no_record_of_cannot_be_found(root: Path) -> None:
    found = readiness.sessions(Home(root / "home"), "/nowhere", untmuxed)
    assert isinstance(found, Unknown) and "this PATH has no `claude` of its own" in found.said


def test_a_claude_that_is_a_script_cannot_tell_its_sessions(root: Path) -> None:
    (root / "bin").mkdir()
    (root / "bin" / "claude").write_text("#!/usr/bin/env node\n")
    (root / "bin" / "claude").chmod(0o755)
    found = readiness.sessions(Home(root / "home"), str(root / "bin"), untmuxed)
    assert isinstance(found, Unknown) and "is a script, so its sessions run as its interpreter" in found.said


def test_sessions_that_cannot_be_looked_for_are_said_beside_the_ones_that_cannot_be_reached() -> None:
    unreadable = liveness.Unreadable(Path("/h/sessions/bad.json"), b"{", Rejected("not json"))
    found = readiness.sessions_found([], set(), [unreadable], readiness.Unfindable("why"), {})
    assert isinstance(found, Missing) and "bad.json names no session" in found.said and "cannot be found: why" in found.said


def test_sessions_that_cannot_be_looked_at_are_unknown(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(_pids: object) -> dict[int, float]:
        raise OSError(1, "sysctl refused")

    monkeypatch.setattr(readiness, "process_starts", refused)
    found = readiness.sessions(Home(root / "home"), installed(root), untmuxed)
    assert isinstance(found, Unknown) and "sysctl refused" in found.said


# Claude Code


def test_a_native_claude_on_path_is_ready(root: Path) -> None:
    found = readiness.claude(installed(root))
    assert found == Ready(f"Claude Code is installed: {root / 'install' / '9.9.9'}")


def test_no_claude_on_path_is_missing_and_names_its_installer(root: Path) -> None:
    found = readiness.claude(f"{root / 'empty'}:/usr/bin:/bin")
    assert isinstance(found, Missing) and "no `claude` of its own" in found.said and readiness.INSTALL in found.said


def test_a_claude_that_is_a_script_is_missing_and_names_the_native_one(root: Path) -> None:
    executable(root / "real" / "claude", "#!/bin/sh\n")
    found = readiness.claude(f"{root / 'real'}:/usr/bin:/bin")
    assert isinstance(found, Missing) and "is a script" in found.said and "`claude install`" in found.said


# PortAudio


def test_portaudio_that_pyaudio_loads_is_ready() -> None:
    found = readiness.portaudio()
    assert isinstance(found, Ready) and "PortAudio" in found.said


def test_a_pyaudio_that_cannot_load_is_missing_and_names_homebrews_portaudio(monkeypatch: pytest.MonkeyPatch) -> None:
    # As when Homebrew's portaudio is gone: importing PyAudio's extension fails.
    monkeypatch.setitem(sys.modules, "pyaudio", None)
    found = readiness.portaudio()
    assert isinstance(found, Missing) and readiness.INSTALL in found.said


# hands on PATH


def hands_printing(root: Path, said: str, code: int = 0) -> str:
    """A PATH whose `hands --version` prints said and exits code."""
    executable(root / "tools" / "hands", f"#!/bin/sh\necho '{said}'\nexit {code}\n")
    return f"{root / 'tools'}:/usr/bin:/bin"


def test_this_hands_on_path_is_ready(root: Path) -> None:
    found = readiness.installed(hands_printing(root, f"hands {version('hands')}"))
    assert isinstance(found, Ready) and str(root / "tools" / "hands") in found.said


def test_no_hands_on_path_is_missing_and_says_the_install_command_puts_it_there(root: Path) -> None:
    found = readiness.installed(f"{root / 'empty'}:/usr/bin:/bin")
    assert isinstance(found, Missing) and "`hands plugin`" in found.said and readiness.INSTALL in found.said


def test_another_hands_on_path_is_missing_naming_both(root: Path) -> None:
    # As after an upgrade that left an older install ahead on PATH: sessions would run its hooks.
    found = readiness.installed(hands_printing(root, "hands 0.0.1"))
    assert isinstance(found, Missing) and "is hands 0.0.1, not this hands" in found.said


def test_a_hands_that_cannot_say_its_version_is_unknown(root: Path) -> None:
    found = readiness.installed(hands_printing(root, "broken", 3))
    assert isinstance(found, Unknown) and "(3)" in found.said


# The backend


def test_a_brain_with_no_login_is_missing_and_names_the_command(root: Path, monkeypatch: pytest.MonkeyPatch, fake_claude: Path) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    found, _ = readiness.configured(Home(root / "home"), os.environ)
    assert isinstance(found, Missing) and "`hands login` gives it one" in found.said


def transcribed(home: Home, url: str) -> None:
    """Settings naming LowTalker at `url`."""
    home.root.mkdir(parents=True, exist_ok=True)
    home.config.write_text(f'[transcription]\nurl = "{url}"\n')


def test_settings_hands_cannot_read_are_missing_naming_the_file(root: Path) -> None:
    home = Home(root / "home")
    home.root.mkdir(parents=True)
    home.config.write_text("[llm\n")
    reached, heard = readiness.configured(home, {})
    assert isinstance(reached, Missing) and str(home.config) in reached.said
    # Where the server is was never read: the step is not known missing, and is not said twice.
    assert heard == Unknown("which server transcribes is in the settings hands cannot read")


def test_a_setting_in_the_environment_is_missing_as_the_start_refuses_it(root: Path) -> None:
    found, _ = readiness.configured(Home(root / "home"), {"HANDS_DEBUG": "1"})
    assert isinstance(found, Missing) and "HANDS_DEBUG set, and hands reads no setting from the environment" in found.said


def test_a_brain_that_cannot_be_asked_is_unknown(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(root / "home")

    def unspawnable(*_: object) -> object:
        raise PermissionError("claude is not executable")

    monkeypatch.setattr(readiness, "resolve", unspawnable)
    found, _ = readiness.configured(home, {})
    assert isinstance(found, Unknown) and "claude is not executable" in found.said


def test_the_brain_says_its_account_and_never_a_key() -> None:
    found = readiness.reaching(ClaudeCodeBackend(model="claude-sonnet-5", config_dir=Path("/h/brain"), account=Account("claude.ai", "brain@example.com")))
    assert found == Ready("the brain is logged in as brain@example.com (claude.ai), and reaches claude-sonnet-5")


# Transcription


class Transcriber(http.server.BaseHTTPRequestHandler):
    """A transcription server answering every upload with the server's status and text."""

    def do_POST(self) -> None:
        server = cast(Answering, self.server)
        length = int(self.headers["Content-Length"])
        server.uploads.append((self.path, self.rfile.read(length)))
        self.send_response(server.status)
        self.end_headers()
        self.wfile.write(server.text.encode())

    def log_message(self, format: str, *args: object) -> None:
        pass


class Answering(http.server.ThreadingHTTPServer):
    def __init__(self, status: int, text: str) -> None:
        super().__init__(("127.0.0.1", 0), Transcriber)
        self.status = status
        self.text = text
        self.uploads: list[tuple[str, bytes]] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


@contextlib.contextmanager
def serving(status: int = 200, text: str = '{"text": "", "segments": []}') -> Generator[Answering]:
    server = Answering(status, text)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def test_a_server_that_transcribes_silence_is_ready_and_was_sent_a_wav() -> None:
    with serving() as server:
        assert readiness.transcription(server.url) == Ready(f"LowTalker at {server.url} transcribes a hold")
    [(path, body)] = server.uploads
    assert path == "/v1/audio/transcriptions" and b"RIFF" in body and b'name="model"' in body


def test_nothing_listening_is_missing_and_names_lowtalker() -> None:
    found = readiness.transcription("http://127.0.0.1:9/v1")
    assert isinstance(found, Missing) and "low-talker" in found.said and "Serve Transcription" in found.said


def test_an_address_that_does_not_resolve_is_missing_and_points_at_the_config_not_lowtalker() -> None:
    found = readiness.transcription("http://lowtalker.invalid:8610/v1")
    assert isinstance(found, Missing) and "cannot be reached" in found.said and "config.toml" in found.said and "Serve Transcription" not in found.said


def test_a_server_still_loading_its_model_is_missing_and_says_so() -> None:
    with serving(503, "model not ready") as server:
        found = readiness.transcription(server.url)
    assert isinstance(found, Missing) and "answered 503: " in found.said and "model not ready" in found.said and "menu says the model is ready" in found.said


def test_a_server_that_refuses_a_hold_is_missing_and_says_its_answer() -> None:
    with serving(404, "no such route") as server:
        found = readiness.transcription(server.url)
    assert isinstance(found, Missing) and "answered 404" in found.said and "no such route" in found.said


def test_a_server_failing_on_its_own_side_is_unknown_not_missing() -> None:
    # A 500 is a hold LowTalker could not decode, not an address that is no transcription server's.
    with serving(500, "decode failed") as server:
        found = readiness.transcription(server.url)
    assert isinstance(found, Unknown) and "answered 500" in found.said and "config.toml" not in found.said


def test_a_server_busy_with_four_holds_is_unknown_not_missing() -> None:
    # LowTalker refuses a fifth upload at once; the next may well be transcribed.
    with serving(429, "busy") as server:
        found = readiness.transcription(server.url)
    assert isinstance(found, Unknown) and "answered 429" in found.said and "busy" in found.said


def test_a_server_answering_with_no_segments_is_missing_as_every_hold_would_fail() -> None:
    with serving(200, '{"text": ""}') as server:
        found = readiness.transcription(server.url)
    assert isinstance(found, Missing) and "not verbose_json" in found.said


def test_a_server_that_does_not_answer_in_time_is_unknown_not_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Accepts the connection and never answers.
    listening = socket.create_server(("127.0.0.1", 0))
    monkeypatch.setattr(transcribing, "PROBE_SECONDS", 0.2)
    with listening:
        found = readiness.transcription(f"http://127.0.0.1:{listening.getsockname()[1]}/v1")
    assert isinstance(found, Unknown) and "did not answer within 0.2 s" in found.said


# hands running


def beating(home: Home) -> None:
    home.root.mkdir(parents=True, exist_ok=True)
    heart = heartbeat.Heart(home.status, os.getpid(), datetime.now(UTC), heartbeat.HEARTBEAT)
    heart.beat("running", None, 0, listening=True, degraded=())


def test_hands_up_is_ready(root: Path) -> None:
    home = Home(root / "home")
    beating(home)
    found = readiness.daemon(home, datetime.now(UTC))
    assert isinstance(found, Ready) and found.said.startswith("hands is up")


def test_hands_that_never_ran_is_missing_and_says_to_run_it(root: Path) -> None:
    found = readiness.daemon(Home(root / "home"), datetime.now(UTC))
    assert isinstance(found, Missing) and "hands has not run" in found.said and "`hands run`" in found.said


def test_a_heartbeat_hands_cannot_read_is_unknown(root: Path) -> None:
    home = Home(root / "home")
    home.root.mkdir(parents=True)
    home.status.write_text("not json")
    assert isinstance(readiness.daemon(home, datetime.now(UTC)), Unknown)


# hands check


STEPS = ["claude", "portaudio", "hands", "shim", "plugin", "first-run", "backend", "transcription", "grant", "running", "sessions"]


def logged_in(_llm: object, home: Home, _environment: object) -> ClaudeCodeBackend:
    """The brain as its login check finds it logged in: that check is the start's, tested beside it."""
    return ClaudeCodeBackend(model="claude-sonnet-5-5", config_dir=home.brain, account=Account("claude.ai", "brain@example.com"))


@pytest.fixture
def lowtalker() -> Iterator[Answering]:
    with serving() as server:
        yield server


def set_up(root: Path, fritter: Path, monkeypatch: pytest.MonkeyPatch, plugins: object, transcriber: Answering) -> Home:
    """A home on which every step of the install is done, with `plugins` listed by its claude, and LowTalker `transcriber`."""
    home = Home(root / "home")
    home.bin.mkdir(parents=True)
    shutil.copy2(fritter, home.bin / "fritter")
    executable(home.shim, shim_script(home.bin / "fritter", home.wire))
    transcribed(home, transcriber.url)
    beating(home)
    monkeypatch.setattr(readiness, "resolve", logged_in)
    hands_printing(root, f"hands {version('hands')}")
    monkeypatch.setenv("PATH", f"{home.bin}:{root / 'tools'}:{claude_listing(root, plugins)}")
    monkeypatch.setattr("hands.voice.talkkey.granted", lambda: True)
    answered(root, home, monkeypatch)

    # The fake claude is a script, whose sessions cannot be told; take it for a native one that runs nowhere.
    def native(_claude: Path | None) -> Path:
        return root / "versions" / "9.9.9"

    monkeypatch.setattr(readiness, "claude_code", native)
    return home


def answered(root: Path, home: Home, monkeypatch: pytest.MonkeyPatch, trusted: bool = True) -> None:
    """The person's Claude Code, in a config of the test's own, through its onboarding, the smoke folder trusted unless not `trusted`."""
    config = root / "config"
    config.mkdir(exist_ok=True)
    (config / ".claude.json").write_text(json.dumps({"hasCompletedOnboarding": True, "projects": {str(home.smoke.resolve()): {"hasTrustDialogAccepted": trusted}}}))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def marks(capsys: pytest.CaptureFixture[str]) -> dict[str, str]:
    lines = [line for line in capsys.readouterr().out.splitlines() if not line.startswith(" ")]
    assert len(lines) == len(STEPS)
    return {step: line.split()[0] for step, line in zip(STEPS, lines)}


def test_every_step_done_is_ok_and_exits_0(root: Path, fritter: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], lowtalker: Answering) -> None:
    home = set_up(root, fritter, monkeypatch, [listed(PLUGIN_ID, True)], lowtalker)
    assert main(["--home", str(home.root), "check"]) == 0
    assert marks(capsys) == dict.fromkeys(STEPS, "ok")


# Each undoes one step of a home set_up made.
type Undo = Callable[[Path, Home, pytest.MonkeyPatch], None]


def no_hands(root: Path, _home: Home, _monkeypatch: pytest.MonkeyPatch) -> None:
    (root / "tools" / "hands").unlink()


def unshimmed(_root: Path, home: Home, _monkeypatch: pytest.MonkeyPatch) -> None:
    home.shim.unlink()


def no_plugin(root: Path, _home: Home, _monkeypatch: pytest.MonkeyPatch) -> None:
    claude_listing(root, [])


def logged_out(_root: Path, _home: Home, monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(*_: object) -> object:
        raise Rejected("the brain has no login: `hands login` gives it one")

    monkeypatch.setattr(readiness, "resolve", refused)


def untrusted(root: Path, home: Home, monkeypatch: pytest.MonkeyPatch) -> None:
    answered(root, home, monkeypatch, trusted=False)


def person_logged_out(root: Path, _home: Home, _monkeypatch: pytest.MonkeyPatch) -> None:
    (root / "real" / "logged-out").touch()


def ungranted(_root: Path, home: Home, monkeypatch: pytest.MonkeyPatch) -> None:
    # A running hands has the grant of the app it runs in, so the grant is missing only where hands is not running.
    monkeypatch.setattr("hands.voice.talkkey.granted", lambda: False)
    home.status.unlink()


def not_running(_root: Path, home: Home, _monkeypatch: pytest.MonkeyPatch) -> None:
    home.status.unlink()


def not_transcribing(_root: Path, home: Home, _monkeypatch: pytest.MonkeyPatch) -> None:
    # Nothing listens on the discard port.
    transcribed(home, "http://127.0.0.1:9/v1")


@pytest.mark.parametrize(
    ("undo", "steps"),
    [
        (no_hands, {"hands"}),
        (unshimmed, {"shim"}),
        (no_plugin, {"plugin"}),
        (untrusted, {"first-run"}),
        (person_logged_out, {"first-run"}),
        (logged_out, {"backend"}),
        (not_transcribing, {"transcription"}),
        (ungranted, {"grant", "running"}),
        (not_running, {"running"}),
    ],
    ids=["hands", "shim", "plugin", "untrusted", "person-logged-out", "backend", "transcription", "grant", "running"],
)
def test_a_home_missing_steps_names_those_steps_and_exits_1(
    root: Path, fritter: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], lowtalker: Answering, undo: Undo, steps: set[str]
) -> None:
    home = set_up(root, fritter, monkeypatch, [listed(PLUGIN_ID, True)], lowtalker)
    undo(root, home, monkeypatch)
    assert main(["--home", str(home.root), "check"]) == 1
    assert marks(capsys) == {**dict.fromkeys(STEPS, "ok"), **dict.fromkeys(steps, "missing")}
    # The command's event carries every step's finding, in the same order.
    [event] = [line for line in map(json.loads, audit.tail(home.audit, 100)[0]) if line.get("event") == "hands.command"]
    assert [finding["type"] for finding in event["facts"]["findings"]] == ["Missing" if each in steps else "Ready" for each in STEPS]


def test_a_step_that_cannot_be_looked_at_exits_2_and_one_missing_outranks_it(
    root: Path, fritter: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], lowtalker: Answering
) -> None:
    home = set_up(root, fritter, monkeypatch, "unreadable", lowtalker)
    assert main(["--home", str(home.root), "check"]) == 2
    assert marks(capsys) == {**dict.fromkeys(STEPS, "ok"), "plugin": "unknown"}
    ungranted(root, home, monkeypatch)
    assert main(["--home", str(home.root), "check"]) == 1
    assert marks(capsys) == {**dict.fromkeys(STEPS, "ok"), "plugin": "unknown", "grant": "missing", "running": "missing"}
