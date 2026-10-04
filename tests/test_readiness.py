"""`hands check`: each piece hands needs, found or named as missing, through the same edges a user's machine has."""

import contextlib
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
from collections.abc import Callable, Generator, Iterator, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pytest

from hands.core.session import Membership, SessionId
from hands.daemon import readiness
from hands.daemon.cli import main
from hands.daemon.readiness import Missing, Ready, Unknown
from hands.sessions import audit, heartbeat, liveness, wrapper
from hands.sessions.hookconfig import PLUGIN_ID
from hands.sessions.home import Home
from hands.sessions.membership import write_membership
from hands.sessions.payload import Rejected
from hands.sessions.terminals import Terminal, attended, terminal_processes
from hands.sessions.wrapper import shim_script
from hands.voice.backends import ClaudeCodeBackend


@pytest.fixture
def root() -> Iterator[Path]:
    # Short on purpose: a test session's fritter socket lives under here, and macOS caps a socket path near 104 bytes.
    root = Path(tempfile.mkdtemp(prefix="ready-", dir="/tmp")).resolve()
    yield root
    shutil.rmtree(root)


def executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def claude_listing(root: Path, plugins: object) -> str:
    """A PATH whose `claude plugin list --json` prints plugins."""
    executable(root / "real" / "claude", f"#!/bin/sh\ncat <<'EOF'\n{json.dumps(plugins)}\nEOF\n")
    return f"{root / 'real'}:/usr/bin:/bin"


def listed(id: str, enabled: bool, scope: str = "user") -> dict[str, object]:
    return {"id": id, "version": "f24447053e9c", "scope": scope, "enabled": enabled}


# The plugin


def test_an_enabled_plugin_is_ready(root: Path) -> None:
    found = readiness.plugin(claude_listing(root, [listed("other@elsewhere", False), listed(PLUGIN_ID, True)]))
    assert isinstance(found, Ready) and "/reload-plugins" in found.said


def test_a_plugin_enabled_at_any_scope_is_ready() -> None:
    assert isinstance(readiness.plugin_listed(json.dumps([listed(PLUGIN_ID, False), listed(PLUGIN_ID, True)])), Ready)


def test_a_disabled_plugin_is_missing_and_says_how_to_enable_it() -> None:
    found = readiness.plugin_listed(json.dumps([listed(PLUGIN_ID, False)]))
    assert isinstance(found, Missing) and f"claude plugin enable --scope user {PLUGIN_ID}" in found.said


def test_a_plugin_not_installed_is_missing_and_says_how_to_install_it(root: Path) -> None:
    found = readiness.plugin(claude_listing(root, [listed("other@elsewhere", True)]))
    assert isinstance(found, Missing) and f"claude plugin install {PLUGIN_ID}" in found.said


def test_an_install_for_one_project_is_not_one_for_every_session() -> None:
    project = {**listed(PLUGIN_ID, True, "local"), "projectPath": "/code/cc-hands"}
    found = readiness.plugin_listed(json.dumps([project]))
    assert isinstance(found, Missing) and f"`claude plugin install {PLUGIN_ID}`" in found.said


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


def test_a_shim_an_older_hands_wrote_says_to_install_it_again(root: Path, fritter: Path) -> None:
    # As after hands is upgraded: the shim is not the script this hands writes, so it may not run what `run` says it does.
    home = Home(root / "home")
    home.bin.mkdir(parents=True)
    shutil.copy2(fritter, home.bin / "fritter")
    executable(home.shim, shim_script(home.bin / "fritter", home.wire).replace("session=no ;;", ";;"))
    found = readiness.shim(home, f"{home.bin}")
    assert found == Missing(f"`claude` on this PATH is hands' shim, {home.shim}, but not the one this hands writes: run `hands install-fritter`")


def test_no_shim_installed_says_to_install_it(root: Path) -> None:
    found = readiness.shim(Home(root / "home"), f"{root / 'empty'}")
    assert isinstance(found, Missing)
    assert "`claude` on this PATH is nothing" in found.said and "hands install-fritter" in found.said


# The grant


def test_the_grant_is_ready_when_given_and_missing_with_where_to_give_it_when_not() -> None:
    assert isinstance(readiness.grant(True), Ready)
    missing = readiness.grant(False)
    assert isinstance(missing, Missing) and "Privacy & Security > Input Monitoring" in missing.said


# The running sessions


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
    assert readiness.sessions(Home(root / "home"), installed(root)) == Ready("running sessions hands knows of: 0, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0, and each can be typed into")


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
        found = readiness.sessions(home, installed(root))
    finally:
        listening.close()
        for sleeper in sleepers:
            sleeper.kill()
            sleeper.wait()
    assert isinstance(found, Missing)
    assert found.said.startswith("running sessions hands knows of: 3, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0, and hands cannot reach these:")
    assert f"/code/unwrapped (pid {sleepers[1].pid}) was started outside fritter" in found.said
    assert f"/code/orphaned (pid {sleepers[2].pid}) has lost its fritter, whose socket {root / 'gone.sock'} is gone" in found.said
    assert "/code/wrapped" not in found.said


def test_a_session_whose_process_has_ended_is_not_running(root: Path) -> None:
    home = Home(root / "home")
    ended = subprocess.Popen(["true"])
    ended.wait()
    joined(home, "ended", ended.pid, None)
    assert readiness.sessions(home, installed(root)) == Ready("running sessions hands knows of: 0, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0, and each can be typed into")


def test_an_unreadable_membership_file_is_named_and_left_where_it_is(root: Path) -> None:
    home = Home(root / "home")
    home.memberships.mkdir(parents=True)
    bad = home.membership(SessionId("bad"))
    bad.write_text("{not json")
    found = readiness.sessions(home, installed(root))
    assert isinstance(found, Missing) and f"{bad} names no session hands can read" in found.said
    assert bad.exists()


def test_a_session_that_never_ran_a_hook_is_named_by_cwd_and_pid_with_the_fix(root: Path) -> None:
    home = Home(root / "home")
    path = installed(root)
    project = root / "project"
    project.mkdir()
    with at_a_terminal(root / "install" / "9.9.9", project) as pid:
        found = readiness.sessions(home, path)
    assert isinstance(found, Missing) and f"{project} (pid {pid}) is a session hands has no record of" in found.said


def test_a_claude_printing_or_piped_into_at_a_terminal_is_no_session(root: Path) -> None:
    home = Home(root / "home")
    path = installed(root)
    shell = root / "install" / "9.9.7"
    shutil.copy("/bin/zsh", shell)
    # The shell waits on its sleep, so it is still the process its -p was given to.
    with at_a_terminal(shell, root, ["-c", "sleep 30; :", "-p", "hello"]), at_a_terminal(root / "install" / "9.9.9", root, piped=True):
        found = readiness.sessions(home, path)
    assert found == Ready("running sessions hands knows of: 0, runs of claude at a terminal that are none: piped 1, subcommand 0, print 1, and each can be typed into")


def test_a_check_run_from_a_removed_directory_says_its_own_config_cannot_be_told(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gone = root / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    found = readiness.unrecorded(Home(root / "home"), installed(root), set())
    assert isinstance(found, readiness.Unfindable) and found.said.startswith("cannot tell which Claude Code config this check runs under")


def test_the_kernel_says_what_a_process_at_a_terminal_was_started_with_and_whether_it_reads_and_writes_it(root: Path) -> None:
    with at_a_terminal(Path("/bin/sleep"), root) as reading, at_a_terminal(Path("/bin/sleep"), root, ["31"], piped=True) as piped:
        found = {process.pid: process for process in terminal_processes()}
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
        found = next(found for found in terminal_processes() if found.pid == process.pid)
        assert attended(found)
    finally:
        process.kill()
        process.wait()
        os.close(controller)
        os.close(terminal)


def test_a_reused_pid_of_an_ended_session_hides_no_session_hands_has_no_record_of(root: Path) -> None:
    home = Home(root / "home")
    path = installed(root)
    with at_a_terminal(root / "install" / "9.9.9", root) as pid:
        # Written before the process under its pid started, as an ended session's file left behind is.
        os.utime(home.membership(joined(home, "ended", pid, None).id), (0, 0))
        found = readiness.sessions(home, path)
    assert isinstance(found, Missing) and f"(pid {pid}) is a session hands has no record of" in found.said


def test_a_session_on_a_version_pruned_since_it_started_is_still_found(root: Path) -> None:
    home = Home(root / "home")
    path = installed(root)
    pruned = root / "install" / "9.9.8"
    shutil.copy("/bin/sleep", pruned)
    with at_a_terminal(pruned, root) as pid:
        pruned.unlink()
        found = readiness.sessions(home, path)
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
            [process] = [process for process in terminal_processes() if process.pid == pid]
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
            found = readiness.sessions(home, path)
    finally:
        listening.close()
    assert found == Ready("running sessions hands knows of: 1, runs of claude at a terminal that are none: piped 0, subcommand 0, print 0, and each can be typed into")


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
    return [session.process for session in readiness.unjoined(home, Path(claude), CHECKED, terminals, members or set(), reads_its_terminal).sessions]


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
    found = readiness.unjoined(Home(Path("/h")), Path("/v/2.1.288"), config, [session], set(), reads_its_terminal)
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
    found = readiness.unjoined(Home(Path("/h")), Path("/v/2.1.288"), CHECKED, [printing, piped, prompted, controlling, asked], set(), reads_its_terminal)
    assert [session.process for session in found.sessions] == [prompted, asked]
    assert found.others == {"print": 1, "piped": 1, "subcommand": 1}


def test_the_helper_of_a_claude_that_is_no_session_is_no_session_either() -> None:
    printing, helper = terminal(10, "/v/2.1.288", arguments=("-p", "hello")), terminal(11, "/v/2.1.288", parent=10)
    assert unjoined("/v/2.1.288", [printing, helper]) == []


def test_a_session_started_outside_fritter_is_told_to_restart_and_one_inside_to_reload(root: Path) -> None:
    home = Home(root / "home")
    fritter = terminal(5, str(home.fritter))
    inside, outside = terminal(10, "/v/2.1.288", "/code/in", parent=5), terminal(11, "/v/2.1.288", "/code/out", parent=6)
    found = readiness.sessions_found([], set(), [], readiness.unjoined(home, Path("/v/2.1.288"), CHECKED, [fritter, inside, outside], set(), reads_its_terminal))
    assert isinstance(found, Missing)
    assert "/code/in (pid 10) is a session hands has no record of, so it cannot be reached: /reload-plugins in it" in found.said
    assert "/code/out (pid 11) is a session hands has no record of, started outside fritter, so it cannot be typed into: restart it" in found.said


def test_with_no_real_claude_a_session_hands_has_no_record_of_cannot_be_found(root: Path) -> None:
    found = readiness.sessions(Home(root / "home"), "/nowhere")
    assert isinstance(found, Unknown) and "this PATH has no `claude` of its own" in found.said


def test_a_claude_that_is_a_script_cannot_tell_its_sessions(root: Path) -> None:
    (root / "bin").mkdir()
    (root / "bin" / "claude").write_text("#!/usr/bin/env node\n")
    (root / "bin" / "claude").chmod(0o755)
    found = readiness.sessions(Home(root / "home"), str(root / "bin"))
    assert isinstance(found, Unknown) and "is a script, so its sessions run as its interpreter" in found.said


def test_sessions_that_cannot_be_looked_for_are_said_beside_the_ones_that_cannot_be_reached() -> None:
    unreadable = liveness.Unreadable(Path("/h/sessions/bad.json"), b"{", Rejected("not json"))
    found = readiness.sessions_found([], set(), [unreadable], readiness.Unfindable("why"))
    assert isinstance(found, Missing) and "bad.json names no session" in found.said and "cannot be found: why" in found.said


def test_sessions_that_cannot_be_looked_at_are_unknown(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(_pids: object) -> dict[int, float]:
        raise OSError(1, "sysctl refused")

    monkeypatch.setattr(readiness, "process_starts", refused)
    found = readiness.sessions(Home(root / "home"), installed(root))
    assert isinstance(found, Unknown) and "sysctl refused" in found.said


# Claude Code


def test_a_native_claude_on_path_is_ready(root: Path) -> None:
    found = readiness.claude(installed(root))
    assert found == Ready(f"Claude Code is installed: {root / 'install' / '9.9.9'}")


def test_no_claude_on_path_is_missing_and_names_its_installer(root: Path) -> None:
    found = readiness.claude(f"{root / 'empty'}:/usr/bin:/bin")
    assert isinstance(found, Missing) and "no `claude` of its own" in found.said and "https://claude.ai/install.sh" in found.said


def test_a_claude_that_is_a_script_is_missing_and_names_the_native_one(root: Path) -> None:
    executable(root / "real" / "claude", "#!/bin/sh\n")
    found = readiness.claude(f"{root / 'real'}:/usr/bin:/bin")
    assert isinstance(found, Missing) and "is a script" in found.said and "https://claude.ai/install.sh" in found.said


# PortAudio


def test_portaudio_that_pyaudio_loads_is_ready() -> None:
    found = readiness.portaudio()
    assert isinstance(found, Ready) and "PortAudio" in found.said


def test_a_pyaudio_that_cannot_load_is_missing_and_names_homebrews_portaudio(monkeypatch: pytest.MonkeyPatch) -> None:
    # As when Homebrew's portaudio is gone: importing PyAudio's extension fails.
    monkeypatch.setitem(sys.modules, "pyaudio", None)
    found = readiness.portaudio()
    assert isinstance(found, Missing) and "`brew install portaudio`" in found.said


# hands on PATH


def hands_printing(root: Path, said: str, code: int = 0) -> str:
    """A PATH whose `hands --version` prints said and exits code."""
    executable(root / "tools" / "hands", f"#!/bin/sh\necho '{said}'\nexit {code}\n")
    return f"{root / 'tools'}:/usr/bin:/bin"


def test_this_hands_on_path_is_ready(root: Path) -> None:
    found = readiness.installed(hands_printing(root, f"hands {version('hands')}"))
    assert isinstance(found, Ready) and str(root / "tools" / "hands") in found.said


def test_no_hands_on_path_is_missing_and_says_how_uv_puts_it_there(root: Path) -> None:
    found = readiness.installed(f"{root / 'empty'}:/usr/bin:/bin")
    assert isinstance(found, Missing) and "`hands plugin`" in found.said and "`uv tool update-shell`" in found.said


def test_another_hands_on_path_is_missing_naming_both(root: Path) -> None:
    # As after an upgrade that left an older install ahead on PATH: sessions would run its hooks.
    found = readiness.installed(hands_printing(root, "hands 0.0.1"))
    assert isinstance(found, Missing) and "is hands 0.0.1, not this hands" in found.said


def test_a_hands_that_cannot_say_its_version_is_unknown(root: Path) -> None:
    found = readiness.installed(hands_printing(root, "broken", 3))
    assert isinstance(found, Unknown) and "(3)" in found.said


# The backend


def keyed(home: Home) -> None:
    home.root.mkdir(parents=True, exist_ok=True)
    home.config.write_text('[llm]\nbackend = "openai"\n')


def test_a_backend_with_its_key_is_ready_and_never_says_the_key(root: Path) -> None:
    home = Home(root / "home")
    keyed(home)
    found = readiness.configured(home, {"OPENAI_API_KEY": "sk-secret"})
    assert isinstance(found, Ready) and "with its key" in found.said and "sk-secret" not in found.said


def test_a_backend_without_its_key_is_missing_and_names_it(root: Path) -> None:
    home = Home(root / "home")
    keyed(home)
    found = readiness.configured(home, {})
    assert isinstance(found, Missing) and "OPENAI_API_KEY is not set" in found.said


def test_settings_hands_cannot_read_are_missing_naming_the_file(root: Path) -> None:
    home = Home(root / "home")
    home.root.mkdir(parents=True)
    home.config.write_text("[llm\n")
    reached = readiness.configured(home, {})
    assert isinstance(reached, Missing) and str(home.config) in reached.said


def test_a_setting_in_the_environment_is_missing_as_the_start_refuses_it(root: Path) -> None:
    home = Home(root / "home")
    keyed(home)
    found = readiness.configured(home, {"OPENAI_API_KEY": "k", "HANDS_DEBUG": "1"})
    assert isinstance(found, Missing) and "HANDS_DEBUG set, and hands reads no setting from the environment" in found.said


def test_a_backend_that_cannot_be_asked_is_unknown(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(root / "home")
    keyed(home)

    def unspawnable(*_: object) -> object:
        raise PermissionError("claude is not executable")

    monkeypatch.setattr(readiness, "resolve", unspawnable)
    found = readiness.configured(home, {})
    assert isinstance(found, Unknown) and "claude is not executable" in found.said


def test_the_brain_says_its_account_and_never_a_key() -> None:
    found = readiness.reaching(ClaudeCodeBackend(model="claude-sonnet-5", config_dir=Path("/h/brain"), account="brain@example.com"))
    assert found == Ready("the brain is logged in as brain@example.com, and reaches claude-sonnet-5")


# hands running


def beating(home: Home) -> None:
    home.root.mkdir(parents=True, exist_ok=True)
    heart = heartbeat.Heart(home.status, os.getpid(), datetime.now(UTC), heartbeat.HEARTBEAT)
    heart.beat("running", None, 0, listening=True, deaf=False)


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


STEPS = ["claude", "portaudio", "hands", "shim", "plugin", "backend", "grant", "running", "sessions"]


def set_up(root: Path, fritter: Path, monkeypatch: pytest.MonkeyPatch, plugins: object) -> Home:
    """A home on which every step of the README is done, with `plugins` listed by its claude."""
    home = Home(root / "home")
    home.bin.mkdir(parents=True)
    shutil.copy2(fritter, home.bin / "fritter")
    executable(home.shim, shim_script(home.bin / "fritter", home.wire))
    keyed(home)
    beating(home)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    hands_printing(root, f"hands {version('hands')}")
    monkeypatch.setenv("PATH", f"{home.bin}:{root / 'tools'}:{claude_listing(root, plugins)}")
    monkeypatch.setattr("hands.voice.talkkey.granted", lambda: True)

    # The fake claude is a script, whose sessions cannot be told; take it for a native one that runs nowhere.
    def native(_claude: Path | None) -> Path:
        return root / "versions" / "9.9.9"

    monkeypatch.setattr(readiness, "claude_code", native)
    return home


def marks(capsys: pytest.CaptureFixture[str]) -> dict[str, str]:
    lines = [line for line in capsys.readouterr().out.splitlines() if not line.startswith(" ")]
    assert len(lines) == len(STEPS)
    return {step: line.split()[0] for step, line in zip(STEPS, lines)}


def test_every_step_done_is_ok_and_exits_0(root: Path, fritter: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    home = set_up(root, fritter, monkeypatch, [listed(PLUGIN_ID, True)])
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


def no_key(_root: Path, _home: Home, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")


def ungranted(_root: Path, _home: Home, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hands.voice.talkkey.granted", lambda: False)


def not_running(_root: Path, home: Home, _monkeypatch: pytest.MonkeyPatch) -> None:
    home.status.unlink()


@pytest.mark.parametrize(
    ("undo", "step"),
    [(no_hands, "hands"), (unshimmed, "shim"), (no_plugin, "plugin"), (no_key, "backend"), (ungranted, "grant"), (not_running, "running")],
    ids=["hands", "shim", "plugin", "backend", "grant", "running"],
)
def test_a_home_missing_one_step_names_that_step_and_exits_1(
    root: Path, fritter: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], undo: Undo, step: str
) -> None:
    home = set_up(root, fritter, monkeypatch, [listed(PLUGIN_ID, True)])
    undo(root, home, monkeypatch)
    assert main(["--home", str(home.root), "check"]) == 1
    assert marks(capsys) == {**dict.fromkeys(STEPS, "ok"), step: "missing"}
    # The command's event carries every step's finding, in the same order.
    [event] = [line for line in map(json.loads, audit.tail(home.audit, 100)[0]) if line.get("event") == "hands.command"]
    assert [finding["type"] for finding in event["facts"]["findings"]] == ["Missing" if each == step else "Ready" for each in STEPS]


def test_a_step_that_cannot_be_looked_at_exits_2_and_one_missing_outranks_it(
    root: Path, fritter: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = set_up(root, fritter, monkeypatch, "unreadable")
    assert main(["--home", str(home.root), "check"]) == 2
    assert marks(capsys) == {**dict.fromkeys(STEPS, "ok"), "plugin": "unknown"}
    monkeypatch.setattr("hands.voice.talkkey.granted", lambda: False)
    assert main(["--home", str(home.root), "check"]) == 1
    assert marks(capsys) == {**dict.fromkeys(STEPS, "ok"), "plugin": "unknown", "grant": "missing"}
