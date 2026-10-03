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
from collections.abc import Generator, Iterator
from pathlib import Path

import pytest

from hands.core.session import Membership, SessionId
from hands.daemon import readiness
from hands.daemon.cli import main
from hands.daemon.readiness import Missing, Ready, Unknown
from hands.sessions.hookconfig import PLUGIN_ID
from hands.sessions.home import Home
from hands.sessions.membership import write_membership
from hands.sessions.processes import Terminal
from hands.sessions.wrapper import shim_script


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


def test_the_shim_first_on_path_is_ready(root: Path) -> None:
    home = Home(root / "home")
    executable(home.bin / "fritter", "#!/bin/sh\n")
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
def at_a_terminal(executable: Path, cwd: Path) -> Generator[int]:
    """A process running executable in cwd with a terminal of its own, as a session runs; its pid, once it runs executable."""
    controller, terminal = pty.openpty()
    # Popen returns only once the child has exec'd, so the process is executable from the first look at it.
    process = subprocess.Popen(
        [executable, "30"],
        cwd=cwd,
        stdin=terminal,
        stdout=terminal,
        stderr=terminal,
        start_new_session=True,
        preexec_fn=lambda: fcntl.ioctl(0, termios.TIOCSCTTY, 0),
    )
    try:
        yield process.pid
    finally:
        process.kill()
        process.wait()
        os.close(controller)
        os.close(terminal)


def test_no_running_session_is_said_as_none_not_left_out(root: Path) -> None:
    assert readiness.sessions(Home(root / "home"), installed(root)) == Ready("running sessions hands knows of: 0, and each can be typed into")


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
    assert found.said.startswith("running sessions hands knows of: 3, and hands cannot reach these:")
    assert f"/code/unwrapped (pid {sleepers[1].pid}) was started outside fritter" in found.said
    assert f"/code/orphaned (pid {sleepers[2].pid}) has lost its fritter, whose socket {root / 'gone.sock'} is gone" in found.said
    assert "/code/wrapped" not in found.said


def test_a_session_whose_process_has_ended_is_not_running(root: Path) -> None:
    home = Home(root / "home")
    ended = subprocess.Popen(["true"])
    ended.wait()
    joined(home, "ended", ended.pid, None)
    assert readiness.sessions(home, installed(root)) == Ready("running sessions hands knows of: 0, and each can be typed into")


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
    assert isinstance(found, Missing)
    assert f"{project} (pid {pid}) has run no hook since the plugin was installed, so hands does not know of it: /reload-plugins in it" in found.said


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
    assert found == Ready("running sessions hands knows of: 1, and each can be typed into")


def terminal(pid: int, executable: str, cwd: str) -> Terminal:
    return Terminal(pid, Path(executable), Path(cwd))


def test_a_session_is_a_terminal_process_of_any_version_of_the_real_claudes_install(root: Path) -> None:
    home = Home(root / "home")
    claude = root / "bin" / "claude"
    claude.parent.mkdir()
    claude.symlink_to(root / "versions" / "2.1.288")
    terminals = [
        terminal(1, f"{root}/versions/2.1.286", "/code/old"),
        terminal(2, f"{root}/versions/2.1.288", "/code/joined"),
        terminal(3, "/bin/zsh", "/code/shell"),
        terminal(4, "/elsewhere/versions/2.1.288", "/code/other-install"),
        terminal(5, f"{root}/versions/2.1.288", f"{home.brain}/cwd"),
    ]
    assert readiness.unjoined(home, claude, terminals, {2}) == [terminals[0]]


def test_with_no_real_claude_a_session_that_never_ran_a_hook_cannot_be_found() -> None:
    assert readiness.unjoined(Home(Path("/nowhere")), None, [terminal(1, "/x/claude", "/code")], set()) is None
    found = readiness.sessions_found([], set(), [], None)
    assert isinstance(found, Missing) and "no `claude` on this PATH but hands' shims" in found.said


def test_sessions_that_cannot_be_looked_at_are_unknown(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(_pids: object) -> dict[int, float]:
        raise OSError(1, "sysctl refused")

    monkeypatch.setattr(readiness, "process_starts", refused)
    found = readiness.sessions(Home(root / "home"), installed(root))
    assert isinstance(found, Unknown) and "sysctl refused" in found.said


# hands check


@pytest.mark.parametrize(
    ("granted", "plugins", "code", "marks"),
    [
        (True, [listed(PLUGIN_ID, True)], 0, ["ok", "ok", "ok", "ok"]),
        (False, [listed(PLUGIN_ID, True)], 1, ["ok", "ok", "missing", "ok"]),
        (True, "unreadable", 2, ["unknown", "ok", "ok", "ok"]),
        (False, "unreadable", 1, ["unknown", "ok", "missing", "ok"]),
    ],
    ids=["ready", "missing", "unknown", "missing-outranks-unknown"],
)
def test_check_says_every_piece_and_exits_by_the_worst(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], granted: bool, plugins: object, code: int, marks: list[str]
) -> None:
    home = Home(root / "home")
    executable(home.bin / "fritter", "#!/bin/sh\n")
    executable(home.shim, shim_script(home.bin / "fritter", home.wire))
    monkeypatch.setenv("PATH", f"{home.bin}:{claude_listing(root, plugins)}")
    monkeypatch.setattr("hands.voice.talkkey.granted", lambda: granted)
    assert main(["--home", str(home.root), "check"]) == code
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in lines] == marks


