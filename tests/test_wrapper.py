"""The claude shim: a session on a terminal runs under fritter, anything else runs the real claude as it is."""

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from hands.daemon.cli import main
from hands.sessions.audit import segment
from hands.sessions.home import Home
from hands.sessions import wrapper
from hands.sessions.wrapper import shim_script

# Each stand-in says what it was run with, and whether it carries a fritter address.
RECORDER = '#!/bin/sh\nprintf "%s %s socket=%s\\n" "$(basename "$0")" "$*" "${FRITTER_SOCKET-unset}"\n'
# Where the shims under test send their sessions' wire; nothing listens there.
WIRE = Path("/tmp/hands-test-wire.sock")
# What fritter is asked for before the claude it runs, when the session was given no API of its own.
TAPPED = f"--tap https://api.anthropic.com --tap-ca NODE_EXTRA_CA_CERTS --tap-to {WIRE}"


@pytest.fixture
def root() -> Iterator[Path]:
    # Short on purpose: a real fritter puts its socket under here, and macOS caps a socket path near 104 bytes.
    root = Path(tempfile.mkdtemp(prefix="wrap-", dir="/tmp")).resolve()
    yield root
    shutil.rmtree(root)


def executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def installed_shim(root: Path) -> Path:
    """A shim in root/bin, beside a recording fritter; the real claude, a recorder too, is in root/real."""
    bin = root / "bin"
    executable(bin / "fritter", RECORDER)
    executable(root / "real" / "claude", RECORDER)
    return executable(bin / "claude", shim_script(bin / "fritter", WIRE))


def on_a_pipe(argv: Sequence[str], path: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, env={"PATH": path, **(env or {})}, stdin=subprocess.DEVNULL, capture_output=True, text=True)


def on_a_terminal(argv: Sequence[str], path: str, env: dict[str, str] | None = None) -> str:
    """What argv prints when a terminal is its stdin, stdout, and stderr."""
    controller, terminal = os.openpty()
    process = subprocess.Popen(argv, env={"PATH": path, "TMPDIR": "/tmp", **(env or {})}, stdin=terminal, stdout=terminal, stderr=terminal, start_new_session=True)
    os.close(terminal)
    printed = b""
    while True:
        try:
            chunk = os.read(controller, 4096)
        except OSError:  # EIO: the last process holding the terminal let go of it
            break
        if not chunk:
            break
        printed += chunk
    process.wait(timeout=10)
    os.close(controller)
    return printed.decode().replace("\r\n", "\n")


def test_a_session_on_a_terminal_runs_under_fritter_around_the_real_claude(root: Path) -> None:
    shim = installed_shim(root)
    printed = on_a_terminal([str(shim), "--resume", "abc"], f"{root / 'bin'}:{root / 'real'}:/usr/bin:/bin")
    assert printed == f"fritter {TAPPED} -- {root / 'real' / 'claude'} --resume abc socket=unset\n"


@pytest.mark.parametrize("args", [["-p", "hello"], ["--print", "hello"], ["--model", "opus", "-p", "hello"], ["-cp", "hello"], ["-pc", "hello"]])
def test_print_on_a_terminal_runs_the_real_claude(root: Path, args: list[str]) -> None:
    shim = installed_shim(root)
    printed = on_a_terminal([str(shim), *args], f"{root / 'bin'}:{root / 'real'}:/usr/bin:/bin")
    assert printed == f"claude {' '.join(args)} socket=unset\n"


def test_a_prompt_after_the_options_that_says_p_is_still_a_session(root: Path) -> None:
    shim = installed_shim(root)
    printed = on_a_terminal([str(shim), "--", "-p"], f"{root / 'bin'}:{root / 'real'}:/usr/bin:/bin")
    assert printed.startswith(f"fritter {TAPPED} -- ")


def test_off_a_terminal_the_real_claude_runs_without_the_address_of_the_session_it_was_started_from(root: Path) -> None:
    shim = installed_shim(root)
    ran = on_a_pipe([str(shim), "mcp", "serve"], f"{root / 'bin'}:{root / 'real'}:/usr/bin:/bin", {"FRITTER_SOCKET": "/tmp/fritter-parent/sock"})
    assert (ran.returncode, ran.stdout, ran.stderr) == (0, "claude mcp serve socket=unset\n", "")


@pytest.mark.parametrize("args", [["-dp"], ["-rp"]])
def test_an_option_whose_value_is_p_is_a_session(root: Path, args: list[str]) -> None:
    # -d and -r take a value, so -dp is a debug filter and -rp a session to resume.
    shim = installed_shim(root)
    printed = on_a_terminal([str(shim), *args], f"{root / 'bin'}:{root / 'real'}:/usr/bin:/bin")
    assert printed.startswith(f"fritter {TAPPED} -- ")


def test_every_hands_shim_is_skipped_however_path_names_it(root: Path) -> None:
    # Two homes' shims on one PATH would each take the other for the real claude, and nest fritters without end.
    shim = installed_shim(root)
    other = executable(root / "other" / "claude", shim_script(root / "other" / "fritter", WIRE))
    (root / "alias").mkdir()
    (root / "alias" / "claude").symlink_to(shim)
    ran = on_a_pipe([str(shim)], f"{root / 'alias'}:{root / 'bin'}:{other.parent}:{root / 'bin'}:{root / 'real'}:/usr/bin:/bin")
    assert ran.stdout == "claude  socket=unset\n"


def test_the_brain_finds_the_real_claude_as_the_shim_does_past_every_shim(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The brain runs under hands' own fritter, so a shim it found would wrap it in a second fritter and tap it as a session.
    shim = installed_shim(root)
    other = executable(root / "other" / "claude", shim_script(root / "other" / "fritter", WIRE))
    (root / "alias").mkdir()
    (root / "alias" / "claude").symlink_to(shim)
    assert wrapper.real_claude(f"{root / 'alias'}:{other.parent}:{root / 'bin'}:{root / 'real'}") == root / "real" / "claude"
    assert wrapper.real_claude(f"{root / 'bin'}:{other.parent}") is None
    monkeypatch.chdir(root / "real")
    assert wrapper.real_claude(f"{root / 'bin'}::/usr/bin") == root / "real" / "claude"


@pytest.mark.parametrize("entries", ["{bin}::/usr/bin:/bin", "{bin}:/usr/bin:/bin:", ":{bin}:/usr/bin:/bin"])
def test_an_empty_path_entry_is_the_current_directory(root: Path, entries: str) -> None:
    shim = installed_shim(root)
    path = entries.format(bin=root / "bin")
    ran = subprocess.run([str(shim)], env={"PATH": path}, cwd=root / "real", stdin=subprocess.DEVNULL, capture_output=True, text=True)
    assert ran.stdout == "claude  socket=unset\n"


def test_any_home_s_shim_names_the_fritter_and_wire_it_runs_and_nothing_else_is_one(root: Path) -> None:
    shim = installed_shim(root)
    other = executable(root / "it's other" / "claude", shim_script(root / "it's other" / "fritter", root / "it's other" / "wire.sock"))
    assert wrapper.shim_of(shim) == wrapper.Shim(root / "bin" / "fritter", WIRE)
    assert wrapper.shim_of(other) == wrapper.Shim(root / "it's other" / "fritter", root / "it's other" / "wire.sock")
    unreadable = executable(root / "locked" / "claude", "#!/bin/sh\n")
    unreadable.chmod(0o111)
    for path in (root / "real" / "claude", root / "nowhere" / "claude", unreadable, root / "bin"):
        assert wrapper.shim_of(path) is None


@pytest.mark.parametrize(
    "lines",
    # The first is a shim as hands wrote it before the wire: a fritter and no wire.
    ["fritter=f\n\nexec f", "fritter=\nwire=w", "fritter='x\nwire=w", "fritter=a b\nwire=w", "fritter=f\nwire='w", "wire=w\nfritter=f"],
)
def test_a_marked_file_that_is_not_the_shim_this_hands_writes_is_stale(root: Path, lines: str) -> None:
    assert wrapper.shim_of(executable(root / "claude", f"#!/bin/sh\n{wrapper.MARK}\n{lines}\n")) == wrapper.Stale()


def test_no_real_claude_on_path_is_said_and_runs_nothing(root: Path) -> None:
    shim = installed_shim(root)
    ran = on_a_pipe([str(shim)], f"{root / 'bin'}:/usr/bin:/bin")
    assert ran.returncode == 127
    assert (ran.stdout, ran.stderr) == ("", "claude: nothing on PATH named claude but hands' shims\n")


def test_a_path_with_spaces_and_quotes_is_the_path_the_shim_names(root: Path) -> None:
    bin = root / "it's a bin"
    executable(bin / "fritter", RECORDER)
    executable(root / "real" / "claude", RECORDER)
    shim = executable(bin / "claude", shim_script(bin / "fritter", WIRE))
    printed = on_a_terminal([str(shim)], f"{bin}:{root / 'real'}:/usr/bin:/bin")
    assert printed == f"fritter {TAPPED} -- {root / 'real' / 'claude'} socket=unset\n"


def test_install_copies_the_packaged_fritter_with_no_go_and_says_whether_path_finds_its_claude(root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(root / "home")
    executable(root / "real" / "claude", RECORDER)
    tools = f"{root / 'real'}:/usr/bin:/bin"
    assert shutil.which("go", path=tools) is None

    monkeypatch.setenv("PATH", tools)
    assert main(["--home", str(home.root), "install-fritter"]) == 1
    said = capsys.readouterr().err
    assert f"`claude` on this PATH is {root / 'real' / 'claude'}, not hands' shim" in said
    assert f'export PATH="{home.bin}:$PATH"' in said and "install-fritter" not in said
    first = (home.bin / "claude").read_bytes()

    monkeypatch.setenv("PATH", f"{home.bin}:{tools}")
    assert main(["--home", str(home.root), "install-fritter"]) == 0
    assert f"copied {wrapper.PACKAGED} to {home.fritter}" in (out := capsys.readouterr().out)
    assert f"`claude` on this PATH is hands' shim, {home.shim}" in out
    assert (home.bin / "claude").read_bytes() == first
    assert home.fritter.read_bytes() == wrapper.PACKAGED.read_bytes()
    assert sorted(entry.name for entry in home.bin.iterdir()) == ["claude", "fritter"]

    printed = on_a_terminal(["claude", "say hi"], f"{home.bin}:{tools}")
    assert printed.startswith("claude say hi socket=/tmp/fritter-")
    # [LAW:nothing-unseen] each install's event: where it copied from and to, and whether PATH found its claude.
    events = [line for line in (json.loads(line) for line in segment(home.audit, 0).read_text().splitlines()) if line.get("event") == "fritter.install"]
    facts = {"packaged": str(wrapper.PACKAGED), "fritter": str(home.fritter), "shim": str(home.shim)}
    assert [(event["event"], event["outcome"], event["facts"]) for event in events] == [
        ("fritter.install", "ok", {**facts, "path_finds_it": False}),
        ("fritter.install", "ok", {**facts, "path_finds_it": True}),
    ]


def test_install_without_a_packaged_fritter_fails_naming_where_it_looked(root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(root / "home")
    monkeypatch.setattr(wrapper, "PACKAGED", root / "package" / "bin" / "fritter")
    assert main(["--home", str(home.root), "install-fritter"]) == 1
    assert f"hands install-fritter: hands' package carries no fritter at {root / 'package' / 'bin' / 'fritter'}: install hands again" in capsys.readouterr().err
    # Refused before anything is written: not even the home's bin.
    assert not home.bin.exists()
    [event] = [line for line in (json.loads(line) for line in segment(home.audit, 0).read_text().splitlines()) if line.get("event") == "fritter.install"]
    assert (event["event"], event["outcome"], event["facts"]) == ("fritter.install", "failed", {"packaged": str(root / "package" / "bin" / "fritter")})
    assert event["error"].startswith("hands' package carries no fritter at")


def test_a_relative_home_is_this_directory_s(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The home is written into the shim, which runs from every directory.
    monkeypatch.chdir(root)
    homes: list[Home] = []

    def install(home: Home) -> wrapper.Installed:
        homes.append(home)
        raise wrapper.Uninstallable("enough")

    monkeypatch.setattr(wrapper, "install", install)
    assert main(["--home", "h", "install-fritter"]) == 1
    assert homes == [Home(root / "h")]


# Says what the claude it stands in for would reach the API through.
REACH_RECORDER = '#!/bin/sh\nprintf "%s api=%s proxy=%s trust=%s tap=%s\\n" "$(basename "$0")" "${ANTHROPIC_BASE_URL-unset}" "${HTTPS_PROXY-unset}" "${NODE_EXTRA_CA_CERTS-unset}" "${FRITTER_TAP-unset}"\n'
TAP = "http://127.0.0.1:40000"
# A session's environment as its fritter left it: its own API, and the tap as its proxy in place of the one it had.
INSIDE = {"ANTHROPIC_BASE_URL": "https://gateway.example", "FRITTER_TAP": TAP, "HTTPS_PROXY": TAP, "NODE_EXTRA_CA_CERTS": "/tmp/fritter-1/trusted.pem", "FRITTER_OUTER_HTTPS_PROXY": "http://corp:3128"}


def test_a_claude_run_from_inside_a_session_reaches_the_api_as_the_session_would_have_without_its_tap(root: Path) -> None:
    # The tap ends when the session does; a run started inside it may not.
    shim = installed_shim(root)
    executable(root / "real" / "claude", REACH_RECORDER)
    ran = on_a_pipe([str(shim), "-p", "hello"], f"{root / 'bin'}:{root / 'real'}:/usr/bin:/bin", INSIDE)
    assert ran.stdout == "claude api=https://gateway.example proxy=http://corp:3128 trust=unset tap=unset\n"


def test_a_session_started_inside_a_session_is_tapped_by_its_own_fritter_alone(root: Path) -> None:
    bin = root / "bin"
    executable(bin / "fritter", REACH_RECORDER)
    executable(root / "real" / "claude", RECORDER)
    shim = executable(bin / "claude", shim_script(bin / "fritter", WIRE))
    printed = on_a_terminal([str(shim)], f"{bin}:{root / 'real'}:/usr/bin:/bin", INSIDE)
    assert printed == "fritter api=https://gateway.example proxy=http://corp:3128 trust=unset tap=unset\n"


def test_a_proxy_set_again_inside_a_session_is_the_one_a_claude_run_there_reaches_through(root: Path) -> None:
    shim = installed_shim(root)
    executable(root / "real" / "claude", REACH_RECORDER)
    ran = on_a_pipe([str(shim), "-p", "hello"], f"{root / 'bin'}:{root / 'real'}:/usr/bin:/bin", {**INSIDE, "HTTPS_PROXY": "http://other:8080"})
    assert ran.stdout == "claude api=https://gateway.example proxy=http://other:8080 trust=/tmp/fritter-1/trusted.pem tap=unset\n"


@pytest.mark.parametrize(("given", "tapped"), [({}, "https://api.anthropic.com"), ({"ANTHROPIC_BASE_URL": "https://gateway.example/v1"}, "https://gateway.example/v1")])
def test_a_session_is_tapped_toward_the_api_it_was_given_and_still_names_it(root: Path, given: dict[str, str], tapped: str) -> None:
    # Claude Code keeps what it keeps for Anthropic's own API only while the API it names is Anthropic's.
    bin = root / "bin"
    executable(bin / "fritter", '#!/bin/sh\nprintf "%s api=%s\\n" "$*" "${ANTHROPIC_BASE_URL-unset}"\n')
    executable(root / "real" / "claude", RECORDER)
    shim = executable(bin / "claude", shim_script(bin / "fritter", WIRE))
    printed = on_a_terminal([str(shim)], f"{bin}:{root / 'real'}:/usr/bin:/bin", given)
    assert printed == f"--tap {tapped} --tap-ca NODE_EXTRA_CA_CERTS --tap-to {WIRE} -- {root / 'real' / 'claude'} api={given.get('ANTHROPIC_BASE_URL', 'unset')}\n"


@pytest.mark.parametrize(
    ("args", "run"),
    [
        ([], "session"),
        (["--resume", "abc"], "session"),
        (["--model", "opus"], "session"),
        (["fix the readme"], "session"),
        (["Hello"], "session"),
        (["--model", "opus", "fix the readme"], "session"),
        (["--", "update"], "session"),
        (["-p", "hi"], "print"),
        (["--print"], "print"),
        (["--model", "opus", "-p", "hi"], "print"),
        (["-cp", "hi"], "print"),
        (["-pc"], "print"),
        (["-dp"], "session"),
        (["-rp"], "session"),
        (["--", "-p"], "session"),
        (["mcp", "serve"], "subcommand"),
        (["update"], "subcommand"),
        (["remote-control", "--spawn", "worktree"], "subcommand"),
        (["setup-token"], "subcommand"),
        (["bg-pty-host"], "subcommand"),
    ],
)
def test_a_claude_already_running_is_a_session_by_the_shims_own_test(root: Path, args: list[str], run: str) -> None:
    shim = installed_shim(root)
    printed = on_a_terminal([str(shim), *args], f"{root / 'bin'}:{root / 'real'}:/usr/bin:/bin")
    assert printed.startswith("fritter ") == (run == "session")
    assert wrapper.run(args, True) == run
    assert wrapper.run(args, False) == "piped"
