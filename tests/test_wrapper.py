"""The claude shim: a session on a terminal runs under fritter, anything else runs the real claude as it is."""

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from hands.daemon.cli import main
from hands.sessions.home import Home
from hands.sessions import wrapper
from hands.sessions.wrapper import shim_script

# Each stand-in says what it was run with, and whether it carries a fritter address.
RECORDER = '#!/bin/sh\nprintf "%s %s socket=%s\\n" "$(basename "$0")" "$*" "${FRITTER_SOCKET-unset}"\n'


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
    return executable(bin / "claude", shim_script(bin / "fritter"))


def on_a_pipe(argv: Sequence[str], path: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, env={"PATH": path, **(env or {})}, stdin=subprocess.DEVNULL, capture_output=True, text=True)


def on_a_terminal(argv: Sequence[str], path: str) -> str:
    """What argv prints when a terminal is its stdin, stdout, and stderr."""
    controller, terminal = os.openpty()
    process = subprocess.Popen(argv, env={"PATH": path, "TMPDIR": "/tmp"}, stdin=terminal, stdout=terminal, stderr=terminal, start_new_session=True)
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
    assert printed == f"fritter -- {root / 'real' / 'claude'} --resume abc socket=unset\n"


@pytest.mark.parametrize("args", [["-p", "hello"], ["--print", "hello"], ["--model", "opus", "-p", "hello"], ["-cp", "hello"], ["-pc", "hello"]])
def test_print_on_a_terminal_runs_the_real_claude(root: Path, args: list[str]) -> None:
    shim = installed_shim(root)
    printed = on_a_terminal([str(shim), *args], f"{root / 'bin'}:{root / 'real'}:/usr/bin:/bin")
    assert printed == f"claude {' '.join(args)} socket=unset\n"


def test_a_prompt_after_the_options_that_says_p_is_still_a_session(root: Path) -> None:
    shim = installed_shim(root)
    printed = on_a_terminal([str(shim), "--", "-p"], f"{root / 'bin'}:{root / 'real'}:/usr/bin:/bin")
    assert printed.startswith("fritter -- ")


def test_off_a_terminal_the_real_claude_runs_without_the_address_of_the_session_it_was_started_from(root: Path) -> None:
    shim = installed_shim(root)
    ran = on_a_pipe([str(shim), "mcp", "serve"], f"{root / 'bin'}:{root / 'real'}:/usr/bin:/bin", {"FRITTER_SOCKET": "/tmp/fritter-parent/sock"})
    assert (ran.returncode, ran.stdout, ran.stderr) == (0, "claude mcp serve socket=unset\n", "")


@pytest.mark.parametrize("args", [["-dp"], ["-rp"]])
def test_an_option_whose_value_is_p_is_a_session(root: Path, args: list[str]) -> None:
    # -d and -r take a value, so -dp is a debug filter and -rp a session to resume.
    shim = installed_shim(root)
    printed = on_a_terminal([str(shim), *args], f"{root / 'bin'}:{root / 'real'}:/usr/bin:/bin")
    assert printed.startswith("fritter -- ")


def test_every_hands_shim_is_skipped_however_path_names_it(root: Path) -> None:
    # Two homes' shims on one PATH would each take the other for the real claude, and nest fritters without end.
    shim = installed_shim(root)
    other = executable(root / "other" / "claude", shim_script(root / "other" / "fritter"))
    (root / "alias").mkdir()
    (root / "alias" / "claude").symlink_to(shim)
    ran = on_a_pipe([str(shim)], f"{root / 'alias'}:{root / 'bin'}:{other.parent}:{root / 'bin'}:{root / 'real'}:/usr/bin:/bin")
    assert ran.stdout == "claude  socket=unset\n"


@pytest.mark.parametrize("entries", ["{bin}::/usr/bin:/bin", "{bin}:/usr/bin:/bin:", ":{bin}:/usr/bin:/bin"])
def test_an_empty_path_entry_is_the_current_directory(root: Path, entries: str) -> None:
    shim = installed_shim(root)
    path = entries.format(bin=root / "bin")
    ran = subprocess.run([str(shim)], env={"PATH": path}, cwd=root / "real", stdin=subprocess.DEVNULL, capture_output=True, text=True)
    assert ran.stdout == "claude  socket=unset\n"


def test_another_home_s_shim_first_on_path_wraps_as_this_one_would(root: Path) -> None:
    shim = installed_shim(root)
    other = executable(root / "other" / "claude", shim_script(root / "other" / "fritter"))
    assert wrapper.Installed(shim, root / "bin" / "fritter", other).on_path
    assert not wrapper.Installed(shim, root / "bin" / "fritter", root / "real" / "claude").on_path
    assert not wrapper.Installed(shim, root / "bin" / "fritter", None).on_path


def test_no_real_claude_on_path_is_said_and_runs_nothing(root: Path) -> None:
    shim = installed_shim(root)
    ran = on_a_pipe([str(shim)], f"{root / 'bin'}:/usr/bin:/bin")
    assert ran.returncode == 127
    assert (ran.stdout, ran.stderr) == ("", "claude: nothing on PATH named claude but hands' shims\n")


def test_a_path_with_spaces_and_quotes_is_the_path_the_shim_names(root: Path) -> None:
    bin = root / "it's a bin"
    executable(bin / "fritter", RECORDER)
    executable(root / "real" / "claude", RECORDER)
    shim = executable(bin / "claude", shim_script(bin / "fritter"))
    printed = on_a_terminal([str(shim)], f"{bin}:{root / 'real'}:/usr/bin:/bin")
    assert printed == f"fritter -- {root / 'real' / 'claude'} socket=unset\n"


@pytest.mark.skipif(shutil.which("go") is None, reason="building fritter needs go")
def test_install_builds_a_fritter_that_gives_the_session_an_address_and_says_whether_path_finds_it(root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(root / "home")
    executable(root / "real" / "claude", RECORDER)
    tools = f"{root / 'real'}:{Path(shutil.which('go') or '').parent}:/usr/bin:/bin"

    monkeypatch.setenv("PATH", tools)
    assert main(["--home", str(home.root), "install-fritter"]) == 1
    said = capsys.readouterr().err
    assert f"`claude` on this PATH is {root / 'real' / 'claude'}, not the shim" in said
    assert f'export PATH="{home.bin}:$PATH"' in said
    first = (home.bin / "claude").read_bytes()

    monkeypatch.setenv("PATH", f"{home.bin}:{tools}")
    assert main(["--home", str(home.root), "install-fritter"]) == 0
    assert "every interactive claude started from this PATH runs under fritter" in capsys.readouterr().out
    assert (home.bin / "claude").read_bytes() == first
    assert sorted(entry.name for entry in home.bin.iterdir()) == ["claude", "fritter"]

    printed = on_a_terminal(["claude", "hi"], f"{home.bin}:{tools}")
    assert printed.startswith("claude hi socket=/tmp/fritter-")


def test_a_relative_home_is_this_directory_s(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The home is written into the shim, which runs from every directory.
    monkeypatch.chdir(root)
    homes: list[Home] = []

    def install(home: Home, path: str) -> wrapper.Installed:
        homes.append(home)
        raise wrapper.Uninstallable("enough")

    monkeypatch.setattr(wrapper, "install", install)
    assert main(["--home", "h", "install-fritter"]) == 1
    assert homes == [Home(root / "h")]
