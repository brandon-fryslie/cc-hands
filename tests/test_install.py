"""install.sh, the README's one command, run in a sandboxed home: what it installs on a bare Mac, what a second run leaves.

The installers it fetches and the tools it drives are stand-ins on PATH that record each call and make the file the real
one would, so a run touches nothing outside its sandbox and needs no network.
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from hands.daemon import readiness
from hands.sessions.hookconfig import MARKETPLACE

REPO = Path(__file__).resolve().parent.parent
INSTALL = REPO / "install.sh"
LOGIN_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"

# Each stand-in appends its name and arguments to $LOG, one call a line.
CURL = r"""#!/bin/bash
echo "curl $*" >>"$LOG"
case "$*" in
  *claude.ai/install.sh*)
    printf '%s\n' 'echo "claude installer" >>"$LOG"' 'mkdir -p "$HOME/.local/bin"' 'printf "#!/bin/sh\n" >"$HOME/.local/bin/claude"' 'chmod +x "$HOME/.local/bin/claude"' ;;
  *Homebrew/install*)
    [ ! -e "$STUBS/homebrew-unreachable" ] || exit 22
    printf '%s\n' 'echo "homebrew installer NONINTERACTIVE=$NONINTERACTIVE" >>"$LOG"' 'mkdir -p "$HOMEBREW_PREFIX/bin"' 'cp "$STUBS/brew.real" "$HOMEBREW_PREFIX/bin/brew"' ;;
  *releases/latest*)
    printf '%s' "https://github.com/brandon-fryslie/cc-hands/releases/tag/$RELEASE" ;;
  *) exit 22 ;;
esac
"""
BREW = r"""#!/bin/bash
echo "brew $*" >>"$LOG"
case "$1" in
  shellenv) printf 'export PATH="%s:$PATH"\n' "$(cd "$(dirname "$0")" && pwd)" ;;
  list) [ -d "$HOMEBREW_PREFIX/Cellar/$3" ] ;;
  install)
    mkdir -p "$HOMEBREW_PREFIX/Cellar/$2"
    [ "$2" != uv ] || cp "$STUBS/uv.real" "$HOMEBREW_PREFIX/bin/uv" ;;
esac
"""
UV = r"""#!/bin/bash
echo "uv $*" >>"$LOG"
case "$1 $2" in
  "tool dir") echo "$HOME/.local/bin" ;;
  "tool install")
    wheel=${!#}
    version=${wheel##*/hands-}
    version=${version%%-*}
    mkdir -p "$HOME/.local/bin"
    sed "s/@VERSION@/$version/" "$STUBS/hands.real" >"$HOME/.local/bin/hands"
    chmod +x "$HOME/.local/bin/hands" ;;
esac
"""
# The hands uv installs: its shim is a claude in ~/.hands/bin, Claude Code's first run is finished and the brain logged in
# unless the person quits them, and its plugin is installed unless the person declines.
HANDS = r"""#!/bin/bash
echo "hands $*" >>"${LOG:-/dev/null}"
case "$1" in
  --version) echo "hands @VERSION@" ;;
  install-fritter) mkdir -p "$HOME/.hands/bin" && printf '#!/bin/sh\n' >"$HOME/.hands/bin/claude" && chmod +x "$HOME/.hands/bin/claude" ;;
  first-run)
    [ ! -e "$STUBS/first-run-unfinished" ] || exit 1
    [ ! -e "$STUBS/first-run-unaskable" ] || exit 2 ;;
  login) [ ! -e "$STUBS/login-unfinished" ] || exit 1 ;;
  install-plugin)
    [ ! -e "$STUBS/plugin-declined" ] || exit 1
    [ ! -e "$STUBS/claude-unaskable" ] || exit 2 ;;
esac
"""
# sudo -v is where the password is asked; the run's output marks the moment. The keeper's sudo -n -v finds nothing cached.
SUDO = r"""#!/bin/bash
echo "sudo $*" >>"$LOG"
case "$*" in
  -v) echo "PASSWORD ASKED" ;;
  *) exit 1 ;;
esac
"""


@dataclass(frozen=True)
class Sandbox:
    root: Path

    @property
    def home(self) -> Path:
        return self.root / "home"

    @property
    def log(self) -> Path:
        return self.root / "log"

    def run(self, release: str = "v9.9.9", shell: str = "/bin/zsh") -> subprocess.CompletedProcess[str]:
        self.log.write_text("")
        environment = {
            "HOME": str(self.home),
            "PATH": f"{self.root / 'stubs'}:{LOGIN_PATH}",
            "SHELL": shell,
            "HOMEBREW_PREFIX": str(self.root / "brew"),
            "STUBS": str(self.root / "stubs"),
            "LOG": str(self.log),
            "RELEASE": release,
        }
        return subprocess.run(["/bin/bash", str(INSTALL)], env=environment, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)

    def calls(self) -> list[str]:
        return self.log.read_text().splitlines()

    def login_finds(self, command: str, shell: str = "/bin/zsh", flags: str = "-ilc") -> str:
        """Where a new terminal's shell, started with these flags, finds `command`, or the empty string."""
        found = subprocess.run(["env", "-i", f"HOME={self.home}", f"PATH={LOGIN_PATH}", shell, flags, f'printf "\\n@found@%s\\n" "$(command -v {command})"'], stdin=subprocess.DEVNULL, capture_output=True, text=True)
        return [line.removeprefix("@found@") for line in found.stdout.splitlines() if line.startswith("@found@")][-1]


def executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    stubs = tmp_path / "stubs"
    executable(stubs / "curl", CURL)
    executable(stubs / "sudo", SUDO)
    executable(stubs / "brew.real", BREW)
    executable(stubs / "uv.real", UV)
    executable(stubs / "hands.real", HANDS)
    (tmp_path / "home").mkdir()
    return Sandbox(tmp_path)


def test_a_bare_mac_gets_claude_code_portaudio_uv_and_the_newest_hands(sandbox: Sandbox) -> None:
    ran = sandbox.run()
    assert ran.returncode == 0, ran.stderr
    calls = sandbox.calls()
    assert "claude installer" in calls and "homebrew installer NONINTERACTIVE=1" in calls
    assert "brew install portaudio" in calls and "brew install uv" in calls
    release = "https://github.com/brandon-fryslie/cc-hands/releases/download/v9.9.9"
    assert f"uv tool install --reinstall --python 3.12 --constraints {release}/constraints.txt {release}/hands-9.9.9-py3-none-macosx_12_0_arm64.whl" in calls
    assert calls.index("hands install-fritter") < calls.index("hands first-run") < calls.index("hands login") < calls.index("hands install-plugin")
    # A new terminal finds each of them without the person touching a profile, and its claude is hands' shim.
    assert sandbox.login_finds("claude") == str(sandbox.home / ".hands/bin/claude")
    assert sandbox.login_finds("hands") == str(sandbox.home / ".local/bin/hands")
    assert sandbox.login_finds("brew") == str(sandbox.root / "brew/bin/brew")
    assert sandbox.login_finds("uv") == str(sandbox.root / "brew/bin/uv")


def test_the_administrator_password_is_announced_before_it_is_asked(sandbox: Sandbox) -> None:
    said = sandbox.run().stdout
    assert "administrator password" in said and said.index("administrator password") < said.index("PASSWORD ASKED")


def test_a_second_run_installs_nothing_and_leaves_the_profile_byte_for_byte(sandbox: Sandbox) -> None:
    assert sandbox.run().returncode == 0
    profile = (sandbox.home / ".zprofile").read_bytes()
    hands = (sandbox.home / ".local/bin/hands").read_bytes()
    again = sandbox.run()
    assert again.returncode == 0, again.stderr
    calls = sandbox.calls()
    assert not [call for call in calls if "installer" in call or " install " in f"{call} " or call.startswith("sudo")], calls
    assert (sandbox.home / ".zprofile").read_bytes() == profile
    assert (sandbox.home / ".local/bin/hands").read_bytes() == hands


def test_a_zshrc_that_puts_another_claude_first_is_answered_in_the_zshrc(sandbox: Sandbox) -> None:
    # As Claude Code's own installer suggests: ~/.local/bin, where its claude is, put first by ~/.zshrc, which a new
    # terminal reads after the profile.
    (sandbox.home / ".zshrc").write_text('export PATH="$HOME/.local/bin:$PATH"\n')
    ran = sandbox.run()
    assert ran.returncode == 0, ran.stderr
    assert sandbox.login_finds("claude") == str(sandbox.home / ".hands/bin/claude")
    rc = (sandbox.home / ".zshrc").read_bytes()
    assert sandbox.run().returncode == 0
    assert (sandbox.home / ".zshrc").read_bytes() == rc


def test_a_run_stopped_part_way_is_finished_by_running_it_again(sandbox: Sandbox) -> None:
    # As after a first run that got Claude Code and Homebrew in and stopped: the second does only the rest.
    assert sandbox.run().returncode == 0
    (sandbox.home / ".local/bin/hands").unlink()
    (sandbox.root / "brew/Cellar/portaudio").rmdir()
    again = sandbox.run()
    assert again.returncode == 0, again.stderr
    calls = sandbox.calls()
    assert "brew install portaudio" in calls and "brew install uv" not in calls
    assert "claude installer" not in calls and not any(call.startswith("homebrew installer") for call in calls)
    assert any(call.startswith("uv tool install") for call in calls)


def test_an_older_hands_is_replaced_by_the_newest_release(sandbox: Sandbox) -> None:
    assert sandbox.run(release="v9.9.8").returncode == 0
    assert sandbox.run(release="v9.9.9").returncode == 0
    said = subprocess.run([str(sandbox.home / ".local/bin/hands"), "--version"], capture_output=True, text=True).stdout.strip()
    assert said == "hands 9.9.9"


def test_a_newest_release_that_is_not_a_version_tag_installs_no_hands(sandbox: Sandbox) -> None:
    ran = sandbox.run(release="v1.2.3-rc.1")
    assert ran.returncode == 1 and "not a vX.Y.Z tag" in ran.stderr
    assert not any(call.startswith("uv tool install") for call in sandbox.calls())


def test_the_wheel_it_installs_carries_the_platform_tag_the_build_gives_it() -> None:
    # hatch_build.py imports hatchling, which only a build has, so its TAG is read off its source.
    tag = re.search(r'^TAG = "(.+)"$', (REPO / "hatch_build.py").read_text(), re.MULTILINE)
    assert tag is not None and f"WHEEL_TAG={tag.group(1)}\n" in INSTALL.read_text()


def test_the_readme_and_hands_check_name_the_same_one_command() -> None:
    command = readiness.INSTALL.strip("`")
    assert command in (REPO / "README.md").read_text()
    assert command in INSTALL.read_text()


def test_a_bash_login_shell_gets_its_lines_in_the_one_profile_it_reads(sandbox: Sandbox) -> None:
    # A bash login shell reads only the first of .bash_profile, .bash_login and .profile that exists: a person's own
    # .profile stays the one it reads, rather than being shadowed by a new .bash_profile.
    (sandbox.home / ".profile").write_text("export OWN=1\n")
    ran = sandbox.run(shell="/bin/bash")
    assert ran.returncode == 0, ran.stderr
    assert not (sandbox.home / ".bash_profile").exists()
    assert (sandbox.home / ".profile").read_text().startswith("export OWN=1\n")
    assert sandbox.login_finds("hands", shell="/bin/bash") == str(sandbox.home / ".local/bin/hands")


def test_a_bash_login_shell_with_no_profile_gets_a_new_bash_profile(sandbox: Sandbox) -> None:
    assert sandbox.run(shell="/bin/bash").returncode == 0
    assert sandbox.login_finds("brew", shell="/bin/bash") == str(sandbox.root / "brew/bin/brew")
    assert (sandbox.home / ".bash_profile").exists()


def test_a_shell_that_is_neither_zsh_nor_bash_is_told_the_lines_and_gets_no_profile(sandbox: Sandbox) -> None:
    ran = sandbox.run(shell="/bin/sh")
    assert ran.returncode == 1 and "is not zsh or bash" in ran.stderr and "brew shellenv" in ran.stderr
    assert not [path for path in sandbox.home.iterdir() if path.name in (".zprofile", ".bash_profile", ".bash_login", ".profile")]


def test_a_profile_that_prints_leaves_a_second_run_s_profile_byte_for_byte(sandbox: Sandbox) -> None:
    # What a profile prints comes before the PATH the probe reads, with no newline of its own.
    (sandbox.home / ".zprofile").write_text("printf welcome\n")
    assert sandbox.run().returncode == 0
    profile = (sandbox.home / ".zprofile").read_bytes()
    assert sandbox.run().returncode == 0
    assert (sandbox.home / ".zprofile").read_bytes() == profile


def test_the_newest_release_is_held_to_the_tag_rule_a_release_is_built_under() -> None:
    rule = re.compile(r"grep -Eqx '([^']+)'")
    release = rule.findall((REPO / ".github/workflows/release.yml").read_text())
    install = rule.findall(INSTALL.read_text())
    assert len(release) == 1 and install == release


def test_a_homebrew_installer_that_cannot_be_fetched_stops_the_run_there(sandbox: Sandbox) -> None:
    (sandbox.root / "stubs" / "homebrew-unreachable").touch()
    ran = sandbox.run()
    assert ran.returncode == 22 and "with Homebrew" not in ran.stdout
    assert not any(call.startswith("brew ") for call in sandbox.calls())


def test_declining_the_plugin_fails_saying_a_second_run_asks_again_with_every_step_before_it_done(sandbox: Sandbox) -> None:
    (sandbox.root / "stubs" / "plugin-declined").touch()
    ran = sandbox.run()
    assert ran.returncode == 1 and "plugin is not installed" in ran.stderr and "running this command again asks again" in ran.stderr
    assert sandbox.login_finds("claude") == str(sandbox.home / ".hands/bin/claude")
    (sandbox.root / "stubs" / "plugin-declined").unlink()
    assert sandbox.run().returncode == 0


def test_a_first_run_left_unfinished_stops_the_run_before_the_plugin_saying_a_second_run_asks_again(sandbox: Sandbox) -> None:
    (sandbox.root / "stubs" / "first-run-unfinished").touch()
    ran = sandbox.run()
    assert ran.returncode == 1 and "Claude Code's first run is not finished" in ran.stderr and "running this command again asks again" in ran.stderr
    assert "hands install-plugin" not in sandbox.calls()
    (sandbox.root / "stubs" / "first-run-unfinished").unlink()
    assert sandbox.run().returncode == 0


def test_a_brain_left_logged_out_stops_the_run_before_the_plugin_saying_a_second_run_asks_again(sandbox: Sandbox) -> None:
    (sandbox.root / "stubs" / "login-unfinished").touch()
    ran = sandbox.run()
    assert ran.returncode == 1 and "the brain is not logged in" in ran.stderr and "running this command again asks again" in ran.stderr
    assert "hands install-plugin" not in sandbox.calls()
    (sandbox.root / "stubs" / "login-unfinished").unlink()
    assert sandbox.run().returncode == 0


def test_a_first_run_claude_code_could_not_be_asked_for_fails_without_saying_a_second_run_asks_again(sandbox: Sandbox) -> None:
    (sandbox.root / "stubs" / "first-run-unaskable").touch()
    ran = sandbox.run()
    assert ran.returncode == 1 and "Claude Code could not be asked" in ran.stderr and "asks again" not in ran.stderr


def test_a_claude_that_cannot_be_asked_fails_without_saying_a_second_run_asks_again(sandbox: Sandbox) -> None:
    (sandbox.root / "stubs" / "claude-unaskable").touch()
    ran = sandbox.run()
    assert ran.returncode == 1 and "Claude Code could not be asked" in ran.stderr and "asks again" not in ran.stderr


def test_a_shell_whose_startup_files_end_it_without_a_terminal_is_judged_from_its_login_shell(sandbox: Sandbox) -> None:
    # As a ~/.zshrc that starts tmux, which with no terminal exits 1, and the shell with it.
    (sandbox.home / ".zshrc").write_text("exit 1\n")
    ran = sandbox.run()
    assert ran.returncode == 0, ran.stderr
    assert "judged from its login shell alone" in ran.stdout
    assert f'export PATH="{sandbox.home}/.hands/bin:$PATH"' in (sandbox.home / ".zprofile").read_text().splitlines()
    assert sandbox.login_finds("claude", flags="-lc") == str(sandbox.home / ".hands/bin/claude")
    profile = (sandbox.home / ".zprofile").read_bytes()
    assert sandbox.run().returncode == 0
    assert (sandbox.home / ".zprofile").read_bytes() == profile


def test_a_startup_file_read_after_the_shim_s_line_that_undoes_it_stops_the_run_saying_so(sandbox: Sandbox) -> None:
    # zsh reads ~/.zlogin after ~/.zshrc.
    (sandbox.home / ".zlogin").write_text('export PATH="$HOME/.local/bin:$PATH"\n')
    ran = sandbox.run()
    assert ran.returncode == 1 and "a startup file read after it puts another claude first" in ran.stderr
    assert "hands install-plugin" not in sandbox.calls()
    rc = (sandbox.home / ".zshrc").read_bytes()
    assert sandbox.run().returncode == 1
    assert (sandbox.home / ".zshrc").read_bytes() == rc


def test_what_a_logout_file_prints_is_not_taken_for_the_path(sandbox: Sandbox) -> None:
    (sandbox.home / ".zlogout").write_text("echo bye\n")
    assert sandbox.run().returncode == 0
    assert sandbox.login_finds("claude") == str(sandbox.home / ".hands/bin/claude")
    profile = (sandbox.home / ".zprofile").read_bytes()
    assert sandbox.run().returncode == 0
    assert (sandbox.home / ".zprofile").read_bytes() == profile


def test_a_shim_behind_the_native_claude_on_the_login_path_is_put_first(sandbox: Sandbox) -> None:
    # As when a person put ~/.hands/bin on PATH themselves, before ~/.local/bin: the native claude is found first.
    (sandbox.home / ".zprofile").write_text('export PATH="$HOME/.hands/bin:$PATH"\nexport PATH="$HOME/.local/bin:$PATH"\n')
    assert sandbox.run().returncode == 0
    assert sandbox.login_finds("claude") == str(sandbox.home / ".hands/bin/claude")


def test_the_releases_it_installs_and_the_marketplace_hands_adds_are_one_repository() -> None:
    assert f"REPO={MARKETPLACE}\n" in INSTALL.read_text()
