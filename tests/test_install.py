"""Tests for install.sh, the README's install command, run in a sandboxed home directory: what it installs on a new Mac,
and what a second run changes.

The installers it downloads and the tools it runs are replaced by stubs on PATH. Each stub records its call and creates
the files the real tool would, so a run changes nothing outside the sandbox and needs no network.
"""

import os
import pty
import re
import select
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from hands.daemon import readiness
from hands.sessions.hookconfig import MARKETPLACE

REPO = Path(__file__).resolve().parent.parent
INSTALL = REPO / "install.sh"
LOGIN_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"

# Each stub appends its name and arguments to $LOG, one call per line.
CURL = r"""#!/bin/bash
echo "curl $*" >>"$LOG"
case "$*" in
  *claude.ai/install.sh*)
    printf '%s\n' 'echo "claude installer" >>"$LOG"' 'mkdir -p "$HOME/.local/bin"' 'printf "#!/bin/sh\n" >"$HOME/.local/bin/claude"' 'chmod +x "$HOME/.local/bin/claude"' ;;
  *Homebrew/install*)
    [ ! -e "$STUBS/homebrew-unreachable" ] || exit 22
    printf '%s\n' 'echo "homebrew installer NONINTERACTIVE=$NONINTERACTIVE" >>"$LOG"' 'mkdir -p "$HOMEBREW_PREFIX/bin"' 'cp "$STUBS/brew.real" "$HOMEBREW_PREFIX/bin/brew"' ;;
  *releases/latest*)
    printf '%s' "https://github.com/promptctl/cc-hands/releases/tag/$RELEASE" ;;
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
# The stub hands that uv installs. install-fritter writes a claude shim in ~/.hands/bin. Each step succeeds unless a marker
# file in $STUBS makes it fail: an unfinished first run or login, a declined plugin, a permission not given, and so on.
# `hands run` records which claude its PATH finds, then exits with 0 (as when the user presses q), or with 3 if hands is
# already running.
HANDS = r"""#!/bin/bash
echo "hands $*" >>"${LOG:-/dev/null}"
case "$1" in
  --version) echo "hands @VERSION@" ;;
  install-fritter) mkdir -p "$HOME/.hands/bin" && printf '#!/bin/sh\n' >"$HOME/.hands/bin/claude" && chmod +x "$HOME/.hands/bin/claude" ;;
  first-run)
    # Like the real command: with no terminal it cannot ask (2), and at a terminal the user may quit before finishing (1).
    if [ -e "$STUBS/first-run-quit-once" ]; then
      [ -t 0 ] || exit 2
      rm "$STUBS/first-run-quit-once"
      exit 1
    fi
    # Unfinished: with no terminal it cannot ask (2); at a terminal the user did not finish (1).
    [ ! -e "$STUBS/first-run-unfinished" ] || { [ -t 0 ] && exit 1; exit 2; }
    [ ! -e "$STUBS/first-run-unaskable" ] || exit 2 ;;
  login)
    [ ! -e "$STUBS/login-unfinished" ] || { [ -t 0 ] && exit 1; exit 2; }
    [ ! -e "$STUBS/login-broken" ] || { echo "hands login: could not write settings" >&2; exit 1; }
    [ ! -e "$STUBS/login-unaskable" ] || exit 2 ;;
  install-plugin)
    [ ! -e "$STUBS/plugin-declined" ] || exit 1
    [ ! -e "$STUBS/claude-unaskable" ] || exit 2 ;;
  grant)
    [ ! -e "$STUBS/grant-not-given" ] || exit 1
    [ ! -e "$STUBS/grant-over-ssh" ] || exit 2
    [ ! -e "$STUBS/grant-crashed" ] || { echo "Traceback (most recent call last):" >&2; exit 70; } ;;
  run)
    echo "hands run finds claude $(command -v claude)" >>"${LOG:-/dev/null}"
    [ ! -e "$STUBS/running" ] || { echo "hands: hands is already running" >&2; exit 3; }
    [ ! -e "$STUBS/run-refused" ] || exit 1 ;;
esac
"""
# The account's login shell. install.sh writes PATH lines for this shell, regardless of which shell started it.
DSCL = r"""#!/bin/bash
[ ! -e "$STUBS/dscl-fails" ] || exit 56
echo "UserShell: $LOGIN_SHELL"
"""
# sudo -v is where the password prompt appears, and the stub prints a marker there. The keep-alive loop's sudo -n -v fails.
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

    def environment(self, release: str, shell: str, running_from: str | None) -> dict[str, str]:
        return {
            "HOME": str(self.home),
            "PATH": f"{self.root / 'stubs'}:{LOGIN_PATH}",
            "SHELL": running_from or shell,
            "LOGIN_SHELL": shell,
            "HOMEBREW_PREFIX": str(self.root / "brew"),
            "STUBS": str(self.root / "stubs"),
            "LOG": str(self.log),
            "RELEASE": release,
        }

    def run_at_a_terminal(self, until: str) -> str:
        """Runs the installer at a terminal, as the README's command does, pressing Enter at each prompt. Returns the output
        up to `until`."""
        self.log.write_text("")
        main, side = pty.openpty()
        started = subprocess.Popen(["/bin/bash", str(INSTALL)], env=self.environment("v9.9.9", "/bin/zsh", None), stdin=side, stdout=side, stderr=side, start_new_session=True)
        os.close(side)
        printed = b""
        answered = 0
        deadline = time.monotonic() + 60
        try:
            while until.encode() not in printed and time.monotonic() < deadline:
                ready, _, _ = select.select([main], [], [], 0.2)
                if ready:
                    printed += os.read(main, 4096)
                    # Press Enter once for each prompt shown so far.
                    pauses = len(re.findall(rb"Press Enter to (?:continue|try it again)", printed))
                    os.write(main, b"\n" * (pauses - answered))
                    answered = pauses
        finally:
            os.close(main)
            started.kill()
            started.wait()
        return printed.decode(errors="replace")

    def run(self, release: str = "v9.9.9", shell: str = "/bin/zsh", running_from: str | None = None) -> subprocess.CompletedProcess[str]:
        self.log.write_text("")
        return subprocess.run(["/bin/bash", str(INSTALL)], env=self.environment(release, shell, running_from), stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)

    def calls(self) -> list[str]:
        return self.log.read_text().splitlines()

    def login_finds(self, command: str, shell: str = "/bin/zsh", flags: str = "-ilc") -> str:
        """Returns the path of `command` as found by a new terminal's shell started with these flags, or an empty string."""
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
    executable(stubs / "dscl", DSCL)
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
    release = "https://github.com/promptctl/cc-hands/releases/download/v9.9.9"
    assert f"uv tool install --reinstall --python 3.12 --constraints {release}/constraints.txt {release}/hands-9.9.9-py3-none-macosx_12_0_arm64.whl" in calls
    assert calls.index("hands install-fritter") < calls.index("hands first-run") < calls.index("hands login") < calls.index("hands install-plugin") < calls.index("hands grant") < calls.index("hands run")
    # After every step is done, hands runs in this terminal with a new terminal's PATH, where claude is hands' shim.
    assert f"hands run finds claude {sandbox.home / '.hands/bin/claude'}" in calls
    # A new terminal finds each tool without the user editing a profile, and its claude is hands' shim.
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
    # Claude Code's installer suggests putting ~/.local/bin (where its claude is) first in PATH in ~/.zshrc, which a new
    # terminal reads after the profile.
    (sandbox.home / ".zshrc").write_text('export PATH="$HOME/.local/bin:$PATH"\n')
    ran = sandbox.run()
    assert ran.returncode == 0, ran.stderr
    assert sandbox.login_finds("claude") == str(sandbox.home / ".hands/bin/claude")
    # The change to the user's ~/.zshrc is shown on screen, not only in the log.
    assert "updated ~/.zshrc" in ran.stdout
    rc = (sandbox.home / ".zshrc").read_bytes()
    assert sandbox.run().returncode == 0
    assert (sandbox.home / ".zshrc").read_bytes() == rc


def test_a_run_stopped_part_way_is_finished_by_running_it_again(sandbox: Sandbox) -> None:
    # Simulates a first run that installed Claude Code and Homebrew, then stopped. The second run does only the rest.
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
    # hatch_build.py imports hatchling, which is only available during a build, so TAG is read from its source text.
    tag = re.search(r'^TAG = "(.+)"$', (REPO / "hatch_build.py").read_text(), re.MULTILINE)
    assert tag is not None and f"WHEEL_TAG={tag.group(1)}\n" in INSTALL.read_text()


def test_the_readme_and_hands_check_name_the_same_one_command() -> None:
    command = readiness.INSTALL.strip("`")
    assert command in (REPO / "README.md").read_text()
    assert command in INSTALL.read_text()


def test_a_bash_login_shell_gets_its_lines_in_the_one_profile_it_reads(sandbox: Sandbox) -> None:
    # A bash login shell reads only the first of .bash_profile, .bash_login, and .profile that exists. The user's existing
    # .profile must remain the file it reads, instead of being overridden by a new .bash_profile.
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
    # Output printed by a profile, with no trailing newline, appears before the PATH line the installer reads.
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
    assert ran.returncode == 1 and "Homebrew's installer could not be downloaded" in ran.stderr
    assert not any(call.startswith("brew ") for call in sandbox.calls())


def test_declining_the_plugin_fails_saying_to_run_it_again_with_every_step_before_it_done(sandbox: Sandbox) -> None:
    (sandbox.root / "stubs" / "plugin-declined").touch()
    ran = sandbox.run()
    assert ran.returncode == 1 and "plugin is not installed" in ran.stderr and "Run the install command again to continue from here" in ran.stderr
    assert sandbox.login_finds("claude") == str(sandbox.home / ".hands/bin/claude")
    (sandbox.root / "stubs" / "plugin-declined").unlink()
    assert sandbox.run().returncode == 0


def test_a_first_run_left_unfinished_with_no_terminal_stops_the_run_before_the_plugin(sandbox: Sandbox) -> None:
    (sandbox.root / "stubs" / "first-run-unfinished").touch()
    ran = sandbox.run()
    assert ran.returncode == 1 and "Claude Code's first run is not finished" in ran.stderr and "could not be asked" in ran.stderr
    # Checked once with no terminal; with no user to ask, it is not run a second time.
    assert sandbox.calls().count("hands first-run") == 1 and "hands install-plugin" not in sandbox.calls()
    (sandbox.root / "stubs" / "first-run-unfinished").unlink()
    assert sandbox.run().returncode == 0


def test_a_brain_check_that_fails_for_another_reason_stops_without_asking(sandbox: Sandbox) -> None:
    (sandbox.root / "stubs" / "login-broken").touch()
    said = sandbox.run_at_a_terminal(until="exit code 1")
    assert "hands login failed with exit code 1" in said and "could not write settings" in said
    assert "Your browser opens" not in said
    assert sandbox.calls().count("hands login") == 1 and "hands install-plugin" not in sandbox.calls()


def test_a_brain_login_claude_code_could_not_be_asked_for_fails_without_saying_to_run_it_again(sandbox: Sandbox) -> None:
    (sandbox.root / "stubs" / "login-unaskable").touch()
    ran = sandbox.run()
    assert ran.returncode == 1 and "the brain is not signed in" in ran.stderr and "could not be asked" in ran.stderr and "continue from here" not in ran.stderr
    assert "hands install-plugin" not in sandbox.calls()


def test_a_first_run_claude_code_could_not_be_asked_for_fails_without_saying_to_run_it_again(sandbox: Sandbox) -> None:
    (sandbox.root / "stubs" / "first-run-unaskable").touch()
    ran = sandbox.run()
    assert ran.returncode == 1 and "Claude Code could not be asked" in ran.stderr and "continue from here" not in ran.stderr


def test_a_claude_that_cannot_be_asked_fails_without_saying_to_run_it_again(sandbox: Sandbox) -> None:
    (sandbox.root / "stubs" / "claude-unaskable").touch()
    ran = sandbox.run()
    assert ran.returncode == 1 and "Claude Code could not be asked" in ran.stderr and "continue from here" not in ran.stderr


def test_a_shell_whose_startup_files_end_it_without_a_terminal_is_judged_from_its_login_shell(sandbox: Sandbox) -> None:
    # Simulates a ~/.zshrc that starts tmux: with no terminal, tmux exits 1 and the shell exits with it.
    (sandbox.home / ".zshrc").write_text("exit 1\n")
    ran = sandbox.run()
    assert ran.returncode == 0, ran.stderr
    assert "checks your login shell instead" in ran.stdout
    assert f'export PATH="{sandbox.home}/.hands/bin:$PATH"' in (sandbox.home / ".zprofile").read_text().splitlines()
    assert sandbox.login_finds("claude", flags="-lc") == str(sandbox.home / ".hands/bin/claude")
    profile = (sandbox.home / ".zprofile").read_bytes()
    assert sandbox.run().returncode == 0
    assert (sandbox.home / ".zprofile").read_bytes() == profile


def test_a_startup_file_read_after_the_shim_s_line_that_undoes_it_stops_the_run_saying_so(sandbox: Sandbox) -> None:
    # zsh reads ~/.zlogin after ~/.zshrc.
    (sandbox.home / ".zlogin").write_text('export PATH="$HOME/.local/bin:$PATH"\n')
    ran = sandbox.run()
    assert ran.returncode == 1 and "A startup file read after it puts another claude first" in ran.stderr
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
    # The user added ~/.hands/bin to PATH before ~/.local/bin themselves, so the native claude is found first.
    (sandbox.home / ".zprofile").write_text('export PATH="$HOME/.hands/bin:$PATH"\nexport PATH="$HOME/.local/bin:$PATH"\n')
    assert sandbox.run().returncode == 0
    assert sandbox.login_finds("claude") == str(sandbox.home / ".hands/bin/claude")


def test_the_releases_it_installs_and_the_marketplace_hands_adds_are_one_repository() -> None:
    assert f"REPO={MARKETPLACE}\n" in INSTALL.read_text()


def test_a_grant_not_given_in_time_stops_the_run_before_hands_runs_saying_to_run_it_again(sandbox: Sandbox) -> None:
    (sandbox.root / "stubs" / "grant-not-given").touch()
    ran = sandbox.run()
    assert ran.returncode == 1 and "without the Input Monitoring permission" in ran.stderr and "Run the install command again to continue from here" in ran.stderr
    assert "hands run" not in sandbox.calls()
    (sandbox.root / "stubs" / "grant-not-given").unlink()
    assert sandbox.run().returncode == 0 and "hands run" in sandbox.calls()


def test_a_grant_that_cannot_be_given_here_fails_without_saying_to_run_it_again(sandbox: Sandbox) -> None:
    (sandbox.root / "stubs" / "grant-over-ssh").touch()
    ran = sandbox.run()
    assert ran.returncode == 1 and "cannot be given here" in ran.stderr and "again" not in ran.stderr
    assert "hands run" not in sandbox.calls()


def test_a_second_run_beside_a_hands_that_runs_starts_no_second_one(sandbox: Sandbox) -> None:
    assert sandbox.run().returncode == 0
    (sandbox.root / "stubs" / "running").touch()
    again = sandbox.run()
    assert again.returncode == 0, again.stderr
    assert "hands is already running" in again.stderr and "hands is already running, so the installer did not start a second copy" in again.stdout


def test_a_step_that_crashed_is_told_as_its_failure_not_as_a_question_to_ask_again(sandbox: Sandbox) -> None:
    (sandbox.root / "stubs" / "grant-crashed").touch()
    ran = sandbox.run()
    assert ran.returncode == 1 and "hands grant failed with exit code 70" in ran.stderr and "again" not in ran.stderr
    assert "hands run" not in sandbox.calls()


def test_a_hands_that_will_not_run_fails_the_run_saying_so(sandbox: Sandbox) -> None:
    (sandbox.root / "stubs" / "run-refused").touch()
    ran = sandbox.run()
    assert ran.returncode == 1 and "hands stopped with exit code 1" in ran.stderr


def test_the_profile_written_is_the_login_shell_s_not_the_one_it_was_run_from(sandbox: Sandbox) -> None:
    # A tool running bash starts the installer for a user whose terminals run zsh.
    ran = sandbox.run(shell="/bin/zsh", running_from="/bin/bash")
    assert ran.returncode == 0, ran.stderr
    assert (sandbox.home / ".zprofile").exists() and not (sandbox.home / ".bash_profile").exists()
    assert sandbox.login_finds("claude") == str(sandbox.home / ".hands/bin/claude")


def test_it_opens_on_what_it_will_ask_and_numbers_every_step(sandbox: Sandbox) -> None:
    said = sandbox.run().stdout
    assert said.index("your Mac password") < said.index("[1/9] Claude Code")
    assert [int(n) for n in re.findall(r"\[(\d)/9\]", said)] == list(range(1, 10))
    assert "All set." in said


def test_the_tools_own_output_goes_to_the_log_not_the_terminal(sandbox: Sandbox) -> None:
    ran = sandbox.run()
    assert "claude installer" not in ran.stdout + ran.stderr
    assert "$ brew install portaudio" in (sandbox.home / "Library/Logs/hands-install.log").read_text()


def test_a_step_the_person_quit_is_offered_again_on_the_spot_at_a_terminal(sandbox: Sandbox) -> None:
    (sandbox.root / "stubs" / "first-run-quit-once").touch()
    said = sandbox.run_at_a_terminal(until="All set.")
    assert "That did not finish" in said and "All set." in said
    calls = sandbox.calls()
    # The check with no terminal, the attempt the user quit, the retry, and then the brain step.
    assert calls.count("hands first-run") == 3 and calls.index("hands login") > max(i for i, call in enumerate(calls) if call == "hands first-run")


def test_a_step_already_done_is_ticked_off_without_a_pause(sandbox: Sandbox) -> None:
    said = sandbox.run().stdout
    assert "already signed in and set up" in said and "type /exit" not in said


def test_a_login_shell_that_cannot_be_looked_up_falls_back_to_the_shell_it_was_run_from(sandbox: Sandbox) -> None:
    # Simulates a network account that the local directory does not contain.
    (sandbox.root / "stubs" / "dscl-fails").touch()
    ran = sandbox.run(shell="/bin/bash", running_from="/bin/zsh")
    assert ran.returncode == 0, ran.stderr
    assert (sandbox.home / ".zprofile").exists() and not (sandbox.home / ".bash_profile").exists()


def test_the_profile_message_names_only_what_was_added(sandbox: Sandbox) -> None:
    # A login PATH that already has Homebrew and ~/.local/bin (where both claude and hands are) gets no profile lines.
    brew = sandbox.root / "brew/bin"
    (sandbox.home / ".zprofile").write_text(f'export PATH="{brew}:$HOME/.local/bin:$PATH"\n')
    ran = sandbox.run()
    assert ran.returncode == 0, ran.stderr
    assert "so new terminals find" not in ran.stdout
    (sandbox.home / ".zprofile").write_text(f'export PATH="{brew}:$PATH"\n')
    again = sandbox.run()
    assert "updated ~/.zprofile so new terminals find Claude Code\n" in again.stdout


def test_a_run_whose_output_is_piped_still_asks_at_the_terminal(sandbox: Sandbox) -> None:
    # As with `/bin/bash -c "$(curl ...)" | tee install.txt`: input is the terminal, output is a pipe.
    (sandbox.root / "stubs" / "first-run-quit-once").touch()
    sandbox.log.write_text("")
    main, side = pty.openpty()
    started = subprocess.Popen(["/bin/bash", str(INSTALL)], env=sandbox.environment("v9.9.9", "/bin/zsh", None), stdin=side, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    os.close(side)
    printed = b""
    assert started.stdout is not None
    try:
        deadline = time.monotonic() + 60
        while b"All set." not in printed and time.monotonic() < deadline:
            # Enter at every prompt: keys typed before a prompt are discarded, so keep pressing until the run ends.
            os.write(main, b"\n")
            ready, _, _ = select.select([started.stdout], [], [], 0.3)
            if ready:
                printed += os.read(started.stdout.fileno(), 4096)
    finally:
        os.close(main)
        started.kill()
        started.wait()
    said = printed.decode(errors="replace")
    assert "That did not finish" in said and "All set." in said
