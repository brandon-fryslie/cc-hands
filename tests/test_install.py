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
    printf '#!/bin/sh\necho "hands %s"\n' "$version" >"$HOME/.local/bin/hands"
    chmod +x "$HOME/.local/bin/hands" ;;
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

    def run(self, release: str = "v9.9.9") -> subprocess.CompletedProcess[str]:
        self.log.write_text("")
        environment = {
            "HOME": str(self.home),
            "PATH": f"{self.root / 'stubs'}:{LOGIN_PATH}",
            "SHELL": "/bin/zsh",
            "HOMEBREW_PREFIX": str(self.root / "brew"),
            "STUBS": str(self.root / "stubs"),
            "LOG": str(self.log),
            "RELEASE": release,
        }
        return subprocess.run(["/bin/bash", str(INSTALL)], env=environment, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)

    def calls(self) -> list[str]:
        return self.log.read_text().splitlines()

    def login_finds(self, command: str) -> str:
        """Where a new terminal's login shell finds `command`, or the empty string."""
        found = subprocess.run(["env", "-i", f"HOME={self.home}", f"PATH={LOGIN_PATH}", "/bin/zsh", "-lc", f"command -v {command}"], capture_output=True, text=True)
        return found.stdout.strip()


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
    (tmp_path / "home").mkdir()
    return Sandbox(tmp_path)


def test_a_bare_mac_gets_claude_code_portaudio_uv_and_the_newest_hands(sandbox: Sandbox) -> None:
    ran = sandbox.run()
    assert ran.returncode == 0, ran.stderr
    calls = sandbox.calls()
    assert "claude installer" in calls and "homebrew installer NONINTERACTIVE=1" in calls
    assert "brew install portaudio" in calls and "brew install uv" in calls
    release = "https://github.com/brandon-fryslie/cc-hands/releases/download/v9.9.9"
    assert f"uv tool install --python 3.12 --constraints {release}/constraints.txt {release}/hands-9.9.9-py3-none-macosx_12_0_arm64.whl" in calls
    # A new terminal finds each of them without the person touching a profile.
    assert sandbox.login_finds("claude") == str(sandbox.home / ".local/bin/claude")
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
