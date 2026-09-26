"""The `claude` that starts every interactive session under fritter, and its install.

    hands install-fritter     # builds fritter from this checkout, and writes that claude beside it in <hands home>/bin

fritter reaches only the sessions started under it. A `claude` earlier on PATH than the real one, which execs
fritter around it, wraps every session nobody remembered to wrap. Sessions already running when it is installed
stay unwrapped until they end.
"""

import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from hands.sessions.files import replace_whole
from hands.sessions.home import Home

# fritter's Go module, in the checkout this hands runs from.
FRITTER_SOURCE = Path(__file__).resolve().parents[3] / "fritter"

# Every shim's second line, by which a shim knows another hands shim on PATH for what it is.
MARK = "# A hands claude shim, written whole by `hands install-fritter`: change hands.sessions.wrapper, not this."


class Uninstallable(Exception):
    """fritter could not be built, or the claude beside it could not be written. The message says which."""


@dataclass(frozen=True)
class Installed:
    shim: Path
    fritter: Path
    # What `claude` names on the PATH the install ran under: the shim, or the claude it would be ahead of.
    found: Path | None

    @property
    def on_path(self) -> bool:
        return self.found is not None and self.found.samefile(self.shim)


def shim_script(fritter: Path) -> str:
    """The shim's text: a claude that runs a session under fritter and anything else as it would run without it."""
    # [LAW:one-source-of-truth] the real claude is looked up on PATH as each run starts, never recorded here, so the
    # installer that moves it and the updater that repoints ~/.local/bin/claude are followed without a reinstall.
    # Every hands shim is skipped by its mark, not only this one: two homes' shims on one PATH would otherwise each
    # take the other for the real claude, and fritter would nest without end. fritter is handed the path, never the
    # name, or it would run a shim again. An empty PATH entry is the current directory, as it is to the shell.
    # A session is a terminal on both ends and no print. A pipe, a script, and `claude -p` are not sessions to
    # drive, and on a pty they would not be what they are; they run the real claude, without the address of any
    # session they were started from, so none of them claims a fritter that does not type into it. -c is the one
    # flag that takes no value and leaves a run going, so `-cp` is print too.
    return f"""#!/bin/sh
{MARK}
fritter={shlex.quote(str(fritter))}

set -f
real=
IFS=:
for dir in $PATH; do
  candidate=${{dir:-.}}/claude
  [ -f "$candidate" ] && [ -x "$candidate" ] || continue
  mark=
  {{ read -r _ && read -r mark; }} < "$candidate"
  [ "$mark" = {shlex.quote(MARK)} ] && continue
  real=$candidate
  break
done
unset IFS
if [ -z "$real" ]; then
  echo "claude: nothing on PATH named claude but hands' shims" >&2
  exit 127
fi

session=yes
[ -t 0 ] && [ -t 1 ] || session=no
for arg; do
  case $arg in
    --) break ;;
    --print|-p*|-cp*) session=no ;;
  esac
done
case $session in
  yes) exec "$fritter" -- "$real" "$@" ;;
  no) unset FRITTER_SOCKET; exec "$real" "$@" ;;
esac
"""


def install(home: Home, path: str) -> Installed:
    """Build fritter and write the shim into the home's bin; `path` is the PATH whose `claude` is reported."""
    fritter = home.bin / "fritter"
    shim = home.bin / "claude"
    if not (FRITTER_SOURCE / "go.mod").is_file():
        raise Uninstallable(f"no fritter source at {FRITTER_SOURCE}: hands install-fritter builds it from a checkout of cc-hands")
    try:
        # Built beside its target and renamed over it, as the shim is written: macOS kills a running process whose
        # executable is rewritten in place, and every session under fritter is one.
        home.bin.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=home.bin, prefix=".fritter.") as staging:
            _build_fritter(Path(staging) / "fritter")
            os.replace(Path(staging) / "fritter", fritter)
        replace_whole(shim, shim_script(fritter), 0o755)
    except OSError as error:
        raise Uninstallable(f"cannot install into {home.bin}: {error}") from error
    found = shutil.which("claude", path=path)
    return Installed(shim, fritter, None if found is None else Path(found))


def _build_fritter(target: Path) -> None:
    try:
        built = subprocess.run(["go", "build", "-o", str(target), "."], cwd=FRITTER_SOURCE, capture_output=True, text=True)
    except OSError as error:
        raise Uninstallable(f"cannot run go to build fritter: {error}") from error
    if built.returncode != 0:
        raise Uninstallable(f"go could not build fritter from {FRITTER_SOURCE} ({built.returncode}):\n{built.stderr}")
