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
from dataclasses import dataclass
from pathlib import Path

from hands.sessions.home import Home

# fritter's Go module, in the checkout this hands runs from.
FRITTER_SOURCE = Path(__file__).resolve().parents[3] / "fritter"


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


def shim_script(shim: Path, fritter: Path) -> str:
    """The shim's text: a claude that runs a session under fritter and anything else as it would run without it."""
    # [LAW:one-source-of-truth] the real claude is looked up on PATH as each run starts, never recorded here, so the
    # installer that moves it and the updater that repoints ~/.local/bin/claude are followed without a reinstall.
    # fritter is handed the real claude's path, never the name, or it would run this file again.
    # A session is a terminal on both ends and no --print. A pipe, a script, and `claude -p` are not sessions to
    # drive, and on a pty they would not be what they are; they run the real claude, without the address of any
    # session they were started from, so none of them claims a fritter that does not type into it.
    return f"""#!/bin/sh
# Written by `hands install-fritter`, which rewrites it whole: change hands.sessions.wrapper, not this.
shim={shlex.quote(str(shim))}
fritter={shlex.quote(str(fritter))}

set -f
real=
IFS=:
for dir in $PATH; do
  if [ -f "$dir/claude" ] && [ -x "$dir/claude" ] && [ ! "$dir/claude" -ef "$shim" ]; then
    real=$dir/claude
    break
  fi
done
unset IFS
if [ -z "$real" ]; then
  echo "claude: nothing on PATH named claude but hands' shim, $shim" >&2
  exit 127
fi

session=yes
[ -t 0 ] && [ -t 1 ] || session=no
for arg; do
  case $arg in
    --) break ;;
    -p|--print) session=no ;;
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
    try:
        home.bin.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise Uninstallable(f"cannot make {home.bin}: {error}") from error
    _build_fritter(fritter)
    try:
        _replace(shim, shim_script(shim, fritter).encode())
    except OSError as error:
        raise Uninstallable(f"cannot write {shim}: {error}") from error
    found = shutil.which("claude", path=path)
    return Installed(shim, fritter, None if found is None else Path(found))


def _build_fritter(target: Path) -> None:
    partial = _partial(target)
    try:
        built = subprocess.run(["go", "build", "-o", str(partial), "."], cwd=FRITTER_SOURCE, capture_output=True, text=True)
    except OSError as error:
        raise Uninstallable(f"cannot run go to build fritter from {FRITTER_SOURCE}: {error}") from error
    if built.returncode != 0:
        raise Uninstallable(f"go could not build fritter from {FRITTER_SOURCE} ({built.returncode}):\n{built.stderr}")
    os.replace(partial, target)


def _replace(target: Path, content: bytes) -> None:
    partial = _partial(target)
    partial.write_bytes(content)
    partial.chmod(0o755)
    os.replace(partial, target)


def _partial(target: Path) -> Path:
    # Written beside the target and renamed over it, never rewritten in place: macOS kills a running process whose
    # executable's pages change under it, and sh reads a script as it runs it, so every session running fritter
    # and every shim mid-exec keeps the file it opened.
    return target.with_name(f".{target.name}.{os.getpid()}")
