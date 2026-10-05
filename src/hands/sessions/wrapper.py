"""The `claude` that starts every interactive session under fritter, and its install.

    hands install-fritter     # copies the fritter hands' package carries, and writes that claude beside it in <hands home>/bin

fritter reaches only the sessions started under it. A `claude` earlier on PATH than the real one, which execs
fritter around it, wraps every session nobody remembered to wrap. Sessions already running when it is installed
stay unwrapped until they end.
"""

import os
import shlex
import shutil
import string
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from itertools import takewhile
from pathlib import Path
from typing import Literal

from hands.core.wire import UPSTREAM
from hands.sessions.files import replace_whole
from hands.sessions.home import Home
from hands.sessions.untap import TRUST, untap_script

# [LAW:one-source-of-truth] the fritter hands' own package carries, put there by hatch_build.py's one build of it: a wheel's
# install and a checkout's editable install alike. It changes only when the package is rebuilt.
PACKAGED = Path(__file__).resolve().parents[1] / "bin" / "fritter"

# Every shim's second line, by which a shim knows another hands shim on PATH for what it is.
MARK = "# A hands claude shim, written whole by `hands install-fritter`: change hands.sessions.wrapper, not this."

# [LAW:one-source-of-truth] the options that make a run print, not a session: the shim's case matches them, and
# run matches them for a claude already running. -c is the one flag that takes no value and leaves a run going,
# so `-cp` is print too.
PRINT = ("--print", "-p*", "-cp*")

# [LAW:one-source-of-truth] a first argument that names a subcommand, as the shim's case and run both match it: a bare
# word of lowercase letters, digits and hyphens, so one that starts a command and has no character that cannot be in
# one. Claude Code dispatches on its first argument, and its help lists only some of what it dispatches on (not
# remote-control, rc, sync, bridge), so no list of its subcommands is complete and the word's shape is the test. Its
# cost: a one-word lowercase opening prompt, `claude review`, is a session to Claude Code and runs as the real claude.
# Only the first argument is read, since an option's value, `--model opus`, has the same shape. The letters are spelled
# out, since a [a-z] range in the shell can follow the locale's collation and take capitals.
STARTS_A_COMMAND = f"[{string.ascii_lowercase}]*"
CANNOT_BE_A_COMMAND = f"*[!{string.ascii_lowercase}{string.digits}-]*"

# Why the shim runs a claude as the real claude and not as a session: not a terminal on both ends, a subcommand first,
# or a print among the options before `--`, tested in that order.
NotASession = Literal["piped", "subcommand", "print"]
type Run = Literal["session"] | NotASession


def run(arguments: Sequence[str], terminal_stdio: bool) -> Run:
    """What the shim runs claude with these arguments as: a session under fritter, or the real claude, and why."""
    first = arguments[0] if arguments else ""
    options = takewhile(lambda argument: argument != "--", arguments)
    if not terminal_stdio:
        return "piped"
    if fnmatchcase(first, STARTS_A_COMMAND) and not fnmatchcase(first, CANNOT_BE_A_COMMAND):
        return "subcommand"
    if any(fnmatchcase(option, pattern) for option in options for pattern in PRINT):
        return "print"
    return "session"


class Uninstallable(Exception):
    """fritter could not be copied, or the claude beside it could not be written. The message says which."""


class Unpackaged(Exception):
    """hands' package carries no fritter: a hands built without it, which only building hands again mends."""


def packaged() -> Path:
    """The fritter hands' package carries, or Unpackaged naming the rebuild.

    [LAW:single-enforcer] the one check that this hands carries its fritter, for everything that runs or copies it: the
    brain and `hands install-fritter` refuse a hands built without it in the same words."""
    if not PACKAGED.is_file():
        raise Unpackaged(
            f"hands' package carries no fritter at {PACKAGED}: install hands again, or in a checkout, `uv sync --reinstall-package hands`"
        )
    return PACKAGED


@dataclass(frozen=True)
class Installed:
    shim: Path
    fritter: Path


@dataclass(frozen=True)
class Shim:
    """A hands shim, this home's or another's, that is the one this hands writes: the fritter it runs and the wire it
    copies API traffic to."""

    fritter: Path
    wire: Path


@dataclass(frozen=True)
class Stale:
    """A hands shim, by its mark, that is not the one this hands writes for any fritter and wire: an older hands wrote it."""


def shim_of(path: Path) -> Shim | Stale | None:
    """What the file at path is as a hands shim; None when it is not one."""
    text = _shim(path)
    if text is None:
        return None
    try:
        words = [shlex.split(line.removeprefix(f"{name}=")) if line.startswith(f"{name}=") else [] for name, line in zip(("fritter", "wire"), text.splitlines()[2:4])]
    except ValueError:  # an unclosed quote: not a line shim_script writes
        words = []
    # [LAW:one-source-of-truth] the shim is shim_script written out for the fritter and wire it names, and `hands check`
    # asks run what that script does, so one an older hands wrote is Stale, never taken to do what this one would.
    match words:
        case [[fritter], [wire]] if text == shim_script(Path(fritter), Path(wire)):
            return Shim(Path(fritter), Path(wire))
        case _:
            return Stale()


def real_claude(search: str) -> Path | None:
    """The first claude on the PATH `search` that is not a hands shim, as the shim itself finds it; None when there is none."""
    # [LAW:one-source-of-truth] the shim's own rule, which lives in shell below: a shim is known by its mark on its
    # second line, and an empty entry is the current directory.
    for entry in search.split(":"):
        candidate = Path(entry or ".") / "claude"
        if candidate.is_file() and os.access(candidate, os.X_OK) and _shim(candidate) is None:
            # Absolute: whoever runs it may run it from another directory than the one an empty entry named.
            return candidate.absolute()
    return None


def _shim(path: Path) -> str | None:
    """The whole text of a hands shim; None when path is not a file marked as a shim."""
    # [LAW:one-source-of-truth] the one reading of the mark in Python. Read as the shim reads each claude on PATH: a file
    # it cannot read is not a shim to it either. Only a marked file is read past its mark: the real claude is large.
    try:
        with path.open("rb") as found:
            head = found.readline() + found.readline()
            if head.splitlines()[1:] != [MARK.encode()]:
                return None
            return (head + found.read()).decode(errors="replace")
    except OSError:
        return None


def shim_script(fritter: Path, wire: Path) -> str:
    """The shim's text: a claude that runs a session under fritter, its API traffic copied to hands at `wire`, and
    anything else as it would run without it."""
    # [LAW:one-source-of-truth] the real claude is looked up on PATH as each run starts, never recorded here, so the
    # installer that moves it and the updater that repoints ~/.local/bin/claude are followed without a reinstall.
    # Every hands shim is skipped by its mark, not only this one: two homes' shims on one PATH would otherwise each
    # take the other for the real claude, and fritter would nest without end. fritter is handed the path, never the
    # name, or it would run a shim again. An empty PATH entry is the current directory, as it is to the shell; the colon
    # added before splitting keeps a trailing one, which splitting on IFS would drop.
    # A session is a terminal on both ends, no subcommand, and no print. A pipe, a script, `claude update`, and
    # `claude -p` are not sessions to drive, and on a pty they would not be what they are; they run the real claude,
    # without the address of any session they were started from, so none of them claims a fritter that does not type
    # into it. run is this same test, asked of a claude already running. A first argument that cannot be a command is
    # tested first: a glob cannot say "only these characters" alone.
    # A session reaches its API as it would without the shim, ANTHROPIC_BASE_URL unchanged, so Claude Code keeps all it
    # keeps for Anthropic's own API: fritter is its proxy instead, and opens only the connections to that API's host. What
    # the tap replaced is given back first, so a claude run from inside a session is tapped once, by its own fritter, and
    # one that outlives the session is not left with a proxy nothing answers.
    return f"""#!/bin/sh
{MARK}
fritter={shlex.quote(str(fritter))}
wire={shlex.quote(str(wire))}

{untap_script()}
set -f
real=
IFS=:
path=$PATH:
for dir in $path; do
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
case ${{1-}} in
  {CANNOT_BE_A_COMMAND}) ;;
  {STARTS_A_COMMAND}) session=no ;;
esac
for arg; do
  case $arg in
    --) break ;;
    {'|'.join(PRINT)}) session=no ;;
  esac
done
case $session in
  yes) exec "$fritter" --tap "${{ANTHROPIC_BASE_URL:-{UPSTREAM}}}" --tap-ca {TRUST} --tap-to "$wire" -- "$real" "$@" ;;
  no) unset FRITTER_SOCKET; exec "$real" "$@" ;;
esac
"""


def install(home: Home) -> Installed:
    """Copy the packaged fritter and write the shim into the home's bin; Unpackaged when the package carries none."""
    fritter = home.fritter
    shim = home.shim
    try:
        # Copied beside its target and renamed over it, as the shim is written: macOS kills a running process whose
        # executable is rewritten in place, and every session under fritter is one.
        home.bin.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=home.bin, prefix=".fritter.") as staging:
            shutil.copy2(packaged(), Path(staging) / "fritter")
            os.replace(Path(staging) / "fritter", fritter)
        replace_whole(shim, shim_script(fritter, home.wire), 0o755)
    except OSError as error:
        raise Uninstallable(f"cannot install {PACKAGED} into {home.bin}: {error}") from error
    return Installed(shim, fritter)
