"""The plugin hands' marketplace entry installs: the directory its command source, `hands plugin`, prints.

Claude Code runs that command at install and again once per session, and copies the directory it prints into its plugin
cache (code.claude.com/docs/en/plugins/marketplace-reference, "command plugin source"), so a session's hooks are always
the installed hands' own. The plugin's files are in hands' package, beside this module. The one thing a package cannot
carry is which interpreter runs it, so the launcher every hook and skill names is written here, for the interpreter
that runs this.
"""

import hashlib
import shlex
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hands.sessions.hookconfig import LAUNCHER
from hands.sessions.home import Home

# The plugin's files as hands' package carries them: everything but its launcher.
PACKAGED = Path(__file__).parent / "plugin"


@dataclass(frozen=True)
class Rendered:
    plugin: Path
    # False when a session before had rendered this same plugin: the directory is named by its content.
    written: bool


def launcher(interpreter: str) -> str:
    """hooks/python: the interpreter given, run on whatever module a hook or skill names."""
    # exec, so the hook is the process Claude Code spawned (hooks.json is exec form, with no shell between), and the
    # shim's parent is the claude whose pid it records. -I keeps the session's directory, where Claude Code runs the
    # hook, and the user's PYTHON* variables off the path, so a project's own json.py or hands/ cannot stand in for
    # hands' modules.
    # [LAW:no-silent-failure] a plugin outlives the interpreter it names when hands is uninstalled or its venv is
    # rebuilt; the hook then says which interpreter is gone and what brings the plugin back, not the shell's exit 127.
    quoted = shlex.quote(interpreter)
    return (
        "#!/bin/sh\n"
        f"[ -x {quoted} ] || {{ echo {shlex.quote(f'hands: the plugin runs {interpreter}, which is gone; install hands, then run `claude plugin update hands@cc-hands`')} >&2; exit 1; }}\n"
        f'exec {quoted} -I "$@"\n'
    )


def render(home: Home, interpreter: str) -> Rendered:
    """Write the plugin, its launcher running `interpreter`, under the home, in a directory named by its content."""
    # [LAW:one-source-of-truth] the plugin's every file by its path within it, the package's and the launcher: what the
    # directory is named by and what is written into it.
    files = {path.relative_to(PACKAGED).as_posix(): path.read_bytes() for path in PACKAGED.rglob("*") if path.is_file()}
    files[LAUNCHER] = launcher(interpreter).encode()
    plugin = home.plugins / digest(files)
    # Claude Code runs this before every session: once a session has written these files, the rest only name them.
    if plugin.is_dir():
        return Rendered(plugin, written=False)
    home.plugins.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".staged-", dir=home.plugins))
    try:
        for name, content in files.items():
            (staged / name).parent.mkdir(parents=True, exist_ok=True)
            (staged / name).write_bytes(content)
        (staged / LAUNCHER).chmod(0o755)
        staged.chmod(0o755)
        # [LAW:no-ambient-temporal-coupling] sessions start together, each running this: a directory is staged whole and
        # renamed into place, so Claude Code never copies one half written, and one already there has these same files.
        try:
            staged.rename(plugin)
        except OSError:
            if not plugin.is_dir():
                raise
            return Rendered(plugin, written=False)
        return Rendered(plugin, written=True)
    finally:
        # Renamed into place, the staged directory is gone; anything else it became, failed or redundant, goes with it.
        if staged.exists():
            shutil.rmtree(staged)


def digest(files: Mapping[str, bytes]) -> str:
    """A name for a directory's files: each one's path within it and its bytes."""
    hashed = hashlib.sha256()
    for name in sorted(files):
        hashed.update(f"{name}\0".encode() + files[name] + b"\0")
    return hashed.hexdigest()[:16]
