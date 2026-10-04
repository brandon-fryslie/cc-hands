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
    return f'#!/bin/sh\nexec {shlex.quote(interpreter)} -I "$@"\n'


def render(home: Home, interpreter: str) -> Rendered:
    """Write the plugin, its launcher running `interpreter`, under the home, in a directory named by its content."""
    home.plugins.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".staged-", dir=home.plugins))
    shutil.copytree(PACKAGED, staged, dirs_exist_ok=True)
    script = staged / LAUNCHER
    script.write_text(launcher(interpreter), encoding="utf-8")
    script.chmod(0o755)
    staged.chmod(0o755)
    # [LAW:no-ambient-temporal-coupling] sessions start together, each running this: a directory is staged whole and
    # renamed into place, so Claude Code never copies one half written, and one already there has these same files.
    plugin = home.plugins / digest(staged)
    try:
        staged.rename(plugin)
    except OSError:
        if not plugin.is_dir():
            raise
        shutil.rmtree(staged)
        return Rendered(plugin, written=False)
    return Rendered(plugin, written=True)


def digest(root: Path) -> str:
    """A name for the directory's files: each one's path within it and its bytes."""
    hashed = hashlib.sha256()
    for path in sorted(entry for entry in root.rglob("*") if entry.is_file()):
        hashed.update(f"{path.relative_to(root)}\0".encode() + path.read_bytes() + b"\0")
    return hashed.hexdigest()[:16]
