"""Claude Code, as hands asks it things outside a session (docs/testing.md): its login, its plugins and the marketplaces
they come from, its login and its first run at the person's terminal, and the .claude.json and settings.json it keeps
its answers in.

[LAW:effects-at-boundaries] `ClaudeCode` is the one route to the real `claude` for all of it. `Installed` makes the real
calls and decides nothing about their answers; tests put the fake in tests/claudecode_fake.py, built from recordings of
a real Claude Code (tests/fixtures/claudecode), in its place.
"""

import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from hands.sessions.files import replace_whole
from hands.sessions.payload import Rejected

# The logins `claude auth login` makes, each named by its own flag: a Claude plan, or an Anthropic Console key.
type Method = Literal["claudeai", "console"]

# `claude auth status` and `claude plugin list` answer in about a second; one that has not answered in this long is not going to.
ANSWER_SECONDS = 20.0


@dataclass(frozen=True)
class Instance:
    """One Claude Code as hands runs it: the executable, found on the environment's PATH when it is a bare name, the
    environment, which names its config, and the directory it runs in, hands' own when None."""

    executable: Path | str
    environment: Mapping[str, str]
    cwd: Path | None = None


@dataclass(frozen=True)
class Answer:
    """What a `claude` subcommand printed and exited with."""

    exit: int
    stdout: bytes
    stderr: bytes


class Unreachable(Exception):
    """`claude` could not be run, or did not answer in time; the message says which."""


class ClaudeCode(Protocol):
    """What hands asks of Claude Code outside a session. A call that runs `claude` raises Unreachable when it cannot."""

    def auth_status(self, claude: Instance) -> Answer:
        """`claude auth status`: the login it holds, as JSON, exiting 1 when it holds none."""
        ...

    def plugins(self, claude: Instance) -> Answer:
        """`claude plugin list --json`."""
        ...

    def marketplaces(self, claude: Instance) -> Answer:
        """`claude plugin marketplace list --json`."""
        ...

    def add_marketplace(self, claude: Instance, source: str) -> int:
        """`claude plugin marketplace add`, saying its progress at the person's terminal; its exit."""
        ...

    def install_plugin(self, claude: Instance, plugin: str) -> int:
        """`claude plugin install --scope user`, at the person's terminal, where Claude Code asks whether to run the
        plugin's install command; its exit, 1 when the person declines."""
        ...

    def auth_login(self, claude: Instance, method: Method | None) -> int:
        """`claude auth login`, with the flag for `method`, or none for Claude Code's own default, at the person's
        terminal; its exit."""
        ...

    def first_run(self, claude: Instance, settings: Mapping[str, object] | None) -> int:
        """Claude Code started at the person's terminal, where it asks what it asks only once, until they quit it: on
        the settings it reads itself when `settings` is None, else on the user's settings alone with `settings` over
        them. Its exit, which is 0 for a run quit before its last screen as for one through them all."""
        ...

    def read(self, file: Path) -> bytes | None:
        """A .claude.json's or settings.json's bytes, or None when there is none; raises Rejected when there is one that
        cannot be read."""
        ...

    def replace(self, file: Path, text: str, mode: int) -> None:
        """Writes `text` as the whole of `file`, with `mode`: a reader sees the old file or the new."""
        ...

    def create(self, file: Path, text: str) -> bool:
        """Writes `text` as `file` when there is none; whether it wrote."""
        ...


class Installed:
    """The Claude Code installed on this Mac: each call runs the real `claude`, or reads or writes the real file."""

    def auth_status(self, claude: Instance) -> Answer:
        return _asked(claude, "auth", "status")

    def plugins(self, claude: Instance) -> Answer:
        return _asked(claude, "plugin", "list", "--json")

    def marketplaces(self, claude: Instance) -> Answer:
        return _asked(claude, "plugin", "marketplace", "list", "--json")

    def add_marketplace(self, claude: Instance, source: str) -> int:
        return _attended(claude, "plugin", "marketplace", "add", source)

    def install_plugin(self, claude: Instance, plugin: str) -> int:
        return _attended(claude, "plugin", "install", "--scope", "user", plugin)

    def auth_login(self, claude: Instance, method: Method | None) -> int:
        return _attended(claude, "auth", "login", *(() if method is None else (f"--{method}",)))

    def first_run(self, claude: Instance, settings: Mapping[str, object] | None) -> int:
        return _attended(claude, *(() if settings is None else ("--setting-sources", "user", "--settings", json.dumps(settings))))

    def read(self, file: Path) -> bytes | None:
        try:
            return file.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise Rejected(f"{file} unreadable: {error}") from error

    def replace(self, file: Path, text: str, mode: int) -> None:
        replace_whole(file, text, mode)

    def create(self, file: Path, text: str) -> bool:
        file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with file.open("x") as made:
                made.write(text)
        except FileExistsError:
            return False
        return True


def _argv(claude: Instance, arguments: tuple[str, ...]) -> list[str]:
    return [str(claude.executable), *arguments]


def _asked(claude: Instance, *arguments: str) -> Answer:
    """A subcommand that answers on its output, never at a terminal."""
    said = " ".join(arguments)
    try:
        # A timed-out child is killed and reaped by run itself.
        ran = subprocess.run(_argv(claude, arguments), env=dict(claude.environment), cwd=claude.cwd, stdin=subprocess.DEVNULL, capture_output=True, timeout=ANSWER_SECONDS)
    except subprocess.TimeoutExpired:
        raise Unreachable(f"`claude {said}` did not answer in {ANSWER_SECONDS:.0f}s") from None
    except OSError as error:
        raise Unreachable(f"`claude {said}` could not be run: {error}") from error
    return Answer(ran.returncode, ran.stdout, ran.stderr)


def _attended(claude: Instance, *arguments: str) -> int:
    """A run at the person's terminal, which it reads and writes as Claude Code does."""
    try:
        return subprocess.run(_argv(claude, arguments), env=dict(claude.environment), cwd=claude.cwd).returncode
    except OSError as error:
        raise Unreachable(f"`claude {' '.join(arguments)}` could not be started: {error}") from error
