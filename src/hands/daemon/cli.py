"""`hands`: run the daemon, ask whether it is up, show it in the menu bar, print their LaunchAgents, install its hooks."""

import argparse
import asyncio
import os
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from hands.daemon import launchd, status
from hands.sessions import audit
from hands.sessions.home import Home, default_home
from hands.sessions.install import install
from hands.sessions.payload import Rejected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hands")
    parser.add_argument("--home", type=Path, default=default_home().root, help="where the socket, sessions, and heartbeat live")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", help="run the daemon in the foreground (launchd runs it this way)")
    commands.add_parser("status", help="say whether the daemon is up, from its heartbeat; exits 0 only when it is")
    commands.add_parser("indicator", help="show the daemon's verdict in the menu bar and post a notification when it stops being up")
    agents = commands.add_parser("launchd", help="print the LaunchAgent property list that keeps the daemon or the indicator up")
    agents.add_argument("agent", choices=sorted(launchd.AGENTS), help="which process the agent keeps up")
    installing = commands.add_parser("install-hooks", help="merge hands' hook entries into a Claude Code settings file; running it again changes nothing")
    installing.add_argument("--settings", type=Path, default=Path.home() / ".claude" / "settings.json", help="the settings file to merge into")
    log = commands.add_parser("log", help="print the newest audit log lines, then each new one as it is written, until Ctrl-C")
    log.add_argument("-n", "--lines", type=int, default=20, help="how many of the newest lines to print first")
    arguments = parser.parse_args(argv)
    home = Home(arguments.home)
    match arguments.command:
        case "run":
            # Read before this run's first heartbeat replaces it.
            after_crash = crashed_before(home)
            heart = status.Heart(home.status, os.getpid(), datetime.now(UTC), status.HEARTBEAT)
            # [LAW:no-ambient-temporal-coupling] the first heartbeat goes out before Pipecat is imported and its
            # models load, seconds of silence in which the file would otherwise still name the process that died.
            heart.beat("starting", None, 0)
            # Imported here, after that heartbeat, and so that `hands status` answers without loading Pipecat.
            from hands.daemon.run import config_from_env, run

            asyncio.run(run(config_from_env(), home, heart, after_crash))
            return 0
        case "status":
            return report(home)
        case "log":
            return tail_log(home, arguments.lines)
        case "install-hooks":
            return install_hooks(arguments.settings, home)
        case "indicator":
            # Imported here so that nothing else in `hands` loads AppKit.
            from hands.daemon.menubar import show

            show(home)
            return 0
        case "launchd":
            sys.stdout.buffer.write(launchd.agent(launchd.AGENTS[arguments.agent], Path(sys.executable), home))
            return 0
        case other:
            raise AssertionError(f"argparse admitted an unknown command {other!r}")


def report(home: Home) -> int:
    now = datetime.now(UTC)
    verdict = status.look(home.status, now)
    match verdict:
        case status.Up():
            out, code = sys.stdout, 0
        case status.NeverRan() | status.Unresponsive() | status.Down() | status.Stopped():
            out, code = sys.stdout, 1
        case status.Unreadable():
            out, code = sys.stderr, 2
    print(status.describe(verdict, now), file=out)
    return code


def install_hooks(settings: Path, home: Home) -> int:
    try:
        installed = install(settings, Path(sys.executable), home)
    except Rejected as error:
        print(f"hands install-hooks: {error}", file=sys.stderr)
        return 2
    # The diff is the output, so it can be read or piped; the one-line verdict goes to stderr beside it.
    sys.stdout.write(installed.diff)
    print(f"hands install-hooks: {'updated' if installed.diff else 'already current'}: {installed.path}", file=sys.stderr)
    return 0


# How often `hands log` looks for new lines.
LOG_POLL_SECONDS = 0.25


def tail_log(home: Home, lines: int) -> int:
    newest, position = audit.tail(home.audit, lines)
    try:
        for line in newest:
            print(line, flush=True)
        for line in audit.follow(home.audit, position, lambda: time.sleep(LOG_POLL_SECONDS)):
            print(line, flush=True)
    except KeyboardInterrupt:
        return 0
    except BrokenPipeError:
        # Piped into head, which has read what it wanted. Python's own flush at exit would raise again into the closed pipe.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    raise AssertionError("following the audit log ends only when interrupted")


def crashed_before(home: Home) -> bool:
    """Whether the last run ended without being stopped: its heartbeat names a pid that is gone, or one that went quiet."""
    now = datetime.now(UTC)
    verdict = status.look(home.status, now)
    if isinstance(verdict, status.Unreadable):
        print(f"hands run: not counted as a crash, since {status.describe(verdict, now)}", file=sys.stderr)
    return isinstance(verdict, status.Down | status.Unresponsive)
