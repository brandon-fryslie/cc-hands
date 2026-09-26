"""`hands`: run the daemon, ask whether it is up, show it in the menu bar, print their LaunchAgents."""

import argparse
import asyncio
import os
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from hands.daemon import launchd
from hands.sessions import audit, heartbeat
from hands.sessions.home import Home, default_home
from hands.sessions.payload import Rejected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hands")
    parser.add_argument("--home", type=Path, help="where the socket, sessions, and heartbeat live (default: HANDS_HOME, or ~/.hands)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", help="run the daemon in the foreground (launchd runs it this way)")
    commands.add_parser("status", help="say whether the daemon is up, from its heartbeat; exits 0 only when it is")
    commands.add_parser("indicator", help="show the daemon's verdict in the menu bar and post a notification when it stops being up")
    agents = commands.add_parser("launchd", help="print the LaunchAgent property list that keeps the daemon or the indicator up")
    agents.add_argument("agent", choices=sorted(launchd.AGENTS), help="which process the agent keeps up")
    log = commands.add_parser("log", help="print the newest audit log lines, then each new one as it is written, until Ctrl-C")
    log.add_argument("-n", "--lines", type=int, default=20, help="how many of the newest lines to print first")
    arguments = parser.parse_args(argv)
    try:
        # HANDS_HOME is read only when --home is not given, so a bad one never stands in the way of an explicit home.
        home = default_home() if arguments.home is None else Home(arguments.home)
    except Rejected as error:
        print(f"hands: {error}", file=sys.stderr)
        return 2
    match arguments.command:
        case "run":
            # Read before this run's first heartbeat replaces it.
            after_crash = crashed_before(home)
            heart = heartbeat.Heart(home.status, os.getpid(), datetime.now(UTC), heartbeat.HEARTBEAT)
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
    verdict = heartbeat.look(home.status, now)
    match verdict:
        case heartbeat.Up():
            out, code = sys.stdout, 0
        case heartbeat.NeverRan() | heartbeat.Unresponsive() | heartbeat.Down() | heartbeat.Stopped():
            out, code = sys.stdout, 1
        case heartbeat.Unreadable():
            out, code = sys.stderr, 2
    print(heartbeat.describe(verdict, now), file=out)
    return code


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
    verdict = heartbeat.look(home.status, now)
    if isinstance(verdict, heartbeat.Unreadable):
        print(f"hands run: not counted as a crash, since {heartbeat.describe(verdict, now)}", file=sys.stderr)
    return isinstance(verdict, heartbeat.Down | heartbeat.Unresponsive)
