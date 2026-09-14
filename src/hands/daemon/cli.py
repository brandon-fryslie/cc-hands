"""`hands`: run the daemon, ask whether it is up, print its LaunchAgent."""

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
from hands.sessions.payload import Rejected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hands")
    parser.add_argument("--home", type=Path, default=default_home().root, help="where the socket, sessions, and heartbeat live")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", help="run the daemon in the foreground (launchd runs it this way)")
    commands.add_parser("status", help="say whether the daemon is up, from its heartbeat; exits 0 only when it is")
    commands.add_parser("launchd", help="print the LaunchAgent property list that keeps the daemon up")
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
        case "launchd":
            sys.stdout.buffer.write(launchd.agent(Path(sys.executable), home, path=os.environ.get("PATH", "")))
            return 0
        case other:
            raise AssertionError(f"argparse admitted an unknown command {other!r}")


def report(home: Home) -> int:
    now = datetime.now(UTC)
    try:
        last = status.read(home.status)
    except Rejected as error:
        # [LAW:no-silent-failure] a heartbeat that does not parse is reported, not taken for "not running".
        print(f"hands status cannot read {home.status}: {error}", file=sys.stderr)
        return 2
    verdict = status.judge(home.status, last, now, alive=last is not None and pid_alive(last.pid))
    print(status.describe(verdict, now))
    return 0 if isinstance(verdict, status.Up) else 1


# How often `hands log` looks for new lines.
LOG_POLL_SECONDS = 0.25


def tail_log(home: Home, lines: int) -> int:
    newest, position = audit.tail(home.audit, lines)
    for line in newest:
        print(line, flush=True)
    try:
        for line in audit.follow(home.audit, position, lambda: time.sleep(LOG_POLL_SECONDS)):
            print(line, flush=True)
    except KeyboardInterrupt:
        return 0
    raise AssertionError("following the audit log ends only when interrupted")


def crashed_before(home: Home) -> bool:
    """Whether the last run ended without being stopped: its heartbeat names a pid that is gone, or one that went quiet."""
    try:
        last = status.read(home.status)
    except Rejected as error:
        print(f"hands run: the last heartbeat at {home.status} does not parse, so it is not taken for a crash: {error}", file=sys.stderr)
        return False
    verdict = status.judge(home.status, last, datetime.now(UTC), alive=last is not None and pid_alive(last.pid))
    return isinstance(verdict, status.Down | status.Unresponsive)


def pid_alive(pid: int) -> bool:
    """Liveness from the OS: signal 0 checks that the process exists without touching it."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists; it belongs to someone else
    return True
