"""`hands`: run the daemon, ask whether it is up, show it in the menu bar, follow what it did."""

import argparse
import asyncio
import os
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from hands.daemon import readiness
from hands.sessions import audit, heartbeat, wrapper
from hands.sessions.home import Home, default_home
from hands.sessions.payload import Rejected

# The lowest level each module's lines reach the terminal at, by loguru's module prefix: "" is every module not named.
TERMINAL_LEVELS: dict[str | None, str | int | bool] = {"": "WARNING", "hands": "INFO"}  # loguru's FilterDict


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hands")
    parser.add_argument("--home", type=Path, help="where the socket, sessions, and heartbeat live (default: HANDS_HOME, or ~/.hands)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", help="run the daemon in this terminal, with its menu-bar indicator beside it")
    commands.add_parser("status", help="say whether the daemon is up, from its heartbeat; exits 0 only when it is")
    commands.add_parser("check", help="say whether hands is set up to work here: its plugin, the claude shim on PATH, this terminal's Input Monitoring grant, and the running sessions; exits 0 only when every piece is there, 1 when one is missing, 2 when one could not be looked at")
    shown = commands.add_parser("indicator", help="show the daemon's verdict in the menu bar, posting a notification when it stops being up, until whatever started it exits (`hands run` starts one)")
    shown.add_argument("--parent", type=int, help="the pid of the process that started it, whose exit ends it (default: its parent now)")
    commands.add_parser("login", help="log the brain (HANDS_LLM=claude) in to the Claude subscription, or in again, at this terminal; exits 0 only when it is logged in after")
    commands.add_parser("install-fritter", help="build fritter and write, beside it in <home>/bin, the claude that runs every interactive session under it; exits 0 only when that claude is the one on PATH")
    log = commands.add_parser("log", help="print the newest audit log lines, then each new one as it is written, until Ctrl-C")
    log.add_argument("-n", "--lines", type=int, default=20, help="how many of the newest lines to print first")
    arguments = parser.parse_args(argv)
    try:
        # HANDS_HOME is read only when --home is not given, so a bad one never stands in the way of an explicit home.
        # A relative --home is this directory's, made absolute here, since the home is written into the shim.
        home = default_home() if arguments.home is None else Home(arguments.home.absolute())
    except Rejected as error:
        print(f"hands: {error}", file=sys.stderr)
        return 2
    match arguments.command:
        case "run":
            # Imported here, like AppKit for the indicator, so that no other command loads Quartz.
            from hands.voice import talkkey

            # [LAW:no-silent-failure] no run without its talk key: a missing grant is named at the door, before a
            # heartbeat says starting, and macOS is asked to show the prompt that adds the terminal to the list.
            granted = talkkey.granted()
            if not granted:
                talkkey.ask()
                print(f"hands: {readiness.grant(granted).said}, then run hands again.", file=sys.stderr)
                return 1
            # [LAW:single-enforcer] the one terminal sink, in place of loguru's DEBUG default: hands' own lines from
            # INFO, and Pipecat's and every other library's only from WARNING, so a run's terminal is hands' to read.
            logger.remove()
            logger.add(sys.stderr, filter=TERMINAL_LEVELS)
            # Read before this run's first heartbeat replaces it.
            after_crash = crashed_before(home)
            heart = heartbeat.Heart(home.status, os.getpid(), datetime.now(UTC), heartbeat.HEARTBEAT)
            # [LAW:no-ambient-temporal-coupling] the first heartbeat goes out before Pipecat is imported and its
            # models load, seconds of silence in which the file would otherwise still name the process that died.
            heart.beat("starting", None, 0, False)
            start_indicator(home)
            # Imported here, after that heartbeat, and so that `hands status` answers without loading Pipecat.
            from hands.daemon.run import config_from_env, run

            path = os.environ.get("PATH", "")
            asyncio.run(run(lambda: config_from_env(home), lambda: survey(readiness.check(home, path, granted)), home, heart, after_crash))
            return 0
        case "status":
            return report(home)
        case "check":
            from hands.voice import talkkey

            return check(home, talkkey.granted())
        case "log":
            return tail_log(home, arguments.lines)
        case "login":
            return login(home)
        case "install-fritter":
            return install_fritter(home)
        case "indicator":
            # Imported here so that nothing else in `hands` loads AppKit.
            # [LAW:no-ambient-temporal-coupling] the parent is read before AppKit loads, not after: a parent that exits
            # in that second would leave this process watching its new one, pid 1, forever.
            parent = os.getppid() if arguments.parent is None else arguments.parent
            from hands.daemon.menubar import show

            show(home, parent)
            return 0
        case other:
            raise AssertionError(f"argparse admitted an unknown command {other!r}")


def start_indicator(home: Home) -> None:
    """The menu-bar indicator for this run, in a process of its own: AppKit wants a main thread, and this one is the daemon's."""
    # [LAW:single-enforcer] the indicator ends itself once the run that started it is gone (menubar.show), however the
    # run ended; a session of its own keeps the terminal's Ctrl-C and hangup from ending it first, before it has said so.
    # Its output shares this terminal, so an indicator that fails is seen where the daemon's own failures are.
    shown = subprocess.Popen(
        [sys.executable, "-m", "hands.daemon", "--home", str(home.root), "indicator", "--parent", str(os.getpid())],
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    threading.Thread(target=reap, args=(shown,), name="indicator", daemon=True).start()


def reap(shown: subprocess.Popen[bytes]) -> None:
    """Wait on the indicator, so one that exits early is reaped and said, not left a zombie under the run."""
    # It exits of its own accord only once the run is gone, so an exit this process lives to see is a failure.
    logger.error(f"the menu-bar indicator exited ({shown.wait()}) while hands runs; hands is not shown in the menu bar")


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


def check(home: Home, granted: bool) -> int:
    findings = readiness.check(home, os.environ.get("PATH", ""), granted)
    for finding in findings:
        print(f"{display(finding)[0]:<8} {finding.said}")
    kinds = {type(finding) for finding in findings}
    # A piece known to be missing outranks one that could not be looked at: hands is not set up, whatever that one is.
    return 1 if readiness.Missing in kinds else 2 if readiness.Unknown in kinds else 0


def survey(findings: Sequence[readiness.Finding]) -> None:
    """Say each finding as a run starts: [LAW:nothing-unseen] an up daemon says nothing of what it cannot reach."""
    for finding in findings:
        logger.log(display(finding)[1], finding.said)


def display(finding: readiness.Finding) -> tuple[str, str]:
    """How `hands check` marks a finding, and the level `hands run` says it at."""
    match finding:
        case readiness.Ready():
            return "ok", "INFO"
        case readiness.Missing():
            return "missing", "WARNING"
        case readiness.Unknown():
            return "unknown", "WARNING"


def login(home: Home) -> int:
    # Imported here, so that no other command loads the brain's process and its aiohttp.
    from hands.brain.process import NotLoggedIn
    from hands.brain.process import login as brain_login
    from hands.core.wire import UPSTREAM

    try:
        account = brain_login(home.brain, UPSTREAM)
    except NotLoggedIn as error:
        print(f"hands login: {error}", file=sys.stderr)
        return 1
    print(f"the brain at {home.brain} is logged in as {account}")
    return 0


def install_fritter(home: Home) -> int:
    try:
        installed = wrapper.install(home)
    except wrapper.Uninstallable as error:
        print(f"hands install-fritter: {error}", file=sys.stderr)
        return 1
    print(f"built {installed.fritter}")
    print(f"wrote {installed.shim}")
    match readiness.shim(home, os.environ.get("PATH", "")):
        case readiness.Ready(said=said):
            print(said)
            return 0
        case readiness.Missing(said=said):
            print(said, file=sys.stderr)
            return 1


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
