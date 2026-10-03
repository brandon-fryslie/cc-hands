"""`hands`: run the daemon, ask whether it is up, show it in the menu bar, follow what it did."""

import argparse
import asyncio
import os
import sys
import threading
import time
from collections.abc import Callable, Coroutine, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from loguru import logger

from hands.daemon import readiness
from hands.daemon.starting import LAST_BEAT, STOP_SIGNALS, Ended, Ending, again, start
from hands.sessions import audit, heartbeat, wrapper
from hands.sessions.home import Home, default_home
from hands.sessions.payload import Rejected
from hands.threads import off_loop

# The lowest level each module's lines reach the terminal at, by loguru's module prefix: "" is every module not named.
TERMINAL_LEVELS: dict[str | None, str | int | bool] = {"": "WARNING", "hands": "INFO"}  # loguru's FilterDict

if TYPE_CHECKING:
    from loguru import Record

# Every C0 and C1 control, DEL, and bidi embedding, override, and isolate, written as its JSON escape: a line's text
# comes from transcripts, replies, and session names, and a raw ESC, BEL, or BS in it would move the cursor, ring the
# bell, or rewrite the line, and a raw RLO would show it reversed. A line break and a tab are layout, and pass.
VISIBLE = {
    code: f"\\u{code:04x}"
    for code in (*range(0x20), *range(0x7F, 0xA0), *range(0x202A, 0x202F), *range(0x2066, 0x206A))
    if chr(code) not in "\n\t"
}
# loguru's default line, with the message in its visible form.
LINE = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{extra[shown]}</level>\n{exception}"
)


def terminal_line(record: "Record") -> str:
    """LINE, with this record's message made visible: loguru's way to give a format a field of its own is extra."""
    record["extra"]["shown"] = record["message"].translate(VISIBLE)
    return LINE


def to_terminal(stream: TextIO) -> int:
    """[LAW:single-enforcer] the one terminal sink: hands' own lines from INFO, every library's only from WARNING, and
    no message read by the terminal as a control."""
    return logger.add(stream, filter=TERMINAL_LEVELS, format=terminal_line)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hands")
    parser.add_argument("--home", type=Path, help="where the socket, sessions, and heartbeat live (default: HANDS_HOME, or ~/.hands)")
    commands = parser.add_subparsers(dest="command", required=True)
    running = commands.add_parser("run", help="run the daemon in this terminal, with its menu-bar indicator beside it")
    running.add_argument("--indicator", type=int, help="the pid of the menu-bar indicator to keep, which a restart hands on to the run it starts, rather than starting one")
    commands.add_parser("status", help="say whether the daemon is up, from its heartbeat; exits 0 only when it is")
    commands.add_parser("check", help="say whether hands is set up to work here: its plugin, the claude shim on PATH, this terminal's Input Monitoring grant, and the running sessions; exits 0 only when every piece is there, 1 when one is missing, 2 when one could not be looked at")
    indicator = commands.add_parser("indicator", help="show the daemon's verdict in the menu bar, posting a notification when it stops being up, until whatever started it exits (`hands run` starts one)")
    indicator.add_argument("--parent", type=int, help="the pid of the process that started it, whose exit ends it (default: its parent now)")
    commands.add_parser("login", help="log the brain (HANDS_LLM=claude) in again, or onto another account, on the Claude subscription at this terminal; exits 0 only when it is on the subscription after")
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
            # In place of loguru's DEBUG default, so a run's terminal is hands' to read.
            logger.remove()
            to_terminal(sys.stderr)
            # Read before this run's first heartbeat replaces it.
            after_crash = crashed_before(home)
            heart = heartbeat.Heart(home.status, os.getpid(), datetime.now(UTC), heartbeat.HEARTBEAT)
            # [LAW:no-ambient-temporal-coupling] the first heartbeat goes out before Pipecat is imported and its
            # models load, seconds of silence in which the file would otherwise still name the process that died.
            heart.beat("starting", None, 0, False)
            shown = start_indicator(home) if arguments.indicator is None else arguments.indicator
            threading.Thread(target=reap, args=(shown,), name="indicator", daemon=True).start()
            audit_log = audit.AuditLog(home.audit, clock=lambda: datetime.now(UTC))
            match asyncio.run(launch(lambda: loaded(home, heart, audit_log, after_crash, granted), heart)):
                case "quit":
                    return 0
                case "restart":
                    again([sys.executable, "-m", "hands.daemon", "--home", str(home.root), "run", "--indicator", str(shown)], audit_log.record)
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
            # It shares the run's terminal, so its lines reach it through the run's sink.
            logger.remove()
            to_terminal(sys.stderr)
            from hands.daemon.menubar import show

            show(home, parent)
            return 0
        case other:
            raise AssertionError(f"argparse admitted an unknown command {other!r}")


# hands' run, given the event that stops it; it ends saying what it knew last.
type Run = Callable[[asyncio.Event], Coroutine[object, object, Ended]]


async def launch(load: Callable[[], Run], heart: heartbeat.Heart) -> Ending:
    """The run `load` makes, with that load, which imports Pipecat, as the first step of its start; then how it was told to end.

    [LAW:single-enforcer] a SIGTERM, a terminal's Ctrl-C, the terminal closing (SIGHUP), the restart signal, the q key,
    and a failed background task all set this one event, and it is installed before the import, so a stop is heard in
    every phase. The last heartbeat is written here, by the one place that knows whether a restart follows it.
    """
    quit_event = asyncio.Event()
    ending: Ending = "quit"

    def stop(how: Ending) -> None:
        nonlocal ending
        ending = how
        quit_event.set()

    loop = asyncio.get_running_loop()
    for signal_number, how in STOP_SIGNALS.items():
        loop.add_signal_handler(signal_number, stop, how)
    try:
        # No session has joined before the hooks are served, which is after the import.
        run = await start(lambda: off_loop(load, "the Pipecat import"), heart, lambda: 0, quit_event)
        last = Ended(None, 0) if run is None else await run(quit_event)
    finally:
        # From here a signal has its default effect again: nothing is left to stop gracefully.
        for signal_number in STOP_SIGNALS:
            loop.remove_signal_handler(signal_number)
    # Written only by a run told to stop: one that raised leaves its last heartbeat naming a pid that is gone, or, as it
    # restarts, one that stops beating, and neither reads as stopped.
    heart.beat(LAST_BEAT[ending], last.last_audio_out, last.live_sessions, False)
    return ending


def loaded(home: Home, heart: heartbeat.Heart, audit_log: audit.AuditLog, after_crash: bool, granted: bool) -> Run:
    """hands' run, once the seconds it takes to import Pipecat have passed."""
    # Imported here, so that `hands status` answers without loading Pipecat.
    from hands.daemon.run import config_from_env, run

    path = os.environ.get("PATH", "")
    return lambda quit_event: run(lambda: config_from_env(home), lambda: survey(readiness.check(home, path, granted)), home, heart, audit_log, quit_event, after_crash)


def start_indicator(home: Home) -> int:
    """The menu-bar indicator for this run, in a process of its own: AppKit wants a main thread, and this one is the daemon's."""
    # [LAW:single-enforcer] the indicator ends itself once the run that started it is gone (menubar.show), however the
    # run ended; a session of its own keeps the terminal's Ctrl-C and hangup from ending it first, before it has said so.
    # Its output shares this terminal, so an indicator that fails is seen where the daemon's own failures are. A restart
    # keeps the pid, so the indicator carries on into the run after it, which reaps it by that pid.
    argv = [sys.executable, "-m", "hands.daemon", "--home", str(home.root), "indicator", "--parent", str(os.getpid())]
    return os.posix_spawn(sys.executable, argv, os.environ, file_actions=[(os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0)], setsid=True)


def reap(shown: int) -> None:
    """Wait on the indicator, so one that exits early is reaped and said, not left a zombie under the run."""
    # It exits of its own accord only once the run is gone, so an exit this process lives to see is a failure.
    _, status = os.waitpid(shown, 0)
    logger.error(f"the menu-bar indicator exited ({os.waitstatus_to_exitcode(status)}) while hands runs; hands is not shown in the menu bar")


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
    from hands.brain.process import LoginFailed, NotLoggedIn, Unstartable
    from hands.brain.process import login as brain_login
    from hands.core.wire import UPSTREAM

    try:
        account = brain_login(home.brain, UPSTREAM)
    except (LoginFailed, NotLoggedIn, Unstartable) as error:
        print(f"hands login: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("hands login: interrupted", file=sys.stderr)
        return 1
    print(f"the brain at {home.brain} is logged in as {account}")
    print("a hands already running started its brain on the login before: restart it to start the brain on this one")
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
        # json.dumps escapes C0 controls but writes DEL, C1, and bidi controls raw; their escapes keep each line JSON.
        for line in newest:
            print(line.translate(VISIBLE), flush=True)
        for line in audit.follow(home.audit, position, lambda: time.sleep(LOG_POLL_SECONDS)):
            print(line.translate(VISIBLE), flush=True)
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
