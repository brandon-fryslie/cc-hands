"""`hands`: run the daemon, ask whether it is up, show it in the menu bar, follow what it did."""

import argparse
import asyncio
import os
import sys
import threading
import time
from collections.abc import Callable, Coroutine, Sequence
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from loguru import logger

from hands.daemon import readiness
from hands.daemon.config import Config, Settings, edited, load
from hands.daemon.starting import LAST_BEAT, STOP_SIGNALS, CannotStart, Ended, Ending, again, invocation, refuse, start
from hands.sessions import audit, heartbeat, marketplace, recall, wide, wrapper
from hands.sessions.home import Home, default_home
from hands.sessions.otlp import exporting
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


def show_phone(home: Home) -> int:
    """Print every address the phone's page opens at, and the first as a QR code a phone's camera opens."""
    import segno

    from hands.voice.phonepage import Untailed, lan_addresses, page_urls, phone_key, tailnet_name

    # The name alone: the certificate is the daemon's to ask Tailscale for, as it serves the page.
    name = asyncio.run(tailnet_name())
    match name:
        case Untailed(reason=reason):
            print(f"hands: no tailnet address, since {reason}; the LAN's alone:", file=sys.stderr)
        case str():
            pass
    try:
        key = phone_key(home)
    except Rejected as error:
        print(f"hands: {error}", file=sys.stderr)
        return 1
    urls = page_urls(name, lan_addresses(), key)
    if not urls:
        print("hands: this machine has no address a phone can reach.", file=sys.stderr)
        return 1
    segno.make(urls[0]).terminal(compact=True)
    print("\n".join(urls))
    print("The page is served while `hands run` is up. Anyone with one of these addresses can talk to hands: keep them to yourself.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hands")
    parser.add_argument("--version", action="version", version=f"hands {version('hands')}", help="print the version of hands installed, which is its release's tag, and exit")
    parser.add_argument("--home", type=Path, help="where the socket, sessions, and heartbeat live (default: HANDS_HOME, or ~/.hands)")
    commands = parser.add_subparsers(dest="command", required=True)
    running = commands.add_parser("run", help="run the daemon in this terminal, with its menu-bar indicator beside it")
    running.add_argument("--restarted", type=int, metavar="INDICATOR_PID", help="this run is a restart, which only hands passes: it is no crash, and the menu-bar indicator INDICATOR_PID, if it is still running, is kept rather than another started")
    commands.add_parser("status", help="say whether the daemon is up, from its heartbeat; exits 0 only when it is")
    commands.add_parser("check", help="say whether hands is set up to work here: its plugin, the claude shim on PATH, this terminal's Input Monitoring grant, and the running sessions; exits 0 only when every piece is there, 1 when one is missing, 2 when one could not be looked at")
    indicator = commands.add_parser("indicator", help="show the daemon's verdict in the menu bar, posting a notification when it stops being up, until whatever started it exits (`hands run` starts one)")
    indicator.add_argument("--parent", type=int, help="the pid of the process that started it, whose exit ends it (default: its parent now)")
    commands.add_parser("login", help="set the brain (the claude backend of the home's config.toml) up on a home with none, or log it in again or onto another account, on the Claude subscription at this terminal; exits 0 only when it is on the subscription after")
    commands.add_parser("install-fritter", help="copy the fritter hands' package carries and write, beside it in <home>/bin, the claude that runs every interactive session under it; exits 0 only when that claude is the one on PATH")
    commands.add_parser("plugin", help="write hands' Claude Code plugin, its hooks and skills run by this hands' Python, and print its directory: the command hands' marketplace entry has Claude Code run, at install and once per session")
    commands.add_parser("phone", help="print the addresses a phone opens hands' talk page at, the tailnet's first as a QR code, each carrying the phone's key")
    log = commands.add_parser("log", help="print the newest audit log lines, then each new one as it is written, until Ctrl-C")
    log.add_argument("-n", "--lines", type=int, default=20, help="how many of the newest lines to print first")
    recalling = commands.add_parser("recall", help="print what was said, sent to a session, and answered for one, oldest first, from the audit log: the newest moments that hold every word given, or the newest of all with none")
    recalling.add_argument("words", nargs="*", help="words every moment printed holds, in any case")
    recalling.add_argument("-n", "--most", type=int, default=20, help="how many of the newest matching moments to print")
    arguments = parser.parse_args(argv)
    try:
        # HANDS_HOME is read only when --home is not given, so a bad one never stands in the way of an explicit home.
        # A relative --home is this directory's, made absolute here, since the home is written into the shim.
        home = default_home(os.environ) if arguments.home is None else Home(arguments.home.absolute())
    except Rejected as error:
        print(f"hands: {error}", file=sys.stderr)
        return 2
    match arguments.command:
        case "run":
            # Read before this run's first heartbeat replaces it. A restart's run before it was told to stop, which is
            # no crash, however long the start took that its last heartbeat may read as gone quiet.
            after_crash = arguments.restarted is None and crashed_before(home)
            heart = heartbeat.Heart(home.status, os.getpid(), datetime.now(UTC), heartbeat.HEARTBEAT)
            audit_log = audit_log_of(home)
            # Refused at the door, a run holds no heartbeat yet, and leaves the one there to what wrote it: a running hands,
            # or a crash the next run must read. A restart's run holds it already: the run before it beat starting under
            # this pid, which it keeps.
            try:
                settings = door(home)
            except CannotStart as cannot:
                refuse(cannot, None if arguments.restarted is None else heart, audit_log.record)
                return 1
            try:
                ending, shown = run_here(home, arguments.restarted, after_crash, settings, heart, audit_log)
            except CannotStart as cannot:
                refuse(cannot, heart, audit_log.record)
                return 1
            match ending:
                case "quit":
                    return 0
                case "restart":
                    again(invocation(home, "run", "--restarted", str(shown)), audit_log.record)
        case "status":
            return report(home)
        case "check":
            from hands.voice import talkkey

            return check(home, talkkey.granted())
        case "log":
            return tail_log(home, arguments.lines)
        case "recall":
            return recall_moments(home, arguments.words, arguments.most)
        case "phone":
            return show_phone(home)
        case "login":
            return login(home)
        case "install-fritter":
            return install_fritter(home)
        case "plugin":
            return render_plugin(home)
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


def door(home: Home) -> Settings:
    """What a run checks before its first heartbeat, each in a moment: the talk key's grant, and the settings it starts
    on. CannotStart where either is missing."""
    # Imported here, like AppKit for the indicator, so that no other command loads Quartz.
    from hands.voice import talkkey

    # [LAW:no-silent-failure] no run without its talk key: a missing grant is named at the door, and macOS is asked to
    # show the prompt that adds the terminal to the list.
    granted = talkkey.granted()
    if not granted:
        talkkey.ask()
        raise CannotStart(f"{readiness.grant(granted).said}, then run hands again.")
    # [LAW:single-enforcer] the one read of the settings a run starts on: the export edge, the run, and the watch for
    # an edit to them all take these.
    try:
        return load(home)
    except Rejected as error:
        raise CannotStart(str(error)) from error


def run_here(home: Home, restarted: int | None, after_crash: bool, settings: Settings, heart: heartbeat.Heart, audit_log: audit.AuditLog) -> tuple[Ending, int]:
    """hands run in this process until it is told to stop: how it was, and the pid of the menu-bar indicator beside it.
    Raises CannotStart where it cannot start, once its heartbeat says starting."""
    # In place of loguru's DEBUG default, so a run's terminal is hands' to read.
    logger.remove()
    to_terminal(sys.stderr)
    # [LAW:no-ambient-temporal-coupling] the first heartbeat goes out before Pipecat is imported and its models load,
    # seconds of silence in which the file would otherwise still name the process that died.
    heart.beat("starting", None, 0, listening=False, deaf=False)
    kept = None if restarted is None else still_shown(restarted)
    shown = start_indicator(home) if kept is None else kept
    threading.Thread(target=reap, args=(shown,), name="indicator", daemon=True).start()
    with exporting(settings.config.collector, audit_log.record) as record:
        ending = asyncio.run(launch(lambda: loaded(home, settings, heart, record, after_crash), heart, lambda: edited(home, record, partial(reachable, home), settings), record))
    return ending, shown


# hands' run, given the event that stops it; it ends saying what it knew last.
type Run = Callable[[asyncio.Event], Coroutine[object, object, Ended]]


async def launch(
    load: Callable[[], Run], heart: heartbeat.Heart, edited: Callable[[], Coroutine[object, object, audit.SettingsEdited]], record: audit.Record
) -> Ending:
    """The run `load` makes, with that load, which imports Pipecat, as the first step of its start; then how it was told to end.

    [LAW:single-enforcer] a SIGTERM, a terminal's Ctrl-C, the terminal closing (SIGHUP), the restart signal, the q key,
    a failed background task, and an edit to the settings (`edited` returning one that parses) all set this one event, and it is
    installed before the import, so a stop is heard in every phase. The last heartbeat is written here, by the one place that knows whether a restart follows it.
    """
    quit_event = asyncio.Event()
    ending: Ending = "quit"
    failed: list[BaseException] = []

    def stop(how: Ending) -> None:
        # The first stop says how the run ends, the q key's and a failed task's included, which set the event alone:
        # a restart asked while a quit winds the run down is not one.
        nonlocal ending
        ending = ending if quit_event.is_set() else how
        quit_event.set()

    def heard(watch: asyncio.Task[audit.SettingsEdited]) -> None:
        if watch.cancelled():
            return
        if (error := watch.exception()) is not None:
            # [LAW:no-silent-failure] a run that cannot weigh its settings would run on settings that are not the
            # file's: it ends raising, as a run whose background task failed does.
            failed.append(error)
            stop("quit")
            return
        # A run already ending says nothing of an edit it does not restart on; the next start reads the file as edited.
        if quit_event.is_set():
            return
        # The settings take effect as everything else on disk does: in a run started again on them.
        record(watch.result())
        stop("restart")

    watching = asyncio.create_task(edited())
    watching.add_done_callback(heard)
    loop = asyncio.get_running_loop()
    for signal_number, how in STOP_SIGNALS.items():
        loop.add_signal_handler(signal_number, stop, how)
    try:
        # No session has joined before the hooks are served, which is after the import.
        run = await start(lambda: off_loop(load, "the Pipecat import"), heart, lambda: 0, quit_event)
        last = Ended(None, 0) if run is None else await run(quit_event)
        if failed:
            raise failed[0]
        # Written only by a run told to stop: a refused start's last is refuse's, and any other run that raised leaves
        # its last heartbeat naming a pid that is gone, or, as it restarts, one that stops beating, and neither reads as stopped. [LAW:no-ambient-temporal-coupling] it
        # goes out while the handlers are in, so a restart is never asked once they are out: the heartbeat no longer
        # says running, and one asked before it reads that is heard by stop, where the first stop already decided.
        heart.beat(LAST_BEAT[ending], last.last_audio_out, last.live_sessions, listening=False, deaf=False)
    finally:
        watching.cancel()
        # From here a signal has its default effect again: nothing is left to stop gracefully.
        for signal_number in STOP_SIGNALS:
            loop.remove_signal_handler(signal_number)
    return ending


def reachable(home: Home, settings: Config) -> None:
    """Raises Rejected where a start on `settings` could not reach its model: the start's own check, made before the
    restart an edit asks for, so an edit naming a key or a login hands lacks is refused and outlived, not restarted on."""
    # Imported here, as in loaded, so that `hands status` answers without loading Pipecat; an edit weighed while the
    # start imports it waits on that import.
    from hands.daemon.run import backend

    backend(settings.llm, home, os.environ)


def loaded(home: Home, settings: Settings, heart: heartbeat.Heart, record: audit.Record, after_crash: bool) -> Run:
    """hands' run, once the seconds it takes to import Pipecat have passed."""
    # Imported here, so that `hands status` answers without loading Pipecat.
    from hands.daemon.run import configured_from, run

    path = os.environ.get("PATH", "")
    return lambda quit_event: run(lambda environment: configured_from(home, settings, environment), lambda: survey(readiness.check(home, path, granted=True)), home, heart, record, quit_event, after_crash, os.environ)


def start_indicator(home: Home) -> int:
    """The menu-bar indicator for this run, in a process of its own: AppKit wants a main thread, and this one is the daemon's."""
    # [LAW:single-enforcer] the indicator ends itself once the run that started it is gone (menubar.show), however the
    # run ended; a session of its own keeps the terminal's Ctrl-C and hangup from ending it first, before it has said so.
    # Its output shares this terminal, so an indicator that fails is seen where the daemon's own failures are. A restart
    # keeps the pid, so the indicator carries on into the run after it, which reaps it by that pid.
    argv = invocation(home, "indicator", "--parent", str(os.getpid()))
    return os.posix_spawn(sys.executable, argv, os.environ, file_actions=[(os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0)], setsid=True)


def still_shown(indicator: int) -> int | None:
    """The indicator a restart handed on, while it runs; one that exited is reaped here, or was by the run before."""
    try:
        exited, _ = os.waitpid(indicator, os.WNOHANG)
    except ChildProcessError:
        return None
    return indicator if exited == 0 else None


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
        case heartbeat.NeverRan() | heartbeat.Unresponsive() | heartbeat.Down() | heartbeat.Stopped() | heartbeat.Refused():
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


def audit_log_of(home: Home) -> audit.AuditLog:
    """The audit log of `home`, on the wall clock: every command's events, and a run's, land in the one log."""
    return audit.AuditLog(home.audit, clock=lambda: datetime.now(UTC))


def login(home: Home) -> int:
    # Imported here, so that no other command loads the brain's process and its aiohttp.
    from hands.brain.process import LoginFailed, NotLoggedIn, Unstartable, starting_settings
    from hands.brain.process import login as brain_login
    from hands.core.wire import UPSTREAM

    audit_log = audit_log_of(home)
    # [LAW:nothing-unseen] a login is a unit of work: whether it wrote the brain's settings, whether it took Claude Code's
    # first run, and the account it ended on. Recorded in the audit log alone, as the plugin's render is: logging in never
    # waits on a collector, or on a config.toml the daemon has yet to accept.
    with wide.unit("brain.login", audit_log.record):
        try:
            # Before any run of Claude Code on this home, so that none ever syncs the account's skills or plugins.
            wide.annotate(settings_written=starting_settings(home.brain))
            signed = brain_login(home.brain, UPSTREAM, os.environ)
        except (LoginFailed, NotLoggedIn, Unstartable, OSError) as error:
            wide.fail(str(error))
            print(f"hands login: {error}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            wide.fail("interrupted")
            print("hands login: interrupted", file=sys.stderr)
            return 1
        wide.annotate(first_run=signed.first_run, account=signed.account)
    print(f"the brain at {home.brain} is logged in as {signed.account}")
    print("a hands already running started its brain on the login before: restart it to start the brain on this one")
    return 0


def install_fritter(home: Home) -> int:
    try:
        settings = load(home)
    except Rejected as error:
        print(f"hands: {error}", file=sys.stderr)
        return 1
    audit_log = audit_log_of(home)
    # [LAW:nothing-unseen] an install is a unit of work: the fritter it copied from, where it put it and the claude
    # beside it, and whether PATH finds that claude, through the same export edge as the run's events.
    with exporting(settings.config.collector, audit_log.record) as record, wide.unit("fritter.install", record):
        wide.annotate(packaged=wrapper.PACKAGED)
        try:
            installed = wrapper.install(home)
        except wrapper.Uninstallable as error:
            wide.fail(str(error))
            print(f"hands install-fritter: {error}", file=sys.stderr)
            return 1
        wide.annotate(fritter=installed.fritter, shim=installed.shim)
        print(f"copied {wrapper.PACKAGED} to {installed.fritter}")
        print(f"wrote {installed.shim}")
        found = readiness.shim(home, os.environ.get("PATH", ""))
        wide.annotate(path_finds_it=isinstance(found, readiness.Ready))
        match found:
            case readiness.Ready(said=said):
                print(said)
                return 0
            case readiness.Missing(said=said) | readiness.Unknown(said=said):
                print(said, file=sys.stderr)
                return 1


def render_plugin(home: Home) -> int:
    audit_log = audit_log_of(home)
    # [LAW:nothing-unseen] Claude Code runs this once per session: the interpreter the hooks run on, the plugin it
    # printed, and whether that plugin was written now or a session before had. Recorded in the audit log alone: Claude
    # Code waits for this command to exit before the session starts, so, like the shim, it waits on no collector and
    # reads no config.toml, whose rejection is the daemon's to report and never costs a session its hooks.
    with wide.unit("plugin.render", audit_log.record):
        wide.annotate(interpreter=sys.executable, packaged=marketplace.PACKAGED)
        rendered = marketplace.render(home, sys.executable)
        wide.annotate(plugin=rendered.plugin, written=rendered.written)
    # [LAW:no-silent-failure] stdout is the path alone, which Claude Code takes as the plugin's directory.
    print(rendered.plugin)
    return 0


# How often `hands log` looks for new lines.
LOG_POLL_SECONDS = 0.25


def tail_log(home: Home, lines: int) -> int:
    newest, offset = audit.tail(home.audit, lines)
    try:
        # json.dumps escapes C0 controls but writes DEL, C1, and bidi controls raw; their escapes keep each line JSON.
        for line in newest:
            print(line.translate(VISIBLE), flush=True)
        for line in audit.follow(home.audit, offset, lambda: time.sleep(LOG_POLL_SECONDS)):
            print(line.translate(VISIBLE), flush=True)
    except KeyboardInterrupt:
        return 0
    except BrokenPipeError:
        # Piped into head, which has read what it wanted. Python's own flush at exit would raise again into the closed pipe.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    raise AssertionError("following the audit log ends only when interrupted")


def recall_moments(home: Home, words: Sequence[str], most: int) -> int:
    """Print the moments `hands recall` found, one a line, each at the time it happened here."""
    try:
        settings = load(home)
    except Rejected as error:
        print(f"hands: {error}", file=sys.stderr)
        return 1
    audit_log = audit_log_of(home)
    # [LAW:nothing-unseen] a recall is a unit of work: what it was asked, how much of the log it read, and what it found,
    # zeros included, through the same export edge as the run's events.
    with exporting(settings.config.collector, audit_log.record) as record, wide.unit("memory.recall", record, ("lines", "unreadable", "moments", "matched", "printed")):
        wide.annotate(words=tuple(words), most=most)
        found = recall.recall(home.audit, words, most)
        wide.annotate(since=found.since)
        wide.count(lines=found.lines, unreadable=found.unreadable, moments=found.found, matched=found.matched, printed=len(found.moments))
        # Retention keeps the log by size, so how far back it reaches is said, and an empty answer is bounded by it.
        print("The log is empty." if found.since is None else f"The log reaches back to {found.since.astimezone():%a %d %b %H:%M}.")
        for moment in found.moments:
            # One line a moment, so a reader can grep it again; the time is this Mac's, as the user says it.
            print(f"{moment.at.astimezone():%a %d %b %H:%M} {moment.heading}: {' '.join(moment.text.split())}".translate(VISIBLE))
    return 0


def crashed_before(home: Home) -> bool:
    """Whether the last run ended without being stopped: its heartbeat names a pid that is gone, or one that went quiet."""
    now = datetime.now(UTC)
    verdict = heartbeat.look(home.status, now)
    if isinstance(verdict, heartbeat.Unreadable):
        print(f"hands run: not counted as a crash, since {heartbeat.describe(verdict, now)}", file=sys.stderr)
    return isinstance(verdict, heartbeat.Down | heartbeat.Unresponsive)
