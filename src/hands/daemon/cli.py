"""`hands`: run the daemon, ask whether it is up, show it in the menu bar and a tmux status line, follow what it did."""

import argparse
import asyncio
import fcntl
import os
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
from collections.abc import Callable, Coroutine, Sequence
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TextIO, cast

from loguru import logger

from hands.daemon import indicator, readiness
from hands.daemon.backend import backend
from hands.daemon.config import ANTHROPIC_MODEL, Config, OwnModel, Settings, edited, load
from hands.daemon.restart import LOOK_SECONDS, NotBack, NotRunning, Restarted, restart, said
from hands.daemon.starting import LAST_BEAT, STOP_SIGNALS, CannotStart, Ended, Ending, Start, again, invocation, refuse, start
from hands.core.tmux import Keyboard
from hands.sessions import audit, heartbeat, marketplace, recall, tmux, wide, wrapper
from hands.sessions.home import Home, default_home
from hands.sessions.hookconfig import MARKETPLACE, PLUGIN_ID
from hands.sessions.otlp import Exports, exporting
from hands.sessions.payload import Rejected
from hands.threads import off_loop


if TYPE_CHECKING:
    from loguru import Message, Record

    from hands.brain.process import Method

# Every C0 and C1 control, DEL, the line and paragraph separators, and bidi embedding, override, and isolate, written as
# its JSON escape: a line's text
# comes from transcripts, replies, and session names, and a raw ESC, BEL, or BS in it would move the cursor, ring the
# bell, or rewrite the line, a raw RLO would show it reversed, and a raw U+2028 is a line's end to some readers. A line
# break and a tab are layout, and pass.
VISIBLE = {
    code: f"\\u{code:04x}"
    for code in (*range(0x20), *range(0x7F, 0xA0), *range(0x2028, 0x202F), *range(0x2066, 0x206A))
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


def on_terminal(record: "Record") -> bool:
    """[LAW:single-enforcer] which records the terminal shows: hands' own from INFO, every library's only from WARNING."""
    name = record["name"] or ""
    floor = "INFO" if name == "hands" or name.startswith("hands.") else "WARNING"
    return record["level"].no >= logger.level(floor).no


def to_terminal(stream: TextIO) -> tuple[int, int]:
    """[LAW:single-enforcer] the one terminal sink: the records on_terminal admits, nothing in them read by the terminal
    as a control. Its two loguru sinks, each admitting the records the other does not.

    A record with no exception is written as loguru colours it. One with an exception is written whole with every
    control made visible, so it is not coloured: loguru fills `{exception}` after the format runs, with the exception's
    own text among its colours, and only the written line holds both, where they cannot be told apart. Its backtrace
    and its diagnosis are loguru's own.
    """

    def traced(message: "Message") -> None:
        stream.write(message.translate(VISIBLE))
        stream.flush()

    # Routed by filter, not format: a sink's filter is the one thing loguru asks before it formats a record, its
    # exception included, and the one thing it asks of a raw record too.
    lines = logger.add(stream, filter=lambda record: on_terminal(record) and not record["exception"], format=terminal_line)
    traces = logger.add(traced, filter=lambda record: on_terminal(record) and bool(record["exception"]), format=terminal_line, colorize=False)
    return lines, traces


def said_failed(record: audit.Record) -> audit.Record:
    """`record`, and each unit of work it records that said it failed, rather than raised, also said on the terminal, by
    its name and its error.

    [LAW:nothing-unseen] the one layer every event of a run passes through: a failure a unit reports on its event, rather
    than logs, still reaches the terminal its run is read on, as a logged error does. A unit that failed by raising has a
    trace, and [LAW:one-source-of-truth] is said by what catches the exception, once, never again by each unit it passed.
    """

    def both(entry: audit.Entry) -> None:
        record(entry)
        match entry:
            case wide.WideEvent(outcome="failed", trace=(), event=event, error=error):
                # A part timed elsewhere, as a tool call whose result was an error, can fail with no error to say.
                audit.said(f"{event} failed" if error is None else f"{event} failed: {error}")
            case _:
                pass

    return both


def show_phone(home: Home) -> int:
    """Print every address the phone's page opens at, and the first as a QR code a phone's camera opens."""
    import segno

    from hands.voice.phoneaddress import Untailed, lan_addresses, page_urls, phone_key, tailnet_name

    # The name alone: the certificate is the daemon's to ask Tailscale for, as it serves the page.
    name = asyncio.run(tailnet_name())
    match name:
        case Untailed(reason=reason):
            wide.annotate(untailed=reason)
            print(f"hands: no tailnet address, since {reason}; the LAN's alone:", file=sys.stderr)
        case str():
            pass
    try:
        key = phone_key(home)
    except Rejected as error:
        wide.annotate(phone_key=str(error))
        print(f"hands: {error}", file=sys.stderr)
        return 1
    urls = page_urls(name, lan_addresses(), key)
    wide.annotate(addresses=len(urls))
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
    running.add_argument("--restarted", type=int, metavar="INDICATOR_PID", help="this run is a restart, which only hands passes: it is no crash, and the menu-bar indicator INDICATOR_PID the run before showed is ended for one this run starts")
    running.add_argument("--model", type=model_id, help="the model to run on in place of the one config.toml names, kept across every restart of this run; a model chosen by voice is refused while it holds")
    commands.add_parser("status", help="say whether the daemon is up, from its heartbeat; exits 0 only when it is")
    commands.add_parser("check", help="say of each step of the README's install whether it is done here: Claude Code, PortAudio, `hands` on PATH, the claude shim on PATH, the plugin, the brain's login, the Input Monitoring grant of the app it runs in, hands running, and the running sessions; exits 0 only when every step is done, 1 when one is missing, 2 when one could not be looked at")
    showing = commands.add_parser("indicator", help="show the daemon's verdict in the menu bar, posting a notification when it stops being up, until whatever started it exits (`hands run` starts one)")
    showing.add_argument("--parent", type=int, help="the pid of the process that started it, whose exit ends it (default: its parent now)")
    commands.add_parser("tmux-status", help="print the menu bar's title for a tmux status line, coloured by the daemon's verdict; always exits 0, since tmux shows what is printed whatever the exit")
    logging_in = commands.add_parser("login", help="set the brain up on a home with none, or log it in again or onto another account, at this terminal; exits 0 only when it holds the login asked for after")
    logging_in.add_argument("--console", action="store_const", const="console", default="claudeai", dest="method", help="log the brain in with an Anthropic Console key, billed to the API, rather than a Claude plan; on a home's first run, pick it on Claude Code's own login screen")
    commands.add_parser("install-fritter", help="copy the fritter hands' package carries and write, beside it in <home>/bin, the claude that runs every interactive session under it; exits 0 only when that claude is the one on PATH")
    commands.add_parser("install-plugin", help=f"install hands' Claude Code plugin, {PLUGIN_ID}, for every session, at this terminal: Claude Code shows the command `hands plugin` and asks the person to accept it, which is said before it asks; exits 0 only when the plugin is installed and enabled, asking nothing when it already is")
    commands.add_parser("plugin", help="write hands' Claude Code plugin, its hooks and skills run by this hands' Python, and print its directory: the command hands' marketplace entry has Claude Code run, at install and once per session")
    commands.add_parser("smoke", help="take three spoken turns through the hands that is running, as a call from the phone's page, with a session started by the `claude` on PATH, and say of each part of the pipeline whether it did its share: heard, answered, spoken, typed into the session, the session's answer told back aloud; exits 0 only when every part did")
    commands.add_parser("restart", help="start the running daemon again, in the same process, on the code, prompt, and brain setup on disk now, and wait until its pipeline is running; exits 0 only when it is (the plugin's /hands:restart runs this)")
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
            return run_daemon(home, arguments.restarted, arguments.model)
        case "tmux-status":
            return show_segment(home)
        case _:
            return commanded(home, arguments)


def model_id(text: str) -> str:
    """[LAW:parse-dont-validate] --model's value as the id it names, which a blank names none of."""
    if not (model := text.strip()):
        raise argparse.ArgumentTypeError(f"a model's id, such as {ANTHROPIC_MODEL}, not {text!r}")
    return model


def run_daemon(home: Home, restarted: int | None, model: str | None) -> int:
    """`hands run`: the daemon, whose units of work are its start and every one it runs, never one command's.

    [LAW:nothing-unseen] it is a command not inside a hands.command event: held open over the run, that event would
    make every unit of work the daemon runs a part of its trace, and a restart, exec'd in its place, never ends it.
    """
    run_start = Start(restarted=restarted is not None)
    # [LAW:nothing-unseen] the --model that outranks the file's, None where there was none, said by a start refused first.
    run_start.heard(model_flag=model)
    heart = heartbeat.Heart(home.status, os.getpid(), datetime.now(UTC), heartbeat.HEARTBEAT)
    audit_log = audit_log_of(home)
    # Refused at the door, a run holds no heartbeat yet, and leaves the one there to what wrote it: a running hands,
    # or a crash the next run must read. A restart's run holds it already, once it holds the home: the run before it
    # beat starting under this pid, which it keeps. Before the settings are read there is no collector: the start ends
    # on the log alone. [LAW:nothing-unseen] so does a start anything else ends before the run's launch could end it.
    held: heartbeat.Heart | None = None
    with run_start.ending(audit_log.record):
        try:
            hold(home)
            held = None if restarted is None else heart
            # Read once the home is this run's, before its first heartbeat replaces it. A restart's run before it was
            # told to stop, which is no crash, however long the start took that its last heartbeat may read as gone quiet.
            after_crash = restarted is None and crashed_before(home)
            run_start.heard(after_crash=after_crash)
            settings = door(home, run_start, model)
        except CannotStart as cannot:
            run_start.ended(audit_log.record, cannot)
            refuse(cannot, held)
            return 1
        try:
            ending, shown = run_here(home, restarted, after_crash, settings, heart, audit_log, run_start)
        except CannotStart as cannot:
            # The run's launch ended the start, failed with this reason, on the edge the settings chose.
            refuse(cannot, heart)
            return 1
        match ending:
            case "quit":
                return 0
            case "restart":
                # [LAW:one-source-of-truth] the run after a restart is this one again, on the --model its settings hold;
                # one argument, so an id that begins with a dash is never read as an option.
                again(invocation(home, "run", "--restarted", str(shown), *(() if settings.model is None else (f"--model={settings.model}",))))


def commanded(home: Home, arguments: argparse.Namespace) -> int:
    """Run the command `arguments` name as one unit of work, and its exit code.

    [LAW:nothing-unseen] the one layer every command but `run` and `tmux-status` passes through, so each invocation is one hands.command
    event: the command, its arguments as parsed, its exit code, and how long it took. It ends failed where the command
    exits nonzero, as the exit code says it did, and the unit of work a command runs inside it is in its trace. Written
    to the audit log alone, as every process but the daemon's run writes its events: a command never waits on a
    collector, or on a config.toml the daemon has yet to accept.
    """
    record = audit_log_of(home).record
    with wide.unit("hands.command", record):
        # The home the command ran on, wherever it came from: --home, HANDS_HOME, or ~/.hands.
        wide.annotate(command=arguments.command, home=home.root, **as_facts(arguments))
        code = dispatch(home, arguments, record)
        wide.annotate(exit_code=code)
        if code != 0:
            wide.fail(f"exited {code}")
    return code


def as_facts(arguments: argparse.Namespace) -> dict[str, wide.Fact]:
    """A command's arguments as argparse parsed them, each under arguments.<its name>, the words it gathers a tuple."""
    return {f"arguments.{name}": as_fact(value) for name, value in vars(arguments).items() if name != "command"}


def as_fact(value: object) -> wide.Fact:
    """One parsed argument as a fact: argparse admits no other kind than these, so one is a parser added here without it."""
    match value:
        case None | bool() | int() | str() | Path():
            return value
        case list():
            # recall's words, nargs="*".
            return tuple(map(as_fact, cast(list[object], value)))
        case other:
            # [LAW:no-silent-failure] raised in the command's event, rather than the event lost as its line is written.
            raise TypeError(f"an argument parsed as a {type(other).__name__}, which no hands.command event carries")


def dispatch(home: Home, arguments: argparse.Namespace, record: audit.Record) -> int:
    """Run the command `arguments` name, other than `run`, recording through `record`; its exit code."""
    match arguments.command:
        case "status":
            return report(home)
        case "check":
            from hands.voice import talkkey

            # In place of loguru's DEBUG default, so the debug lines of the tmux and process reads a check makes do not print
            # over its findings; a read that broke still prints, as an error.
            logger.remove()
            to_terminal(sys.stderr)
            return check(home, talkkey.granted())
        case "log":
            return tail_log(home, arguments.lines)
        case "recall":
            return recall_moments(home, record, arguments.words, arguments.most)
        case "phone":
            return show_phone(home)
        case "login":
            return login(home, arguments.method, record)
        case "install-fritter":
            return install_fritter(home, record)
        case "install-plugin":
            return install_plugin(record)
        case "plugin":
            return render_plugin(home, record)
        case "restart":
            return asked_to_restart(home)
        case "smoke":
            # Imported here so that no other command loads aiortc.
            from hands.daemon.smoke import run_smoke

            return run_smoke(home)
        case "indicator":
            # Imported here so that nothing else in `hands` loads AppKit.
            # [LAW:no-ambient-temporal-coupling] the parent is read before AppKit loads, not after: a parent that exits
            # in that second would leave this process watching its new one, pid 1, forever.
            parent = os.getppid() if arguments.parent is None else arguments.parent
            # It shares the run's terminal, so its lines reach it through the run's sink.
            logger.remove()
            to_terminal(sys.stderr)
            from hands.daemon.menubar import show

            return show(home, parent)
        case other:
            raise AssertionError(f"argparse admitted an unknown command {other!r}")


def door(home: Home, run_start: Start, model: str | None) -> Settings:
    """What a run that holds its home checks before its first heartbeat, each in a moment: the talk key's grant, the
    home's copy of fritter, and the settings it starts on. CannotStart where any is missing or stale."""
    # Imported here, like AppKit for the indicator, so that no other command loads Quartz.
    from hands.voice import talkkey

    # [LAW:no-silent-failure] no run without its talk key: a missing grant is named at the door, and macOS is asked to
    # show the prompt that adds the app hands runs in to the list.
    granted = talkkey.granted()
    if not granted:
        talkkey.ask()
        raise CannotStart(f"{readiness.grant(granted).said}, then run hands again.")
    # [LAW:no-silent-failure] every session this home's shim starts runs the home's copy of fritter, which updating hands
    # leaves as the older hands copied it, so this hands may ask it what it cannot do: a stale copy is refused here,
    # naming its fix, before any brain types through it. No copy yet, a hands that carries none, or a PATH whose claude
    # is not this home's shim, is the survey's to say.
    try:
        copy = wrapper.copy_of(home)
    except OSError as error:
        run_start.heard(fritter="unreadable")
        raise CannotStart(f"cannot tell whether {home.fritter} is the fritter this hands carries: {error}") from error
    run_start.heard(fritter=copy)
    if copy == "stale":
        raise CannotStart(f"{home.fritter} is not the fritter this hands carries, {wrapper.PACKAGED}, so sessions started under it may not do what this hands asks of them: run `hands install-fritter`, then run hands again.")
    # [LAW:single-enforcer] the one read of the settings a run starts on: the export edge, the run, and the watch for
    # an edit to them all take these.
    try:
        return load(home, model)
    except Rejected as error:
        raise CannotStart(str(error)) from error


def hold(home: Home) -> None:
    """Lock the home for the rest of this process, or CannotStart, naming the daemon that holds it.

    [LAW:single-enforcer] the one test of whether a daemon runs on the home, made before anything of the home is
    written: a second run refused here leaves the heartbeat, the socket, and the indicator to the daemon running.
    A POSIX record lock is the process's, kept across a restart's exec through the inheritable descriptor, so the same
    pid takes it again; the kernel lets it go when the process ends, however it ends, so a crash leaves no stale lock.
    """
    home.root.mkdir(parents=True, exist_ok=True)
    # Never closed: closing any descriptor of the file would let the process's lock go. A restart's run locks again
    # through the one the run before it opened, so restarts open no more of them.
    descriptor = inherited(home.lock)
    if descriptor is None:
        descriptor = os.open(home.lock, os.O_RDWR | os.O_CREAT, 0o600)
        os.set_inheritable(descriptor, True)
    while True:
        try:
            fcntl.lockf(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError as error:
            # [LAW:one-source-of-truth] the lock names who holds it; the heartbeat may still be an earlier run's.
            # None where the holder let go between the two asks, so the home is free to take.
            pid = holder(descriptor)
            if pid is not None:
                os.close(descriptor)
                raise CannotStart(f"hands is already running on {home.root}, as pid {pid}") from error


def inherited(path: Path) -> int | None:
    """The descriptor on `path` this process already has open: a restart's run has the one the run before it opened."""
    try:
        target = path.stat()
    except FileNotFoundError:
        return None
    for name in os.listdir("/dev/fd"):
        try:
            found = os.fstat(int(name))
        except OSError:
            # The descriptor listdir read /dev/fd through, closed before it is asked about.
            continue
        if (found.st_dev, found.st_ino) == (target.st_dev, target.st_ino):
            return int(name)
    return None


def holder(descriptor: int) -> int | None:
    """The pid of the process whose lock on `descriptor`'s file refused this one's, or None where none holds it now."""
    # macOS's struct flock: l_start, l_len, l_pid, l_type, l_whence; a length of 0 asks about the whole file.
    shape = "qqihh"
    _, _, pid, kind, _ = struct.unpack(shape, fcntl.fcntl(descriptor, fcntl.F_GETLK, struct.pack(shape, 0, 0, 0, fcntl.F_WRLCK, os.SEEK_SET)))
    return None if kind == fcntl.F_UNLCK else pid


def run_here(home: Home, restarted: int | None, after_crash: bool, settings: Settings, heart: heartbeat.Heart, audit_log: audit.AuditLog, run_start: Start) -> tuple[Ending, int]:
    """hands run in this process until it is told to stop: how it was, and the pid of the menu-bar indicator beside it.
    Raises CannotStart where it cannot start, once its heartbeat says starting."""
    # In place of loguru's DEBUG default, so a run's terminal is hands' to read.
    logger.remove()
    to_terminal(sys.stderr)
    # [LAW:no-ambient-temporal-coupling] the first heartbeat goes out before Pipecat is imported and its models load,
    # seconds of silence in which the file would otherwise still name the process that died.
    heart.beat("starting", None, 0, listening=False, degraded=())
    # [LAW:dataflow-not-control-flow] every run shows itself through an indicator it started, on the code it runs: one
    # kept across a restart would judge this run's heartbeat by the code the run before had.
    run_start.heard(previous_indicator=None if restarted is None else retire(restarted))
    shown = start_indicator(home)
    threading.Thread(target=reap, args=(shown,), name="indicator", daemon=True).start()
    # [LAW:one-source-of-truth] what the collector is failing to take is folded from the Exported lines as they are written.
    exports = Exports(audit_log.record, lambda: datetime.now(UTC))
    with exporting(settings.config.collector, exports.record) as exported:
        record = said_failed(exported)
        ending = asyncio.run(launch(lambda: loaded(home, settings, heart, record, exports.degraded, after_crash, run_start), heart, exports.degraded, lambda: edited(home, record, partial(reachable, home), settings), record, run_start))
    return ending, shown


# hands' run, given the event that stops it; it ends saying what it knew last.
type Run = Callable[[asyncio.Event], Coroutine[object, object, Ended]]


async def launch(
    load: Callable[[], Run], heart: heartbeat.Heart, degraded: Callable[[], tuple[heartbeat.Degradation, ...]], edited: Callable[[], Coroutine[object, object, audit.SettingsEdited]], record: audit.Record, run_start: Start
) -> Ending:
    """The run `load` makes, with that load, which imports Pipecat, as the first step of its start; then how it was told to end.
    A run that ends before it was ready ends its start here: failed with what it raised, or cancelled, told to stop first.

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
        with run_start.ending(record):
            # No session has joined before the hooks are served, which is after the import.
            run = await start(lambda: off_loop(load, "the Pipecat import"), heart, lambda: 0, degraded, quit_event)
            last = Ended(None, 0) if run is None else await run(quit_event)
            if failed:
                raise failed[0]
        # Written only by a run told to stop: a refused start's last is refuse's, and any other run that raised leaves
        # its last heartbeat naming a pid that is gone, or, as it restarts, one that stops beating, and neither reads as stopped. [LAW:no-ambient-temporal-coupling] it
        # goes out while the handlers are in, so a restart is never asked once they are out: the heartbeat no longer
        # says running, and one asked before it reads that is heard by stop, where the first stop already decided.
        heart.beat(LAST_BEAT[ending], last.last_audio_out, last.live_sessions, listening=False, degraded=())
    finally:
        watching.cancel()
        # From here a signal has its default effect again: nothing is left to stop gracefully.
        for signal_number in STOP_SIGNALS:
            loop.remove_signal_handler(signal_number)
    return ending


def reachable(home: Home, settings: Config) -> None:
    """Raises Rejected where a start on `settings` could not reach its model: the start's own check, made before the
    restart an edit asks for, so an edit hands could not start on, its brain logged out among them, is refused and
    outlived, not restarted on."""
    backend(settings.llm, home, os.environ)


def loaded(home: Home, settings: Settings, heart: heartbeat.Heart, record: audit.Record, degraded: Callable[[], tuple[heartbeat.Degradation, ...]], after_crash: bool, run_start: Start) -> Run:
    """hands' run, once the seconds it takes to import Pipecat have passed."""
    # Imported here, so that `hands status` answers without loading Pipecat.
    from hands.daemon.run import Configured, configured_from, run

    path = os.environ.get("PATH", "")
    # This run is hands running, and the backend it reaches is the one it was configured on.
    running = readiness.Ready(f"hands is running here: pid {os.getpid()}")

    def surveyed(read: Configured | CannotStart) -> None:
        match read:
            case Configured(voice=voice):
                reached: readiness.Finding = readiness.reaching(voice.llm)
            case CannotStart() as error:
                # Refused on its backend or on its kept voice: the reason names which, so the line claims neither.
                reached = readiness.Missing(f"hands cannot start on its settings: {error}")
        survey(readiness.check(home, path, True, reached, running, keyboards))

    return lambda quit_event: run(lambda environment: configured_from(home, settings, environment), surveyed, home, heart, record, degraded, quit_event, after_crash, os.environ, run_start, OwnModel(home, settings, partial(reachable, home)))


def start_indicator(home: Home) -> int:
    """The menu-bar indicator for this run, in a process of its own: AppKit wants a main thread, and this one is the daemon's."""
    # [LAW:single-enforcer] the indicator ends itself once the run that started it is gone (menubar.show), however the
    # run ended; a session of its own keeps the terminal's Ctrl-C and hangup from ending it first, before it has said so.
    # Its output shares this terminal, so an indicator that fails is seen where the daemon's own failures are. A restart
    # keeps the pid, so the indicator stays a child of the run after it, which ends it by that pid (retire).
    argv = invocation(home, "indicator", "--parent", str(os.getpid()))
    return os.posix_spawn(sys.executable, argv, os.environ, file_actions=[(os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0)], setsid=True)


# What became of the indicator the run before a restart showed: ended as this run asked, killed by it once it had not
# within the grace, found exited on its own and said, or reaped already by the run before, which said so as it did.
type Retired = Literal["ended", "killed", "exited", "reaped"]

# How long the indicator has to end once asked before its process group is killed.
RETIRE_GRACE_SECONDS = 1.0


def retire(indicator: int, grace: float = RETIRE_GRACE_SECONDS) -> Retired:
    """End the indicator the run before a restart showed, so the one this run starts is the only one in the menu bar."""
    try:
        exited, status = os.waitpid(indicator, os.WNOHANG)
    except ChildProcessError:
        return "reaped"
    if exited:
        said_exited(status)
        return "exited"
    # Still this process's child and unreaped, so the pid is the indicator's and no other process's. Its group goes
    # with it, the notices it was posting included, as Child.killed ends a child hands started.
    signalled(indicator, signal.SIGTERM)
    statuses: list[int] = []
    waiter = threading.Thread(target=lambda: statuses.append(os.waitpid(indicator, 0)[1]), name="retire", daemon=True)
    waiter.start()
    waiter.join(grace)
    if waiter.is_alive():
        signalled(indicator, signal.SIGKILL)
        waiter.join()
    # [LAW:no-silent-failure] the status says how it ended, not the asking: one that crashed as it was asked is said.
    code = os.waitstatus_to_exitcode(statuses[0])
    if code in (0, -signal.SIGTERM):
        return "ended"
    if code == -signal.SIGKILL:
        logger.info(f"killed the menu-bar indicator ({indicator}) and everything it started, because it had not ended {grace:.1f}s after it was asked to")
        return "killed"
    said_exited(statuses[0])
    return "exited"


def signalled(group: int, signum: signal.Signals) -> None:
    """Send `signum` to the process group `group` leads, where any of it has yet to exit."""
    try:
        os.killpg(group, signum)
    except (ProcessLookupError, PermissionError):
        # Every process in the group had exited: gone, ProcessLookupError; exited and not yet reaped, macOS says
        # PermissionError. The wait that follows reaps the leader all the same.
        pass


def reap(shown: int) -> None:
    """Wait on the indicator, so one that exits early is reaped and said, not left a zombie under the run."""
    # It exits of its own accord only once the run is gone, so an exit this process lives to see is a failure.
    _, status = os.waitpid(shown, 0)
    said_exited(status)


def said_exited(status: int) -> None:
    logger.error(f"the menu-bar indicator exited ({os.waitstatus_to_exitcode(status)}) while hands runs; hands is not shown in the menu bar")


def report(home: Home) -> int:
    now = datetime.now(UTC)
    verdict = heartbeat.look(home.status, now)
    # [LAW:nothing-unseen] what the exit code was decided by.
    wide.annotate(verdict=verdict)
    match verdict:
        case heartbeat.Up():
            out, code = sys.stdout, 0
        case heartbeat.NeverRan() | heartbeat.Unresponsive() | heartbeat.Down() | heartbeat.Stopped() | heartbeat.Refused():
            out, code = sys.stdout, 1
        case heartbeat.Unreadable():
            out, code = sys.stderr, 2
    print(heartbeat.describe(verdict, now), file=out)
    return code


def show_segment(home: Home) -> int:
    """`hands tmux-status`: one look at the heartbeat, drawn as a tmux status-line segment.

    [LAW:nothing-unseen] it is a command not inside a hands.command event, as the menu bar's looks are not: tmux runs
    one per client every status-interval, so an event for each would repeat the heartbeat into the audit log at that
    rate and push the daemon's own history out of its two segments.
    """
    print(indicator.segment(heartbeat.look(home.status, datetime.now(UTC))))
    # A status line shows the segment whatever the exit, so the segment carries the verdict and the exit nothing.
    return 0


def check(home: Home, granted: bool) -> int:
    reached = readiness.configured(home, os.environ)
    findings = readiness.check(home, os.environ.get("PATH", ""), granted, reached, readiness.daemon(home, datetime.now(UTC)), keyboards)
    wide.annotate(findings=tuple(findings))
    for finding in findings:
        print(f"{display(finding)[0]:<8} {finding.said}")
    kinds = {type(finding) for finding in findings}
    # A piece known to be missing outranks one that could not be looked at: hands is not set up, whatever that one is.
    return 1 if readiness.Missing in kinds else 2 if readiness.Unknown in kinds else 0


def keyboards(pids: Sequence[int]) -> list[Keyboard]:
    """The tmux pane whose keys reach each of `pids`, read on a loop of its own: the readiness check runs on none."""
    return asyncio.run(tmux.keyboards(pids, os.environ))


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


def asked_to_restart(home: Home) -> int:
    """`hands restart`: ask the running daemon to start again, and say how that went."""
    outcome = restart(home, lambda: datetime.now(UTC), lambda: time.sleep(LOOK_SECONDS))
    match outcome:
        case Restarted():
            out, code = sys.stdout, 0
        case NotRunning() | NotBack():
            out, code = sys.stderr, 1
    # What the heartbeat said, and how long the restart took or was waited on.
    wide.annotate(outcome=outcome)
    print(said(outcome, datetime.now(UTC)), file=out)
    return code


def audit_log_of(home: Home) -> audit.AuditLog:
    """The audit log of `home`, on the wall clock: every command's events, and a run's, land in the one log."""
    return audit.AuditLog(home.audit, clock=lambda: datetime.now(UTC))


def login(home: Home, method: "Method", record: audit.Record) -> int:
    # Imported here, so that no other command loads the brain's process and its aiohttp.
    from hands.brain.process import LoginFailed, NotLoggedIn, Unstartable, starting_settings
    from hands.brain.process import login as brain_login
    from hands.core.wire import UPSTREAM

    # [LAW:nothing-unseen] a login is a unit of work: the login asked for, whether it wrote the brain's settings, whether
    # it took Claude Code's first run, and the account it ended on.
    with wide.unit("brain.login", record):
        wide.annotate(asked=method)
        try:
            # Before any run of Claude Code on this home, so that none ever syncs the account's skills or plugins.
            wide.annotate(settings_written=starting_settings(home.brain))
            signed = brain_login(home.brain, UPSTREAM, os.environ, method)
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


def install_fritter(home: Home, record: audit.Record) -> int:
    # [LAW:nothing-unseen] an install is a unit of work: the fritter it copied from, where it put it and the claude
    # beside it, and whether PATH finds that claude.
    with wide.unit("fritter.install", record):
        wide.annotate(packaged=wrapper.PACKAGED)
        try:
            installed = wrapper.install(home)
        except (wrapper.Uninstallable, wrapper.Unpackaged) as error:
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


def install_plugin(record: audit.Record) -> int:
    # [LAW:nothing-unseen] an install is a unit of work: what Claude Code said of the plugin before, what its install
    # exited with when it asked, and whether the plugin is there after.
    path = os.environ.get("PATH", "")
    with wide.unit("plugin.install", record):
        before = readiness.plugin(path)
        wide.annotate(before=type(before).__name__.lower())
        match before:
            case readiness.Ready(said=said):
                print(said)
                return 0
            case readiness.Unknown(said=said):
                wide.fail(said)
                print(f"hands install-plugin: {said}", file=sys.stderr)
                return 2
            case readiness.Missing():
                pass
        added = subprocess.run(["claude", "plugin", "marketplace", "add", MARKETPLACE])
        wide.annotate(marketplace_add_exit=added.returncode)
        if added.returncode != 0:
            wide.fail(f"`claude plugin marketplace add {MARKETPLACE}` failed ({added.returncode})")
            print(f"hands install-plugin: `claude plugin marketplace add {MARKETPLACE}` failed ({added.returncode})", file=sys.stderr)
            return 1
        print(f"Claude Code now shows the command `hands plugin`, which installs {PLUGIN_ID}, and asks whether to run it: answer y", flush=True)
        # Keys pressed during what ran before would reach Claude Code's [y/N] as an answer the person never gave it.
        if sys.stdin.isatty():
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        installed = subprocess.run(["claude", "plugin", "install", "--scope", "user", PLUGIN_ID])
        wide.annotate(install_exit=installed.returncode)
        after = readiness.plugin(path)
        wide.annotate(installed=isinstance(after, readiness.Ready))
        match after:
            case readiness.Ready(said=said):
                print(said)
                return 0
            case readiness.Missing(said=said) | readiness.Unknown(said=said):
                wide.fail(said)
                print(f"hands install-plugin: {said}", file=sys.stderr)
                return 1


def render_plugin(home: Home, record: audit.Record) -> int:
    # [LAW:nothing-unseen] Claude Code runs this once per session: the interpreter the hooks run on, the plugin it
    # printed, and whether that plugin was written now or a session before had. Claude Code waits for this command to
    # exit before the session starts, so, like the shim, it reads no config.toml, whose rejection is the daemon's to
    # report and never costs a session its hooks.
    with wide.unit("plugin.render", record):
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
    # [LAW:nothing-unseen] how many lines the log had to print first, and the log offset following began at.
    wide.annotate(tailed=len(newest), followed_from=offset)
    try:
        # json.dumps escapes C0 controls but writes DEL, C1, U+2028, U+2029, and bidi controls raw; their escapes keep each
        # line JSON, and one line to any reader.
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


def recall_moments(home: Home, record: audit.Record, words: Sequence[str], most: int) -> int:
    """Print the moments `hands recall` found, one a line, each at the time it happened here."""
    # [LAW:nothing-unseen] a recall is a unit of work: what it was asked, how much of the log it read, and what it found,
    # zeros included.
    with wide.unit("memory.recall", record, ("lines", "unreadable", "moments", "matched", "printed")):
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
