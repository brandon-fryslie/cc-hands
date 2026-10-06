"""Whether hands is set up to work here: each step of the README's install, named, and done or missing.

    hands check     # one line a step; exits 0 only when every step is done

The steps, in the README's order: Claude Code from its installer; PortAudio, which the microphone opens through; the
installed `hands` on PATH, which Claude Code runs for the plugin; the claude shim that runs sessions under fritter;
the plugin that joins sessions to hands; a backend with its key or login; the Input
Monitoring grant that lets hands hear the talk key; hands running; and the running sessions themselves. `hands run` says the same lines as it starts,
and a daemon that is up says nothing about any of them, so this is where a missing one is heard.
"""

import os
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import get_args

from hands.core.effects import Fritter
from hands.core.events import Attached
from hands.core.reach import Unwrapped, through, writer
from hands.core.tmux import Behind, Keyboard, NotInTmux, Pane, PaneUnread
from hands.core.session import Membership
from hands.daemon.backend import backend as resolve
from hands.daemon.config import load
from hands.sessions import heartbeat, liveness, wrapper
from hands.sessions.hookconfig import PLUGIN_ID
from hands.sessions.home import Home
from hands.sessions.payload import Payload, Rejected
from hands.sessions.processes import process_starts
from hands.sessions.terminals import Terminal, Terminals, Undescribed, attended, terminal_processes
from hands.voice import backends


@dataclass(frozen=True)
class Ready:
    said: str


@dataclass(frozen=True)
class Missing:
    said: str  # what is missing, what that stops, and what puts it there


@dataclass(frozen=True)
class Unknown:
    said: str  # why the piece could not be looked at


Finding = Ready | Missing | Unknown

# The tmux pane whose keys reach each of a list of processes, as `hands.sessions.tmux.keyboards` reads it.
Keyboards = Callable[[Sequence[int]], list[Keyboard]]

# `claude plugin list` answers in a quarter of a second; one that has not answered in this long is not going to.
LIST_TIMEOUT_SECONDS = 20.0
# `hands --version` imports hands' CLI, a couple of seconds; one that has not answered in this long is not going to.
VERSION_TIMEOUT_SECONDS = 30.0
INSTALL_CLAUDE = "`curl -fsSL https://claude.ai/install.sh | bash`"


def check(home: Home, path: str, granted: bool, reached: Finding, running: Finding, keyboards: Keyboards) -> list[Finding]:
    """Every step, in the README's order. `path` is the PATH sessions are started from; `granted`, this terminal's grant;
    `reached`, whether the settings' backend has its key or login; `running`, whether hands is up; `keyboards`, what
    reads the tmux pane in front of each running session."""
    # [LAW:dataflow-not-control-flow] every step is looked at every time: one that is missing hides none after it.
    return [claude(path), portaudio(), installed(path), shim(home, path), plugin(path), reached, grant(granted), running, sessions(home, path, keyboards)]


def claude(path: str) -> Finding:
    """Whether this PATH has a Claude Code of its own, apart from any hands shim, installed as its installer puts it."""
    match claude_code(wrapper.real_claude(path)):
        case Unfindable(said):
            return Missing(f"no Claude Code that hands can join its sessions of: {said}. Claude Code's installer puts it in: {INSTALL_CLAUDE}")
        case executable:
            return Ready(f"Claude Code is installed: {executable}")


def portaudio() -> Finding:
    """Whether PyAudio, which hands opens the microphone through, can load Homebrew's PortAudio."""
    try:
        import pyaudio
    except ImportError as error:
        return Missing(f"PyAudio cannot load PortAudio ({error}), so hands cannot open the microphone: `brew install portaudio`, with Homebrew from https://brew.sh")
    return Ready(f"PortAudio is there for the microphone: {pyaudio.get_portaudio_version_text()}")


def installed(path: str) -> Finding:
    """Whether `hands` on this PATH, which Claude Code runs for the plugin's hooks and skills, is this hands."""
    this = f"hands {version('hands')}"
    found = shutil.which("hands", path=path)
    if found is None:
        return Missing(
            f"this PATH has no `hands`, so Claude Code cannot run `hands plugin` and no session gets hands' hooks: "
            f"`uv tool update-shell` puts the directory `uv tool install` writes it to on PATH"
        )
    try:
        said = subprocess.run([found, "--version"], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=VERSION_TIMEOUT_SECONDS)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return Unknown(f"cannot ask {found} which hands it is: {error}")
    if said.returncode != 0:
        return Unknown(f"`{found} --version` failed ({said.returncode}), so which hands it is is unknown: {said.stderr.strip()}")
    # [LAW:one-source-of-truth] the plugin's hooks are whichever hands Claude Code finds, so one that is not this hands
    # is said here, where this hands' own lines would otherwise vouch for it.
    if said.stdout.strip() != this:
        return Missing(f"`hands` on this PATH, {found}, is {said.stdout.strip()}, not this {this}, so sessions run its hooks: put this one first on PATH, or install it again")
    return Ready(f"`hands` on this PATH is {found}, {this}: Claude Code runs it for the plugin in every session")


def configured(home: Home, environment: Mapping[str, str]) -> Finding:
    """Whether the backend the home's config.toml names has the key or the login it reaches its model with."""
    try:
        config = load(home).config
    except Rejected as error:
        return Missing(f"hands cannot read its settings: {error}")
    try:
        reached: Finding = reaching(resolve(config.llm, home, environment))
    except Rejected as error:
        reached = Missing(f"hands has no model to talk with: {error}")
    except OSError as error:
        # The keychain or the brain's claude could not be asked, which says nothing of whether they hold a key or a login.
        reached = Unknown(f"cannot tell whether the [llm] backend can reach its model: {error}")
    return reached


def reaching(reached: backends.LLMBackend) -> Ready:
    """What a backend that has its key or login reaches, never its key."""
    match backends.account(reached):
        case None:
            return Ready(f"the [llm] backend reaches {reached.model} at {backends.server(reached)} with its key")
        case account:
            return Ready(f"the brain is logged in as {account}, and reaches {reached.model}")


def daemon(home: Home, now: datetime) -> Finding:
    """Whether hands is running, from its heartbeat."""
    verdict = heartbeat.look(home.status, now)
    said = heartbeat.describe(verdict, now)
    match verdict:
        case heartbeat.Up():
            return Ready(said)
        case heartbeat.Unreadable():
            return Unknown(said)
        case heartbeat.NeverRan() | heartbeat.Down() | heartbeat.Unresponsive() | heartbeat.Stopped() | heartbeat.Refused():
            return Missing(f"{said}: `hands run`, in a terminal that has the Input Monitoring grant")


def plugin(path: str) -> Finding:
    """Whether Claude Code, as `claude` on this PATH runs it, has hands' plugin installed and enabled."""
    # [LAW:one-source-of-truth] Claude Code is asked, never its files read: where it keeps its plugins is its own.
    # Not a session to the shim, since nothing here is a terminal, so the shim runs the real claude.
    try:
        listed = subprocess.run(
            ["claude", "plugin", "list", "--json"],
            env={**os.environ, "PATH": path},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=LIST_TIMEOUT_SECONDS,
        )
    # A ValueError is output that is not text.
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return Unknown(f"cannot ask `claude plugin list` whether the plugin {PLUGIN_ID} is installed: {error}")
    if listed.returncode != 0:
        return Unknown(f"`claude plugin list` failed ({listed.returncode}), so whether the plugin {PLUGIN_ID} is installed is unknown: {listed.stderr.strip()}")
    return plugin_listed(listed.stdout)


def plugin_listed(raw: str) -> Finding:
    """What `claude plugin list --json` says of hands' plugin."""
    try:
        # Only a user-scope install joins every session; a project or local one joins the sessions of one directory,
        # and is listed as enabled or not by the directory the listing was asked from.
        hands = [entry for entry in Payload.parse_list(raw.encode(), "each plugin") if entry.fields.get("id") == PLUGIN_ID]
        everywhere = {entry.flag("enabled") for entry in hands if entry.text("scope") == "user"}
    except Rejected as error:
        return Unknown(f"`claude plugin list --json` printed what hands cannot read: {error}")
    if True in everywhere:
        return Ready(f"the plugin {PLUGIN_ID} is installed and enabled for every session: a session started since, or reloaded with /reload-plugins, joins hands")
    if everywhere:
        return Missing(f"the plugin {PLUGIN_ID} is disabled, so no session joins hands: `claude plugin enable --scope user {PLUGIN_ID}`, then /reload-plugins in each running session")
    return Missing(
        f"the plugin {PLUGIN_ID} is not installed for every session, so only the sessions of a project it is installed in "
        f"join hands: `claude plugin install {PLUGIN_ID}` (after `claude plugin marketplace add brandon-fryslie/cc-hands`), then "
        f"/reload-plugins in each running session"
    )


def shim(home: Home, path: str) -> Finding:
    """Whether `claude` on this PATH is the shim this hands writes, with the fritter this hands carries, so it starts
    every interactive session under fritter."""
    # Every fix below is `hands install-fritter`'s, which a hands built without its fritter cannot run: the rebuild is said instead.
    try:
        carried = wrapper.packaged()
    except wrapper.Unpackaged as error:
        return Missing(str(error))
    found = shutil.which("claude", path=path)
    claude = None if found is None else Path(found)
    # [LAW:dataflow-not-control-flow] what to do is read off what is there: an installed shim wants only the PATH.
    first = f'put {home.bin} first on PATH: export PATH="{home.bin}:$PATH"'
    fix = first if isinstance(wrapper.shim_of(home.shim), wrapper.Shim) else f"run `hands install-fritter`, then {first}"
    match None if claude is None else wrapper.shim_of(claude):
        case None:
            return Missing(f"`claude` on this PATH is {found or 'nothing'}, not hands' shim, so no session started from it can be typed into: {fix}")
        case wrapper.Stale():
            return Missing(f"`claude` on this PATH is hands' shim, {found}, but not the one this hands writes: run `hands install-fritter`")
        case wrapper.Shim(fritter=runs) if not os.access(runs, os.X_OK):
            return Missing(f"`claude` on this PATH is hands' shim, {found}, but its fritter {runs} is not there to run, so every interactive claude fails to start: run `hands install-fritter`")
        case wrapper.Shim(fritter=runs):
            # [LAW:one-source-of-truth] the fritter in a home's bin is a copy of the packaged one, so a copy that has
            # drifted from it, as one does when hands is upgraded or a checkout's fritter rebuilt, is said, never trusted.
            try:
                current = wrapper.carried(runs)
            except OSError as error:
                return Unknown(f"cannot tell whether {runs} is the fritter this hands carries, {carried}: {error}")
            if not current:
                return Missing(f"`claude` on this PATH is hands' shim, {found}, but its fritter {runs} is not the one this hands carries, {carried}: run `hands install-fritter`")
            return Ready(f"`claude` on this PATH is hands' shim, {found}: every interactive session started from it can be typed into")


def grant(granted: bool) -> Ready | Missing:
    """Whether this terminal's app may show hands the keys typed in other apps."""
    if granted:
        return Ready("this terminal has the Input Monitoring grant, so hands run here hears the talk key (Right Shift)")
    return Missing(
        "this terminal has no Input Monitoring grant, so hands run here cannot hear the talk key (Right Shift). Grant it "
        "to the app this terminal runs in, in System Settings > Privacy & Security > Input Monitoring, and restart that app"
    )


def sessions(home: Home, path: str, keyboards: Keyboards) -> Finding:
    """Whether every running session can be typed into, those hands knows of and those it does not, each by the fritter
    that wrapped it or the tmux pane in front of it, as `keyboards` reads them. Nothing is removed or dialled."""
    try:
        records, unreadable = liveness.recorded(home)
        started = process_starts({record.membership.pid for record in records})
        # [LAW:single-enforcer] a session is running when the sweep would attach it, and by no other test.
        seen, _ = liveness.observations((), records, started, frozenset())
        running = [observed.membership for observed in seen if isinstance(observed, Attached)]
        # A socket that is gone is a fritter that is gone. It is never dialled here: fritter writes what it cannot read
        # into the session's own terminal.
        listening = {member.fritter for member in running if member.fritter is not None and member.fritter.is_socket()}
    except OSError as error:
        return Unknown(f"cannot look at the running sessions in {home.memberships}: {error}")
    found = unrecorded(home, path, {member.pid for member in running})
    # One read of the panes, for every session the check looks at.
    joining = found.sessions if isinstance(found, Unrecorded) else []
    pids = [*(member.pid for member in running), *(session.process.pid for session in joining)]
    return sessions_found(running, listening, unreadable, found, dict(zip(pids, keyboards(pids), strict=True)))


@dataclass(frozen=True)
class Unfindable:
    said: str  # why a running session hands has no record of cannot be told from the other programs at a terminal


@dataclass(frozen=True)
class Unjoined:
    """A running session hands has no record of, and the fritter that wrapped it, if one did."""

    process: Terminal
    fritter: Terminal | None


@dataclass(frozen=True)
class Unrecorded:
    """The sessions at a terminal hands has no record of, how many runs of claude beside them are no session, by why, and
    the processes at a terminal the kernel would not describe, any of which may be one."""

    sessions: list[Unjoined]
    others: Counter[wrapper.NotASession]
    unread: list[Undescribed]


def unrecorded(home: Home, path: str, members: Collection[int]) -> Unrecorded | Unfindable:
    """The sessions running at a terminal that hands has no record of, among this user's processes now."""
    match claude_code(wrapper.real_claude(path)):
        case Unfindable() as unfindable:
            return unfindable
        case executable:
            try:
                config = config_dir(os.environ, Path.cwd())
            except OSError as error:
                return Unfindable(f"cannot tell which Claude Code config this check runs under: {error}")
            try:
                return unjoined(home, executable, config, terminal_processes(), members, attended)
            except OSError as error:
                return Unfindable(f"cannot look at this user's processes at a terminal: {error}")


def claude_code(claude: Path | None) -> Path | Unfindable:
    """The executable a session of the real `claude` on PATH runs as."""
    if claude is None:
        return Unfindable("this PATH has no `claude` of its own, apart from any hands shim")
    executable = claude.resolve()
    try:
        with executable.open("rb") as start:
            script = start.read(2) == b"#!"
    except OSError as error:
        return Unfindable(f"cannot read the real `claude`, {executable}: {error}")
    if script:
        return Unfindable(f"the real `claude`, {executable}, is a script, so its sessions run as its interpreter: `claude install` puts in the native one, whose sessions hands can tell")
    return executable


def config_dir(environment: Mapping[str, str], cwd: Path) -> Path:
    """The Claude Code config a process with this environment, working in `cwd`, runs under: its plugins, and so whether
    hands' is one. It is the directory itself, by whatever path or link it was named, read as that process reads it."""
    named = environment.get("CLAUDE_CONFIG_DIR") or Path(environment.get("HOME") or Path.home()) / ".claude"
    # realpath, not Path.resolve: on Python 3.12 resolve raises RuntimeError on a link loop, where realpath stops.
    return Path(os.path.realpath(cwd / named))


def unjoined(
    home: Home, claude: Path, config: Path, terminals: Terminals, members: Collection[int], attended: Callable[[Terminal], bool | Undescribed]
) -> Unrecorded:
    """The sessions at a terminal under `config` that no running membership names: started before the plugin, and not
    reloaded since.

    A session is a process at a terminal running `claude`, the Claude Code executable, or any other version of it, since
    an update leaves running sessions on the version they started on, that the shim would have run as a session: a
    `claude -p`, a subcommand, or a claude piped into, is none. One whose parent runs claude is that run's own helper. The
    hook records that same process, so it is matched by pid. One under another config, as the brain is, has other
    plugins, and is no session of the plugin this check looks at; one under the same directory by another path, through
    a link, is, since `config` and each process's are both as `config_dir` reads them. `attended` says whether a process
    reads and writes its terminal; it is asked only of a run of claude, the one process it matters for. A process the
    kernel would not describe may be a session, unless it is one hands knows or a run's own helper.
    """
    by_pid = {process.pid: process for process in terminals.found}
    install = _unversioned(claude)

    def runs_claude(pid: int) -> bool:
        return pid in by_pid and _unversioned(by_pid[pid].executable) == install

    fritter = home.fritter.resolve()
    runs = [
        process
        for process in terminals.found
        if runs_claude(process.pid) and not runs_claude(process.parent) and config_dir(process.environment, process.cwd) == config and process.pid not in members
    ]
    told = [(process, attended(process)) for process in runs]
    described = [(process, reads) for process, reads in told if isinstance(reads, bool)]
    # [LAW:one-source-of-truth] a session is what the shim would run as one, by the shim's own test.
    ran: list[tuple[Terminal, wrapper.Run]] = [(process, wrapper.run(process.arguments, reads)) for process, reads in described]
    return Unrecorded(
        [Unjoined(process, wrapped if (wrapped := by_pid.get(process.parent)) and wrapped.executable == fritter else None) for process, why in ran if why == "session"],
        Counter(why for _, why in ran if why != "session"),
        [
            *(each for each in terminals.unread if each.pid not in members and not runs_claude(each.parent)),
            *(reads for _, reads in told if isinstance(reads, Undescribed)),
        ],
    )


# Claude Code keeps each version under a name that is the version: a file in the native installer's versions
# directory, a directory in a Homebrew cask's. A semantic version, with any pre-release and build.
_VERSION = re.compile(r"\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?")


def _unversioned(executable: Path) -> tuple[str | None, ...]:
    """The executable's path with each part that names a version blanked: the same for every version of one install."""
    return tuple(None if _VERSION.fullmatch(part) else part for part in executable.parts)


def sessions_found(
    running: Sequence[Membership],
    listening: Collection[Path],
    unreadable: Sequence[liveness.Unreadable],
    unrecorded: Unrecorded | Unfindable,
    keyboards: Mapping[int, Keyboard],
) -> Finding:
    """What the running sessions are to hands, given which fritter sockets are there, which files did not parse, which
    sessions at a terminal hands has no record of, and the tmux pane in front of each session, by its pid."""
    match unrecorded:
        case Unfindable(said):
            unjoined, unread, beside = [], [Unknown(f"a session hands has no record of cannot be found: {said}")], ""
        case Unrecorded(sessions, others, undescribed):
            # [LAW:nothing-unseen] a claude at a terminal set aside as no session is counted by why, so none goes unseen;
            # nor does a process the kernel would not describe, which may be a session.
            counts = ", ".join(f"{why} {others[why]}" for why in get_args(wrapper.NotASession))
            unjoined, beside = [_unjoined(session, keyboards[session.process.pid]) for session in sessions], f", runs of claude at a terminal that are none: {counts}"
            said = ", ".join(f"pid {each.pid} ({each.call}: {os.strerror(each.errno)})" for each in undescribed)
            unread = [Unknown(f"the kernel would not describe these processes at a terminal, so whether any is a session hands has no record of is unknown: {said}")] if undescribed else []
    lines = [
        *(_typable(member, listening, keyboards[member.pid]) for member in running),
        *(Missing(f"{file.path} names no session hands can read ({file.error}): hands run removes it") for file in unreadable),
        *unjoined,
        *unread,
    ]
    unreached = [line.said for line in lines if isinstance(line, Missing)]
    unseen = [line.said for line in lines if isinstance(line, Unknown)]
    known = f"running sessions hands knows of: {len(running)}{beside}"
    # [LAW:no-silent-failure] sessions that could not be looked for are said, never taken for none.
    if unreached:
        return Missing(f"{known}, and hands cannot reach these:" + "".join(f"\n    {line}" for line in [*unreached, *unseen]))
    if unseen:
        return Unknown(f"{known}, and whether hands can reach these is unknown:" + "".join(f"\n    {line}" for line in unseen))
    return Ready(f"{known}, and each can be typed into")


def _unjoined(session: Unjoined, pane: Keyboard) -> Missing:
    where = f"{session.process.cwd} (pid {session.process.pid}) is a session hands has no record of"
    # [LAW:single-enforcer] whether it can be typed into once it joins is decided as a joined session's writer is.
    match through(session.fritter, pane):
        case Terminal() | Pane():
            return Missing(f"{where}, so it cannot be reached: /reload-plugins in it")
        case Behind(pane=behind):
            return Missing(f"{where}, started outside fritter, {_behind(behind)}, so it cannot be typed into: {_FRONT}, then /reload-plugins in it")
        case NotInTmux():
            return Missing(f"{where}, started outside fritter and in no tmux pane, so it cannot be typed into: {_RESTART}")
        case PaneUnread(reason):
            return Missing(
                f"{where}, started outside fritter, and which tmux pane it runs in could not be read ({reason}): "
                f"/reload-plugins in it joins it, and whether it can be typed into then is unknown"
            )


_RESTART = "restart it from a PATH whose `claude` is hands' shim"
_FRONT = "bring it back to the front of its pane"


def _behind(pane: Pane) -> str:
    return f"and another program has the keyboard of the tmux pane it runs in, {pane.id} (window {pane.window} of {pane.session})"


def _typable(member: Membership, listening: Collection[Path], pane: Keyboard) -> Finding:
    where = f"{member.cwd} (pid {member.pid})"
    # [LAW:single-enforcer] a session's reach is judged by the writer that would type into it, and by no other test.
    match writer(member, pane):
        case Fritter(socket) if socket not in listening:
            return Missing(f"{where} has lost its fritter, whose socket {socket} is gone, so it cannot be typed into: {_RESTART}")
        case Fritter():
            return Ready(f"{where} is typed into through its fritter")
        case Pane(id=id):
            return Ready(f"{where} is typed into through its tmux pane {id}")
        case Unwrapped(pane=missing):
            match missing:
                case Behind(pane=behind):
                    return Missing(f"{where} was started outside fritter, {_behind(behind)}, so it cannot be typed into: {_FRONT}, or {_RESTART}")
                case NotInTmux():
                    return Missing(f"{where} was started outside fritter and runs in no tmux pane, so it cannot be typed into: {_RESTART}")
                case PaneUnread(reason):
                    return Unknown(f"{where} was started outside fritter, and which tmux pane it runs in could not be read, so whether it can be typed into is unknown: {reason}")
