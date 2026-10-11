"""Whether hands is set up to work here: each step of the install, named, and done or missing.

    hands check     # one line a step; exits 0 only when every step is done

The steps, in the order they are said: what the one install command puts in, which is Claude Code, PortAudio, which the
microphone opens through, and the installed `hands` on PATH, which Claude Code runs for the plugin; the claude shim that runs sessions under fritter;
the plugin that joins sessions to hands; Claude Code's own first run and login, so a session it starts takes what is typed; the brain with its login; the Input
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
from hands.sessions import claudecode, firstrun, heartbeat, liveness, wrapper
from hands.sessions.hookconfig import MARKETPLACE, MARKETPLACE_NAME, PLUGIN_ID
from hands.sessions.home import Home
from hands.sessions.payload import Payload, Rejected
from hands.sessions.startsession import as_from_a_terminal
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

# `hands --version` imports hands' CLI, a couple of seconds; one that has not answered in this long is not going to.
VERSION_TIMEOUT_SECONDS = 30.0
# [LAW:one-source-of-truth] the README's one install command, which installs whichever of Claude Code, PortAudio and hands is missing.
INSTALL = '`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/promptctl/cc-hands/master/install.sh)"`'


def check(claude_code: claudecode.ClaudeCode, home: Home, environment: Mapping[str, str], granted: bool, reached: Finding, running: Finding, keyboards: Keyboards) -> list[Finding]:
    """Every step, in the order the module names them, Claude Code asked through `claude_code`. `environment` is the one sessions are started from; `granted`, the grant of the app this runs in;
    `reached`, whether the brain has its login; `running`, whether hands is up; `keyboards`, what
    reads the tmux pane in front of each running session."""
    path = environment.get("PATH", "")
    # [LAW:dataflow-not-control-flow] every step is looked at every time: one that is missing hides none after it.
    return [claude(path), portaudio(), installed(path), shim(home, path), plugin(claude_code, path), first_run(claude_code, home, environment), reached, hears(granted, running), running, sessions(home, path, keyboards)]


def claude(path: str) -> Finding:
    """Whether this PATH has a Claude Code of its own, apart from any hands shim, installed as its installer puts it."""
    match claude_code(wrapper.real_claude(path)):
        case Unfindable(said):
            return Missing(f"no Claude Code that hands can join its sessions of: {said}")
        case executable:
            return Ready(f"Claude Code is installed: {executable}")


def portaudio() -> Finding:
    """Whether PyAudio, which hands opens the microphone through, can load Homebrew's PortAudio."""
    try:
        import pyaudio
    except ImportError as error:
        return Missing(f"PyAudio cannot load PortAudio ({error}), so hands cannot open the microphone: the install command puts it in: {INSTALL}")
    return Ready(f"PortAudio is there for the microphone: {pyaudio.get_portaudio_version_text()}")


def installed(path: str) -> Finding:
    """Whether `hands` on this PATH, which Claude Code runs for the plugin's hooks and skills, is this hands."""
    this = f"hands {version('hands')}"
    found = shutil.which("hands", path=path)
    if found is None:
        return Missing(
            "this PATH has no `hands`, so Claude Code cannot run `hands plugin` and no session gets hands' hooks: "
            f"the install command puts it there: {INSTALL}"
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


def configured(claude_code: claudecode.ClaudeCode, home: Home, environment: Mapping[str, str]) -> Finding:
    """Whether the brain has the login it reaches its model with."""
    try:
        config = load(home).config
    except Rejected as error:
        return Missing(f"hands cannot read its settings: {error}")
    try:
        reached: Finding = reaching(resolve(claude_code, config.llm, home, environment))
    except Rejected as error:
        reached = Missing(f"hands has no model to talk with: {error}")
    except claudecode.Unreachable as error:
        # The brain's claude could not be asked, which says nothing of whether it holds a login.
        reached = Unknown(f"cannot tell whether the [llm] backend can reach its model: {error}")
    return reached


def reaching(reached: backends.ClaudeCodeBackend) -> Ready:
    """What a brain that has its login reaches."""
    return Ready(f"the brain is logged in as {reached.account}, and reaches {reached.model}")


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
            return Missing(f"{said}: open hands.app, or `hands run` in a terminal that has the Input Monitoring grant")


def plugin(claude_code: claudecode.ClaudeCode, path: str) -> Finding:
    """Whether Claude Code, as `claude` on this PATH runs it, has hands' plugin installed and enabled."""
    match listing(claude_code.plugins, path, "plugin", f"whether the plugin {PLUGIN_ID} is installed"):
        case Unknown() as unknown:
            return unknown
        case bytes() as listed:
            return plugin_listed(listed)


def marketplace(claude_code: claudecode.ClaudeCode, path: str) -> Finding:
    """Whether Claude Code, as `claude` on this PATH runs it, has the marketplace hands' plugin is installed from, from
    whichever source the person added it: this repository on GitHub, or a checkout of it."""
    match listing(claude_code.marketplaces, path, "plugin marketplace", f"whether the marketplace {MARKETPLACE_NAME} is added"):
        case Unknown() as unknown:
            return unknown
        case bytes() as listed:
            return marketplace_listed(listed)


def listing(listed: Callable[[claudecode.Instance], claudecode.Answer], path: str, said: str, unknown: str) -> bytes | Unknown:
    """What `claude <said> list --json`, which `listed` asks, prints, as `claude` on this PATH runs it."""
    # [LAW:one-source-of-truth] Claude Code is asked, never its files read: where it keeps its plugins is its own.
    # Not a session to the shim, since nothing here is a terminal, so the shim runs the real claude.
    try:
        answer = listed(claudecode.Instance("claude", {**os.environ, "PATH": path}))
    except claudecode.Unreachable as error:
        return Unknown(f"cannot ask `claude {said} list` {unknown}: {error}")
    if answer.exit != 0:
        return Unknown(f"`claude {said} list` failed ({answer.exit}), so {unknown} is unknown: {answer.stderr.decode(errors='replace').strip()}")
    return answer.stdout


def marketplace_listed(raw: bytes) -> Finding:
    """What `claude plugin marketplace list --json` says of the marketplace hands' plugin is installed from."""
    try:
        names = {entry.text("name") for entry in Payload.parse_list(raw, "each marketplace")}
    except Rejected as error:
        return Unknown(f"`claude plugin marketplace list --json` printed what hands cannot read: {error}")
    if MARKETPLACE_NAME in names:
        return Ready(f"the marketplace {MARKETPLACE_NAME} is added")
    return Missing(f"the marketplace {MARKETPLACE_NAME} is not added: `claude plugin marketplace add {MARKETPLACE}`")


def plugin_listed(raw: bytes) -> Finding:
    """What `claude plugin list --json` says of hands' plugin."""
    try:
        # Only a user-scope install joins every session; a project or local one joins the sessions of one directory,
        # and is listed as enabled or not by the directory the listing was asked from.
        hands = [entry for entry in Payload.parse_list(raw, "each plugin") if entry.fields.get("id") == PLUGIN_ID]
        everywhere = {entry.flag("enabled") for entry in hands if entry.text("scope") == "user"}
    except Rejected as error:
        return Unknown(f"`claude plugin list --json` printed what hands cannot read: {error}")
    if True in everywhere:
        return Ready(f"the plugin {PLUGIN_ID} is installed and enabled for every session: a session started since, or reloaded with /reload-plugins, joins hands")
    if everywhere:
        return Missing(f"the plugin {PLUGIN_ID} is disabled, so no session joins hands: `hands install-plugin`, then /reload-plugins in each running session")
    return Missing(
        f"the plugin {PLUGIN_ID} is not installed for every session, so only the sessions of a project it is installed in "
        f"join hands: `hands install-plugin`, then /reload-plugins in each running session"
    )


@dataclass(frozen=True)
class FirstRun:
    """What the person's own Claude Code would ask before its input in the folder `hands smoke` starts its session in,
    and whether it holds a login it would use: its account's, which a Claude Code done with its onboarding does not ask
    for again, or an API key it was told to use."""

    claude: Path  # the real claude, past any hands shim, that was asked
    unanswered: firstrun.Unanswered | None
    logged_in: bool


def first_run_state(claude_code: claudecode.ClaudeCode, home: Home, environment: Mapping[str, str]) -> FirstRun | Unknown:
    """What `claude` on this environment's PATH, past any hands shim, would ask first in `home.smoke`, run there as `hands
    smoke` runs it."""
    # [LAW:one-source-of-truth] the environment `hands smoke` starts its session in, whichever command looks.
    environment = as_from_a_terminal(environment)
    claude = wrapper.real_claude(environment.get("PATH", ""))
    if claude is None:
        return Unknown("there is no Claude Code on this PATH to ask whether it has been through its first run")
    try:
        asked = firstrun.persons(claude_code, environment, home.smoke, config_dir(environment, home.smoke))
    except Rejected as error:
        return Unknown(f"cannot tell what Claude Code would ask first: {error}")
    # `auth status` takes an ANTHROPIC_API_KEY for a login, one it was told not to use among them (2.1.294), so it is asked
    # without the environment's, and a login from that key, as one settings.json sets, counts only where its first run
    # approved it. A Console login is an API key too, from another source, and counts.
    unkeyed = {name: value for name, value in environment.items() if name != firstrun.API_KEY}
    try:
        status = claude_code.auth_status(claudecode.Instance(claude, unkeyed))
    except claudecode.Unreachable as error:
        return Unknown(f"cannot ask {claude} whether it is logged in: {error}")
    try:
        # It exits 1 when logged out, saying so in its JSON as when logged in (2.1.289).
        said = Payload.parse(status.stdout)
        account = said.flag("loggedIn") and said.optional_text("apiKeySource") != firstrun.API_KEY
    except Rejected as error:
        return Unknown(f"`{claude} auth status` answered {status.stdout[:200]!r} {status.stderr[:200]!r}, not its status: {error}")
    return FirstRun(claude, asked.unanswered, account or asked.keyed)


def first_run(claude_code: claudecode.ClaudeCode, home: Home, environment: Mapping[str, str]) -> Finding:
    """Whether the person's own Claude Code is logged in and would take what `hands smoke` types into its session."""
    match first_run_state(claude_code, home, environment):
        case Unknown() as unknown:
            return unknown
        case FirstRun(unanswered=None, logged_in=True):
            return Ready(f"Claude Code is logged in and asks nothing first in {home.smoke}, where `hands smoke` starts its session")
        case FirstRun(unanswered=asked, logged_in=logged_in):
            owed = ([] if asked is None else [f"would first ask {asked.listed} ({asked.why})"]) + ([] if logged_in else ["has no login"])
            return Missing(f"Claude Code {' and '.join(owed)}, so the session `hands smoke` starts in {home.smoke} would wait on it: `hands first-run` answers them at this terminal")


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


def hears(granted: bool, running: Finding) -> Ready | Missing:
    """Whether hands hears the keys typed in other apps. A running hands does: it starts only with the Input Monitoring
    grant of the app it runs in, hands.app or a terminal's, which a check run elsewhere cannot see. Otherwise, whether
    the app this runs in has the grant."""
    if isinstance(running, Ready):
        return Ready("hands is running, so it started with the Input Monitoring grant of the app it runs in, and hears the talk key (Right Shift)")
    return grant(granted)


def grant(granted: bool) -> Ready | Missing:
    """Whether the app this runs in, hands.app or a terminal's, may show hands the keys typed in other apps."""
    if granted:
        return Ready("the app this runs in has the Input Monitoring grant, so hands run here hears the talk key (Right Shift)")
    return Missing(
        "the app this runs in has no Input Monitoring grant, so hands run here cannot hear the talk key (Right Shift). Grant it "
        "to that app, hands.app or the terminal's, in System Settings > Privacy & Security > Input Monitoring; `hands grant`, run in a "
        "terminal, gives it to that terminal"
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
        return Unfindable(f"this PATH has no `claude` of its own, apart from any hands shim; the install command puts it in: {INSTALL}")
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
    # [LAW:nothing-unseen] each session is said under what it is to hands, one that is typed into with the way it is.
    headed = ((Missing, "hands cannot reach these"), (Unknown, "whether hands can reach these is unknown"), (Ready, "these are typed into"))
    grouped = [(kind, heading, [line.said for line in lines if type(line) is kind]) for kind, heading in headed]
    said = "".join(f"\n  {heading}:" + "".join(f"\n    {line}" for line in group) for _, heading, group in grouped if group)
    # [LAW:no-silent-failure] the step is the worst any session is to hands: one that could not be looked for is never taken for none.
    verdict = next((kind for kind, _, group in grouped if group), Ready)
    return verdict(f"running sessions hands knows of: {len(running)}{beside}{said}")


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
