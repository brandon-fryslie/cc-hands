"""Whether hands is set up to work here: each step of the README's install, named, and done or missing.

    hands check     # one line a step; exits 0 only when every step is done

The steps, in the README's order: Claude Code from its installer; PortAudio, which the microphone opens through; the
installed `hands` on PATH, which Claude Code runs for the plugin; the claude shim that runs sessions under fritter;
the plugin that joins sessions to hands; a backend with its key or login; LowTalker serving transcription; the Input
Monitoring grant that lets hands hear the talk key; hands running; and the running sessions themselves. `hands run` says the same lines as it starts,
and a daemon that is up says nothing about any of them, so this is where a missing one is heard.
"""

import filecmp
import io
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
import wave
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from hands.core.events import Attached
from hands.core.session import Membership
from hands.daemon.backend import backend as resolve
from hands.daemon.config import load
from hands.sessions import heartbeat, liveness, wrapper
from hands.sessions.hookconfig import PLUGIN_ID
from hands.sessions.home import Home
from hands.sessions.payload import Payload, Rejected
from hands.sessions.processes import process_starts
from hands.sessions.terminals import Terminal, terminal_processes
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

# `claude plugin list` answers in a quarter of a second; one that has not answered in this long is not going to.
LIST_TIMEOUT_SECONDS = 20.0
# `hands --version` imports hands' CLI, a couple of seconds; one that has not answered in this long is not going to.
VERSION_TIMEOUT_SECONDS = 30.0
# LowTalker answers silence at once; one still loading its model answers 503 at once.
TRANSCRIBE_TIMEOUT_SECONDS = 10.0
INSTALL_CLAUDE = "`curl -fsSL https://claude.ai/install.sh | bash`"


def check(home: Home, path: str, granted: bool, reached: Finding, heard: Finding, running: Finding) -> list[Finding]:
    """Every step, in the README's order. `path` is the PATH sessions are started from; `granted`, this terminal's grant;
    `reached`, whether the settings' backend has its key or login; `heard`, whether their transcription server
    transcribes; `running`, whether hands is up."""
    # [LAW:dataflow-not-control-flow] every step is looked at every time: one that is missing hides none after it.
    return [claude(path), portaudio(), installed(path), shim(home, path), plugin(path), reached, heard, grant(granted), running, sessions(home, path)]


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
        return Missing(f"PyAudio cannot load PortAudio ({error}), so hands cannot open the microphone: `brew install portaudio`")
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


def configured(home: Home, environment: Mapping[str, str]) -> tuple[Finding, Finding]:
    """Whether the backend the home's config.toml names has the key or the login it reaches its model with, and whether
    its transcription server transcribes."""
    try:
        config = load(home).config
    except Rejected as error:
        unread = Missing(f"hands cannot read its settings: {error}")
        return unread, unread
    try:
        reached = reaching(resolve(config.llm, home, environment))
    except Rejected as error:
        reached = Missing(f"hands has no model to talk with: {error}")
    return reached, transcription(config.transcription)


def reaching(reached: backends.LLMBackend) -> Ready:
    """What a backend that has its key or login reaches, never its key."""
    match backends.account(reached):
        case None:
            return Ready(f"the [llm] backend reaches {reached.model} at {backends.server(reached)} with its key")
        case account:
            return Ready(f"the brain is logged in as {account}, and reaches {reached.model}")


def transcription(url: str) -> Finding:
    """Whether the transcription server at `url` transcribes: a quarter second of silence uploaded as a hold is, which
    LowTalker answers with no words."""
    silence = io.BytesIO()
    with wave.open(silence, "wb") as written:
        written.setnchannels(1)
        written.setsampwidth(2)
        written.setframerate(16_000)
        written.writeframes(bytes(8_000))
    boundary = "hands-check"
    body = b"".join(
        [
            f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\nwhisper-1\r\n'.encode(),
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="silence.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode(),
            silence.getvalue(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    upload = urllib.request.Request(f"{url}/audio/transcriptions", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    # [LAW:no-silent-failure] what the server said back is said, so a refusal names its own cause.
    try:
        with urllib.request.urlopen(upload, timeout=TRANSCRIBE_TIMEOUT_SECONDS) as answered:
            answered.read()
    except urllib.error.HTTPError as error:
        said = error.read().decode(errors="replace").strip()
        if error.code == 503:
            return Missing(f"the transcription server at {url} is not ready (503: {said}): LowTalker answers once its menu says the model is ready")
        return Missing(f"the transcription server at {url} answered {error.code} to a hold: {said}")
    except urllib.error.URLError as error:
        return Missing(
            f"nothing transcribes at {url} ({error.reason}), so hands cannot hear what is said: install LowTalker's network build "
            f"(https://github.com/brandon-fryslie/low-talker) and switch Serve Transcription on in its menu"
        )
    except (OSError, TimeoutError) as error:
        return Unknown(f"cannot tell whether {url} transcribes: {error}")
    return Ready(f"the transcription server at {url} transcribes a hold")


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
    """Whether `claude` on this PATH is a hands shim whose fritter is the one this hands carries, so it starts every
    interactive session under fritter."""
    found = shutil.which("claude", path=path)
    fritter = None if found is None else wrapper.fritter_of(Path(found))
    # [LAW:dataflow-not-control-flow] what to do is read off what is there: an installed shim wants only the PATH.
    first = f'put {home.bin} first on PATH: export PATH="{home.bin}:$PATH"'
    fix = first if wrapper.fritter_of(home.shim) is not None else f"run `hands install-fritter`, then {first}"
    match fritter:
        case None:
            return Missing(f"`claude` on this PATH is {found or 'nothing'}, not hands' shim, so no session started from it can be typed into: {fix}")
        case runs if not os.access(runs, os.X_OK):
            return Missing(f"`claude` on this PATH is hands' shim, {found}, but its fritter {runs} is not there to run, so every interactive claude fails to start: run `hands install-fritter`")
        case runs:
            # [LAW:one-source-of-truth] the fritter in a home's bin is a copy of the packaged one, so a copy that has
            # drifted from it, as one does when hands is upgraded or a checkout's fritter rebuilt, is said, never trusted.
            try:
                current = filecmp.cmp(runs, wrapper.PACKAGED, shallow=False)
            except OSError as error:
                return Unknown(f"cannot tell whether {runs} is the fritter this hands carries, {wrapper.PACKAGED}: {error}")
            if not current:
                return Missing(f"`claude` on this PATH is hands' shim, {found}, but its fritter {runs} is not the one this hands carries, {wrapper.PACKAGED}: run `hands install-fritter`")
            return Ready(f"`claude` on this PATH is hands' shim, {found}: every interactive session started from it can be typed into")


def grant(granted: bool) -> Ready | Missing:
    """Whether this terminal's app may show hands the keys typed in other apps."""
    if granted:
        return Ready("this terminal has the Input Monitoring grant, so hands run here hears the talk key (Right Shift)")
    return Missing(
        "this terminal has no Input Monitoring grant, so hands run here cannot hear the talk key (Right Shift). Grant it "
        "to the app this terminal runs in, in System Settings > Privacy & Security > Input Monitoring, and restart that app"
    )


def sessions(home: Home, path: str) -> Finding:
    """Whether every running session can be typed into, those hands knows of and those it does not. Nothing is removed or dialled."""
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
    return sessions_found(running, listening, unreadable, unrecorded(home, path, {member.pid for member in running}))


@dataclass(frozen=True)
class Unfindable:
    said: str  # why a running session hands has no record of cannot be told from the other programs at a terminal


@dataclass(frozen=True)
class Unjoined:
    """A running session hands has no record of, and whether hands could type into it once it joins."""

    process: Terminal
    under_fritter: bool


def unrecorded(home: Home, path: str, members: Collection[int]) -> list[Unjoined] | Unfindable:
    """The sessions running at a terminal that hands has no record of, among this user's processes now."""
    match claude_code(wrapper.real_claude(path)):
        case Unfindable() as unfindable:
            return unfindable
        case executable:
            try:
                terminals = terminal_processes()
            except OSError as error:
                return Unfindable(f"cannot look at this user's processes at a terminal: {error}")
            return unjoined(home, executable, config_dir(os.environ), terminals, members)


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


def config_dir(environment: Mapping[str, str]) -> Path:
    """The Claude Code config a process with this environment runs under: its plugins, and so whether hands' is one."""
    return Path(environment.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")


def unjoined(home: Home, claude: Path, config: Path, terminals: Sequence[Terminal], members: Collection[int]) -> list[Unjoined]:
    """The sessions at a terminal under `config` that no running membership names: started before the plugin, and not
    reloaded since.

    A session is a process at a terminal running `claude`, the Claude Code executable, or any other version of it, since
    an update leaves running sessions on the version they started on; one whose parent is one is that session's own
    helper. The hook records that same process, so it is matched by pid. One under another config, as the brain is,
    has other plugins, and is no session of the plugin this check looks at.
    """
    by_pid = {process.pid: process for process in terminals}
    install = _unversioned(claude)

    def runs_claude(pid: int) -> bool:
        return pid in by_pid and _unversioned(by_pid[pid].executable) == install

    fritter = home.fritter.resolve()
    return [
        Unjoined(process, process.parent in by_pid and by_pid[process.parent].executable == fritter)
        for process in terminals
        if runs_claude(process.pid) and not runs_claude(process.parent) and config_dir(process.environment) == config and process.pid not in members
    ]


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
    unrecorded: Sequence[Unjoined] | Unfindable,
) -> Finding:
    """What the running sessions are to hands, given which fritter sockets are there, which files did not parse, and
    which sessions at a terminal hands has no record of."""
    match unrecorded:
        case Unfindable(said):
            unknown, unseen = [], [f"a session hands has no record of cannot be found: {said}"]
        case sessions:
            unknown, unseen = [_unjoined(session) for session in sessions], []
    unreached = [
        *(line for member in running for line in _untypable(member, listening)),
        *(f"{file.path} names no session hands can read ({file.error}): hands run removes it" for file in unreadable),
        *unknown,
    ]
    known = f"running sessions hands knows of: {len(running)}"
    # [LAW:no-silent-failure] sessions that could not be looked for are said, never taken for none.
    if unreached:
        return Missing(f"{known}, and hands cannot reach these:" + "".join(f"\n    {line}" for line in [*unreached, *unseen]))
    if unseen:
        return Unknown(f"{known}, and each can be typed into, but {unseen[0]}")
    return Ready(f"{known}, and each can be typed into")


def _unjoined(session: Unjoined) -> str:
    where = f"{session.process.cwd} (pid {session.process.pid}) is a session hands has no record of"
    if session.under_fritter:
        return f"{where}, so it cannot be reached: /reload-plugins in it"
    return f"{where}, started outside fritter, so it cannot be typed into: {_RESTART}"


_RESTART = "restart it from a PATH whose `claude` is hands' shim"


def _untypable(member: Membership, listening: Collection[Path]) -> list[str]:
    where = f"{member.cwd} (pid {member.pid})"
    match member.fritter:
        case None:
            return [f"{where} was started outside fritter, so it cannot be typed into: {_RESTART}"]
        case socket if socket not in listening:
            return [f"{where} has lost its fritter, whose socket {socket} is gone, so it cannot be typed into: {_RESTART}"]
        case _:
            return []
