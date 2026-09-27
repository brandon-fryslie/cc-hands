"""Whether hands is set up to work here: each piece it needs, named, and found or missing.

    hands check     # one line a piece; exits 0 only when every piece is there

The pieces are the plugin that joins sessions to hands, the claude shim that runs them under fritter, the Input
Monitoring grant that lets hands hear the talk key, and the running sessions themselves. `hands run` says the same
lines as it starts, and a daemon that is up says nothing about any of them, so this is where a missing one is heard.
"""

import os
import shutil
import subprocess
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

from hands.core.events import Attached
from hands.core.session import Membership
from hands.sessions import liveness, wrapper
from hands.sessions.hookconfig import PLUGIN_ID
from hands.sessions.home import Home
from hands.sessions.payload import Payload, Rejected
from hands.sessions.processes import process_starts


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


def check(home: Home, path: str, granted: bool) -> list[Finding]:
    """Every piece, in the order a user sets them up. `path` is the PATH sessions are started from; `granted`, this terminal's grant."""
    # [LAW:dataflow-not-control-flow] every piece is looked at every time: one that is missing hides none after it.
    return [plugin(path), shim(home, path), grant(granted), sessions(home)]


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
        f"join hands: `claude plugin install {PLUGIN_ID}` (after `claude plugin marketplace add` of this checkout), then "
        f"/reload-plugins in each running session"
    )


def shim(home: Home, path: str) -> Ready | Missing:
    """Whether `claude` on this PATH is a hands shim whose fritter is there, so it starts every interactive session under fritter."""
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
        case _:
            return Ready(f"`claude` on this PATH is hands' shim, {found}: every interactive session started from it can be typed into")


def grant(granted: bool) -> Ready | Missing:
    """Whether this terminal's app may show hands the keys typed in other apps."""
    if granted:
        return Ready("this terminal has the Input Monitoring grant, so hands run here hears the talk key (Right Shift)")
    return Missing(
        "this terminal has no Input Monitoring grant, so hands run here cannot hear the talk key (Right Shift). Grant it "
        "to the app this terminal runs in, in System Settings > Privacy & Security > Input Monitoring, and restart that app"
    )


def sessions(home: Home) -> Finding:
    """Whether every running session hands knows of can be typed into. Nothing is removed or dialled."""
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
    return sessions_found(running, listening, unreadable)


def sessions_found(running: Sequence[Membership], listening: Collection[Path], unreadable: Sequence[liveness.Unreadable]) -> Ready | Missing:
    """What the running sessions are to hands, given which fritter sockets are there and which files did not parse."""
    unreached = [
        *(line for member in running for line in _untypable(member, listening)),
        *(f"{file.path} names no session hands can read ({file.error}): hands run removes it" for file in unreadable),
    ]
    known = f"running sessions hands knows of: {len(running)}"
    if unreached:
        return Missing(f"{known}, and hands cannot reach these:" + "".join(f"\n    {line}" for line in unreached))
    return Ready(f"{known}, and each can be typed into")


def _untypable(member: Membership, listening: Collection[Path]) -> list[str]:
    where = f"{member.cwd} (pid {member.pid})"
    restart = "restart it from a PATH whose `claude` is hands' shim"
    match member.fritter:
        case None:
            return [f"{where} was started outside fritter, so it cannot be typed into: {restart}"]
        case socket if socket not in listening:
            return [f"{where} has lost its fritter, whose socket {socket} is gone, so it cannot be typed into: {restart}"]
        case _:
            return []
