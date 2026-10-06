"""Starting a Claude Code session for the user, as a person at a terminal starts one: the `start_session` tool.

The session runs `claude` in a tmux window of the tmux session named for its folder, made when there is none, so it has a
terminal the user can attach to from any terminal app, and hands can find and type into as it does any session's. It is
started by the daemon, in the environment hands was started in, which is the user's: never from the brain's shell, whose
environment hands made the brain's and holds none of the user's credentials. What a session the daemon itself was started
inside gave it is left out. The start returns once the session joined hands, naming it, or says why it did not.
"""

import asyncio
import shlex
import shutil
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hands.core.session import Membership, SessionId
from hands.sessions import audit, wide
from hands.sessions.child import Ran, run
from hands.sessions.home import Home
from hands.sessions.terminals import Process, process_table
from hands.sessions.untap import untap_script, untapped

# What a session gives each process it runs, naming itself as their parent: Claude Code's (2.1.288), and fritter's
# address. A `claude` started under Claude Code's is a child of that session, not one of its own: it writes its turns
# into the parent's transcript, and its own is never made, so hands has nothing to read; and one started outside hands'
# shim under fritter's would join hands as that session's.
SESSION_GIVEN = (
    "FRITTER_SOCKET", "CLAUDECODE", "CLAUDE_PID", "CLAUDE_EFFORT", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_EXECPATH", "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_SESSION_ATTENDED", "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_CODE_TMUX_TRUECOLOR",
)

# How long a started session has to join hands: Claude Code's start and its first hook. A folder it has never been
# trusted in holds it at the trust dialog, which no hook outlasts.
JOIN_SECONDS = 30.0
# How often the sessions and the session's pane are looked at while it joins.
LOOK_SECONDS = 0.25
# What tmux says when the session a new one would be named for is there already (tmux 3.6).
DUPLICATE = "duplicate session"
# How long tmux has to answer: a server that does not is wedged, and the start says so.
ANSWER_SECONDS = 5.0


class NotStarted(Exception):
    """The session was not started, or did not join hands. The message says why, with what its pane showed when it had one."""


@dataclass(frozen=True)
class Started:
    session: SessionId
    # The tmux session the window is in, and the pane `claude` runs in, as tmux names them (`%12`).
    tmux_session: str
    pane: str


def as_from_a_terminal(environment: Mapping[str, str]) -> dict[str, str]:
    """`environment` as a terminal outside any session has it: no tap, and nothing else of a session it may be inside."""
    return {name: value for name, value in untapped(environment).items() if name not in SESSION_GIVEN}


def as_from_a_terminal_script(claude: Path) -> str:
    """as_from_a_terminal, as sh run in the environment to clean, from the same names, ending by running `claude` with the
    script's arguments."""
    # [LAW:one-source-of-truth] the same names as as_from_a_terminal.
    return f'{untap_script()}unset {shlex.join(SESSION_GIVEN)}\nexec {shlex.quote(str(claude))} "$@"\n'


def joined(members: Iterable[Membership], root: int, table: Mapping[int, Process]) -> Membership | None:
    """The session among `members` whose process is `root` or one it started, by the process `table`: a session already
    running when `root` was started is never one."""
    return next((member for member in members if descends(member.pid, root, table)), None)


def descends(pid: int, root: int, table: Mapping[int, Process]) -> bool:
    """Whether process `pid` is `root` or one of the processes it started, by each process's parent in `table`."""
    while pid != root:
        if pid not in table:
            return False
        pid = table[pid].parent
    return True


async def start(home: Home, record: audit.Record, folder: Path, model: str | None, environment: Mapping[str, str], members: Callable[[], Iterable[Membership]]) -> Started:
    """Start hands' `claude` in `folder` on `model`, or on its own default with none, and wait until it is among the
    `members` of hands."""
    # [LAW:nothing-unseen] a start is one unit of work: where, on what, in which tmux session and pane, which session
    # joined, and whether tmux's session was made for it.
    with wide.unit("session.start", record):
        wide.annotate(folder=folder, model=model)
        where = _folder(folder)
        terminal = as_from_a_terminal(environment)
        tmux = shutil.which("tmux", path=terminal.get("PATH"))
        if tmux is None:
            raise NotStarted("there is no tmux on PATH, and hands starts a session in a tmux window")
        name = tmux_name(where)
        # [LAW:single-enforcer] the window runs `claude` as from a terminal outside any session whatever the tmux server
        # it opens in holds: one already running gives a window its own environment, which a server started inside a
        # session holds that session's in. Through sh, never the user's shell, whose startup files would set a PATH of
        # their own. hands' own `claude`, its shim, whatever comes first on the window's PATH: the session runs under
        # fritter, so hands can type into it.
        claude = ["/bin/sh", "-c", as_from_a_terminal_script(home.bin / "claude"), "claude", *(() if model is None else (f"--model={model}",))]
        made, pane, pid = await _opened(tmux, terminal, name, where, ("-e", f"HANDS_HOME={home.root}", "--", *claude))
        wide.annotate(tmux_session=name, made_tmux_session=made, pane=pane)
        member = await _joined(members, where, tmux, terminal, pane, pid)
        wide.annotate(session=member.id, under_fritter=member.fritter is not None)
        if member.fritter is None:
            raise NotStarted(f"session {member.id} joined hands in tmux pane {pane}, but not under fritter, so hands cannot type into it: {home.bin / 'claude'} did not wrap it")
        return Started(member.id, name, pane)


def _folder(folder: Path) -> Path:
    # [LAW:parse-dont-validate] the daemon's own working directory is no folder the user named.
    if not folder.expanduser().is_absolute():
        raise NotStarted(f"{folder} is no absolute folder, nor one from ~: give the folder's whole path")
    try:
        resolved = folder.expanduser().resolve(strict=True)
    # A symlink loop is a RuntimeError before Python 3.13.
    except (OSError, RuntimeError) as error:
        raise NotStarted(f"there is no folder {folder}: {error}") from error
    if not resolved.is_dir():
        raise NotStarted(f"{folder} is not a folder")
    return resolved


def tmux_name(folder: Path) -> str:
    """The tmux session a session in `folder` is started in: named for the folder, as tmux would spell it (tmux turns a
    '.' or ':', which its targets read as separators, into '_')."""
    return folder.name.replace(".", "_").replace(":", "_")


async def _opened(tmux: str, terminal: Mapping[str, str], name: str, folder: Path, command: Sequence[str]) -> tuple[bool, str, int]:
    """Open a new window, detached, of tmux session `name`, made when it is not there, running what `command` ends with:
    whether it was made, and the window's pane and the pid of the process it runs."""
    shown = ("-d", "-P", "-F", "#{pane_id} #{pane_pid}")
    # [LAW:no-ambient-temporal-coupling] made or not is what tmux answers the making with, never a look beforehand that a
    # second start could make stale.
    made = await _tmux(tmux, terminal, "new-session", *shown, "-s", name, "-c", str(folder), *command)
    if made.returncode == 0:
        return True, *_pane(made)
    if not _said(made).startswith(DUPLICATE):
        raise NotStarted(f"tmux could not make session {name}: {_said(made)}")
    window = await _tmux(tmux, terminal, "new-window", *shown, "-t", f"={name}:", "-c", str(folder), *command)
    if window.returncode != 0:
        raise NotStarted(f"tmux could not open a window in session {name}: {_said(window)}")
    return False, *_pane(window)


def _pane(opened: Ran) -> tuple[str, int]:
    pane, pid = opened.out.decode().split()
    return pane, int(pid)


async def _joined(members: Callable[[], Iterable[Membership]], folder: Path, tmux: str, terminal: Mapping[str, str], pane: str, pid: int) -> Membership:
    deadline = time.monotonic() + JOIN_SECONDS
    while (member := joined(members(), pid, _processes())) is None:
        # list-panes, since display-message answers for a pane that is gone as though it were there (tmux 3.6).
        state = await _tmux(tmux, terminal, "list-panes", "-t", pane, "-f", f"#{{==:#{{pane_id}},{pane}}}", "-F", "#{pane_dead}")
        if state.returncode != 0:
            raise NotStarted(f"`claude` in {folder} ended before it joined hands, and its window closed with it")
        if state.out.decode().strip() == "1":
            raise NotStarted(f"`claude` in {folder} ended before it joined hands; tmux pane {pane} showed:\n{await _shown(tmux, terminal, pane)}")
        if time.monotonic() > deadline:
            raise NotStarted(f"`claude` in {folder} has not joined hands in {JOIN_SECONDS:.0f} seconds, and is still running in tmux pane {pane}, which shows:\n{await _shown(tmux, terminal, pane)}")
        await asyncio.sleep(LOOK_SECONDS)
    return member


async def _shown(tmux: str, terminal: Mapping[str, str], pane: str) -> str:
    shown = await _tmux(tmux, terminal, "capture-pane", "-p", "-t", pane)
    return shown.out.decode().rstrip() if shown.returncode == 0 else f"(tmux could not show it: {_said(shown)})"


def _said(ran: Ran) -> str:
    return ran.err.decode(errors="replace").strip()


async def _tmux(tmux: str, terminal: Mapping[str, str], *arguments: str) -> Ran:
    # The terminal's environment: a tmux server this starts takes it as every window's, so it holds nothing of hands'.
    # The home the session reports to is given its window by name.
    try:
        return await run(tmux, *arguments, timeout=ANSWER_SECONDS, env=terminal)
    except TimeoutError as error:
        raise NotStarted(f"tmux {arguments[0]} did not answer in {ANSWER_SECONDS:.0f} seconds") from error


def _processes() -> dict[int, Process]:
    try:
        return process_table()
    except OSError as error:
        raise NotStarted(f"the processes could not be read to tell which session is the one started: {error}") from error
