"""The brain's process: one long-lived, slim Claude Code on the subscription, behind hands' proxy, reaching hands over MCP.

It is Claude Code as anyone runs it: interactive, on a terminal hands holds, under fritter, and asked nothing a person
at its keyboard could not ask. A turn is typed into its input and sent with Return, and an interrupt is Escape. Nothing
else is typed into it: its input is the user's, and what hands asks in the background is asked of a Claude Code of its
own (`hands.brain.asides`). What it says is read from the wire, not from its screen [the design's rule: primary facts
from the wire, derivative ones from the harness], and the harness is heard only through the hooks it posts to a
listener of hands' own: that a typed turn was taken, and that it ended, or that the API failed it.

Its login, settings, and skills live in a directory hands owns, set up once, as any Claude Code is, by running it there:

    mkdir -p ~/.hands/brain/cwd && cd ~/.hands/brain/cwd && CLAUDE_CONFIG_DIR=~/.hands/brain claude

and it runs in that empty directory of hands' own, never in a project. What it may use, what it may do without asking,
and which MCP servers it has are that directory's to say, as they are for any Claude Code: its settings.json and its
.claude.json. hands adds only its own server and hooks, and keeps out what the login brings from the account.
"""

import asyncio
import contextlib
import fcntl
import json
import os
import pty
import re
import shutil
import struct
import subprocess
import tempfile
import termios
import threading
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import web
from loguru import logger

from hands.brain.mcp import SERVER_NAME
from hands.core.effects import Allow, Deny, Text
from hands.core.session import ESCAPES, Permission, SessionId, pasted
from hands.core.wire import MainTurn, Observed, Sent, tool_names
from hands.sessions.audit import BrainAnswered, BrainAsked, BrainExited, BrainLaunched, BrainOffered, BrainPermission, BrainRefused, Record
from hands.sessions.hookconfig import PERMISSION_DEADLINE_SECONDS, declared
from hands.sessions.hooks import called, hook_output
from hands.sessions.payload import Payload, Rejected
from hands.sessions.typing import Typist, Untyped
from hands.sessions.untap import untapped
from hands.sessions.wrapper import real_claude

# What --bare would have switched off, switched off one by one so the OAuth login stays on (hands-wire-6ic.8wu, 2.1.284).
# LSP needs no switch: it comes only from plugins, and the brain's own setup installs none.
SLIM = {
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION": "false",
    # A background command's notification opens a turn of its own, under a prompt id no turn hands typed carries, which
    # would take a typed turn's place or leave it never ending (hands-wire-6ic.99l review).
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    # A scheduled prompt opens a turn of its own the same way when it fires (hands-brain-d8g.9d6 review, 2.1.288).
    "CLAUDE_CODE_DISABLE_CRON": "1",
    # The account's claude.ai connectors are Brandon's, never the brain's: loaded, they joined every request after the
    # first (turn 2 grew from 8.7 KB to 112 KB; hands-wire-6ic.eph, 2.1.284). The skills and plugins the account syncs
    # are turned off in the brain's settings.json, the only place Claude Code reads that switch from (2.1.288).
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
}

# Credentials Claude Code prefers to its own login. Inherited from hands' environment, any of them would put the brain
# on another account or off the subscription without a word, so none is passed on.
FOREIGN_CREDENTIALS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")

# A dialog nobody can see would hold its turn open forever with no hook to say so (2.1.288). A permission its setup asks
# about is held at its hook while the user is asked by voice, in a turn of theirs; anything else is answered no there,
# and the brain hears why.
NOBODY = "Nobody can be asked: no turn of the user's is in flight to ask in, so what this Claude Code's settings would ask about is refused."
UNREAD = "hands could not read this permission request, so it is refused."
UNVOICED = "Nobody sees this dialog: ask the user in your reply instead, and they will answer in their next turn."
UNANSWERED = "No answer came from the user in time, so it did not run."
SPOKEN_OVER = "The user spoke over this turn before they could be asked, so it did not run."
STOPPED = "The user stopped this turn, so it did not run."
DECLINED: Mapping[str, object] = {"hookSpecificOutput": {"hookEventName": "Elicitation", "action": "decline"}}
# What each hook is answered with when its body cannot be read: a dialog is kept shut, and the rest ask nothing.
UNREADABLE: Mapping[str, Mapping[str, object]] = {"PermissionRequest": hook_output(Deny(UNREAD)), "Elicitation": DECLINED}

# What the brain posts to hands, each to its own path: a typed turn taken, a turn ended, a turn the API failed, and a
# dialog about to open. Escape ends a turn with none of them (measured on 2.1.285), so a turn told to stop is over when
# it is told.
HOOKS = ("UserPromptSubmit", "Stop", "StopFailure", "PermissionRequest", "Elicitation")

# The terminal the brain draws on. Nobody looks at it; it is sized so a long line is not wrapped into many.
ROWS, COLS = 50, 200
# How much of what the brain last showed is kept for the line that says it exited.
SHOWN_BYTES = 16 * 1024
SHOWN_LINES = 20
# `claude auth status` answers in about a second; one that has not answered in this long is not going to.
AUTH_STATUS_SECONDS = 20.0
# How long fritter has to start the brain and open its socket.
START_SECONDS = 10.0
# How long the brain's input takes to come up after it starts (about 0.22s in, 2.1.285, measured 2026-09-30). Text
# typed before it is up is lost, and a turn lost so is untaken, which says so.
SETTLE_SECONDS = 0.5
# How long a typed turn has to be taken: a cold start loads the MCP server before the input is read.
TAKE_SECONDS = 30.0
# How far apart two Ctrl-Cs are pressed into the brain: Claude Code exits on a second within 800ms of one that found its
# input empty.
EXIT_SECONDS = 1.0
# How long a Claude Code told to stop has before it is killed.
STOP_SECONDS = 5.0

# What a terminal is told rather than shown, and the keys it is sent.
_CONTROL = re.compile(rf"{ESCAPES.pattern}|[\x00-\x09\x0b-\x1f\x7f]")


@dataclass(frozen=True)
class Station:
    """What every slim Claude Code of hands' own runs on, the brain and each one a side question is asked of."""

    config_dir: Path
    cwd: Path
    model: str
    proxy_url: str
    # hands' own environment, which each is started in less what environment() keeps out; kept out of the repr, as it holds keys.
    inherited: Mapping[str, str] = field(repr=False)


@dataclass(frozen=True)
class Launch:
    """Everything the brain is started with."""

    station: Station
    instruction: str
    mcp_config: str
    # [LAW:one-source-of-truth] chosen by hands, so the brain's requests are known as its own from the first one on the
    # wire, and its hooks from the first one posted.
    session: SessionId
    # hands' own fritter, from `hands install-fritter`: the brain is typed into through it.
    fritter: Path


def slim(claude: Path, model: str, session: SessionId) -> list[str]:
    """A slim Claude Code's command line: the real claude, interactive, reading no settings but its config directory's."""
    return [
        str(claude),
        "--model", model,
        "--session-id", session,
        # The config directory's settings.json is the only settings file read: none from the working directory.
        "--setting-sources", "user",
    ]


def command(launch: Launch, claude: Path, hooks: str) -> list[str]:
    """The brain's command line: a slim Claude Code on its own setup, plus hands' server, instruction, and hooks posted to the listener at `hooks`."""
    return [
        *slim(claude, launch.station.model, launch.session),
        # Beside the MCP servers its own setup names, never in place of them.
        "--mcp-config", launch.mcp_config,
        # After Claude Code's own system prompt, never in place of it: the API checks that it opens as Claude Code's does.
        "--append-system-prompt", launch.instruction,
        # hands' tools are how the brain reaches the sessions at all; a deny rule in its own setup still outranks this.
        "--allowedTools", f"mcp__{SERVER_NAME}",
        # [LAW:single-enforcer] each hook declared as the plugin declares it: a held permission's lives as long as a working
        # session's, and is denied by the same deadline.
        "--settings", json.dumps({"hooks": {event: [{"hooks": [{"type": "http", "url": f"{hooks}/{event}", **declared(event)}]}] for event in HOOKS}}),
    ]


def environment(config_dir: Path, base_url: str, inherited: Mapping[str, str]) -> dict[str, str]:
    """A slim Claude Code's environment: hands' own, less any credential that is not the login in `config_dir`, reaching the API at `base_url`."""
    # [LAW:one-source-of-truth] the brain is the same brain wherever hands was started: a daemon started inside a tapped
    # session does not hand the brain that session's tap.
    kept = {name: value for name, value in untapped(inherited).items() if name not in FOREIGN_CREDENTIALS}
    return {**kept, **SLIM, "CLAUDE_CONFIG_DIR": str(config_dir), "ANTHROPIC_BASE_URL": base_url}


def workdir(config_dir: Path) -> Path:
    """The empty directory a slim Claude Code on the login in `config_dir` runs in, made if it is not there: never a project."""
    cwd = _cwd(config_dir)
    cwd.mkdir(parents=True, exist_ok=True)
    return cwd


def _cwd(config_dir: Path) -> Path:
    return config_dir / "cwd"


class BrainGone(Exception):
    """The brain ended, or cannot be typed into, while a turn waited on it, or before one was asked."""


class Untaken(Exception):
    """The brain was typed a turn and never took it: it is on a screen that is not its input."""


class NotLoggedIn(Exception):
    """The brain's config directory holds no login on the Claude subscription, so every turn would fail or be billed to a key."""


class LoginFailed(Exception):
    """Claude Code's own login did not finish; whatever login the brain held before, it still holds."""


class Unstartable(Exception):
    """A Claude Code of hands' own could not be started: no claude to run, or for the brain no fritter to run it under, or a fritter that never opened its socket."""


def brain_claude(inherited: Mapping[str, str]) -> Path:
    """The Claude Code hands runs as its own: the real claude on PATH, past every hands shim, which would run it as a session."""
    claude = real_claude(inherited.get("PATH", ""))
    if claude is None:
        raise Unstartable("no claude on PATH but hands' shims, so there is no Claude Code for hands to run as its own")
    return claude


def login(config_dir: Path, base_url: str, inherited: Mapping[str, str]) -> str:
    """Log `config_dir` in to the Claude subscription with Claude Code's own login, at this terminal; the account it holds after."""
    # [LAW:one-source-of-truth] the brain's own claude and environment, so the login lands in its config directory,
    # which the daemon reads, and no credential of this shell's stands in for the one being made.
    signed = subprocess.run([brain_claude(inherited), "auth", "login", "--claudeai"], env=environment(config_dir, base_url, inherited))
    if signed.returncode != 0:
        raise LoginFailed(f"`claude auth login` for the brain exited {signed.returncode}")
    return logged_in(config_dir, base_url, inherited)


def account_kept_out(config_dir: Path) -> None:
    """Raises Unstartable unless the brain's settings.json keeps out the skills and plugins its login's account syncs.

    They are Brandon's, never the brain's, and settings.json is the only place Claude Code reads either switch from
    (2.1.288): not --settings, and not the environment. Left out, each is on."""
    settings = config_dir / "settings.json"
    fix = f'set "syncClaudeAiSkills": false and "syncClaudeAiPlugins": false in {settings}'
    try:
        said = Payload.parse(settings.read_bytes())
        synced = [switch for switch in ("syncClaudeAiSkills", "syncClaudeAiPlugins") if said.fields.get(switch) is not False]
    except (OSError, Rejected) as error:
        raise Unstartable(f"the brain's settings could not be read ({error}): {fix}") from None
    if synced:
        raise Unstartable(f"the brain would load its account's {' and '.join(synced)}: {fix}")


def logged_in(config_dir: Path, base_url: str, inherited: Mapping[str, str]) -> str:
    """The subscription account `config_dir` is logged in as; raises NotLoggedIn, naming the command that makes a login, when it has none."""
    try:
        # A timed-out child is killed and reaped by run itself.
        asked = subprocess.run(
            [brain_claude(inherited), "auth", "status"],
            env=environment(config_dir, base_url, inherited),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=AUTH_STATUS_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise NotLoggedIn(f"`claude auth status` for the brain did not answer in {AUTH_STATUS_SECONDS:.0f}s") from None
    try:
        status = Payload.parse(asked.stdout)
        if not status.flag("loggedIn"):
            raise NotLoggedIn(f"the brain has no login; run: {setup(config_dir)}")
        # [LAW:no-silent-failure] a key the config directory resolves would answer every turn, billed to the API.
        if (method := status.text("authMethod")) != "claude.ai":
            raise NotLoggedIn(
                f"the brain is logged in by {method}, not on the Claude subscription: `hands login` puts it there,"
                f" unless {config_dir / 'settings.json'} or hands' environment sets {method} ahead of its login"
            )
        return status.text("email")
    except Rejected as error:
        raise NotLoggedIn(f"`claude auth status` for the brain answered {asked.stdout[:200]!r} {asked.stderr[:200]!r}, not its status: {error}") from None


def setup(config_dir: Path) -> str:
    """The command that sets the brain up as any Claude Code is set up: run once, where it runs."""
    # The directory is made here too: it is asked for before hands has ever run the brain, and so before workdir made it.
    return f"mkdir -p {_cwd(config_dir)} && cd {_cwd(config_dir)} && CLAUDE_CONFIG_DIR={config_dir} claude"


class _Terminal:
    """The terminal a Claude Code of hands' own runs on, held by hands: read as fast as it is written, and the last of it kept."""

    def __init__(self, master: int) -> None:
        self._master = master
        self._shown = b""
        self._lock = threading.Lock()
        loop = asyncio.get_running_loop()
        self.closed: asyncio.Future[None] = loop.create_future()

        def read() -> None:
            # A thread of its own: nothing is drawn when nothing reads, and a terminal's reads block.
            while True:
                try:
                    data = os.read(master, 65536)
                except OSError:
                    data = b""
                if not data:
                    break
                with self._lock:
                    self._shown = (self._shown + data)[-SHOWN_BYTES:]
            os.close(master)
            loop.call_soon_threadsafe(lambda: None if self.closed.done() else self.closed.set_result(None))

        threading.Thread(target=read, name="a Claude Code's terminal", daemon=True).start()

    def last(self) -> str:
        """The last lines it showed, as text."""
        with self._lock:
            shown = self._shown.decode(errors="replace")
        lines = [line.rstrip() for line in _CONTROL.sub("", shown.replace("\r", "\n")).split("\n") if line.strip()]
        return "\n".join(lines[-SHOWN_LINES:])


class ClaudeCode:
    """A slim Claude Code of hands' own, running on a terminal hands holds, until it ends or is stopped."""

    def __init__(self, process: asyncio.subprocess.Process, terminal: _Terminal) -> None:
        self._process = process
        self._terminal = terminal
        # Its exit code, once it has ended and what it showed last is read.
        self.exit = asyncio.ensure_future(self._run_out())

    @property
    def pid(self) -> int:
        return self._process.pid

    def shown(self) -> str:
        """The last lines it showed on its terminal."""
        return self._terminal.last()

    async def stop(self) -> None:
        if self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), STOP_SECONDS)
            except TimeoutError:
                self._kill()
            except asyncio.CancelledError:
                # [LAW:no-silent-failure] a stopper told to leave waits no longer, and leaves nothing running behind it.
                self._kill()
                raise
        await asyncio.shield(self.exit)

    def _kill(self) -> None:
        # One that ended as it was about to be killed is what killing it is for.
        with contextlib.suppress(ProcessLookupError):
            self._process.kill()

    async def _run_out(self) -> int:
        code = await self._process.wait()
        try:
            # What it showed last, read to its end; a terminal some child of it still holds is not waited on for long.
            await asyncio.wait_for(asyncio.shield(self._terminal.closed), 1.0)
        except TimeoutError:
            pass
        return code


async def spawn(station: Station, argv: Sequence[str]) -> ClaudeCode:
    """Run a slim Claude Code's command on a terminal of hands' own, in its own directory and environment."""
    master, slave = pty.openpty()
    try:
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
        process = await asyncio.create_subprocess_exec(
            *_holding_terminal(os.ttyname(slave), argv),
            cwd=station.cwd,
            env={**environment(station.config_dir, station.proxy_url, station.inherited), "TERM": "xterm-256color"},
            stdin=slave,
            stdout=slave,
            stderr=slave,
            start_new_session=True,
        )
    except BaseException:
        os.close(master)
        raise
    finally:
        os.close(slave)
    try:
        await _held(master, process)
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        os.close(master)
        raise
    return ClaudeCode(process, _Terminal(master))


async def _held(terminal: int, process: asyncio.subprocess.Process) -> None:
    """Until `process` holds `terminal` as its session's, or has ended without taking it.

    [LAW:no-ambient-temporal-coupling] what is spawned is ended by hands' end only once it holds its terminal: a hands
    that died before then would hang up nothing, and leave the child opening a terminal with no other end, for good."""
    while process.returncode is None and os.tcgetpgrp(terminal) != process.pid:
        await asyncio.sleep(0.002)


def _holding_terminal(terminal: str, argv: Sequence[str]) -> list[str]:
    """argv, run so that its terminal is its session's controlling terminal: hands' end, however it comes, hangs the
    terminal up and ends what runs on it, as closing a window does. Without it, a hands that dies without stopping it
    leaves it running for good.

    A session leader with no controlling terminal takes the first terminal it opens, so the shell opens it and execs
    argv in its place, keeping its pid. [LAW:no-ambient-temporal-coupling] no Python runs between fork and exec, where
    a lock another of hands' threads held at the fork would hang the child, and hands with it."""
    return ["/bin/sh", "-c", ': <>"$0"; exec "$@"', terminal, *argv]


# Compared by identity: each is its own request, however alike two calls are.
@dataclass(frozen=True, eq=False)
class Asked:
    """A permission the brain's own setup asks about, held at its hook while the user is asked: settled once, by what they
    answer, by nobody answering in time, or by the turn's end, whichever comes first."""

    permission: Permission
    decision: "asyncio.Future[Allow | Deny]"

    @property
    def open(self) -> bool:
        return not self.decision.done()

    def settle(self, decision: Allow | Deny) -> None:
        # The first to settle it answers its hook; any later one comes after that answer went.
        if self.open:
            self.decision.set_result(decision)


@dataclass(frozen=True)
class _Posted:
    """A hook the brain posted, to the path of its event, and the body its post is answered with."""

    event: str
    said: Payload
    reply: "asyncio.Future[Mapping[str, object]]"


@dataclass
class _Turn:
    answered: asyncio.Future[BrainAnswered]
    taken: asyncio.Future[str]  # the prompt id Claude Code gave the turn when it took it
    # Told of each permission the turn holds at its hook, to put it to the user.
    asks: Callable[[Asked], None]

    @property
    def prompt(self) -> str | None:
        return self.taken.result() if self.taken.done() else None


class Brain:
    """A running brain: a turn is asked with `ask`, which returns once the brain has said the turn is over."""

    def __init__(
        self,
        claude: ClaudeCode,
        session: SessionId,
        typist: Typist,
        hooks: "asyncio.Queue[_Posted]",
        listener: web.AppRunner,
        sockets: Path,
        config_dir: Path,
        record: Record,
    ) -> None:
        self._claude = claude
        self._config_dir = config_dir
        self.session = session
        self._typist = typist
        self._listener = listener
        self._sockets = sockets
        self._record = record
        # The tools its requests last offered, so the audit says them once and again only when they change.
        self._offered: tuple[str, ...] | None = None
        # [LAW:single-enforcer] the input is the user's, and only two things are ever typed into it: a turn, and the
        # keys that stop one. A turn keeps it until its hook says Claude Code took the turn, and a stop's keys go in alone.
        self._input = asyncio.Lock()
        # [LAW:no-ambient-temporal-coupling] the turn in flight is the brain's own state, not its asker's: it is over
        # when its hook says so, whether or not anyone still waits on it, and the next is typed only then.
        self._turn: _Turn | None = None
        # The permissions held at their hooks now: one for each call of a reply that asks. Nothing is typed into the brain
        # while one is held but the Escape that stops its turn: what is typed goes to the dialog Claude Code draws, whose
        # Return says yes (2.1.288, spike hands-brain-d8g.aur). A turn is typed only once the last has ended, and the stage
        # presses no Escape while the user is being asked.
        self._held: set[Asked] = set()
        self._typing: set[asyncio.Task[None]] = set()
        # When the last stop pressed Ctrl-C, on the event loop's clock.
        self._cleared = float("-inf")
        self._heard = asyncio.ensure_future(self._hear_hooks(hooks))
        self._exit = asyncio.ensure_future(self._run_out())

    @property
    def pid(self) -> int:
        return self._claude.pid

    async def ask(self, text: str, asks: Callable[[Asked], None]) -> BrainAnswered:
        """One turn of the brain's, from its typing to its end; `asks` is told of each permission the turn holds for the user."""
        while self._turn is not None:
            await asyncio.wait({self._turn.answered})
        if self._exit.done():
            raise BrainGone(f"the brain had exited ({self._exit.result()}) before it was asked")
        loop = asyncio.get_running_loop()
        turn = self._turn = _Turn(loop.create_future(), loop.create_future(), asks)
        # An asker that stops waiting leaves the turn to be typed and to run to its end, which is still the brain's to hear.
        self._keep(self._send(text, turn))
        return await asyncio.shield(turn.answered)

    def _keep(self, typing: Coroutine[None, None, None]) -> None:
        """Runs on in a task of the brain's own, whoever asked for it."""
        task = asyncio.create_task(typing)
        self._typing.add(task)
        task.add_done_callback(self._typing.discard)

    async def _send(self, text: str, turn: _Turn) -> None:
        try:
            # [LAW:no-ambient-temporal-coupling] the input is the turn's from its first key until Claude Code says it took
            # the turn. Claude Code reads keys that reach it together as one paste, and a Return inside a paste sends
            # nothing: what was typed 10ms behind a turn joined the turn's prompt, or was left in the input with it
            # (2.1.286, measured 2026-09-30, 2 of 4; none of 8 typed once the turn was taken).
            async with self._input:
                # Behind a space, as every prompt hands types: a leading / or ! is then the character it is.
                await self._type(lambda typist: typist.type(Text(pasted(text)).typed))
                self._record(BrainAsked(text))
                await asyncio.wait({turn.taken, turn.answered}, timeout=TAKE_SECONDS, return_when=asyncio.FIRST_COMPLETED)
            if not (turn.taken.done() or turn.answered.done()):
                raise Untaken(f"the brain did not take the turn typed into it in {TAKE_SECONDS:.0f}s; if it is on a setup screen, run: {setup(self._config_dir)}")
        except (BrainGone, Untaken) as error:
            self._over(turn, error)

    def interrupt(self) -> None:
        """Stop the turn in flight with Escape, as at the keyboard. Returns at once: the Escape is the brain's to press,
        and no hook says a turn was stopped, so the turn ends where it is pressed."""
        turn = self._turn
        if turn is not None:
            self._keep(self._stop(turn))

    async def _stop(self, turn: _Turn) -> None:
        # [LAW:no-ambient-temporal-coupling] Escape goes once Claude Code has taken the turn, never before: a turn ended
        # while its UserPromptSubmit hook is still coming would leave that hook to be taken for the next turn's.
        await asyncio.wait({turn.taken, turn.answered}, timeout=TAKE_SECONDS, return_when=asyncio.FIRST_COMPLETED)
        if self._turn is not turn:
            return
        if not turn.taken.done():
            # [LAW:no-silent-failure] the turn runs on, told to stop by nobody: its words are the stage's to hold.
            logger.warning(f"the brain was not stopped: its turn was not taken in {TAKE_SECONDS:.0f}s")
            return
        try:
            async with self._input:
                if self._turn is not turn:
                    return
                await self._type(lambda typist: typist.press("escape"))
                # Escape refuses a permission held at its dialog and leaves its hook unanswered for good (2.1.288, spike
                # hands-brain-d8g.aur): it is settled here, and nothing can answer it later.
                self._settle(Deny(STOPPED))
                # Escape puts a prompt stopped before any reply back in the input, which the next turn typed would join;
                # Ctrl-C clears it. On an input left empty it arms Claude Code's exit instead, which a second Ctrl-C within
                # 800ms takes, whatever was typed between (2.1.285, read from its source 2026-09-30); so no two are
                # pressed that close, counted from when each went and with room for Claude Code to read the first late.
                loop = asyncio.get_running_loop()
                await asyncio.sleep(self._cleared + EXIT_SECONDS - loop.time())
                await self._type(lambda typist: typist.press("ctrl_c"))
                self._cleared = loop.time()
        except BrainGone as error:
            self._over(turn, error)
            return
        if turn.answered.done():
            # Its Stop came while the Escape was pressed: the turn ended by itself, and says so once.
            return
        stopped = BrainAnswered(turn.taken.result(), None)
        self._record(stopped)
        self._over(turn, stopped)

    def hear(self, observed: Observed) -> None:
        """The brain's own requests, read from the wire: the tools its setup gave it, and whether a turn reached hands' tools."""
        match observed:
            case Sent(session=session, kind=MainTurn(), body=body) if session == self.session:
                tools = tool_names(body)
                if tools != self._offered:
                    self._offered = tools
                    self._record(BrainOffered(tools))
                if not any(name.startswith(f"mcp__{SERVER_NAME}__") for name in tools):
                    # [LAW:no-silent-failure] a brain without hands' tools answers every question about the sessions from nothing.
                    logger.error(f"the brain's turn went to the model without hands' tools: it did not connect to hands' MCP server ({tools})")
            case _:
                pass

    async def exited(self) -> int:
        """Waits for the brain to end, and returns its exit code; the line that says it ended is written once, however many wait."""
        return await asyncio.shield(self._exit)

    async def stop(self) -> None:
        await self._claude.stop()
        await self.exited()
        self._heard.cancel()
        await self._listener.cleanup()
        shutil.rmtree(self._sockets, ignore_errors=True)

    async def _type(self, typing: Callable[[Typist], None]) -> None:
        try:
            await asyncio.to_thread(typing, self._typist)
        except Untyped as error:
            raise BrainGone(f"the brain cannot be typed into: {error}") from error

    async def _run_out(self) -> int:
        code = await asyncio.shield(self._claude.exit)
        self._settle(Deny(f"the brain exited ({code})"))
        # [LAW:no-silent-failure] a turn that can never end is said to have failed, not left waiting.
        if self._turn is not None:
            self._over(self._turn, BrainGone(f"the brain exited ({code}) before it answered"))
        self._record(BrainExited(code, self._claude.shown()))
        return code

    async def _hear_hooks(self, hooks: "asyncio.Queue[_Posted]") -> None:
        while True:
            posted = await hooks.get()
            match posted.event:
                case "PermissionRequest":
                    self._keep(self._permit(posted))
                case "Elicitation":
                    posted.reply.set_result(DECLINED)
                    self._refused(posted.said)
                case _:
                    # Asks nothing of the turn, so it is answered at once, whatever it says.
                    posted.reply.set_result({})
                    self._hook(posted.said)

    async def _permit(self, posted: _Posted) -> None:
        """Answers a permission request: held while the user is asked, when a turn of theirs is in flight to ask in."""
        loop = asyncio.get_running_loop()
        began = loop.time()
        try:
            prompt, asked = posted.said.optional_text("prompt_id"), called(posted.said)
        except Rejected as error:
            logger.warning(f"the brain posted a permission request hands cannot read: {error}")
            posted.reply.set_result(UNREADABLE["PermissionRequest"])
            return
        turn = self._turn
        match asked:
            # [LAW:single-enforcer] the turn's own, as its Stop is: one a turn before it left behind is nobody's to answer.
            case Permission(tool=tool) if turn is not None and not turn.answered.done() and turn.prompt == prompt:
                held = Asked(asked, loop.create_future())
                self._held.add(held)
                try:
                    turn.asks(held)
                    await asyncio.wait({held.decision}, timeout=PERMISSION_DEADLINE_SECONDS)
                finally:
                    self._held.discard(held)
                held.settle(Deny(UNANSWERED))
                decision = held.decision.result()
            case Permission(tool=tool):
                decision = Deny(NOBODY)
            case _:
                # A dialog of questions or a plan is never put to the user by voice: the brain's own words ask them.
                decision, tool = Deny(UNVOICED), posted.said.text("tool_name")
        # [LAW:nothing-unseen] one line for each, however it was settled.
        self._record(BrainPermission(prompt, tool, decision, loop.time() - began))
        posted.reply.set_result(hook_output(decision))

    def _settle(self, decision: Allow | Deny) -> None:
        """Settles every permission held now: its turn is over, so no answer of the user's can reach it."""
        for held in self._held:
            held.settle(decision)

    def _refused(self, said: Payload) -> None:
        """An MCP server's ask for input, answered no in a turn or between turns, where it has no prompt id; said here too,
        so the refusal is not only the brain's to tell."""
        try:
            self._record(BrainRefused(said.optional_text("prompt_id"), "Elicitation", said.optional_text("mcp_server_name")))
        except Rejected as error:
            logger.warning(f"the brain posted an Elicitation hook that does not parse: {error}")

    def _hook(self, said: Payload) -> None:
        try:
            event, session = said.text("hook_event_name"), said.session_id()
            prompt = said.text("prompt_id")
            failed = f"{said.optional_text('error')}: {said.optional_text('last_assistant_message')}"
        except Rejected as error:
            logger.warning(f"the brain posted a hook that does not parse: {error}")
            return
        turn = self._turn
        if session != self.session or turn is None or turn.answered.done():
            # A turn's own hook arriving after an Escape ended it, or a hook no turn of this brain's asked for.
            logger.info(f"the brain's {event} hook for prompt {prompt} came with no turn of its own in flight")
            return
        match event:
            case "UserPromptSubmit" if not turn.taken.done():
                turn.taken.set_result(prompt)
            case "Stop" | "StopFailure" if turn.prompt == prompt:
                error = None if event == "Stop" else failed
                answered = BrainAnswered(prompt, error)
                self._record(answered)
                self._over(turn, answered)
            case _:
                logger.warning(f"the brain's {event} hook for prompt {prompt} does not fit the turn in flight (prompt {turn.prompt})")

    def _over(self, turn: _Turn, outcome: BrainAnswered | Exception) -> None:
        if self._turn is turn:
            self._turn = None
        if turn.answered.done():
            return
        match outcome:
            case BrainAnswered():
                turn.answered.set_result(outcome)
            case _:
                turn.answered.set_exception(outcome)


async def start(launch: Launch, record: Record) -> Brain:
    """Start the brain under fritter on a terminal of hands' own, on the login its backend was parsed with."""
    claude = brain_claude(launch.station.inherited)
    if not launch.fritter.is_file():
        raise Unstartable(f"no fritter at {launch.fritter} to run the brain under: run `hands install-fritter`")
    hooks: asyncio.Queue[_Posted] = asyncio.Queue()
    listener, url = await _listen(hooks)
    # A unix socket's path is capped near 104 bytes on macOS, so not under the brain's own directory.
    sockets = Path(tempfile.mkdtemp(prefix="hands-brain-"))
    running: ClaudeCode | None = None
    try:
        running = await spawn(launch.station, [str(launch.fritter), "--socket-dir", str(sockets), "--", *command(launch, claude, url)])
        record(BrainLaunched(running.pid, launch.station.config_dir, launch.station.cwd, launch.station.model))
        typist = await _typist(running, sockets, launch.session)
        await asyncio.sleep(SETTLE_SECONDS)
    except BaseException:
        # [LAW:no-silent-failure] a start that fails or is cancelled leaves nothing running: the brain is in a session of
        # its own, which nothing but hands' own end would hang up.
        if running is not None:
            await running.stop()
        await listener.cleanup()
        shutil.rmtree(sockets, ignore_errors=True)
        raise
    return Brain(running, launch.session, typist, hooks, listener, sockets, launch.station.config_dir, record)


async def _listen(hooks: "asyncio.Queue[_Posted]") -> tuple[web.AppRunner, str]:
    """The listener the brain posts its hooks to, on loopback, each hook put on `hooks` as it comes and answered with what
    the brain settles for it."""

    async def hook(request: web.Request) -> web.Response:
        # The event read off the path, so a body hands cannot read is still answered as its hook asks.
        event = request.match_info["event"]
        try:
            said = Payload.parse(await request.read())
        except Rejected as error:
            logger.warning(f"the brain posted a {event} hook hands cannot read: {error}")
            return web.json_response(UNREADABLE.get(event, {}))
        reply: asyncio.Future[Mapping[str, object]] = asyncio.get_running_loop().create_future()
        hooks.put_nowait(_Posted(event, said, reply))
        # Shielded: a handler aiohttp cancels leaves the reply to be set by the brain, which is never refused a hook.
        return web.json_response(await asyncio.shield(reply))

    app = web.Application()
    app.router.add_post("/{event}", hook)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    return runner, f"http://{host}:{port}"


async def _typist(running: ClaudeCode, sockets: Path, session: SessionId) -> Typist:
    """The brain as fritter types into it: the socket fritter opened under `sockets`, and the claude it started."""
    deadline = asyncio.get_running_loop().time() + START_SECONDS
    while asyncio.get_running_loop().time() < deadline and not running.exit.done():
        found = [path for path in sockets.rglob("*") if path.is_socket()]
        child = await asyncio.to_thread(_child_of, running.pid)
        if found and child is not None:
            return Typist(session, found[0], child)
        await asyncio.sleep(0.05)
    # What fritter's terminal showed says why: a fritter that could not be run at all fails there, in the shell it was run by.
    if running.exit.done():
        raise Unstartable(f"fritter exited ({running.exit.result()}) before it started the brain and opened its socket; it showed:\n{running.shown()}")
    raise Unstartable(f"fritter did not start the brain and open its socket in {START_SECONDS:.0f}s; it showed:\n{running.shown()}")


def _child_of(pid: int) -> int | None:
    """The process `pid` started: fritter starts one, the claude it wraps."""
    found = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    return int(found[0]) if found else None
