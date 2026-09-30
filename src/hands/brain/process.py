"""The brain's process: one long-lived, slim Claude Code on the subscription, behind hands' proxy, reaching hands over MCP.

It is Claude Code as anyone runs it: interactive, on a terminal hands holds, under fritter, and asked nothing a person
at its keyboard could not ask. A turn is typed into its input and sent with Return, a side question is `/btw` typed
the same way, and an interrupt is Escape. What it says is read from the wire, not from its screen [the design's rule:
primary facts from the wire, derivative ones from the harness], and the harness is heard only through the hooks it
posts to a listener of hands' own: that a typed turn was taken, and that it ended, or that the API failed it.

Its login, settings, and skills live in a directory hands owns, set up once, as any Claude Code is, by running it there:

    mkdir -p ~/.hands/brain/cwd && cd ~/.hands/brain/cwd && CLAUDE_CONFIG_DIR=~/.hands/brain claude

and it runs in that empty directory of hands' own, never in a project.
"""

import asyncio
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
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from pathlib import Path

from aiohttp import web
from loguru import logger

from hands.brain.mcp import SERVER_NAME
from hands.core.effects import Command, Text
from hands.core.session import ESCAPES, CommandName, Keystroke, PromptText, SessionId, pasted
from hands.core.wire import Exchanged, Fork, MainTurn, Observed, Reached, Sent, Streamed, asked, tool_names
from hands.core.wire import Text as Said
from hands.sessions.audit import BrainAnswered, BrainAsked, BrainExited, BrainForked, BrainLaunched, Record
from hands.sessions.payload import Payload, Rejected
from hands.sessions.typing import Typist, Untyped
from hands.sessions.wrapper import SESSION_TAP, real_claude

# The built-in tools the brain is given: it reads, searches, runs commands, and uses skills. It edits nothing itself,
# and it reaches the working sessions only through hands' tools over MCP.
BUILTIN_TOOLS = ("Read", "Glob", "Grep", "Bash", "Skill")

# What --bare would have switched off, switched off one by one so the OAuth login stays on (hands-wire-6ic.8wu, 2.1.284).
# LSP and plugin sync need no switch: they come only from plugins, and the brain's config directory has none.
SLIM = {
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION": "false",
    # A background command's notification opens a turn of its own, under a prompt id no turn hands typed carries, which
    # would take a typed turn's place or leave it never ending (hands-wire-6ic.99l review).
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
}

# Credentials Claude Code prefers to its own login. Inherited from hands' environment, any of them would put the brain
# on another account or off the subscription without a word, so none is passed on.
FOREIGN_CREDENTIALS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")

# What the brain posts to hands, each to its own path: a typed turn taken, a turn ended, and a turn the API failed.
# Escape ends a turn with none of them (measured on 2.1.285), so a turn told to stop is over when it is told.
HOOKS = ("UserPromptSubmit", "Stop", "StopFailure")

# The side question Claude Code asks of a fork of the session, which answers from its context and writes nothing into it.
ASIDE = CommandName("btw")

# The terminal the brain draws on. Nobody looks at it; it is sized so a long line is not wrapped into many.
ROWS, COLS = 50, 200
# How much of what the brain last showed is kept for the line that says it exited.
SHOWN_BYTES = 16 * 1024
SHOWN_LINES = 20
# `claude auth status` answers in about a second; one that has not answered in this long is not going to.
AUTH_STATUS_SECONDS = 20.0
# How long fritter has to start the brain and open its socket.
START_SECONDS = 10.0
# How long the brain's input takes to come up: after it starts (about 0.22s in, 2.1.285, measured 2026-09-30), and
# after a side question's answer is dismissed. Text typed before it is up is lost, and a turn lost so is untaken, which
# says so.
SETTLE_SECONDS = 0.5
# How long a typed turn has to be taken: a cold start loads the MCP server before the input is read.
TAKE_SECONDS = 30.0
# How long a brain told to stop has before it is killed.
STOP_SECONDS = 5.0
# How long a side question is waited on: one reply of one sentence, with thinking, read from the brain's cache.
FORK_SECONDS = 120.0

# What a terminal is told rather than shown, and the keys it is sent.
_CONTROL = re.compile(rf"{ESCAPES.pattern}|[\x00-\x09\x0b-\x1f\x7f]")


@dataclass(frozen=True)
class Launch:
    """Everything the brain is started with."""

    config_dir: Path
    cwd: Path
    model: str
    instruction: str
    proxy_url: str
    mcp_config: str
    # [LAW:one-source-of-truth] chosen by hands, so the brain's requests are known as its own from the first one on the
    # wire, and its hooks from the first one posted.
    session: SessionId
    # hands' own fritter, from `hands install-fritter`.
    fritter: Path


def command(launch: Launch, claude: Path, hooks: str) -> list[str]:
    """The brain's command line: the real claude, interactive, posting its hooks to the listener at `hooks`."""
    return [
        str(claude),
        "--model", launch.model,
        "--session-id", launch.session,
        # After Claude Code's own system prompt, never in place of it: the API checks that it opens as Claude Code's does.
        "--append-system-prompt", launch.instruction,
        "--tools", ",".join(BUILTIN_TOOLS),
        # [LAW:single-enforcer] what the brain may do without asking is said here, and nothing else is allowed: nobody
        # sits at its keyboard to be asked, so anything outside the list is denied rather than left waiting.
        "--allowedTools", ",".join((*BUILTIN_TOOLS, f"mcp__{SERVER_NAME}")),
        "--permission-mode", "dontAsk",
        # Without it the claude.ai connectors on the account load after the first turn and join every request after it
        # (turn 2 grew from 8.7 KB to 112 KB; hands-wire-6ic.eph, 2.1.284).
        "--strict-mcp-config",
        "--mcp-config", launch.mcp_config,
        # The config directory's settings.json is the only settings file read: none from the working directory.
        "--setting-sources", "user",
        "--settings", json.dumps({"hooks": {event: [{"hooks": [{"type": "http", "url": f"{hooks}/{event}"}]}] for event in HOOKS}}),
    ]


def environment(config_dir: Path, base_url: str, inherited: Mapping[str, str]) -> dict[str, str]:
    """A slim Claude Code's environment: hands' own, less any credential that is not the login in `config_dir`, reaching the API at `base_url`."""
    # [LAW:one-source-of-truth] the brain is the same brain wherever hands was started: a daemon started inside a tapped
    # session does not hand the brain that session's tap.
    kept = {name: value for name, value in inherited.items() if name not in (*FOREIGN_CREDENTIALS, *SESSION_TAP)}
    return {**kept, **SLIM, "CLAUDE_CONFIG_DIR": str(config_dir), "ANTHROPIC_BASE_URL": base_url}


def workdir(config_dir: Path) -> Path:
    """The empty directory a slim Claude Code on the login in `config_dir` runs in, made if it is not there: never a project."""
    cwd = _cwd(config_dir)
    cwd.mkdir(parents=True, exist_ok=True)
    return cwd


def _cwd(config_dir: Path) -> Path:
    return config_dir / "cwd"


class BrainGone(Exception):
    """The brain ended, or cannot be typed into, while a turn or a side question waited on it, or before one was asked."""


class Untaken(Exception):
    """The brain was typed a turn and never took it: it is on a screen that is not its input."""


class ForkFailed(Exception):
    """The brain's side question was answered with no words, or not at all."""


class NotLoggedIn(Exception):
    """The brain's config directory holds no login, so every turn would fail."""


class Unstartable(Exception):
    """The brain could not be started: no claude to run, no fritter to run it under, or a fritter that never opened its socket."""


def logged_in(config_dir: Path, base_url: str) -> None:
    """Returns when `config_dir` holds a login; raises NotLoggedIn, naming the command that makes one, when not."""
    try:
        # A timed-out child is killed and reaped by run itself.
        asked = subprocess.run(
            ["claude", "auth", "status"],
            env=environment(config_dir, base_url, os.environ),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=AUTH_STATUS_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise NotLoggedIn(f"`claude auth status` for the brain did not answer in {AUTH_STATUS_SECONDS:.0f}s") from None
    try:
        status = Payload.parse(asked.stdout).flag("loggedIn")
    except Rejected as error:
        raise NotLoggedIn(f"`claude auth status` for the brain answered {asked.stdout[:200]!r} {asked.stderr[:200]!r}, not its status: {error}") from None
    if not status:
        raise NotLoggedIn(f"the brain has no login; run: {setup(config_dir)}")


def setup(config_dir: Path) -> str:
    """The command that sets the brain up as any Claude Code is set up: run once, where it runs."""
    # The directory is made here too: it is asked for before hands has ever run the brain, and so before workdir made it.
    return f"mkdir -p {_cwd(config_dir)} && cd {_cwd(config_dir)} && CLAUDE_CONFIG_DIR={config_dir} claude"


class _Terminal:
    """The terminal the brain runs on, held by hands: read as fast as it is written, and the last of it kept."""

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

        threading.Thread(target=read, name="the brain's terminal", daemon=True).start()

    def last(self) -> str:
        """The last lines the brain showed, as text."""
        with self._lock:
            shown = self._shown.decode(errors="replace")
        lines = [line.rstrip() for line in _CONTROL.sub("", shown.replace("\r", "\n")).split("\n") if line.strip()]
        return "\n".join(lines[-SHOWN_LINES:])


@dataclass
class _Aside:
    """A side question typed into the brain: its text as typed, the exchanges on the wire that ask it, and its answer."""

    question: PromptText
    answer: asyncio.Future[str]
    # [LAW:one-source-of-truth] the answer is read from the brain's own request for this question, never from whichever
    # fork of the session happens to end first.
    exchanges: set[str]
    # Whether the answer came, and so is what covers the input: Return dismisses an answer, and Escape a question still
    # waiting on one, which it cancels and leaves the turn running (2.1.285, measured 2026-09-30).
    shown: bool = False

    @property
    def dismissal(self) -> Keystroke:
        return "enter" if self.shown else "escape"


@dataclass
class _Turn:
    answered: asyncio.Future[BrainAnswered]
    taken: asyncio.Future[str]  # the prompt id Claude Code gave the turn when it took it

    @property
    def prompt(self) -> str | None:
        return self.taken.result() if self.taken.done() else None


class Brain:
    """A running brain: a turn is asked with `ask`, which returns once the brain has said the turn is over."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        session: SessionId,
        typist: Typist,
        terminal: _Terminal,
        hooks: "asyncio.Queue[Payload]",
        listener: web.AppRunner,
        sockets: Path,
        config_dir: Path,
        record: Record,
    ) -> None:
        self._process = process
        self._config_dir = config_dir
        self.session = session
        self._typist = typist
        self._terminal = terminal
        self._listener = listener
        self._sockets = sockets
        self._record = record
        # [LAW:single-enforcer] one thing is typed at a time, and nothing while a side question's answer covers the input,
        # where whatever is typed goes into the answer instead. Side questions wait their turn for the input in the queue,
        # so a turn the user asked for, or a stop, each of which takes the input alone, is next once whatever holds it lets go.
        self._queue = asyncio.Lock()
        self._input = asyncio.Lock()
        # [LAW:no-ambient-temporal-coupling] the turn in flight is the brain's own state, not its asker's: it is over
        # when its hook says so, whether or not anyone still waits on it, and the next is typed only then.
        self._turn: _Turn | None = None
        self._typing: set[asyncio.Task[None]] = set()
        self._fork: _Aside | None = None
        self._heard = asyncio.ensure_future(self._hear_hooks(hooks))
        self._exit = asyncio.ensure_future(self._run_out())

    @property
    def pid(self) -> int:
        return self._process.pid

    async def ask(self, text: str) -> BrainAnswered:
        while self._turn is not None:
            await asyncio.wait({self._turn.answered})
        if self._exit.done():
            raise BrainGone(f"the brain had exited ({self._process.returncode}) before it was asked")
        loop = asyncio.get_running_loop()
        turn = self._turn = _Turn(loop.create_future(), loop.create_future())
        # An asker that stops waiting leaves the turn to be typed and to run to its end, which is still the brain's to hear.
        self._keep(self._send(text, turn))
        return await asyncio.shield(turn.answered)

    def _keep(self, typing: Coroutine[None, None, None]) -> None:
        """Types on in a task of the brain's own, whoever asked for it."""
        task = asyncio.create_task(typing)
        self._typing.add(task)
        task.add_done_callback(self._typing.discard)

    async def _send(self, text: str, turn: _Turn) -> None:
        try:
            async with self._input:
                # Behind a space, as every prompt hands types: a leading / or ! is then the character it is.
                await self._type(lambda: self._typist.type(Text(pasted(text)).typed))
            self._record(BrainAsked(text))
            await asyncio.wait({turn.taken, turn.answered}, timeout=TAKE_SECONDS, return_when=asyncio.FIRST_COMPLETED)
            if not (turn.taken.done() or turn.answered.done()):
                raise Untaken(f"the brain did not take the turn typed into it in {TAKE_SECONDS:.0f}s; if it is on a setup screen, run: {setup(self._config_dir)}")
        except (BrainGone, Untaken) as error:
            self._over(turn, error)

    async def fork(self, question: str) -> str:
        """The brain's answer to a side question, asked with /btw of a fork that shares its context, writes nothing into its
        history, and may run while a turn is in flight; raises ForkFailed on an answer of no words or none in time, and
        BrainGone when the brain ends or cannot be typed into."""
        if self._exit.done():
            raise BrainGone(f"the brain had exited ({self._process.returncode}) before it was asked a side question")
        async with self._queue, self._input:
            typed = pasted(question)
            fork = self._fork = _Aside(typed, asyncio.get_running_loop().create_future(), set())
            try:
                await self._type(lambda: self._typist.command(Command(ASIDE, typed)))
                reply = await asyncio.wait_for(asyncio.shield(fork.answer), FORK_SECONDS)
            except TimeoutError as error:
                # [LAW:no-silent-failure] a question the brain never answers ends here, said as such.
                # What the brain showed says why: a question it never saw, one it is still answering, or a screen over its input.
                failure = ForkFailed(f"no answer in {FORK_SECONDS:.0f}s; the brain showed:\n{self._terminal.last()}")
                self._record(BrainForked(question, str(failure), failed=True))
                raise failure from error
            except (ForkFailed, BrainGone) as error:
                self._record(BrainForked(question, str(error), failed=True))
                raise
            except asyncio.CancelledError:
                self._record(BrainForked(question, "its asker stopped waiting", failed=True))
                raise
            finally:
                self._fork = None
                # The question or its answer covers the input until it is dismissed.
                if not self._exit.done():
                    try:
                        await self._type(lambda: self._typist.press(fork.dismissal))
                    except BrainGone:
                        logger.exception("the brain's side question could not be dismissed")
                    await asyncio.sleep(SETTLE_SECONDS)
        self._record(BrainForked(question, reply, failed=False))
        return reply

    def interrupt(self) -> None:
        """Stop the turn in flight with Escape, as at the keyboard. Returns at once: the Escape is the brain's to press,
        before anything else waiting to type, once a side question on the screen is done with it (hands-wire-6ic.ulk), and
        no hook says a turn was stopped, so the turn ends where it is pressed."""
        turn = self._turn
        if turn is not None:
            self._keep(self._stop(turn))

    async def _stop(self, turn: _Turn) -> None:
        # [LAW:no-ambient-temporal-coupling] Escape goes once Claude Code has taken the turn, never before: a turn ended
        # while its UserPromptSubmit hook is still coming would leave that hook to be taken for the next turn's.
        await asyncio.wait({turn.taken, turn.answered}, timeout=TAKE_SECONDS, return_when=asyncio.FIRST_COMPLETED)
        try:
            async with self._input:
                if self._turn is not turn:
                    return
                if not turn.taken.done():
                    # [LAW:no-silent-failure] the turn runs on, told to stop by nobody: its words are the stage's to hold.
                    logger.warning(f"the brain was not stopped: its turn was not taken in {TAKE_SECONDS:.0f}s")
                    return
                await self._type(lambda: self._typist.press("escape"))
                # Escape puts the stopped prompt back in the input, which the next turn typed would join; Ctrl-C clears it.
                # It arms Claude Code's exit for a second Ctrl-C on an empty input, which no stop presses: each follows an
                # Escape that refilled it (2.1.285, measured 2026-09-30, two stops 45ms apart).
                await self._type(lambda: self._typist.press("ctrl_c"))
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
        """The brain's own requests, read from the wire: whether a turn reached hands' tools, and a side question's answer."""
        fork = self._fork
        match observed:
            case Sent(session=session, kind=MainTurn(), body=body) if session == self.session and not any(
                name.startswith(f"mcp__{SERVER_NAME}__") for name in tool_names(body)
            ):
                # [LAW:no-silent-failure] a brain without hands' tools answers every question about the sessions from nothing.
                logger.error(f"the brain's turn went to the model without hands' tools: it did not connect to hands' MCP server ({tool_names(body)})")
            case Sent(exchange=exchange, session=session, kind=Fork(), body=body) if (
                session == self.session and fork is not None and any(fork.question in text for text in asked(body))
            ):
                fork.exchanges.add(exchange)
            case Exchanged(exchange=exchange, reply=reply) if fork is not None and exchange in fork.exchanges and not fork.answer.done():
                match reply:
                    case Reached(status=200, body=Streamed(message=message)):
                        fork.shown = True
                        said = " ".join(block.text for block in message.content if isinstance(block, Said)).strip()
                        if said:
                            fork.answer.set_result(said)
                        else:
                            fork.answer.set_exception(ForkFailed(f"the brain answered the side question with no words ({message.stop_reason})"))
                    case _:
                        # Claude Code asks again after a request that failed; the question waits for that, or for its time.
                        logger.warning(f"a side question of the brain's was answered {reply}")
            case _:
                pass

    async def exited(self) -> int:
        """Waits for the brain to end, and returns its exit code; the line that says it ended is written once, however many wait."""
        return await asyncio.shield(self._exit)

    async def stop(self) -> None:
        if self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), STOP_SECONDS)
            except TimeoutError:
                self._process.kill()
        await self.exited()
        self._heard.cancel()
        await self._listener.cleanup()
        shutil.rmtree(self._sockets, ignore_errors=True)

    async def _type(self, typing: Callable[[], None]) -> None:
        try:
            await asyncio.to_thread(typing)
        except Untyped as error:
            raise BrainGone(f"the brain cannot be typed into: {error}") from error

    async def _run_out(self) -> int:
        code = await self._process.wait()
        try:
            # What it showed last, read to its end; a terminal some child of it still holds is not waited on for long.
            await asyncio.wait_for(asyncio.shield(self._terminal.closed), 1.0)
        except TimeoutError:
            pass
        # [LAW:no-silent-failure] a turn or a side question that can never end is said to have failed, not left waiting.
        if self._turn is not None:
            self._over(self._turn, BrainGone(f"the brain exited ({code}) before it answered"))
        if self._fork is not None and not self._fork.answer.done():
            self._fork.answer.set_exception(BrainGone(f"the brain exited ({code}) before it answered a side question"))
        self._record(BrainExited(code, self._terminal.last()))
        return code

    async def _hear_hooks(self, hooks: "asyncio.Queue[Payload]") -> None:
        while True:
            self._hook(await hooks.get())

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
    claude = real_claude(os.environ.get("PATH", ""))
    if claude is None:
        raise Unstartable("no claude on PATH but hands' shims, so there is no Claude Code to run as the brain")
    if not launch.fritter.is_file():
        raise Unstartable(f"no fritter at {launch.fritter} to run the brain under: run `hands install-fritter`")
    hooks: asyncio.Queue[Payload] = asyncio.Queue()
    listener, url = await _listen(hooks)
    # A unix socket's path is capped near 104 bytes on macOS, so not under the brain's own directory.
    sockets = Path(tempfile.mkdtemp(prefix="hands-brain-"))
    process: asyncio.subprocess.Process | None = None
    try:
        master, slave = pty.openpty()
        try:
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
            process = await asyncio.create_subprocess_exec(
                str(launch.fritter), "--socket-dir", str(sockets), "--", *command(launch, claude, url),
                cwd=launch.cwd,
                env={**environment(launch.config_dir, launch.proxy_url, os.environ), "TERM": "xterm-256color"},
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
        terminal = _Terminal(master)
        record(BrainLaunched(process.pid, launch.config_dir, launch.cwd, launch.model))
        typist = await _typist(process, sockets, launch.session)
        await asyncio.sleep(SETTLE_SECONDS)
    except BaseException:
        # [LAW:no-silent-failure] a start that fails or is cancelled leaves nothing running: the brain is in a session of
        # its own, so nothing else would end it when hands does.
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        await listener.cleanup()
        shutil.rmtree(sockets, ignore_errors=True)
        raise
    return Brain(process, launch.session, typist, terminal, hooks, listener, sockets, launch.config_dir, record)


async def _listen(hooks: "asyncio.Queue[Payload]") -> tuple[web.AppRunner, str]:
    """The listener the brain posts its hooks to, on loopback, each hook put on `hooks` as it comes."""

    async def hook(request: web.Request) -> web.Response:
        try:
            hooks.put_nowait(Payload.parse(await request.read()))
        except Rejected as error:
            logger.warning(f"the brain posted a hook hands cannot read: {error}")
            return web.Response(status=400, text=str(error))
        # An empty answer: the hook asks nothing of the turn.
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/{event}", hook)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    return runner, f"http://{host}:{port}"


async def _typist(process: asyncio.subprocess.Process, sockets: Path, session: SessionId) -> Typist:
    """The brain as fritter types into it: the socket fritter opened under `sockets`, and the claude it started."""
    deadline = asyncio.get_running_loop().time() + START_SECONDS
    while asyncio.get_running_loop().time() < deadline and process.returncode is None:
        found = [path for path in sockets.rglob("*") if path.is_socket()]
        child = await asyncio.to_thread(_child_of, process.pid)
        if found and child is not None:
            return Typist(session, found[0], child)
        await asyncio.sleep(0.05)
    raise Unstartable(f"fritter did not start the brain and open its socket in {START_SECONDS:.0f}s (exit {process.returncode})")


def _child_of(pid: int) -> int | None:
    """The process `pid` started: fritter starts one, the claude it wraps."""
    found = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    return int(found[0]) if found else None
