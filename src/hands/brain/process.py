"""The brain's process: one long-lived, slim Claude Code on the subscription, behind hands' proxy, reaching hands over MCP.

It runs `claude -p` with stream-json both ways: a turn is a user message written to its stdin, and ends with the
`result` line it writes to stdout. What it says is read from the wire, not from stdout [the design's rule: primary
facts from the wire, derivative ones from the harness]: stdout is read here only for what the harness alone knows,
its session and what it loaded, and when a turn ended and how.

Its login, settings, and skills live in a directory hands owns, logged in once with

    CLAUDE_CONFIG_DIR=~/.hands/brain claude auth login

and it runs in an empty directory of hands' own, never in a project.
"""

import asyncio
import json
import os
import subprocess
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from loguru import logger

from hands.brain.mcp import SERVER_NAME
from hands.core.session import SessionId
from hands.sessions.audit import BrainAnswered, BrainAsked, BrainExited, BrainForked, BrainLaunched, BrainReady, Record
from hands.sessions.payload import Payload, Rejected

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
}

# Credentials Claude Code prefers to its own login. Inherited from hands' environment, any of them would put the brain
# on another account or off the subscription without a word, so none is passed on.
FOREIGN_CREDENTIALS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")

# How much of stderr is kept for the line that says the brain exited.
STDERR_LINES = 20
# `claude auth status` answers in about a second; one that has not answered in this long is not going to.
AUTH_STATUS_SECONDS = 20.0
# How long a brain told to stop has before it is killed.
STOP_SECONDS = 5.0
# How long a side question is waited on: one reply of one sentence, with thinking, read from the brain's cache.
FORK_SECONDS = 120.0


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
    # wire, before its init line is read off stdout.
    session: SessionId


def command(launch: Launch) -> list[str]:
    """The brain's command line."""
    return [
        "claude",
        "-p",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--include-partial-messages",
        # stream-json output under -p is refused without it.
        "--verbose",
        "--model", launch.model,
        "--session-id", launch.session,
        "--system-prompt", launch.instruction,
        # Variadic: it would swallow a positional prompt after it, and none follows, since the prompt comes on stdin.
        "--tools", ",".join(BUILTIN_TOOLS),
        # [LAW:single-enforcer] what the brain may do without asking is said here, and nothing else is allowed: under -p
        # there is nobody to ask, so anything outside the list is denied rather than left waiting.
        "--allowedTools", ",".join((*BUILTIN_TOOLS, f"mcp__{SERVER_NAME}")),
        "--permission-mode", "dontAsk",
        "--permission-prompts", "none",
        # Without it the claude.ai connectors on the account load after the first turn and join every request after it
        # (turn 2 grew from 8.7 KB to 112 KB; hands-wire-6ic.eph, 2.1.284).
        "--strict-mcp-config",
        "--mcp-config", launch.mcp_config,
        # The config directory's settings.json is the only settings file read: none from the working directory.
        "--setting-sources", "user",
    ]


def environment(config_dir: Path, base_url: str, inherited: Mapping[str, str]) -> dict[str, str]:
    """A slim Claude Code's environment: hands' own, less any credential that is not the login in `config_dir`, reaching the API at `base_url`."""
    kept = {name: value for name, value in inherited.items() if name not in FOREIGN_CREDENTIALS}
    return {**kept, **SLIM, "CLAUDE_CONFIG_DIR": str(config_dir), "ANTHROPIC_BASE_URL": base_url}


def workdir(config_dir: Path) -> Path:
    """The empty directory a slim Claude Code on the login in `config_dir` runs in, made if it is not there: never a project."""
    cwd = config_dir / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    return cwd


class BrainGone(Exception):
    """The brain's output ended while a turn waited on it, or before one was asked."""


class ForkFailed(Exception):
    """The brain answered a side question with an error, or with no reply."""


class NotLoggedIn(Exception):
    """The brain's config directory holds no login, so every turn would fail."""


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
        raise NotLoggedIn(f"the brain has no login; run: CLAUDE_CONFIG_DIR={config_dir} claude auth login")


class Brain:
    """A running brain: a turn is asked with `ask`, which returns once the brain has said the turn is over."""

    def __init__(self, process: asyncio.subprocess.Process, session: SessionId, record: Record) -> None:
        self._process = process
        self.session = session
        self._record = record
        # [LAW:no-ambient-temporal-coupling] the turn in flight is the brain's own state, not its asker's: it is over
        # when its result line is read, whether or not anyone still waits on it, and the next is written only then, so
        # a result always belongs to the turn written before it.
        self._turn: asyncio.Future[BrainAnswered] | None = None
        # Side questions asked and not yet answered, by request id: they run beside the turn, each ended by its own response.
        self._forks: dict[str, asyncio.Future[str]] = {}
        self._stderr: deque[str] = deque(maxlen=STDERR_LINES)
        self._read = asyncio.ensure_future(self._read_stdout())
        self._said = asyncio.ensure_future(self._read_stderr())
        self._exit = asyncio.ensure_future(self._run_out())

    @property
    def pid(self) -> int:
        return self._process.pid

    async def ask(self, text: str) -> BrainAnswered:
        while self._turn is not None:
            await asyncio.wait({self._turn})
        if self._read.done():
            raise BrainGone(f"the brain's output had ended ({self._process.returncode}) before it was asked")
        turn = self._turn = asyncio.get_running_loop().create_future()
        stdin = self._process.stdin
        assert stdin is not None, "the brain is started with a stdin pipe"
        stdin.write(json.dumps({"type": "user", "message": {"role": "user", "content": text}}).encode() + b"\n")
        self._record(BrainAsked(text))
        await stdin.drain()
        # An asker that stops waiting leaves the turn running to its result line, which is still the brain's to read.
        return await asyncio.shield(turn)

    async def fork(self, question: str) -> str:
        """The brain's answer to a side question, asked of a fork that shares its context, writes nothing into its
        history, and may run while a turn is in flight; raises ForkFailed on an error response, and BrainGone when the
        brain's output ends first."""
        if self._read.done():
            raise BrainGone(f"the brain's output had ended ({self._process.returncode}) before it was asked a side question")
        request = uuid4().hex
        answer = self._forks[request] = asyncio.get_running_loop().create_future()
        stdin = self._process.stdin
        assert stdin is not None, "the brain is started with a stdin pipe"
        try:
            stdin.write(json.dumps({"type": "control_request", "request_id": request, "request": {"subtype": "side_question", "question": question}}).encode() + b"\n")
            await stdin.drain()
            reply = await asyncio.wait_for(asyncio.shield(answer), FORK_SECONDS)
        except (ConnectionError, TimeoutError) as error:
            # [LAW:no-silent-failure] a question the brain cannot take, or never answers, ends here, said as such.
            failure = BrainGone(f"the brain's stdin closed: {error}") if isinstance(error, ConnectionError) else ForkFailed(f"no answer in {FORK_SECONDS:.0f}s")
            self._record(BrainForked(request, question, str(failure), failed=True))
            raise failure from error
        except (ForkFailed, BrainGone) as error:
            self._record(BrainForked(request, question, str(error), failed=True))
            raise
        finally:
            # However the asker stops waiting, cancelled included, no one is left to be told the answer.
            self._forks.pop(request, None)
        self._record(BrainForked(request, question, reply, failed=False))
        return reply

    async def interrupt(self) -> None:
        """Tell the turn in flight to stop. It still ends at its result line, which the brain writes once it has stopped."""
        stdin = self._process.stdin
        assert stdin is not None, "the brain is started with a stdin pipe"
        stdin.write(json.dumps({"type": "control_request", "request_id": uuid4().hex, "request": {"subtype": "interrupt"}}).encode() + b"\n")
        await stdin.drain()

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

    async def _run_out(self) -> int:
        try:
            await self._read
        finally:
            # A brain whose output has ended, or broken, can answer nothing more.
            if self._process.returncode is None:
                self._process.kill()
            code = await self._process.wait()
            await self._said
            self._record(BrainExited(code, "\n".join(self._stderr)))
        return code

    async def _read_stdout(self) -> None:
        stdout = self._process.stdout
        assert stdout is not None, "the brain is started with a stdout pipe"
        try:
            async for line in stdout:
                self._heard(line)
        finally:
            # [LAW:no-silent-failure] a turn or a side question that can never end is said to have failed, not left waiting.
            self._over(BrainGone("the brain's output ended before it answered"))
            for answer in self._forks.values():
                answer.set_exception(BrainGone("the brain's output ended before it answered a side question"))
            self._forks.clear()

    async def _read_stderr(self) -> None:
        stderr = self._process.stderr
        assert stderr is not None, "the brain is started with a stderr pipe"
        async for line in stderr:
            text = line.decode(errors="replace").rstrip()
            self._stderr.append(text)
            logger.warning(f"the brain said on stderr: {text}")

    def _heard(self, line: bytes) -> None:
        try:
            said = Payload.parse(line)
        except Rejected as error:
            # [LAW:no-silent-failure] stream-json is one object a line; anything else is a harness change to hear about.
            logger.warning(f"the brain wrote a line that is not a JSON object ({error}): {line[:200]!r}")
            return
        match said.fields.get("type"), said.fields.get("subtype"):
            case "system", "init":
                self._ready(said)
            case "control_response", _:
                self._answered(said)
            case "result", _:
                # The turn is over whatever else the line holds; one that does not parse fails it, loudly.
                try:
                    answered = BrainAnswered(said.text("subtype"), said.flag("is_error"), said.integer("num_turns"), said.integer("duration_ms"))
                except Rejected as error:
                    self._over(error)
                    return
                self._record(answered)
                self._over(answered)
            case _:
                # Everything else is what the model said and did, which hands reads from the wire, where all of it is.
                pass

    def _answered(self, said: Payload) -> None:
        """A control response: the end of the side question it names, or of a control request no one waits on."""
        try:
            response = Payload.of(said.fields.get("response"), "a control response")
            answer = self._forks.pop(response.text("request_id"), None)
        except Rejected as error:
            logger.warning(f"the brain wrote a control response that does not parse: {error}")
            return
        if answer is None:
            # An interrupt's acknowledgement: nothing waits on it.
            return
        match response.fields.get("subtype"), response.fields.get("response"):
            case "success", {"response": str() as reply}:
                answer.set_result(reply)
            case "success", given:
                answer.set_exception(ForkFailed(f"the brain answered the side question with no reply: {given!r}"))
            case _:
                answer.set_exception(ForkFailed(f"the brain failed the side question: {response.fields.get('error')!r}"))

    def _over(self, outcome: BrainAnswered | Exception) -> None:
        match self._turn, outcome:
            case None, BrainGone():
                pass
            case None, _:
                logger.warning(f"the brain ended a turn nobody asked: {outcome!r}")
            case turn, BrainAnswered():
                self._turn = None
                turn.set_result(outcome)
            case turn, _:
                self._turn = None
                turn.set_exception(outcome)

    def _ready(self, init: Payload) -> None:
        try:
            servers = {server.text("name"): server.text("status") for server in (Payload.of(entry, "an MCP server") for entry in init.optional_items("mcp_servers"))}
            ready = BrainReady(init.text("session_id"), init.text("model"), tuple(str(tool) for tool in init.items("tools")), servers)
        except Rejected as error:
            logger.warning(f"the brain's init line does not parse: {error}")
            return
        self._record(ready)
        if servers.get(SERVER_NAME) != "connected":
            # [LAW:no-silent-failure] a brain without hands' tools answers every question about the sessions from nothing.
            logger.error(f"the brain did not connect to hands' MCP server: {servers}")


async def start(launch: Launch, record: Record) -> Brain:
    """Start the brain, on the login its backend was parsed with."""
    process = await asyncio.create_subprocess_exec(
        *command(launch),
        cwd=launch.cwd,
        env=environment(launch.config_dir, launch.proxy_url, os.environ),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # A stream-json line holds a whole message; asyncio's 64 KiB default cuts a long one off mid-line.
        limit=16 * 1024 * 1024,
    )
    record(BrainLaunched(process.pid, launch.config_dir, launch.cwd, launch.model))
    return Brain(process, launch.session, record)
