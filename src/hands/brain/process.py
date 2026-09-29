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
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from loguru import logger

from hands.brain.mcp import SERVER_NAME
from hands.sessions.audit import BrainAnswered, BrainAsked, BrainExited, BrainLaunched, BrainReady, Record

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


@dataclass(frozen=True)
class Launch:
    """Everything the brain is started with."""

    config_dir: Path
    cwd: Path
    model: str
    instruction: str
    proxy_url: str
    mcp_config: str


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


class BrainGone(Exception):
    """The brain's process ended while a turn waited on it."""


class NotLoggedIn(Exception):
    """The brain's config directory holds no login, so every turn would fail."""


async def logged_in(launch: Launch) -> None:
    """Returns when the brain's config directory holds a login; raises NotLoggedIn, naming the command that makes one, when not."""
    asked = await asyncio.create_subprocess_exec(
        "claude", "auth", "status",
        env=environment(launch.config_dir, launch.proxy_url, os.environ),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(asked.communicate(), AUTH_STATUS_SECONDS)
    except TimeoutError:
        asked.kill()
        raise NotLoggedIn(f"`claude auth status` for the brain did not answer in {AUTH_STATUS_SECONDS:.0f}s") from None
    try:
        status = cast(object, json.loads(out))
    except json.JSONDecodeError:
        raise NotLoggedIn(f"`claude auth status` for the brain answered {out[:200]!r} {err[:200]!r}, not its status") from None
    match status:
        case {"loggedIn": True}:
            return
        case _:
            raise NotLoggedIn(f"the brain has no login; run: CLAUDE_CONFIG_DIR={launch.config_dir} claude auth login")


class Brain:
    """A running brain: a turn is asked with `ask`, which returns once the brain has said the turn is over."""

    def __init__(self, process: asyncio.subprocess.Process, record: Record) -> None:
        self._process = process
        self._record = record
        # [LAW:no-ambient-temporal-coupling] one turn at a time is the brain's own shape: the next is written once the
        # last one's result line has been read, so a result always belongs to the turn that is waiting on it.
        self._turn = asyncio.Lock()
        self._answers: asyncio.Queue[BrainAnswered] = asyncio.Queue()
        self._stderr: deque[str] = deque(maxlen=STDERR_LINES)
        self._reading = asyncio.gather(self._read_stdout(), self._read_stderr())

    @property
    def pid(self) -> int:
        return self._process.pid

    async def ask(self, text: str) -> BrainAnswered:
        async with self._turn:
            stdin = self._process.stdin
            assert stdin is not None, "the brain is started with a stdin pipe"
            stdin.write(json.dumps({"type": "user", "message": {"role": "user", "content": text}}).encode() + b"\n")
            await stdin.drain()
            self._record(BrainAsked(text))
            answer = asyncio.ensure_future(self._answers.get())
            ended = asyncio.ensure_future(self._process.wait())
            await asyncio.wait({answer, ended}, return_when=asyncio.FIRST_COMPLETED)
            ended.cancel()
            if not answer.done():
                answer.cancel()
                # [LAW:no-silent-failure] a turn that can never end is said to have failed, not left waiting.
                raise BrainGone(f"the brain exited ({self._process.returncode}) before it answered")
            return answer.result()

    async def exited(self) -> int:
        """Waits for the process to end, writes the line that says so, and returns its exit code."""
        code = await self._process.wait()
        await self._reading
        self._record(BrainExited(code, "\n".join(self._stderr)))
        return code

    async def stop(self) -> None:
        if self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), STOP_SECONDS)
            except TimeoutError:
                self._process.kill()
        await self.exited()

    async def _read_stdout(self) -> None:
        stdout = self._process.stdout
        assert stdout is not None, "the brain is started with a stdout pipe"
        async for line in stdout:
            self._heard(line)

    async def _read_stderr(self) -> None:
        stderr = self._process.stderr
        assert stderr is not None, "the brain is started with a stderr pipe"
        async for line in stderr:
            text = line.decode(errors="replace").rstrip()
            self._stderr.append(text)
            logger.warning(f"the brain said on stderr: {text}")

    def _heard(self, line: bytes) -> None:
        try:
            said = cast(object, json.loads(line))
        except json.JSONDecodeError:
            # [LAW:no-silent-failure] stream-json is one object a line; anything else is a harness change to hear about.
            logger.warning(f"the brain wrote a line that is not JSON: {line[:200]!r}")
            return
        match said:
            case {"type": "system", "subtype": "init"}:
                self._ready(cast(Mapping[str, object], said))
            case {"type": "result"}:
                # The turn is over whatever else the line holds; a field it lacks is recorded as absent, not guessed.
                result = cast(Mapping[str, object], said)
                answered = BrainAnswered(str(result.get("subtype")), result.get("is_error") is True, _int(result.get("num_turns")), _int(result.get("duration_ms")))
                self._record(answered)
                self._answers.put_nowait(answered)
            case _:
                # Everything else is what the model said and did, which hands reads from the wire, where all of it is.
                pass

    def _ready(self, init: Mapping[str, object]) -> None:
        servers = {str(server.get("name")): str(server.get("status")) for server in _objects(init.get("mcp_servers"))}
        self._record(BrainReady(str(init.get("session_id")), str(init.get("model")), tuple(str(tool) for tool in _list(init.get("tools"))), servers))
        if servers.get(SERVER_NAME) != "connected":
            # [LAW:no-silent-failure] a brain without hands' tools answers every question about the sessions from nothing.
            logger.error(f"the brain did not connect to hands' MCP server: {servers}")


async def start(launch: Launch, record: Record) -> Brain:
    """Start the brain; raises NotLoggedIn before starting it when its config directory holds no login."""
    await logged_in(launch)
    launch.cwd.mkdir(parents=True, exist_ok=True)
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
    return Brain(process, record)


def _int(value: object) -> int | None:
    match value:
        case bool():
            return None
        case int():
            return value
        case _:
            return None


def _list(value: object) -> Sequence[object]:
    match value:
        case list():
            return cast(list[object], value)
        case _:
            return []


def _objects(value: object) -> list[Mapping[str, object]]:
    return [cast(Mapping[str, object], item) for item in _list(value) if isinstance(item, dict)]
