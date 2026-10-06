"""Fixtures more than one test module needs."""

import asyncio
import json
import os
import stat
import subprocess
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import mlx_whisper
import pytest
from aiohttp import web
from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner

from hands.sessions.audit import Entry, SettingsEdited
from hands.sessions.home import Home
from hands.sessions.marketplace import render
from hands.sessions.wide import WideEvent
from hands.sessions.wrapper import PACKAGED
from hands.voice.wake import fetched
from hands.voice.wakeword import Pretrained

def events(recorded: Sequence[Entry], name: str) -> list[WideEvent]:
    """The wide events named `name`, in the order they were emitted."""
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == name]


# How long a pipeline may take to start before a test fails on it.
STARTUP_SECS = 5.0


@pytest.fixture(autouse=True)
def no_whisper_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building a Whisper transcribes nothing: its load would fetch 1.6 GB of weights, which no test is about."""

    def transcribe(_audio: object, **_options: object) -> dict[str, object]:
        return {"segments": []}

    monkeypatch.setattr(mlx_whisper, "transcribe", transcribe)


async def unprimed() -> None:
    """Whisper's prompt where no test is about its vocabulary: none, so it transcribes unprimed."""


def _dead_pid() -> int:
    process = subprocess.Popen(["true"])
    process.wait()
    return process.pid


@pytest.fixture
async def models(pytestconfig: pytest.Config) -> Path:
    """The wake word's models, Hey Jarvis's and Hey Mycroft's, fetched from openWakeWord's release once and kept in
    pytest's cache."""
    directory = pytestconfig.cache.mkdir("wake-word") if pytestconfig.cache else pytest.fail("the cache provider is off")
    for word in (Pretrained("Hey Jarvis"), Pretrained("Hey Mycroft")):
        await fetched(directory, word)
    return directory


@pytest.fixture
def dead_pid() -> Callable[[], int]:
    """Makes the pid of a process that has just exited, which no process holds for now, fresh at each call."""
    return _dead_pid


@pytest.fixture
def plugin(tmp_path: Path) -> Path:
    """hands' plugin as `hands plugin` writes it for the hands under test: its launcher runs this interpreter."""
    return render(Home(tmp_path / "rendering"), sys.executable).plugin


# Where the plugin's hooks and skills run: a PATH with no Python new enough for hands, since /usr/bin's python3 on macOS
# is 3.9, so the launcher can only be running the interpreter it names.
NO_PYTHON = "/usr/bin:/bin"


@dataclass(frozen=True)
class Api:
    """A served model: a chat completions endpoint under `url` and an Anthropic messages one under `anthropic_url`."""

    url: str
    # The Anthropic SDK appends /v1/messages itself, so its base is the bare host.
    anthropic_url: str


Endpoint = Callable[[web.Request], Awaitable[web.StreamResponse]]
ServeApi = Callable[[Endpoint, Endpoint], Awaitable[Api]]


@pytest.fixture
async def api_server() -> AsyncIterator[ServeApi]:
    """Serves the chat completions endpoint and the Anthropic messages one, each by the handler given; stopped after the test."""
    runners: list[web.AppRunner] = []

    async def serve(complete: Endpoint, message: Endpoint) -> Api:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", complete)
        app.router.add_post("/v1/messages", message)
        runner = web.AppRunner(app)
        runners.append(runner)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        host = f"http://127.0.0.1:{runner.addresses[0][1]}"
        return Api(url=f"{host}/v1", anthropic_url=host)

    yield serve
    for runner in runners:
        await runner.cleanup()


@dataclass
class ChatServer:
    """A served model answering every request whole, and what it has been asked: each request's body and key."""

    url: str
    anthropic_url: str
    asked: list[dict[str, object]]
    keys: list[str]


ServeChat = Callable[[str | None], Awaitable[ChatServer]]


@pytest.fixture
async def chat_server(api_server: ServeApi) -> ServeChat:
    """Starts a server whose two endpoints answer every request with the content given; stopped after the test."""

    async def serve(content: str | None) -> ChatServer:
        asked: list[dict[str, object]] = []
        keys: list[str] = []

        async def complete(request: web.Request) -> web.Response:
            asked.append(await request.json())
            keys.append(request.headers["Authorization"].removeprefix("Bearer "))
            return web.json_response(
                {
                    "id": "c1", "object": "chat.completion", "created": 0, "model": "m",
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
                }
            )

        async def message(request: web.Request) -> web.Response:
            asked.append(await request.json())
            keys.append(request.headers["x-api-key"])
            return web.json_response(
                {
                    "id": "m1", "type": "message", "role": "assistant", "model": "m", "stop_reason": "end_turn", "stop_sequence": None,
                    "content": [] if content is None else [{"type": "text", "text": content}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

        api = await api_server(complete, message)
        return ChatServer(url=api.url, anthropic_url=api.anthropic_url, asked=asked, keys=keys)

    return serve


@dataclass
class Running:
    """A pipeline running as the daemon runs one, and the errors it has raised: what the system channel would say."""

    worker: PipelineWorker
    errors: list[ErrorFrame]


@asynccontextmanager
async def running(processors: list[FrameProcessor], observers: Sequence[BaseObserver] = ()) -> AsyncGenerator[Running]:
    """Runs the processors as one pipeline, watched by the observers, from its start until the block ends, then cancels it."""
    worker = PipelineWorker(Pipeline(processors), observers=list(observers), idle_timeout_secs=None)
    started = asyncio.Event()
    errors: list[ErrorFrame] = []

    @worker.event_handler("on_pipeline_started")
    async def _started(_worker: PipelineWorker, _frame: Frame) -> None:  # pyright: ignore[reportUnusedFunction]
        started.set()

    @worker.event_handler("on_pipeline_error")
    async def _failed(_worker: PipelineWorker, error: ErrorFrame) -> None:  # pyright: ignore[reportUnusedFunction]
        errors.append(error)

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    task = asyncio.create_task(runner.run())
    await asyncio.wait_for(started.wait(), STARTUP_SECS)
    try:
        yield Running(worker, errors)
    finally:
        await worker.cancel()
        await task


@pytest.fixture(scope="session")
def fritter() -> Path:
    """The fritter this checkout's hands package carries, which its editable install built."""
    return PACKAGED


def onboard(brain: Path, settings: bytes = b'{"syncClaudeAiSkills": false, "syncClaudeAiPlugins": false}', trusted: bool = True) -> None:
    """A brain home Claude Code has been through its onboarding on, holding `settings`; the directory it runs in trusted, as
    Claude Code trusts it, by a directory above it, unless not `trusted`."""
    brain.mkdir(parents=True)
    projects = {str(brain.resolve().parent): {"hasTrustDialogAccepted": trusted}}
    (brain / ".claude.json").write_text(json.dumps({"hasCompletedOnboarding": True, "projects": projects}))
    (brain / "settings.json").write_bytes(settings)


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A `claude` first on PATH that reports its login from LOGGED_IN, or from an `auth login` or a first run (`--setting-sources user` alone, which also records its onboarding, the trust of the directory it ran in, and its approval of an API key its settings.json sets) it recorded, made by AUTH_METHOD (claude.ai unless named, api_key after an `auth login --console`, holding what 2.1.289 says of each) through API_PROVIDER (firstParty unless named), and as the brain is Claude Code at a keyboard: it
    reads its terminal raw, STARTS_AFTER seconds in if named, in bursts, takes a prompt when a Return that ends a burst sends it, and posts the hooks its --settings name. Everything it
    reads is written, one line each, to the file TYPED names, a side question with the session it was asked under; a side
    question it is started with, after `--`, is taken as if typed. A turn "wait" runs until Escape, "fail" is failed by the API, "deaf"
    is never taken, "late" is taken only once the next prompt is sent, ahead of it, "later" is too, and then runs LATER_SECONDS before its Stop, what is sent while it runs queued behind it until its Stop or an Escape that stops it puts the queue back in the input, unless a "racy" is queued, which is then taken as the late one ends just ahead of the Escape, that stops it instead, "garbled" is taken as other words and runs until Escape, "write" asks permission to write notes.txt beside TYPED, writing it only if allowed, then an MCP server's input, "stray" asks one under an earlier turn's prompt id, and "die", as a turn or a side question, ends the program; a side question "stubborn" writes that it was told to end, and does not."""
    script = tmp_path / "bin" / "claude"
    script.parent.mkdir()
    script.write_text(f"""#!{sys.executable}
import itertools, json, os, select, signal, sys, termios, time, tty, urllib.request
first_run = sys.argv[1:] == ["--setting-sources", "user"]
if sys.argv[1:2] == ["auth"] or first_run:
    login = os.path.join(os.environ["CLAUDE_CONFIG_DIR"], "login.json")
# `auth login`, or a first run, its screens answered and its login made at its keyboard; one that fails finishes neither.
if sys.argv[1:3] == ["auth", "login"] or first_run:
    os.makedirs(os.environ["CLAUDE_CONFIG_DIR"], exist_ok=True)
    if os.environ.get("LOGIN_EXIT", "0") != "0":
        sys.exit(int(os.environ["LOGIN_EXIT"]))
    with open(login, "w") as made:
        json.dump({{"argv": sys.argv[1:], "cwd": os.getcwd(), "settings": os.path.exists(os.path.join(os.environ["CLAUDE_CONFIG_DIR"], "settings.json")), "credentials": sorted(set(os.environ) & {{"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"}})}}, made)
    # LOGIN_UNANSWERED: quit before its last screen, which Claude Code exits 0 on as on any other.
    if first_run and not os.environ.get("LOGIN_UNANSWERED"):
        with open(os.path.join(os.environ["CLAUDE_CONFIG_DIR"], ".claude.json"), "w") as state:
            settings = os.path.join(os.environ["CLAUDE_CONFIG_DIR"], "settings.json")
            key = json.load(open(settings)).get("env", {{}}).get("ANTHROPIC_API_KEY") if os.path.exists(settings) else None
            json.dump({{"hasCompletedOnboarding": True, "projects": {{os.getcwd(): {{"hasTrustDialogAccepted": True}}}}, "customApiKeyResponses": {{"approved": [key[-20:]] if key else [], "rejected": []}}}}, state)
    sys.exit(0)
if sys.argv[1:3] == ["auth", "status"]:
    console = os.path.exists(login) and "--console" in json.load(open(login))["argv"]
    method = "api_key" if console else os.environ.get("AUTH_METHOD", "claude.ai")
    holder = {{"claude.ai": {{"email": "brain@example.com"}}, "api_key": {{"email": "brain@example.com", "apiKeySource": "/login managed key"}}, "oauth_token": {{}}}}[method]
    print(json.dumps({{"loggedIn": os.environ["LOGGED_IN"] == "1" or os.path.exists(login), "authMethod": method, "apiProvider": os.environ.get("API_PROVIDER", "firstParty"), **holder}}))
    sys.exit(0)
hooks = json.loads(sys.argv[sys.argv.index("--settings") + 1])["hooks"] if "--settings" in sys.argv else {{}}
session = sys.argv[sys.argv.index("--session-id" if "--session-id" in sys.argv else "--resume") + 1]
def post(event, **fields):
    [[url]] = [[hook["url"] for hook in matcher["hooks"]] for matcher in hooks[event]]
    body = json.dumps({{"session_id": session, "hook_event_name": event, **fields}}).encode()
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, body, {{"Content-Type": "application/json"}}), timeout=30).read())
def typed(line):
    with open(os.environ["TYPED"], "a") as log:
        log.write(json.dumps(line) + "\\n")
# As a loaded machine starts it slowly.
time.sleep(float(os.environ.get("STARTS_AFTER", "0")))
# Raw without flushing: Claude Code keeps what was typed while it started (2.1.289, measured 2026-10-04, keys typed
# 0.05s in).
tty.setraw(0, termios.TCSANOW)
# Claude Code asks its terminal for bracketed paste, and fritter pastes only into a program that asked.
os.write(1, b"\\x1b[?2004h> ")
pending, box, turn, turn_text, prompts, late, running, behind = b"", "", None, "", 0, None, None, []
# A prompt as Claude Code's UserPromptSubmit says it took it (2.1.289): a paste of four lines or more in its tags, with a
# line end added where it had none, and any other with its trailing whitespace trimmed.
def taken(text):
    if text.count("\\n") < 3:
        return text.rstrip()
    return '\\n\\n<pasted_content id="4c2e">\\n' + (text if text.endswith("\\n") else text + "\\n") + '</pasted_content id="4c2e">\\n'
def submit(text):
    global turn, turn_text, prompts, late, running
    if text.startswith("/btw "):
        if text == "/btw stubborn":
            signal.signal(signal.SIGTERM, lambda *_: typed(["sigterm", text, session]))
        typed(["btw", text[len("/btw "):], session])
        if text == "/btw die":
            os.write(1, b"bye\\r\\n")
            sys.exit(3)
        return
    if running is not None:
        behind.append(text)
        return
    if late is not None and late[1].strip() == "later":
        # Taken late, and runs on: what was sent is queued behind it.
        post("UserPromptSubmit", prompt_id=late[0], prompt=taken(late[1]))
        running, late = (late[0], late[1], time.monotonic() + float(os.environ["LATER_SECONDS"])), None
        behind.append(text)
        return
    typed(["prompt", text])
    prompts += 1
    prompt = f"p{{prompts}}"
    said = text.strip()
    if late is not None:
        # Taken late, as Claude Code takes a prompt it read after its turn had ended: then the prompt sent behind it.
        post("UserPromptSubmit", prompt_id=late[0], prompt=taken(late[1]))
        post("Stop", prompt_id=late[0], last_assistant_message="Late.")
        late = None
    if said == "deaf":
        return
    if said in ("late", "later"):
        late = prompt, text
        return
    if said == "garbled":
        post("UserPromptSubmit", prompt_id=prompt, prompt=taken(text) + " (as kept)")
        turn, turn_text = prompt, text
        return
    post("UserPromptSubmit", prompt_id=prompt, prompt=taken(text))
    if said == "die":
        os.write(1, b"bye\\r\\n")
        sys.exit(3)
    if said == "wait":
        turn, turn_text = prompt, text
        return
    if said == "slow":
        time.sleep(0.5)
    if said == "write":
        # Its setup asks before a write, held at its hook until it is answered: the file is written only on an allow,
        # and the turn goes on either way. Then an MCP server asks for input.
        notes = os.path.join(os.path.dirname(os.environ["TYPED"]), "notes.txt")
        decision = post("PermissionRequest", prompt_id=prompt, tool_name="Write", tool_input={{"file_path": notes, "content": "hello"}})["hookSpecificOutput"]["decision"]
        if decision["behavior"] == "allow":
            with open(notes, "w") as written:
                written.write("hello")
        typed(["permission", decision["behavior"], decision.get("message", "")])
        declined = post("Elicitation", prompt_id=prompt, mcp_server_name="probe", message="Which?")
        typed(["elicitation", declined["hookSpecificOutput"]["action"]])
    if said == "stray":
        # A permission an earlier turn left behind, posted under that turn's prompt id.
        stray = post("PermissionRequest", prompt_id="p0", tool_name="Write", tool_input={{"file_path": "notes.txt", "content": "hello"}})["hookSpecificOutput"]["decision"]
        typed(["permission", stray["behavior"], stray.get("message", "")])
    if said == "fail":
        post("StopFailure", prompt_id=prompt, error="unknown", last_assistant_message="API Error: 400 refused")
    else:
        post("Stop", prompt_id=prompt, last_assistant_message="Two.")
# As Claude Code reads its options: --mcp-config takes every word up to the next option or `--`, each a config or a
# config's file, and one that is neither ends the program (a prompt read as a path: ENAMETOOLONG, exit 1).
end = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
if "--mcp-config" in sys.argv:
    for value in itertools.takewhile(lambda word: not word.startswith("-"), sys.argv[sys.argv.index("--mcp-config") + 1:end]):
        try:
            json.loads(value)
        except ValueError:
            if not os.path.exists(value):
                sys.stderr.write(f"Error: MCP config file not found: {{value}}\\n")
                sys.exit(1)
# The prompt it opens with is the one word after `--`.
if end < len(sys.argv):
    [opening] = sys.argv[end + 1:]
    submit(opening)
while True:
    if running is not None and not select.select([0], [], [], max(0.0, running[2] - time.monotonic()))[0]:
        post("Stop", prompt_id=running[0], last_assistant_message="Late.")
        running, queued = None, behind
        behind = []
        for text in queued:
            submit(text)
        continue
    data = os.read(0, 65536)
    if not data:
        break
    pending += data
    # Claude Code reads what reaches it close together as one burst (2.1.286).
    time.sleep(0.05)
    while select.select([0], [], [], 0)[0]:
        more = os.read(0, 65536)
        if not more:
            break
        pending += more
    while pending:
        if pending.startswith(b"\\x1b[200~"):
            end = pending.find(b"\\x1b[201~")
            if end < 0:
                break
            box += pending[6:end].decode()
            pending = pending[end + 6:]
        elif pending.startswith(b"\\x1b"):
            pending = pending[1:]
            typed(["escape", box])
            if running is not None and [queued.strip() for queued in behind] == ["racy"]:
                # The late prompt ended, and the one queued behind it was taken, just before the Escape, which stops it.
                post("Stop", prompt_id=running[0], last_assistant_message="Late.")
                prompts += 1
                post("UserPromptSubmit", prompt_id=f"p{{prompts}}", prompt=taken(behind[0]))
                box, running = behind[0], None
                behind = []
            if running is not None:
                # As Claude Code does, the stopped prompt is put back in the input, and the queue behind it with it; no Stop.
                box, running = running[1] + "".join(behind), None
                behind = []
            if turn is not None:
                # As Claude Code does, the stopped prompt is put back in the input.
                box, turn = turn_text, None
        elif pending.startswith(b"\x03"):
            pending = pending[1:]
            typed(["ctrl_c", box])
            box = ""
        # A newline, as a Return typed before it read raw reaches it, is one more line in the input: Claude Code takes
        # Ctrl+J as multiline input (its docs), and sends on a Return alone.
        elif pending.startswith(b"\\r"):
            pending = pending[1:]
            if pending:
                # A Return with more behind it in the same burst is read as pasted, and sends nothing (2.1.286).
                continue
            text, box = box, ""
            submit(text)
        else:
            box += pending[:1].decode()
            pending = pending[1:]
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("LOGGED_IN", "1")
    monkeypatch.setenv("TYPED", str(tmp_path / "typed.jsonl"))
    monkeypatch.setenv("LATER_SECONDS", "1")
    return script


async def unedited() -> SettingsEdited:
    """Settings a run's start watches that are never edited."""
    await asyncio.Event().wait()
    raise AssertionError("an event nothing sets was set")
