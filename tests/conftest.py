"""Fixtures more than one test module needs."""

import asyncio
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import mlx_whisper
import pytest
from aiohttp import web
from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner

from hands.sessions.wrapper import FRITTER_SOURCE

# How long a pipeline may take to start before a test fails on it.
STARTUP_SECS = 5.0


@pytest.fixture(autouse=True)
def no_whisper_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building a Whisper transcribes nothing: its load would fetch 1.6 GB of weights, which no test is about."""

    def transcribe(_audio: object, **_options: object) -> dict[str, object]:
        return {"segments": []}

    monkeypatch.setattr(mlx_whisper, "transcribe", transcribe)


def _dead_pid() -> int:
    process = subprocess.Popen(["true"])
    process.wait()
    return process.pid


@pytest.fixture
def dead_pid() -> Callable[[], int]:
    """Makes the pid of a process that has just exited, which no process holds for now, fresh at each call."""
    return _dead_pid


@pytest.fixture
def python312(tmp_path: Path) -> str:
    """A PATH whose only Python new enough is a python3.12 outside any venv, as the plugin's launcher finds one; /usr/bin's python3 on macOS is 3.9."""
    interpreters = tmp_path / "bin"
    interpreters.mkdir()
    (interpreters / "python3.12").symlink_to(Path(getattr(sys, "_base_executable", sys.executable)).resolve())
    return f"/usr/bin:/bin:{interpreters}"


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
async def running(processors: list[FrameProcessor]) -> AsyncGenerator[Running]:
    """Runs the processors as one pipeline from its start until the block ends, then cancels it."""
    worker = PipelineWorker(Pipeline(processors), idle_timeout_secs=None)
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
def fritter(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """fritter, built from this checkout once for the run."""
    if shutil.which("go") is None:
        pytest.skip("needs go to build fritter")
    built = tmp_path_factory.mktemp("fritter") / "fritter"
    subprocess.run(["go", "build", "-o", str(built), "."], cwd=FRITTER_SOURCE, check=True, capture_output=True)
    return built


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A `claude` first on PATH that reports its login from LOGGED_IN, or from an `auth login` it recorded, made by AUTH_METHOD (claude.ai unless named), and as the brain is Claude Code at a keyboard: it
    reads its terminal raw, in bursts, takes a prompt when a Return that ends a burst sends it, and posts the hooks its --settings name. Everything it
    reads is written, one line each, to the file TYPED names, a side question with the session it was asked under; a side
    question it is started with is taken as if typed. A turn "wait" runs until Escape, "fail" is failed by the API, "deaf"
    is never taken, and "die", as a turn or a side question, ends the program; a side question "stubborn" writes that it was told to end, and does not."""
    script = tmp_path / "bin" / "claude"
    script.parent.mkdir()
    script.write_text(f"""#!{sys.executable}
import json, os, select, signal, sys, time, tty, urllib.request
if sys.argv[1] == "auth":
    login = os.path.join(os.environ["CLAUDE_CONFIG_DIR"], "login.json")
if sys.argv[1:3] == ["auth", "login"]:
    os.makedirs(os.environ["CLAUDE_CONFIG_DIR"], exist_ok=True)
    with open(login, "w") as made:
        json.dump({{"argv": sys.argv[1:], "credentials": sorted(set(os.environ) & {{"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"}})}}, made)
    sys.exit(int(os.environ.get("LOGIN_EXIT", "0")))
if sys.argv[1:3] == ["auth", "status"]:
    print(json.dumps({{"loggedIn": os.environ["LOGGED_IN"] == "1" or os.path.exists(login), "authMethod": os.environ.get("AUTH_METHOD", "claude.ai"), "email": "brain@example.com"}}))
    sys.exit(0)
hooks = json.loads(sys.argv[sys.argv.index("--settings") + 1])["hooks"] if "--settings" in sys.argv else {{}}
session = sys.argv[sys.argv.index("--session-id") + 1]
def post(event, **fields):
    [[url]] = [[hook["url"] for hook in matcher["hooks"]] for matcher in hooks[event]]
    body = json.dumps({{"session_id": session, "hook_event_name": event, **fields}}).encode()
    urllib.request.urlopen(urllib.request.Request(url, body, {{"Content-Type": "application/json"}}), timeout=5).read()
def typed(line):
    with open(os.environ["TYPED"], "a") as log:
        log.write(json.dumps(line) + "\\n")
tty.setraw(0)
# Claude Code asks its terminal for bracketed paste, and fritter pastes only into a program that asked.
os.write(1, b"\\x1b[?2004h> ")
pending, box, turn, turn_text, prompts = b"", "", None, "", 0
def submit(text):
    global turn, turn_text, prompts
    if text.startswith("/btw "):
        if text == "/btw stubborn":
            signal.signal(signal.SIGTERM, lambda *_: typed(["sigterm", text, session]))
        typed(["btw", text[len("/btw "):], session])
        if text == "/btw die":
            os.write(1, b"bye\\r\\n")
            sys.exit(3)
        return
    typed(["prompt", text])
    prompts += 1
    prompt = f"p{{prompts}}"
    said = text.strip()
    if said == "deaf":
        return
    post("UserPromptSubmit", prompt_id=prompt, prompt=text)
    if said == "die":
        os.write(1, b"bye\\r\\n")
        sys.exit(3)
    if said == "wait":
        turn, turn_text = prompt, text
        return
    if said == "slow":
        time.sleep(0.5)
    if said == "fail":
        post("StopFailure", prompt_id=prompt, error="unknown", last_assistant_message="API Error: 400 refused")
    else:
        post("Stop", prompt_id=prompt, last_assistant_message="Two.")
if sys.argv[-1].startswith("/btw "):
    submit(sys.argv[-1])
while True:
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
            if turn is not None:
                # As Claude Code does, the stopped prompt is put back in the input.
                box, turn = turn_text, None
        elif pending.startswith(b"\x03"):
            pending = pending[1:]
            typed(["ctrl_c", box])
            box = ""
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
    return script
