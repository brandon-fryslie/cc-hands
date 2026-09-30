"""Fixtures more than one test module needs."""

import os
import stat
import subprocess
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import mlx_whisper
import pytest
from aiohttp import web


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


@dataclass
class ChatServer:
    """A chat completions endpoint at `url` and an Anthropic messages one at `anthropic_url`, and what they have been asked: each request's body and key."""

    url: str
    # The Anthropic SDK appends /v1/messages itself, so its base is the bare host.
    anthropic_url: str
    asked: list[dict[str, object]]
    keys: list[str]


ServeChat = Callable[[str | None], Awaitable[ChatServer]]


@pytest.fixture
async def chat_server() -> AsyncIterator[ServeChat]:
    """Starts a server whose two endpoints answer every request with the content given; stopped after the test."""
    runners: list[web.AppRunner] = []

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

        app = web.Application()
        app.router.add_post("/v1/chat/completions", complete)
        app.router.add_post("/v1/messages", message)
        runner = web.AppRunner(app)
        runners.append(runner)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        host = f"http://127.0.0.1:{runner.addresses[0][1]}"
        return ChatServer(url=f"{host}/v1", anthropic_url=host, asked=asked, keys=keys)

    yield serve
    for runner in runners:
        await runner.cleanup()


INIT = {"type": "system", "subtype": "init", "session_id": "b1", "model": "claude-sonnet-5", "tools": ["Read", "mcp__hands__list_sessions"], "mcp_servers": [{"name": "hands", "status": "connected"}]}
RESULT = {"type": "result", "subtype": "success", "is_error": False, "num_turns": 1, "duration_ms": 812, "result": "Two."}


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A `claude` first on PATH that reports its login from LOGGED_IN, and as the brain answers each turn with RESULT and
    each side question with "Forked: " and the question."""
    script = tmp_path / "bin" / "claude"
    script.parent.mkdir()
    script.write_text(f"""#!{sys.executable}
import json, os, sys, time
if sys.argv[1:3] == ["auth", "status"]:
    print(json.dumps({{"loggedIn": os.environ["LOGGED_IN"] == "1"}}))
    sys.exit(0)
if "json" in sys.argv and "stream-json" not in sys.argv:
    turn = sys.stdin.read()
    print(json.dumps({{"type": "result", "is_error": turn == "fail", "result": "Summed: " + turn + " in " + os.getcwd() + " via " + os.environ["ANTHROPIC_BASE_URL"]}}))
    sys.exit(0)
print(json.dumps({INIT!r}), flush=True)
print("not json", file=sys.stderr, flush=True)
def respond(request_id, question):
    if question == "fail":
        response = {{"subtype": "error", "request_id": request_id, "error": "no snapshot"}}
    else:
        reply = None if question == "silent" else "Forked: " + question
        response = {{"subtype": "success", "request_id": request_id, "response": {{"response": reply, "synthetic": False}}}}
    print(json.dumps({{"type": "control_response", "response": response}}), flush=True)
# A turn "wait" is answered after the next side question, and a side question "hold" after the one that follows it.
waiting, holding = False, None
for line in sys.stdin:
    said = json.loads(line)
    if said["type"] == "control_request":
        request_id, question = said["request_id"], said["request"].get("question")
        if question is None:
            print(json.dumps({{"type": "control_response", "response": {{"subtype": "success", "request_id": request_id}}}}), flush=True)
            continue
        if question == "hold":
            holding = request_id
            continue
        respond(request_id, question)
        if holding is not None:
            respond(holding, "hold")
            holding = None
        if waiting:
            waiting = False
            print(json.dumps({RESULT!r}), flush=True)
        continue
    content = said["message"]["content"]
    if content == "wait":
        waiting = True
        continue
    if content == "die":
        sys.exit(3)
    if content == "slow":
        time.sleep(0.5)
    if content == "flood":
        print("x" * (17 * 1024 * 1024), flush=True)
        time.sleep(60)
    print(json.dumps({{"type": "stream_event"}}), flush=True)
    print(json.dumps({RESULT!r}), flush=True)
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("LOGGED_IN", "1")
    return script
