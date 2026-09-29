"""Fixtures more than one test module needs."""

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
