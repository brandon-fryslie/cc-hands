"""Fixtures more than one test module needs."""

import subprocess
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import pytest
from aiohttp import web


def _dead_pid() -> int:
    process = subprocess.Popen(["true"])
    process.wait()
    return process.pid


@pytest.fixture
def dead_pid() -> Callable[[], int]:
    """Makes the pid of a process that has just exited, which no process holds for now, fresh at each call."""
    return _dead_pid


@dataclass
class ChatServer:
    """A chat completions endpoint at `url`, and what it has been asked: each request's body and bearer token."""

    url: str
    asked: list[dict[str, object]]
    keys: list[str]


ServeChat = Callable[[str | None], Awaitable[ChatServer]]


@pytest.fixture
async def chat_server() -> AsyncIterator[ServeChat]:
    """Starts a chat completions endpoint that answers every request with the content given; stopped after the test."""
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

        app = web.Application()
        app.router.add_post("/v1/chat/completions", complete)
        runner = web.AppRunner(app)
        runners.append(runner)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        return ChatServer(url=f"http://127.0.0.1:{runner.addresses[0][1]}/v1", asked=asked, keys=keys)

    yield serve
    for runner in runners:
        await runner.cleanup()
