"""The pipeline's LLM service reaches the backend it was built from, with that backend's key, and a reply with nothing in it is the model's failure."""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

import pytest
from aiohttp import web
from pipecat.frames.frames import ErrorFrame, Frame, InterruptionFrame, LLMContextFrame, LLMFullResponseEndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.workers.runner import WorkerRunner

from conftest import ServeChat
from hands.sessions.model_facts import ModelFault, ModelReplyEmpty
from hands.voice.pipeline import AnthropicBackend, OpenAICompatibleBackend, build_llm
from hands.voice.tools import pipecat_function, stay_silent_tool

PATIENCE_SECS = 5.0


async def test_the_openai_compatible_service_sends_the_backend_key_to_the_backend_url(chat_server: ServeChat) -> None:
    server = await chat_server("Two sessions are running.")
    llm = build_llm(OpenAICompatibleBackend(base_url=server.url, api_key="k", model="m"), instruction="Speak.", max_tokens=50)
    reply = await llm.run_inference(LLMContext(messages=[{"role": "user", "content": "What is running?"}]))
    assert reply == "Two sessions are running."
    [request] = server.asked
    assert request["model"] == "m"
    assert server.keys == ["k"]


async def test_the_anthropic_service_sends_the_backend_key_to_the_backend_url(chat_server: ServeChat) -> None:
    server = await chat_server("Two sessions are running.")
    llm = build_llm(AnthropicBackend(base_url=server.anthropic_url, api_key="k", model="m"), instruction="Speak.", max_tokens=50)
    reply = await llm.run_inference(LLMContext(messages=[{"role": "user", "content": "What is running?"}]))
    assert reply == "Two sessions are running."
    [request] = server.asked
    assert request["model"] == "m"
    assert server.keys == ["k"]


# What each API streams for a reply: nothing at all, words, or a call to stay_silent, which is the model choosing silence.
Reply = Literal["empty", "words", "stay_silent"]


def openai_stream(reply: Reply) -> list[str]:
    """The chunks a chat completions server streams; "empty" is the one codexapi.pro sends for a turn that wanted a tool."""
    def chunk(delta: dict[str, object], finish: str | None) -> dict[str, object]:
        return {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    match reply:
        case "empty":
            chunks = [chunk({"role": "assistant", "content": ""}, "stop")]
        case "words":
            chunks = [chunk({"role": "assistant", "content": "Two sessions are running."}, None), chunk({}, "stop")]
        case "stay_silent":
            call = {"index": 0, "id": "call_1", "type": "function", "function": {"name": "stay_silent", "arguments": "{}"}}
            chunks = [chunk({"role": "assistant", "tool_calls": [call]}, None), chunk({}, "tool_calls")]
    return [*(f"data: {json.dumps(each)}\n\n" for each in chunks), "data: [DONE]\n\n"]


def anthropic_stream(reply: Reply) -> list[str]:
    """The events an Anthropic messages server streams for the same three replies."""
    started: dict[str, object] = {"type": "message_start", "message": {"id": "m1", "type": "message", "role": "assistant", "model": "m", "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}}
    match reply:
        case "empty":
            blocks: list[dict[str, object]] = []
        case "words":
            blocks = [
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Two sessions are running."}},
                {"type": "content_block_stop", "index": 0},
            ]
        case "stay_silent":
            blocks = [
                {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "stay_silent", "input": {}}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
                {"type": "content_block_stop", "index": 0},
            ]
    stop = "tool_use" if reply == "stay_silent" else "end_turn"
    ended: dict[str, object] = {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": {"input_tokens": 1, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}
    return [f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in [started, *blocks, ended, {"type": "message_stop"}]]


@dataclass
class Streaming:
    """A server streaming one reply from both endpoints, until `release` is set when the test holds it open."""

    url: str
    anthropic_url: str
    release: asyncio.Event


@pytest.fixture
async def streaming() -> AsyncIterator[Callable[[Reply, bool], Awaitable[Streaming]]]:
    runners: list[web.AppRunner] = []

    async def serve(reply: Reply, held: bool) -> Streaming:
        release = asyncio.Event()
        if not held:
            release.set()

        async def stream(request: web.Request, events: list[str]) -> web.StreamResponse:
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await release.wait()
            for event in events:
                await response.write(event.encode())
            return response

        async def complete(request: web.Request) -> web.StreamResponse:
            return await stream(request, openai_stream(reply))

        async def message(request: web.Request) -> web.StreamResponse:
            return await stream(request, anthropic_stream(reply))

        app = web.Application()
        app.router.add_post("/v1/chat/completions", complete)
        app.router.add_post("/v1/messages", message)
        runner = web.AppRunner(app)
        runners.append(runner)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        host = f"http://127.0.0.1:{runner.addresses[0][1]}"
        return Streaming(f"{host}/v1", host, release)

    yield serve
    for runner in runners:
        await runner.cleanup()


def service(shape: Literal["openai", "anthropic"], server: Streaming) -> AnthropicLLMService | OpenAILLMService:
    match shape:
        case "openai":
            return build_llm(OpenAICompatibleBackend(base_url=server.url, api_key="k", model="m"), instruction="Speak.", max_tokens=50)
        case "anthropic":
            return build_llm(AnthropicBackend(base_url=server.anthropic_url, api_key="k", model="m"), instruction="Speak.", max_tokens=50)


@dataclass
class Running:
    worker: PipelineWorker
    errors: list[ErrorFrame]
    ended: asyncio.Event


@asynccontextmanager
async def running(llm: FrameProcessor) -> AsyncGenerator[Running]:
    """The service in a pipeline of its own, the way the daemon runs it: its errors are what the system channel says."""
    worker = PipelineWorker(Pipeline([llm, Ends(ended := asyncio.Event())]), idle_timeout_secs=None)
    errors: list[ErrorFrame] = []
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def _started(_worker: PipelineWorker, _frame: Frame) -> None:  # pyright: ignore[reportUnusedFunction]
        started.set()

    @worker.event_handler("on_pipeline_error")
    async def _failed(_worker: PipelineWorker, error: ErrorFrame) -> None:  # pyright: ignore[reportUnusedFunction]
        errors.append(error)

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    task = asyncio.create_task(runner.run())
    await asyncio.wait_for(started.wait(), PATIENCE_SECS)
    try:
        yield Running(worker, errors, ended)
    finally:
        await worker.cancel()
        await task


class Ends(FrameProcessor):
    """Sets `ended` once a reply's end frame has passed the service."""

    def __init__(self, ended: asyncio.Event) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._ended = ended

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseEndFrame):
            self._ended.set()
        await self.push_frame(frame, direction)


def asked() -> LLMContextFrame:
    return LLMContextFrame(LLMContext(messages=[{"role": "user", "content": "What is running?"}], tools=[pipecat_function(stay_silent_tool())]))  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("shape", ["openai", "anthropic"])
async def test_an_empty_reply_is_reported_as_the_models_empty_reply(streaming: Callable[[Reply, bool], Awaitable[Streaming]], shape: Literal["openai", "anthropic"]) -> None:
    llm = service(shape, await streaming("empty", False))
    async with running(llm) as run:
        await run.worker.queue_frame(asked())
        await asyncio.wait_for(run.ended.wait(), PATIENCE_SECS)
        [error] = run.errors
        assert error.processor is llm
        assert isinstance(error.exception, ModelFault) and error.exception.fact == ModelReplyEmpty()


@pytest.mark.parametrize("shape", ["openai", "anthropic"])
@pytest.mark.parametrize("reply", ["words", "stay_silent"])
async def test_a_reply_with_words_or_a_call_to_stay_silent_is_no_failure(streaming: Callable[[Reply, bool], Awaitable[Streaming]], shape: Literal["openai", "anthropic"], reply: Reply) -> None:
    llm = service(shape, await streaming(reply, False))
    async with running(llm) as run:
        await run.worker.queue_frame(asked())
        await asyncio.wait_for(run.ended.wait(), PATIENCE_SECS)
        await asyncio.sleep(0.1)
        assert run.errors == []


@pytest.mark.parametrize("shape", ["openai", "anthropic"])
async def test_a_reply_the_user_spoke_over_is_no_empty_reply(streaming: Callable[[Reply, bool], Awaitable[Streaming]], shape: Literal["openai", "anthropic"]) -> None:
    server = await streaming("empty", True)
    llm = service(shape, server)
    async with running(llm) as run:
        await run.worker.queue_frame(asked())
        await asyncio.sleep(0.2)
        # Pipecat closes the reply it cancels with the same end frame a finished reply has.
        await run.worker.queue_frame(InterruptionFrame())
        await asyncio.wait_for(run.ended.wait(), PATIENCE_SECS)
        server.release.set()
        await asyncio.sleep(0.2)
        assert run.errors == []
