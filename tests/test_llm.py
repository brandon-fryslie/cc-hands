"""The pipeline's LLM service reaches the backend it was built from, with that backend's key, and a reply with nothing in it is the model's failure."""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

import pytest
from aiohttp import web
from anthropic import AsyncAnthropic
from openai.types.chat import ChatCompletionMessageFunctionToolCallParam, ChatCompletionUserMessageParam
from pipecat.frames.frames import Frame, InterruptionFrame, LLMContextFrame, LLMFullResponseEndFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.openai.llm import OpenAILLMService

from conftest import Api, ServeApi, ServeChat, running
from hands.sessions.model_facts import ModelFault, ModelReplyEmpty, ModelUnreachable
from hands.voice.pipeline import AnthropicBackend, AnthropicService, OpenAICompatibleBackend, build_llm
from hands.voice.system import model_fact
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
    """A served model streaming one reply from both endpoints: `received` is set as a request arrives, and the reply is
    streamed once `release` is set, at once when the test does not hold it."""

    api: Api
    received: asyncio.Event
    release: asyncio.Event


@pytest.fixture
async def streaming(api_server: ServeApi) -> AsyncIterator[Callable[[Reply, bool], Awaitable[Streaming]]]:
    releases: list[asyncio.Event] = []

    async def serve(reply: Reply, held: bool) -> Streaming:
        received, release = asyncio.Event(), asyncio.Event()
        releases.append(release)
        if not held:
            release.set()

        async def stream(request: web.Request, events: list[str]) -> web.StreamResponse:
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            received.set()
            await release.wait()
            for event in events:
                await response.write(event.encode())
            return response

        async def complete(request: web.Request) -> web.StreamResponse:
            return await stream(request, openai_stream(reply))

        async def message(request: web.Request) -> web.StreamResponse:
            return await stream(request, anthropic_stream(reply))

        return Streaming(await api_server(complete, message), received, release)

    yield serve
    # A reply still held would keep its handler, and the server's cleanup, waiting for good.
    for release in releases:
        release.set()


Shape = Literal["openai", "anthropic"]


def service(shape: Shape, server: Streaming) -> AnthropicLLMService | OpenAILLMService:
    match shape:
        case "openai":
            return build_llm(OpenAICompatibleBackend(base_url=server.api.url, api_key="k", model="m"), instruction="Speak.", max_tokens=50)
        case "anthropic":
            return build_llm(AnthropicBackend(base_url=server.api.anthropic_url, api_key="k", model="m"), instruction="Speak.", max_tokens=50)


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
async def test_an_empty_reply_is_reported_as_the_models_empty_reply(streaming: Callable[[Reply, bool], Awaitable[Streaming]], shape: Shape) -> None:
    llm = service(shape, await streaming("empty", False))
    async with running([llm, Ends(ended := asyncio.Event())]) as run:
        await run.worker.queue_frame(asked())
        await asyncio.wait_for(ended.wait(), PATIENCE_SECS)
        [error] = run.errors
        assert error.processor is llm
        assert isinstance(error.exception, ModelFault) and error.exception.fact == ModelReplyEmpty()


@pytest.mark.parametrize("shape", ["openai", "anthropic"])
@pytest.mark.parametrize("reply", ["words", "stay_silent"])
async def test_a_reply_with_words_or_a_call_to_stay_silent_is_no_failure(streaming: Callable[[Reply, bool], Awaitable[Streaming]], shape: Shape, reply: Reply) -> None:
    llm = service(shape, await streaming(reply, False))
    async with running([llm, Ends(ended := asyncio.Event())]) as run:
        await run.worker.queue_frame(asked())
        await asyncio.wait_for(ended.wait(), PATIENCE_SECS)
        await asyncio.sleep(0.1)
        assert run.errors == []


@pytest.mark.parametrize("shape", ["openai", "anthropic"])
async def test_a_reply_the_user_spoke_over_is_no_empty_reply(streaming: Callable[[Reply, bool], Awaitable[Streaming]], shape: Shape) -> None:
    server = await streaming("empty", True)
    llm = service(shape, server)
    async with running([llm, Ends(ended := asyncio.Event())]) as run:
        await run.worker.queue_frame(asked())
        await asyncio.wait_for(server.received.wait(), PATIENCE_SECS)
        # Pipecat closes the reply it cancels with the same end frame a finished reply has.
        await run.worker.queue_frame(InterruptionFrame())
        await asyncio.wait_for(ended.wait(), PATIENCE_SECS)
        await asyncio.sleep(0.2)
        assert run.errors == []


@pytest.mark.parametrize("shape", ["openai", "anthropic"])
async def test_a_reply_cut_off_by_the_pipeline_stopping_is_no_empty_reply(streaming: Callable[[Reply, bool], Awaitable[Streaming]], shape: Shape) -> None:
    server = await streaming("empty", True)
    llm = service(shape, server)
    async with running([llm, Ends(asyncio.Event())]) as run:
        await run.worker.queue_frame(asked())
        await asyncio.wait_for(server.received.wait(), PATIENCE_SECS)
    # Stopped mid-request, as the daemon stops: its errors are all in by the time the pipeline has.
    assert run.errors == []


async def test_an_anthropic_request_that_times_out_is_said_as_the_model_out_of_reach(streaming: Callable[[Reply, bool], Awaitable[Streaming]]) -> None:
    server = await streaming("empty", True)
    llm = AnthropicService(
        api_key="k",
        client=AsyncAnthropic(base_url=server.api.anthropic_url, api_key="k", max_retries=0, timeout=0.2),
        settings=AnthropicService.Settings(model="m", system_instruction="Speak.", max_tokens=50),
    )
    async with running([llm, Ends(ended := asyncio.Event())]) as run:
        await run.worker.queue_frame(asked())
        await asyncio.wait_for(ended.wait(), PATIENCE_SECS)
        [error] = run.errors
        assert model_fact(error) == ModelUnreachable()


def answered(*after: ChatCompletionUserMessageParam) -> LLMContextFrame:
    """The context Pipecat asks the model again with once a call's result is in, with any note added while the call ran."""
    call: ChatCompletionMessageFunctionToolCallParam = {"id": "call_1", "type": "function", "function": {"name": "read_session", "arguments": "{}"}}
    return LLMContextFrame(
        LLMContext(
            messages=[
                {"role": "user", "content": "What is api doing?"},
                {"role": "assistant", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "call_1", "content": '{"steps": []}'},
                *after,
            ],
            tools=[pipecat_function(stay_silent_tool())],  # pyright: ignore[reportArgumentType]
        )
    )


@pytest.mark.parametrize("shape", ["openai", "anthropic"])
@pytest.mark.parametrize("after", [(), ({"role": "user", "content": "[hands] web finished a turn."},)])
async def test_an_empty_reply_to_a_calls_result_is_no_failure(streaming: Callable[[Reply, bool], Awaitable[Streaming]], shape: Shape, after: tuple[ChatCompletionUserMessageParam, ...]) -> None:
    """Claude often ends a turn with nothing once a call has done what it was for, as under the brain."""
    llm = service(shape, await streaming("empty", False))
    async with running([llm, Ends(ended := asyncio.Event())]) as run:
        await run.worker.queue_frame(answered(*after))
        await asyncio.wait_for(ended.wait(), PATIENCE_SECS)
        await asyncio.sleep(0.1)
        assert run.errors == []
