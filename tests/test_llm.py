"""The pipeline's LLM service reaches the backend it was built from, with that backend's key."""

from pipecat.processors.aggregators.llm_context import LLMContext

from conftest import ServeChat
from hands.voice.pipeline import OpenAICompatibleBackend, build_llm


async def test_the_openai_compatible_service_sends_the_backend_key_to_the_backend_url(chat_server: ServeChat) -> None:
    server = await chat_server("Two sessions are running.")
    llm = build_llm(OpenAICompatibleBackend(base_url=server.url, api_key="k", model="m"), instruction="Speak.", max_tokens=50)
    reply = await llm.run_inference(LLMContext(messages=[{"role": "user", "content": "What is running?"}]))
    assert reply == "Two sessions are running."
    [request] = server.asked
    assert request["model"] == "m"
    assert server.keys == ["k"]
