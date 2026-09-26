"""The summariser: one stateless call on the configured model, apart from the intermediary's conversation."""

from collections.abc import Awaitable, Callable

from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
from openai import AsyncOpenAI

from hands.voice.pipeline import AnthropicBackend, LLMBackend, OpenAICompatibleBackend

# A rendered turn in, the spoken summary out.
Summariser = Callable[[str], Awaitable[str]]


class SummaryFailed(Exception):
    """The model answered, but with nothing that can be spoken."""


def summariser(backend: LLMBackend, instruction: str, max_tokens: int, timeout: float) -> Summariser:
    """The one place the backend variant is inspected for summaries."""
    # [LAW:one-type-per-behavior] both backends take the same turn and give the same text; only the client differs.
    # No retries: a summary that fails is said at once, not after a backoff that sounds like nothing happened.
    match backend:
        case OpenAICompatibleBackend(base_url=base_url, api_key=api_key, model=model):
            openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=timeout)

            async def from_openai(turn: str) -> str:
                completion = await openai_client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "system", "content": instruction}, {"role": "user", "content": turn}],
                )
                return _spoken([choice.message.content or "" for choice in completion.choices[:1]])

            return from_openai
        case AnthropicBackend(api_key=api_key, model=model):
            anthropic_client = AsyncAnthropic(api_key=api_key, max_retries=0, timeout=timeout)

            async def from_anthropic(turn: str) -> str:
                message = await anthropic_client.messages.create(
                    model=model, max_tokens=max_tokens, system=instruction, messages=[{"role": "user", "content": turn}]
                )
                return _spoken([block.text for block in message.content if isinstance(block, TextBlock)])

            return from_anthropic


def _spoken(parts: list[str]) -> str:
    text = " ".join(part.strip() for part in parts).strip()
    if not text:
        raise SummaryFailed("the model returned no summary")
    return text
