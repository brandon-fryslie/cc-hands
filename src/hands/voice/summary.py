"""The summariser: one stateless call on the configured model, apart from the intermediary's conversation, or under
the brain, a side question asked of a Claude Code of its own."""

from collections.abc import Awaitable, Callable

from anthropic import AnthropicError, AsyncAnthropic, Omit, omit
from anthropic.types import TextBlock, ThinkingConfigDisabledParam
from openai import AsyncOpenAI, OpenAIError
from pipecat.services.anthropic.llm import _SONNET_THINKS_BY_DEFAULT_FROM, _sonnet_generation  # pyright: ignore[reportPrivateUsage]

from hands.brain.asides import AsideFailed, AsideKind, Deadline, Within
from hands.sessions.audit import Record
from hands.sessions.wide import annotate, unit
from hands.voice.backends import AnthropicBackend, OpenAICompatibleBackend
from hands.voice.token_usage import USAGE, openai_usage, reported

# A rendered turn in, the spoken summary out.
Summariser = Callable[[str], Awaitable[str]]


class SummaryFailed(Exception):
    """The model answered, but with nothing that can be spoken, or its side question had no answer."""


# Everything one summariser call is expected to fail with: the model's answer unusable, or its API refusing.
SUMMARY_FAILURES = (SummaryFailed, OpenAIError, AnthropicError, OSError)


def summariser(
    backend: AnthropicBackend | OpenAICompatibleBackend, record: Record, kind: AsideKind, instruction: str, max_tokens: int, timeout: float
) -> Summariser:
    """The summariser on an API: the one place its variant is inspected for summaries. Each call is one wide event,
    `model.request`, saying what it was asked for, how it ended, and the usage its API reported."""
    # [LAW:one-type-per-behavior] both backends take the same turn and give the same text; only the client differs.
    # No retries: a summary that fails is said at once, not after a backoff that sounds like nothing happened.
    match backend:
        case OpenAICompatibleBackend(base_url=base_url, api_key=api_key, model=model):
            openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=timeout)

            async def from_openai(turn: str) -> str:
                with unit("model.request", record):
                    annotate(kind=kind)
                    completion = await openai_client.chat.completions.create(
                        model=model,
                        max_tokens=max_tokens,
                        messages=[{"role": "system", "content": instruction}, {"role": "user", "content": turn}],
                    )
                    annotate(**openai_usage(completion.usage))
                    return _spoken([choice.message.content or "" for choice in completion.choices[:1]])

            return from_openai
        case AnthropicBackend(base_url=base_url, api_key=api_key, model=model):
            anthropic_client = AsyncAnthropic(base_url=base_url, api_key=api_key, max_retries=0, timeout=timeout)

            async def from_anthropic(turn: str) -> str:
                with unit("model.request", record):
                    annotate(kind=kind)
                    message = await anthropic_client.messages.create(
                        model=model, max_tokens=max_tokens, system=instruction, messages=[{"role": "user", "content": turn}], thinking=thinking(model)
                    )
                    annotate(**reported(message.usage, USAGE))
                    return _spoken([block.text for block in message.content if isinstance(block, TextBlock)])

            return from_anthropic


def aside(ask: Callable[[str, Within], Awaitable[str]], instruction: str, timeout: float) -> Summariser:
    """The summariser under the brain: the turn asked as a side question, with what to make of it, of a Claude Code that
    is asked nothing else, within `timeout`, as the API's are, whatever it waited on."""

    async def from_an_aside(turn: str) -> str:
        try:
            return _spoken([await ask(f"{instruction}\n\nSummarize this:\n\n{turn}", Deadline(timeout))])
        except AsideFailed as error:
            raise SummaryFailed(f"the side question had no answer: {error}") from error

    return from_an_aside


def thinking(model: str) -> ThinkingConfigDisabledParam | Omit:
    """Thinking off for a model that would otherwise think, and left out for the rest, which reject the setting or do not need it."""
    # [LAW:one-source-of-truth] Pipecat's rule for which models think unasked, the one the pipeline's own service applies:
    # a summary that thinks spends its few tokens before it says a word.
    generation = _sonnet_generation(model)
    return {"type": "disabled"} if generation is not None and generation >= _SONNET_THINKS_BY_DEFAULT_FROM else omit


def _spoken(parts: list[str]) -> str:
    text = " ".join(part.strip() for part in parts).strip()
    if not text:
        raise SummaryFailed("the model returned no summary")
    return text
