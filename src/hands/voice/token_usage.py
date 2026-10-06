"""The token usage a model API reports for a request, as the facts of its `model.request` event."""

from typing import Any

from openai.types import CompletionUsage
from pipecat.metrics.metrics import LLMTokenUsage

from hands.sessions.wide import Fact

# What a request read fresh, wrote to the prompt cache, and read from it, as message_start and message_delta report them;
# and what the model wrote back, which only message_delta does: message_start's is a count before the reply is written.
INPUT_USAGE = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
USAGE = (*INPUT_USAGE, "output_tokens")


def reported(usage: Any, names: tuple[str, ...]) -> dict[str, Fact]:
    """The `names` an Anthropic usage reports; a field it leaves out is not one it reported."""
    return {name: value for name in names if (value := getattr(usage, name)) is not None}


def openai_usage(usage: CompletionUsage | None) -> dict[str, Fact]:
    """An OpenAI-compatible usage under the names Pipecat gives a streamed one, so every `model.request` on these APIs
    says the same facts by the same names; none where the API reported none, as these APIs may. Its prompt_tokens is
    gross of the cache, where Anthropic's input_tokens is net."""
    if usage is None:
        return {}
    prompt, completion = usage.prompt_tokens_details, usage.completion_tokens_details
    tokens = LLMTokenUsage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        cache_read_input_tokens=prompt.cached_tokens if prompt else None,
        reasoning_tokens=completion.reasoning_tokens if completion else None,
    )
    return tokens.model_dump(exclude_none=True)
