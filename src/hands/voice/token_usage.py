"""The token usage a model API reports for a request, as the facts of its `model.request` event."""

from functools import reduce
from typing import Any

from openai.types import CompletionUsage

from hands.sessions.wide import Fact

# What a request read fresh, wrote to the prompt cache, and read from it, as message_start and message_delta report them;
# and what the model wrote back, which only message_delta does: message_start's is a count before the reply is written.
INPUT_USAGE = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
USAGE = (*INPUT_USAGE, "output_tokens")

# Where each fact sits on an OpenAI-compatible usage, under the name Pipecat's LLMTokenUsage gives it on a streamed one,
# so every `model.request` on these APIs says the same facts by the same names. Its prompt_tokens is gross of the cache,
# where Anthropic's input_tokens is net.
OPENAI_USAGE = {
    "prompt_tokens": ("prompt_tokens",),
    "completion_tokens": ("completion_tokens",),
    "total_tokens": ("total_tokens",),
    "cache_read_input_tokens": ("prompt_tokens_details", "cached_tokens"),
    "cache_creation_input_tokens": ("prompt_tokens_details", "cache_write_tokens"),
    "reasoning_tokens": ("completion_tokens_details", "reasoning_tokens"),
}


def reported(usage: Any, names: tuple[str, ...]) -> dict[str, Fact]:
    """The `names` an Anthropic usage reports. The SDK builds what the API sent unchecked, so the usage, or any count in
    it, may be missing: one left out is not one it reported, never a failed request."""
    return {name: value for name in names if (value := getattr(usage, name, None)) is not None}


def openai_usage(usage: CompletionUsage | None) -> dict[str, Fact]:
    """The facts an OpenAI-compatible usage reports. These APIs may report none, or leave any count out or null, and
    the SDK builds what they sent unchecked: a count left out is not one they reported, never a failed request."""
    # [LAW:nothing-unseen] the usage is read, not parsed: what the API left out cannot fail the summary it paid for.
    facts = {name: reduce(lambda part, field: getattr(part, field, None), path, usage) for name, path in OPENAI_USAGE.items()}
    return {name: value for name, value in facts.items() if value is not None}
