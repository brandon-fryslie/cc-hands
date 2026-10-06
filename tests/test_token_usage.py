"""The usage a model API reports, as the facts of a `model.request` event."""

from typing import cast

import pytest
from anthropic._models import construct_type as anthropic_type  # pyright: ignore[reportPrivateUsage]
from anthropic.types import Message
from openai._models import construct_type  # pyright: ignore[reportPrivateUsage]
from openai.types import CompletionUsage
from pipecat.metrics.metrics import LLMTokenUsage

from hands.voice.token_usage import OPENAI_USAGE, USAGE, openai_usage, reported


def sent(usage: object) -> CompletionUsage:
    """A usage as the SDK builds it from what the API sent: unchecked, so a count left out or null is None."""
    return construct_type(type_=CompletionUsage, value=usage)  # pyright: ignore[reportReturnType]


def test_its_facts_are_named_as_pipecat_names_a_streamed_usage() -> None:
    assert set(OPENAI_USAGE) <= set(LLMTokenUsage.model_fields)


@pytest.mark.parametrize(
    ("usage", "facts"),
    [
        (None, {}),
        (
            {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
            {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
        ),
        ({"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": None}, {"prompt_tokens": 9, "completion_tokens": 2}),
        ({"prompt_tokens": 9}, {"prompt_tokens": 9}),
        (
            {
                "prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11,
                "prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 5},
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
            {
                "prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11,
                "cache_read_input_tokens": 3, "cache_creation_input_tokens": 5, "reasoning_tokens": 1,
            },
        ),
        ({"prompt_tokens": 9, "prompt_tokens_details": {"cached_tokens": None}}, {"prompt_tokens": 9}),
    ],
)
def test_a_usage_says_each_count_the_api_reported_and_none_it_left_out(usage: dict[str, object] | None, facts: dict[str, int]) -> None:
    assert openai_usage(None if usage is None else sent(usage)) == facts


@pytest.mark.parametrize(
    ("message", "facts"),
    [
        ({"usage": {"input_tokens": 5, "cache_read_input_tokens": 4, "output_tokens": 2}}, {"input_tokens": 5, "cache_read_input_tokens": 4, "output_tokens": 2}),
        ({"usage": {"output_tokens": 2}}, {"output_tokens": 2}),
        ({}, {}),
    ],
)
def test_an_anthropic_usage_says_each_count_the_api_reported_and_none_it_left_out(message: dict[str, object], facts: dict[str, int]) -> None:
    reply = anthropic_type(type_=Message, value={"id": "m1", "type": "message", "role": "assistant", "model": "m", "content": [], **message})
    assert reported(cast(Message, reply).usage, USAGE) == facts
