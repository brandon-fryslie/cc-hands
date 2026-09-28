"""What a model's API answered when it refused a call, read for the one case a listener can act on: a spent usage limit.

[LAW:single-enforcer] the pipeline's own model calls and the summariser's reach the same API, so both read a refusal
here. The API's text is read and never said: what is said comes from a closed set, so it stays short, and a burst of the
same refusal stays one burst.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime

import anthropic
import openai

# Anthropic's words for an account that has spent its limit (2026-09-27), and the instant it names.
_USAGE_LIMIT = re.compile(r"usage limit", re.IGNORECASE)
_RETURNS = re.compile(r"regain access on (\d{4}-\d{2}-\d{2}) at (\d{2}:\d{2}) UTC")


@dataclass(frozen=True)
class UsageLimitReached:
    """The model's API account has reached its usage limit; `returns` is when access comes back, if the API said."""

    returns: datetime | None


def usage_limit(exception: BaseException | None) -> UsageLimitReached | None:
    """The spent usage limit this refusal reports, streamed or not; None for any other failure."""
    match _message(exception):
        case str() as message if _USAGE_LIMIT.search(message):
            return UsageLimitReached(_returns(message))
        case _:
            return None


def reached(limit: UsageLimitReached) -> str:
    """The limit as a clause, with when it lifts in the listener's own time: "the language model's usage limit is reached, until ..."."""
    match limit.returns:
        case None:
            return "the language model's usage limit is reached"
        case datetime() as returns:
            local = returns.astimezone()
            return f"the language model's usage limit is reached, until {local:%B} {local.day} at {local:%-I:%M %p}"


def _message(exception: BaseException | None) -> str | None:
    """The message an API put in the body of its error: Anthropic nests it under `error`, and the OpenAI SDK hands over
    that inner object as the body."""
    match exception:
        case anthropic.APIError(body={"error": {"message": str() as message}}):
            return message
        case openai.APIError(body={"message": str() as message}):
            return message
        case _:
            return None


def _returns(message: str) -> datetime | None:
    match _RETURNS.search(message):
        case re.Match() as found:
            return datetime.fromisoformat(f"{found[1]}T{found[2]}").replace(tzinfo=UTC)
        case None:
            return None
