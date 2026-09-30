"""What a model's API answered when it refused a call, read for the one case a listener can act on: a spent usage limit.

[LAW:single-enforcer] the pipeline's own model calls and the summariser's reach the same API, so both read a refusal
here and hand what they found to the system channel, which says it. The API's text is read and never said.
"""

import re
from datetime import UTC, datetime

import anthropic
import openai

from hands.core.wire import Seconds, UsageLimitReached

# Anthropic's words for an account that has spent its limit (2026-09-27), and the instant it names. Matched exactly: a
# throttle that mentions a per-minute usage limit lifts in seconds and is not this.
_USAGE_LIMIT = re.compile(r"You have reached your specified API usage limits")
_RETURNS = re.compile(r"regain access on (\d{4}-\d{2}-\d{2}) at (\d{2}:\d{2}) UTC")


def usage_limit(exception: BaseException | None) -> UsageLimitReached | None:
    """The spent usage limit this refusal reports, streamed or not; None for any other failure."""
    match _message(exception):
        case str() as message if _USAGE_LIMIT.search(message):
            return UsageLimitReached(_returns(message))
        case _:
            return None


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


def _returns(message: str) -> Seconds | None:
    match _RETURNS.search(message):
        case re.Match() as found:
            return datetime.fromisoformat(f"{found[1]}T{found[2]}").replace(tzinfo=UTC).timestamp()
        case None:
            return None
