"""The summariser: a side question asked of a Claude Code of hands' own, apart from the brain's conversation."""

from collections.abc import Awaitable, Callable

from hands.brain.asides import AsideFailed, Deadline, Within

# A rendered turn in, the spoken summary out.
Summariser = Callable[[str], Awaitable[str]]


class SummaryFailed(Exception):
    """The model answered, but with nothing that can be spoken, or its side question had no answer."""


def aside(ask: Callable[[str, Within], Awaitable[str]], instruction: str, timeout: float) -> Summariser:
    """The turn asked as a side question, with what to make of it, of a Claude Code that is asked nothing else, within
    `timeout`, whatever it waited on."""

    async def from_an_aside(turn: str) -> str:
        try:
            return _spoken(await ask(f"{instruction}\n\nSummarize this:\n\n{turn}", Deadline(timeout)))
        except AsideFailed as error:
            raise SummaryFailed(f"the side question had no answer: {error}") from error

    return from_an_aside


def _spoken(answer: str) -> str:
    text = answer.strip()
    if not text:
        raise SummaryFailed("the model returned no summary")
    return text
