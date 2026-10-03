"""What the intermediary is told of how the sessions stand, and never what any of them did.

Pointers, not content (docs/architecture.md, 'Push pointers, pull content'): a session's history stays in its
transcript, where read_session pulls it on demand, so what the model is told is the same size after a week of work as
after none.

A model hands reaches only through Pipecat is told by notes in its context: one when hands starts, and one at each
change. The brain is told at the tail of every request its turn makes, composed fresh each time, so it reads how the
sessions stand, and what hands itself said to the user lately, as the request leaves and no stale note piles up in its
history.
"""

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence

from pipecat.frames.frames import Frame, LLMMessagesAppendFrame

from hands.sessions.registry import Sessions
from hands.voice.speech import Pushed, Tailed, Telling, bounded
from hands.voice.tools import standing


def briefing(listed: Sequence[Mapping[str, str]]) -> str:
    """The note, from sessions as list_sessions describes them, so the two can never tell the model different things."""
    # [LAW:one-source-of-truth] the id, name, state, and mode are describe_listing's words, the ones list_sessions returns.
    if not listed:
        return "[hands] hands has just started, and no Claude Code sessions are running. Say nothing about this unless the user asks."
    return (
        f"[hands] hands has just started. The Claude Code sessions running now: {_running(listed)}. "
        "That is how they stood at the start, and they change; list_sessions says how they stand now. "
        "Say nothing about this unless the user asks."
    )


# How much of one line the brain is reminded of: a session's question is read out whole, however long, and the brain
# can read the rest with read_session.
HEARD_CHARS = 400


def tail(listed: Sequence[Mapping[str, str]], heard: Sequence[str] = ()) -> str:
    """How the sessions stand as a request leaves, in list_sessions' words, and the last lines hands said to the user
    in its own words: what hands appends to its newest message."""
    sessions = (
        "[hands] No Claude Code sessions are running now."
        if not listed
        else f"[hands] The Claude Code sessions running now: {_running(listed)}. That is how they stand as this message is sent."
    )
    # [LAW:one-source-of-truth] what the user heard hands say is the speaker's ledger, read as the request leaves; the
    # brain's own words are its history's and are not repeated to it.
    # Each line quoted as JSON, so a question with quotes in it reads as one line; and cut, since every request carries it.
    lines = "; ".join(json.dumps(bounded(line, HEARD_CHARS), ensure_ascii=False) for line in heard)
    said = f" Lately hands said to the user, oldest first: {lines}. What the user says may answer one of these; when it does, act on it." if heard else ""
    # The sessions are said nothing of unless asked; what hands said is there to be answered, so it comes after.
    return f"{sessions} Say nothing about this unless the user asks.{said}"


def _running(listed: Sequence[Mapping[str, str]]) -> str:
    return "; ".join(f'"{session["name"]}" (id {session["id"]}), {session["state"]}, permission mode: {session["mode"]}' for session in listed)


async def brief(sessions: Sessions, telling: Telling, queue_frame: Callable[[Frame], Awaitable[None]]) -> None:
    """Queue the note with the model left unrun: it knows, and says nothing. A tailed model is queued nothing.

    [LAW:no-ambient-temporal-coupling] awaited before the relay and the narrator start, so the note is the first
    thing in the model's context and every change it hears of came after what the note says.
    """
    for note in _notes(telling, sessions):
        await queue_frame(LLMMessagesAppendFrame([{"role": "user", "content": note}], run_llm=False))


def _notes(telling: Telling, sessions: Sessions) -> Sequence[str]:
    match telling:
        case Pushed():
            return (briefing(standing(sessions)),)
        case Tailed():
            return ()
