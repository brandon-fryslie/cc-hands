"""What the intermediary is told of how the sessions stand, and never what any of them did.

Pointers, not content (docs/architecture.md, 'Push pointers, pull content'): a session's history stays in its
transcript, where read_session pulls it on demand, so what the model is told is the same size after a week of work as
after none.

A model hands reaches only through Pipecat is told by notes in its context: one when hands starts, and one at each
change. The brain is told at the tail of every request its turn makes, composed fresh each time, so it reads how the
sessions stand as the request leaves and no stale note piles up in its history.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence

from loguru import logger
from pipecat.frames.frames import Frame, LLMMessagesAppendFrame

from hands.sessions.focus import focused
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions
from hands.voice.speech import Pushed, Tailed, Telling
from hands.voice.tools import standing


# The focus, said when the user has focused no session.
UNFOCUSED = "No session is focused."


def briefing(listed: Sequence[Mapping[str, str]], focus: str) -> str:
    """The note, from sessions as list_sessions describes them, so the two can never tell the model different things,
    and the focus as `focus_said` says it."""
    # [LAW:one-source-of-truth] the id, name, state, and mode are describe_listing's words, the ones list_sessions returns.
    if not listed:
        return f"[hands] hands has just started, and no Claude Code sessions are running. {focus} Say nothing about this unless the user asks."
    return (
        f"[hands] hands has just started. The Claude Code sessions running now: {_running(listed)}. {focus} "
        "That is how they stood at the start, and they change; list_sessions says how they stand now. "
        "Say nothing about this unless the user asks."
    )


def tail(listed: Sequence[Mapping[str, str]], focus: str) -> str:
    """How the sessions stand as a request leaves, in list_sessions' words, and the focus as `focus_said` says it: what
    hands appends to its newest message."""
    if not listed:
        return f"[hands] No Claude Code sessions are running now. {focus} Say nothing about this unless the user asks."
    return (
        f"[hands] The Claude Code sessions running now: {_running(listed)}. {focus} "
        "That is how they stand as this message is sent. Say nothing about this unless the user asks."
    )


def as_sent(sessions: Sessions, home: Home) -> str:
    """The tail as a request leaves: the sessions and the focus, both read as it is composed."""
    listed = standing(sessions)
    return tail(listed, focus_said(listed, home))


def focus_said(listed: Sequence[Mapping[str, str]], home: Home) -> str:
    """The focus, read from the home as this is said, in a sentence: the session the user's words go to when they name none."""
    try:
        focus = focused(home)
    except (Rejected, OSError) as error:
        # [LAW:no-silent-failure] the model is told the focus is unknown, never that there is none.
        logger.error(f"cannot read the focus to tell the brain: {error}")
        return f"Which session is focused cannot be read: {error}."
    if focus is None:
        return UNFOCUSED
    match [session["name"] for session in listed if session["id"] == focus]:
        case [name]:
            return f'The focused session, the one the user\'s words go to when they name none, is "{name}" (id {focus}).'
        case _:
            return f"The focused session (id {focus}) is not running now."


def _running(listed: Sequence[Mapping[str, str]]) -> str:
    return "; ".join(f'"{session["name"]}" (id {session["id"]}), {session["state"]}, permission mode: {session["mode"]}' for session in listed)


async def brief(sessions: Sessions, home: Home, telling: Telling, queue_frame: Callable[[Frame], Awaitable[None]]) -> None:
    """Queue the note with the model left unrun: it knows, and says nothing. A tailed model is queued nothing.

    [LAW:no-ambient-temporal-coupling] awaited before the relay and the narrator start, so the note is the first
    thing in the model's context and every change it hears of came after what the note says.
    """
    for note in _notes(telling, sessions, home):
        await queue_frame(LLMMessagesAppendFrame([{"role": "user", "content": note}], run_llm=False))


def _notes(telling: Telling, sessions: Sessions, home: Home) -> Sequence[str]:
    match telling:
        case Pushed():
            listed = standing(sessions)
            return (briefing(listed, focus_said(listed, home)),)
        case Tailed():
            return ()
