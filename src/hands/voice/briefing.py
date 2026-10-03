"""What the intermediary is told of how the sessions stand, and never what any of them did.

Pointers, not content (docs/architecture.md, 'Push pointers, pull content'): a session's history stays in its
transcript, where read_session pulls it on demand, so what the model is told is the same size after a week of work as
after none.

A model hands reaches only through Pipecat is told by notes in its context: one when hands starts, and one at each
change. The brain is told at the tail of every request its turn makes, composed fresh each time, so it reads how the
sessions stand as the request leaves and no stale note piles up in its history.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence

from pipecat.frames.frames import Frame

from hands.sessions.focus import Focus, Unreadable, focused
from hands.sessions.home import Home
from hands.sessions.registry import Sessions
from hands.core.pending import Briefing
from hands.voice.speech import Pushed, Tailed, Telling, Unprompted
from hands.voice.tools import standing


def briefing(listed: Sequence[Mapping[str, str]], focus: Focus) -> str:
    """The note, from sessions as list_sessions describes them, so the two can never tell the model different things,
    and the focus."""
    # [LAW:one-source-of-truth] the id, name, state, and mode are describe_listing's words, the ones list_sessions returns.
    if not listed:
        return f"[hands] hands has just started, and no Claude Code sessions are running. {_focus_said(listed, focus)} Say nothing about this unless the user asks."
    return (
        f"[hands] hands has just started. The Claude Code sessions running now: {_running(listed)}. {_focus_said(listed, focus)} "
        "That is how they stood at the start, and they change; list_sessions says how they stand now. "
        "Say nothing about this unless the user asks."
    )


def tail(listed: Sequence[Mapping[str, str]], focus: Focus) -> str:
    """How the sessions stand as a request leaves, in list_sessions' words, and the focus: what hands appends to its
    newest message."""
    if not listed:
        return f"[hands] No Claude Code sessions are running now. {_focus_said(listed, focus)} Say nothing about this unless the user asks."
    return (
        f"[hands] The Claude Code sessions running now: {_running(listed)}. {_focus_said(listed, focus)} "
        "That is how they stand as this message is sent. Say nothing about this unless the user asks."
    )


def as_sent(sessions: Sessions, home: Home) -> str:
    """The tail as a request leaves: the sessions and the focus, both read as it is composed."""
    return tail(standing(sessions), focused(home))


def _focus_said(listed: Sequence[Mapping[str, str]], focus: Focus) -> str:
    """The focus in a sentence: the session the user's words go to when they name none."""
    match focus:
        case None:
            return "No session is focused."
        case Unreadable(reason):
            # [LAW:no-silent-failure] the model is told the focus is unknown, never that there is none.
            return f"Which session is focused cannot be read: {reason}."
        case session:
            match [listing["name"] for listing in listed if listing["id"] == session]:
                case [name]:
                    return f'The focused session, the one the user\'s words go to when they name none, is "{name}" (id {session}).'
                case _:
                    return f"The focused session (id {session}) is not running now."


def _running(listed: Sequence[Mapping[str, str]]) -> str:
    return "; ".join(f'"{session["name"]}" (id {session["id"]}), {session["state"]}, permission mode: {session["mode"]}' for session in listed)


async def brief(sessions: Sessions, home: Home, telling: Telling, queue_frame: Callable[[Frame], Awaitable[None]]) -> None:
    """Queue the note with the model left unrun: it knows, and says nothing. A tailed model is queued nothing.

    [LAW:no-ambient-temporal-coupling] awaited before the relay and the narrator start, so the note is the first
    thing in the model's context and every change it hears of came after what the note says.
    """
    for note in _notes(telling, sessions, home):
        await queue_frame(Unprompted(Briefing(note)))


def _notes(telling: Telling, sessions: Sessions, home: Home) -> Sequence[str]:
    match telling:
        case Pushed():
            return (briefing(standing(sessions), focused(home)),)
        case Tailed():
            return ()
