"""Pending speech: what hands has to tell of the sessions and has not started, and the order it is told in.

Nothing here is a frame. The floor holds these values while the user's turn is open and makes frames of them only as it
lets them go, so the order they are told in and what they fold into are decided over values, by `coalesce`, which a
table test can hold to without a pipeline [LAW:effects-at-boundaries].
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from hands.core.effects import DeadlineNear, Narrate, Note, SessionGone, Speak
from hands.core.narration import Segment
from hands.core.session import Held, SessionId


@dataclass(frozen=True)
class News:
    """One telling of a finished turn: the last thing the session said, what hands adds from its record, what it is
    waiting on the user to answer, and the parts of the narration tree it was cut from, each naming its records."""

    reply: str | None
    facts: str
    asked: str
    parts: tuple[Segment, ...]


@dataclass(frozen=True)
class Finished:
    """A session finished one or more turns, told together: `coalesce` folds a session's pending tellings into one."""

    session: SessionId
    news: tuple[News, ...]


@dataclass(frozen=True)
class Unread:
    """A session finished a turn whose transcript could not be read: said as a system fact is, with no model."""

    session: SessionId


@dataclass(frozen=True)
class Briefing:
    """How the sessions stood as hands started, for the model to know before anything else is told."""

    note: str


Pending = Speak | Narrate | Note | Finished | Unread | SessionGone | Briefing

# How soon a pending thing is told, soonest first. "known" goes into the model's context and is never spoken, so it
# costs the user nothing to have it first, and what is spoken after it is said knowing it. "blocking" is something a
# session waits on the user for. "result" is what a session did. "fyi" is the rest.
Priority = Literal["known", "blocking", "result", "fyi"]
_SOONEST: Sequence[Priority] = ("known", "blocking", "result", "fyi")


def priority(pending: Pending) -> Priority:
    # [LAW:one-source-of-truth] read off the variant, never stored beside it.
    match pending:
        case Note() | Briefing():
            return "known"
        case Narrate() | Speak():
            return "blocking"
        case Finished() | Unread():
            return "result"
        case SessionGone():
            return "fyi"


def coalesce(pending: Sequence[Pending], held: Mapping[SessionId, Held]) -> tuple[Pending, ...]:
    """What is told of `pending`, in the order it is told: what no longer waits on the user dropped, each session's
    finished turns folded into one telling where the first of them stood, then soonest first, in arrival order within
    a priority.

    `held` is each session's dialog that still waits on an answer, read as the floor lets go: a request answered at
    the keyboard while the user talked is no longer one, and neither is a deadline counted down on it.
    """
    # [LAW:dataflow-not-control-flow] sorted is stable, so arrival order holds within a priority with no second key.
    return tuple(sorted(_folded(each for each in pending if _waits(each, held)), key=lambda each: _SOONEST.index(priority(each))))


def _waits(pending: Pending, held: Mapping[SessionId, Held]) -> bool:
    """Whether what `pending` tells is still so as it is told."""
    match pending:
        case Narrate(moment=moment):
            return (dialog := held.get(moment.session)) is not None and dialog.request == moment.request
        case Speak(announcement=DeadlineNear(session=session, on=on)):
            return (dialog := held.get(session)) is not None and dialog.on == on
        case _:
            return True


def _folded(pending: Iterable[Pending]) -> list[Pending]:
    """Each session's finished turns as one Finished, where its first stood: three Stops heard during one held key
    are one telling whose headline covers them all, and every telling in it keeps the parts it was cut from."""
    news: dict[SessionId, tuple[News, ...]] = {}
    # A dict keeps a key where it was first put, so a session's slot stays where its first turn stood as later ones join it.
    slots: dict[SessionId | int, Pending] = {}
    for at, each in enumerate(pending):
        match each:
            case Finished(session=session):
                news[session] = (*news.get(session, ()), *each.news)
                slots[session] = Finished(session, news[session])
            case _:
                slots[at] = each
    return list(slots.values())
