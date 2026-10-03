"""Pending speech: what hands has to tell of the sessions and has not started, and the order it is told in.

Nothing here is a frame. The floor holds these values while the user's turn is open and makes frames of them only as it
lets them go, so the order they are told in and what they fold into are decided over values, by `coalesce`, which a
table test can hold to without a pipeline [LAW:effects-at-boundaries].
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from hands.core.effects import DeadlineNear, Expired, Narrate, Note, SessionGone, Speak
from hands.core.narration import Segment
from hands.core.session import Held, PromptId, SessionId


@dataclass(frozen=True)
class News:
    """One telling of a finished turn: which turn, the last thing the session said, what hands adds from its record, what
    it is waiting on the user to answer, and the parts of the narration tree it was cut from, each naming its records.

    A turn that went on past its Stop is told again, as a second telling of the same turn; a turn with no id is never
    taken for another."""

    turn: PromptId | None
    reply: str | None
    facts: str
    asked: str
    parts: tuple[Segment, ...]


def went_on(before: News, after: News) -> bool:
    """Whether `after` tells more of the turn `before` told, rather than the turn after it."""
    return after.turn is not None and after.turn == before.turn


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
    finished turns folded into one telling, then soonest first, in arrival order within a priority.

    A session's story is told in the order it happened: what it said before something sooner is told with that sooner
    thing, never after it, so its next turn's request is not heard ahead of the turn that came before it. What is only
    known is apart from any story: it is never spoken, so it goes first without telling anything out of order.

    `held` is each session's dialog that still waits on an answer, read as the floor lets go: a request answered at
    the keyboard while the user talked is no longer one, and neither is a deadline counted down on it.
    """
    told = _folded([each for each in pending if _waits(each, held)])
    stories = [_story(each, at) for at, each in enumerate(told)]
    # Walked from the last: each thing is told as soon as the soonest thing its story tells after it.
    soonest: dict[SessionId | int, int] = {}
    ranks = [0] * len(told)
    for at in reversed(range(len(told))):
        ranks[at] = soonest[stories[at]] = min(_SOONEST.index(priority(told[at])), soonest.get(stories[at], len(_SOONEST)))
    # [LAW:dataflow-not-control-flow] sorted is stable, so arrival order holds within a priority with no second key.
    return tuple(each for _, each in sorted(zip(ranks, told, strict=True), key=lambda ranked: ranked[0]))


def _waits(pending: Pending, held: Mapping[SessionId, Held]) -> bool:
    """Whether what `pending` tells is still so as it is told."""
    match pending:
        case Narrate(moment=moment):
            return (dialog := held.get(moment.session)) is not None and dialog.request == moment.request
        case Speak(announcement=DeadlineNear(session=session, request=request)):
            return (dialog := held.get(session)) is not None and dialog.request == request
        case _:
            return True


def _story(pending: Pending, at: int) -> SessionId | int:
    """Whose story `pending` is told in: its session's, or, for what is only known, its own, by where it stands."""
    match pending:
        case Speak(announcement=DeadlineNear(session=session) | Expired(session=session)):
            return session
        case Narrate(moment=moment):
            return moment.session
        case Finished(session=session) | Unread(session=session) | SessionGone(session=session):
            return session
        case Note() | Briefing():
            return at


def _folded(pending: Sequence[Pending]) -> list[Pending]:
    """Each session's finished turns as one Finished, where the first of them stood: three Stops heard during one held
    key are one telling whose headline covers them all, and every telling in it keeps the parts it was cut from.

    A fold covers the turns between the other things the session's story tells: a turn that could not be read stands
    between the turns before and after it, as it happened.
    """
    news: dict[tuple[SessionId | int, int], tuple[News, ...]] = {}
    # A dict keeps a key where it was first put, so a fold's slot stays where its first turn stood as later ones join it.
    slots: dict[tuple[SessionId | int, int], Pending] = {}
    # How many things other than a finished turn each story has told so far: a fold is the turns between two of them.
    between: dict[SessionId | int, int] = {}
    for at, each in enumerate(pending):
        story = _story(each, at)
        match each:
            case Finished(session=session):
                slot = (story, between.get(story, 0))
                news[slot] = (*news.get(slot, ()), *each.news)
                slots[slot] = Finished(session, news[slot])
            case _:
                between[story] = between.get(story, 0) + 1
                slots[(at, -1)] = each
    return list(slots.values())
