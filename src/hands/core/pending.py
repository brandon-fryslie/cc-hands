"""Pending speech: what hands has to tell of the sessions and has not started, and the order it is told in.

Nothing here is a frame. The floor holds these values while the user's turn is open and makes frames of them only as it
lets them go, so the order they are told in and what they fold into are decided over values, by `coalesce`, which a
table test can hold to without a pipeline [LAW:effects-at-boundaries].
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from hands.core.attention import Amount
from hands.core.effects import DeadlineNear, Expired, Narrate, Note, SessionGone, Speak
from hands.core.narration import Segment
from hands.core.occurrences import Occurrence
from hands.core.progress import Doing
from hands.core.session import Held, Opened, PromptId, RequestId, Session, SessionId, ids
from hands.core.turn import AgentId, AgentTask


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
    # The subagents whose reports it carries, and with them their work.
    reported: frozenset[AgentId]


def went_on(before: News, after: News) -> bool:
    """Whether `after` tells more of the turn `before` told, rather than the turn after it."""
    return after.turn is not None and after.turn == before.turn


@dataclass(frozen=True)
class Finished:
    """A session finished one or more turns, told together, as much of them as the user set: `coalesce` folds a
    session's pending tellings into one, told as the last of them was set to be."""

    session: SessionId
    news: tuple[News, ...]
    amount: Amount


@dataclass(frozen=True)
class Unread:
    """A session finished a turn whose transcript could not be read: said as a system fact is, with no model."""

    session: SessionId


@dataclass(frozen=True)
class Working:
    """What a session is doing as it works, said as it happens: the focused session's progress."""

    session: SessionId
    # Whose doing it is: the turn's, by every id the turn goes by, or a subagent's.
    of: frozenset[PromptId] | AgentTask
    doings: tuple[Doing, ...]


@dataclass(frozen=True)
class Mentioned:
    """Something a session's hook said happened, said as written, as much of it as its kind is set to say."""

    session: SessionId
    occurrence: Occurrence
    amount: Amount


Pending = Speak | Narrate | Note | Finished | Unread | SessionGone | Working | Mentioned

# How soon a pending thing is told, soonest first. "known" goes into the model's context and is never spoken, so it
# costs the user nothing to have it first, and what is spoken after it is said knowing it. "blocking" is something a
# session waits on the user for. "result" is what a session did. "fyi" is the rest.
Priority = Literal["known", "blocking", "result", "fyi"]
_SOONEST: Sequence[Priority] = ("known", "blocking", "result", "fyi")


def priority(pending: Pending) -> Priority:
    # [LAW:one-source-of-truth] read off the variant, never stored beside it.
    match pending:
        case Note():
            return "known"
        case Narrate() | Speak():
            return "blocking"
        case Finished() | Unread():
            return "result"
        case SessionGone() | Working() | Mentioned():
            return "fyi"


@dataclass(frozen=True)
class Coalesced:
    """One thing told: what is said, and where in what was pending each thing it tells stood, more than one where a
    session's turns or its progress were folded into it."""

    pending: Pending
    sources: tuple[int, ...]


def coalesce(pending: Sequence[Pending], live: Mapping[SessionId, Session]) -> tuple[Coalesced, ...]:
    """What is told of `pending`, in the order it is told: what no longer waits on the user dropped, each session's
    finished turns folded into one telling, then soonest first, in arrival order within a priority. What was dropped is
    each thing whose place no telling names.

    A session's story is told in the order it happened: what it said before something sooner is told with that sooner
    thing, never after it, so its next turn's request is not heard ahead of the turn that came before it. What is only
    known is apart from any story: it is never spoken, so it goes first without telling anything out of order.

    `live` is each session that has not ended, read as the floor lets go: a request answered at the keyboard while the
    user talked is no longer one, and neither is a deadline counted down on it; and progress of a turn that ended
    meanwhile is out of date, however the ending was told, or whether it was told at all.
    """
    told = _folded(_current([Coalesced(each, (at,)) for at, each in enumerate(pending) if _waits(each, live)]))
    stories = [_story(each.pending, at) for at, each in enumerate(told)]
    # Walked from the last: each thing is told as soon as the soonest thing its story tells after it.
    soonest: dict[SessionId | int, int] = {}
    ranks = [0] * len(told)
    for at in reversed(range(len(told))):
        ranks[at] = soonest[stories[at]] = min(_SOONEST.index(priority(told[at].pending)), soonest.get(stories[at], len(_SOONEST)))
    # [LAW:dataflow-not-control-flow] sorted is stable, so arrival order holds within a priority with no second key.
    return tuple(each for _, each in sorted(zip(ranks, told, strict=True), key=lambda ranked: ranked[0]))


def _waits(pending: Pending, live: Mapping[SessionId, Session]) -> bool:
    """Whether what `pending` tells is still so as it is told."""
    match pending:
        case Narrate(moment=moment):
            return _asks(live.get(moment.session), moment.request)
        case Speak(announcement=DeadlineNear(session=session, request=request)):
            return _asks(live.get(session), request)
        case Working(session=session, of=of):
            return current(live.get(session), of)
        case _:
            return True


def _asks(session: Session | None, request: RequestId) -> bool:
    """Whether the session's dialog still waits on an answer to `request`."""
    match session:
        case Session(dialog=Held() as dialog):
            return dialog.request == request
        case _:
            return False


def current(session: Session | None, of: frozenset[PromptId] | AgentTask) -> bool:
    """Whether the session still runs the turn that goes by any of these ids; for a subagent's work, whether the session
    is still live, since the turn its work is told with may not have opened yet."""
    match session, of:
        case Session(turn=Opened() as opened), frozenset() as turn:
            return not ids(opened).isdisjoint(turn)
        case Session(), AgentTask():
            return True
        case _:
            return False


def _story(pending: Pending, at: int) -> SessionId | int:
    """Whose story `pending` is told in: its session's, or, for what is only known, its own, by where it stands."""
    match pending:
        case Speak(announcement=DeadlineNear(session=session) | Expired(session=session)):
            return session
        case Narrate(moment=moment):
            return moment.session
        case Finished(session=session) | Unread(session=session) | SessionGone(session=session) | Working(session=session) | Mentioned(session=session):
            return session
        case Note():
            return at


def _current(pending: Sequence[Coalesced]) -> list[Coalesced]:
    """What a session was doing, dropped where its story tells how that turn ended: the result says it better, and
    progress heard after it would be heard out of date. Progress of the turn after it is news, wherever the result of
    the turn before stands: a result is told once it is summarised, and the next turn's calls do not wait on that."""
    ends = [(at, each.pending) for at, each in enumerate(pending) if isinstance(each.pending, Finished | Unread | SessionGone)]
    return [each for at, each in enumerate(pending) if not (isinstance(each.pending, Working) and any(_ends(end, each.pending, at < where) for where, end in ends))]


def _ends(end: Finished | Unread | SessionGone, progress: Working, before: bool) -> bool:
    """Whether `end` tells how the turn `progress` was made in went: a result of that turn, or the session gone; and a
    turn that could not be read, which goes by no id, if it is told after the progress came. A subagent's work is told
    with the turn it reports back to, wherever that stands: a burst that settled after the report was read is the work
    the report tells. Any other result leaves it news, as it is of a subagent working on in the background."""
    match end, progress.of:
        case Finished(session=session, news=news), frozenset() as turn:
            return session == progress.session and any(each.turn in turn for each in news)
        case Finished(session=session, news=news), AgentTask(id=agent):
            return session == progress.session and any(agent in each.reported for each in news)
        case SessionGone(session=session), _:
            return session == progress.session
        case Unread(session=session), _:
            return session == progress.session and before


def _folded(pending: Sequence[Coalesced]) -> list[Coalesced]:
    """Each session's finished turns as one Finished, and its progress as one Working, where the first of them stood:
    three Stops heard during one held key are one telling whose headline covers them all, and every telling in it keeps
    the parts it was cut from; ten edits are one sentence.

    A fold covers what came between the other things the session's story tells: a turn that could not be read stands
    between the turns before and after it, as it happened.
    """
    # A dict keeps a key where it was first put, so a fold's slot stays where its first thing stood as later ones join it.
    slots: dict[tuple[object, ...], Coalesced] = {}
    # How many things that do not fold each story has told so far: a fold is what came between two of them.
    between: dict[SessionId | int, int] = {}
    for at, each in enumerate(pending):
        story = _story(each.pending, at)
        match each.pending:
            case Finished() | Working() as folding:
                # A subagent's work folds only with its own: each is said as the work of the call that started it.
                slot = (type(folding), story, between.get(story, 0), folding.of if isinstance(folding, Working) and isinstance(folding.of, AgentTask) else None)
                before = slots.get(slot)
                slots[slot] = each if before is None else Coalesced(_joined(before.pending, folding), (*before.sources, *each.sources))
            case _:
                between[story] = between.get(story, 0) + 1
                slots[(at,)] = each
    return list(slots.values())


def _joined(before: Pending, each: Finished | Working) -> Pending:
    """`each` folded into the telling of its kind that came before it in its slot."""
    match before, each:
        case Finished(news=earlier), Finished(session=session, news=news, amount=amount):
            return Finished(session, (*earlier, *news), amount)
        case Working(of=frozenset() as was, doings=earlier), Working(session=session, of=frozenset() as turn, doings=doings):
            return Working(session, was | turn, (*earlier, *doings))
        case Working(of=AgentTask() as agent, doings=earlier), Working(session=session, doings=doings):
            return Working(session, agent, (*earlier, *doings))
        case _:
            return each
