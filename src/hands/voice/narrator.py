"""Sessions' stories, heard: each turn a session finishes is summarised once, for the model to say in its own words, and
each session gone is said after its last turn.

The model is the one that says a turn, not a summariser beside it, so what the user heard is in the model's history and
it can answer about it. The reply is the session's own account: its author knows what "PR 68" is, and a summariser
reading a trimmed transcript does not.

[LAW:one-source-of-truth] every telling of a finished turn is the one summary `speech.told` makes, however it reaches the
user: handed to the model as the turn finishes, as finished turns are set to be told or the session is watched, or held
in `Recounts` until the user asks for it (tell_turn), as a muted session's always is. Nothing of a turn is said as written past the model.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

from loguru import logger
from pipecat.frames.frames import Frame

from hands.core.attention import DEFAULT as DEFAULT_OVERLAY, Amount, Attention, Delivery, EndedRoute, Overlay, Spoken, Withheld, delivery, ended_route
from hands.core.delta import Delta
from hands.core.effects import SessionGone, Summarise
from hands.core.narration import Segment, narration
from hands.core.pending import Finished, News, Unread
from hands.core.session import PromptId, SessionId
from hands.core.subagents import Subagent, reporting
from hands.core.turn import Said
from hands.sessions.delta import Changes, NoChanges
from hands.sessions.focus import focused
from hands.sessions.home import Home
from hands.sessions.overlays import Overlays
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions
from hands.sessions.subagents import read_subagent
from hands.sessions.tail import Tails, Telling
from hands.voice.speech import Unprompted
from hands.voice.utterance import Utterance, Utterances

# Everything reading a turn is expected to fail with; each is said, and the next turn is still heard.
_FAILURES = (Rejected, OSError)


@dataclass(frozen=True)
class Recount:
    """What was told of one turn, each telling of it in the order it was told, whether the last reading of it failed,
    and each part the user asked for more of, once per asking.

    A reading that fails marks nothing told, so the next that succeeds tells what it could not, and clears it. How deep
    a part has been opened is how many times it was asked for, read off `opened` rather than kept beside it
    [LAW:one-source-of-truth], and a new telling starts it over, since there is more of the turn to open.
    """

    turn: PromptId | None
    tellings: tuple[News, ...]
    unread: bool = False
    opened: tuple[str, ...] = ()

    @property
    def parts(self) -> tuple[Segment, ...]:
        return tuple(part for telling in self.tellings for part in telling.parts)


class Recounts:
    """Each session's last finished turn, for the user to ask for: tell_turn hands it to the model.

    One per session, replaced by its next turn's and let go of when the session is gone [LAW:carrying-cost]. A turn told
    again, as one that went on past its Stop is, adds what it did since to what was told of it, so the turn held is the
    whole turn. A turn with no id is never taken for the one held.
    """

    def __init__(self) -> None:
        self._last: dict[SessionId, Recount] = {}

    def put(self, session: SessionId, turn: PromptId | None, told: News | None) -> None:
        """`told` is None when the telling found nothing new: the turn held stays as it is, and another replaces it."""
        held = self._held(session, turn)
        self._last[session] = replace(held, unread=False) if told is None else Recount(turn, (*held.tellings, told))

    def open(self, session: SessionId, part: str) -> int:
        """The user asked for more of `part` of the session's held turn: how many times they asked for it before."""
        held = self._last[session]
        self._last[session] = replace(held, opened=(*held.opened, part))
        return held.opened.count(part)

    def unread(self, session: SessionId, turn: PromptId | None) -> None:
        self._last[session] = replace(self._held(session, turn), unread=True)

    def _held(self, session: SessionId, turn: PromptId | None) -> Recount:
        """The recount held for `turn`, or a fresh one where the turn held is another."""
        held = self._last.get(session)
        return held if held is not None and turn is not None and held.turn == turn else Recount(turn, ())

    def of(self, session: SessionId) -> Recount | None:
        return self._last.get(session)

    def gone(self, session: SessionId) -> None:
        self._last.pop(session, None)


async def narrate(
    sessions: Sessions,
    utterances: Utterances,
    tails: Tails,
    queue_frame: Callable[[Frame], Awaitable[None]],
    aloud: Callable[[], Attention],
    overlays: Overlays,
    recounts: Recounts,
    changes: Changes | None = None,
) -> None:
    """Summarise each finished turn and tell each session gone, in the order they happened, until cancelled.

    `aloud`, what the user set hands to say unprompted, and `overlays` are read at every finished turn and every session
    ending, so a change to either is heard from the next one.
    """
    read = changes or NoChanges()
    while True:
        story = await sessions.story()
        utterance = utterances.heard(story.session, story)
        match story:
            case Summarise(session=session, turn=turn, closing=closing):
                delivered = delivery(await set_to(aloud), await _overlay(overlays, session))
                told = await recount(tails, session, turn, closing, utterance, await read.taken(session), delivered, recounts)
            case SessionGone(session=session):
                recounts.gone(session)
                attention = await set_to(aloud)
                route = ended_route(attention)
                # [LAW:nothing-unseen] whether the ending was said, and what was set that decided it.
                utterance.annotate(attention=attention, route=route)
                told = _routed(route, story)
        match told:
            case None:
                utterance.settle("noted")
            case _:
                await queue_frame(Unprompted(told, (utterance,)))


async def recount(
    tails: Tails,
    session: SessionId,
    turn: PromptId | None,
    closing: str | None,
    utterance: Utterance,
    delta: Delta,
    delivered: Delivery,
    recounts: Recounts,
) -> Finished | Unread | None:
    """Summarise what the turn did beyond what was told before, and what the floor hands the model of it as the turn
    finishes; None when there is nothing new, or when it is held until the user asks.

    `utterance` is the turn's, which carries what was told of it and how it was delivered.
    `delta` is what the repository says the turn did, read when it stopped. A turn with no steps left to tell
    but a repository that moved is still worth telling: that is a formatter or a code generator, and naming
    what it changed is the whole point of reading git at all.
    """
    utterance.annotate(delivered=delivered)
    try:
        told = await tails.tell(session, turn, closing)
    except _FAILURES as error:
        _unread(session, error)
        utterance.fail(f"the turn could not be read: {type(error).__name__}: {error}")
        recounts.unread(session, turn)
        return _delivered(delivered, lambda _: Unread(session))
    if told is None or not (told.turn.steps or delta):
        logger.info(f"session {session} stopped with no untold turn, so there is nothing to tell")
        recounts.put(session, turn, None)
        return None
    subagents, unread = await _subagents(session, told)
    tree = narration(told.turn, delta, subagents)
    # The last words wherever they fall: a turn interrupted mid-work, or one ending on a dialog, said what it had done
    # before the step that ended it.
    replied = [step.text for step in told.turn.steps if isinstance(step, Said)][-1:]
    news = News(turn, replied[0] if replied else None, tree.facts(), tree.asked(), tree.parts, frozenset(task.id for task in reporting(told.turn)))
    # [LAW:nothing-unseen] what the model is handed to say of the turn: the last thing the session said and what hands adds
    # from its record. `topics` is every part of the narration built and not played, which makes this the one place a
    # developer who cannot see the screen finds what "more on that" has to open, and `questions` what the session waits
    # on an answer to. `opened` is the kind of thing that opened the turn, so a turn the user's own command opened is told
    # apart from one they asked for; `subagents` each one that reported back and was read, and `unread_subagents` each
    # whose transcript could not be.
    utterance.annotate(
        reply=news.reply,
        facts=news.facts,
        topics=tuple(dict.fromkeys(segment.topic.name for segment in (*tree.sections, *tree.settled, *tree.subagents))),
        questions=tuple(question.text for question in tree.questions),
        opened=type(told.turn.opening).__name__,
        subagents=tuple(subagent.id for subagent in subagents),
        unread_subagents=unread,
    )
    # Marked told however it is delivered, so the tail lets the steps go: what is kept of them is the tree's parts, held
    # with the telling until the session's next turn, for the user to open.
    await tails.spoken(told)
    recounts.put(session, turn, news)
    return _delivered(delivered, lambda amount: Finished(session, (news,), amount))


async def _subagents(session: SessionId, told: Telling) -> tuple[tuple[Subagent, ...], tuple[str, ...]]:
    """The subagents that reported back in the turn, each read from its own transcript, and the ids of those that could not be.

    A subagent whose transcript cannot be read is said in the log and on the turn's utterance, and the turn is told
    without it: the parent's own record of it, the call and the report, is still the turn's [LAW:no-silent-failure].
    """
    read: list[Subagent] = []
    unread: list[str] = []
    for task in reporting(told.turn):
        try:
            read.append(await asyncio.to_thread(read_subagent, told.transcript, task))
        except _FAILURES as error:
            logger.error(f"cannot read the work of subagent {task.id} of session {session}, so its turn is told without it: {type(error).__name__}: {error}")
            unread.append(task.id)
    return tuple(read), tuple(unread)


def _delivered[T](delivered: Delivery, told: Callable[[Amount], T]) -> T | None:
    """What is told, as much of it as is set, as the turn finishes; none when it waits to be asked for."""
    match delivered:
        case Spoken(amount=amount):
            return told(amount)
        case Withheld():
            return None


def _routed(route: EndedRoute, gone: SessionGone) -> SessionGone | None:
    match route:
        case "said":
            return gone
        case "note":
            # [LAW:one-source-of-truth] the audit log holds the ending, for catch_up to tell when asked.
            return None


async def set_to(aloud: Callable[[], Attention]) -> Attention:
    """What the user set hands to say unprompted, read off the loop the speaker runs on; the defaults where it cannot be read.

    [LAW:no-silent-failure] a setting that cannot be read is logged as the error it is, which is an audit line, and
    what it decides is decided as the defaults decide it.
    """
    try:
        return await asyncio.to_thread(aloud)
    except (Rejected, OSError) as error:
        logger.error(f"cannot read what hands is set to say unprompted, so this is handled as the defaults handle it, {Attention()}: {error}")
        return Attention()


async def _overlay(overlays: Overlays, session: SessionId) -> Overlay:
    """The session's overlay, read off the loop the speaker runs on; the default where it cannot be read.

    [LAW:no-silent-failure] logged as the error it is, which is an audit line, and the turn is handled as the default's.
    """
    try:
        return await asyncio.to_thread(overlays.of, session)
    except (Rejected, OSError) as error:
        logger.error(f"cannot read the overlay of session {session}, so this turn is handled as a {DEFAULT_OVERLAY} session's: {error}")
        return DEFAULT_OVERLAY


async def attending(home: Home, overlays: Overlays, aloud: Callable[[], Attention], session: SessionId) -> tuple[Attention, bool, Overlay]:
    """What hands is set to say unprompted, whether the session is the focus, and its overlay, each read as its progress
    is relayed: a setting or an overlay that cannot be read is the default, and a focus that cannot be read is no focus,
    each logged as the error it is."""
    focus = await asyncio.to_thread(focused, home)
    return await set_to(aloud), focus == session, await _overlay(overlays, session)


def _unread(session: SessionId, error: Exception) -> None:
    """[LAW:no-silent-failure] a turn whose transcript could not be read is logged with the reason, which is an audit
    line, and said as a system fact is. Nothing is marked told, so a later telling of the same turn tells it whole."""
    logger.error(f"cannot read the turn session {session} finished: {type(error).__name__}: {error}")
