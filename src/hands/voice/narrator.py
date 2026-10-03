"""Sessions' stories, heard: each turn a session finishes is summarised once, for the model to say in its own words, and
each session gone is said after its last turn.

The model is the one that says a turn, not a summariser beside it, so what the user heard is in the model's history and
it can answer about it. The reply is the session's own account: its author knows what "PR 68" is, and a summariser
reading a trimmed transcript does not.

[LAW:one-source-of-truth] every telling of a finished turn is the one summary `speech.told` makes, however it reaches the
user: handed to the model as the turn finishes, with spoken summaries on or the session watched, or held in `Recounts`
until the user asks for it (tell_turn), as a muted session's always is. Nothing of a turn is said as written past the model.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

from loguru import logger
from pipecat.frames.frames import Frame

from hands.core.attention import DEFAULT as DEFAULT_OVERLAY, Delivery, Overlay
from hands.core.delta import Delta
from hands.core.effects import SessionGone, Summarise
from hands.core.narration import Segment, narration
from hands.core.pending import Finished, News, Unread
from hands.core.session import PromptId, SessionId
from hands.core.turn import Said
from hands.sessions.audit import Recounted, Record
from hands.sessions.delta import Changes, NoChanges
from hands.sessions.overlays import Overlays
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions
from hands.sessions.summaries import DEFAULT, Summaries
from hands.sessions.tail import Tails
from hands.voice.speech import Unprompted

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


def delivery(switch: Summaries, overlay: Overlay) -> Delivery:
    """How a finished turn reaches the user: a muted session's only when the user asks for it; otherwise every session's
    as it finishes with summaries on, a watched one's as it finishes with them off, and any other's when asked for."""
    # [LAW:dataflow-not-control-flow] a table over the two settings, every pair of them a row the type checker holds to.
    match switch, overlay:
        case _, "muted":
            return "muted"
        case "on", _:
            return "summaries"
        case "off", "watched":
            return "watched"
        case "off", "normal":
            return "on request"


async def narrate(
    sessions: Sessions,
    tails: Tails,
    queue_frame: Callable[[Frame], Awaitable[None]],
    record: Record,
    aloud: Callable[[], Summaries],
    overlays: Overlays,
    recounts: Recounts,
    changes: Changes | None = None,
) -> None:
    """Summarise each finished turn and tell each session gone, in the order they happened, until cancelled.

    `aloud` and `overlays` are read at every finished turn, so a change to either is heard from the next one.
    """
    read = changes or NoChanges()
    while True:
        story = await sessions.story()
        match story:
            case Summarise(session=session, turn=turn, closing=closing):
                delivered = delivery(await switch(aloud), await _overlay(overlays, session))
                told = await recount(tails, session, turn, closing, record, await read.taken(session), delivered, recounts)
            case SessionGone(session=session):
                recounts.gone(session)
                told = story
        if told is not None:
            await queue_frame(Unprompted(told))


async def recount(
    tails: Tails,
    session: SessionId,
    turn: PromptId | None,
    closing: str | None,
    record: Record,
    delta: Delta,
    delivered: Delivery,
    recounts: Recounts,
) -> Finished | Unread | None:
    """Summarise what the turn did beyond what was told before, and what the floor hands the model of it as the turn
    finishes; None when there is nothing new, or when it is held until the user asks.

    `delta` is what the repository says the turn did, read when it stopped. A turn with no steps left to tell
    but a repository that moved is still worth telling: that is a formatter or a code generator, and naming
    what it changed is the whole point of reading git at all.
    """
    try:
        told = await tails.tell(session, turn, closing)
    except _FAILURES as error:
        _unread(session, error)
        recounts.unread(session, turn)
        return _delivered(delivered, Unread(session))
    if told is None or not (told.turn.steps or delta):
        logger.info(f"session {session} stopped with no untold turn, so there is nothing to tell")
        recounts.put(session, turn, None)
        return None
    tree = narration(told.turn, delta)
    # The last words wherever they fall: a turn interrupted mid-work, or one ending on a dialog, said what it had done
    # before the step that ended it.
    replied = [step.text for step in told.turn.steps if isinstance(step, Said)][-1:]
    news = News(replied[0] if replied else None, tree.facts(), tree.asked(), tree.parts)
    record(
        Recounted(
            session,
            news.reply,
            news.facts,
            tuple(dict.fromkeys(segment.topic.name for segment in (*tree.sections, *tree.settled))),
            tuple(question.text for question in tree.questions),
            delivered,
            opened=type(told.turn.opening).__name__,
        )
    )
    # Marked told however it is delivered, so the tail lets the steps go: what is kept of them is the tree's parts, held
    # with the telling until the session's next turn, for the user to open.
    await tails.spoken(told)
    recounts.put(session, turn, news)
    return _delivered(delivered, Finished(session, (news,)))


def _delivered[T](delivered: Delivery, told: T) -> T | None:
    """What is told, as the turn finishes with summaries on or the session watched; none when it waits to be asked for."""
    match delivered:
        case "summaries" | "watched":
            return told
        case "on request" | "muted":
            return None


async def switch(aloud: Callable[[], Summaries]) -> Summaries:
    """Where the summaries switch stands, read off the loop the speaker runs on; the default where it cannot be read.

    [LAW:no-silent-failure] a switch that cannot be read is logged as the error it is, which is an audit line, and the
    turn is handled as the default handles it.
    """
    try:
        return await asyncio.to_thread(aloud)
    except (Rejected, OSError) as error:
        logger.error(f"cannot read whether spoken summaries are on, so this turn is handled as they are by default, {DEFAULT}: {error}")
        return DEFAULT


async def _overlay(overlays: Overlays, session: SessionId) -> Overlay:
    """The session's overlay, read off the loop the speaker runs on; the default where it cannot be read.

    [LAW:no-silent-failure] logged as the error it is, which is an audit line, and the turn is handled as the default's.
    """
    try:
        return await asyncio.to_thread(overlays.of, session)
    except (Rejected, OSError) as error:
        logger.error(f"cannot read the overlay of session {session}, so this turn is handled as a {DEFAULT_OVERLAY} session's: {error}")
        return DEFAULT_OVERLAY


def _unread(session: SessionId, error: Exception) -> None:
    """[LAW:no-silent-failure] a turn whose transcript could not be read is logged with the reason, which is an audit
    line, and said as a system fact is. Nothing is marked told, so a later telling of the same turn tells it whole."""
    logger.error(f"cannot read the turn session {session} finished: {type(error).__name__}: {error}")
