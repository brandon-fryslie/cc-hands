"""Sessions' stories, heard: each turn a session finishes is summarised once, for the model to say in its own words, and
each session gone is said after its last turn.

The model is the one that says a turn, not a summariser beside it, so what the user heard is in the model's history and
it can answer about it. The reply is the session's own account: its author knows what "PR 68" is, and a summariser
reading a trimmed transcript does not.

[LAW:one-source-of-truth] every telling of a finished turn is the one summary `_news` makes, however it reaches the
user: handed to the model as the turn finishes, with spoken summaries on or the session watched, or held in `Recounts`
until the user asks for it (tell_turn). Nothing of a turn is said as written past the model.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from loguru import logger
from pipecat.frames.frames import Frame, TTSSpeakFrame

from hands.core.attention import DEFAULT as UNWATCHED, Delivery, Overlay
from hands.core.delta import Delta
from hands.core.effects import SessionGone, Summarise
from hands.core.narration import Narration, narration
from hands.core.session import PromptId, SessionId
from hands.core.turn import Said, Turn
from hands.sessions.audit import Recounted, Record
from hands.sessions.delta import Changes, NoChanges
from hands.sessions.overlays import Overlays
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions
from hands.sessions.summaries import DEFAULT, Summaries
from hands.sessions.tail import Tails
from hands.voice.readback import spoken_name
from hands.voice.speech import Telling, as_written, bounded, handed

# How much of a reply is handed to the model. Every turn told grows the model's history toward compaction, so what is
# handed is bounded; a session asked to end on a concise overview writes far less than this.
REPLY_SHOWN = 1500

# Everything reading a turn is expected to fail with; each is said, and the next turn is still heard.
_FAILURES = (Rejected, OSError)


@dataclass(frozen=True)
class Recount:
    """What was told of one turn, each telling of it in the order it was told; none when it finished with nothing to tell."""

    turn: PromptId | None
    tellings: tuple[str, ...]


class Recounts:
    """Each session's last finished turn, for the user to ask for: tell_turn hands it to the model.

    One per session, replaced by its next turn's and let go of when the session is gone [LAW:carrying-cost]. A turn told
    again, as one that went on past its Stop is, adds what it did since to what was told of it, so the turn held is the
    whole turn. A turn with no id is never taken for the one held.
    """

    def __init__(self) -> None:
        self._last: dict[SessionId, Recount] = {}

    def put(self, session: SessionId, turn: PromptId | None, told: str | None) -> None:
        """`told` is None when the telling found nothing new: the turn held stays as it is, and another replaces it."""
        held = self._last.get(session)
        earlier = held.tellings if held is not None and turn is not None and held.turn == turn else ()
        self._last[session] = Recount(turn, (*earlier, *([] if told is None else [told])))

    def of(self, session: SessionId) -> Recount | None:
        return self._last.get(session)

    def gone(self, session: SessionId) -> None:
        self._last.pop(session, None)


def delivery(switch: Summaries, overlay: Overlay) -> Delivery:
    """How a finished turn reaches the user: every session's as it finishes with summaries on, a watched one's as it
    finishes with them off, and any other's when the user asks for it."""
    match switch, overlay:
        case "on", _:
            return "summaries"
        case "off", "watched":
            return "watched"
        case "off", "normal":
            return "on request"


async def narrate(
    sessions: Sessions,
    tails: Tails,
    telling: Telling,
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
        name = spoken_name(sessions, story.session)
        match story:
            case Summarise(session=session, turn=turn, closing=closing):
                delivered = delivery(await _switch(aloud), await _overlay(overlays, session))
                told = await recount(tails, session, turn, closing, name, record, await read.taken(session), delivered, recounts, telling)
            case SessionGone(session=session):
                recounts.gone(session)
                told = as_written(TTSSpeakFrame(f"The session {name} is gone."), telling)
        if told is not None:
            await queue_frame(told)


async def recount(
    tails: Tails,
    session: SessionId,
    turn: PromptId | None,
    closing: str | None,
    name: str,
    record: Record,
    delta: Delta,
    delivered: Delivery,
    recounts: Recounts,
    telling: Telling,
) -> Frame | None:
    """Summarise what the turn did beyond what was told before, and the frame that hands it to the model as the turn
    finishes; None when there is nothing new, or when it is held until the user asks.

    `delta` is what the repository says the turn did, read when it stopped. A turn with no steps left to tell
    but a repository that moved is still worth telling: that is a formatter or a code generator, and naming
    what it changed is the whole point of reading git at all.
    """
    try:
        told = await tails.tell(session, turn, closing)
    except _FAILURES as error:
        unread = _unread(session, name, error)
        recounts.put(session, turn, f"[hands] The Claude Code session {name} finished a turn, and hands could not read it. Tell the user so.")
        return _delivered(delivered, as_written(unread, telling))
    if told is None or not (told.turn.steps or delta):
        logger.info(f"session {session} stopped with no untold turn, so there is nothing to tell")
        recounts.put(session, turn, None)
        return None
    tree = narration(told.turn, delta)
    news = _news(name, told.turn, tree)
    record(
        Recounted(
            session,
            news,
            tuple(dict.fromkeys(segment.topic.name for segment in (*tree.sections, *tree.settled))),
            tuple(question.text for question in tree.questions),
            delivered,
            opened=type(told.turn.opening).__name__,
        )
    )
    # Marked told however it is delivered: the summary holds what the user is told of it, so the steps are let go of.
    await tails.spoken(told)
    recounts.put(session, turn, news)
    return _delivered(delivered, handed(news, f"{name} finished a turn, and I could not tell it.", telling))


def _delivered(delivered: Delivery, frame: Frame) -> Frame | None:
    """The frame, as the turn finishes with summaries on or the session watched; none when it waits to be asked for."""
    match delivered:
        case "summaries" | "watched":
            return frame
        case "on request":
            return None


def _news(name: str, turn: Turn, tree: Narration) -> str:
    """The turn as the model is handed it: the last thing the session said, what hands read of it that those words may
    not say, and what it is waiting on, which the model ends by asking.

    The last words wherever they fall: a turn interrupted mid-work, or one ending on a dialog, said what it had done
    before the step that ended it.
    """
    replied = [step.text for step in turn.steps if isinstance(step, Said)][-1:]
    reply = f"The last thing it said was:\n\n{bounded(replied[0], REPLY_SHOWN)}\n\n" if replied else "It said nothing. "
    read = tree.facts()
    facts = f"From its record, hands adds: {read} " if read else ""
    asked = tree.asked()
    ending = (
        f"It is waiting on the user's answer to this, so end by asking it, with what it refers to, so they can answer without looking at the screen: {asked}"
        if asked
        else "It asks the user nothing."
    )
    return (
        f"[hands] The Claude Code session {name} finished a turn. {reply}{facts}"
        f"Tell the user what it concretely did, in your own words, in one or two spoken sentences, naming the session. {ending}"
    )


async def _switch(aloud: Callable[[], Summaries]) -> Summaries:
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
    """Whether the session is watched, read off the loop the speaker runs on; the default where it cannot be read.

    [LAW:no-silent-failure] logged as the error it is, which is an audit line, and the turn is handled as an unwatched one's.
    """
    try:
        return await asyncio.to_thread(overlays.of, session)
    except (Rejected, OSError) as error:
        logger.error(f"cannot read whether session {session} is watched, so this turn is handled as an unwatched one's: {error}")
        return UNWATCHED


def _unread(session: SessionId, name: str, error: Exception) -> TTSSpeakFrame:
    """What is said of a turn whose transcript could not be read: that, as a system fact is, with no model.

    [LAW:no-silent-failure] logged with the reason, which is an audit line. Nothing is marked told, so a later telling
    of the same turn tells it whole.
    """
    logger.error(f"cannot read the turn session {session} finished: {type(error).__name__}: {error}")
    return TTSSpeakFrame(f"{name} finished a turn, and I could not read it.", append_to_context=False)
