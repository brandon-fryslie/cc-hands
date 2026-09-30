"""Sessions' stories, heard: each turn a session finishes is handed to the model with the reply that ended it, and said
in the model's own words, and each session gone is said after its last turn.

The model is the one that says a turn, not a summariser beside it, so what the user heard is in the model's history and
it can answer about it. The reply is the session's own account: its author knows what "PR 68" is, and a summariser
reading a trimmed transcript does not.
"""

import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger
from pipecat.frames.frames import Frame, TTSSpeakFrame

from hands.core.delta import Delta
from hands.core.effects import SessionGone, Summarise
from hands.core.narration import Narration, narration
from hands.core.session import PromptId, SessionId
from hands.core.turn import Interruption, Said, Turn
from hands.sessions.audit import Recounted, Record
from hands.sessions.delta import Changes, NoChanges
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


async def narrate(
    sessions: Sessions,
    tails: Tails,
    telling: Telling,
    queue_frame: Callable[[Frame], Awaitable[None]],
    record: Record,
    aloud: Callable[[], Summaries],
    changes: Changes | None = None,
) -> None:
    """Tell each finished turn and each session gone, in the order they happened, until cancelled.

    `aloud` reads where the summaries switch stands, at every finished turn, so a change is heard from the next one told.
    """
    read = changes or NoChanges()
    while True:
        story = await sessions.story()
        name = spoken_name(sessions, story.session)
        match story:
            case Summarise(session=session, turn=turn, closing=closing):
                switch = await _switch(aloud)
                told = await recount(tails, session, turn, closing, name, record, await read.taken(session), switch, telling)
            case SessionGone():
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
    switch: Summaries,
    telling: Telling,
) -> Frame | None:
    """The frame that tells the user what the turn did beyond what was told before, or None when there is nothing new.

    `delta` is what the repository says the turn did, read when it stopped. A turn with no steps left to tell
    but a repository that moved is still worth telling: that is a formatter or a code generator, and naming
    what it changed is the whole point of reading git at all.

    With summaries off, what the turn is waiting on the user to answer is all that plays, said as written: the switch
    quiets the report, never a question.
    """
    try:
        told = await tails.tell(session, turn, closing)
    except _FAILURES as error:
        return as_written(_unread(session, name, error), telling)
    if told is None or not (told.turn.steps or delta):
        logger.info(f"session {session} stopped with no untold turn, so there is nothing to tell")
        return None
    tree = narration(told.turn, delta)
    # A turn stopped before it did anything has nothing for the model to tell: what the types say of it is all there is.
    did = bool(delta) or any(not isinstance(step, Interruption) for step in told.turn.steps)
    match switch:
        case "on" if did:
            text = _news(name, told.turn, tree)
            # Said as written if the model cannot take it: what the turn is waiting on is the daemon's, and still heard.
            unsaid = " ".join(part for part in (f"{name} finished a turn, and I could not tell it.", tree.asked()) if part)
            frame: Frame | None = handed(text, unsaid, telling)
        case "on":
            text = tree.said()
            frame = as_written(TTSSpeakFrame(f"{name}: {text}"), telling)
        case "off":
            text = tree.asked()
            frame = as_written(TTSSpeakFrame(f"{name}: {text}"), telling) if text else None
            logger.info(f"session {session} finished a turn, and spoken summaries are off, so {'only its question is' if text else 'nothing is'} said")
    record(
        Recounted(
            session,
            text,
            tuple(dict.fromkeys(segment.topic.name for segment in (*tree.sections, *tree.settled))),
            tuple(question.text for question in tree.questions),
            by_model=switch == "on" and did,
        )
    )
    # Marked told either way: a turn the switch kept quiet was heard as much as it will be, and is not told later.
    await tails.spoken(told)
    return frame


def _news(name: str, turn: Turn, tree: Narration) -> str:
    """The turn as the model is handed it: the last thing the session said, what hands read of it that those words may
    not say, and what it is waiting on, which the model ends by asking.

    The last words wherever they fall: a turn interrupted mid-work, or one ending on a dialog, said what it had done
    before the step that ended it.
    """
    replied = [step.text for step in turn.steps if isinstance(step, Said)][-1:]
    reply = f"The last thing it said was:\n\n{bounded(replied[0], REPLY_SHOWN)}\n\n" if replied else "It said nothing. "
    read = " ".join(segment.text for segment in (*tree.interrupted, *tree.repository))
    facts = f"From its record, hands adds: {read} " if read else ""
    asked = tree.asked()
    ending = f"It is waiting on the user's answer to this, so end by asking it: {asked}" if asked else "It asks the user nothing."
    return (
        f"[hands] The Claude Code session {name} finished a turn. {reply}{facts}"
        f"Tell the user what it did, in your own words, in one or two spoken sentences, naming the session. {ending}"
    )


async def _switch(aloud: Callable[[], Summaries]) -> Summaries:
    """Where the summaries switch stands, read off the loop the speaker runs on; the default where it cannot be read.

    [LAW:no-silent-failure] a switch that cannot be read is logged as the error it is, which is an audit line, and the
    turn is still told, as the default tells it: its question is never lost to a file edited by hand.
    """
    try:
        return await asyncio.to_thread(aloud)
    except (Rejected, OSError) as error:
        logger.error(f"cannot read whether spoken summaries are on, so this turn is told as they are by default, {DEFAULT}: {error}")
        return DEFAULT


def _unread(session: SessionId, name: str, error: Exception) -> TTSSpeakFrame:
    """What is said of a turn whose transcript could not be read: that, as a system fact is, with no model.

    [LAW:no-silent-failure] logged with the reason, which is an audit line. Nothing is marked told, so a later telling
    of the same turn tells it whole.
    """
    logger.error(f"cannot read the turn session {session} finished: {type(error).__name__}: {error}")
    return TTSSpeakFrame(f"{name} finished a turn, and I could not read it.", append_to_context=False)
