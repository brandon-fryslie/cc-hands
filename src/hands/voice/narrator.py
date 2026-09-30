"""Sessions' stories, heard: each turn a session finishes is told from the tail, summarised, and spoken with its name, and each session gone is said after its last turn."""

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping

from loguru import logger
from pipecat.frames.frames import Frame, TTSSpeakFrame

from hands.core.delta import Delta
from hands.core.effects import SessionGone, Summarise
from hands.core.narration import narration, shown
from hands.core.session import PromptId, SessionId
from hands.core.sentences import Digest, turn_digest
from hands.core.turn import Answering, Budget, Interruption
from hands.sessions.audit import Recounted, Record
from hands.sessions.delta import Changes, NoChanges
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions
from hands.sessions.summaries import DEFAULT, Summaries
from hands.sessions.tail import Tails
from hands.voice.readback import spoken_name
from hands.voice.refusal import UsageLimitReached, usage_limit
from hands.voice.summary import SUMMARY_FAILURES, Summariser
from hands.voice.summary_instruction import HEADLINE_SENTENCES

# How much of a turn the summariser is shown: enough to name its results, few enough tokens for a local model to answer in seconds.
TURN_BUDGET = Budget(opening=600, said=1500, input=200, result=400, steps=40, files=25, commits=10, changes=2000)

# Where a turn's report goes to be kept as its sentence: the summary store.
Keep = Callable[[Mapping[Digest, str]], None]

# Where a model refusal the listener can act on goes: the system channel, which says it once a burst.
Refused = Callable[[UsageLimitReached], Awaitable[None]]

# Everything reading and summarising a turn is expected to fail with; each is said, and the next turn is still heard.
_FAILURES = (Rejected, *SUMMARY_FAILURES)


async def narrate(
    sessions: Sessions,
    tails: Tails,
    summarise: Summariser,
    queue_frame: Callable[[Frame], Awaitable[None]],
    record: Record,
    aloud: Callable[[], Summaries],
    refused: Refused,
    keep: Keep,
    budget: Budget = TURN_BUDGET,
    changes: Changes | None = None,
) -> None:
    """Speak each finished turn and each session gone, in the order they happened, until cancelled.

    `aloud` reads where the summaries switch stands, at every finished turn, so a change is heard from the next one told.
    """
    read = changes or NoChanges()
    while True:
        story = await sessions.story()
        name = spoken_name(sessions, story.session)
        match story:
            case Summarise(session=session, turn=turn, closing=closing):
                switch = await _switch(aloud)
                spoken = await recount(tails, session, turn, closing, name, summarise, record, budget, await read.taken(session), switch, refused, keep)
            case SessionGone():
                spoken = TTSSpeakFrame(f"The session {name} is gone.")
        if spoken is not None:
            await queue_frame(spoken)


async def recount(
    tails: Tails,
    session: SessionId,
    turn: PromptId | None,
    closing: str | None,
    name: str,
    summarise: Summariser,
    record: Record,
    budget: Budget,
    delta: Delta,
    switch: Summaries,
    refused: Refused,
    keep: Keep,
) -> Frame | None:
    """The frame that tells the user what the turn did beyond what was told before, or None when there is nothing new.

    `delta` is what the repository says the turn did, read when it stopped. A turn with no steps left to tell
    but a repository that moved is still worth telling: that is a formatter or a code generator, and naming
    what it changed is the whole point of reading git at all.

    With summaries off, what the turn is waiting on the user to answer is all that plays, and no model is asked:
    the switch quiets the report, never a question.
    """
    try:
        telling = await tails.tell(session, turn, closing)
    except _FAILURES as error:
        return _unsummarised(session, name, error, "")
    if telling is None or not (telling.turn.steps or delta):
        logger.info(f"session {session} stopped with no untold turn, so there is nothing to tell")
        return None
    # What plays with no model: whatever the turn is waiting on the listener to answer.
    unsaid = narration("", telling.turn, delta, HEADLINE_SENTENCES)
    match switch:
        case "on":
            began = time.monotonic()
            # A turn stopped before it did anything has nothing for a model to report, and a model told not to say it was
            # interrupted would report something anyway: its narration is the interruption alone.
            did = delta or any(not isinstance(step, Interruption) for step in telling.turn.steps)
            try:
                headline = await summarise(shown(telling.turn, delta, budget)) if did else ""
            except _FAILURES as error:
                # A spent usage limit fails every summary until a stated date, so it goes to the system channel, which
                # says it once a burst however many sessions finish a turn inside one.
                match usage_limit(error):
                    case UsageLimitReached() as limit:
                        await refused(limit)
                    case None:
                        pass
                # What the turn is waiting on is the daemon's to find and needs no model, and a question is always said.
                return _unsummarised(session, name, error, unsaid.asked())
            # The number this whole epic turns on, and until now invisible: how long a finished turn waited on the
            # model before it could be spoken at all.
            logger.info(f"session {session} was summarised in {time.monotonic() - began:.2f} s")
            # The tree is cut from the turn after the headline comes back, so the sections cost nothing at the Stop that
            # matters: the counts are arithmetic over steps already read, and only the headline waits on a model.
            told = narration(headline, telling.turn, delta, HEADLINE_SENTENCES)
            spoken = told.said()
            # [LAW:one-source-of-truth] a report of the whole turn is its sentence in the summary store, under the key
            # read_session finds it by, so a turn told aloud is never summarised again. A telling of the rest of a turn
            # reported once already is not the whole turn, and is not kept.
            key = turn_digest((telling.turn.opening, *telling.turn.steps)) if headline and isinstance(telling.turn.standing, Answering) else None
            if key is not None:
                keep({key: headline})
        case "off":
            told, spoken, key = unsaid, unsaid.asked(), None
            logger.info(f"session {session} finished a turn, and spoken summaries are off, so {'only its question is' if spoken else 'nothing is'} said")
    record(
        Recounted(
            session,
            spoken,
            tuple(dict.fromkeys(segment.topic.name for segment in (*told.sections, *told.settled))),
            tuple(question.text for question in told.questions),
            kept=key is not None,
        )
    )
    # Marked told either way: a turn the switch kept quiet was heard as much as it will be, and is not told later.
    await tails.spoken(telling)
    # Kept in the intermediary's context, so it can answer about what the user heard.
    return TTSSpeakFrame(f"{name}: {spoken}") if spoken else None


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


def _unsummarised(session: SessionId, name: str, error: Exception, asked: str) -> Frame:
    """What is said of a turn that could not be summarised: that, and what it is waiting on the listener to answer.

    [LAW:no-silent-failure] said without the model, as a system fact is, and logged with the reason, which is an audit
    line. Nothing is marked told, so a later telling of the same turn — its Stop after an interrupt was read — tells it
    whole.
    """
    logger.error(f"cannot summarise the turn session {session} finished: {type(error).__name__}: {error}")
    return TTSSpeakFrame(" ".join(part for part in (f"{name} finished a turn, and I could not summarise it.", asked) if part), append_to_context=False)
