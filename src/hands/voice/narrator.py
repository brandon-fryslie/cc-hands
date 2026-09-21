"""Sessions' stories, heard: each turn a session finishes is told from the tail, summarised, and spoken with its name, and each session gone is said after its last turn."""

from collections.abc import Awaitable, Callable

import anthropic
import openai
from loguru import logger
from pipecat.frames.frames import Frame, TTSSpeakFrame

from hands.core.delta import Delta
from hands.core.effects import SessionGone, Summarise
from hands.core.session import SessionId
from hands.core.turn import Budget, render
from hands.sessions.audit import Recounted, Record
from hands.sessions.delta import Changes, NoChanges
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions
from hands.sessions.tail import Tails
from hands.voice.readback import spoken_name
from hands.voice.summary import Summariser, SummaryFailed

# How much of a turn the summariser is shown: enough to name its results, few enough tokens for a local model to answer in seconds.
TURN_BUDGET = Budget(opening=600, said=1500, input=200, result=400, steps=40, files=25, changes=2000)

# Everything reading and summarising a turn is expected to fail with; each is said, and the next turn is still heard.
_FAILURES = (Rejected, OSError, SummaryFailed, openai.OpenAIError, anthropic.AnthropicError)


async def narrate(
    sessions: Sessions,
    tails: Tails,
    summarise: Summariser,
    queue_frame: Callable[[Frame], Awaitable[None]],
    record: Record,
    budget: Budget = TURN_BUDGET,
    changes: Changes | None = None,
) -> None:
    """Speak each finished turn and each session gone, in the order they happened, until cancelled."""
    read = changes or NoChanges()
    while True:
        story = await sessions.story()
        name = spoken_name(sessions, story.session)
        match story:
            case Summarise(session=session, closing=closing):
                spoken = await recount(tails, session, closing, name, summarise, record, budget, read.taken(session))
            case SessionGone():
                spoken = TTSSpeakFrame(f"The session {name} is gone.")
        if spoken is not None:
            await queue_frame(spoken)


async def recount(
    tails: Tails, session: SessionId, closing: str | None, name: str, summarise: Summariser, record: Record, budget: Budget, delta: Delta
) -> Frame | None:
    """The frame that tells the user what the turn did beyond what was told before, or None when there is nothing new.

    `delta` is what the repository says the turn did, read when it stopped. A turn with no steps left to tell
    but a repository that moved is still worth telling: that is a formatter or a code generator, and naming
    what it changed is the whole point of reading git at all.
    """
    try:
        telling = await tails.tell(session, closing)
        if telling is None or not (telling.turn.steps or delta):
            logger.info(f"session {session} stopped with no untold turn, so there is nothing to tell")
            return None
        summary = await summarise(render(telling.turn, delta, budget))
    except _FAILURES as error:
        # [LAW:no-silent-failure] said without the model, as a system fact is, and logged with the reason, which is an audit line.
        # Nothing is marked told, so what could not be summarised is told again at the next Stop, which may summarise it.
        logger.error(f"cannot summarise the turn session {session} finished: {type(error).__name__}: {error}")
        return TTSSpeakFrame(f"{name} finished a turn, and I could not summarise it.", append_to_context=False)
    record(Recounted(session, summary))
    await tails.spoken(telling)
    # Kept in the intermediary's context, so it can answer about what the user heard.
    return TTSSpeakFrame(f"{name}: {summary}")
