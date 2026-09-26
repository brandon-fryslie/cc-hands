"""Sessions' stories, heard: each turn a session finishes is told from the tail, summarised, and spoken with its name, and each session gone is said after its last turn."""

import time
from collections.abc import Awaitable, Callable

import anthropic
import openai
from loguru import logger
from pipecat.frames.frames import Frame, TTSSpeakFrame

from hands.core.delta import Delta
from hands.core.effects import SessionGone, Summarise
from hands.core.narration import Segment, narration, shown
from hands.core.session import PromptId, SessionId
from hands.core.turn import Budget, Interruption
from hands.sessions.audit import Recounted, Record
from hands.sessions.delta import Changes, NoChanges
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions
from hands.sessions.tail import Tails
from hands.voice.readback import spoken_name
from hands.voice.summary import Summariser, SummaryFailed
from hands.voice.summary_instruction import HEADLINE_SENTENCES

# How much of a turn the summariser is shown: enough to name its results, few enough tokens for a local model to answer in seconds.
TURN_BUDGET = Budget(opening=600, said=1500, input=200, result=400, steps=40, files=25, commits=10, changes=2000)

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
            case Summarise(session=session, turn=turn, closing=closing):
                spoken = await recount(tails, session, turn, closing, name, summarise, record, budget, await read.taken(session))
            case SessionGone():
                spoken = TTSSpeakFrame(f"The session {name} is gone.")
        if spoken is not None:
            await queue_frame(spoken)


async def recount(
    tails: Tails, session: SessionId, turn: PromptId | None, closing: str | None, name: str, summarise: Summariser, record: Record, budget: Budget, delta: Delta
) -> Frame | None:
    """The frame that tells the user what the turn did beyond what was told before, or None when there is nothing new.

    `delta` is what the repository says the turn did, read when it stopped. A turn with no steps left to tell
    but a repository that moved is still worth telling: that is a formatter or a code generator, and naming
    what it changed is the whole point of reading git at all.
    """
    try:
        telling = await tails.tell(session, turn, closing)
    except _FAILURES as error:
        return _unsummarised(session, name, error, ())
    if telling is None or not (telling.turn.steps or delta):
        logger.info(f"session {session} stopped with no untold turn, so there is nothing to tell")
        return None
    began = time.monotonic()
    # A turn stopped before it did anything has nothing for a model to report, and a model told not to say it was
    # interrupted would report something anyway: its narration is the interruption alone.
    did = delta or any(not isinstance(step, Interruption) for step in telling.turn.steps)
    try:
        headline = await summarise(shown(telling.turn, delta, budget)) if did else ""
    except _FAILURES as error:
        # What the turn is waiting on is the daemon's to find and needs no model, and a question is always said.
        return _unsummarised(session, name, error, narration("", telling.turn, delta, HEADLINE_SENTENCES).questions)
    # The number this whole epic turns on, and until now invisible: how long a finished turn waited on the
    # model before it could be spoken at all.
    logger.info(f"session {session} was summarised in {time.monotonic() - began:.2f} s")
    # The tree is cut from the turn after the headline comes back, so the sections cost nothing at the Stop that
    # matters: the counts are arithmetic over steps already read, and only the headline waits on a model.
    told = narration(headline, telling.turn, delta, HEADLINE_SENTENCES)
    spoken = told.said()
    record(
        Recounted(
            session,
            spoken,
            tuple(dict.fromkeys(segment.topic.name for segment in (*told.sections, *told.answered))),
            tuple(question.text for question in told.questions),
        )
    )
    await tails.spoken(telling)
    # Kept in the intermediary's context, so it can answer about what the user heard.
    return TTSSpeakFrame(f"{name}: {spoken}")


def _unsummarised(session: SessionId, name: str, error: Exception, questions: tuple[Segment, ...]) -> Frame:
    """What is said of a turn that could not be summarised: that, and what it is waiting on the listener to answer.

    [LAW:no-silent-failure] said without the model, as a system fact is, and logged with the reason, which is an audit
    line. Nothing is marked told, so a later telling of the same turn — its Stop after an interrupt was read — tells it
    whole.
    """
    logger.error(f"cannot summarise the turn session {session} finished: {type(error).__name__}: {error}")
    return TTSSpeakFrame(" ".join([f"{name} finished a turn, and I could not summarise it.", *(question.text for question in questions)]), append_to_context=False)
