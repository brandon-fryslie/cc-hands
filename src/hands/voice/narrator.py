"""Sessions' stories, heard: each turn a session finishes is read from its transcript, summarised, and spoken with its name, and each session gone is said after its last turn."""

import asyncio
from collections.abc import Awaitable, Callable

import anthropic
import openai
from loguru import logger
from pipecat.frames.frames import Frame, TTSSpeakFrame

from hands.core.effects import SessionGone, Summarise
from hands.core.session import SessionId
from hands.core.turn import Budget, render
from hands.sessions.audit import Recounted, Record
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions
from hands.sessions.transcript import UNTOLD, Told, read_turn
from hands.voice.readback import spoken_name
from hands.voice.summary import Summariser, SummaryFailed

# How much of a turn the summariser is shown: enough to name its results, few enough tokens for a local model to answer in seconds.
TURN_BUDGET = Budget(opening=600, said=1500, input=200, result=400, steps=40)

# Everything reading and summarising a turn is expected to fail with; each is said, and the next turn is still heard.
_FAILURES = (Rejected, OSError, SummaryFailed, openai.OpenAIError, anthropic.AnthropicError)


async def narrate(
    sessions: Sessions, summarise: Summariser, queue_frame: Callable[[Frame], Awaitable[None]], record: Record, budget: Budget = TURN_BUDGET
) -> None:
    """Speak each finished turn and each session gone, in the order they happened, until cancelled."""
    # How much of its newest turn each session has been told. Only this loop reads or writes it.
    told: dict[SessionId, Told] = {}
    while True:
        story = await sessions.story()
        name = spoken_name(sessions, story.session)
        match story:
            case Summarise(session=session):
                spoken, told[session] = await recount(story, told.get(session, UNTOLD), name, summarise, record, budget)
            case SessionGone(session=session):
                told.pop(session, None)
                spoken = TTSSpeakFrame(f"The session {name} is gone.")
        if spoken is not None:
            await queue_frame(spoken)


async def recount(
    finished: Summarise, told: Told, name: str, summarise: Summariser, record: Record, budget: Budget
) -> tuple[Frame | None, Told]:
    """The frame that tells the user what the turn did beyond `told`, or None when there is nothing new; and what it has been told once it is spoken."""
    try:
        # Off the loop: a long session's transcript is tens of megabytes.
        reading = await asyncio.to_thread(read_turn, finished.transcript, told, finished.closing)
        if reading is None or not reading.turn.steps:
            logger.info(f"session {finished.session} stopped with no untold turn in {finished.transcript}, so there is nothing to tell")
            return None, UNTOLD if reading is None else reading.told
        summary = await summarise(render(reading.turn, budget))
    except _FAILURES as error:
        # [LAW:no-silent-failure] said without the model, as a system fact is, and logged with the reason, which is an audit line.
        # What could not be read is told again at the next Stop, which may read it.
        logger.error(f"cannot summarise the turn session {finished.session} finished, from {finished.transcript}: {type(error).__name__}: {error}")
        return TTSSpeakFrame(f"{name} finished a turn, and I could not summarise it.", append_to_context=False), told
    record(Recounted(finished.session, summary))
    # Kept in the intermediary's context, so it can answer about what the user heard.
    return TTSSpeakFrame(f"{name}: {summary}"), reading.told
