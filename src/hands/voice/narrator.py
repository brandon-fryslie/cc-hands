"""Finished turns, heard: each turn a session finishes is read from its transcript, summarised, and spoken with its name."""

import asyncio
from collections.abc import Awaitable, Callable

import anthropic
import openai
from loguru import logger
from pipecat.frames.frames import Frame, TTSSpeakFrame

from hands.core.effects import Summarise
from hands.core.turn import Budget, render
from hands.sessions.audit import Recounted, Record
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions
from hands.sessions.transcript import read_turn
from hands.voice.readback import spoken_name
from hands.voice.summary import Summariser, SummaryFailed

# How much of a turn the summariser is shown: enough to name its results, few enough tokens for a local model to answer in seconds.
TURN_BUDGET = Budget(prompt=600, said=1500, input=200, result=400, steps=40)

# Everything reading and summarising a turn is expected to fail with; each is said, and the next turn is still heard.
_FAILURES = (Rejected, OSError, SummaryFailed, openai.OpenAIError, anthropic.AnthropicError)


async def narrate_turns(
    sessions: Sessions, summarise: Summariser, queue_frame: Callable[[Frame], Awaitable[None]], record: Record, budget: Budget = TURN_BUDGET
) -> None:
    """Speak each finished turn, in the order the sessions stopped, until cancelled."""
    while True:
        finished = await sessions.finished()
        spoken = await recount(finished, spoken_name(sessions, finished.session), summarise, record, budget)
        if spoken is not None:
            await queue_frame(spoken)


async def recount(finished: Summarise, name: str, summarise: Summariser, record: Record, budget: Budget) -> Frame | None:
    """The frame that tells the user what the turn did; None for a session that stopped before it was ever prompted."""
    try:
        # Off the loop: a long session's transcript is tens of megabytes.
        turn = await asyncio.to_thread(read_turn, finished.transcript)
        if turn is None:
            logger.info(f"session {finished.session} stopped with no prompt in {finished.transcript}, so there is no turn to tell")
            return None
        summary = await summarise(render(turn, budget))
    except _FAILURES as error:
        # [LAW:no-silent-failure] said without the model, as a system fact is, and logged with the reason, which is an audit line.
        logger.error(f"cannot summarise the turn session {finished.session} finished, from {finished.transcript}: {type(error).__name__}: {error}")
        return TTSSpeakFrame(f"{name} finished a turn, and I could not summarise it.", append_to_context=False)
    record(Recounted(finished.session, summary))
    # Kept in the intermediary's context, so it can answer about what the user heard.
    return TTSSpeakFrame(f"{name}: {summary}")
