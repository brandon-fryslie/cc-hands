"""The focused session heard as it works: each burst of its progress played in the order it settled, its text said by
a summary of it."""

import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger
from pipecat.frames.frames import Frame

from hands.core.effects import Progress
from hands.core.pending import Working
from hands.core.progress import EXPLAINING, Doing, explained
from hands.core.session import Opened, PromptId, Session, SessionId, ids
from hands.sessions.audit import ProgressTold, Record
from hands.voice.speech import Unprompted
from hands.voice.summary import SUMMARY_FAILURES, Summariser

# The one lane progress to be played goes by, from the relay that routes it to the task that plays it.
Playing = asyncio.Queue[Progress]


async def keep_playing(
    playing: Playing, live_session: Callable[[SessionId], Session | None], queue_frame: Callable[[Frame], Awaitable[None]], record: Record, explain: Summariser
) -> None:
    """Hand each progress routed to be played to the floor, in the order it settled, once its text is summarised, until
    cancelled.

    [LAW:no-ambient-temporal-coupling] a summary takes seconds, and the turn may end in them; the registry says whether
    it did as the summary is ready, and progress of a turn that has ended is not played, since that turn's result is
    told instead.
    """
    while True:
        progress = await playing.get()
        explaining, failed = await _explained(progress, explain)
        current = _running(live_session(progress.session), frozenset(progress.turn))
        # [LAW:nothing-unseen] what the text came to, and whether it was played.
        record(ProgressTold(progress.session, len(progress.written), None if explaining is None else explaining.alone, failed, current))
        if current:
            said = progress.doings if explaining is None else (explaining, *progress.doings)
            await queue_frame(Unprompted(Working(progress.session, frozenset(progress.turn), said)))


async def _explained(progress: Progress, explain: Summariser) -> tuple[Doing | None, str | None]:
    """What the text in a burst is said as, ahead of its calls, since Claude says what it will do before it does it:
    None for a burst with no text. And why it could not be summarised, when it could not; it is then said to have been
    written, and never read out."""
    match progress.written.strip():
        case "":
            return None, None
        case text:
            try:
                return explained(await explain(text)), None
            except SUMMARY_FAILURES as error:
                # [LAW:no-silent-failure] the burst is still said, its text as written and never as what it says.
                logger.error(f"the text session {progress.session} wrote could not be summarised, so it is said to have been written: {type(error).__name__}: {error}")
                return Doing(EXPLAINING, None), f"{type(error).__name__}: {error}"


def _running(session: Session | None, turn: frozenset[PromptId]) -> bool:
    """Whether the session still runs the turn that goes by any of these ids."""
    match session:
        case Session(turn=Opened() as opened):
            return not ids(opened).isdisjoint(turn)
        case _:
            return False
