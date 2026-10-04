"""The focused session heard as it works: each burst of its progress played in the order it settled, its text said by
a summary of it; briefly, what it says it is doing alone, without the calls it makes."""

import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger
from pipecat.frames.frames import Frame

from hands.core.attention import Amount
from hands.core.effects import Progress
from hands.core.pending import Working, current
from hands.core.progress import WRITING, Doing, explained
from hands.core.session import Session, SessionId
from hands.voice.speech import Unprompted
from hands.voice.summary import SUMMARY_FAILURES, Summariser
from hands.voice.utterance import Utterance

# The one lane progress to be played goes by, from the relay that routes it to the task that plays it, with how much of
# it is said and the utterance it is.
Playing = asyncio.Queue[tuple[Progress, Amount, Utterance]]


async def keep_playing(
    playing: Playing, live_session: Callable[[SessionId], Session | None], queue_frame: Callable[[Frame], Awaitable[None]], explain: Summariser
) -> None:
    """Hand each progress routed to be played to the floor, in the order it settled, once its text is summarised, until
    cancelled. Each summary is begun as its progress is routed, so a burst waits on its own summary and never on the
    one ahead of it as well: a turn that keeps writing settles a burst faster than one summary may take.

    [LAW:no-ambient-temporal-coupling] a summary takes seconds, and the turn may end in them; the registry says whether
    it did as the summary is ready, and progress of a turn that has ended is not played, since that turn's result is
    told instead.
    """
    summarising: asyncio.Queue[tuple[Progress, Amount, Utterance, asyncio.Task[tuple[Doing | None, str | None]]]] = asyncio.Queue()

    async def begin(group: asyncio.TaskGroup) -> None:
        while True:
            progress, amount, utterance = await playing.get()
            summarising.put_nowait((progress, amount, utterance, group.create_task(_explained(progress, explain))))

    async with asyncio.TaskGroup() as group:
        group.create_task(begin(group))
        while True:
            progress, amount, utterance, summary = await summarising.get()
            explaining, failed = await summary
            # [LAW:nothing-unseen] what the text came to, or why it could not be summarised, which fails the utterance
            # and leaves it said all the same.
            utterance.annotate(explained=None if explaining is None else explaining.alone)
            if failed is not None:
                utterance.fail(f"the text could not be summarised: {failed}")
            said = _said(amount, explaining, progress.doings)
            # Progress of a turn that has ended is not played: its result is told instead. Briefly, a burst of calls
            # alone has nothing to say.
            match current(live_session(progress.session), progress.of), said:
                case False, _:
                    utterance.settle("dropped")
                case True, ():
                    utterance.settle("noted")
                case True, _:
                    await queue_frame(Unprompted(Working(progress.session, progress.of, said), (utterance,)))


def _said(amount: Amount, explaining: Doing | None, calls: tuple[Doing, ...]) -> tuple[Doing, ...]:
    """What of a burst is said: all of it; or, briefly, what the session says it is doing, which is nothing for a burst
    of calls alone."""
    told = () if explaining is None else (explaining,)
    match amount:
        case "full":
            return (*told, *calls)
        case "brief":
            return told


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
                return Doing(WRITING, None), f"{type(error).__name__}: {error}"

