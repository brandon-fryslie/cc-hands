"""The one task that says what the summary store is asked for, off the voice path: backlogs and finished turns of sessions.

A backlog pass reads the backlog fresh, finds what has no sentence under its current content, and asks the
summariser for those, a batch at a time, leaves before the parents that are keyed by their sentences. A turns pass
asks for the turns a reading of a session found unsaid. Every pass is one audit line.
"""

import itertools
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from hands.core.sentences import Due, answered, page
from hands.sessions.audit import BacklogUnread, Record, Summarised, TurnsSummarised
from hands.sessions.backlog import Unread, read_backlog
from hands.sessions.payload import Rejected
from hands.voice.sentences import Backlog, SummaryStore, Turns
from hands.voice.summary import SUMMARY_FAILURES, Summariser

# How many things one summariser call is asked for: a few calls for a whole backlog, each reply short enough to finish.
BATCH = 20

# How much of one thing's text the summariser is shown. A ticket's median is 1.3 KB; the long tail is design notes
# whose first pages say what the ticket is.
TEXT_LIMIT = 4000

# Room for BATCH lines of about thirty words each, and the time a slim Claude Code takes to write them.
SENTENCES_MAX_TOKENS = 2000
SENTENCES_TIMEOUT_SECONDS = 180.0

async def keep_summarising(store: SummaryStore, summarise: Summariser, record: Record) -> None:
    """Say each wanted backlog and each session's wanted turns, one at a time, until cancelled."""
    while True:
        match await store.wanted():
            case Backlog(project=project):
                await summarise_backlog(project, store, summarise, record)
            case Turns() as turns:
                await summarise_turns(turns, store, summarise, record)


@dataclass
class _Tally:
    """What one pass's calls came to, for its audit line."""

    said: int = 0
    calls: int = 0
    failed: int = 0
    stray: int = 0
    left_out: list[str] = field(default_factory=list[str])


async def _say(due: Sequence[Due], of: str, store: SummaryStore, summarise: Summariser, tally: _Tally, batch: int) -> None:
    """Ask the summariser for a sentence for each of `due`, a batch at a time, and keep what comes back."""
    for asked in itertools.batched(due, batch):
        tally.calls += 1
        try:
            reply = await summarise(page(asked, TEXT_LIMIT))
        except SUMMARY_FAILURES as error:
            tally.failed += 1
            logger.error(f"the summariser failed on {len(asked)} things from {of}: {type(error).__name__}: {error}")
            continue
        answer = answered(reply, asked)
        store.keep(answer.said)
        tally.said += len(answer.said)
        tally.left_out.extend(answer.missing)
        tally.stray += len(answer.stray)
        if answer.missing or answer.stray:
            logger.warning(f"the summariser's reply for {of} left {list(answer.missing)} unsaid and gave {len(answer.stray)} lines that name nothing asked: {list(answer.stray)[:3]}")


async def summarise_turns(turns: Turns, store: SummaryStore, summarise: Summariser, record: Record, batch: int = BATCH) -> None:
    """Make a sentence for each turn asked for, and audit the pass."""
    began = time.monotonic()
    tally = _Tally()
    # A turn is let go of as it is taken, so a read while it is being said queues it again: this pass runs after both,
    # and asks only for what is still unsaid.
    unsaid = [due for due in turns.due if store.known(due.digest) is None]
    await _say(unsaid, f"session {turns.session}", store, summarise, tally, batch)
    record(
        TurnsSummarised(
            turns.session,
            outcome="said" if tally.said == len(unsaid) else "partial",
            known=len(turns.due) - len(unsaid),
            asked=len(unsaid),
            said=tally.said,
            calls=tally.calls,
            failed_calls=tally.failed,
            left_out=tuple(tally.left_out),
            stray=tally.stray,
            seconds=time.monotonic() - began,
        )
    )


async def summarise_backlog(project: Path, store: SummaryStore, summarise: Summariser, record: Record, batch: int = BATCH) -> None:
    """Make every sentence the backlog in `project` is missing under its current content, and audit the pass."""
    began = time.monotonic()
    try:
        backlog = await read_backlog(project)
    except (Unread, Rejected) as error:
        # [LAW:no-silent-failure] a project hands cannot read is said in the log, and its pass still has its line.
        logger.error(f"cannot read the backlog in {project}, so none of it is summarised: {error}")
        record(BacklogUnread(str(project), str(error), time.monotonic() - began))
        return
    thing = backlog.thing()
    first = store.reckon(thing)
    tally = _Tally()
    rounds = 0
    reckoning = first
    # A round says what is due; the parents it unblocks are due in the next. A round that says nothing ends the pass,
    # since another would ask the same again.
    while reckoning.due:
        rounds += 1
        before = tally.said
        await _say(reckoning.due, f"the backlog in {project}", store, summarise, tally, batch)
        if tally.said == before:
            break
        reckoning = store.reckon(thing)
    known = len(first.said)
    record(
        Summarised(
            str(project),
            outcome="said" if not reckoning.due and not reckoning.waiting else "partial",
            things=known + len(first.due) + first.waiting,
            known=known,
            said=tally.said,
            unsaid=len(reckoning.due) + reckoning.waiting,
            rounds=rounds,
            calls=tally.calls,
            failed_calls=tally.failed,
            left_out=tuple(tally.left_out),
            stray=tally.stray,
            seconds=time.monotonic() - began,
        )
    )
