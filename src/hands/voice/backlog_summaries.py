"""The one task that says backlogs, off the voice path: each wanted backlog read fresh, and every sentence it is missing asked for.

A pass finds what has no sentence under its current content and asks the summariser for those, a batch at a time,
leaves before the parents that are keyed by their sentences. Every pass is one audit line.
"""

import itertools
import time
from pathlib import Path

from loguru import logger

from hands.core.sentences import answered, page
from hands.sessions.audit import BacklogUnread, Record, Summarised
from hands.sessions.backlog import Unread, read_backlog
from hands.sessions.payload import Rejected
from hands.voice.sentences import SummaryStore
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
    """Say each wanted backlog, one at a time, until cancelled."""
    while True:
        await summarise_backlog(await store.wanted(), store, summarise, record)


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
    said = rounds = calls = failed = stray = 0
    left_out: list[str] = []
    reckoning = first
    # A round says what is due; the parents it unblocks are due in the next. A round that says nothing ends the pass,
    # since another would ask the same again.
    while reckoning.due:
        rounds += 1
        before = said
        for asked in itertools.batched(reckoning.due, batch):
            calls += 1
            try:
                reply = await summarise(page(asked, TEXT_LIMIT))
            except SUMMARY_FAILURES as error:
                failed += 1
                logger.error(f"the summariser failed on {len(asked)} things from the backlog in {project}: {type(error).__name__}: {error}")
                continue
            answer = answered(reply, asked)
            store.keep(answer.said)
            said += len(answer.said)
            left_out.extend(answer.missing)
            stray += len(answer.stray)
            if answer.missing or answer.stray:
                logger.warning(f"the summariser's reply for the backlog in {project} left {list(answer.missing)} unsaid and gave {len(answer.stray)} lines that name nothing asked: {list(answer.stray)[:3]}")
        if said == before:
            break
        reckoning = store.reckon(thing)
    known = len(first.said)
    record(
        Summarised(
            str(project),
            outcome="said" if not reckoning.due and not reckoning.waiting else "partial",
            things=known + len(first.due) + first.waiting,
            known=known,
            said=said,
            unsaid=len(reckoning.due) + reckoning.waiting,
            rounds=rounds,
            calls=calls,
            failed_calls=failed,
            left_out=tuple(left_out),
            stray=stray,
            seconds=time.monotonic() - began,
        )
    )
