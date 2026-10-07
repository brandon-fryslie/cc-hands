"""The one task that says what the summary store is asked for, off the voice path: backlogs and finished turns of sessions.

A backlog pass reads the backlog fresh, finds what has no sentence under its current content, and asks the
summariser for those, a batch at a time, leaves before the parents that are keyed by their sentences. A turns pass
asks for the turns a reading of a session found unsaid. Every pass is one unit of work, and its event says what it
found, asked, said, and failed on.
"""

import itertools
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from hands.core.sentences import Due, answered, page
from hands.sessions.audit import Record
from hands.sessions.backlog import Unread, Untracked, read_backlog
from hands.sessions.payload import Rejected
from hands.sessions.wide import Fact, annotate, count, fail, unit
from hands.voice.sentences import Backlog, SummaryStore, Turns
from hands.voice.summary import Summariser, SummaryFailed

# How many things one summariser call is asked for: a few calls for a whole backlog, each reply short enough to finish.
BATCH = 20

# How much of one thing's text the summariser is shown. A ticket's median is 1.3 KB; the long tail is design notes
# whose first pages say what the ticket is.
TEXT_LIMIT = 4000

# The time a slim Claude Code takes to write BATCH lines of about thirty words each.
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
class _Missed:
    """What one pass's replies gave no sentence for, and what each call that failed raised."""

    left_out: list[str] = field(default_factory=list[str])
    errors: list[str] = field(default_factory=list[str])


# What every pass counts of the calls it makes, after the counts of its own.
CALL_COUNTS = ("said", "calls", "failed_calls", "stray")


@contextmanager
def _pass(event: str, record: Record, counts: tuple[str, ...], **facts: Fact) -> Generator[_Missed]:
    """Run the body as one pass, a unit of work whose event carries what its replies left out and its calls raised,
    written as the body ends however it ends, so a pass cut short still says what it had."""
    with unit(event, record, counts=(*counts, *CALL_COUNTS)):
        annotate(**facts)
        missed = _Missed()
        try:
            yield missed
        finally:
            annotate(left_out=tuple(missed.left_out), errors=tuple(missed.errors))


async def _say(due: Sequence[Due], store: SummaryStore, summarise: Summariser, missed: _Missed, batch: int) -> int:
    """Ask the summariser for a sentence for each of `due`, a batch at a time, keep what comes back, and say how many."""
    said = 0
    for asked in itertools.batched(due, batch):
        count(calls=1)
        try:
            reply = await summarise(page(asked, TEXT_LIMIT))
        except SummaryFailed as error:
            count(failed_calls=1)
            missed.errors.append(f"{type(error).__name__}: {error}")
            # [LAW:no-silent-failure] a call that failed fails its pass, so the pass is read as an error; the pass goes on
            # to its other batches.
            fail(f"the summariser failed on {len(asked)} things: {type(error).__name__}: {error}")
            continue
        answer = answered(reply, asked)
        store.keep(answer.said)
        said += len(answer.said)
        count(said=len(answer.said), stray=len(answer.stray))
        missed.left_out.extend(answer.missing)
    return said


async def summarise_turns(turns: Turns, store: SummaryStore, summarise: Summariser, record: Record, batch: int = BATCH) -> None:
    """Make a sentence for each turn asked for, as one unit of work."""
    # [LAW:nothing-unseen] `known` is what was said since the turns were queued, `asked` what this pass asked for; a
    # pass that found everything said is one of zeros.
    with _pass("summary.turns", record, ("known", "asked"), session=turns.session) as missed:
        # A turn is let go of as it is taken, so a read while it is being said queues it again: this pass runs after
        # both, and asks only for what is still unsaid.
        unsaid = [due for due in turns.due if store.known(due.digest) is None]
        count(known=len(turns.due) - len(unsaid), asked=len(unsaid))
        await _say(unsaid, store, summarise, missed, batch)


async def summarise_backlog(project: Path, store: SummaryStore, summarise: Summariser, record: Record, batch: int = BATCH) -> None:
    """Make every sentence the backlog in `project` is missing under its current content, as one unit of work."""
    # [LAW:nothing-unseen] `unsaid` is what is still without a sentence when the pass ends: what the summariser failed
    # on or left out, and the parents above them. `left_out` names what a reply gave no sentence for, and `stray`
    # counts the reply lines that named nothing asked, so a model that skips items reads apart from one whose calls
    # failed.
    with _pass("summary.backlog", record, ("things", "known", "unsaid", "rounds"), project=project) as missed:
        try:
            backlog = await read_backlog(project)
        except (Unread, Rejected) as error:
            # [LAW:no-silent-failure] a project hands cannot read fails its pass, saying why.
            fail(f"cannot read the backlog, so none of it is summarised: {error}")
            return
        # [LAW:nothing-unseen] a directory lit has no workspace in has nothing to say, and its event says that is why.
        match backlog:
            case Untracked():
                annotate(tracked=False)
                return
            case _:
                annotate(tracked=True)
        thing = backlog.thing()
        first = store.reckon(thing)
        count(things=len(first.said) + len(first.due) + first.waiting, known=len(first.said))
        reckoning = first
        # A round says what is due; the parents it unblocks are due in the next. A round that says nothing ends the
        # pass, since another would ask the same again.
        while reckoning.due:
            count(rounds=1)
            if not await _say(reckoning.due, store, summarise, missed, batch):
                break
            reckoning = store.reckon(thing)
        count(unsaid=len(reckoning.due) + reckoning.waiting)
