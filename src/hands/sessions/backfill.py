"""What a session did before anyone was listening, read from its transcript through the tail's own recognisers.

The daemon attaches to sessions that have been running for an hour, and "catch me up" is answered from the file
rather than from memory. [LAW:one-source-of-truth] the same fold the tail runs live, over the same records, so
what a session is told of a turn it lived through and what it is told of one it missed cannot differ in kind.
"""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from hands.core.turn import Happening, Opening, Ref, Step
from hands.sessions.payload import Rejected
from hands.sessions.transcript import ref_of, turn_record
from hands.sessions.turning import Turning


class Unseen(Exception):
    """The record a reading was to go on from is not in this transcript."""


@dataclass(frozen=True)
class Reading:
    """What a session did after the mark it was read from, and how much of that can be marked as read.

    A call whose result has not been written yet is still shown — the session is in the middle of something and
    that is worth saying — but it is not a place to go on from. A mark names a record and a reading goes on from
    after it, so marking a call that has not come back spends its result on nobody: the reader would be told the
    suite was being run and never told what failed. `settled` is how much of this a reader may mark as read; the
    rest is what the session is still waiting on, and is read again, with its results, next time.
    """

    happenings: list[Happening]
    settled: int


@dataclass(frozen=True)
class _Opened:
    """A turn's request, and how much of the session had happened when it arrived."""

    after: int  # steps read when the turn opened, which is where in the order the request belongs
    opening: Opening


def read_since(transcript: Path, since: Ref | None) -> Reading:
    """Everything a session did after the record named, or all of it for a session nobody has heard of yet.

    The whole file is folded, and only then cut at the record named, so that a call made before the cut and
    answered after it is still one step that knows its result — read from the mark on, the result would have
    arrived with no call to belong to. Raises OSError, which the caller decides what to make of, and Unseen for
    a record this transcript does not hold.

    Folded again for every reading rather than kept between them: the file is the one source of what a session
    did, and a fold held on the side is a second copy of it that the session writing the file can make wrong
    [LAW:one-source-of-truth]. The largest transcript on this machine is 30 MB and folds in 0.19 s into 2,966
    happenings, which the caller runs in a thread, so the cost is paid off the event loop.
    """
    turning = Turning()
    openings: list[_Opened] = []
    read = 0  # how much of the session the reader has already been given
    seen = since is None
    # [LAW:no-ambient-temporal-coupling] a session may be writing while this reads, and a record is whole only
    # once its newline is written, so the bytes after the last newline are not read as a record.
    *complete, _unfinished = transcript.read_bytes().split(b"\n")
    for line in complete:
        try:
            record = turn_record(line)
        except Rejected as error:
            # [LAW:no-silent-failure] one unreadable line is skipped and said; the rest of the session is still read.
            logger.error(f"a record in {transcript} could not be read, so it is left out of the backfill: {error}")
            continue
        if record is None:
            continue
        opening = turning.consume(record)
        if opening is not None:
            # The tail lets go of the turn before this one; a reading keeps every one of them, each in its place.
            # What was asked is most of what a session's morning means: the steps alone say how, never what for.
            openings.append(_Opened(len(turning.slots), opening))
        if since is not None and ref_of(record) == since:
            seen = True
            read = len(turning.slots) + len(openings)
    if not seen:
        # [LAW:no-silent-failure] a mark this transcript never held would otherwise read as a mark at the very
        # start, and the whole session would be told again as though it were new.
        raise Unseen(f"{transcript} holds no record {since}")
    happenings, places = _in_order(turning.steps(), openings)
    return Reading(happenings[read:], max(0, _settled(happenings, _waiting(turning, places)) - read))


def _waiting(turning: Turning, places: list[int]) -> set[int]:
    """Where in the order the calls are that no result has been written for."""
    return {places[place] for place in turning.places.values() if isinstance(turning.slots[place], str)}


def _settled(happenings: list[Happening], waiting: set[int]) -> int:
    """How much of a reading is finished business, which is all of it up to the calls it ends on.

    Only the calls it *ends* on: a call with work after it is one the session moved on from, whose result is
    never coming, and holding the mark behind it would re-read the rest of the session for ever after.
    """
    settled = len(happenings)
    while settled > 0 and settled - 1 in waiting:
        settled -= 1
    return settled


def _in_order(steps: list[Step], openings: list[_Opened]) -> tuple[list[Happening], list[int]]:
    """Every request and every step in the order they happened, and where in that order each step ended up."""
    happenings: list[Happening] = []
    places: list[int] = []
    opened = 0
    for index, step in enumerate(steps):
        while opened < len(openings) and openings[opened].after <= index:
            happenings.append(openings[opened].opening)
            opened += 1
        places.append(len(happenings))
        happenings.append(step)
    happenings.extend(rest.opening for rest in openings[opened:])
    return happenings, places
