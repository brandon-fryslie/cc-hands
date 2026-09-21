"""What a session did before anyone was listening, read from its transcript through the tail's own recognisers.

The daemon attaches to sessions that have been running for an hour, and "catch me up" is answered from the file
rather than from memory. [LAW:one-source-of-truth] the same fold the tail runs live, over the same records, so
what a session is told of a turn it lived through and what it is told of one it missed cannot differ in kind.
"""

from pathlib import Path

from loguru import logger

from hands.core.turn import Ref, Step
from hands.sessions.payload import Rejected
from hands.sessions.transcript import ref_of, turn_record
from hands.sessions.turning import Turning


def steps_since(transcript: Path, since: Ref | None) -> list[Step]:
    """Every step the session took after the record named, or all of them for a session nobody has heard of yet.

    The whole file is folded, and only then cut at the record named, so that a call made before it and answered
    after it is still one step that knows its result — read from the mark on, the result would have had no call
    to belong to. Raises OSError, which the caller decides what to make of.
    """
    turning = Turning()
    mark = 0
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
        turning.consume(record)
        if since is not None and ref_of(record) == since:
            # Everything through the record named is what the reader already has, however much of it is one step.
            mark = len(turning.slots)
    return turning.steps()[mark:]
