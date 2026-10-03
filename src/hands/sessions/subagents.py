"""A subagent's own transcript, read beside the session that ran it."""

from pathlib import Path

from hands.core.subagents import Subagent
from hands.core.turn import AgentTask, Opening
from hands.sessions.backfill import read_transcript
from hands.sessions.payload import Payload


def read_subagent(transcript: Path, task: AgentTask) -> Subagent:
    """The work of the subagent `task` names, read from its own transcript beside the session's.

    Raises OSError for a transcript that cannot be read, a missing one among them, and Rejected for a record that is not JSON.
    """
    own = transcript.with_suffix("") / "subagents" / f"agent-{task.id}.jsonl"
    reading = read_transcript(own)
    # The record the transcript starts from is the job the subagent was given, which `task` already names: its prompt,
    # or for a fork, the parent's call that launched it, copied in ahead of the fork's own work.
    job = _started_from(own)
    steps = tuple(happening for happening in reading.happenings if not isinstance(happening, Opening) and happening.ref != job)
    return Subagent(task.id, task.description, steps)


def _started_from(own: Path) -> str | None:
    """The uuid of the record a transcript starts from, the one naming no parent; None for a transcript with none."""
    with own.open("rb") as lines:
        for line in lines:
            match Payload.parse(line).fields:
                case {"parentUuid": None, "uuid": str() as uuid}:
                    return uuid
                case _:
                    pass
    return None
