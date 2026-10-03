"""A subagent's own transcript, read beside the session that ran it."""

from pathlib import Path

from hands.core.subagents import Subagent
from hands.core.turn import AgentId, AgentTask, Opening
from hands.sessions.backfill import read_transcript
from hands.sessions.payload import Payload


def read_subagent(transcript: Path, task: AgentTask) -> Subagent:
    """The work of the subagent `task` names, read from its own transcript beside the session's.

    Raises OSError for a transcript that cannot be read, a missing one among them, and Rejected for a record that is not JSON.
    """
    own = transcript_of(transcript, task.id)
    reading = read_transcript(own)
    # The record the transcript starts from is the job the subagent was given, which `task` already names: its prompt,
    # or for a fork, the parent's call that launched it, copied in ahead of the fork's own work.
    job = _started_from(own)
    steps = tuple(happening for happening in reading.happenings if not isinstance(happening, Opening) and happening.ref != job)
    return Subagent(task.id, task.description, steps)


def subagents_of(transcript: Path) -> Path:
    """The folder Claude Code keeps the session's subagents in, each its transcript and the file naming its job, every
    one of them there whatever started it: a subagent's own subagents beside the session's."""
    return transcript.with_suffix("") / "subagents"


def transcript_of(transcript: Path, id: AgentId) -> Path:
    """The subagent's own transcript, beside the session's."""
    return subagents_of(transcript) / f"agent-{id}.jsonl"


def started_from(record: Payload) -> bool | None:
    """Whether the transcript starts from this record, which is its job then and no work of its own: the first record of
    the work names no parent. None for a record of Claude Code's own bookkeeping, as a fork's `fork-context-ref` is,
    which is no record of the work, so the one after it is still to say. A transcript that starts part way through its
    parent's, as a fork's did before Claude Code copied the launching call in, starts from no job."""
    match record.fields:
        case {"uuid": str(), "parentUuid": None}:
            return True
        case {"uuid": str()}:
            return False
        case _:
            return None


def _started_from(own: Path) -> str | None:
    """The uuid of the job the transcript starts from, if it starts from one."""
    with own.open("rb") as lines:
        for line in lines:
            record = Payload.parse(line)
            match started_from(record), record.fields.get("uuid"):
                case True, str() as uuid:
                    return uuid
                case False, _:
                    return None
                case _:
                    pass
    return None
