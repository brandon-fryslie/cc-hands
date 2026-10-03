"""A subagent's own transcript, read beside the session that ran it."""

import json
from pathlib import Path

from hands.core.subagents import Subagent
from hands.core.turn import AgentId, Opening
from hands.sessions.backfill import read_transcript
from hands.sessions.payload import Rejected


def read_subagent(transcript: Path, task: str) -> Subagent | None:
    """The subagent `task` names, read from its own transcript beside the session's; None where there is none, as for a
    notification from a background command or a monitor, which is a task and no subagent.

    Raises OSError for a transcript or a meta file that cannot be read, and Rejected for a meta file that does not say
    what job the subagent was given.
    """
    folder = transcript.with_suffix("") / "subagents"
    own = folder / f"agent-{task}.jsonl"
    if not own.exists():
        return None
    described = folder / f"agent-{task}.meta.json"
    try:
        meta: object = json.loads(described.read_text())
    except ValueError as error:
        raise Rejected(f"{described} is not JSON: {error}") from error
    match meta:
        case {"description": str() as description} if description:
            pass
        case _:
            raise Rejected(f"{described} names no description of the subagent's job")
    # The prompt it was given opens its transcript, and is the job the description already names.
    steps = tuple(happening for happening in read_transcript(own).happenings if not isinstance(happening, Opening))
    return Subagent(AgentId(task), description, steps)
