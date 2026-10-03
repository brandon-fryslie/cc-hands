"""A subagent's own work, read from its own transcript and told beside the turn it reported back to.

On the parent's transcript a subagent is one call and one report, and the report is often a line: "the review found
three issues". What it found is in its own transcript, `<session>/subagents/agent-<id>.jsonl`, which is folded by the
same fold as a session's [LAW:one-source-of-truth] and joined to the parent's turn by the id both name.
"""

from dataclasses import dataclass

from hands.core.turn import AgentId, Delegated, Notified, Step, Turn


@dataclass(frozen=True)
class Subagent:
    """One subagent's work: the job it was given, as its parent described it, and every step its transcript records."""

    id: AgentId
    description: str
    steps: tuple[Step, ...]


def reporting(turn: Turn) -> tuple[str, ...]:
    """The ids of the tasks whose reports this turn carries, which name subagents where a subagent reported.

    A subagent reports where its work is done: in the result of the call that ran it, or, run in the background, in
    the notification that opens a turn of its own. A call that launched one in the background carries no report, and
    its work is told with the notification that does. A notification's task may be a background command or a
    monitor, which has no transcript of its own; only the transcript says which it is.
    """
    notified = (turn.opening.task,) if isinstance(turn.opening, Notified) and turn.opening.task is not None else ()
    returned = tuple(step.id for step in turn.steps if isinstance(step, Delegated) and step.id is not None and step.report is not None)
    return tuple(dict.fromkeys((*notified, *returned)))
