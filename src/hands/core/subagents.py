"""A subagent's own work, read from its own transcript and told beside the turn it reported back to.

On the parent's transcript a subagent is one call and one report, and the report is often a line: "the review found
three issues". What it found is in its own transcript, `<session>/subagents/agent-<id>.jsonl`, which is folded by the
same fold as a session's [LAW:one-source-of-truth] and joined to the parent's turn by the id both name.
"""

from dataclasses import dataclass

from hands.core.turn import AgentId, AgentTask, Answering, Delegated, Notified, Reported, Step, Turn


@dataclass(frozen=True)
class Subagent:
    """One subagent's work: the job it was given, as its parent described it, and every step its transcript records."""

    id: AgentId
    description: str
    steps: tuple[Step, ...]


def reporting(turn: Turn) -> tuple[AgentTask, ...]:
    """The subagents whose reports this telling of the turn carries, each named with the job it was given.

    A subagent reports where its work is done: in the result of the call that ran it, or, run in the background, in
    a notification, which opens a turn of its own or, handed to a session still working, is a step of the turn under
    way. A call that launched one in the background carries no report, and its work is told with the notification that
    does. A notification that opens the turn is carried only by the telling that answers it: a later telling of the same
    turn tells only the steps since, and the work again with them would be told twice. One that is a step is carried,
    as any step is, by the one telling whose steps hold it.
    """
    opening = turn.opening
    notified = (opening.agent,) if isinstance(opening, Notified) and opening.agent is not None and isinstance(turn.standing, Answering) else ()
    returned = tuple(AgentTask(step.id, step.description) for step in turn.steps if isinstance(step, Delegated) and step.id is not None and step.report is not None)
    reported = tuple(step.agent for step in turn.steps if isinstance(step, Reported) and step.agent is not None)
    return tuple({task.id: task for task in (*notified, *returned, *reported)}.values())
