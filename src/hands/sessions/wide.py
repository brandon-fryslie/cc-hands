"""Wide events: one record for each unit of work, opened as it begins and emitted as it ends, however it ends.

    with unit("delta.read", record, counts=("commits", "files")):
        annotate(session=session)       # a fact about this run, from anywhere inside it
        count(commits=len(commits))     # a count it declared; one it never set is written as 0

Code inside a unit of work never emits; it annotates the event the nearest open unit holds, and the unit emits it, once,
on success, on failure, and on cancellation alike [LAW:nothing-unseen]. The event leaves through the `emit` the unit was
opened with, which in the daemon is the audit log's record: the log is the one export edge, and an event is one more line
in it, so no fact about a run is kept both as an event and as a line beside it [LAW:one-source-of-truth].
"""

import asyncio
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

Outcome = Literal["ok", "failed", "cancelled"]


@dataclass(frozen=True)
class WideEvent:
    """One run of one unit of work: what it was, when it began and how long it took, how it ended, and every count and
    fact the run annotated it with. `trace_id` is shared by every unit opened inside another, so a run's parts are one
    trace. `counts` holds every count the unit declared, 0 for one the run never set: ran and did nothing is a count of
    zero, where never ran is no event at all."""

    event: str
    trace_id: str
    started_at: datetime
    duration_ms: float
    outcome: Outcome
    error: str | None
    counts: Mapping[str, int]
    facts: Mapping[str, object]


@dataclass
class _Open:
    """The event a unit is building while it runs."""

    trace_id: str
    counts: dict[str, int]
    facts: dict[str, object] = field(default_factory=dict[str, object])


# [LAW:no-shared-mutable-globals] owned by unit alone, which sets it as a unit opens and resets it as it closes; a task
# started inside a unit copies the context it was started in, so its annotations land on that unit's event.
_open: ContextVar[_Open | None] = ContextVar("the unit of work open here", default=None)


@contextmanager
def unit(event: str, emit: Callable[[WideEvent], None], counts: tuple[str, ...] = ()) -> Generator[None]:
    """Run the body as one unit of work named `event`, and emit its event as the body ends, however it ends.

    The body's exception is the body's: it is recorded on the event and raised on, never swallowed here.
    """
    enclosing = _open.get()
    opened = _Open(uuid4().hex if enclosing is None else enclosing.trace_id, dict.fromkeys(counts, 0))
    started_at, began = datetime.now(UTC), time.monotonic()
    token = _open.set(opened)
    outcome: Outcome = "ok"
    error: str | None = None
    try:
        yield
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except BaseException as raised:
        outcome, error = "failed", f"{type(raised).__name__}: {raised}"
        raise
    finally:
        _open.reset(token)
        duration_ms = round((time.monotonic() - began) * 1000, 3)
        emit(WideEvent(event, opened.trace_id, started_at, duration_ms, outcome, error, opened.counts, opened.facts))


def annotate(**facts: object) -> None:
    """Add facts to the event of the unit of work open here. A later fact of the same name replaces an earlier one."""
    _current().facts.update(facts)


def count(**counts: int) -> None:
    """Set counts the unit of work open here declared. A name it did not declare is a bug, refused out loud."""
    opened = _current()
    undeclared = counts.keys() - opened.counts.keys()
    if undeclared:
        raise KeyError(f"counts this unit of work did not declare: {sorted(undeclared)}")
    opened.counts.update(counts)


def _current() -> _Open:
    opened = _open.get()
    if opened is None:
        # [LAW:no-silent-failure] a fact with no unit to land on is code running outside the layer it was written for.
        raise LookupError("no unit of work is open here to annotate")
    return opened
