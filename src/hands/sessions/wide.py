"""Wide events: one record for each unit of work, opened as it begins and emitted as it ends, however it ends.

    with unit("delta.read", record, counts=("commits", "files")):
        annotate(session=session)       # a fact about this run, from anywhere inside it
        count(commits=len(commits))     # added to a count it declared; one it never counted is written as 0
        child("git.log", at, ms, "ok")  # a part of this run measured where it happened, as a span under it
        fail("the forge refused")       # this run failed, said rather than raised: it ends failed, and nothing is thrown

Code inside a unit of work never emits; it annotates the event the nearest open unit holds, and the unit emits it, once,
on success, on failure, and on cancellation alike [LAW:nothing-unseen]. The event leaves through the `emit` the unit was
opened with, which in the daemon is the audit log's record: the log is the one export edge, and an event is one more line
in it, so no fact about a run is kept both as an event and as a line beside it [LAW:one-source-of-truth].
"""

import asyncio
import time
import traceback
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from secrets import token_hex
from typing import Literal, cast
from uuid import uuid4

from hands.core.trace import Span

Outcome = Literal["ok", "failed", "cancelled"]


@dataclass(frozen=True)
class WideEvent:
    """One run of one unit of work: what it was, when it began and how long it took, how it ended, and every count and
    fact the run annotated it with. `trace_id` is shared by every unit opened inside another, so a run's parts are one
    trace, and `span_id` is this run's own within it, named as `parent_id` by each unit opened inside it. `error` and `trace` say what a failed run raised, as a Failure line does. `counts` holds every count the unit
    declared, 0 for one the run never counted: ran and did nothing is a count of zero, where never ran is no event at all."""

    event: str
    trace_id: str
    span_id: str
    parent_id: str | None
    started_at: datetime
    duration_ms: float
    outcome: Outcome
    error: str | None
    trace: tuple[str, ...]
    counts: Mapping[str, int]
    facts: Mapping[str, object]


@dataclass
class _Open:
    """The event a unit is building while it runs, and whether it has ended, after which nothing more lands on it."""

    trace_id: str
    span_id: str
    parent_id: str | None
    emit: Callable[[WideEvent], None]
    counts: dict[str, int]
    facts: dict[str, object] = field(default_factory=dict[str, object])
    # Why the run failed, where it said so rather than raising.
    failure: str | None = None
    closed: bool = False


# [LAW:no-shared-mutable-globals] owned by unit alone, which sets it as a unit opens and resets it as it closes; a task
# started inside a unit copies the context it was started in, so its annotations land on that unit's event.
_open: ContextVar[_Open | None] = ContextVar("the unit of work open here", default=None)


@contextmanager
def unit(event: str, emit: Callable[[WideEvent], None], counts: tuple[str, ...] = ()) -> Generator[None]:
    """Run the body as one unit of work named `event`, and emit its event as the body ends, however it ends.

    The body's exception is the body's: it is recorded on the event and raised on, never swallowed here.
    """
    enclosing = _open.get()
    # The W3C Trace Context size of a trace id, as OTLP carries it: 16 bytes, in hex.
    opened = _Open(uuid4().hex if enclosing is None else enclosing.trace_id, _span_id(), None if enclosing is None else enclosing.span_id, emit, dict.fromkeys(counts, 0))
    started_at, began = datetime.now(UTC), time.monotonic()
    token = _open.set(opened)
    outcome: Outcome = "ok"
    error: str | None = None
    trace: tuple[str, ...] = ()
    try:
        yield
        if opened.failure is not None:
            outcome, error = "failed", opened.failure
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except BaseException as raised:
        outcome, error, trace = "failed", f"{type(raised).__name__}: {raised}", chain(raised)
        raise
    finally:
        _open.reset(token)
        # A task the body started copied this unit along and may outlive it: what it adds now would change an event
        # already emitted and never reach the log, so it is refused instead.
        opened.closed = True
        duration_ms = round((time.monotonic() - began) * 1000, 3)
        emit(WideEvent(event, opened.trace_id, opened.span_id, opened.parent_id, started_at, duration_ms, outcome, error, trace, opened.counts, opened.facts))


def annotate(**facts: object) -> None:
    """Add facts to the event of the unit of work open here. A later fact of the same name replaces an earlier one."""
    _current().facts.update(facts)


def count(**counts: int) -> None:
    """Add to counts the unit of work open here declared. A name it did not declare is a bug, refused out loud."""
    opened = _current()
    undeclared = counts.keys() - opened.counts.keys()
    if undeclared:
        raise KeyError(f"counts this unit of work did not declare: {sorted(undeclared)}")
    for name, n in counts.items():
        opened.counts[name] += n


def fail(error: str) -> None:
    """End the unit of work open here failed, with `error`, for a failure it reports rather than raises: the run goes on
    to its end, and its event says it failed and why. A later failure of the same run replaces an earlier one."""
    _current().failure = error


def child(event: str, started_at: datetime, duration_ms: float, outcome: Outcome, error: str | None = None, **facts: object) -> None:
    """Emit a part of the unit of work open here that was timed where it happened rather than run inside it, such as a
    request another process made on its behalf: its own event, in the unit's trace, naming the unit as its parent."""
    opened = _current()
    opened.emit(WideEvent(event, opened.trace_id, _span_id(), opened.span_id, started_at, duration_ms, outcome, error, (), {}, facts))


def here() -> Span:
    """The span of the unit of work open here, for a part of it that runs where the unit is not open."""
    opened = _current()
    return Span(opened.trace_id, opened.span_id, opened.parent_id)


def within(parent: Span) -> Span:
    """A new span inside `parent`, in its trace."""
    return Span(parent.trace_id, _span_id(), parent.span_id)


def _span_id() -> str:
    # The W3C Trace Context size of a span id, as OTLP carries it: 8 bytes, in hex.
    return token_hex(8)


def _current() -> _Open:
    opened = _open.get()
    if opened is None or opened.closed:
        # [LAW:no-silent-failure] a fact with no unit to land on is code running outside the layer it was written for.
        raise LookupError("no unit of work is open here to annotate")
    return opened


def chain(error: BaseException) -> tuple[str, ...]:
    """What raised error: it and each exception it was raised from or while handling, the first cause first, each named
    and then followed by the frames it came up through, the raising one last; and inside a group, each it gathered."""
    links: list[BaseException] = []
    link: BaseException | None = error
    while link is not None and link not in links:
        links.append(link)
        link = link.__cause__ or (None if link.__suppress_context__ else link.__context__)
    return tuple(line for cause in reversed(links) for line in (f"{type(cause).__name__}: {cause}", *_frames(cause), *_gathered(cause)))


def _gathered(error: BaseException) -> tuple[str, ...]:
    # A TaskGroup raises its children's errors as one group, whose own name says only how many there were.
    children = cast(BaseExceptionGroup[BaseException], error).exceptions if isinstance(error, BaseExceptionGroup) else ()
    return tuple(line for child in children for line in chain(child))


def _frames(error: BaseException) -> tuple[str, ...]:
    # Without the source lines, which nothing here reads: looking them up opens every frame's file inside the sink.
    frames = traceback.StackSummary.extract(traceback.walk_tb(error.__traceback__), lookup_lines=False)
    return tuple(f"{frame.filename}:{frame.lineno} in {frame.name}" for frame in frames)
