"""Wide events: one record for each unit of work, opened as it begins and emitted as it ends, however it ends.

    with unit("delta.read", record, counts=("commits", "files")):
        annotate(session=session)       # a fact about this run, from anywhere inside it
        count(commits=len(commits))     # added to a count it declared; one it never counted is written as 0
        child("git.log", within(here()), at, ms, "ok")  # a part of this run measured where it happened, as a span under it
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
from datetime import UTC, datetime, timedelta
from enum import IntEnum, StrEnum
from pathlib import Path
from secrets import token_hex
from typing import TYPE_CHECKING, Literal, cast
from uuid import uuid4

from hands.core.trace import Span

if TYPE_CHECKING:
    # Defined only in the standard library's type stubs: any instance of a dataclass.
    from _typeshed import DataclassInstance

Outcome = Literal["ok", "failed", "cancelled"]

# [LAW:types-are-the-program] a fact is a value both the audit log's line (hands.sessions.audit.jsonable) and an OTLP
# span's attribute can carry, so one neither can is refused by pyright where it is annotated, rather than found as its
# line is written. Containers are the immutable ones, since an event emitted is never changed, and an enum is one whose
# values are strings or integers. A dataclass's own fields are its type's to admit: pyright cannot follow them here.
type Fact = None | bool | int | float | str | datetime | timedelta | Path | StrEnum | IntEnum | DataclassInstance | tuple[Fact, ...] | frozenset[Fact]


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
    facts: Mapping[str, Fact]


@dataclass
class _Open:
    """The event a unit is building while it runs, and whether it has ended, after which nothing more lands on it."""

    trace_id: str
    span_id: str
    parent_id: str | None
    emit: Callable[[WideEvent], None]
    counts: dict[str, int]
    facts: dict[str, Fact] = field(default_factory=dict[str, Fact])
    # Why the run failed, where it said so rather than raising.
    failure: str | None = None
    closed: bool = False


# [LAW:no-shared-mutable-globals] owned by unit and continuing alone, each setting it for its body and resetting it
# after; a task started inside a unit copies the context it was started in, so its annotations land on that unit's
# event. A Span is a part of a trace begun elsewhere, which the next unit opened here continues.
_open: ContextVar[_Open | Span | None] = ContextVar("the unit of work open here", default=None)


@dataclass(frozen=True)
class Begun:
    """A unit of work that began before the body that runs it opened: its span, and when it began, on the wall clock and
    by time.monotonic(). Work started for it in between is its child all the same, and its event is timed from here."""

    span: Span
    started_at: datetime
    began: float


def begun() -> Begun:
    """A unit of work beginning here, now, inside the unit open here, in its trace, or the root of a new one."""
    return Begun(_minted(), datetime.now(UTC), time.monotonic())


@contextmanager
def unit(event: str, emit: Callable[[WideEvent], None], counts: tuple[str, ...] = (), began: Begun | None = None) -> Generator[None]:
    """Run the body as one unit of work named `event`, and emit its event as the body ends, however it ends.

    `began` is the unit `begun` here before the body opened it; by default it begins as it opens. The body's exception is
    the body's: it is recorded on the event and raised on, never swallowed here.
    """
    if began is not None and began.span.parent_id != _minted().parent_id:
        # [LAW:no-silent-failure] a unit begun under another would be written into a trace it is no part of.
        raise LookupError(f"{event} was begun under another unit of work than the one open here")
    beginning = began or begun()
    opened = _Open(beginning.span.trace_id, beginning.span.span_id, beginning.span.parent_id, emit, dict.fromkeys(counts, 0))
    token = _open.set(opened)
    raised: BaseException | None = None
    try:
        yield
    except BaseException as error:
        raised = error
        raise
    finally:
        _open.reset(token)
        # A task the body started copied this unit along and may outlive it: what it adds now would change an event
        # already emitted and never reach the log, so it is refused instead.
        opened.closed = True
        outcome, error, trace = _how(raised, opened.failure)
        emit(WideEvent(event, opened.trace_id, opened.span_id, opened.parent_id, beginning.started_at, since(beginning.began), outcome, error, trace, opened.counts, opened.facts))


def ended(event: str, emit: Callable[[WideEvent], None], began: Begun, raised: BaseException | None, failure: str | None = None, /, **facts: Fact) -> None:
    """Emit the event of a unit of work `began` elsewhere that no one body runs, as it ends: ok where nothing `raised`,
    cancelled where a CancelledError did, failed with what did otherwise, or with the `failure` it said, as `fail` says
    one. What raised is only written down, never raised again here, so its traceback stays the one it came up through."""
    outcome, error, trace = _how(raised, failure)
    emit(WideEvent(event, began.span.trace_id, began.span.span_id, began.span.parent_id, began.started_at, since(began.began), outcome, error, trace, {}, facts))


def _how(raised: BaseException | None, failure: str | None) -> tuple[Outcome, str | None, tuple[str, ...]]:
    """How a run ended, the error it is written with, and what raised it: by what `raised`, or by the `failure` it said."""
    match raised:
        case None:
            return ("ok", None, ()) if failure is None else ("failed", failure, ())
        case asyncio.CancelledError():
            return "cancelled", None, ()
        case _:
            return "failed", f"{type(raised).__name__}: {raised}", chain(raised)


@contextmanager
def continuing(parent: Span | None) -> Generator[None]:
    """Run the body as work done for `parent`, a span of a trace begun elsewhere, as W3C Trace Context propagates one
    with a request: a unit opened inside is its child, in its trace. None is no trace to continue: such a unit is a root."""
    token = _open.set(parent)
    try:
        yield
    finally:
        _open.reset(token)


def annotate(**facts: Fact) -> None:
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


def child(event: str, span: Span, started_at: datetime, duration_ms: float, outcome: Outcome, error: str | None = None, **facts: Fact) -> None:
    """Emit a part of the unit of work open here that was timed where it happened rather than run inside it, such as a
    request another process made on its behalf: its own event, as `span`, which `within` made inside the unit's, minted
    as the part began so that work done for it elsewhere can be its child."""
    opened = _current()
    if (span.trace_id, span.parent_id) != (opened.trace_id, opened.span_id):
        # [LAW:no-silent-failure] a span minted under another unit would be written into a trace it is no part of.
        raise LookupError(f"{event} is no part of the unit of work open here")
    opened.emit(WideEvent(event, span.trace_id, span.span_id, span.parent_id, started_at, duration_ms, outcome, error, (), {}, facts))


def here() -> Span:
    """The span of the unit of work open here, for a part of it that runs where the unit is not open."""
    opened = _current()
    return Span(opened.trace_id, opened.span_id, opened.parent_id)


def _minted() -> Span:
    """The span a unit of work opened here would have: inside the unit open here, in its trace, or the root of a new one."""
    match _open.get():
        case None:
            return root()
        case _Open(trace_id=trace_id, span_id=parent_id) | Span(trace_id=trace_id, span_id=parent_id):
            return Span(trace_id, _span_id(), parent_id)


def since(began: float) -> float:
    """The milliseconds since `began`, a reading of time.monotonic(), as every event's duration is written."""
    return round((time.monotonic() - began) * 1000, 3)


def root() -> Span:
    """The root span of a new trace, for work done as part of no unit of work."""
    # The W3C Trace Context size of a trace id, as OTLP carries it: 16 bytes, in hex.
    return Span(uuid4().hex, _span_id(), None)


def within(parent: Span) -> Span:
    """A new span inside `parent`, in its trace."""
    return Span(parent.trace_id, _span_id(), parent.span_id)


def _span_id() -> str:
    # The W3C Trace Context size of a span id, as OTLP carries it: 8 bytes, in hex.
    return token_hex(8)


def _current() -> _Open:
    opened = _open.get()
    if not isinstance(opened, _Open) or opened.closed:
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
