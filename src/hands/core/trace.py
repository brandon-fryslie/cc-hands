"""Where a unit of work sits in a trace, as W3C Trace Context names it: carried to a part of it that runs elsewhere."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    """One span of a trace: its trace's id, its own id, and the id of the span it is part of, None at a trace's root.
    In hex, at the sizes OTLP carries: a 16-byte trace id and 8-byte span ids."""

    trace_id: str
    span_id: str
    parent_id: str | None
