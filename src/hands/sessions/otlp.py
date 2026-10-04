"""The OTLP export edge: each wide event the audit log records is also sent to an OpenTelemetry collector, as a span.

    [telemetry]
    collector = "http://otel.example:4318"

The audit log stays the whole record, written first and whatever becomes of the collector: hands reads it back (the
catch-up, `hands log`), so what it holds never depends on a network [LAW:dataflow-not-control-flow]. The collector is
sent the same event, encoded as OTLP/HTTP JSON, so the homelab's stores and Grafana hold it beside every other
service's. What sits behind the collector is the homelab's: hands names only its address.

[LAW:nothing-unseen] each batch sent is an Exported line in the log, naming each span by its id, how long the send took,
and, where the collector did not take it, why: a telemetry failure is itself telemetry, and a run whose events all
reached the collector says so rather than saying nothing. Events are sent from a thread of their own, in batches, so a
collector that is slow or gone costs a unit of work nothing.
"""

import json
import math
import queue
import threading
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Literal
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from hands.core.wire import Exchanged, Garbled, Held, Reached, Uncopied, Unreached
from hands.sessions.audit import Entry, Exported, Record, jsonable, level
from hands.sessions.wide import Fact, Outcome, WideEvent

# OpenTelemetry's service.name, which every span carries on its resource.
SERVICE = "hands"
# How long a batch waits for more events after its first, and the most it holds.
LINGER_SECONDS = 1.0
BATCH_SPANS = 512
# How long one batch's request may take before the collector is counted unreachable.
TIMEOUT_SECONDS = 5.0
# Why a batch a stop left no time for, or one sent once the exporter was closed, was not sent.
STOPPED = "hands stopped before the batch could be sent"

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
# OTLP's Status codes: UNSET, OK, ERROR. A cancelled run neither succeeded nor failed.
_STATUS: Mapping[Outcome, int] = {"cancelled": 0, "ok": 1, "failed": 2}
# OTLP's SPAN_KIND_INTERNAL: a unit of work inside hands, neither serving a request nor making one.
_INTERNAL = 1
# Straight to the address the settings name, past every proxy: the environment's, which in a session fritter taps is the
# tap (hands.sessions.untap), and the system's.
_DIRECT = build_opener(ProxyHandler({}))


@contextmanager
def exporting(collector: str | None, record: Record) -> Generator[Record]:
    """`record`, and where a collector is set, each wide event recorded through it also sent there; what is still to be
    sent as this ends is sent, or recorded unsent, before it returns."""
    if collector is None:
        yield record
        return
    exporter = Exporter(collector, record)

    def both(entry: Entry) -> None:
        record(entry)
        if (event := traced(entry)) is not None:
            exporter.send(event)

    try:
        yield both
    finally:
        exporter.close()


def traced(entry: Entry) -> WideEvent | None:
    """The span a line of the audit log is in a trace: a wide event as it is, and the record of an exchange the proxy or
    tap handled, as the part of the unit of work it was made for, such as a voice turn's round trip to the model, or as
    the root of a trace of its own; None for a line in no trace."""
    match entry:
        case WideEvent():
            return entry
        case Exchanged(span=span, sent_at=sent_at, reply=reply):
            # [LAW:one-source-of-truth] a view of the record, under its own names, never a second record kept of it; and
            # [LAW:single-enforcer] failed where the audit log judges the line an error.
            failed = level(entry) == "error"
            ended, error, facts = _reply_end(sent_at, reply)
            return WideEvent(
                "proxy.exchange", span.trace_id, span.span_id, span.parent_id, datetime.fromtimestamp(sent_at, UTC), round((ended - sent_at) * 1000, 3),
                "failed" if failed else "ok", error if failed else None, (), {}, {"exchange": entry.exchange, "session": entry.session, "final": entry.final, **facts},
            )
        case _:
            return None


def _reply_end(sent_at: float, reply: Reached | Unreached | Held | Uncopied) -> tuple[float, str | None, Mapping[str, Fact]]:
    """When a request's reply ended, what it failed of where it did, and what it says of the reply."""
    match reply:
        case Unreached(error=error, failed_at=at) | Uncopied(reason=error, lost_at=at):
            return at, error, {}
        case Held(answered_at=at):
            return at, None, {}
        case Reached(status=status, first_byte_at=first, last_byte_at=last, reply_bytes=size, body=body):
            facts = {"status": status, "first_byte_ms": round((first - sent_at) * 1000, 3), "reply_bytes": size}
            return last, body.reason if isinstance(body, Garbled) else f"the API answered {status}", facts


class _Closed:
    pass


_CLOSED = _Closed()


class Exporter:
    """Sends wide events to `collector` in batches, from a thread of its own, recording each batch and what became of it."""

    def __init__(self, collector: str, record: Callable[[Exported], None], linger: float = LINGER_SECONDS, timeout: float = TIMEOUT_SECONDS) -> None:
        self._collector = collector
        self._record = record
        self._linger = linger
        self._timeout = timeout
        # [LAW:no-shared-mutable-globals] close alone writes it, once, finite from then on; the thread reads it as it
        # sends each batch, and send to know the queue's end is marked.
        self._deadline = math.inf
        # [LAW:no-shared-mutable-globals] the thread alone writes it, as each send begins; close reads it once the thread
        # has outlived the timeout.
        self._sending: tuple[str, ...] = ()
        # [LAW:no-shared-mutable-globals] send puts and the thread takes; close takes what is left once the thread has
        # outlived the timeout.
        self._queue: queue.SimpleQueue[WideEvent | _Closed] = queue.SimpleQueue()
        # [LAW:no-ambient-temporal-coupling] held by send and close alike, so no event is put behind the end's mark.
        self._marking = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="otlp export", daemon=True)
        self._thread.start()

    def send(self, event: WideEvent) -> None:
        """Queue `event`; one sent once this is closed, as work a stop did not wait for ends, is recorded unsent."""
        with self._marking:
            closed = math.isfinite(self._deadline)
            if not closed:
                self._queue.put(event)
        if closed:
            self._record(Exported(self._collector, (event.span_id,), 0.0, STOPPED))

    def close(self) -> None:
        """Send what is queued and stop: every event sent before this is in an Exported line when it returns. The batches
        still to send share the one timeout from now, and those it leaves no time for are recorded unsent, so a collector
        that is gone holds up a stop by the timeout, however much is queued."""
        # [LAW:no-ambient-temporal-coupling] set before the end of the queue is marked, so every batch sent from here on
        # reads it, whether or not the thread has reached the mark yet.
        with self._marking:
            self._deadline = time.monotonic() + self._timeout
            self._queue.put(_CLOSED)
        self._thread.join(self._timeout + 1)
        if self._thread.is_alive():
            # [LAW:nothing-unseen] a send that outlived the timeout, which bounds each socket operation and not a name's
            # resolution: what it holds and what is queued behind it are said here, as the process may end before it.
            unsent = [taken.span_id for taken in _drained(self._queue) if isinstance(taken, WideEvent)]
            self._record(Exported(self._collector, (*self._sending, *unsent), 0.0, f"{STOPPED}: a send was still waiting on the collector"))

    def _run(self) -> None:
        closed = False
        while not closed:
            first = self._queue.get()
            if isinstance(first, _Closed):
                return
            batch, closed = self._gathered(first)
            self._deliver(batch)

    def _gathered(self, first: WideEvent) -> tuple[list[WideEvent], bool]:
        """A batch beginning with `first`: every event sent within the linger after it, up to BATCH_SPANS; and whether
        the exporter was closed while it gathered."""
        batch = [first]
        deadline = time.monotonic() + self._linger
        while len(batch) < BATCH_SPANS and (left := deadline - time.monotonic()) > 0:
            try:
                taken = self._queue.get(timeout=left)
            except queue.Empty:
                break
            if isinstance(taken, _Closed):
                return batch, True
            batch.append(taken)
        return batch, False

    def _deliver(self, batch: Sequence[WideEvent]) -> None:
        began = time.monotonic()
        left = min(self._timeout, self._deadline - began)
        self._sending = tuple(event.span_id for event in batch)
        refused = self._sent(batch, left) if left > 0 else STOPPED
        self._record(Exported(self._collector, self._sending, (time.monotonic() - began) * 1000, refused))

    def _sent(self, batch: Sequence[WideEvent], timeout: float) -> str | None:
        """Why the collector did not take `batch`, None where it took every span."""
        try:
            request = Request(f"{self._collector}/v1/traces", data=json.dumps(spans(batch), ensure_ascii=False, allow_nan=False).encode(), headers={"Content-Type": "application/json"}, method="POST")
            with _DIRECT.open(request, timeout=timeout) as response:
                return rejected(response.read(), len(batch))
        except Exception as error:
            # Unreachable, an HTTP error status, or an event that would not encode: the batch is lost to the collector
            # alike, and said alike.
            return _why(error)


def _drained(queued: "queue.SimpleQueue[WideEvent | _Closed]") -> list[WideEvent | _Closed]:
    taken: list[WideEvent | _Closed] = []
    while True:
        try:
            taken.append(queued.get_nowait())
        except queue.Empty:
            return taken


def _why(error: Exception) -> str:
    said = f"{type(error).__name__}: {error}"
    if not isinstance(error, HTTPError):
        return said
    # The collector's reason for an error status is the body it answered with, an OTLP Status.
    try:
        return f"{said}: {error.read().decode(errors='replace')}"
    # [LAW:no-silent-failure] said in the reason: a body cut short (IncompleteRead) or a dropped connection must not end
    # the thread, and every batch after this one with it.
    except Exception as unread:
        return f"{said}, its body unread: {type(unread).__name__}: {unread}"


def _json_or_none(body: bytes) -> object:
    # A success answered with a body that is not JSON, as a gateway's "OK", says nothing of rejected spans.
    try:
        return json.loads(body)
    except ValueError:
        return None


def rejected(body: bytes, sent: int) -> str | None:
    """Why the collector rejected spans of a batch of `sent` it answered with success, from its
    ExportTraceServiceResponse; None where it took them all."""
    # rejectedSpans is an int64, which OTLP's JSON may spell as a string; a body of any other shape rejects nothing.
    match _json_or_none(body):
        case {"partialSuccess": {"rejectedSpans": int() | str() as count, "errorMessage": str(why)}} if str(count).isdecimal() and int(count) > 0:
            return f"the collector rejected {int(count)} of {sent} spans: {why}"
        case {"partialSuccess": {"rejectedSpans": int() | str() as count}} if str(count).isdecimal() and int(count) > 0:
            return f"the collector rejected {int(count)} of {sent} spans"
        case _:
            return None


def spans(events: Sequence[WideEvent]) -> dict[str, object]:
    """An OTLP ExportTraceServiceRequest, in its JSON encoding, holding each event as one span."""
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _attributes({"service.name": SERVICE})},
                "scopeSpans": [{"scope": {"name": "hands.sessions.wide"}, "spans": [_span(event) for event in events]}],
            }
        ]
    }


def _span(event: WideEvent) -> dict[str, object]:
    # [LAW:one-source-of-truth] the attributes are the event's own fields under the names its audit line gives them, so
    # one query reads either; counts and facts are flattened under their names, as OTLP attributes are spelled.
    started = (event.started_at - _EPOCH) // timedelta(microseconds=1) * 1000
    failed: dict[str, object] = {} if event.error is None else {"error": event.error, "trace": "\n".join(event.trace)}
    return {
        "traceId": event.trace_id,
        "spanId": event.span_id,
        # OTLP's encoding of a span with no parent: the root of its trace.
        "parentSpanId": "" if event.parent_id is None else event.parent_id,
        "name": event.event,
        "kind": _INTERNAL,
        "startTimeUnixNano": str(started),
        "endTimeUnixNano": str(started + round(event.duration_ms * 1_000_000)),
        "attributes": _attributes(
            {"outcome": event.outcome, **failed, **{f"counts.{name}": n for name, n in event.counts.items()}, **{f"facts.{name}": fact for name, fact in event.facts.items()}}
        ),
        "status": {"code": _STATUS[event.outcome], "message": "" if event.error is None else event.error},
    }


def _attributes(values: Mapping[str, object]) -> list[dict[str, object]]:
    return [{"key": key, "value": _value(value)} for key, value in values.items()]


AnyValue = Literal["boolValue", "intValue", "doubleValue", "stringValue"]


def _value(value: object) -> dict[AnyValue, object]:
    # [LAW:one-source-of-truth] OTLP's AnyValue of the value as its audit line writes it, whose 64-bit integers JSON
    # carries as strings; anything not a scalar is that line's JSON.
    match written := jsonable(value):
        case None:
            # OTLP's empty AnyValue: the attribute is there, holding nothing, as null is on the line.
            return {}
        case bool():
            return {"boolValue": written}
        case int():
            return {"intValue": str(written)}
        case float() if math.isnan(written):
            # proto3 JSON's spellings of the doubles JSON has no number for.
            return {"doubleValue": "NaN"}
        case float() if math.isinf(written):
            return {"doubleValue": "Infinity" if written > 0 else "-Infinity"}
        case float():
            return {"doubleValue": written}
        case str():
            return {"stringValue": written}
        case _:
            return {"stringValue": json.dumps(written, ensure_ascii=False)}
