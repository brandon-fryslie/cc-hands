"""The OTLP export edge: each wide event the audit log records is also sent to an OpenTelemetry collector, as a span and
as a log record.

    [telemetry]
    collector = "http://otel.example:4318"

The audit log stays the whole record, written first and whatever becomes of the collector: hands reads it back (the
catch-up, `hands log`), so what it holds never depends on a network [LAW:dataflow-not-control-flow]. The collector is
sent the same event, encoded as OTLP/HTTP JSON, so the homelab's stores and Grafana hold it beside every other
service's: the span for the trace store, and the log record, carrying the span's trace and span ids, for the event
store. What sits behind the collector is the homelab's: hands names only its address.

[LAW:nothing-unseen] each batch sent as each signal is an Exported line in the log, naming each event by its span id, how
long the send took, and, where the collector did not take it, why: a telemetry failure is itself telemetry, and a run
whose events all reached the collector says so rather than saying nothing. Events are sent in batches, as each signal
from a thread of its own, so a collector that is slow or gone costs a unit of work nothing, and each signal waits on it
beside the other, never behind it.
"""

import json
import math
import queue
import threading
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, cast
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from hands.core.wire import Exchanged, Garbled, Held, Reached, Uncopied, Unreached, Written
from hands.sessions.audit import Entry, Exported, Level, Record, Signal, jsonable, level
from hands.sessions.heartbeat import Degradation
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
# How many of a signal's batches in a row the collector must not take before hands says it is failing: one is a blip.
FAILING_BATCHES = 3

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
# OTLP's Status codes: UNSET, OK, ERROR. A cancelled run neither succeeded nor failed.
_STATUS: Mapping[Outcome, int] = {"cancelled": 0, "ok": 1, "failed": 2}
# OTLP's SPAN_KIND_INTERNAL: a unit of work inside hands, neither serving a request nor making one.
_INTERNAL = 1
# OTLP's SeverityNumber of each audit level: INFO and ERROR.
_SEVERITY: Mapping[Level, int] = {"info": 9, "error": 17}
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


@dataclass(frozen=True)
class Failing:
    """A run of a signal's batches the collector did not take all of, each after the one before: how many, and when the
    first was said. Why each was not taken is in its Exported line."""

    batches: int
    since: datetime


def failing(streak: Failing | None, exported: Exported, at: datetime) -> Failing | None:
    """The run of untaken batches once `exported`, said at `at`, follows `streak`: a batch taken ends it."""
    # [LAW:one-source-of-truth] a fold over the Exported lines the log already holds, never a counter kept beside them.
    match exported.error, streak:
        case None, _:
            return None
        case str(), None:
            return Failing(1, at)
        case str(), Failing(batches=batches, since=since):
            return Failing(batches + 1, since)


def degradation(collector: str, signal: Signal, streak: Failing | None) -> Degradation | None:
    """What hands says of `signal`'s run of untaken batches, once it is long enough to be more than a blip."""
    if streak is None or streak.batches < FAILING_BATCHES:
        return None
    # The words hold nothing a batch would change, neither a count nor an error: the same failure is not news again each
    # time it repeats, however the collector words it. Each error is in its Exported line.
    since = streak.since.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    return Degradation(f"can't export {signal}", f"the collector at {collector} has not taken all of its {signal} since {since}")


class Exports:
    """`record`, folding each Exported line it records into what the collector is failing to take, per signal, which
    the up heartbeat carries as its degradations."""

    def __init__(self, record: Record, clock: Callable[[], datetime]) -> None:
        self._record = record
        self._clock = clock
        # [LAW:no-shared-mutable-globals] replaced whole under the lock, by each signal's exporter thread and by a stop;
        # the heartbeat reads it.
        self._streaks: Mapping[tuple[str, Signal], Failing] = {}
        self._folding = threading.Lock()

    def record(self, entry: Entry) -> None:
        self._record(entry)
        if not isinstance(entry, Exported):
            return
        key = (entry.collector, entry.signal)
        with self._folding:
            streak = failing(self._streaks.get(key), entry, self._clock())
            self._streaks = {**{held: kept for held, kept in self._streaks.items() if held != key}, **({} if streak is None else {key: streak})}

    def degraded(self) -> tuple[Degradation, ...]:
        """Each signal the collector is failing to take, as a degradation."""
        streaks = self._streaks
        return tuple(said for (collector, signal), streak in streaks.items() if (said := degradation(collector, signal, streak)) is not None)


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
                "failed" if failed else "ok", error if failed else None, (), {}, {"exchange": entry.exchange, "session": entry.session, "kind": entry.kind, "path": entry.path, "final": entry.final, **facts},
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
            # A message says which way it came: streamed, or sent whole when Claude Code asked again without a stream.
            written = {"streamed": body.streamed} if isinstance(body, Written) else {}
            facts = {"status": status, "first_byte_ms": round((first - sent_at) * 1000, 3), "reply_bytes": size, **written}
            return last, body.reason if isinstance(body, Garbled) else f"the API answered {status}", facts


class _Closed:
    pass


_CLOSED = _Closed()


class _Lane:
    """One signal's share of an Exporter: the events still to be sent as it, and the thread that sends them, so each
    signal waits on the collector beside the other, never behind it."""

    def __init__(self, encoding: "Encoding", run: Callable[["_Lane"], None]) -> None:
        self.encoding = encoding
        # [LAW:no-shared-mutable-globals] send puts and the thread takes; close takes what is left once the thread has
        # outlived the timeout.
        self.queue: queue.SimpleQueue[WideEvent | _Closed] = queue.SimpleQueue()
        # [LAW:no-shared-mutable-globals] the thread alone writes it, as each send begins; close reads it once the thread
        # has outlived the timeout.
        self.sending: tuple[str, ...] = ()
        self.thread = threading.Thread(target=run, args=(self,), name=f"otlp export {encoding.signal}", daemon=True)


class Exporter:
    """Sends wide events to `collector` in batches, as each signal from a thread of its own, recording each batch and
    what became of it."""

    def __init__(self, collector: str, record: Callable[[Exported], None], linger: float = LINGER_SECONDS, timeout: float = TIMEOUT_SECONDS) -> None:
        self._collector = collector
        self._record = record
        self._linger = linger
        self._timeout = timeout
        # [LAW:no-shared-mutable-globals] close alone writes it, once, finite from then on; each lane's thread reads it
        # as it sends each batch, and send to know the queues' end is marked.
        self._deadline = math.inf
        # [LAW:no-ambient-temporal-coupling] held by send and close alike, so no event is put behind the end's mark.
        self._marking = threading.Lock()
        self._lanes = tuple(_Lane(encoding, self._run) for encoding in ENCODINGS)
        for lane in self._lanes:
            lane.thread.start()

    def send(self, event: WideEvent) -> None:
        """Queue `event`; one sent once this is closed, as work a stop did not wait for ends, is recorded unsent."""
        with self._marking:
            closed = math.isfinite(self._deadline)
            if not closed:
                for lane in self._lanes:
                    lane.queue.put(event)
        if closed:
            for lane in self._lanes:
                self._record(Exported(self._collector, lane.encoding.signal, (event.span_id,), 0.0, STOPPED, None))

    def close(self) -> None:
        """Send what is queued and stop: every event sent before this is in an Exported line when it returns. The batches
        still to send, as every signal, share the one timeout from now, and those it leaves no time for are recorded
        unsent, so a collector that is gone holds up a stop by the timeout, however much is queued."""
        # [LAW:no-ambient-temporal-coupling] set before the end of the queues is marked, so every batch sent from here on
        # reads it, whether or not its thread has reached the mark yet.
        with self._marking:
            self._deadline = time.monotonic() + self._timeout
            for lane in self._lanes:
                lane.queue.put(_CLOSED)
        for lane in self._lanes:
            lane.thread.join(max(0.0, self._deadline + 1 - time.monotonic()))
        for lane in self._lanes:
            if lane.thread.is_alive():
                # [LAW:nothing-unseen] a send that outlived the timeout, which bounds each socket operation and not a
                # name's resolution: what it holds and what is queued behind it are said here, as the process may end
                # before it.
                unsent = tuple(taken.span_id for taken in _drained(lane.queue) if isinstance(taken, WideEvent))
                # The end marked again, so the thread ends once its send does rather than waiting on the queue forever.
                lane.queue.put(_CLOSED)
                stuck = f"{STOPPED}: a send was still waiting on the collector"
                self._record(Exported(self._collector, lane.encoding.signal, (*lane.sending, *unsent), 0.0, stuck, None))

    def _run(self, lane: _Lane) -> None:
        closed = False
        while not closed:
            first = lane.queue.get()
            if isinstance(first, _Closed):
                return
            batch, closed = self._gathered(lane, first)
            self._deliver(lane, batch)

    def _gathered(self, lane: _Lane, first: WideEvent) -> tuple[list[WideEvent], bool]:
        """A batch beginning with `first`: every event sent within the linger after it, up to BATCH_SPANS; and whether
        the exporter was closed while it gathered."""
        batch = [first]
        deadline = time.monotonic() + self._linger
        while len(batch) < BATCH_SPANS and (left := deadline - time.monotonic()) > 0:
            try:
                taken = lane.queue.get(timeout=left)
            except queue.Empty:
                break
            if isinstance(taken, _Closed):
                return batch, True
            batch.append(taken)
        return batch, False

    def _deliver(self, lane: _Lane, batch: Sequence[WideEvent]) -> None:
        began = time.monotonic()
        left = min(self._timeout, self._deadline - began)
        lane.sending = tuple(event.span_id for event in batch)
        refused, warning = self._sent(lane.encoding, batch, left) if left > 0 else (STOPPED, None)
        self._record(Exported(self._collector, lane.encoding.signal, lane.sending, (time.monotonic() - began) * 1000, refused, warning))

    def _sent(self, encoding: "Encoding", batch: Sequence[WideEvent], timeout: float) -> tuple[str | None, str | None]:
        """Why the collector did not take `batch` as `encoding`'s signal, None where it took every event; and what it
        warned of where it took them all."""
        try:
            body = json.dumps(encoding.request(batch), ensure_ascii=False, allow_nan=False).encode()
            request = Request(f"{self._collector}/v1/{encoding.signal}", data=body, headers={"Content-Type": "application/json"}, method="POST")
            with _DIRECT.open(request, timeout=timeout) as response:
                return answered(encoding, response.read(), len(batch))
        except Exception as error:
            # Unreachable, an HTTP error status, or an event that would not encode: the batch is lost to the collector
            # alike, and said alike.
            return _why(error), None


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


def answered(encoding: "Encoding", body: bytes, sent: int) -> tuple[str | None, str | None]:
    """What the collector's answer of success to a batch of `sent` says, from its response for `encoding`'s signal: why
    it rejected events of it, None where it took them all; and, where it took them all, what it warned of, None where it
    said nothing."""
    match parsed := _json_or_none(body):
        case {"partialSuccess": dict()}:
            # A JSON object's keys are strings.
            said = cast(dict[str, dict[str, object]], parsed)["partialSuccess"]
            message = said.get("errorMessage")
            why = message if isinstance(message, str) and message else None
        case _:
            # A body of any other shape rejects nothing and says nothing.
            return None, None
    # The count is an int64, which OTLP's JSON may spell as a string.
    match said.get(encoding.rejected):
        case int() | str() as count if str(count).isdecimal() and int(count) > 0:
            return f"the collector rejected {int(count)} of {sent} {encoding.noun}" + (f": {why}" if why else ""), None
        case _:
            # OTLP's warning: a partial success that rejected nothing, with a message.
            return None, why


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


def logs(events: Sequence[WideEvent]) -> dict[str, object]:
    """An OTLP ExportLogsServiceRequest, in its JSON encoding, holding each event as one log record."""
    return {
        "resourceLogs": [
            {
                "resource": {"attributes": _attributes({"service.name": SERVICE})},
                "scopeLogs": [{"scope": {"name": "hands.sessions.wide"}, "logRecords": [_log_record(event) for event in events]}],
            }
        ]
    }


@dataclass(frozen=True)
class Encoding:
    """How a batch is sent as one OTLP signal: the request its events are encoded as, and the field of the signal's
    Export*ServiceResponse that counts what the collector rejected, and what it counts."""

    signal: Signal
    request: Callable[[Sequence[WideEvent]], dict[str, object]]
    rejected: str
    noun: str


# [LAW:one-source-of-truth] every signal a batch is sent as, and all that differs between them.
ENCODINGS = (Encoding("traces", spans, "rejectedSpans", "spans"), Encoding("logs", logs, "rejectedLogRecords", "log records"))
TRACES, LOGS = ENCODINGS


def _started(event: WideEvent) -> int:
    """When `event` began, in nanoseconds since the epoch, to the microsecond."""
    return (event.started_at - _EPOCH) // timedelta(microseconds=1) * 1000


def _ended(event: WideEvent) -> int:
    """When `event` ended and its audit line was written, in nanoseconds since the epoch."""
    return _started(event) + round(event.duration_ms * 1_000_000)


def _fields(event: WideEvent) -> dict[str, object]:
    # [LAW:one-source-of-truth] the event's own fields under the names its audit line gives them, so one query reads
    # either; counts and facts are flattened under their names, as OTLP attributes are spelled.
    failed: dict[str, object] = {} if event.error is None else {"error": event.error, "trace": "\n".join(event.trace)}
    return {"outcome": event.outcome, **failed, **{f"counts.{name}": n for name, n in event.counts.items()}, **{f"facts.{name}": fact for name, fact in event.facts.items()}}


def _span(event: WideEvent) -> dict[str, object]:
    started = _started(event)
    return {
        "traceId": event.trace_id,
        "spanId": event.span_id,
        # OTLP's encoding of a span with no parent: the root of its trace.
        "parentSpanId": "" if event.parent_id is None else event.parent_id,
        "name": event.event,
        "kind": _INTERNAL,
        "startTimeUnixNano": str(started),
        "endTimeUnixNano": str(_ended(event)),
        "attributes": _attributes(_fields(event)),
        "status": {"code": _STATUS[event.outcome], "message": "" if event.error is None else event.error},
    }


def _log_record(event: WideEvent) -> dict[str, object]:
    # The event store's row of the event: its name as the body, its audit level as the severity, and its span's ids, which
    # open the trace it is part of. It is stamped as it ended, as its audit line is; a log record holds one time and no
    # parent, so how long it took and the unit it ran inside are attributes.
    severity = level(event)
    return {
        "timeUnixNano": str(_ended(event)),
        "severityNumber": _SEVERITY[severity],
        "severityText": severity.upper(),
        "body": {"stringValue": event.event},
        "attributes": _attributes({"duration_ms": event.duration_ms, "parent_id": event.parent_id, **_fields(event)}),
        "traceId": event.trace_id,
        "spanId": event.span_id,
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
