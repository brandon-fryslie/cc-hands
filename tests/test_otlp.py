"""The OTLP export edge: each wide event goes to the audit log, and to the collector where one is set, as a span and as a
log record; each batch sent as each signal is said in the log, by span id, with why where the collector did not take it."""

import io
import json
import os
import socket
import threading
import time
from collections.abc import Generator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.request import Request

import pytest

from hands.core.session import SessionId
from hands.core.trace import Span
from hands.core.wire import Answered, Exchanged, Garbled, Held, MainTurn, Reached, Unreached
from hands.daemon import indicator
from hands.daemon.cli import main
from hands.sessions import heartbeat, otlp
from hands.sessions.audit import AuditLog, Entry, Exported, segment, segments
from hands.sessions.home import Home
from hands.sessions.otlp import BATCH_SPANS, ENCODINGS, FAILING_BATCHES, LOGS, STOPPED, TIMEOUT_SECONDS, TRACES, Exporter, Exports, Failing, answered, degradation, exporting, failing, logs, spans, traced
from hands.sessions.wide import WideEvent, annotate, count, root, unit

STARTED = datetime(2026, 10, 3, 12, 0, 0, 250_000, tzinfo=UTC)


class Collector:
    """A stand-in for the collector's OTLP/HTTP receiver: it keeps every request body posted to it, and answers a post to
    /v1/<signal> with `status` where `refusing` names the signal and 200 otherwise, `answer` the body either way."""

    def __init__(self, answer: bytes = b"{}", status: int = 200, refusing: tuple[str, ...] = ("traces", "logs")) -> None:
        self.bodies: list[dict[str, object]] = []
        # What it answers each request with from now on, so a collector that refused can come back.
        self.status = status
        self.answer = answer
        self.paths: list[str] = []
        received = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                received.paths.append(self.path)
                received.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(received.status if self.path.removeprefix("/v1/") in refusing else 200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(received.answer)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def spans(self) -> list[dict[str, Any]]:
        return [span for body in self.bodies if "resourceSpans" in body for span in _spans(body)]

    def log_records(self) -> list[dict[str, Any]]:
        return [record for body in self.bodies if "resourceLogs" in body for record in _log_records(body)]


def _spans(request: dict[str, object]) -> list[dict[str, Any]]:
    """The spans of an ExportTraceServiceRequest, as JSON reads them."""
    resources = cast(list[dict[str, Any]], request["resourceSpans"])
    return [span for resource in resources for scope in resource["scopeSpans"] for span in scope["spans"]]


def _log_records(request: dict[str, object]) -> list[dict[str, Any]]:
    """The log records of an ExportLogsServiceRequest, as JSON reads them."""
    resources = cast(list[dict[str, Any]], request["resourceLogs"])
    return [record for resource in resources for scope in resource["scopeLogs"] for record in scope["logRecords"]]


@pytest.fixture
def collector() -> Generator[Collector]:
    running = Collector()
    yield running
    running.close()


def _stopped() -> str:
    """The address of a collector that is not running: a port nothing listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{probe.getsockname()[1]}"


def _lines(directory: Path) -> list[dict[str, object]]:
    return [json.loads(line) for base in segments(directory) for line in segment(directory, base).read_text().splitlines()]


def _event(**overrides: Any) -> WideEvent:
    fields: dict[str, Any] = {
        "event": "delta.read", "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736", "span_id": "00f067aa0ba902b7", "parent_id": None,
        "started_at": STARTED, "duration_ms": 12.5, "outcome": "ok", "error": None, "trace": (),
        "counts": {"commits": 2, "files": 0}, "facts": {"session": "s1", "clean": True, "paths": ("a", "b")},
    }  # fmt: skip
    return WideEvent(**{**fields, **overrides})


def test_an_event_is_one_span_under_its_trace_with_its_counts_and_facts_as_attributes() -> None:
    sent = spans([_event()])
    [resource] = cast(list[dict[str, Any]], sent["resourceSpans"])
    assert resource["resource"] == {"attributes": [{"key": "service.name", "value": {"stringValue": "hands"}}]}
    [span] = _spans(sent)
    assert (span["traceId"], span["spanId"], span["parentSpanId"], span["name"]) == ("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7", "", "delta.read")
    # Nanoseconds since the epoch, as strings, the start to the microsecond and the end duration_ms after it.
    assert span["startTimeUnixNano"] == "1791028800250000000" and span["endTimeUnixNano"] == "1791028800262500000"
    assert span["attributes"] == [
        {"key": "outcome", "value": {"stringValue": "ok"}},
        # A count of zero is sent as zero, as the log writes it.
        {"key": "counts.commits", "value": {"intValue": "2"}},
        {"key": "counts.files", "value": {"intValue": "0"}},
        {"key": "facts.session", "value": {"stringValue": "s1"}},
        {"key": "facts.clean", "value": {"boolValue": True}},
        {"key": "facts.paths", "value": {"stringValue": '["a", "b"]'}},
    ]
    assert span["status"] == {"code": 1, "message": ""}


def test_a_failed_event_is_an_error_span_carrying_what_it_raised_and_a_cancelled_one_is_unset() -> None:
    failure = _event(outcome="failed", error="RuntimeError: the disk is full", trace=("RuntimeError: the disk is full", "delta.py:9 in read"))
    failed, cancelled = _spans(spans([failure, _event(outcome="cancelled")]))
    assert failed["status"] == {"code": 2, "message": "RuntimeError: the disk is full"}
    assert {"key": "trace", "value": {"stringValue": "RuntimeError: the disk is full\ndelta.py:9 in read"}} in failed["attributes"]
    assert cancelled["status"] == {"code": 0, "message": ""}


def test_an_event_is_one_log_record_named_by_its_body_at_its_audit_level_carrying_its_span_and_its_attributes() -> None:
    sent = logs([_event(), _event(outcome="failed", error="RuntimeError: the disk is full", trace=("RuntimeError: the disk is full",))])
    [resource] = cast(list[dict[str, Any]], sent["resourceLogs"])
    assert resource["resource"] == {"attributes": [{"key": "service.name", "value": {"stringValue": "hands"}}]}
    ok, failed = _log_records(sent)
    # Its span's ids, which open the trace it is part of in the trace store.
    assert (ok["traceId"], ok["spanId"], ok["body"]) == ("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7", {"stringValue": "delta.read"})
    # Stamped as it ended, as its audit line is.
    assert (ok["timeUnixNano"], ok["severityNumber"], ok["severityText"]) == ("1791028800262500000", 9, "INFO")
    # The span's attributes, and how long it took and the unit it ran inside, which a log record has no end or parent to say.
    assert ok["attributes"] == [{"key": "duration_ms", "value": {"doubleValue": 12.5}}, {"key": "parent_id", "value": {}}, *_spans(spans([_event()]))[0]["attributes"]]
    [child] = _log_records(logs([_event(parent_id="1111111111111111")]))
    assert {"key": "parent_id", "value": {"stringValue": "1111111111111111"}} in child["attributes"]
    assert (failed["severityNumber"], failed["severityText"]) == (17, "ERROR")
    assert {"key": "error", "value": {"stringValue": "RuntimeError: the disk is full"}} in failed["attributes"]


def test_with_a_collector_each_event_is_in_the_log_and_reaches_the_collector_and_a_unit_inside_another_is_its_child(collector: Collector, tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit", clock=datetime.now)
    with exporting(collector.url, log.record) as record:
        with unit("turn", record):
            with unit("delta.read", record, counts=("commits",)):
                annotate(session="s1")
                count(commits=1)
    lines = _lines(tmp_path / "audit")
    assert [line["type"] for line in lines] == ["WideEvent", "WideEvent", "Exported", "Exported"]
    inner, outer, *exported = lines
    # A batch the collector took is said too, as each signal, so a run whose events all reached it is told from one that sent none.
    assert sorted((line["signal"], line["level"], line["collector"], line["spans"], line["error"]) for line in exported) == [
        (signal, "info", collector.url, [inner["span_id"], outer["span_id"]], None) for signal in ("logs", "traces")
    ]
    assert sorted(collector.paths) == ["/v1/logs", "/v1/traces"]
    sent = {span["spanId"]: span for span in collector.spans()}
    assert set(sent) == {inner["span_id"], outer["span_id"]}
    assert sent[inner["span_id"]]["parentSpanId"] == outer["span_id"] and sent[outer["span_id"]]["parentSpanId"] == ""
    assert {sent[line["span_id"]]["traceId"] for line in (inner, outer)} == {outer["trace_id"]}
    # The event store's rows of the same events, each opening the trace above.
    assert [(row["body"]["stringValue"], row["traceId"], row["spanId"]) for row in collector.log_records()] == [
        (line["event"], line["trace_id"], line["span_id"]) for line in (inner, outer)
    ]


TURN = _event(event="voice.turn")


def _exchanged(reply: Reached | Unreached | Held, span: Span = Span(TURN.trace_id, "1111111111111111", TURN.span_id)) -> Exchanged:
    return Exchanged("x1", SessionId("brain"), MainTurn(None), "POST", "/v1/messages", 2, (), 1000.0, 1000.5, reply, True, span)


def test_a_request_made_for_a_unit_of_work_reaches_the_collector_as_a_span_under_it_and_one_made_for_none_as_a_root(collector: Collector, tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit", clock=datetime.now)
    alone = root()
    with exporting(collector.url, log.record) as record:
        record(_exchanged(Reached(200, 1001.0, 1002.25, 10, Garbled("the stream ended mid-frame"))))
        # A wrapped session's, or the brain's outside a turn: the root of a trace of its own.
        record(_exchanged(Reached(200, 1001.0, 1002.0, 10, Answered({})), span=alone))
    span, rooted = collector.spans()
    assert (span["traceId"], span["spanId"], span["parentSpanId"], span["name"]) == (TURN.trace_id, "1111111111111111", TURN.span_id, "proxy.exchange")
    assert (rooted["traceId"], rooted["spanId"], rooted["parentSpanId"], rooted["name"]) == (alone.trace_id, alone.span_id, "", "proxy.exchange")
    # Which session made it, and what it asked where, as nothing else in a trace of its own says.
    rooted_facts = {attribute["key"]: attribute["value"] for attribute in rooted["attributes"]}
    assert (rooted_facts["facts.session"], rooted_facts["facts.path"]) == ({"stringValue": "brain"}, {"stringValue": "/v1/messages"})
    assert json.loads(rooted_facts["facts.kind"]["stringValue"]) == {"type": "MainTurn", "prompt": None}
    # From the request leaving to the reply's last byte, its first byte half a second in.
    assert int(span["endTimeUnixNano"]) - int(span["startTimeUnixNano"]) == 1_750_000_000
    attributes = {attribute["key"]: attribute["value"] for attribute in span["attributes"]}
    assert (attributes["facts.exchange"], attributes["facts.status"], attributes["facts.first_byte_ms"]) == ({"stringValue": "x1"}, {"intValue": "200"}, {"doubleValue": 500.0})
    # A 200 whose stream broke is failed, as the audit log judges its line an error.
    assert span["status"] == {"code": 2, "message": "the stream ended mid-frame"}


def test_a_request_the_api_refused_or_never_heard_is_a_failed_span_and_one_answered_whole_is_ok() -> None:
    refused, unreached, answered = (
        traced(_exchanged(reply))
        for reply in (Reached(529, 1001.0, 1001.0, 10, Answered({})), Unreached("ClientConnectorError: no route", 1000.75), Reached(200, 1001.0, 1001.0, 10, Answered({})))
    )
    assert refused is not None and (refused.outcome, refused.error) == ("failed", "the API answered 529")
    assert unreached is not None and (unreached.outcome, unreached.error, unreached.duration_ms) == ("failed", "ClientConnectorError: no route", 250.0)
    assert answered is not None and (answered.outcome, answered.error) == ("ok", None)
    # One hands answered itself is answered whole, as the API never saw it.
    held = traced(_exchanged(Held("(stayed silent)", 1000.75)))
    assert held is not None and (held.outcome, held.error, held.duration_ms) == ("ok", None, 250.0)


def test_with_the_collector_stopped_each_event_is_in_the_log_and_said_unexported_by_span_id(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit", clock=datetime.now)
    collector = _stopped()
    with exporting(collector, log.record) as record:
        with unit("delta.read", record):
            pass
    event, *exported = _lines(tmp_path / "audit")
    assert event["type"] == "WideEvent"
    assert sorted((line["signal"], line["type"], line["level"], line["collector"], line["spans"]) for line in exported) == [
        (signal, "Exported", "error", collector, [event["span_id"]]) for signal in ("logs", "traces")
    ]
    assert all("Connection refused" in str(line["error"]) for line in exported)


def _sent(recorded: list[Entry]) -> tuple[WideEvent, Exported, Exported]:
    """The one event recorded, and the batch holding it as it was sent as spans and as log records, in whichever order
    the two signals' threads recorded them."""
    event, *exported = recorded
    assert isinstance(event, WideEvent) and all(isinstance(line, Exported) for line in exported)
    by_signal = {line.signal: line for line in cast(list[Exported], exported)}
    assert len(exported) == 2 and set(by_signal) == {"traces", "logs"}
    return event, by_signal["traces"], by_signal["logs"]


def test_events_the_collector_rejected_are_said_with_its_reason(tmp_path: Path) -> None:
    refusing = Collector(json.dumps({"partialSuccess": {"rejectedSpans": "1", "rejectedLogRecords": "1", "errorMessage": "too old"}}).encode())
    try:
        recorded: list[Entry] = []
        with exporting(refusing.url, recorded.append) as record:
            with unit("delta.read", record):
                pass
    finally:
        refusing.close()
    event, traces, logged = _sent(recorded)
    assert (traces.collector, traces.spans, traces.error) == (refusing.url, (event.span_id,), "the collector rejected 1 of 1 spans: too old")
    assert (logged.spans, logged.error) == ((event.span_id,), "the collector rejected 1 of 1 log records: too old")


def test_a_request_the_collector_refused_is_said_with_the_reason_its_answer_gave() -> None:
    refusing = Collector(b'{"code": 3, "message": "invalid traceId"}', status=400)
    try:
        recorded: list[Entry] = []
        with exporting(refusing.url, recorded.append) as record:
            with unit("delta.read", record):
                pass
    finally:
        refusing.close()
    _, traces, _ = _sent(recorded)
    assert traces.error is not None and "400" in traces.error and "invalid traceId" in traces.error


def test_a_collector_that_takes_the_spans_and_refuses_the_log_records_is_said_to_have_refused_only_those(tmp_path: Path) -> None:
    # A collector with no logs pipeline, which answers its logs path 404.
    refusing = Collector(status=404, refusing=("logs",))
    try:
        log = AuditLog(tmp_path / "audit", clock=datetime.now)
        with exporting(refusing.url, log.record) as record:
            with unit("delta.read", record):
                pass
    finally:
        refusing.close()
    exported = {line["signal"]: line for line in _lines(tmp_path / "audit")[1:]}
    assert (exported["traces"]["level"], exported["traces"]["error"]) == ("info", None)
    assert exported["logs"]["level"] == "error" and "404" in str(exported["logs"]["error"])


def test_a_collector_that_took_every_event_answers_with_no_rejection() -> None:
    assert answered(TRACES, b"", 3) == (None, None)
    assert answered(TRACES, b"{}", 3) == (None, None)
    assert answered(TRACES, b'{"partialSuccess": {}}', 3) == (None, None)
    # A body of another shape than an Export*ServiceResponse rejects nothing; it does not turn a taken batch into a failed one.
    assert answered(TRACES, b"null", 3) == (None, None)
    assert answered(TRACES, b'{"partialSuccess": {"rejectedSpans": ""}}', 3) == (None, None)
    assert answered(TRACES, b'{"partialSuccess": {"rejectedSpans": 2}}', 3) == ("the collector rejected 2 of 3 spans", None)
    # Each signal's response counts what it rejected under its own name, and never under the other's.
    assert answered(LOGS, b'{"partialSuccess": {"rejectedSpans": 2}}', 3) == (None, None)
    assert answered(LOGS, b'{"partialSuccess": {"rejectedLogRecords": "2"}}', 3) == ("the collector rejected 2 of 3 log records", None)


def test_a_collector_that_took_every_event_and_warned_of_something_is_said_to_have_warned() -> None:
    # OTLP's warning: a partial success that rejected nothing, with a message.
    warned = b'{"partialSuccess": {"rejectedLogRecords": "0", "errorMessage": "attribute truncated"}}'
    assert answered(LOGS, warned, 3) == (None, "attribute truncated")
    assert answered(LOGS, b'{"partialSuccess": {"errorMessage": ""}}', 3) == (None, None)
    warning = Collector(warned)
    try:
        recorded: list[Entry] = []
        with exporting(warning.url, recorded.append) as record:
            with unit("delta.read", record):
                pass
    finally:
        warning.close()
    _, traces, logged = _sent(recorded)
    assert (traces.error, traces.warning) == (None, "attribute truncated") and (logged.error, logged.warning) == (None, "attribute truncated")


def test_a_fact_json_has_no_number_for_is_sent_as_proto3_json_spells_it() -> None:
    [span] = _spans(spans([_event(counts={}, facts={"ratio": float("nan"), "ceiling": float("inf"), "floor": float("-inf")})]))
    assert span["attributes"][1:] == [
        {"key": "facts.ratio", "value": {"doubleValue": "NaN"}},
        {"key": "facts.ceiling", "value": {"doubleValue": "Infinity"}},
        {"key": "facts.floor", "value": {"doubleValue": "-Infinity"}},
    ]


def test_a_stop_waits_on_a_collector_that_never_answers_for_one_timeout_and_says_every_batch_it_could_not_send() -> None:
    # A collector that takes the connection and never answers: a host the network has gone dark to.
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen(16)
        recorded: list[Exported] = []
        exporter = Exporter(f"http://127.0.0.1:{silent.getsockname()[1]}", recorded.append, linger=0.01, timeout=0.5)
        # Three batches' worth, each of which alone could wait out the timeout.
        events = [_event(span_id=f"{n:016x}") for n in range(3 * BATCH_SPANS)]
        for event in events:
            exporter.send(event)
        began = time.monotonic()
        exporter.close()
        stopped = time.monotonic() - began
    assert stopped < 1.0
    for signal in ("traces", "logs"):
        assert sorted(span for exported in recorded if exported.signal == signal for span in exported.spans) == sorted(event.span_id for event in events)
    assert all(exported.error is not None for exported in recorded)
    assert any(exported.error == STOPPED for exported in recorded)


def test_with_no_collector_the_log_alone_records_each_event(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit", clock=datetime.now)
    with exporting(None, log.record) as record:
        assert record == log.record


def test_an_event_sent_once_the_exporter_is_closed_is_recorded_unsent(collector: Collector) -> None:
    recorded: list[Exported] = []
    exporter = Exporter(collector.url, recorded.append)
    exporter.close()
    late = _event()
    exporter.send(late)
    assert [(exported.signal, exported.spans, exported.error) for exported in recorded] == [("traces", (late.span_id,), STOPPED), ("logs", (late.span_id,), STOPPED)]


def test_an_error_answer_cut_short_is_said_and_the_batches_after_it_are_still_sent() -> None:
    # A collector, or a proxy before it, that answers an error status and drops the connection inside its body.
    with socket.socket() as cut:
        cut.bind(("127.0.0.1", 0))
        cut.listen(16)

        def answer() -> None:
            while True:
                try:
                    connection, _ = cut.accept()
                except OSError:
                    return
                with connection:
                    # The whole request read first, so the client is reading the answer, not still sending, as it is cut.
                    received = b""
                    while b"\r\n\r\n" not in received:
                        received += connection.recv(1 << 16)
                    head, body = received.split(b"\r\n\r\n", 1)
                    length = int(next(line.split(b":")[1] for line in head.split(b"\r\n") if line.lower().startswith(b"content-length")))
                    while len(body) < length:
                        body += connection.recv(1 << 16)
                    connection.sendall(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 100\r\n\r\ncut")

        threading.Thread(target=answer, daemon=True).start()
        recorded: list[Exported] = []
        exporter = Exporter(f"http://127.0.0.1:{cut.getsockname()[1]}", recorded.append, linger=0.01)
        first, second = _event(span_id="0000000000000001"), _event(span_id="0000000000000002")
        exporter.send(first)
        # Each signal gathers its batch on its own thread, so the second is sent only once both have sent the first alone.
        while {exported.signal for exported in recorded} != {"traces", "logs"}:
            time.sleep(0.01)
        exporter.send(second)
        exporter.close()
    for signal in ("traces", "logs"):
        cut_short, after = (exported for exported in recorded if exported.signal == signal)
        assert (cut_short.spans, after.spans) == ((first.span_id,), (second.span_id,))
        assert cut_short.error is not None and "IncompleteRead" in cut_short.error


def test_a_batch_goes_straight_to_the_collector_past_a_proxy_in_the_environment(collector: Collector, monkeypatch: pytest.MonkeyPatch) -> None:
    # A daemon started inside a session fritter taps holds the tap as its proxy, and the tap is not the way to the collector.
    monkeypatch.setenv("HTTP_PROXY", _stopped())
    monkeypatch.setenv("http_proxy", _stopped())
    recorded: list[Entry] = []
    with exporting(collector.url, recorded.append) as record:
        with unit("delta.read", record):
            pass
    _, traces, logged = _sent(recorded)
    assert traces.error is None and logged.error is None and len(collector.spans()) == len(collector.log_records()) == 1


class _Stuck:
    """A way to the collector on which a send to any of `paths` outlives its timeout, as one waiting on a name that will
    not resolve does, until the test releases it; a send to any other path is taken whole."""

    def __init__(self, *paths: str) -> None:
        self.paths = paths
        self.sending = threading.Event()
        self.released = threading.Event()

    def open(self, request: Request, timeout: float) -> object:
        if not request.full_url.endswith(self.paths):
            return io.BytesIO(b"{}")
        self.sending.set()
        self.released.wait()
        raise TimeoutError("never answered")


def test_a_send_still_waiting_as_the_stop_gives_up_is_said_with_every_event_queued_behind_it(monkeypatch: pytest.MonkeyPatch) -> None:
    stuck_on = _Stuck("/v1/traces", "/v1/logs")
    monkeypatch.setattr(otlp, "_DIRECT", stuck_on)
    recorded: list[Exported] = []
    exporter = Exporter("http://otel.example:4318", recorded.append, linger=0.01, timeout=0.2)
    first, behind = _event(span_id="0000000000000001"), _event(span_id="0000000000000002")
    exporter.send(first)
    stuck_on.sending.wait()
    exporter.send(behind)
    exporter.close()
    stuck_on.released.set()
    stuck = f"{STOPPED}: a send was still waiting on the collector"
    assert sorted((exported.signal, exported.spans, exported.error) for exported in recorded) == [(signal, (first.span_id, behind.span_id), stuck) for signal in ("logs", "traces")]


def test_a_send_stuck_on_the_log_records_holds_up_none_of_the_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    stuck_on = _Stuck("/v1/logs")
    monkeypatch.setattr(otlp, "_DIRECT", stuck_on)
    recorded: list[Exported] = []
    exporter = Exporter("http://otel.example:4318", recorded.append, linger=0.01, timeout=0.2)
    first, behind = _event(span_id="0000000000000001"), _event(span_id="0000000000000002")
    exporter.send(first)
    stuck_on.sending.wait()
    exporter.send(behind)
    exporter.close()
    stuck_on.released.set()
    stuck = f"{STOPPED}: a send was still waiting on the collector"
    # Each span taken, in one batch or two: `behind` may come within the spans' linger after `first`.
    traces = [exported for exported in recorded if exported.signal == "traces"]
    assert [span for exported in traces for span in exported.spans] == [first.span_id, behind.span_id] and {exported.error for exported in traces} == {None}
    assert [(exported.spans, exported.error) for exported in recorded if exported.signal == "logs"] == [((first.span_id, behind.span_id), stuck)]


def _refused(error: str = "HTTPError: HTTP Error 503: Service Unavailable") -> Exported:
    return Exported("http://otel.example:4318", "traces", ("00f067aa0ba902b7",), 3.0, error, None)


def test_a_run_of_untaken_batches_is_folded_from_the_exported_lines_and_a_taken_one_ends_it() -> None:
    at = [STARTED + timedelta(seconds=seconds) for seconds in range(4)]
    streak = None
    for when, exported in zip(at, [_refused(), _refused(), _refused("URLError: Connection refused")]):
        streak = failing(streak, exported, when)
    # Since the first it did not take, whatever each one's error.
    assert streak == Failing(3, at[0])
    assert failing(streak, replace(_refused(), error=None), at[3]) is None


def test_a_signal_is_said_failing_only_once_its_run_is_more_than_a_blip_and_in_words_its_length_does_not_change() -> None:
    def said(batches: int) -> heartbeat.Degradation | None:
        return degradation("http://otel.example:4318", "traces", Failing(batches, STARTED))

    assert said(FAILING_BATCHES - 1) is None
    failing_now = said(FAILING_BATCHES)
    assert failing_now is not None and failing_now.brief == "can't export traces"
    # Since a day and a time, so an outage begun on an earlier day does not read as one begun later today.
    assert failing_now.said == f"the collector at http://otel.example:4318 has not taken all of its traces since {STARTED.astimezone():%Y-%m-%d %H:%M:%S}"
    # The same failure, repeating, is not news again.
    assert said(FAILING_BATCHES + 5) == failing_now


def test_a_collector_that_keeps_refusing_one_signal_is_said_by_the_heartbeat_the_menu_bar_a_notice_and_hands_status_until_it_takes_a_batch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # It takes the log records all along: only the spans are failing.
    refusing = Collector(b'{"code": 14, "message": "the store is down"}', status=503, refusing=("traces",))
    home = Home(tmp_path)
    heart = heartbeat.Heart(home.status, os.getpid(), datetime.now(UTC), heartbeat.HEARTBEAT)
    entries: list[Entry] = []
    exports = Exports(entries.append, lambda: datetime.now(UTC))
    exporter = Exporter(refusing.url, exports.record, linger=0.01)

    def look(before: indicator.Shown | None) -> indicator.Shown:
        heart.beat("running", None, 0, listening=False, degraded=exports.degraded())
        return indicator.show(before, heartbeat.look(home.status, datetime.now(UTC)), datetime.now(UTC))

    def batch(span_id: str) -> None:
        # Once the batch is said as every signal.
        owed = len(entries) + len(ENCODINGS)
        exporter.send(_event(span_id=span_id))
        deadline = time.monotonic() + TIMEOUT_SECONDS * 2
        while len(entries) < owed:
            assert time.monotonic() < deadline, f"no Exported lines for {span_id}"
            time.sleep(0.01)

    try:
        healthy = look(None)
        for n in range(FAILING_BATCHES - 1):
            batch(f"{n:016x}")
        # One refused batch, or two, is a blip: nothing is said.
        blip = look(healthy)
        assert (blip.title, blip.notices) == ("✋", ())
        batch(f"{FAILING_BATCHES:016x}")
        failing_now = look(blip)
        assert failing_now.title == "⚠︎ hands can't export traces"
        assert failing_now.text.startswith(f"hands is up but the collector at {refusing.url} has not taken all of its traces since")
        assert failing_now.notices == (failing_now.text,)
        assert main(["--home", str(tmp_path), "status"]) == 0
        assert capsys.readouterr().out.startswith(f"hands is up but the collector at {refusing.url} has not taken all of its traces since")
        # Refusing on, in other words each time, it stays said and is not news again.
        refusing.answer = b'{"code": 14, "message": "the store is still down"}'
        batch("00000000000000ff")
        still = look(failing_now)
        assert (still.title, still.notices) == (failing_now.title, ())
        # One batch it takes clears it.
        refusing.status = 200
        batch("0000000000000100")
        assert look(still).title == "✋"
    finally:
        exporter.close()
        refusing.close()
    # Why each batch was not taken is said by its own line, never by the heartbeat.
    traces = [entry for entry in entries if isinstance(entry, Exported) and entry.signal == "traces"]
    assert len(entries) == len(ENCODINGS) * len(traces) == len(ENCODINGS) * (FAILING_BATCHES + 2)
    assert "the store is down" in str(traces[0]) and "the store is still down" in str(traces[FAILING_BATCHES])


def test_a_batch_to_a_collector_that_never_answers_waits_one_timeout_for_both_signals() -> None:
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen(16)
        recorded: list[Exported] = []
        exporter = Exporter(f"http://127.0.0.1:{silent.getsockname()[1]}", recorded.append, linger=0.01, timeout=0.4)
        began = time.monotonic()
        exporter.send(_event())
        while len(recorded) < 2:
            time.sleep(0.01)
        waited = time.monotonic() - began
        exporter.close()
    # The two signals wait on the collector side by side: one timeout, not one after the other's.
    assert sorted(exported.signal for exported in recorded) == ["logs", "traces"] and waited < 0.7
