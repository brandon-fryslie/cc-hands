"""The OTLP export edge: each wide event goes to the audit log, and to the collector where one is set, as a span; a batch
the collector did not take is said in the log, by span id."""

import json
import socket
import threading
from collections.abc import Generator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest

from hands.sessions.audit import AuditLog, Entry, Undelivered, segment, segments
from hands.sessions.otlp import exporting, rejected, spans
from hands.sessions.wide import WideEvent, annotate, count, unit

STARTED = datetime(2026, 10, 3, 12, 0, 0, 250_000, tzinfo=UTC)


class Collector:
    """A stand-in for the collector's OTLP/HTTP receiver: it keeps every request body posted to /v1/traces and answers
    with `answer`."""

    def __init__(self, answer: bytes = b"{}") -> None:
        self.bodies: list[dict[str, object]] = []
        self.paths: list[str] = []
        received = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                received.paths.append(self.path)
                received.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(answer)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def spans(self) -> list[dict[str, Any]]:
        return [span for body in self.bodies for span in _spans(body)]


def _spans(request: dict[str, object]) -> list[dict[str, Any]]:
    """The spans of an ExportTraceServiceRequest, as JSON reads them."""
    resources = cast(list[dict[str, Any]], request["resourceSpans"])
    return [span for resource in resources for scope in resource["scopeSpans"] for span in scope["spans"]]


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


def test_with_a_collector_each_event_is_in_the_log_and_reaches_the_collector_and_a_unit_inside_another_is_its_child(collector: Collector, tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit", clock=datetime.now)
    with exporting(collector.url, log.record) as record:
        with unit("turn", record):
            with unit("delta.read", record, counts=("commits",)):
                annotate(session="s1")
                count(commits=1)
    lines = _lines(tmp_path / "audit")
    assert [line["type"] for line in lines] == ["WideEvent", "WideEvent"]
    inner, outer = lines
    assert collector.paths == ["/v1/traces"]
    sent = {span["spanId"]: span for span in collector.spans()}
    assert set(sent) == {inner["span_id"], outer["span_id"]}
    assert sent[inner["span_id"]]["parentSpanId"] == outer["span_id"] and sent[outer["span_id"]]["parentSpanId"] == ""
    assert {sent[line["span_id"]]["traceId"] for line in lines} == {outer["trace_id"]}


def test_with_the_collector_stopped_each_event_is_in_the_log_and_said_undelivered_by_span_id(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit", clock=datetime.now)
    collector = _stopped()
    with exporting(collector, log.record) as record:
        with unit("delta.read", record):
            pass
    event, undelivered = _lines(tmp_path / "audit")
    assert event["type"] == "WideEvent"
    assert (undelivered["type"], undelivered["level"], undelivered["collector"], undelivered["spans"]) == ("Undelivered", "error", collector, [event["span_id"]])
    assert "Connection refused" in str(undelivered["error"])


def test_spans_the_collector_rejected_are_said_undelivered_with_its_reason(tmp_path: Path) -> None:
    refusing = Collector(json.dumps({"partialSuccess": {"rejectedSpans": "1", "errorMessage": "span too old"}}).encode())
    try:
        recorded: list[Entry] = []
        with exporting(refusing.url, recorded.append) as record:
            with unit("delta.read", record):
                pass
    finally:
        refusing.close()
    event, undelivered = recorded
    assert isinstance(event, WideEvent)
    assert undelivered == Undelivered(refusing.url, (event.span_id,), "the collector rejected 1 of 1 spans: span too old")


def test_a_collector_that_took_every_span_answers_with_no_rejection() -> None:
    assert rejected(b"", 3) is None
    assert rejected(b"{}", 3) is None
    assert rejected(b'{"partialSuccess": {}}', 3) is None
    assert rejected(b'{"partialSuccess": {"rejectedSpans": 2}}', 3) == "the collector rejected 2 of 3 spans"


def test_with_no_collector_the_log_alone_records_each_event(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit", clock=datetime.now)
    with exporting(None, log.record) as record:
        assert record == log.record
