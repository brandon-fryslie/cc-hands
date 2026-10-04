"""The OTLP export edge: each wide event goes to the audit log, and to the collector where one is set, as a span; each
batch sent is said in the log, by span id, with why where the collector did not take it."""

import json
import socket
import threading
import time
from collections.abc import Generator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest

from hands.core.session import SessionId
from hands.core.trace import Span
from hands.core.wire import Answered, Exchanged, Garbled, Held, MainTurn, Reached, Unreached
from hands.sessions.audit import AuditLog, Entry, Exported, segment, segments
from hands.sessions import otlp
from hands.sessions.otlp import BATCH_SPANS, STOPPED, Exporter, exporting, rejected, spans, traced
from hands.sessions.wide import WideEvent, annotate, count, unit

STARTED = datetime(2026, 10, 3, 12, 0, 0, 250_000, tzinfo=UTC)


class Collector:
    """A stand-in for the collector's OTLP/HTTP receiver: it keeps every request body posted to /v1/traces and answers
    with `answer`."""

    def __init__(self, answer: bytes = b"{}", status: int = 200) -> None:
        self.bodies: list[dict[str, object]] = []
        self.paths: list[str] = []
        received = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                received.paths.append(self.path)
                received.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(status)
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
    assert [line["type"] for line in lines] == ["WideEvent", "WideEvent", "Exported"]
    inner, outer, exported = lines
    # A batch the collector took is said too, so a run whose events all reached it is told from one that sent none.
    assert (exported["level"], exported["collector"], exported["spans"], exported["error"]) == ("info", collector.url, [inner["span_id"], outer["span_id"]], None)
    assert collector.paths == ["/v1/traces"]
    sent = {span["spanId"]: span for span in collector.spans()}
    assert set(sent) == {inner["span_id"], outer["span_id"]}
    assert sent[inner["span_id"]]["parentSpanId"] == outer["span_id"] and sent[outer["span_id"]]["parentSpanId"] == ""
    assert {sent[line["span_id"]]["traceId"] for line in (inner, outer)} == {outer["trace_id"]}


TURN = _event(event="voice.turn")


def _exchanged(reply: Reached | Unreached | Held, span: Span | None = Span(TURN.trace_id, "1111111111111111", TURN.span_id)) -> Exchanged:
    return Exchanged("x1", SessionId("brain"), MainTurn(None), "POST", "/v1/messages", 2, (), 1000.0, 1000.5, reply, True, span)


def test_a_request_made_for_a_unit_of_work_reaches_the_collector_as_a_span_under_it(collector: Collector, tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit", clock=datetime.now)
    with exporting(collector.url, log.record) as record:
        record(_exchanged(Reached(200, 1001.0, 1002.25, 10, Garbled("the stream ended mid-frame"))))
        # One made for no unit of work, and one hands answered itself, are in no trace.
        record(_exchanged(Reached(200, 1001.0, 1002.0, 10, Answered({})), span=None))
    [span] = collector.spans()
    assert (span["traceId"], span["spanId"], span["parentSpanId"], span["name"]) == (TURN.trace_id, "1111111111111111", TURN.span_id, "proxy.exchange")
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
    assert traced(_exchanged(Held("(stayed silent)", 1000.5), span=None)) is None


def test_with_the_collector_stopped_each_event_is_in_the_log_and_said_unexported_by_span_id(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit", clock=datetime.now)
    collector = _stopped()
    with exporting(collector, log.record) as record:
        with unit("delta.read", record):
            pass
    event, exported = _lines(tmp_path / "audit")
    assert event["type"] == "WideEvent"
    assert (exported["type"], exported["level"], exported["collector"], exported["spans"]) == ("Exported", "error", collector, [event["span_id"]])
    assert "Connection refused" in str(exported["error"])


def _sent(recorded: list[Entry]) -> tuple[WideEvent, Exported]:
    event, exported = recorded
    assert isinstance(event, WideEvent) and isinstance(exported, Exported)
    return event, exported


def test_spans_the_collector_rejected_are_said_with_its_reason(tmp_path: Path) -> None:
    refusing = Collector(json.dumps({"partialSuccess": {"rejectedSpans": "1", "errorMessage": "span too old"}}).encode())
    try:
        recorded: list[Entry] = []
        with exporting(refusing.url, recorded.append) as record:
            with unit("delta.read", record):
                pass
    finally:
        refusing.close()
    event, exported = _sent(recorded)
    assert (exported.collector, exported.spans, exported.error) == (refusing.url, (event.span_id,), "the collector rejected 1 of 1 spans: span too old")


def test_a_request_the_collector_refused_is_said_with_the_reason_its_answer_gave() -> None:
    refusing = Collector(b'{"code": 3, "message": "invalid traceId"}', status=400)
    try:
        recorded: list[Entry] = []
        with exporting(refusing.url, recorded.append) as record:
            with unit("delta.read", record):
                pass
    finally:
        refusing.close()
    _, exported = _sent(recorded)
    assert exported.error is not None and "400" in exported.error and "invalid traceId" in exported.error


def test_a_collector_that_took_every_span_answers_with_no_rejection() -> None:
    assert rejected(b"", 3) is None
    assert rejected(b"{}", 3) is None
    assert rejected(b'{"partialSuccess": {}}', 3) is None
    # A body of another shape than an ExportTraceServiceResponse rejects nothing; it does not turn a taken batch into a failed one.
    assert rejected(b"null", 3) is None
    assert rejected(b'{"partialSuccess": {"rejectedSpans": ""}}', 3) is None
    assert rejected(b'{"partialSuccess": {"rejectedSpans": 2}}', 3) == "the collector rejected 2 of 3 spans"


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
    assert sorted(span for exported in recorded for span in exported.spans) == sorted(event.span_id for event in events)
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
    assert [(exported.spans, exported.error) for exported in recorded] == [((late.span_id,), STOPPED)]


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
        while not recorded:
            time.sleep(0.01)
        exporter.send(second)
        exporter.close()
    assert [exported.spans for exported in recorded] == [(first.span_id,), (second.span_id,)]
    assert recorded[0].error is not None and "IncompleteRead" in recorded[0].error


def test_a_batch_goes_straight_to_the_collector_past_a_proxy_in_the_environment(collector: Collector, monkeypatch: pytest.MonkeyPatch) -> None:
    # A daemon started inside a session fritter taps holds the tap as its proxy, and the tap is not the way to the collector.
    monkeypatch.setenv("HTTP_PROXY", _stopped())
    monkeypatch.setenv("http_proxy", _stopped())
    recorded: list[Entry] = []
    with exporting(collector.url, recorded.append) as record:
        with unit("delta.read", record):
            pass
    _, exported = _sent(recorded)
    assert exported.error is None and len(collector.spans()) == 1


def test_a_send_still_waiting_as_the_stop_gives_up_is_said_with_every_event_queued_behind_it(monkeypatch: pytest.MonkeyPatch) -> None:
    # A send that outlives its timeout, as one waiting on a name that will not resolve does.
    sending = threading.Event()

    class Stuck:
        def open(self, request: object, timeout: float) -> object:
            sending.set()
            time.sleep(3)
            raise TimeoutError("never answered")

    monkeypatch.setattr(otlp, "_DIRECT", Stuck())
    recorded: list[Exported] = []
    exporter = Exporter("http://otel.example:4318", recorded.append, linger=0.01, timeout=0.2)
    first, behind = _event(span_id="0000000000000001"), _event(span_id="0000000000000002")
    exporter.send(first)
    sending.wait()
    exporter.send(behind)
    exporter.close()
    assert [(exported.spans, exported.error) for exported in recorded] == [((first.span_id, behind.span_id), f"{STOPPED}: a send was still waiting on the collector")]
