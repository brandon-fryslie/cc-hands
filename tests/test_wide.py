"""Wide events: one per unit of work, however it ends, carrying what the run annotated it with, out through the audit log."""

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import pytest

from hands.sessions.audit import AuditLog, segment, segments
from hands.sessions.otlp import spans
from hands.core.trace import Span
from hands.sessions.wide import WideEvent, annotate, child, continuing, count, fail, here, unit, within


def test_a_unit_that_ends_well_is_one_event_with_its_facts_and_its_counts_zeros_included() -> None:
    emitted: list[WideEvent] = []
    with unit("job", emitted.append, counts=("seen", "failed")):
        annotate(session="s1")
        count(seen=3)
    [event] = emitted
    assert (event.event, event.outcome, event.error) == ("job", "ok", None)
    assert event.counts == {"seen": 3, "failed": 0}
    assert event.facts == {"session": "s1"}
    assert event.duration_ms >= 0 and event.started_at.tzinfo is UTC and len(event.trace_id) == 32


def test_a_unit_that_raises_is_a_failed_event_saying_what_it_raised_and_the_error_goes_on_up() -> None:
    emitted: list[WideEvent] = []
    with pytest.raises(RuntimeError, match="git is on fire"):
        with unit("job", emitted.append):
            annotate(step="reading")
            raise RuntimeError("git is on fire")
    [event] = emitted
    assert (event.outcome, event.error, event.facts) == ("failed", "RuntimeError: git is on fire", {"step": "reading"})
    # As a Failure line does: what was raised, and the frames it came up through.
    assert event.trace[0] == "RuntimeError: git is on fire" and any("in test_a_unit_that_raises" in line for line in event.trace)


async def test_a_unit_cancelled_mid_run_is_a_cancelled_event_and_not_a_failure() -> None:
    emitted: list[WideEvent] = []

    async def endless() -> None:
        with unit("job", emitted.append):
            await asyncio.Event().wait()

    task = asyncio.create_task(endless())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [(event.outcome, event.error) for event in emitted] == [("cancelled", None)]


async def test_a_unit_opened_inside_another_is_its_child_in_one_trace_and_each_annotates_only_its_own_event() -> None:
    emitted: list[WideEvent] = []
    with unit("turn", emitted.append):
        annotate(level="outer")

        async def inner() -> None:
            with unit("reading", emitted.append):
                annotate(level="inner")

        # A task started inside a unit carries it along, as the delta's reading task does.
        await asyncio.create_task(inner())
        annotate(after=True)
    inner_event, outer_event = emitted
    assert inner_event.trace_id == outer_event.trace_id
    assert (inner_event.parent_id, outer_event.parent_id) == (outer_event.span_id, None) and len(outer_event.span_id) == 16
    assert (inner_event.facts, outer_event.facts) == ({"level": "inner"}, {"level": "outer", "after": True})


def test_two_units_one_after_the_other_are_two_traces() -> None:
    emitted: list[WideEvent] = []
    for _ in range(2):
        with unit("job", emitted.append):
            pass
    assert emitted[0].trace_id != emitted[1].trace_id


def test_counting_adds_to_what_the_unit_has_counted_so_far() -> None:
    emitted: list[WideEvent] = []
    with unit("job", emitted.append, counts=("retries",)):
        count(retries=1)
        count(retries=2)
    assert emitted[0].counts == {"retries": 3}


async def test_a_task_that_outlives_its_unit_cannot_change_the_event_already_emitted() -> None:
    emitted: list[WideEvent] = []
    ended = asyncio.Event()

    async def straggler() -> None:
        await ended.wait()
        annotate(late=True)

    with unit("job", emitted.append):
        task = asyncio.create_task(straggler())
    ended.set()
    with pytest.raises(LookupError):
        await task
    assert emitted[0].facts == {}


def test_a_count_the_unit_did_not_declare_is_refused_and_fails_the_unit() -> None:
    emitted: list[WideEvent] = []
    with pytest.raises(KeyError, match="pushed"):
        with unit("job", emitted.append, counts=("seen",)):
            count(pushed=1)
    assert [event.outcome for event in emitted] == ["failed"]


def test_a_unit_that_says_it_failed_ends_failed_with_why_and_runs_to_its_end() -> None:
    emitted: list[WideEvent] = []
    with unit("job", emitted.append):
        fail("the forge refused")
        annotate(after="still ran")
    [event] = emitted
    assert (event.outcome, event.error, event.trace, event.facts) == ("failed", "the forge refused", (), {"after": "still ran"})


def test_a_part_timed_elsewhere_is_its_own_event_under_the_unit_in_its_trace() -> None:
    emitted: list[WideEvent] = []
    at = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)
    with unit("turn", emitted.append):
        span = within(here())
        child("tool.call", span, at, 700.0, "ok", call="t1")
    part, turn = emitted
    assert (part.event, part.started_at, part.duration_ms, part.outcome, part.facts) == ("tool.call", at, 700.0, "ok", {"call": "t1"})
    assert (part.trace_id, part.span_id, part.parent_id) == (turn.trace_id, span.span_id, turn.span_id) and part.span_id != turn.span_id


def test_a_part_minted_under_another_unit_is_refused_rather_than_written_into_a_trace_it_is_no_part_of() -> None:
    emitted: list[WideEvent] = []
    with unit("turn", emitted.append):
        earlier = within(here())
    with unit("turn", emitted.append), pytest.raises(LookupError, match="tool.call is no part of the unit of work open here"):
        child("tool.call", earlier, datetime.now(UTC), 1.0, "ok")
    assert [event.event for event in emitted] == ["turn", "turn"]


def test_a_unit_opened_for_a_span_begun_elsewhere_continues_its_trace_and_one_for_none_is_a_root() -> None:
    emitted: list[WideEvent] = []
    with unit("turn", emitted.append):
        part = within(here())
    with continuing(part):
        with unit("tool.run", emitted.append):
            pass
    with unit("turn", emitted.append), continuing(None):
        with unit("tool.run", emitted.append):
            pass
    _, ran, rooted, _ = emitted
    assert (ran.trace_id, ran.parent_id) == (part.trace_id, part.span_id) and ran.span_id != part.span_id
    assert rooted.parent_id is None and rooted.trace_id != part.trace_id


def test_a_fact_inside_a_continued_span_with_no_unit_open_is_refused() -> None:
    with continuing(Span("t" * 32, "s" * 16, None)), pytest.raises(LookupError):
        annotate(session="s1")


def test_a_unit_hands_its_span_to_a_part_of_it_that_runs_where_it_is_not_open() -> None:
    emitted: list[WideEvent] = []
    with unit("turn", emitted.append):
        with unit("step", emitted.append):
            step = here()
        turn = here()
    inside = within(turn)
    stepped, turned = emitted
    assert (step.trace_id, step.span_id, step.parent_id) == (stepped.trace_id, stepped.span_id, turned.span_id)
    assert (turn.span_id, turn.parent_id) == (turned.span_id, None)
    assert (inside.trace_id, inside.parent_id) == (turned.trace_id, turned.span_id) and inside.span_id not in (turned.span_id, stepped.span_id)


def test_a_fact_with_no_unit_open_to_land_on_is_refused() -> None:
    with pytest.raises(LookupError):
        annotate(session="s1")
    with pytest.raises(LookupError):
        count(seen=1)


def test_an_event_is_an_audit_line_with_its_start_written_as_a_time_and_a_failed_one_is_an_error(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit", clock=lambda: datetime(2026, 10, 3, tzinfo=UTC))
    with unit("job", log.record, counts=("seen",)):
        annotate(session="s1")
    with pytest.raises(ValueError):
        with unit("job", log.record):
            raise ValueError("bad")
    written = [json.loads(line) for base in segments(tmp_path / "audit") for line in segment(tmp_path / "audit", base).read_text().splitlines()]
    assert [(line["type"], line["level"], line["outcome"]) for line in written] == [("WideEvent", "info", "ok"), ("WideEvent", "error", "failed")]
    assert written[0]["counts"] == {"seen": 0} and written[0]["facts"] == {"session": "s1"}
    assert datetime.fromisoformat(written[0]["started_at"]).tzinfo is not None


class Reading(StrEnum):
    READ = "read"


@dataclass(frozen=True)
class Pushed:
    branch: str


def test_every_kind_of_fact_is_written_on_its_audit_line_and_carried_by_its_otlp_span(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit", clock=lambda: datetime(2026, 10, 3, tzinfo=UTC))
    emitted: list[WideEvent] = []

    def record(event: WideEvent) -> None:
        log.record(event)
        emitted.append(event)

    at = datetime(2026, 10, 3, 1, 2, 3, tzinfo=UTC)
    with unit("job", record):
        annotate(none=None, flag=True, n=3, ms=1.5, word="s1", at=at, path=Path("/x"), reading=Reading.READ, change=Pushed("fix"))
        annotate(changes=(Pushed("a"), 2), seen=frozenset({"b", "a"}))
    [line] = [json.loads(line) for base in segments(tmp_path / "audit") for line in segment(tmp_path / "audit", base).read_text().splitlines()]
    pushed = {"type": "Pushed", "branch": "fix"}
    assert line["facts"] == {
        "none": None, "flag": True, "n": 3, "ms": 1.5, "word": "s1", "at": "2026-10-03T01:02:03.000+00:00", "path": "/x", "reading": "read",
        "change": pushed, "changes": [{"type": "Pushed", "branch": "a"}, 2], "seen": ["a", "b"],
    }
    [span] = json.loads(json.dumps(spans(emitted)))["resourceSpans"][0]["scopeSpans"][0]["spans"]
    carried = {attribute["key"]: attribute["value"] for attribute in span["attributes"]}
    # [LAW:one-source-of-truth] a scalar on the line is the same scalar on the span, and anything else is the line's JSON.
    assert {key.removeprefix("facts."): value for key, value in carried.items() if key.startswith("facts.")} == {
        "none": {}, "flag": {"boolValue": True}, "n": {"intValue": "3"}, "ms": {"doubleValue": 1.5}, "word": {"stringValue": "s1"},
        "at": {"stringValue": "2026-10-03T01:02:03.000+00:00"}, "path": {"stringValue": "/x"}, "reading": {"stringValue": "read"},
        "change": {"stringValue": json.dumps(pushed)}, "changes": {"stringValue": json.dumps(line["facts"]["changes"])}, "seen": {"stringValue": '["a", "b"]'},
    }


def test_a_fact_neither_the_audit_log_nor_otlp_can_carry_is_refused_by_pyright_where_it_is_annotated(tmp_path: Path) -> None:
    annotated = tmp_path / "annotated.py"
    annotated.write_text(
        "import socket\n"
        "from hands.sessions.wide import annotate, child, here, within\n"
        "from datetime import UTC, datetime\n"
        "annotate(n=1, words=('a', 'b'), pairs=frozenset({(None, 1.5)}))\n"
        "annotate(socket=socket.socket())\n"
        "annotate(listed=[1])\n"
        "annotate(by={'a': 1})\n"
        "child('part', within(here()), datetime.now(UTC), 1.0, 'ok', handler=print)\n"
        "from enum import Enum\n"
        "class Held(Enum):\n"
        "    SOCKET = socket.socket()\n"
        "annotate(held=Held.SOCKET)\n"
    )
    checked = subprocess.run([sys.executable, "-m", "pyright", "--outputjson", str(annotated)], capture_output=True, text=True, cwd=Path(__file__).parents[1])
    assert checked.stdout, checked.stderr
    diagnostics = json.loads(checked.stdout)["generalDiagnostics"]
    refused = [(diagnostic["range"]["start"]["line"] + 1, diagnostic["rule"], '"Fact"' in diagnostic["message"]) for diagnostic in diagnostics]
    assert refused == [(line, "reportArgumentType", True) for line in (5, 6, 7, 8, 12)]
