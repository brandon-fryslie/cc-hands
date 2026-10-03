"""Wide events: one per unit of work, however it ends, carrying what the run annotated it with, out through the audit log."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hands.sessions.audit import AuditLog, segment, segments
from hands.sessions.wide import WideEvent, annotate, count, unit


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


async def test_a_unit_opened_inside_another_shares_its_trace_and_each_annotates_only_its_own_event() -> None:
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
    assert (inner_event.facts, outer_event.facts) == ({"level": "inner"}, {"level": "outer", "after": True})


def test_two_units_one_after_the_other_are_two_traces() -> None:
    emitted: list[WideEvent] = []
    for _ in range(2):
        with unit("job", emitted.append):
            pass
    assert emitted[0].trace_id != emitted[1].trace_id


def test_a_count_the_unit_did_not_declare_is_refused_and_fails_the_unit() -> None:
    emitted: list[WideEvent] = []
    with pytest.raises(KeyError, match="pushed"):
        with unit("job", emitted.append, counts=("seen",)):
            count(pushed=1)
    assert [event.outcome for event in emitted] == ["failed"]


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
