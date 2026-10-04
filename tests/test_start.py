"""A start of hands is one wide event, `hands.start`: ready, refused, or told to stop first, with what it learned on the way."""

import asyncio
import os
from pathlib import Path

import pytest

from hands.daemon.starting import CannotStart, Start
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent


def events(recorded: list[Entry]) -> list[WideEvent]:
    return [entry for entry in recorded if isinstance(entry, WideEvent)]


def test_a_start_that_comes_up_is_one_event_with_which_run_it_is_and_what_it_learned() -> None:
    run_start = Start(restarted=True, after_crash=False)
    run_start.heard(proxy="http://127.0.0.1:5000", tap=Path("/home/wire.sock"))
    run_start.heard(voice="charles")
    recorded: list[Entry] = []
    run_start.ended(recorded.append, None)
    [event] = events(recorded)
    assert (event.event, event.outcome, event.error, event.parent_id) == ("hands.start", "ok", None, None)
    assert dict(event.facts) == {"pid": os.getpid(), "restarted": True, "after_crash": False, "proxy": "http://127.0.0.1:5000", "tap": Path("/home/wire.sock"), "voice": "charles"}
    assert event.duration_ms >= 0 and len(recorded) == 1


def test_a_refused_start_is_failed_with_its_reason_and_says_how_far_it_got() -> None:
    run_start = Start(restarted=False, after_crash=True)
    run_start.heard(proxy="http://127.0.0.1:5000")
    recorded: list[Entry] = []
    with pytest.raises(CannotStart, match="no key"), run_start.ending(recorded.append):
        raise CannotStart("no key")
    [event] = events(recorded)
    assert (event.outcome, event.error) == ("failed", "CannotStart: no key")
    assert event.facts["proxy"] == "http://127.0.0.1:5000" and "voice" not in event.facts


def test_a_failed_start_traces_only_the_frames_its_error_came_up_through() -> None:
    # Written down, never raised again to be written: no frame of the event's own making is in its trace.
    run_start = Start(restarted=False, after_crash=False)
    recorded: list[Entry] = []
    with pytest.raises(ValueError, match="x"), run_start.ending(recorded.append):
        raise ValueError("x")
    [event] = events(recorded)
    assert event.trace[0] == "ValueError: x"
    assert not [line for line in event.trace if "wide.py" in line or line.endswith(" in ended")]


def test_a_start_told_to_stop_before_it_was_ready_is_cancelled() -> None:
    run_start = Start(restarted=False, after_crash=False)
    recorded: list[Entry] = []
    with run_start.ending(recorded.append):
        pass
    assert [(event.outcome, event.error) for event in events(recorded)] == [("cancelled", None)]


def test_a_run_that_ends_after_its_start_was_ready_leaves_the_start_as_it_was() -> None:
    run_start = Start(restarted=False, after_crash=False)
    recorded: list[Entry] = []
    with pytest.raises(RuntimeError, match="the pipeline ended"), run_start.ending(recorded.append):
        run_start.ended(recorded.append, None)
        raise RuntimeError("the pipeline ended without being told to stop")
    assert [event.outcome for event in events(recorded)] == ["ok"]


async def test_a_stop_while_the_start_awaits_is_cancelled_not_failed() -> None:
    run_start = Start(restarted=False, after_crash=False)
    recorded: list[Entry] = []

    async def starting() -> None:
        with run_start.ending(recorded.append):
            await asyncio.Event().wait()

    task = asyncio.create_task(starting())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [event.outcome for event in events(recorded)] == ["cancelled"]


def test_a_start_ends_once_and_learns_nothing_after() -> None:
    run_start = Start(restarted=False, after_crash=False)
    run_start.ended(lambda _entry: None, None)
    with pytest.raises(RuntimeError, match="already ended"):
        run_start.ended(lambda _entry: None, None)
    with pytest.raises(LookupError, match="came too late"):
        run_start.heard(voice="charles")
