"""The audit log: what is written, how it reads back, and what becomes a line when writing or encoding fails."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from loguru import logger
from pipecat.utils.errors import ErrorCategory

from hands.core.effects import Holding, Reply, Unmatched, Withdraw
from hands.core.events import Abandoned, Closed, Joined, Prompted, Read, StatusReported, Stopped, Tick
from hands.core.session import Membership, PromptId, RequestId, SessionId, Told
from hands.daemon import cli
from hands.sessions.audit import (
    Applied,
    AuditLog,
    BrainSpoke,
    Called,
    Entry,
    Failure,
    Performed,
    START,
    Replied,
    Transcribed,
    encoded,
    failures_to,
    follow,
    tail,
)
from hands.sessions.home import Home
from hands.sessions.model_facts import ModelFailed
from hands.sessions.registry import Sessions
from hands.voice.tools import Result, Tool, audited, draft_tools, tool
from hands.core.status import Busy, Report, Stamp

# When hands heard a Stop, on the clock Claude Code stamps a status with.
STOP_HEARD = Stamp(1500)
STOP_REQUEST = RequestId("stop")

AT = datetime(2026, 9, 14, 12, 0, 0, 123000, tzinfo=UTC)


def member() -> Membership:
    return Membership(SessionId("s1"), pid=4242, cwd=Path("/code/cc-hands"), transcript=Path("/nowhere/s1.jsonl"))


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


async def invoke(tool: Tool, **arguments: object) -> object:
    return await tool.body(**arguments)


def test_an_entry_is_its_type_and_fields_nested_values_alike() -> None:
    assert encoded(Performed(Reply(SessionId("s1"), RequestId("r1"), Withdraw()))) == {
        "type": "Performed",
        "effect": {"type": "Reply", "session": "s1", "request": "r1", "reply": {"type": "Withdraw"}},
    }
    assert encoded(Applied(Joined(member(), "startup")))["event"] == {
        "type": "Joined",
        # fritter is the socket to type into this session, and null for a session nobody
        # wrapped; it is in the log because "why could hands not type into that one" is
        # a question the log should be able to answer.
        "membership": {"type": "Membership", "id": "s1", "pid": 4242, "cwd": "/code/cc-hands", "transcript": "/nowhere/s1.jsonl", "fritter": None},
        "source": "startup",
    }


def test_a_value_the_log_cannot_write_is_refused_rather_than_guessed_at() -> None:
    with pytest.raises(TypeError, match="cannot encode a set"):
        encoded(Called("t", {"odd": {1, 2}}, None))


def test_the_log_is_the_user_s_alone_to_read_whether_it_is_new_or_was_there(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh.jsonl"
    AuditLog(fresh, clock=lambda: AT).record(Transcribed("send it"))
    there = tmp_path / "there.jsonl"
    there.write_text("")
    there.chmod(0o644)
    AuditLog(there, clock=lambda: AT)
    assert [path.stat().st_mode & 0o777 for path in (fresh, there)] == [0o600, 0o600]


def test_each_entry_is_one_line_stamped_with_when_it_was_written(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "deep" / "audit.jsonl", clock=lambda: AT)
    log.record(Transcribed("send it"))
    log.record(Replied("Sent to cc-hands.", interrupted=False))
    assert lines(tmp_path / "deep" / "audit.jsonl") == [
        {"at": "2026-09-14T12:00:00.123+00:00", "type": "Transcribed", "text": "send it"},
        {"at": "2026-09-14T12:00:00.123+00:00", "type": "Replied", "text": "Sent to cc-hands.", "interrupted": False},
    ]


def test_the_tail_is_the_newest_whole_lines_and_following_picks_up_where_it_ended(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    assert tail(path, 5) == ([], START)
    path.write_text("one\ntwo\nthree\npart")
    newest, position = tail(path, 2)
    assert newest == ["two", "three"]

    looks: list[str] = []
    followed = follow(path, position, lambda: looks.append("look"))
    with path.open("a") as log:
        log.write("ial\nfour\n")
    assert [next(followed), next(followed)] == ["partial", "four"]
    # Cut short in place: the log is read again from its first line.
    path.write_text("fresh\n")
    assert next(followed) == "fresh"


def test_a_log_moved_aside_is_followed_from_the_first_line_of_the_new_one_even_when_it_has_grown_past_the_old_offset(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text("old\n")
    _, position = tail(path, 1)
    followed = follow(path, position, lambda: None)
    path.rename(tmp_path / "audit.jsonl.1")
    # Longer than the old log, and the old offset falls inside a two-byte character.
    path.write_text("\u00e9t\u00e9\nsecond\n")
    assert [next(followed), next(followed)] == ["\u00e9t\u00e9", "second"]


def test_a_line_the_disk_will_not_take_is_lost_out_loud_and_the_daemon_carries_on(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path, clock=lambda: AT)
    path.mkdir()  # opening a directory to append fails as a full or read-only disk does
    warnings: list[str] = []
    sink = logger.add(lambda message: warnings.append(message.record["message"]), level="WARNING")
    try:
        log.record(Transcribed("send it"))
    finally:
        logger.remove(sink)
    [warning] = warnings
    assert warning.startswith(f"the audit log {path} lost a Transcribed line: ")


def test_following_stops_at_ctrl_c_after_printing_the_newest_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path)
    home.audit.write_text("a\nb\nc\n")

    def interrupted(_: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", interrupted)
    assert cli.main(["--home", str(tmp_path), "log", "-n", "2"]) == 0
    assert capsys.readouterr().out == "b\nc\n"


def test_an_error_logged_anywhere_is_a_failure_line_with_its_exception() -> None:
    recorded: list[Entry] = []
    sink = logger.add(failures_to(recorded.append), level="ERROR")
    try:
        logger.info("not a failure")
        try:
            raise OSError("no space left")
        except OSError:
            logger.exception("cannot write the heartbeat")
    finally:
        logger.remove(sink)
    assert recorded == [Failure(source="test_audit:test_an_error_logged_anywhere_is_a_failure_line_with_its_exception", message="cannot write the heartbeat: OSError: no space left")]


async def test_an_event_that_changed_nothing_is_not_a_line_and_one_that_did_is() -> None:
    recorded: list[Entry] = []
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=recorded.append)
    await sessions.apply(Tick(1.0))
    assert recorded == []
    await sessions.apply(Joined(member(), "startup"))
    assert recorded == [Applied(Joined(member(), "startup"))]


async def test_a_stop_that_ends_no_turn_is_a_line_saying_so() -> None:
    recorded: list[Entry] = []
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=recorded.append)
    await sessions.apply(Joined(member(), "startup"))
    stop = Stopped(member().id, "done", mode=None, prompt=PromptId("p1"), again=False, heard=STOP_HEARD, request=STOP_REQUEST)
    await sessions.apply(stop)
    assert Unmatched(stop.session, stop.prompt) not in recorded
    # Heard again once its turn was told, as an interrupted turn's late Stop is.
    await sessions.apply(stop)
    assert recorded[-2:] == [Unmatched(stop.session, stop.prompt), Performed(Reply(stop.session, stop.request, Withdraw()))]


async def test_a_stop_held_for_its_record_is_a_line_as_it_is_heard_and_again_once_read_through_without_one() -> None:
    recorded: list[Entry] = []
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=recorded.append)
    await sessions.apply(Joined(member(), "startup"))
    await sessions.apply(Prompted(member().id, at=1.0, mode=None, prompt=PromptId("p1")))
    # Its transcript is read once a status is, so a record could name the Stop.
    await sessions.apply(StatusReported(member().id, Report(Busy(), Stamp(1000)), at=1.0))
    stop = Stopped(member().id, "done", mode=None, prompt=PromptId("p2"), again=False, heard=STOP_HEARD, request=STOP_REQUEST)
    await sessions.apply(stop)
    assert recorded[-2:] == [Applied(stop), Holding(stop.session, stop.prompt)]
    await sessions.apply(Read(member().id, Stamp(STOP_HEARD + 999)))
    assert Unmatched(stop.session, stop.prompt) not in recorded
    await sessions.apply(Read(member().id, Stamp(STOP_HEARD + 1000)))
    # Its hook is let go only once it is decided.
    assert recorded[-2:] == [Unmatched(stop.session, stop.prompt), Performed(Reply(stop.session, stop.request, Withdraw()))]


async def test_a_stop_hook_is_answered_only_once_its_stop_is_decided() -> None:
    """Claude Code waits on the hook, so what deciding the Stop calls for is done before it goes on."""
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(member(), "startup"))
    await sessions.apply(Prompted(member().id, at=1.0, mode=None, prompt=PromptId("p1")))
    # Its transcript is read once a status is, so a record could name the Stop.
    await sessions.apply(StatusReported(member().id, Report(Busy(), Stamp(1000)), at=1.0))
    hook = asyncio.create_task(sessions.stop(Stopped(member().id, "done", mode=None, prompt=PromptId("p2"), again=False, heard=STOP_HEARD, request=STOP_REQUEST)))
    await asyncio.sleep(0)
    assert not hook.done()
    await sessions.apply(Read(member().id, Stamp(STOP_HEARD + 1000)))
    await asyncio.wait_for(hook, 1.0)


async def test_a_stop_hook_is_let_go_once_the_hold_passes_and_the_stop_decided_later_answers_no_hook() -> None:
    """The hold bounds how long Claude Code waits, however slowly the transcript is read."""
    recorded: list[Entry] = []
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=recorded.append, stop_hold=0.01)
    await sessions.apply(Joined(member(), "startup"))
    await sessions.apply(Prompted(member().id, at=1.0, mode=None, prompt=PromptId("p1")))
    await sessions.apply(StatusReported(member().id, Report(Busy(), Stamp(1000)), at=1.0))
    stop = Stopped(member().id, "done", mode=None, prompt=PromptId("p2"), again=False, heard=STOP_HEARD, request=STOP_REQUEST)
    await asyncio.wait_for(sessions.stop(stop), 1.0)
    assert recorded[-1] == Applied(Abandoned(stop.session, stop.request, 0.0))
    await sessions.apply(Read(member().id, Stamp(STOP_HEARD + 1000)))
    assert recorded[-1] == Unmatched(stop.session, stop.prompt)


async def test_a_stop_heard_as_the_daemon_shuts_down_is_still_applied() -> None:
    """Letting every hook go at shutdown lets go of its wait, not of what it says happened."""
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(member(), "startup"))
    await sessions.apply(Prompted(member().id, at=1.0, mode=None, prompt=PromptId("p1")))
    sessions.release_waiting()
    await asyncio.wait_for(sessions.stop(Stopped(member().id, "done", mode=None, prompt=PromptId("p1"), again=False, heard=STOP_HEARD, request=STOP_REQUEST)), 3.0)
    live = sessions.live_session(member().id)
    assert live is not None and live.turn == Told(PromptId("p1"))


async def test_an_audited_tool_keeps_its_schema_and_writes_its_call_beside_its_result() -> None:
    recorded: list[Entry] = []
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    [stage, *_] = draft_tools(sessions)
    wrapped = audited(stage, recorded.append)
    assert (wrapped.name, wrapped.description, wrapped.input_schema, wrapped.completes) == (stage.name, stage.description, stage.input_schema, stage.completes)
    result = await invoke(wrapped, session="nobody", text="hi", resolutions=[])
    assert recorded == [Called("stage_draft", {"session": "nobody", "text": "hi", "resolutions": []}, result)]


async def test_a_tool_that_raises_is_a_failure_line_naming_it_and_its_arguments() -> None:
    recorded: list[Entry] = []

    async def broken(session: str) -> Result:
        """Fail."""
        raise RuntimeError("the transcript went away")

    sink = logger.add(failures_to(recorded.append), level="ERROR", filter="hands")
    try:
        with pytest.raises(RuntimeError):
            await invoke(audited(tool(broken), recorded.append), session="s1")
    finally:
        logger.remove(sink)
    assert recorded == [Failure(source="hands.voice.tools:call", message="the tool broken raised, called with {'session': 's1'}: RuntimeError: the transcript went away")]


def test_hands_log_piped_into_a_reader_that_stops_ends_quietly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    home.audit.write_text("a\n")

    def closed(*_: object, **__: object) -> None:
        raise BrokenPipeError

    def discarded(*_: object) -> None:
        pass

    monkeypatch.setattr("builtins.print", closed)
    monkeypatch.setattr(cli.os, "dup2", discarded)
    assert cli.main(["--home", str(tmp_path), "log"]) == 0


def test_a_log_cut_short_and_regrown_past_the_offset_between_looks_is_read_from_its_first_line(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text("old\n")
    _, position = tail(path, 1)
    followed = follow(path, position, lambda: None)
    path.write_text("regrown\nlines\n")
    assert [next(followed), next(followed)] == ["regrown", "lines"]


def test_an_entry_the_log_cannot_encode_is_a_failure_line_and_the_daemon_carries_on(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path, clock=lambda: AT)
    sink = logger.add(failures_to(log.record), level="ERROR", filter="hands")
    try:
        log.record(Called("list_sessions", {}, object()))
    finally:
        logger.remove(sink)
    [line] = lines(path)
    assert line["type"] == "Failure"
    assert line["message"].startswith("the audit log cannot encode a Called line: the audit log cannot encode a object")


async def test_what_was_heard_where_nothing_waits_on_it_still_fails_loudly() -> None:
    logged: list[str] = []
    sink = logger.add(lambda message: logged.append(message.record["message"]), level="ERROR")

    def record(entry: Entry) -> None:
        if isinstance(entry, Performed):
            raise RuntimeError("the log is full")

    try:
        sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=record)
        await sessions.apply(Joined(member(), "startup"))
        with pytest.raises(RuntimeError):
            await sessions.apply(Prompted(member().id, at=1.0, mode=None, prompt=PromptId("p1")))
        sessions.hear(Closed(member().id, PromptId("p1"), "Done."))

        async def failed() -> None:
            while "performing what was heard failed: RuntimeError: the log is full" not in logged:
                await asyncio.sleep(0)

        await asyncio.wait_for(failed(), 2.0)
    finally:
        logger.remove(sink)


def test_a_brain_turn_that_failed_is_written_with_what_it_failed_of(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    AuditLog(path, clock=lambda: AT).record(BrainSpoke(("x1",), "", (), False, "user", 0.0, ModelFailed(ErrorCategory.SERVER)))
    [line] = lines(path)
    assert line["failed"] == {"type": "ModelFailed", "category": "server"}
