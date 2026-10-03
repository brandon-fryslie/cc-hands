"""The audit log: what is written, how it reads back, and what becomes a line when writing or encoding fails."""

import asyncio
import inspect
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
from hands.sessions import audit
from hands.sessions.audit import (
    Applied,
    AuditLog,
    AsideAnswered,
    BacklogUnread,
    BrainAnswered,
    BrainSpoke,
    Called,
    Entry,
    Failure,
    Named,
    Performed,
    Replied,
    Transcribed,
    encoded,
    failures_to,
    follow,
    segment,
    segments,
    tail,
)
from hands.sessions.home import Home
from hands.sessions.model_facts import ModelFailed, ModelReplyEmpty
from hands.sessions.registry import Sessions
from hands.voice.tools import Result, Tool, audited, draft_tools, tool
from hands.core.status import Busy, Report, Stamp
from hands.core.wire import Answered, Exchanged, Garbled, MainTurn, Reached, Uncopied, Unreached

# When hands heard a Stop, on the clock Claude Code stamps a status with.
STOP_HEARD = Stamp(1500)
STOP_REQUEST = RequestId("stop")

AT = datetime(2026, 9, 14, 12, 0, 0, 123000, tzinfo=UTC)


def member() -> Membership:
    return Membership(SessionId("s1"), pid=4242, cwd=Path("/code/cc-hands"), transcript=Path("/nowhere/s1.jsonl"))


def lines(log: Path) -> list[dict[str, Any]]:
    """Every line of the log, its segments read oldest first."""
    return [json.loads(line) for base in segments(log) for line in segment(log, base).read_text().splitlines()]


async def invoke(tool: Tool, **arguments: object) -> Result:
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
        encoded(Called("t", {"odd": {1, 2}}, {}))


def test_the_log_is_the_user_s_alone_to_read_whether_it_is_new_or_was_there(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh"
    AuditLog(fresh, clock=lambda: AT).record(Transcribed("send it"))
    there = tmp_path / "there"
    there.mkdir(mode=0o755)
    AuditLog(there, clock=lambda: AT)
    assert [path.stat().st_mode & 0o777 for path in (fresh, segment(fresh, 0), there)] == [0o700, 0o600, 0o700]


def test_each_entry_is_one_line_stamped_with_when_it_was_written(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "deep" / "audit", clock=lambda: AT)
    log.record(Transcribed("send it"))
    log.record(Replied("Sent to cc-hands.", interrupted=False))
    assert lines(tmp_path / "deep" / "audit") == [
        {"at": "2026-09-14T12:00:00.123+00:00", "level": "info", "type": "Transcribed", "text": "send it"},
        {"at": "2026-09-14T12:00:00.123+00:00", "level": "info", "type": "Replied", "text": "Sent to cc-hands.", "interrupted": False},
    ]


def test_a_line_is_an_error_when_it_is_a_failure_or_says_what_failed_and_nothing_else_is(tmp_path: Path) -> None:
    path = tmp_path / "audit"
    log = AuditLog(path, clock=lambda: AT)
    log.record(Failure(source="hands.x:f", message="broke", where="/x.py:1", trace=()))
    log.record(BacklogUnread(project="/code/p", error="lit exited 3", seconds=0.1))
    log.record(BrainAnswered(prompt="p1", error="rate_limit"))
    log.record(BrainAnswered(prompt="p2", error=None))
    log.record(Named(session="s1", outcome="kept", before="a b", name=None, reply="a b", error=None, seconds=0.1))
    log.record(Named(session="s1", outcome="failed", before="a b", name=None, reply=None, error="timed out", seconds=0.1))
    log.record(AsideAnswered("q", "", True, SessionId("s2"), 0.0, 9.0))
    log.record(BrainSpoke(("x1",), "", (), False, "user", 0.0, ModelReplyEmpty()))
    log.record(BrainSpoke(("x1",), "Sent.", (), False, "user", 0.0, None))
    log.record(Called("tell_turn", {}, {"error": "no running session has the id 'x'"}))
    # An "error" deep in a line, in what a tool handed back, does not make the line hands' error.
    log.record(Called("read_turn", {}, {"turn": {"error": {"type": "rate_limit_error"}}}))
    for reply in (
        Reached(200, 0.0, 0.0, 2, Answered({"input_tokens": 3})),
        Reached(429, 0.0, 0.0, 2, Answered({"type": "error", "error": {"type": "rate_limit_error"}})),
        Reached(200, 0.0, 0.0, 2, Garbled("the stream ended early")),
        Unreached("ClientConnectorError: no route", 0.0),
        Uncopied("the copy broke off", 0.0),
    ):
        log.record(Exchanged("x", SessionId("s1"), MainTurn(None), "POST", "/v1/messages", 2, (), 0.0, 0.0, reply, False))
    assert [(line["type"], line["level"]) for line in lines(path)] == [
        ("Failure", "error"),
        ("BacklogUnread", "error"),
        ("BrainAnswered", "error"),
        ("BrainAnswered", "info"),
        ("Named", "info"),
        ("Named", "error"),
        ("AsideAnswered", "error"),
        ("BrainSpoke", "error"),
        ("BrainSpoke", "info"),
        ("Called", "error"),
        ("Called", "info"),
        ("Exchanged", "info"),
        ("Exchanged", "error"),
        ("Exchanged", "error"),
        ("Exchanged", "error"),
        ("Exchanged", "error"),
    ]


def test_text_cut_mid_emoji_is_a_line_that_reads_back_as_it_was_and_whole_characters_are_written_as_themselves(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit", clock=lambda: AT)
    log.record(Transcribed("cut \ud83d, whole \U0001f600 é"))
    assert lines(tmp_path / "audit") == [{"at": "2026-09-14T12:00:00.123+00:00", "level": "info", "type": "Transcribed", "text": "cut \ud83d, whole \U0001f600 é"}]
    assert "cut \\ud83d, whole \U0001f600 é" in segment(tmp_path / "audit", 0).read_text(encoding="utf-8")


def test_the_tail_is_the_newest_complete_lines_and_following_picks_up_where_it_ended(tmp_path: Path) -> None:
    log = tmp_path / "audit"
    assert tail(log, 5) == ([], 0)
    log.mkdir()
    segment(log, 0).write_text("one\ntwo\nthree\npart")
    newest, offset = tail(log, 2)
    assert (newest, offset) == (["two", "three"], 14)

    looks: list[str] = []
    followed = follow(log, offset, lambda: looks.append("look"))
    with segment(log, 0).open("a") as active:
        active.write("ial\nfour\n")
    assert [next(followed), next(followed)] == ["partial", "four"]


def test_the_log_never_holds_more_than_two_segments_of_the_bound(tmp_path: Path) -> None:
    log = tmp_path / "audit"
    writer = AuditLog(log, clock=lambda: AT, segment_bytes=1000)
    for number in range(200):
        writer.record(Transcribed(f"line {number}"))
        assert len(segments(log)) <= 2
        assert all(segment(log, base).stat().st_size <= 1000 for base in segments(log))
    closed, active = segments(log)
    assert segment(log, closed).stat().st_mode & 0o777 == 0o600
    # A segment is named by its base offset: where the one before it ended.
    assert active == closed + segment(log, closed).stat().st_size
    # The active segment opens with the roll that began it, and which segments retention deleted.
    first, second = [json.loads(line) for line in segment(log, active).read_text().splitlines()][:2]
    assert first["type"] == "Rolled" and first["base"] == active and first["deleted"] == [max(base for base in first["deleted"])]
    assert second["type"] == "Transcribed"
    assert lines(log)[-1]["text"] == "line 199"


def test_a_reader_following_the_log_across_a_roll_loses_no_line(tmp_path: Path) -> None:
    log = tmp_path / "audit"
    writer = AuditLog(log, clock=lambda: AT, segment_bytes=1000)
    writer.record(Transcribed("before"))
    _, offset = tail(log, 1)
    followed = follow(log, offset, lambda: None)
    # Enough between two reads to roll the log once, with lines left unread in the segment closed.
    for number in range(15):
        writer.record(Transcribed(f"line {number}"))
    assert len(segments(log)) == 2
    read = [json.loads(next(followed)) for _ in range(16)]
    assert [line["text"] for line in read if line["type"] == "Transcribed"] == [f"line {number}" for number in range(15)]
    assert [line["type"] for line in read].count("Rolled") == 1


def test_a_roll_just_after_the_reader_lists_the_segments_loses_no_line_and_tells_none_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "audit"
    writer = AuditLog(log, clock=lambda: AT, segment_bytes=400)
    writer.record(Transcribed("before"))
    _, offset = tail(log, 1)
    listed = audit.segments
    rolled = iter([True])

    def then_rolled(directory: Path) -> list[int]:
        found = listed(directory)
        for _ in zip(range(1), rolled):
            for number in range(5):
                writer.record(Transcribed(f"line {number}"))
        return found

    monkeypatch.setattr(audit, "segments", then_rolled)
    followed = follow(log, offset, lambda: None)
    read = [json.loads(next(followed)) for _ in range(6)]
    assert len(listed(log)) == 2
    assert [line["text"] for line in read if line["type"] == "Transcribed"] == [f"line {number}" for number in range(5)]


def test_a_reader_behind_retention_goes_on_at_the_oldest_segment_kept(tmp_path: Path) -> None:
    log = tmp_path / "audit"
    log.mkdir()
    segment(log, 100).write_text("kept\n")
    assert next(follow(log, 40, lambda: None)) == "kept"


def test_the_tail_reaches_into_the_older_segment_when_the_active_one_is_short(tmp_path: Path) -> None:
    log = tmp_path / "audit"
    log.mkdir()
    segment(log, 0).write_text("one\ntwo\n")
    segment(log, 8).write_text("three\n")
    assert tail(log, 2) == (["two", "three"], 14)


def test_a_line_that_rolls_the_log_is_stamped_with_the_rolled_line_in_front_of_it(tmp_path: Path) -> None:
    log = tmp_path / "audit"
    ticks = iter(range(1000))
    writer = AuditLog(log, clock=lambda: AT.replace(second=next(ticks) % 60), segment_bytes=200)
    for number in range(5):
        writer.record(Transcribed(f"line {number}"))
    stamps = [line["at"] for line in lines(log)]
    assert stamps == sorted(stamps)
    newest = [json.loads(line) for line in segment(log, segments(log)[-1]).read_text().splitlines()]
    assert [line["type"] for line in newest][:2] == ["Rolled", "Transcribed"]


def test_a_line_the_disk_will_not_take_is_lost_out_loud_and_the_daemon_carries_on(tmp_path: Path) -> None:
    path = tmp_path / "audit"
    log = AuditLog(path, clock=lambda: AT)
    segment(path, 0).mkdir()  # opening a directory to append fails as a full or read-only disk does
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
    home.audit.mkdir()
    segment(home.audit, 0).write_text("a\nb\nc\n")

    def interrupted(_: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", interrupted)
    assert cli.main(["--home", str(tmp_path), "log", "-n", "2"]) == 0
    assert capsys.readouterr().out == "b\nc\n"


def _out_of_space() -> None:
    raise OSError("no space left")


def _write_heartbeat() -> None:
    logger.info("not a failure")
    try:
        _out_of_space()
    except OSError:
        logger.exception("cannot write the heartbeat")
    logger.error("no exception here")
    try:
        try:
            _out_of_space()
        except OSError as error:
            raise RuntimeError("the heartbeat is lost") from error
    except RuntimeError:
        logger.exception("the daemon is not beating")


def test_an_error_logged_anywhere_is_a_failure_line_with_its_exception_where_it_was_logged_and_the_frames_it_came_up_through() -> None:
    recorded: list[Entry] = []
    sink = logger.add(failures_to(recorded.append), level="ERROR")
    try:
        _write_heartbeat()
    finally:
        logger.remove(sink)
    # Each line of _write_heartbeat and _out_of_space, counted from its def.
    at = {name: inspect.getsourcelines(function)[1] for name, function in (("write", _write_heartbeat), ("raise", _out_of_space))}
    assert recorded == [
        Failure(
            source="test_audit:_write_heartbeat",
            message="cannot write the heartbeat: OSError: no space left",
            where=f"{__file__}:{at['write'] + 5}",
            trace=("OSError: no space left", f"{__file__}:{at['write'] + 3} in _write_heartbeat", f"{__file__}:{at['raise'] + 1} in _out_of_space"),
        ),
        Failure(source="test_audit:_write_heartbeat", message="no exception here", where=f"{__file__}:{at['write'] + 6}", trace=()),
        # Raised from another, it comes after the one it was raised from: the first cause, where it began, is first.
        Failure(
            source="test_audit:_write_heartbeat",
            message="the daemon is not beating: RuntimeError: the heartbeat is lost",
            where=f"{__file__}:{at['write'] + 13}",
            trace=(
                "OSError: no space left",
                f"{__file__}:{at['write'] + 9} in _write_heartbeat",
                f"{__file__}:{at['raise'] + 1} in _out_of_space",
                "RuntimeError: the heartbeat is lost",
                f"{__file__}:{at['write'] + 11} in _write_heartbeat",
            ),
        ),
    ]


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
    [failure] = recorded
    assert isinstance(failure, Failure)
    assert (failure.source, failure.message) == ("hands.voice.tools:call", "the tool broken raised, called with {'session': 's1'}: RuntimeError: the transcript went away")
    # Where it was raised, a frame of the test's own, is the last frame; where hands logged it is in the tools.
    assert failure.trace[-1].endswith(" in broken") and failure.where.split("/")[-1].startswith("tools.py:")


def test_hands_log_piped_into_a_reader_that_stops_ends_quietly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    home.audit.mkdir()
    segment(home.audit, 0).write_text("a\n")

    def closed(*_: object, **__: object) -> None:
        raise BrokenPipeError

    def discarded(*_: object) -> None:
        pass

    monkeypatch.setattr("builtins.print", closed)
    monkeypatch.setattr(cli.os, "dup2", discarded)
    assert cli.main(["--home", str(tmp_path), "log"]) == 0


def test_hands_log_prints_a_control_json_left_raw_as_its_escape_and_the_line_is_still_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path)
    home.audit.mkdir()
    segment(home.audit, 0).write_text(json.dumps({"t": "a\x9b2J\x7f\u202eb\x1b"}, ensure_ascii=False) + "\n")

    def interrupted(*_: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "LOG_POLL_SECONDS", 0)
    monkeypatch.setattr(cli.time, "sleep", interrupted)
    assert cli.main(["--home", str(tmp_path), "log"]) == 0
    printed = capsys.readouterr().out
    assert printed == '{"t": "a\\u009b2J\\u007f\\u202eb\\u001b"}\n'
    assert json.loads(printed) == {"t": "a\x9b2J\x7f\u202eb\x1b"}


def test_an_entry_the_log_cannot_encode_is_a_failure_line_and_the_daemon_carries_on(tmp_path: Path) -> None:
    path = tmp_path / "audit"
    log = AuditLog(path, clock=lambda: AT)
    sink = logger.add(failures_to(log.record), level="ERROR", filter="hands")
    try:
        log.record(Called("list_sessions", {}, {"sessions": object()}))
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
    path = tmp_path / "audit"
    AuditLog(path, clock=lambda: AT).record(BrainSpoke(("x1",), "", (), False, "user", 0.0, ModelFailed(ErrorCategory.SERVER)))
    [line] = lines(path)
    assert line["failed"] == {"type": "ModelFailed", "category": "server"}
