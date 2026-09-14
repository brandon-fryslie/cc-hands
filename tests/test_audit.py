"""The audit log: what is written, how it reads back, and a send traced from the user's words to the keys typed."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from loguru import logger
from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
from pipecat.services.llm_service import FunctionCallParams

from hands.core.effects import Text, Type
from hands.core.events import Joined, Tick
from hands.core.session import Membership, PromptText, SessionId, TmuxPane
from hands.daemon import cli
from hands.sessions.audit import (
    Applied,
    AuditLog,
    Called,
    EffectFailed,
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
from hands.sessions.registry import Sessions
from hands.voice.tools import Tool, audited, draft_tools

AT = datetime(2026, 9, 14, 12, 0, 0, 123000, tzinfo=UTC)


def member(pane: TmuxPane | None) -> Membership:
    return Membership(SessionId("s1"), pid=4242, pane=pane, cwd=Path("/code/cc-hands"), transcript=Path("/nowhere/s1.jsonl"))


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


async def invoke(tool: Tool, **arguments: object) -> object:
    results: list[object] = []

    async def capture(result: object, *, properties: object = None) -> None:
        results.append(result)

    params = FunctionCallParams(
        function_name=DirectFunctionWrapper(tool).name, tool_call_id="call-1", arguments=arguments,
        llm=cast(Any, None), pipeline_worker=cast(Any, None), context=cast(Any, None), result_callback=capture,
    )
    await DirectFunctionWrapper(tool).invoke(arguments, params)
    [result] = results
    return result


def test_an_entry_is_its_type_and_fields_nested_values_alike() -> None:
    assert encoded(Performed(Type(TmuxPane("%3"), Text(PromptText("fix it"))))) == {
        "type": "Performed",
        "effect": {"type": "Type", "pane": "%3", "input": {"type": "Text", "body": "fix it"}},
    }
    assert encoded(Applied(Joined(member(None), "startup")))["event"] == {
        "type": "Joined",
        "membership": {"type": "Membership", "id": "s1", "pid": 4242, "pane": None, "cwd": "/code/cc-hands", "transcript": "/nowhere/s1.jsonl"},
        "source": "startup",
    }


def test_a_value_the_log_cannot_write_is_refused_rather_than_guessed_at() -> None:
    with pytest.raises(TypeError, match="cannot encode a set"):
        encoded(Called("t", {"odd": {1, 2}}, None))


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
    await sessions.apply(Joined(member(None), "startup"))
    assert recorded == [Applied(Joined(member(None), "startup"))]


async def test_an_audited_tool_keeps_its_schema_and_writes_its_call_beside_its_result() -> None:
    recorded: list[Entry] = []
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    [stage, *_] = draft_tools(sessions)
    wrapped = audited(stage, recorded.append)
    assert DirectFunctionWrapper(wrapped).to_function_schema().to_default_dict() == DirectFunctionWrapper(stage).to_function_schema().to_default_dict()
    result = await invoke(wrapped, session="nobody", text="hi", resolutions=[])
    assert recorded == [Called("stage_draft", {"session": "nobody", "text": "hi", "resolutions": []}, result)]


async def test_a_send_that_fails_to_type_is_written_as_failed(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=recorded.append)
    await sessions.apply(Joined(member(TmuxPane("%999999")), "startup"))
    [stage, _, _, send] = [audited(tool, recorded.append) for tool in draft_tools(sessions)]
    await invoke(stage, session="s1", text="hello", resolutions=[])
    await invoke(send, session="s1")
    typed = Type(TmuxPane("%999999"), Text(PromptText("hello")))
    [failed] = [entry for entry in recorded if isinstance(entry, EffectFailed)]
    assert failed.effect == typed and "can't find pane" in failed.error
    assert Performed(typed) not in recorded
