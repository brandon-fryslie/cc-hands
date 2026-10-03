"""The intermediary is told which sessions run and never their history, can decline to answer, and its eval judges the tools the daemon gives it."""

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from collections.abc import Awaitable, Callable
from pipecat.frames.frames import FunctionCallResultProperties
from pipecat.services.llm_service import FunctionCallParams
import pytest

from hands.core.session import SessionId
from hands.core.wire import Exchanged, MainTurn, Unreached
from hands.sessions.audit import AuditLog, BacklogUnread, Called, Transcribed, retired
from hands.sessions.home import Home
from hands.sessions.registry import Sessions
from hands.voice.briefing import brief, briefing, tail
from hands.voice.speech import Tailed
from hands.voice.intermediary_instruction import INTERMEDIARY_INSTRUCTION, brain_instruction
from hands.sessions.sentences import Sentences
from hands.voice.sentences import SummaryStore
from hands.voice.narrator import Recounts
from hands.voice.player import Player
from hands.voice.tools import intermediary_tools, pipecat_function, stay_silent_tool

_SPEC = importlib.util.spec_from_file_location("intermediary_eval", Path(__file__).parents[1] / "evals" / "intermediary.py")
assert _SPEC is not None and _SPEC.loader is not None
evaluation = importlib.util.module_from_spec(_SPEC)
sys.modules["intermediary_eval"] = evaluation
_SPEC.loader.exec_module(evaluation)

AUTH = {"id": "5b0e2f4e-3c1a-4d8e-9f21-7a6c0d9e1b34", "name": "cc-hands, auth refactor", "state": "idle", "mode": "manual mode"}
FRESH = {"id": "c7d1a9e2-8f40-4b6a-a2d3-1e5f9c0b7a68", "name": "cc-hands", "state": "working", "mode": "not reported yet"}


def names(sessions: Sessions) -> list[str]:
    with tempfile.TemporaryDirectory() as home:
        return [tool.name for tool in intermediary_tools(sessions, SummaryStore(Sentences(Path(home) / "sentences.db")), Home(Path(home)), Recounts(), Player(lambda _entry: None))]


def test_the_briefing_names_each_session_by_name_state_and_mode_with_the_id_for_the_tools() -> None:
    note = briefing([AUTH, FRESH])
    assert note.startswith("[hands] ")
    assert f'"cc-hands, auth refactor" (id {AUTH["id"]}), idle, permission mode: manual mode' in note
    assert f'"cc-hands" (id {FRESH["id"]}), working, permission mode: not reported yet' in note
    assert "Say nothing about this unless the user asks." in note


def test_the_briefing_with_nothing_running_says_so() -> None:
    assert briefing([]) == "[hands] hands has just started, and no Claude Code sessions are running. Say nothing about this unless the user asks."


def test_the_tail_names_each_session_as_the_briefing_does_and_says_it_is_current() -> None:
    told = tail([AUTH])
    assert told.startswith("[hands] ")
    assert f'"cc-hands, auth refactor" (id {AUTH["id"]}), idle, permission mode: manual mode' in told
    assert "as this message is sent" in told and "Say nothing about this unless the user asks." in told


def test_the_tail_with_nothing_running_says_so() -> None:
    assert tail([]) == "[hands] No Claude Code sessions are running now. Say nothing about this unless the user asks."


async def test_a_brain_read_from_the_tail_is_given_no_briefing() -> None:
    queued: list[object] = []

    async def queue(frame: object) -> None:
        queued.append(frame)

    await brief(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None), Tailed(), queue)
    assert queued == []


async def test_stay_silent_ends_the_turn_without_running_the_model_again() -> None:
    answered: list[tuple[object, FunctionCallResultProperties | None]] = []

    async def capture(result: object, *, properties: FunctionCallResultProperties | None = None) -> None:
        answered.append((result, properties))

    handler = cast(Callable[[FunctionCallParams], Awaitable[None]], pipecat_function(stay_silent_tool())._handler)  # pyright: ignore[reportPrivateUsage]
    await handler(cast(FunctionCallParams, SimpleNamespace(result_callback=capture, arguments={})))
    [(result, properties)] = answered
    assert result == {"silent": True}
    assert properties is not None and properties.run_llm is False


def test_the_prompt_names_no_tool_the_daemon_does_not_give() -> None:
    # The rule in intermediary_instruction: a prompt that asks for a tool before it exists gets that tool paraphrased.
    # Every tool is snake_case, so every snake_case name in the prompt is a tool, bar the code names it quotes as ones never to say.
    quoted_code_names = {"parse_date", "test_invoice_total"}
    given = set(names(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)))
    for prompt in (INTERMEDIARY_INSTRUCTION, brain_instruction(Path("/home/hands/audit.jsonl"))):
        named = set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", prompt)) - quoted_code_names
        assert named, "the prompt names no tool at all"
        assert named <= given, f"the prompt names {sorted(named - given)}, which the daemon does not give"


def test_only_the_brain_which_has_bash_is_told_of_the_log_and_it_keeps_the_closing_words_last() -> None:
    told = brain_instruction(Path("/my home/audit.jsonl"))
    assert "audit.jsonl" not in INTERMEDIARY_INSTRUCTION and "Bash" not in INTERMEDIARY_INSTRUCTION
    assert told.startswith(INTERMEDIARY_INSTRUCTION.split("\n\n# Above all")[0]) and told.endswith(INTERMEDIARY_INSTRUCTION.split("\n\n")[-1])
    # A home with a space in it is one argument to every command the brain is shown.
    assert "'/my home/audit.jsonl'" in told


@pytest.mark.skipif(shutil.which("jq") is None, reason="the brain's commands read the log with jq")
def test_the_commands_the_brain_is_shown_find_in_a_log_hands_wrote_what_they_say_they_find(tmp_path: Path) -> None:
    path = tmp_path / "a home" / "audit.jsonl"
    log = AuditLog(path, clock=lambda: datetime(2026, 10, 3, tzinfo=UTC))
    log.record(Transcribed("send it"))
    # A line cut short, as a write that failed part way leaves it: the lines after it are still read.
    with path.open("a", encoding="utf-8") as torn:
        torn.write('{"at": "2026-10-03T00:00:00.000+00:00", "level": "err\n')
    log.record(Exchanged("x", SessionId("s1"), MainTurn(None), "POST", "/v1/messages", 2, (), 0.0, 0.0, Unreached("no route", 0.0), True))
    log.record(Called("list_sessions", {}, {"sessions": []}))
    log.record(BacklogUnread(project="/code/p", error="lit exited 3", seconds=0.1))
    shown = [line[2:].partition(": ") for line in brain_instruction(path).splitlines() if line.startswith("- ")]
    commands = {label: command for label, _, command in shown if " | jq " in command}
    assert len(commands) == 3

    def found(label: str) -> list[str]:
        ran = subprocess.run(commands[label], shell=True, capture_output=True, text=True, check=True)
        return [json.loads(line)["type"] for line in ran.stdout.splitlines()]

    assert found("the latest errors") == ["Exchanged", "BacklogUnread"]
    assert found("what happened lately") == ["Transcribed", "Called", "BacklogUnread"]
    assert found("one kind of line") == ["Called"]
    # Once the log has been retired, what it held is found too, before what came after.
    path.rename(retired(path))
    log.record(Called("list_sessions", {}, {"sessions": []}))
    assert found("one kind of line") == ["Called", "Called"]
    assert found("the latest errors") == ["Exchanged", "BacklogUnread"]


def test_every_conversation_case_loads_with_exactly_one_expectation_and_names_only_tools_the_daemon_gives() -> None:
    given = set(names(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)))
    loaded = evaluation.cases()
    assert loaded, "evals/conversations holds no case"
    for case in loaded:
        for wanted in evaluation.wanted(case.expect):
            assert wanted in given, f"{case.name} expects {wanted}, which the daemon does not give"


def test_a_reply_that_names_an_id_or_a_code_name_fails_and_a_plain_one_holds() -> None:
    [case] = [case for case in evaluation.cases() if case.name == "sessions-by-title"]
    plain = evaluation.judge(case, "The auth refactor is idle, and the hands daemon is working.", ())
    assert all(check.held for check in plain), plain
    leaked = evaluation.judge(case, f"auth refactor is {AUTH['id']}, and hands daemon runs src/hands/daemon.py.", ())
    assert {check.name for check in leaked if not check.held} >= {"no ids", "spoken"}


def test_a_silent_case_holds_only_for_stay_silent_alone() -> None:
    [case] = [case for case in evaluation.cases() if case.name == "not-for-me-door"]
    silent = evaluation.Call("stay_silent", {})
    assert all(check.held for check in evaluation.judge(case, "", (silent,)))
    assert not all(check.held for check in evaluation.judge(case, "Sure!", (silent,)))
    assert not all(check.held for check in evaluation.judge(case, "Sure!", ()))


async def test_words_said_beside_a_look_are_judged_with_the_step_after_it() -> None:
    [case] = [case for case in evaluation.cases() if case.name == "not-for-me-door"]
    asked = iter([(f"Let me check session {AUTH['id']}.", (evaluation.Call("list_sessions", {}),)), ("", (evaluation.Call("stay_silent", {}),))])

    async def ask(_messages: object) -> tuple[str, tuple[object, ...]]:
        return next(asked)

    said, calls, looks = await evaluation.answer(ask, case)
    assert looks == 1 and said == f"Let me check session {AUTH['id']}."
    assert not all(check.held for check in evaluation.judge(case, said, calls))


def test_a_call_case_fails_on_the_wrong_arguments_and_on_a_stray_tool() -> None:
    [case] = [case for case in evaluation.cases() if case.name == "dictation-is-staged"]
    right = evaluation.Call("stage_draft", {"session": AUTH["id"], "text": "Use the new token helper in the login flow.", "resolutions": []})
    assert all(check.held for check in evaluation.judge(case, "", (right,)))
    wrong_session = evaluation.Call("stage_draft", {**right.arguments, "session": FRESH["id"]})
    assert not all(check.held for check in evaluation.judge(case, "", (wrong_session,)))
    sent_too = evaluation.Call("send_draft", {"session": AUTH["id"]})
    assert not all(check.held for check in evaluation.judge(case, "", (right, sent_too)))
    staged_twice = evaluation.Call("stage_draft", {**right.arguments, "text": "Use the old token helper."})
    assert not all(check.held for check in evaluation.judge(case, "", (right, staged_twice)))
