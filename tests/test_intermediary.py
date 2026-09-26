"""The intermediary is told which sessions run and never their history, can decline to answer, and its eval judges the tools the daemon gives it."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
from pipecat.frames.frames import FunctionCallResultProperties
from pipecat.services.llm_service import FunctionCallParams

from hands.sessions.registry import Sessions
from hands.voice.briefing import briefing
from hands.voice.intermediary_instruction import INTERMEDIARY_INSTRUCTION
from hands.voice.tools import intermediary_tools, stay_silent_tool

_SPEC = importlib.util.spec_from_file_location("intermediary_eval", Path(__file__).parents[1] / "evals" / "intermediary.py")
assert _SPEC is not None and _SPEC.loader is not None
evaluation = importlib.util.module_from_spec(_SPEC)
sys.modules["intermediary_eval"] = evaluation
_SPEC.loader.exec_module(evaluation)

AUTH = {"id": "5b0e2f4e-3c1a-4d8e-9f21-7a6c0d9e1b34", "title": "auth refactor", "state": "idle", "mode": "manual mode"}
FRESH = {"id": "c7d1a9e2-8f40-4b6a-a2d3-1e5f9c0b7a68", "title": "untitled, in cc-hands", "state": "working", "mode": "not reported yet"}


def names(sessions: Sessions) -> list[str]:
    return [DirectFunctionWrapper(tool).name for tool in intermediary_tools(sessions)]


def test_the_briefing_names_each_session_by_title_state_and_mode_with_the_id_for_the_tools() -> None:
    note = briefing([AUTH, FRESH])
    assert note.startswith("[hands] ")
    assert f'"auth refactor" (id {AUTH["id"]}), idle, in manual mode' in note
    assert f'"untitled, in cc-hands" (id {FRESH["id"]}), working, its mode not reported yet' in note
    assert "Say nothing about this unless the user asks." in note


def test_the_briefing_with_nothing_running_says_so() -> None:
    assert briefing([]) == "[hands] hands has just started, and no Claude Code sessions are running. Say nothing about this unless the user asks."


async def test_stay_silent_ends_the_turn_without_running_the_model_again() -> None:
    answered: list[tuple[object, FunctionCallResultProperties | None]] = []

    async def capture(result: object, *, properties: FunctionCallResultProperties | None = None) -> None:
        answered.append((result, properties))

    await stay_silent_tool()(cast(FunctionCallParams, SimpleNamespace(result_callback=capture)))
    [(result, properties)] = answered
    assert result == {"silent": True}
    assert properties is not None and properties.run_llm is False


def test_every_tool_the_intermediary_is_given_is_named_by_its_prompt() -> None:
    # A tool the prompt never mentions is one the model has to discover from its schema alone; a tool the
    # prompt names that is not here is worse, and is what the ordering rule in intermediary_instruction forbids.
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    given = names(sessions)
    assert "stay_silent" in given
    # The answer tools and amend/discard are reached for from the narration and the readback, not the prompt.
    named_by_prompt = {"list_sessions", "read_session", "stage_draft", "send_draft", "stay_silent"}
    assert named_by_prompt <= set(given)
    for name in named_by_prompt:
        assert name in INTERMEDIARY_INSTRUCTION


def test_every_conversation_case_loads_with_exactly_one_expectation_and_names_only_tools_the_daemon_gives() -> None:
    given = set(names(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)))
    loaded = evaluation.cases()
    assert loaded, "evals/conversations holds no case"
    for case in loaded:
        for wanted in evaluation.wanted(case.expect):
            assert wanted in given, f"{case.name} expects {wanted}, which the daemon does not give"
        assert case.messages[0]["content"].startswith("[hands] hands has just started")


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


def test_a_call_case_fails_on_the_wrong_arguments_and_on_a_stray_tool() -> None:
    [case] = [case for case in evaluation.cases() if case.name == "dictation-is-staged"]
    right = evaluation.Call("stage_draft", {"session": AUTH["id"], "text": "Use the new token helper in the login flow.", "resolutions": []})
    assert all(check.held for check in evaluation.judge(case, "", (right,)))
    wrong_session = evaluation.Call("stage_draft", {**right.arguments, "session": FRESH["id"]})
    assert not all(check.held for check in evaluation.judge(case, "", (wrong_session,)))
    sent_too = evaluation.Call("send_draft", {"session": AUTH["id"]})
    assert not all(check.held for check in evaluation.judge(case, "", (right, sent_too)))
