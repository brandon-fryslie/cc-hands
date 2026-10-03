"""The intermediary is told which sessions run and never their history, can decline to answer, and its eval judges the tools the daemon gives it."""

import importlib.util
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from collections.abc import Awaitable, Callable
from pipecat.frames.frames import FunctionCallResultProperties
from pipecat.services.llm_service import FunctionCallParams

from hands.sessions.home import Home
from hands.sessions.overlays import Overlays
from hands.sessions.registry import Sessions
from hands.voice.briefing import brief, briefing, tail
from hands.voice.speech import Tailed
from hands.voice.intermediary_instruction import INTERMEDIARY_INSTRUCTION
from hands.sessions.sentences import Sentences
from hands.voice.sentences import SummaryStore
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
        return [tool.name for tool in intermediary_tools(sessions, SummaryStore(Sentences(Path(home) / "sentences.db")), Overlays(Home(Path(home))))]


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


def test_the_tail_reminds_the_model_of_what_the_user_heard_hands_say() -> None:
    """A question hands read out as written was answered on 2026-10-03 and the brain asked what the user was talking about."""
    told = tail([], ("cc-hands has a question for you.", 'cc-hands: Which flag? --force ("overwrite"); --dry-run.'))
    # Each line is quoted whole, so the quotes and semicolons a session's question carries read as one line.
    assert told == (
        "[hands] No Claude Code sessions are running now. Say nothing about this unless the user asks. "
        'Lately hands said to the user, oldest first: "cc-hands has a question for you."; "cc-hands: Which flag? --force (\\"overwrite\\"); --dry-run.". '
        "What the user says may answer one of these; when it does, act on it."
    )


def test_a_long_line_hands_said_is_cut_in_the_tail() -> None:
    told = tail([], ("x" * 1000,))
    assert "x" * 400 + "... (cut short)" in told and "x" * 401 not in told


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
    named = set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", INTERMEDIARY_INSTRUCTION)) - quoted_code_names
    given = set(names(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)))
    assert named, "the prompt names no tool at all"
    assert named <= given, f"the prompt names {sorted(named - given)}, which the daemon does not give"


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
