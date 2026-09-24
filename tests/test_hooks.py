"""Hook bodies parse once into core events, or are rejected with a reason. Payload fields as captured from Claude Code 2.1.270."""

import json
from pathlib import Path

import pytest

from hands.core.events import Ended, Joined, PermissionRequested, Prompted, Stopped, ToolFinished, Waited
from hands.core.session import AskedQuestion, Membership, Option, Permission, Plan, PlanApproved, Question, RequestId, SessionId
from hands.sessions.home import Home
from hands.sessions.hooks import parse_hook
from hands.sessions.membership import write_membership
from hands.sessions.payload import Rejected

SID = SessionId("bf411065-dc5c-4ec9-8302-61b84bdb5c53")
TRANSCRIPT = f"/Users/me/.claude/projects/-code-a/{SID}.jsonl"
COMMON = {"session_id": SID, "transcript_path": TRANSCRIPT, "cwd": "/code/a"}
REQUEST = RequestId("r1")


def body(**fields: object) -> bytes:
    return json.dumps({**COMMON, **fields}).encode()


def parse(home: Home, raw: bytes):  # noqa: ANN201 - the event union
    return parse_hook(raw, home=home, at=12.5, request=REQUEST)


@pytest.fixture
def home(tmp_path: Path) -> Home:
    return Home(tmp_path)


def test_a_start_reads_the_membership_the_shim_wrote(home: Home) -> None:
    membership = Membership(SID, pid=51810, cwd=Path("/code/a"), transcript=Path(TRANSCRIPT))
    write_membership(home, membership)
    assert parse(home, body(hook_event_name="SessionStart", source="startup")) == Joined(membership, "startup")
    assert parse(home, body(hook_event_name="SessionStart", source="compact")) == Joined(membership, "compact")


def test_a_start_with_no_membership_file_is_rejected(home: Home) -> None:
    with pytest.raises(Rejected, match="no membership file"):
        parse(home, body(hook_event_name="SessionStart", source="startup"))


def test_the_turn_hooks(home: Home) -> None:
    prompt = body(hook_event_name="UserPromptSubmit", prompt="hi", prompt_id="p", permission_mode="default")
    stop = body(hook_event_name="Stop", stop_hook_active=False, last_assistant_message="ok", background_tasks=[])
    end = body(hook_event_name="SessionEnd", reason="other")
    assert parse(home, prompt) == Prompted(SID, at=12.5)
    assert parse(home, stop) == Stopped(SID, "ok")
    assert parse(home, end) == Ended(SID, "other")


def test_a_stop_is_still_the_end_of_a_turn_when_the_reply_it_carries_is_not_a_string(home: Home) -> None:
    """The turn is what the hook says; the reply is how it is narrated. A session stuck working is the worse wrong."""
    assert parse(home, body(hook_event_name="Stop", stop_hook_active=False, last_assistant_message={"text": "ok"})) == Stopped(SID, None)
    assert parse(home, body(hook_event_name="Stop", stop_hook_active=False, last_assistant_message=None)) == Stopped(SID, None)


def test_a_permission_request_carries_the_tool_and_its_input(home: Home) -> None:
    raw = body(hook_event_name="PermissionRequest", tool_name="Bash", tool_input={"command": "rm -r build"}, permission_suggestions=[])
    permission = Permission(tool="Bash", input={"command": "rm -r build"})
    assert parse(home, raw) == PermissionRequested(SID, at=12.5, request=REQUEST, on=permission)


def test_a_finished_or_failed_tool_names_its_call_as_a_permission_does(home: Home) -> None:
    done = body(hook_event_name="PostToolUse", tool_name="Bash", tool_input={"command": "ls"}, tool_use_id="t", tool_response={})
    failed = body(hook_event_name="PostToolUseFailure", tool_name="Bash", tool_input={"command": "ls"}, tool_use_id="t", error="no")
    call = Permission(tool="Bash", input={"command": "ls"})
    assert parse(home, done) == parse(home, failed) == ToolFinished(SID, at=12.5, call=call)


# As Claude Code 2.1.280 posted it, captured live.
QUESTIONS: dict[str, object] = {
    "questions": [
        {"question": "Which color do you prefer?", "header": "Color", "options": [{"label": "red", "description": "The color red"}, {"label": "green", "description": "The color green"}], "multiSelect": False},
        {"question": "Which fruits do you like?", "header": "Fruits", "options": [{"label": "pear", "description": "A sweet fruit"}, {"label": "plum", "description": "A small stone fruit"}], "multiSelect": True},
    ]
}


def test_an_ask_user_question_request_is_a_question_carrying_what_it_asks_and_its_input(home: Home) -> None:
    raw = body(hook_event_name="PermissionRequest", tool_name="AskUserQuestion", tool_input=QUESTIONS, permission_suggestions=[])
    question = Question(
        (
            AskedQuestion("Which color do you prefer?", (Option("red", "The color red"), Option("green", "The color green")), several=False),
            AskedQuestion("Which fruits do you like?", (Option("pear", "A sweet fruit"), Option("plum", "A small stone fruit")), several=True),
        ),
        QUESTIONS,
    )
    event = parse(home, raw)
    assert event == PermissionRequested(SID, at=12.5, request=REQUEST, on=question)
    assert isinstance(event, PermissionRequested) and isinstance(event.on, Question) and event.on.input == QUESTIONS


def test_a_question_answered_at_the_keyboard_finishes_as_the_question_it_was(home: Home) -> None:
    asked = parse(home, body(hook_event_name="PermissionRequest", tool_name="AskUserQuestion", tool_input=QUESTIONS, permission_suggestions=[]))
    answered = {**QUESTIONS, "answers": {"Which color do you prefer?": "green", "Which fruits do you like?": "pear, plum"}}
    finished = parse(home, body(hook_event_name="PostToolUse", tool_name="AskUserQuestion", tool_input=answered, tool_use_id="t", tool_response={}))
    assert isinstance(asked, PermissionRequested) and isinstance(finished, ToolFinished)
    assert isinstance(asked.on, Question) and isinstance(finished.call, Question) and finished.call.asked == asked.on.asked


def test_a_question_with_no_options_and_no_descriptions_is_still_a_question(home: Home) -> None:
    asked = {"questions": [{"question": "Name it?"}, {"question": "Pick", "options": [{"label": "a"}, {"label": "b", "description": ""}]}]}
    raw = body(hook_event_name="PermissionRequest", tool_name="AskUserQuestion", tool_input=asked, permission_suggestions=[])
    on = Question((AskedQuestion("Name it?", (), several=False), AskedQuestion("Pick", (Option("a", None), Option("b", None)), several=False)), asked)
    assert parse(home, raw) == PermissionRequested(SID, at=12.5, request=REQUEST, on=on)


# As Claude Code 2.1.281 sends it: the plan file already read into the input.
PLAN_INPUT = {"plan": "# Plan\n\n1. Create hello.txt.\n2. Write hi into it.\n", "planFilePath": "/Users/me/.claude/plans/noble-goose.md"}


def test_an_exit_plan_mode_request_is_a_plan_carrying_its_text(home: Home) -> None:
    raw = body(hook_event_name="PermissionRequest", tool_name="ExitPlanMode", tool_input=PLAN_INPUT, permission_suggestions=[])
    assert parse(home, raw) == PermissionRequested(SID, at=12.5, request=REQUEST, on=Plan(PLAN_INPUT["plan"]))


@pytest.mark.parametrize("ran", [{}, {"plan": "# Plan, edited at the dialog"}])
def test_an_exit_plan_mode_that_ran_is_a_plan_approved_whatever_its_input_kept(home: Home, ran: dict[str, object]) -> None:
    raw = body(hook_event_name="PostToolUse", tool_name="ExitPlanMode", tool_input=ran, tool_use_id="t", tool_response={})
    assert parse(home, raw) == ToolFinished(SID, at=12.5, call=PlanApproved())


def test_a_plan_with_no_text_is_rejected_by_name(home: Home) -> None:
    with pytest.raises(Rejected, match="missing field 'plan'"):
        parse(home, body(hook_event_name="PermissionRequest", tool_name="ExitPlanMode", tool_input={"planFilePath": "/p.md"}, permission_suggestions=[]))


@pytest.mark.parametrize(
    ("asked", "reason"),
    [
        ({}, "missing field 'questions'"),
        ({"questions": "which?"}, "'questions' should be a list"),
        ({"questions": ["which?"]}, "each question should be a JSON object"),
        ({"questions": [{"question": "q", "options": ["a"]}]}, "each option should be a JSON object"),
        ({"questions": [{"question": "q", "multiSelect": "yes"}]}, "'multiSelect' should be a boolean"),
    ],
)
def test_a_question_that_does_not_parse_is_rejected_by_name(home: Home, asked: dict[str, object], reason: str) -> None:
    with pytest.raises(Rejected, match=reason):
        parse(home, body(hook_event_name="PermissionRequest", tool_name="AskUserQuestion", tool_input=asked, permission_suggestions=[]))


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (b"not json", "not JSON"),
        (body(hook_event_name="SessionStart", source="teleport"), "source 'teleport' is not one hands knows"),
        (b"[1, 2]", "JSON object"),
        (body(hook_event_name="PreCompact"), "'PreCompact' is not one hands handles"),
        (body(hook_event_name="PermissionRequest", tool_input={}), "missing field 'tool_name'"),
        (body(hook_event_name="PermissionRequest", tool_name="Bash", tool_input="ls"), "'tool_input' should be an object"),
        (json.dumps({**COMMON, "session_id": "../../etc/x", "hook_event_name": "Stop"}).encode(), "not a Claude Code session id"),
    ],
)
def test_what_does_not_parse_is_rejected_by_name(home: Home, raw: bytes, reason: str) -> None:
    with pytest.raises(Rejected, match=reason):
        parse(home, raw)


def test_an_end_reason_this_version_does_not_know_is_an_end_nobody_chose(home: Home) -> None:
    assert parse(home, body(hook_event_name="SessionEnd", reason="solar_flare")) == Ended(SID, "other")


def test_the_idle_notification_is_a_session_waiting(home: Home) -> None:
    idle = body(hook_event_name="Notification", message="Claude is waiting for your input", notification_type="idle_prompt")
    assert parse(home, idle) == Waited(SID)


def test_a_notification_hands_did_not_ask_for_is_rejected(home: Home) -> None:
    with pytest.raises(Rejected, match="permission_prompt"):
        parse(home, body(hook_event_name="Notification", message="Claude needs your permission", notification_type="permission_prompt"))
