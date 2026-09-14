"""Hook bodies parse once into core events, or are rejected with a reason. Payload fields as captured from Claude Code 2.1.270."""

import json
from pathlib import Path

import pytest

from hands.core.events import Ended, Joined, PermissionRequested, Prompted, Stopped, ToolFinished
from hands.core.session import Membership, Permission, RequestId, SessionId, TmuxPane
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
    membership = Membership(SID, pid=51810, pane=TmuxPane("%8"), cwd=Path("/code/a"), transcript=Path(TRANSCRIPT))
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
    assert parse(home, stop) == Stopped(SID)
    assert parse(home, end) == Ended(SID)


def test_a_permission_request_carries_the_tool_and_its_input(home: Home) -> None:
    raw = body(hook_event_name="PermissionRequest", tool_name="Bash", tool_input={"command": "rm -r build"}, permission_suggestions=[])
    permission = Permission(tool="Bash", input={"command": "rm -r build"})
    assert parse(home, raw) == PermissionRequested(SID, at=12.5, request=REQUEST, permission=permission)


def test_a_finished_or_failed_tool_names_its_call_as_a_permission_does(home: Home) -> None:
    done = body(hook_event_name="PostToolUse", tool_name="Bash", tool_input={"command": "ls"}, tool_use_id="t", tool_response={})
    failed = body(hook_event_name="PostToolUseFailure", tool_name="Bash", tool_input={"command": "ls"}, tool_use_id="t", error="no")
    call = Permission(tool="Bash", input={"command": "ls"})
    assert parse(home, done) == parse(home, failed) == ToolFinished(SID, at=12.5, call=call)


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
