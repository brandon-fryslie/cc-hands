"""list_sessions answers with live sessions, labelled by Claude Code's own ai-title."""

import zlib
import json
from pathlib import Path


from hands.core.events import Ended, Joined, PermissionRequested, Prompted, StatusReported, Taken
from hands.core.session import Membership, Permission, PromptId, RequestId, SessionId, Question
from hands.core.status import Busy, Idle, Report, Stamp, Waiting
from hands.sessions.registry import Sessions
from hands.sessions.transcript import ai_title
from hands.voice.tools import list_sessions_tool


def membership(tmp_path: Path, name: str) -> Membership:
    # A distinct pid per name: one process holds one session.
    return Membership(SessionId(name), pid=zlib.crc32(name.encode()), cwd=Path("/code") / name, transcript=tmp_path / f"{name}.jsonl")


def titled(path: Path, *titles: str) -> None:
    records: list[dict[str, object]] = [
        {"type": "user", "message": {"content": 'grep "type":"ai-title" says hi'}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "input": {"record": {"type": "ai-title"}}}]}},
    ]
    title_records: list[dict[str, object]] = [{"type": "ai-title", "aiTitle": t, "sessionId": path.stem} for t in titles]
    records += title_records
    path.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records))


async def call(sessions: Sessions) -> object:
    return await list_sessions_tool(sessions).body()


async def test_live_sessions_are_labelled_with_their_newest_ai_title_and_their_project(tmp_path: Path) -> None:
    working, untitled, blocked, asking, ended = (membership(tmp_path, n) for n in ("working", "untitled", "blocked", "asking", "ended"))
    titled(working.transcript, "first guess", "pipeline spike")
    titled(blocked.transcript, "auth refactor")
    titled(ended.transcript, "old work")
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    for event in (
        Joined(working, "startup"),
        StatusReported(working.id, Report(Busy(), Stamp(1)), at=1.0),
        Prompted(working.id, at=1.0, mode="acceptEdits", prompt=PromptId("p1")),
        Taken(working.id, PromptId("p1"), None, 5.0),
        Joined(untitled, "startup"),
        StatusReported(untitled.id, Report(Idle(), Stamp(1)), at=1.0),
        Joined(blocked, "startup"),
        StatusReported(blocked.id, Report(Waiting("permission prompt"), Stamp(1)), at=2.0),
        PermissionRequested(blocked.id, at=2.0, request=RequestId("r"), on=Permission("Bash", {}), mode="default"),
        Joined(asking, "startup"),
        StatusReported(asking.id, Report(Waiting("input needed"), Stamp(1)), at=3.0),
        PermissionRequested(asking.id, at=3.0, request=RequestId("q"), on=Question((), {}), mode=None),
        Joined(ended, "startup"),
        Ended(ended.id, "prompt_input_exit"),
    ):
        await sessions.apply(event)

    assert await call(sessions) == {
        "sessions": [
            {"id": "working", "title": "pipeline spike in working", "state": "working", "mode": "accept edits mode"},
            {"id": "untitled", "title": "untitled in untitled", "state": "idle", "mode": "not reported yet"},
            {"id": "blocked", "title": "auth refactor in blocked", "state": "waiting for permission to use Bash", "mode": "manual mode"},
            {"id": "asking", "title": "untitled in asking", "state": "waiting for the user to answer its question", "mode": "not reported yet"},
        ]
    }


async def test_a_session_that_joins_on_its_permission_request_is_listed_as_waiting_on_it(tmp_path: Path) -> None:
    """Its status is not read yet, and what it waits on is what its hook asked."""
    lagging = membership(tmp_path, "lagging")
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    for event in (Joined(lagging, "startup"), PermissionRequested(lagging.id, at=1.0, request=RequestId("r"), on=Permission("Bash", {}), mode="default")):
        await sessions.apply(event)
    assert await call(sessions) == {"sessions": [{"id": "lagging", "title": "untitled in lagging", "state": "waiting for permission to use Bash", "mode": "manual mode"}]}


def test_a_title_record_still_being_written_is_not_read(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    titled(transcript, "finished title")
    whole = transcript.read_bytes()
    for unfinished in (b'{"type":"ai-title","aiTitle":"half', '{"type":"ai-title","aiTitle":"caf\u00e9'.encode()[:-1]):
        transcript.write_bytes(whole + unfinished)
        assert ai_title(transcript) == "finished title"


async def test_a_transcript_whose_title_cannot_be_read_lists_the_session_untitled(tmp_path: Path) -> None:
    broken = membership(tmp_path, "broken")
    broken.transcript.write_text('{"type":"ai-title","sessionId":"broken"}\n')
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(broken, "startup"))
    assert await call(sessions) == {"sessions": [{"id": "broken", "title": "untitled in broken", "state": "not reported yet", "mode": "not reported yet"}]}


def test_the_tool_is_named_and_described_from_its_body() -> None:
    tool = list_sessions_tool(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None))
    assert (tool.name, tool.required) == ("list_sessions", ())
    assert "running Claude Code sessions" in tool.description
