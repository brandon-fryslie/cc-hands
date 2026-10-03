"""list_sessions answers with live sessions, each by its project and its newest name, never Claude Code's ai-title."""

import zlib
import json
from pathlib import Path


from hands.core.events import Ended, Joined, PermissionRequested, Prompted, StatusReported, Taken
from hands.core.session import Membership, Permission, PromptId, RequestId, SessionId, Question
from hands.core.status import Busy, Idle, Report, Stamp, Waiting
from hands.sessions.home import Home
from hands.sessions.overlays import Overlays
from hands.sessions.registry import Sessions
from hands.sessions.transcript import session_name
from hands.voice.tools import list_sessions_tool


def membership(tmp_path: Path, name: str) -> Membership:
    # A distinct pid per name: one process holds one session.
    return Membership(SessionId(name), pid=zlib.crc32(name.encode()), cwd=Path("/code") / name, transcript=tmp_path / f"{name}.jsonl")


def named(path: Path, *names: str) -> None:
    records: list[dict[str, object]] = [
        {"type": "user", "message": {"content": 'grep "type":"custom-title" says hi'}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "input": {"record": {"type": "custom-title"}}}]}},
        # Claude Code's own title, long and never seen, is not a name.
        {"type": "ai-title", "aiTitle": "Investigating why the pipeline drops frames under load", "sessionId": path.stem},
    ]
    name_records: list[dict[str, object]] = [{"type": "custom-title", "customTitle": n, "sessionId": path.stem} for n in names]
    records += name_records
    path.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records))


async def call(sessions: Sessions, tmp_path: Path) -> object:
    return await list_sessions_tool(sessions, Overlays(Home(tmp_path / "home"))).body()


async def test_live_sessions_are_named_by_their_project_then_their_newest_name(tmp_path: Path) -> None:
    working, untitled, blocked, asking, ended = (membership(tmp_path, n) for n in ("working", "untitled", "blocked", "asking", "ended"))
    named(working.transcript, "first guess", "pipeline spike")
    named(blocked.transcript, "auth refactor")
    named(ended.transcript, "old work")
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
    Overlays(Home(tmp_path / "home")).set(working.id, "watched")

    assert await call(sessions, tmp_path) == {
        "sessions": [
            {"id": "working", "name": "working, pipeline spike", "state": "working", "mode": "accept edits mode", "watched": "yes"},
            {"id": "untitled", "name": "untitled", "state": "idle", "mode": "not reported yet", "watched": "no"},
            {"id": "blocked", "name": "blocked, auth refactor", "state": "waiting for permission to use Bash", "mode": "manual mode", "watched": "no"},
            {"id": "asking", "name": "asking", "state": "waiting for the user to answer its question", "mode": "not reported yet", "watched": "no"},
        ]
    }


async def test_a_session_that_joins_on_its_permission_request_is_listed_as_waiting_on_it(tmp_path: Path) -> None:
    """Its status is not read yet, and what it waits on is what its hook asked."""
    lagging = membership(tmp_path, "lagging")
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    for event in (Joined(lagging, "startup"), PermissionRequested(lagging.id, at=1.0, request=RequestId("r"), on=Permission("Bash", {}), mode="default")):
        await sessions.apply(event)
    assert await call(sessions, tmp_path) == {"sessions": [{"id": "lagging", "name": "lagging", "state": "waiting for permission to use Bash", "mode": "manual mode", "watched": "no"}]}


def test_a_name_record_still_being_written_is_not_read(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    named(transcript, "finished name")
    whole = transcript.read_bytes()
    for unfinished in (b'{"type":"custom-title","customTitle":"half', '{"type":"custom-title","customTitle":"caf\u00e9'.encode()[:-1]):
        transcript.write_bytes(whole + unfinished)
        assert session_name(transcript) == "finished name"


async def test_a_transcript_whose_name_cannot_be_read_lists_the_session_by_its_project(tmp_path: Path) -> None:
    broken = membership(tmp_path, "broken")
    broken.transcript.write_text('{"type":"custom-title","sessionId":"broken"}\n')
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(broken, "startup"))
    assert await call(sessions, tmp_path) == {"sessions": [{"id": "broken", "name": "broken", "state": "not reported yet", "mode": "not reported yet", "watched": "no"}]}


def test_the_tool_is_named_and_described_from_its_body() -> None:
    tool = list_sessions_tool(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None), Overlays(Home(Path("/nonexistent"))))
    assert (tool.name, tool.required) == ("list_sessions", ())
    assert "running Claude Code sessions" in tool.description
