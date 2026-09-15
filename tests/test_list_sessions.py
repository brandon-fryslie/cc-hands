"""list_sessions answers with live sessions, labelled by Claude Code's own ai-title."""

import zlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
from pipecat.services.llm_service import FunctionCallParams

from hands.core.events import Ended, Joined, PermissionRequested, Prompted
from hands.core.session import Membership, Permission, RequestId, SessionId
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
    results: list[object] = []

    async def capture(result: object, **_: object) -> None:
        results.append(result)

    tool = list_sessions_tool(sessions)
    await tool(cast(FunctionCallParams, SimpleNamespace(result_callback=capture)))
    [result] = results
    return result


async def test_live_sessions_are_labelled_with_their_newest_ai_title(tmp_path: Path) -> None:
    working, untitled, blocked, ended = (membership(tmp_path, n) for n in ("working", "untitled", "blocked", "ended"))
    titled(working.transcript, "first guess", "pipeline spike")
    titled(blocked.transcript, "auth refactor")
    titled(ended.transcript, "old work")
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    for event in (
        Joined(working, "startup"),
        Prompted(working.id, at=1.0),
        Joined(untitled, "startup"),
        Joined(blocked, "startup"),
        PermissionRequested(blocked.id, at=2.0, request=RequestId("r"), permission=Permission("Bash", {})),
        Joined(ended, "startup"),
        Ended(ended.id, "prompt_input_exit"),
    ):
        await sessions.apply(event)

    assert await call(sessions) == {
        "sessions": [
            {"id": "working", "title": "pipeline spike", "state": "working"},
            {"id": "untitled", "title": "untitled, in untitled", "state": "idle"},
            {"id": "blocked", "title": "auth refactor", "state": "waiting for permission to use Bash"},
        ]
    }


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
    assert await call(sessions) == {"sessions": [{"id": "broken", "title": "untitled, in broken", "state": "idle"}]}


def test_the_tool_is_a_valid_pipecat_direct_function() -> None:
    wrapper = DirectFunctionWrapper(list_sessions_tool(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)))
    assert wrapper.name == "list_sessions"
    assert "running Claude Code sessions" in (wrapper.description or "")
