"""list_sessions answers with live sessions, labelled by Claude Code's own ai-title."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
from pipecat.services.llm_service import FunctionCallParams

from hands.core.events import Ended, Joined, PermissionRequested, Prompted
from hands.core.session import Membership, Permission, RequestId, SessionId
from hands.sessions.registry import Sessions
from hands.voice.tools import list_sessions_tool


def membership(tmp_path: Path, name: str) -> Membership:
    return Membership(SessionId(name), pid=1, pane=None, cwd=Path("/code") / name, transcript=tmp_path / f"{name}.jsonl")


def titled(path: Path, *titles: str) -> None:
    records = [{"type": "user", "message": {"content": 'grep "type":"ai-title" says hi'}}]
    records += [{"type": "ai-title", "aiTitle": title, "sessionId": path.stem} for title in titles]
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
    sessions = Sessions(permission_timeout=60.0)
    for event in (
        Joined(working),
        Prompted(working.id, at=1.0),
        Joined(untitled),
        Joined(blocked),
        PermissionRequested(blocked.id, at=2.0, request=RequestId("r"), permission=Permission("Bash", {})),
        Joined(ended),
        Ended(ended.id),
    ):
        sessions.apply(event)

    assert await call(sessions) == {
        "sessions": [
            {"id": "working", "title": "pipeline spike", "state": "working"},
            {"id": "untitled", "title": "untitled, in untitled", "state": "idle"},
            {"id": "blocked", "title": "auth refactor", "state": "waiting for permission to use Bash"},
        ]
    }


def test_the_tool_is_a_valid_pipecat_direct_function() -> None:
    wrapper = DirectFunctionWrapper(list_sessions_tool(Sessions(permission_timeout=60.0)))
    assert wrapper.name == "list_sessions"
    assert "running Claude Code sessions" in (wrapper.description or "")
