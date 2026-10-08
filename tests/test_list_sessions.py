"""list_sessions answers with live sessions, each by its project and its newest name, never Claude Code's ai-title."""

import zlib
import json
from pathlib import Path
from typing import NoReturn

import pytest


from hands.core.events import Ended, Joined, Launched, PermissionRequested, Prompted, StatusReported, Stopped, Taken
from hands.core.session import Membership, Permission, PromptId, RequestId, SessionId, Question
from hands.core.status import Busy, Idle, Report, Shell, Stamp, Waiting
from hands.core.turn import AgentId
from hands.sessions.focus import set_focus
from hands.sessions import tmux
from hands.sessions.home import Home
from hands.sessions.names import Names
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
    # No tmux server of the test's own runs: every session is in none.
    return await list_sessions_tool(sessions, Overlays(Home(tmp_path / "home")), Home(tmp_path / "home"), {"TMUX_TMPDIR": str(tmp_path)}).body()


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
        PermissionRequested(blocked.id, at=2.0, request=RequestId("r"), on=Permission("Bash", {}), mode="default", timeout=None),
        Joined(asking, "startup"),
        StatusReported(asking.id, Report(Waiting("input needed"), Stamp(1)), at=3.0),
        PermissionRequested(asking.id, at=3.0, request=RequestId("q"), on=Question((), {}), mode=None, timeout=None),
        Joined(ended, "startup"),
        Ended(ended.id, "prompt_input_exit"),
    ):
        await sessions.apply(event)
    Overlays(Home(tmp_path / "home")).set(working.id, "watched")
    set_focus(Home(tmp_path / "home"), working.id)

    assert await call(sessions, tmp_path) == {
        "sessions": [
            {"id": "working", "name": "working, pipeline spike", "state": "working", "mode": "accept edits mode", "overlay": "watched", "tmux": "not in tmux"},
            {"id": "untitled", "name": "untitled", "state": "idle", "mode": "not reported yet", "overlay": "normal", "tmux": "not in tmux"},
            {"id": "blocked", "name": "blocked, auth refactor", "state": "waiting for permission to use Bash", "mode": "manual mode", "overlay": "normal", "tmux": "not in tmux"},
            {"id": "asking", "name": "asking", "state": "waiting for the user to answer its question", "mode": "not reported yet", "overlay": "normal", "tmux": "not in tmux"},
        ],
        "focus": "working",
    }


async def test_a_session_that_joins_on_its_permission_request_is_listed_as_waiting_on_it(tmp_path: Path) -> None:
    """Its status is not read yet, and what it waits on is what its hook asked."""
    lagging = membership(tmp_path, "lagging")
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    for event in (Joined(lagging, "startup"), PermissionRequested(lagging.id, at=1.0, request=RequestId("r"), on=Permission("Bash", {}), mode="default", timeout=None)):
        await sessions.apply(event)
    assert await call(sessions, tmp_path) == {"sessions": [{"id": "lagging", "name": "lagging", "state": "waiting for permission to use Bash", "mode": "manual mode", "overlay": "normal", "tmux": "not in tmux"}], "focus": None}


async def test_a_session_at_its_prompt_with_a_background_shell_running_is_listed_idle(tmp_path: Path) -> None:
    """As seen live on 2.1.289: a session cleared while a background poll it started ran on reported shell, and sat at
    its prompt doing nothing."""
    cleared = membership(tmp_path, "cleared")
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    for event in (Joined(cleared, "clear"), StatusReported(cleared.id, Report(Shell(), Stamp(1)), at=1.0)):
        await sessions.apply(event)
    listed = {"id": "cleared", "name": "cleared", "state": "idle, with a shell command it started in the background still running", "mode": "not reported yet", "overlay": "normal", "tmux": "not in tmux"}
    assert await call(sessions, tmp_path) == {"sessions": [listed], "focus": None}


async def test_a_session_at_its_prompt_with_a_subagent_working_in_the_background_is_listed_idle(tmp_path: Path) -> None:
    """2.1.289 keeps busy from a background subagent's launch until the turn reporting it back ends; the turn that
    launched it has stopped, so the session sits at its prompt, and is said to."""
    waiting = membership(tmp_path, "waiting")
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    turn = PromptId("p1")
    for event in (
        Joined(waiting, "startup"),
        StatusReported(waiting.id, Report(Idle(), Stamp(900)), at=1.0),
        Prompted(waiting.id, at=2.0, mode=None, prompt=turn),
        StatusReported(waiting.id, Report(Busy(), Stamp(1000)), at=2.1),
        Launched(waiting.id, AgentId("a1"), Stamp(1200)),
        Launched(waiting.id, AgentId("a2"), Stamp(1300)),
        Stopped(waiting.id, "Started them.", mode=None, prompt=turn, again=False, heard=Stamp(1500), request=RequestId("stop")),
    ):
        await sessions.apply(event)
    listed = {"id": "waiting", "name": "waiting", "state": "idle, with two subagents it started in the background still working", "mode": "not reported yet", "overlay": "normal", "tmux": "not in tmux"}
    assert await call(sessions, tmp_path) == {"sessions": [listed], "focus": None}


async def test_a_session_idle_since_the_turn_that_renamed_it_is_listed_by_its_new_name(tmp_path: Path) -> None:
    """As seen live: a session whose last turn moved its work on was named for it after its Stop, and had no prompt since
    to hand Claude Code the name, so it stayed listed by the name of the work before."""
    idle, renamed = membership(tmp_path, "idle"), membership(tmp_path, "renamed")
    named(idle.transcript, "pr 231 review")
    named(renamed.transcript, "auth refactor")
    names = Names()
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None, names=names)
    for event in (Joined(idle, "startup"), Joined(renamed, "startup")):
        await sessions.apply(event)
    names.rename(idle.id, "monetization review", "pr 231 review")
    names.rename(renamed.id, "naming fix", "auth refactor")
    # The user's /rename since the decision outranks it.
    named(renamed.transcript, "auth refactor", "my own name")
    assert [listing.name for listing in sessions.live()] == ["monetization review", "my own name"]


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
    assert await call(sessions, tmp_path) == {"sessions": [{"id": "broken", "name": "broken", "state": "not reported yet", "mode": "not reported yet", "overlay": "normal", "tmux": "not in tmux"}], "focus": None}


async def test_a_pane_read_that_breaks_leaves_each_session_listed_and_saying_why_its_pane_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken(_environment: object, _pids: object, _processes: object) -> NoReturn:
        raise ValueError("not enough values to unpack")

    monkeypatch.setattr(tmux, "servers", broken)
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(membership(tmp_path, "plain"), "startup"))
    listed = {"id": "plain", "name": "plain", "state": "not reported yet", "mode": "not reported yet", "overlay": "normal", "tmux": {"cannot_read": "ValueError: not enough values to unpack"}}
    assert await call(sessions, tmp_path) == {"sessions": [listed], "focus": None}


def test_the_tool_is_named_and_described_from_its_body() -> None:
    tool = list_sessions_tool(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None), Overlays(Home(Path("/nonexistent"))), Home(Path("/nonexistent")), {})
    assert (tool.name, tool.required) == ("list_sessions", ())
    assert "running Claude Code sessions" in tool.description
