"""The brain: hands' tools over MCP as a Claude Code client reaches them, the brain's process as hands drives it, and the
side questions hands asks of a Claude Code of their own."""

import asyncio
import json
import os
import pickle
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import aiohttp
import pytest
from loguru import logger

from hands.core.front import FrontUnread, InFront
from hands.brain.mcp import TOOL_USE_ID, CallSpans, McpServer, serve_mcp
from hands.brain.asides import AsideFailed, AsideKind, Asides, Deadline, TimeLimit, Unanswered, Within, aside_command
from hands.brain.process import BROKEN, NOBODY, SLIM, STOPPED, TAKE_SECONDS, UNANSWERED, UNREAD, UNVOICED, Asked, Brain, BrainAnswered, BrainGone, Fresh, Launch, NotLoggedIn, Resumed, Station, Unstartable, Untaken, _listen, _Posted, _Turn, account_kept_out, command, conversation, environment, logged_in, start, workdir  # pyright: ignore[reportPrivateUsage]
from hands.core.effects import Allow, Deny
from hands.core.permissions import heard
from hands.core.session import Permission
from hands.sessions.hookconfig import PERMISSION_HOOK_TIMEOUT_SECONDS
from hands.sessions.audit import Entry, level, segment
from conftest import events, onboard
from pipecat.services.anthropic.llm import AnthropicLLMService

from hands.brain.stage import BrainStage
from hands.core.session import PromptText, SessionId, pasted
from hands.core.wire import Exchanged, Fork, Garbled, Heard, MainTurn, Message, MessageStarted, Reached, Send, Sent, Written
from hands.core.wire import Text as Said
from hands.daemon.cli import main
from hands.daemon import run
from hands.daemon.run import mind
from hands.daemon.starting import CannotStart
from hands.sessions.proxy import Wire
from hands.sessions.sentences import Sentences
from hands.sessions import wrapper
from hands.sessions.wrapper import MARK
from hands.sessions.home import Home
from hands.sessions.pseudoterminal import STOP_SECONDS
from hands.sessions.registry import Sessions
from hands.voice.refocus import Refocus
from hands.voice.sentences import SummaryStore
from hands.voice.speech import Pushed, Tailed
from hands.voice.backends import Account, AnthropicBackend, ClaudeCodeBackend
from hands.voice.beside import Noting
from hands.voice.pipeline import VoiceConfig
from hands.voice.summary import SummaryFailed, aside
from hands.sessions.wide import Fact, WideEvent, begun, continuing, here, root, unit, within
from hands.voice.tool import Result, tool
from hands.voice.tools import Called, audited
from hands.voice import voices


async def echo(text: str, times: int = 1) -> Result:
    """Say it back.

    Args:
        text: What to say.
        times: How many times.
    """
    return {"said": " ".join([text] * times)}


async def broken() -> Result:
    """Fail."""
    raise RuntimeError("the transcript went away")


async def rpc(server: McpServer, message: dict[str, object]) -> tuple[int, object]:
    async with aiohttp.ClientSession() as client, client.post(server.url, json=message, headers={"Authorization": f"Bearer {server.token}"}) as reply:
        return reply.status, (await reply.json() if reply.status == 200 else None)


def ran(recorded: Sequence[Entry]) -> list[tuple[str, str, Mapping[str, Fact]]]:
    return [(entry.event, entry.outcome, entry.facts) for entry in recorded if isinstance(entry, WideEvent)]


async def test_a_client_opens_lists_and_calls_the_tools_and_each_request_is_one_event() -> None:
    recorded: list[Entry] = []
    server = await serve_mcp([audited(tool(echo), recorded.append)], recorded.append, CallSpans())
    try:
        status, opened = await rpc(server, {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "clientInfo": {"name": "claude-code"}}})
        assert status == 200 and opened == {
            "jsonrpc": "2.0", "id": 0,
            "result": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "hands", "version": "0"}},
        }
        assert (await rpc(server, {"jsonrpc": "2.0", "method": "notifications/initialized"}))[0] == 202
        _, listed = await rpc(server, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert listed == {"jsonrpc": "2.0", "id": 1, "result": {"tools": [{
            "name": "echo", "description": "Say it back.",
            "inputSchema": {"type": "object", "properties": {"text": {"type": "string", "description": "What to say."}, "times": {"type": "integer", "description": "How many times."}}, "required": ["text"]},
        }]}}
        _, called = await rpc(server, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "echo", "arguments": {"text": "hi", "times": 2}}})
        assert called == {"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": '{"said": "hi hi"}'}], "isError": False}}
        assert ran(recorded) == [
            ("mcp.request", "ok", {"method": "initialize", "client": "claude-code", "client_version": None, "protocol": "2025-06-18"}),
            ("mcp.request", "ok", {"method": "tools/list"}),
            ("tool.run", "ok", {"tool": "echo", "called": Called({"text": "hi", "times": 2}, {"said": "hi hi"})}),
        ]
    finally:
        await server.close()


async def test_a_call_the_tool_cannot_answer_is_told_to_the_model_and_one_the_server_cannot_is_an_error() -> None:
    recorded: list[Entry] = []
    server = await serve_mcp([audited(tool(echo), recorded.append), audited(tool(broken), recorded.append)], recorded.append, CallSpans())
    errors, sink = failures()
    try:
        _, wrong = await rpc(server, {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "echo", "arguments": {"words": "hi"}}})
        refusal = "echo was called with the wrong arguments: missing a required argument: 'text'"
        # A refusal is a result, as Claude Code hands it to the model unwrapped; its event is what ends failed.
        assert wrong == {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": json.dumps({"error": refusal})}], "isError": False}}
        _, failed = await rpc(server, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "broken"}})
        assert failed == {"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "broken failed: RuntimeError: the transcript went away"}], "isError": True}}
        # What the tool raised ends where the model is told, so it is said there, on the terminal, once.
        assert errors == ["broken failed: RuntimeError: the transcript went away"]
        _, unknown = await rpc(server, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "resume"}})
        assert unknown == {"jsonrpc": "2.0", "id": 3, "error": {"code": -32602, "message": "no tool resume taking arguments None"}}
        _, unshaped = await rpc(server, {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "echo", "arguments": '{"text": "hi"}'}})
        assert unshaped == {"jsonrpc": "2.0", "id": 4, "error": {"code": -32602, "message": """no tool echo taking arguments '{"text": "hi"}'"""}}
        _, unasked = await rpc(server, {"jsonrpc": "2.0", "id": 5, "method": "resources/list"})
        assert unasked == {"jsonrpc": "2.0", "id": 5, "error": {"code": -32601, "message": "no method resources/list"}}
        # Claude Code asks it of every server it opens: the known no is no failure.
        _, discovered = await rpc(server, {"jsonrpc": "2.0", "id": 6, "method": "server/discover"})
        assert discovered == {"jsonrpc": "2.0", "id": 6, "error": {"code": -32601, "message": "no method server/discover"}}
        async with aiohttp.ClientSession() as client, client.post(server.url, data=b"{not json", headers={"Authorization": f"Bearer {server.token}"}) as garbled:
            assert (await garbled.json())["error"]["code"] == -32700
        async with aiohttp.ClientSession() as client, client.get(server.url, headers={"Authorization": f"Bearer {server.token}"}) as stream:
            assert stream.status == 405
        assert [(event, outcome, facts.get("tool", facts.get("method"))) for event, outcome, facts in ran(recorded)] == [
            ("tool.run", "failed", "echo"),
            ("tool.run", "failed", "broken"),
            ("mcp.request", "failed", "tools/call"),
            ("mcp.request", "failed", "tools/call"),
            ("mcp.request", "failed", "resources/list"),
            ("mcp.request", "ok", "server/discover"),
            ("mcp.request", "failed", None),
        ]
        [garbled_event] = [entry for entry in recorded if isinstance(entry, WideEvent)][-1:]
        assert garbled_event.error is not None and garbled_event.error.startswith("not JSON: ")
    finally:
        logger.remove(sink)
        await server.close()


async def test_a_call_a_turn_is_running_is_run_as_a_part_of_the_turns_call_and_one_no_turn_is_running_is_a_root() -> None:
    recorded: list[Entry] = []
    spans = CallSpans()
    server = await serve_mcp([audited(tool(echo), recorded.append)], recorded.append, spans)
    with unit("voice.turn", recorded.append):
        call = within(here())
    spans.opened("toolu_1", call)
    try:
        for id, used in enumerate(("toolu_1", "toolu_2")):
            await rpc(server, {"jsonrpc": "2.0", "id": id, "method": "tools/call", "params": {"name": "echo", "arguments": {"text": "hi"}, "_meta": {TOOL_USE_ID: used}}})
        spans.ended(["toolu_1"])
        await rpc(server, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "echo", "arguments": {"text": "hi"}, "_meta": {TOOL_USE_ID: "toolu_1"}}})
    finally:
        await server.close()
    _, in_turn, outside, after = recorded
    assert isinstance(in_turn, WideEvent) and (in_turn.trace_id, in_turn.parent_id) == (call.trace_id, call.span_id)
    assert all(isinstance(run, WideEvent) and run.parent_id is None and run.trace_id != call.trace_id for run in (outside, after))


async def test_a_request_without_the_brains_token_reaches_no_tool() -> None:
    recorded: list[Entry] = []
    server = await serve_mcp([audited(tool(echo), recorded.append)], recorded.append, CallSpans())
    call ={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "echo", "arguments": {"text": "hi"}}}
    try:
        async with aiohttp.ClientSession() as client:
            # What a web page can send with no preflight: a text/plain POST, with no token or the wrong one.
            for headers in ({"Content-Type": "text/plain"}, {"Authorization": "Bearer guessed"}):
                async with client.post(server.url, data=json.dumps(call), headers=headers) as reply:
                    assert reply.status == 401
        assert recorded == []
    finally:
        await server.close()


def station(tmp: Path) -> Station:
    return Station(tmp / "brain", workdir(tmp / "brain"), "claude-sonnet-5", "http://127.0.0.1:1", dict(os.environ))


def unasked(asked: Asked) -> None:
    """A turn's teller of permissions, for a turn that asks none."""
    pytest.fail(f"a turn that asks nothing held {asked.permission}")


def launch(tmp: Path) -> Launch:
    return Launch(station(tmp), Account("claude.ai", "brain@example.com"), "You are hands.", '{"mcpServers": {}}', Fresh(SessionId("b1"), None))


def typed(tmp: Path) -> list[list[str]]:
    """What the fake claude read from its terminal, in order."""
    path = tmp / "typed.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


async def until(check: Callable[[], bool]) -> None:
    for _ in range(200):
        if check():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("never")


def running(tmp: Path) -> list[str]:
    """The fake Claude Codes still running."""
    return subprocess.run(["pgrep", "-f", str(tmp / "bin" / "claude")], capture_output=True, text=True).stdout.split()


def answered(session: str, words: str, stop: str = "end_turn", exchange: str = "x1") -> Exchanged:
    message = Message("m1", "claude-sonnet-5", (Said(words),) if words else (), stop, {})
    return Exchanged(exchange, SessionId(session), Fork(), "POST", "/v1/messages", 10, (), 0.0, 0.0, Reached(200, 0.0, 0.0, 10, Written(message, True)), False, root())


def test_the_brain_is_interactive_on_its_own_setup_beside_hands_server_and_its_own_login_through_the_proxy(tmp_path: Path) -> None:
    argv = command(launch(tmp_path), Path("/real/claude"), "http://127.0.0.1:7")
    # The real claude, interactive: no -p, and no pipe to speak over.
    assert argv[0] == "/real/claude" and "-p" not in argv and "--print" not in argv
    # Its tools, what it may do without asking, and its MCP servers are its config directory's: nothing here narrows them.
    assert not {"--tools", "--disallowedTools", "--permission-mode", "--strict-mcp-config", "--system-prompt", "--bare"} & set(argv)
    assert argv[argv.index("--allowedTools") + 1 : argv.index("--plugin-dir")] == ["mcp__hands", "Skill(hands:chat)", "Skill(hands:prompt)", "Skill(hands:recall)", "Skill(hands:start)"]
    assert argv[argv.index("--mcp-config") + 1] == launch(tmp_path).mcp_config
    # Beside its setup's skills, hands gives it its own: how it talks with the user, and how it writes a session's prompt.
    plugin = Path(argv[argv.index("--plugin-dir") + 1])
    assert json.loads((plugin / ".claude-plugin" / "plugin.json").read_text())["name"] == "hands"
    for skill in ("chat", "prompt"):
        assert (plugin / "skills" / skill / "SKILL.md").read_text().startswith(f"---\nname: {skill}\n")
    assert [argv[argv.index(flag) + 1] for flag in ("--setting-sources", "--append-system-prompt", "--session-id")] == ["user", "You are hands.", "b1"]
    # The instruction this hands writes goes on every request, never the one its conversation first had.
    assert argv[argv.index("--system-prompt-snapshot") + 1] == "off"
    settings = json.loads(argv[argv.index("--settings") + 1])
    # hands' tools are offered to the model directly, never deferred behind ToolSearch: set where neither the brain's
    # settings.json env nor hands' environment outranks it.
    assert settings["env"] == {"ENABLE_TOOL_SEARCH": "false"}
    hooks = settings["hooks"]
    assert {event: hooks.pop(event) for event in ("UserPromptSubmit", "Stop", "StopFailure", "Elicitation")} == {
        event: [{"hooks": [{"type": "http", "url": f"http://127.0.0.1:7/{event}"}]}] for event in ("UserPromptSubmit", "Stop", "StopFailure", "Elicitation")
    }
    # A permission is held while the user is asked, as long as a working session's is.
    assert hooks == {"PermissionRequest": [{"hooks": [{"type": "http", "url": "http://127.0.0.1:7/PermissionRequest", "timeout": PERMISSION_HOOK_TIMEOUT_SECONDS}]}]}
    # A side question's Claude Code is the same slim one, closed whatever the brain's setup holds: no tools, no server.
    bare = aside_command(Path("/real/claude"), "claude-sonnet-5", SessionId("a1"), "what now?")
    assert bare[bare.index("--tools") + 1] == "" and bare[bare.index("--session-id") + 1] == "a1" and "--strict-mcp-config" in bare
    # Its options end before its prompt: --mcp-config takes every word up to the next option.
    assert bare[-2:] == ["--", "/btw what now?"]
    assert json.loads(bare[bare.index("--mcp-config") + 1]) == {"mcpServers": {}}
    assert not {"-p", "--print", "--settings", "--allowedTools", "--append-system-prompt"} & set(bare)
    env = environment(tmp_path / "brain", "http://127.0.0.1:1", {
        "PATH": "/bin",
        # A command its setup's skill runs, firecrawl among them, is logged in by its own key or by what it stored under HOME.
        "HOME": "/home/u",
        "FIRECRAWL_API_KEY": "fc",
        "ANTHROPIC_API_KEY": "sk",
        "CLAUDE_CODE_OAUTH_TOKEN": "t",
        "ANTHROPIC_BASE_URL": "http://elsewhere",
        # A daemon started inside a tapped session inherits the session's tap; the brain is not that session.
        "FRITTER_TAP": "http://127.0.0.1:40000",
        "HTTPS_PROXY": "http://127.0.0.1:40000",
        "NODE_EXTRA_CA_CERTS": "/tmp/fritter-1/trusted.pem",
        "FRITTER_OUTER_HTTPS_PROXY": "http://corp:3128",
    })
    assert env == {"PATH": "/bin", "HOME": "/home/u", "FIRECRAWL_API_KEY": "fc", "HTTPS_PROXY": "http://corp:3128", **SLIM, "CLAUDE_CONFIG_DIR": str(tmp_path / "brain"), "ANTHROPIC_BASE_URL": "http://127.0.0.1:1"}
    # The account's claude.ai connectors stay out of every request, whatever the brain's own setup names.
    assert env["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"
    # No turn opens but the ones hands types: no background task and no scheduled prompt opens one of its own.
    assert env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == env["CLAUDE_CODE_DISABLE_CRON"] == "1"


async def test_a_turn_is_typed_behind_a_space_and_ends_at_its_stop_hook_and_the_brains_launch_turns_and_run_are_one_event_each(tmp_path: Path, fake_claude: Path, fritter: Path) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        assert await brain.ask("what is running?", unasked) == BrainAnswered("p1", None)
        assert await brain.ask("/and now?", unasked) == BrainAnswered("p2", None)
    finally:
        await brain.stop()
    assert (tmp_path / "brain" / "cwd").is_dir()
    # Behind a space, so a turn that opens with a slash is the words it is and not a command.
    assert typed(tmp_path) == [["prompt", " what is running?"], ["prompt", " /and now?"]]
    launched, first, second, ran = recorded
    assert isinstance(launched, WideEvent) and isinstance(first, WideEvent) and isinstance(second, WideEvent) and isinstance(ran, WideEvent)
    # Each turn is its own, with the prompt Claude Code took it as.
    assert [(turn.event, turn.outcome, turn.facts) for turn in (first, second)] == [
        ("brain.turn", "ok", {"prompts": ("p1",), "typings": 1, "offered": (), "others": ()}),
        ("brain.turn", "ok", {"prompts": ("p2",), "typings": 1, "offered": (), "others": ()}),
    ]
    assert (launched.event, launched.outcome, launched.parent_id) == ("brain.launch", "ok", None)
    assert launched.facts == {
        "session": "b1", "account": Account("claude.ai", "brain@example.com"), "model": "claude-sonnet-5", "config_dir": tmp_path / "brain", "cwd": tmp_path / "brain" / "cwd",
        "conversation": "fresh", "untranscribed": None, "fritter": fritter, "pid": brain.pid,
    }
    # Held once the brain is up, for the next start to resume.
    assert conversation(tmp_path / "brain", SessionId("b2")) == Fresh(SessionId("b2"), SessionId("b1"))
    transcript(tmp_path / "brain", "b1")
    assert conversation(tmp_path / "brain", SessionId("b2")) == Resumed(SessionId("b1"))
    # Its run is a part of its launch, from its input coming up to its process's end.
    assert (ran.event, ran.outcome, ran.trace_id, ran.parent_id) == ("brain.run", "ok", launched.trace_id, launched.span_id)
    assert ran.facts["pid"] == brain.pid and isinstance(ran.facts["code"], int) and isinstance(ran.facts["shown"], str)


def transcript(config_dir: Path, session: str) -> None:
    """Claude Code's transcript of `session`, under the project directory it names the brain's cwd by."""
    project = config_dir / "projects" / "-brain-cwd"
    project.mkdir(parents=True, exist_ok=True)
    (project / f"{session}.jsonl").write_text("{}\n")


def test_a_brain_starts_new_until_one_has_held_a_conversation_and_resumes_it_while_claude_code_keeps_its_transcript(tmp_path: Path) -> None:
    brain = tmp_path / "brain"
    brain.mkdir()
    assert conversation(brain, SessionId("n1")) == Fresh(SessionId("n1"), None)
    (brain / "conversation").write_text("c1\n")
    # A conversation Claude Code has no transcript of has nothing to resume: the one held is named as untranscribed.
    assert conversation(brain, SessionId("n1")) == Fresh(SessionId("n1"), SessionId("c1"))
    transcript(brain, "c1")
    assert conversation(brain, SessionId("n1")) == Resumed(SessionId("c1"))
    # Resumed under its own session, which Claude Code keeps: never a session id of its own beside it.
    argv = command(replace(launch(tmp_path), conversation=Resumed(SessionId("c1"))), Path("/real/claude"), "http://127.0.0.1:7")
    assert argv[argv.index("--resume") + 1] == "c1" and "--session-id" not in argv


def permissions(recorded: Sequence[Entry]) -> list[tuple[Fact, Fact, Fact]]:
    """The permissions the brain's setup asked about, as the log says each was settled."""
    return [(event.facts["prompt"], event.facts["tool"], event.facts["decision"]) for event in events(recorded, "brain.permission")]


async def test_a_permission_the_brains_setup_asks_about_holds_its_turn_until_answered_and_only_a_yes_runs_the_tool(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    held: list[Asked] = []
    notes = tmp_path / "notes.txt"
    no = heard("No, leave it.")
    assert isinstance(no, Deny)
    brain = await start(launch(tmp_path), recorded.append)
    try:
        turn = asyncio.create_task(brain.ask("write", held.append))
        await until(lambda: len(held) == 1)
        assert held[0].permission == Permission("Write", {"file_path": str(notes), "content": "hello"})
        # Held: the turn waits on the user, and the tool has not run.
        await asyncio.sleep(0.3)
        assert not turn.done() and not notes.exists()
        held[0].settle(Allow())
        assert await asyncio.wait_for(turn, 10) == BrainAnswered("p1", None)
        assert notes.read_text() == "hello"
        notes.unlink()
        turn = asyncio.create_task(brain.ask("write", held.append))
        await until(lambda: len(held) == 2)
        held[1].settle(no)
        assert await asyncio.wait_for(turn, 10) == BrainAnswered("p2", None)
        assert not notes.exists()
    finally:
        await brain.stop()
    # The brain heard each decision, the no with the user's words; nothing was typed into it while it held one.
    assert typed(tmp_path) == [
        ["prompt", " write"], ["permission", "allow", ""], ["elicitation", "decline"],
        ["prompt", " write"], ["permission", "deny", no.message], ["elicitation", "decline"],
    ]
    assert permissions(recorded) == [("p1", "Write", Allow()), ("p2", "Write", no)]
    assert [event.facts for event in events(recorded, "brain.elicitation")] == [{"prompt": "p1", "server": "probe"}, {"prompt": "p2", "server": "probe"}]


async def test_a_permission_nobody_answers_is_refused_at_its_deadline(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hands.brain.process.PERMISSION_DEADLINE_SECONDS", 0.3)
    recorded: list[Entry] = []
    held: list[Asked] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        with unit("voice.turn", recorded.append):
            asking = here()
            turn = asyncio.create_task(brain.ask("write", held.append))
            await until(lambda: len(held) == 1)
            # The turn's request, read off the wire while the turn holds its permission.
            brain.hear(Sent("x1", SessionId("b1"), MainTurn(None), {"messages": [], "tools": [{"name": "Write"}, {"name": "mcp__hands__read_session"}]}))
            assert await asyncio.wait_for(turn, 10) == BrainAnswered("p1", None)
    finally:
        await brain.stop()
    assert held[0].decision.result() == Deny(UNANSWERED)
    assert permissions(recorded) == [("p1", "Write", Deny(UNANSWERED))]
    # The brain's turn is a part of the voice turn that asked it, with the tools its request offered; the permission is a
    # part of the brain's turn it was held for, timed from its post to its answer.
    [brain_turn] = events(recorded, "brain.turn")
    assert (brain_turn.trace_id, brain_turn.parent_id) == (asking.trace_id, asking.span_id)
    assert brain_turn.facts == {"prompts": ("p1",), "typings": 1, "offered": ("Write", "mcp__hands__read_session"), "others": ()}
    [permission] = events(recorded, "brain.permission")
    assert (permission.trace_id, permission.parent_id) == (asking.trace_id, brain_turn.span_id) and permission.duration_ms >= 300
    assert not (tmp_path / "notes.txt").exists()


async def test_a_permission_posted_under_another_turns_prompt_is_refused_and_never_asked(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    held: list[Asked] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        assert await asyncio.wait_for(brain.ask("stray", held.append), 10) == BrainAnswered("p1", None)
    finally:
        await brain.stop()
    assert held == []
    assert permissions(recorded) == [("p0", "Write", Deny(NOBODY))]
    # No turn's own, so a part of the brain's launch.
    [launched] = events(recorded, "brain.launch")
    [permission] = events(recorded, "brain.permission")
    assert permission.parent_id == launched.span_id


async def test_a_turn_stopped_while_it_holds_a_permission_refuses_it(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    held: list[Asked] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        turn = asyncio.create_task(brain.ask("write", held.append))
        await until(lambda: len(held) == 1)
        brain.interrupt()
        assert await asyncio.wait_for(turn, 10) == BrainAnswered("p1", None)
    finally:
        await brain.stop()
    assert permissions(recorded) == [("p1", "Write", Deny(STOPPED))]
    assert not (tmp_path / "notes.txt").exists()


async def test_a_dialog_between_turns_or_with_a_body_that_does_not_parse_is_answered_no_and_said() -> None:
    hooks: asyncio.Queue[_Posted] = asyncio.Queue()  # pyright: ignore[reportPrivateUsage]
    listener, url = await _listen(hooks)  # pyright: ignore[reportPrivateUsage]
    recorded: list[Entry] = []
    brain = object.__new__(Brain)
    brain.session = SessionId("b1")
    brain._record = recorded.append  # pyright: ignore[reportPrivateUsage]
    with unit("brain.launch", recorded.append):
        launched = here()
    brain._launched = launched  # pyright: ignore[reportPrivateUsage]
    brain._turn = None  # pyright: ignore[reportPrivateUsage]
    brain._held = set()  # pyright: ignore[reportPrivateUsage]
    brain._typing = set()  # pyright: ignore[reportPrivateUsage]
    with continuing(launched):
        hearing = asyncio.create_task(brain._hear_hooks(hooks))  # pyright: ignore[reportPrivateUsage]

    async def answered(event: str, body: bytes) -> dict[str, object]:
        async with aiohttp.ClientSession() as client, client.post(f"{url}/{event}", data=body) as reply:
            assert reply.status == 200
            return (await reply.json())["hookSpecificOutput"]

    try:
        assert await answered("Elicitation", b"not json") == {"hookEventName": "Elicitation", "action": "decline"}
        assert await answered("PermissionRequest", b"not json") == {"hookEventName": "PermissionRequest", "decision": {"behavior": "deny", "message": UNREAD}}
        # An MCP server asking while it connects, and a permission with no turn in flight to ask the user in: no prompt id.
        elicited = {"hook_event_name": "Elicitation", "session_id": "b1", "mcp_server_name": "probe", "message": "Which?"}
        assert (await answered("Elicitation", json.dumps(elicited).encode()))["action"] == "decline"
        permission = {"hook_event_name": "PermissionRequest", "session_id": "b1", "tool_name": "Write", "tool_input": {"file_path": "/tmp/x"}}
        assert (await answered("PermissionRequest", json.dumps(permission).encode()))["decision"] == {"behavior": "deny", "message": NOBODY}
        # A dialog of questions is never put to the user by voice, in a turn or not: the brain is told to ask in words.
        questions: dict[str, object] = {**permission, "tool_name": "AskUserQuestion", "tool_input": {"questions": []}}
        assert (await answered("PermissionRequest", json.dumps(questions).encode()))["decision"] == {"behavior": "deny", "message": UNVOICED}
    finally:
        hearing.cancel()
        await listener.cleanup()
    # Each is one event, a part of the brain's launch, the unreadable ones failed saying so.
    unread, elicited_event = events(recorded, "brain.elicitation")
    assert unread.outcome == "failed" and unread.error is not None and unread.error.startswith("hands could not read the Elicitation hook")
    assert (elicited_event.outcome, elicited_event.facts) == ("ok", {"prompt": None, "server": "probe"})
    unreadable, *read = events(recorded, "brain.permission")
    assert unreadable.outcome == "failed" and unreadable.error is not None and unreadable.error.startswith("hands could not read the permission request")
    assert permissions(read) == [(None, "Write", Deny(NOBODY)), (None, "AskUserQuestion", Deny(UNVOICED))]
    assert {event.parent_id for event in events(recorded, "brain.permission") + events(recorded, "brain.elicitation")} == {launched.span_id}


async def test_a_turn_the_api_fails_ends_at_its_stop_failure_hook_saying_what_failed_it(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        assert await brain.ask("fail", unasked) == BrainAnswered("p1", "unknown: API Error: 400 refused")
        # An asker that stops waiting leaves the turn to run to its end, and its event still says what failed it.
        asking = asyncio.create_task(brain.ask("fail", unasked))
        await until(lambda: sum(line[0] == "prompt" for line in typed(tmp_path)) == 2)
        asking.cancel()
        await until(lambda: len(events(recorded, "brain.turn")) == 2)
        assert await brain.ask("and now?", unasked) == BrainAnswered("p3", None)
    finally:
        await brain.stop()
    failed, left, _ = events(recorded, "brain.turn")
    assert [(turn.outcome, turn.error, turn.facts["prompts"]) for turn in (failed, left)] == [
        ("failed", "the brain's turn ended in error: unknown: API Error: 400 refused", ("p1",)),
        ("failed", "the brain's turn ended in error: unknown: API Error: 400 refused", ("p2",)),
    ]


async def test_an_interrupt_is_escape_and_ends_the_turn_in_flight_and_the_next_turn_is_its_own(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        # With no turn in flight there is nothing to stop, and no Escape is pressed.
        brain.interrupt()
        waiting = asyncio.create_task(brain.ask("wait", unasked))
        # As soon as it is typed, before Claude Code has said it took it: the Escape waits for that.
        await until(lambda: [" wait"] == [line[1] for line in typed(tmp_path) if line[0] == "prompt"])
        brain.interrupt()
        # Pressed once the turn is taken, not once it ends: the turn waiting on the Escape never ends without it.
        assert await asyncio.wait_for(waiting, 5) == BrainAnswered("p1", None)
        assert await brain.ask("and now?", unasked) == BrainAnswered("p2", None)
    finally:
        await brain.stop()
    # The Escape puts the stopped prompt back in the input, and Ctrl-C clears it before the next is typed.
    assert typed(tmp_path) == [["prompt", " wait"], ["escape", ""], ["ctrl_c", " wait"], ["prompt", " and now?"]]


async def test_no_stop_presses_ctrl_c_within_claude_codes_exit_window_of_the_last(tmp_path: Path, fake_claude: Path) -> None:
    # A Ctrl-C that finds the input empty arms Claude Code's exit, which a second within 800ms takes.
    brain = await start(launch(tmp_path), lambda _entry: None)
    pressed: list[float] = []
    try:
        for stops in (1, 2):
            waiting = asyncio.create_task(brain.ask("wait", unasked))
            await until(lambda: sum(line[0] == "prompt" for line in typed(tmp_path)) == stops)
            brain.interrupt()
            await until(lambda: sum(line[0] == "ctrl_c" for line in typed(tmp_path)) == stops)
            pressed.append(asyncio.get_running_loop().time())
            assert await asyncio.wait_for(waiting, 5) == BrainAnswered(f"p{stops}", None)
    finally:
        await brain.stop()
    assert pressed[1] - pressed[0] > 0.8


# An asker's deadline no question here comes near, but the one that is to pass it.
WAITED = Deadline(60.0)


async def test_a_side_question_is_the_prompt_a_claude_code_of_its_own_opens_with_answered_from_the_wire_and_its_claude_code_ended(
    tmp_path: Path, fake_claude: Path
) -> None:
    recorded: list[Entry] = []
    asides = Asides(station(tmp_path), recorded.append)
    asked = asyncio.create_task(asides.ask(AsideKind.SUMMARY, "what did\n\tthe read \ud83d say?", WAITED))
    await until(lambda: len(typed(tmp_path)) == 1)
    [[_, question, first]] = typed(tmp_path)
    # Another session's side question, and this one's request that failed and is asked again, answer nothing.
    asides.hear(answered("elsewhere", "Not this."))
    asides.hear(replace(answered(first, "x"), reply=Reached(529, 0.0, 0.0, 10, Garbled("overloaded"))))
    await asyncio.sleep(0.1)
    assert not asked.done()
    # The question's own request is inside its unit of work; another session's goes as it was routed.
    routed = Send(span=root())
    own = asides.adopted(Sent("x1", SessionId(first), Fork(), None), routed)
    assert asides.adopted(Sent("x2", SessionId("elsewhere"), Fork(), None), routed) == routed
    asides.hear(answered(first, "It said four."))
    assert await asked == "It said four."
    # The command and its question whole, newline and all, with a tab as its spaces and half an emoji spelled out.
    assert question == "what did\n    the read \\ud83d say?"
    assert running(tmp_path) == []
    # The next question has a Claude Code of its own, under a session of its own, which carries nothing of the first.
    # Asked inside another unit of work, as a background pass asks one, it is a part of that unit.
    with unit("pass", recorded.append):
        again = asyncio.create_task(asides.ask(AsideKind.NAME, "and then?", WAITED))
        passing = here()
    await until(lambda: len(typed(tmp_path)) == 2)
    second = typed(tmp_path)[1][2]
    assert second != first
    # A late answer to the first finds no question waiting on it.
    asides.hear(answered(first, "Too late."))
    asides.hear(answered(second, "Two."))
    assert await again == "Two."
    assert running(tmp_path) == []
    # One event for each question, answered: what it was for, what was asked and answered, and under which session.
    assert [
        (event.outcome, event.facts["kind"], event.facts["question"], event.facts["reply"], event.facts["aside_session"], "unanswered" in event.facts)
        for event in events(recorded, "brain.aside")
    ] == [
        ("ok", "summary", "what did\n\tthe read \ud83d say?", "It said four.", first, False),
        ("ok", "name", "and then?", "Two.", second, False),
    ]
    [alone, within_pass] = events(recorded, "brain.aside")
    assert alone.parent_id is None and (within_pass.trace_id, within_pass.parent_id) == (passing.trace_id, passing.span_id)
    assert (own.span.trace_id, own.span.parent_id) == (alone.trace_id, alone.span_id)


async def test_a_side_question_with_no_answer_fails_saying_why_and_leaves_no_claude_code_running(
    tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: list[Entry] = []
    asides = Asides(station(tmp_path), recorded.append)

    def sessions_asked(question: str) -> list[str]:
        return [session for kind, text, session in typed(tmp_path) if (kind, text) == ("btw", question)]

    # Only the questions that are never answered are given less than a second: nothing else here races a clock.
    with pytest.raises(AsideFailed, match=r"no answer within Deadline\(seconds=0.5\)"):
        await asides.ask(AsideKind.LINE, "hold", Deadline(0.5))
    # A reply that did not end in words: Claude Code shows words of its own for it, which are never the answer.
    silent = asyncio.create_task(asides.ask(AsideKind.LINE, "silent", WAITED))
    await until(lambda: sessions_asked("silent") != [])
    asides.hear(answered(sessions_asked("silent")[0], "", stop="tool_use"))
    with pytest.raises(AsideFailed, match="ended 'tool_use'"):
        await silent
    with pytest.raises(AsideFailed, match=r"exited \(3\) before it answered; it showed:\n(.|\n)*bye"):
        await asides.ask(AsideKind.LINE, "die", WAITED)
    # An asker that stops waiting ends its Claude Code too; one that leaves before its turn never had one, and one whose
    # deadline passes before its turn never had one either.
    leaving = asyncio.create_task(asides.ask(AsideKind.LINE, "stay", WAITED))
    await until(lambda: sessions_asked("stay") != [])
    behind = asyncio.create_task(asides.ask(AsideKind.LINE, "behind", WAITED))
    await asyncio.sleep(0.1)
    behind.cancel()
    with pytest.raises(asyncio.CancelledError):
        await behind
    with pytest.raises(AsideFailed, match=r"no answer within Deadline\(seconds=0.3\)"):
        await asides.ask(AsideKind.LINE, "late", Deadline(0.3))
    leaving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leaving
    assert running(tmp_path) == [] and sessions_asked("behind") == sessions_asked("late") == []
    # A fault of hands' own is the line's reason, as itself: never an asker that left.
    async def fault(*_: object) -> None:
        raise RuntimeError("no thread")

    with monkeypatch.context() as broken:
        broken.setattr("hands.brain.asides.spawn", fault)
        with pytest.raises(RuntimeError, match="no thread"):
            await asides.ask(AsideKind.LINE, "broken?", WAITED)
    # With no claude on the PATH hands was started with there is no Claude Code to ask.
    nowhere = station(tmp_path)
    nowhere = replace(nowhere, inherited={**nowhere.inherited, "PATH": str(tmp_path / "nowhere")})
    with pytest.raises(AsideFailed, match="no claude on PATH"):
        await Asides(nowhere, recorded.append).ask(AsideKind.LINE, "anyone?", WAITED)
    # One event for each question, however it ended: why it has no answer, typed, beside what was seen of it.
    said = events(recorded, "brain.aside")
    assert [(event.facts["question"], event.outcome, event.facts.get("unanswered"), (event.error or "").split(";")[0].split(":\n")[0]) for event in said] == [
        ("hold", "failed", Unanswered.TIMED_OUT, "AsideFailed: no answer within Deadline(seconds=0.5)"),
        ("silent", "failed", Unanswered.WORDLESS, "AsideFailed: the model's reply ended 'tool_use' with ''"),
        ("die", "failed", Unanswered.EXITED, "AsideFailed: its Claude Code exited (3) before it answered"),
        ("behind", "cancelled", None, ""),
        ("late", "failed", Unanswered.TIMED_OUT, "AsideFailed: no answer within Deadline(seconds=0.3)"),
        ("stay", "cancelled", None, ""),
        ("broken?", "failed", None, "RuntimeError: no thread"),
        ("anyone?", "failed", Unanswered.UNSTARTED, "AsideFailed: no Claude Code to ask: no claude on PATH but hands' shims, so there is no Claude Code for hands to run as its own"),
    ]
    assert all("reply" not in event.facts for event in said)
    # How long each waited its turn, the rest of its duration its Claude Code answering; none for one that never had a turn.
    [hold, _, _, behind_said, late, _, _, _] = said
    assert hold.facts["queued_ms"] < 500 <= hold.duration_ms  # pyright: ignore[reportOperatorIssue]
    assert "queued_ms" not in behind_said.facts and "queued_ms" not in late.facts and late.duration_ms >= 300
    # What its Claude Code showed as its asker stopped waiting: none for one that never had a Claude Code.
    assert ">" in str(hold.facts["shown"]) and "shown" not in late.facts
    # A question whose asker's deadline passed is an error; one whose asker left, as hands stopping leaves it, is not.
    assert [level(event) for event in (hold, behind_said, late)] == ["error", "info", "error"]


async def test_a_side_question_whose_claude_code_will_not_end_still_fails_at_its_deadline(tmp_path: Path, fake_claude: Path) -> None:
    asides = Asides(station(tmp_path), lambda _entry: None)
    started = asyncio.get_running_loop().time()
    # Its Claude Code is told to end as the deadline passes and does not: it is given the time any Claude Code is to end
    # in, never killed as it may be writing the brain's config directory, and then killed.
    with pytest.raises(AsideFailed, match=r"no answer within Deadline\(seconds=0.5\)"):
        await asides.ask(AsideKind.LINE, "stubborn", Deadline(0.5))
    assert asyncio.get_running_loop().time() - started < 0.5 + STOP_SECONDS + 1.5
    assert running(tmp_path) == []


async def test_an_asker_in_no_hurry_for_its_turn_has_its_whole_time_limit_once_it_has_it(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    asides = Asides(station(tmp_path), recorded.append)
    first = asyncio.create_task(asides.ask(AsideKind.SUMMARY, "first", WAITED))
    await until(lambda: len(typed(tmp_path)) == 1)
    patient = asyncio.create_task(asides.ask(AsideKind.LINE, "patient", TimeLimit(0.3)))
    # Longer behind the first than its time limit, which is not yet running.
    await asyncio.sleep(0.5)
    assert not patient.done()
    asides.hear(answered(typed(tmp_path)[0][2], "One."))
    assert await first == "One."
    # Never answered, it is given the whole of its time limit from its turn: every bound here is one a slow machine only
    # widens, as a timer is never run before its time but by the clock's resolution, and the log writes each to 0.001ms.
    with pytest.raises(AsideFailed, match=r"no answer within TimeLimit\(seconds=0.3\)"):
        await patient
    [_, waited] = events(recorded, "brain.aside")
    assert waited.facts["queued_ms"] >= 500  # pyright: ignore[reportOperatorIssue]
    assert waited.duration_ms - waited.facts["queued_ms"] >= 300 - time.get_clock_info("monotonic").resolution * 1000 - 0.001  # pyright: ignore[reportOperatorIssue]


async def test_an_asker_told_to_leave_again_while_its_claude_code_is_ending_leaves_none_running(tmp_path: Path, fake_claude: Path) -> None:
    asides = Asides(station(tmp_path), lambda _entry: None)
    leaving = asyncio.create_task(asides.ask(AsideKind.LINE, "stubborn", WAITED))
    await until(lambda: [line[0] for line in typed(tmp_path)] == ["btw"])
    leaving.cancel()
    # Its Claude Code is told to end and does not. The asker is told to leave again while it waits on that, as a daemon
    # shutting down tells it, and the Claude Code is killed rather than left.
    await until(lambda: [line[0] for line in typed(tmp_path)] == ["btw", "sigterm"])
    leaving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leaving
    await until(lambda: running(tmp_path) == [])


async def test_a_turn_is_typed_into_the_brain_at_once_while_a_side_question_waits_on_an_answer_that_never_comes(
    tmp_path: Path, fake_claude: Path
) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    asides = Asides(station(tmp_path), recorded.append)
    try:
        stuck = asyncio.create_task(asides.ask(AsideKind.SUMMARY, "hold", WAITED))
        await until(lambda: any(line[0] == "btw" for line in typed(tmp_path)))
        assert await asyncio.wait_for(brain.ask("are you listening?", unasked), 5) == BrainAnswered("p1", None)
        assert not stuck.done()
        stuck.cancel()
        await asyncio.wait({stuck})
    finally:
        await brain.stop()
    # The question went to a Claude Code that is not the brain, and nothing but the turn was typed into the brain.
    [question, turn] = typed(tmp_path)
    assert question[:2] == ["btw", "hold"] and question[2] != "b1"
    assert turn == ["prompt", " are you listening?"]


def test_text_passed_on_to_the_brain_is_typed_as_the_characters_it_shows() -> None:
    # What a terminal is told is dropped, line ends are newlines, a tab is its spaces, any other control is spelled out,
    # and a closing backslash is kept from the Return that sends it.
    assert pasted("\x1b[31mred\x1b[0m\r\nnext\rline\n\tgo\tthere\x07 C:\\") == "red\nnext\nline\n    go  there\\x07 C:\\ "
    assert pasted("plain\nwords") == "plain\nwords"
    # Half of an emoji that was cut in two is spelled out; a whole one is itself.
    assert pasted("cut \ud83d, whole \U0001f600") == "cut \\ud83d, whole \U0001f600"


def test_a_hook_with_a_field_that_does_not_parse_is_passed_over_and_hooks_are_still_heard() -> None:
    brain = object.__new__(Brain)
    brain.session = SessionId("b1")
    brain._turn = None  # pyright: ignore[reportPrivateUsage]
    # An error that is not text: the hook is logged and passed over, never raised out of the loop that hears hooks.
    brain._hook(json.dumps({"hook_event_name": "StopFailure", "session_id": "b1", "prompt_id": "p1", "error": {"kind": "odd"}}).encode())  # pyright: ignore[reportPrivateUsage]


def test_a_brain_turn_without_hands_tools_is_an_error() -> None:
    recorded: list[Entry] = []
    brain = object.__new__(Brain)
    brain.session = SessionId("b1")
    brain._record = recorded.append  # pyright: ignore[reportPrivateUsage]
    brain._turn = None  # pyright: ignore[reportPrivateUsage]
    errors, sink = failures()
    try:
        body = {"messages": [{"role": "user", "content": "hi"}], "tools": [{"name": "Read"}]}
        with_hands = {**body, "tools": [{"name": "Read"}, {"name": "mcp__hands__read_session"}]}
        brain.hear(Sent("x1", SessionId("b1"), MainTurn(None), with_hands))
        brain.hear(Sent("x2", SessionId("b1"), MainTurn(None), with_hands))
        brain.hear(Sent("x3", SessionId("elsewhere"), MainTurn(None), body))
        assert errors == []
        brain.hear(Sent("x4", SessionId("b1"), MainTurn(None), body))
        # Deferred, as 2.1.288 offers MCP tools under tool search: the server may be connected, and is not blamed.
        deferred = {**body, "tools": [{"name": "Read"}, {"name": "ToolSearch"}, {"name": "DeferredToolPlaceholder", "defer_loading": True}]}
        brain.hear(Sent("x5", SessionId("b1"), MainTurn(None), deferred))
    finally:
        logger.remove(sink)
    assert errors == [
        "the brain's turn went to the model without hands' tools: it did not connect to hands' MCP server (('Read',))",
        "the brain's turn went to the model without hands' tools and with ToolSearch: tool search is on though hands' --settings"
        " turn it off (('Read', 'ToolSearch', 'DeferredToolPlaceholder'))",
    ]
    # What a turn's requests offered is on its brain turn's event.
    assert recorded == []


async def test_a_brain_that_dies_mid_turn_fails_the_turn_and_says_once_how_it_ended(
    tmp_path: Path, fake_claude: Path
) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        # Bounded, so a brain that never dies fails here, naming the wait it hung in.
        async with asyncio.timeout(TAKE_SECONDS * 2):
            with pytest.raises(BrainGone, match="exited"):
                await brain.ask("die", unasked)
            assert await brain.exited() == 3
            with pytest.raises(BrainGone, match="before it was asked"):
                await brain.ask("anyone?", unasked)
    finally:
        # The watch that saw it die and this stop both wait on the one exit; a brain that did not die is ended here.
        await brain.stop()
    [ran] = events(recorded, "brain.run")
    shown = ran.facts["shown"]
    assert ran.facts["code"] == 3 and isinstance(shown, str) and "bye" in shown


async def test_a_brain_slow_to_read_its_terminal_is_typed_into_once_its_input_is_up(
    tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A loaded machine starts it slowly: a turn typed before it reads raw has its Return made a newline, and is untaken.
    monkeypatch.setenv("STARTS_AFTER", "1")
    brain = await start(launch(tmp_path), lambda _entry: None)
    try:
        async with asyncio.timeout(TAKE_SECONDS * 2):
            assert await brain.ask("what is running?", unasked) == BrainAnswered("p1", None)
    finally:
        await brain.stop()


async def test_a_turn_never_taken_fails_naming_hands_login_and_the_next_turn_is_its_own(
    tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    brain = await start(launch(tmp_path), lambda _entry: None)
    try:
        # Only the deaf turn is held to a short limit: the next is a real turn, given the brain's own.
        with monkeypatch.context() as short, pytest.raises(Untaken, match="`hands login` answers them"):
            short.setattr("hands.brain.process.TAKE_SECONDS", 0.3)
            await brain.ask("deaf", unasked)
        assert await brain.ask("and now?", unasked) == BrainAnswered("p2", None)
    finally:
        await brain.stop()


async def test_a_turn_claude_code_takes_after_it_ended_untaken_is_not_the_next_turns(
    tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        with monkeypatch.context() as short, pytest.raises(Untaken):
            short.setattr("hands.brain.process.TAKE_SECONDS", 0.3)
            await brain.ask("late", unasked)
        # Its prompt and its Stop come while the next turn is in flight, ahead of that turn's own: words the next turn's
        # are a part of are still not the next turn's.
        assert await asyncio.wait_for(brain.ask("lat", unasked), 10) == BrainAnswered("p2", None)
    finally:
        await brain.stop()
    _, answered_turn = events(recorded, "brain.turn")
    assert answered_turn.facts == {"prompts": ("p2",), "typings": 1, "offered": (), "others": ("p1",)}


async def test_a_turn_taken_late_that_runs_past_the_take_limit_leaves_the_next_turns_taken(
    tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every turn has half a second to be taken, and each late one would run a second ahead of what was typed behind it.
    monkeypatch.setattr("hands.brain.process.TAKE_SECONDS", 0.5)
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        with pytest.raises(Untaken):
            await brain.ask("later", unasked)
        assert await asyncio.wait_for(brain.ask("next", unasked), 10) == BrainAnswered("p2", None)
        # The same words asked again after one ended untaken: the late prompt takes the new turn, as it answers those
        # words, and runs to its Stop; the new turn's own then runs ahead of the turn after it, which still is taken.
        with pytest.raises(Untaken):
            await brain.ask("later", unasked)
        await asyncio.wait_for(brain.ask("later", unasked), 10)
        assert await asyncio.wait_for(brain.ask("after", unasked), 10) == BrainAnswered("p5", None)
    finally:
        await brain.stop()
    # Each turn that a prompt no turn's ran ahead of says which it stopped, Escape and Ctrl-C, before it was typed again.
    assert [(turn.facts["others"], turn.facts["typings"]) for turn in events(recorded, "brain.turn")] == [((), 1), (("p1",), 2), ((), 1), ((), 1), (("p4",), 2)]
    # Its Escape puts the stopped prompt and the turn queued behind it back in the input, and its Ctrl-C clears them.
    assert typed(tmp_path) == [
        ["prompt", " later"],
        ["escape", ""],
        ["ctrl_c", " later next"],
        ["prompt", " next"],
        ["prompt", " later"],
        ["prompt", " later"],
        ["escape", ""],
        ["ctrl_c", " later after"],
        ["prompt", " after"],
    ]


async def test_a_turn_taken_as_the_escape_that_stops_what_ran_ahead_of_it_goes_is_answered_by_its_typing_again(
    tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hands.brain.process.TAKE_SECONDS", 0.5)
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        with pytest.raises(Untaken):
            await brain.ask("later", unasked)
        # The Escape meant for the late prompt stops the turn's own p2, which posts no Stop: p3, its typing again, answers it.
        assert await asyncio.wait_for(brain.ask("racy", unasked), 10) == BrainAnswered("p3", None)
    finally:
        await brain.stop()
    assert typed(tmp_path) == [["prompt", " later"], ["escape", ""], ["ctrl_c", " racy"], ["prompt", " racy"]]
    _, answered = events(recorded, "brain.turn")
    assert answered.facts == {"prompts": ("p2", "p3"), "typings": 2, "offered": (), "others": ("p1",)}


async def test_an_interrupt_while_a_turn_waits_behind_a_late_prompt_stops_it_once_it_is_typed_again(
    tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hands.brain.process.TAKE_SECONDS", 0.5)
    brain = await start(launch(tmp_path), lambda _entry: None)
    try:
        with pytest.raises(Untaken):
            await brain.ask("later", unasked)
        asked = asyncio.ensure_future(brain.ask("wait", unasked))
        # Within its first wait, while the late prompt runs ahead of it.
        await asyncio.sleep(0.1)
        brain.interrupt()
        assert await asyncio.wait_for(asked, 10) == BrainAnswered("p2", None)
    finally:
        await brain.stop()


async def test_a_turn_is_typed_at_most_twice_when_what_ran_ahead_of_it_is_its_own_kept_as_other_words(
    tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hands.brain.process.TAKE_SECONDS", 0.3)
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        with pytest.raises(Untaken):
            await asyncio.wait_for(brain.ask("garbled", unasked), 10)
    finally:
        await brain.stop()
    assert typed(tmp_path) == [["prompt", " garbled"], ["escape", ""], ["ctrl_c", " garbled"], ["prompt", " garbled"]]
    [untaken] = events(recorded, "brain.turn")
    assert (untaken.facts["others"], untaken.facts["typings"]) == (("p1", "p2"), 2)


async def test_a_turn_is_taken_as_claude_code_keeps_what_it_typed(tmp_path: Path, fake_claude: Path) -> None:
    brain = await start(launch(tmp_path), lambda _entry: None)
    try:
        # A long paste comes back in its tags, and a closing backslash's space comes back trimmed.
        assert await asyncio.wait_for(brain.ask("one\ntwo\nthree\nfour", unasked), 10) == BrainAnswered("p1", None)
        assert await asyncio.wait_for(brain.ask("path C:\\", unasked), 10) == BrainAnswered("p2", None)
    finally:
        await brain.stop()


class _Jammed:
    """A typist whose every key fails as a terminal gone from under it does: not the Untyped a stale socket raises."""

    def type(self, text: object) -> None:
        raise OSError("jammed")

    press = type


def failures() -> tuple[list[str], int]:
    """What is said on the terminal as an error, as it is said, and the sink that hears it."""
    said: list[str] = []
    return said, logger.add(lambda message: said.append(message.record["message"]), level="ERROR")


async def test_a_turn_whose_typing_fails_unexpectedly_fails_its_asker_says_so_once_and_the_next_turn_is_its_own(
    tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    said, sink = failures()
    try:
        typist = brain._typist  # pyright: ignore[reportPrivateUsage]
        brain._typist = _Jammed()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
        # Nothing was typed, so Claude Code never takes the turn, and its stop ends it once it was not taken in time:
        # a short time, which the next turn, a real one, is not held to.
        with monkeypatch.context() as short, pytest.raises(OSError, match="jammed"):
            short.setattr("hands.brain.process.TAKE_SECONDS", 0.3)
            await asyncio.wait_for(brain.ask("what is running?", unasked), 10)
        brain._typist = typist  # pyright: ignore[reportPrivateUsage]
        assert await asyncio.wait_for(brain.ask("and now?", unasked), 10) == BrainAnswered("p1", None)
    finally:
        logger.remove(sink)
        await brain.stop()
    assert said == ["the brain's own work failed: OSError('jammed')"]
    failed, _ = events(recorded, "brain.turn")
    assert (failed.outcome, failed.error) == ("failed", "OSError: jammed")


async def test_a_permission_that_fails_unexpectedly_is_refused_stops_its_turn_and_says_so_once(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    asked: list[Asked] = []

    def asks(held: Asked) -> None:
        asked.append(held)
        raise RuntimeError("no voice")

    brain = await start(launch(tmp_path), recorded.append)
    said, sink = failures()
    try:
        with pytest.raises(RuntimeError, match="no voice"):
            await asyncio.wait_for(brain.ask("write", asks), 10)
        # The brain heard the refusal, so its dialog did not hold it: its Stop comes, and the next turn is its own.
        await until(lambda: ["permission", "deny", BROKEN] in typed(tmp_path))
        assert await asyncio.wait_for(brain.ask("and now?", unasked), 10) == BrainAnswered("p2", None)
    finally:
        logger.remove(sink)
        await brain.stop()
    assert said == ["the brain's own work failed: RuntimeError('no voice')"]
    # The user's side is told the refusal the brain heard, and the turn is stopped as an Escape stops it.
    assert [held.decision.result() for held in asked] == [Deny(BROKEN)]
    assert ["escape", ""] in typed(tmp_path)
    [permission] = events(recorded, "brain.permission")
    assert (permission.outcome, permission.error, permission.facts["decision"]) == ("failed", "RuntimeError: no voice", Deny(BROKEN))
    assert not (tmp_path / "notes.txt").exists()


async def test_a_stop_whose_keys_fail_unexpectedly_ends_its_turn_says_so_once_and_is_never_pressed_again(tmp_path: Path, fake_claude: Path) -> None:
    brain = await start(launch(tmp_path), lambda _entry: None)
    pressed: list[object] = []

    class Jammed(_Jammed):
        def press(self, key: object) -> None:
            pressed.append(key)
            raise OSError("jammed")

    said, sink = failures()
    try:
        waiting = asyncio.create_task(brain.ask("wait", unasked))
        await until(lambda: [" wait"] == [line[1] for line in typed(tmp_path) if line[0] == "prompt"])
        brain._typist = Jammed()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
        brain.interrupt()
        with pytest.raises(OSError, match="jammed"):
            await asyncio.wait_for(waiting, 5)
    finally:
        logger.remove(sink)
        await brain.stop()
    # A second stop would press Escape again, and its Ctrl-C would land inside Claude Code's exit window.
    assert (pressed, said) == (["escape"], ["the brain's own work failed: OSError('jammed')"])


async def test_a_hook_whose_hearing_fails_unexpectedly_is_said_once_and_the_hooks_after_it_are_still_heard() -> None:
    hooks: asyncio.Queue[_Posted] = asyncio.Queue()  # pyright: ignore[reportPrivateUsage]
    listener, url = await _listen(hooks)  # pyright: ignore[reportPrivateUsage]
    recorded: list[Entry] = []

    def record(entry: Entry) -> None:
        # The first event the log is handed cannot be written.
        if not recorded:
            recorded.append(entry)
            raise ValueError("the log is full")
        recorded.append(entry)

    brain = object.__new__(Brain)
    brain.session = SessionId("b1")
    brain._record = record  # pyright: ignore[reportPrivateUsage]
    with unit("brain.launch", lambda _entry: None):
        brain._launched = here()  # pyright: ignore[reportPrivateUsage]
    # The elicitation that breaks is a part of the turn in flight: it was declined before it was heard, so the turn runs on.
    loop = asyncio.get_running_loop()
    turn = _Turn(loop.create_future(), loop.create_future(), loop.create_future(), unasked, begun(), TAKE_SECONDS, PromptText(" probe"))  # pyright: ignore[reportPrivateUsage]
    turn.taken.set_result("p1")
    brain._turn = turn  # pyright: ignore[reportPrivateUsage]
    brain._held = set()  # pyright: ignore[reportPrivateUsage]
    brain._typing = set()  # pyright: ignore[reportPrivateUsage]
    hearing = asyncio.create_task(brain._hear_hooks(hooks))  # pyright: ignore[reportPrivateUsage]
    said, sink = failures()

    async def answered(event: str, body: Mapping[str, object]) -> dict[str, object]:
        async with aiohttp.ClientSession() as client, client.post(f"{url}/{event}", data=json.dumps(body).encode()) as reply:
            assert reply.status == 200
            return (await reply.json())["hookSpecificOutput"]

    try:
        elicited = {"hook_event_name": "Elicitation", "session_id": "b1", "prompt_id": "p1", "mcp_server_name": "probe", "message": "Which?"}
        assert (await answered("Elicitation", elicited))["action"] == "decline"
        permission = {"hook_event_name": "PermissionRequest", "session_id": "b1", "tool_name": "Write", "tool_input": {"file_path": "/tmp/x"}}
        assert (await answered("PermissionRequest", permission))["decision"] == {"behavior": "deny", "message": NOBODY}
    finally:
        logger.remove(sink)
        hearing.cancel()
        await listener.cleanup()
    assert said == ["the brain's own work failed: ValueError('the log is full')"]
    assert (turn.broken, turn.answered.done(), brain._turn) == (None, False, turn)  # pyright: ignore[reportPrivateUsage]
    assert [event.event for event in recorded if isinstance(event, WideEvent)] == ["brain.elicitation", "brain.permission"]


async def test_an_asker_that_stops_waiting_leaves_the_turn_to_its_stop_and_the_next_turn_gets_its_own(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        asked = asyncio.create_task(brain.ask("slow", unasked))
        await asyncio.sleep(0.1)
        asked.cancel()
        assert await brain.ask("and now?", unasked) == BrainAnswered("p2", None)
    finally:
        await brain.stop()
    assert [event.event for event in events(recorded, "brain.launch") + events(recorded, "brain.run")] == ["brain.launch", "brain.run"]


# How a hands starts a Claude Code of its own, and the pids of what it started: its brain under fritter, or an aside's
# Claude Code run directly, as the leader of its own session.
BRAIN = """
brain = await start(launch, lambda _entry: None)
print(brain.pid, _child_of(brain.pid), flush=True)
"""
ASIDE = """
claude = await spawn(launch.station, [str(brain_claude(launch.station.inherited)), "--session-id", "s1"])
print(claude.pid, flush=True)
"""


@pytest.mark.parametrize("started", [BRAIN, ASIDE], ids=["brain", "aside"])
async def test_a_hands_that_dies_without_stopping_its_claude_code_leaves_nothing_it_started_running(tmp_path: Path, fake_claude: Path, started: str) -> None:
    # hands killed outright: no stop, no cleanup, only the kernel's hangup of the terminal it held.
    # Its temp dir is the test's, so what a dead hands leaves there - the brain's socket dir - goes with the test.
    temp = Path(tempfile.mkdtemp(dir="/tmp"))  # short: the sockets in it are held to the unix socket path limit
    pids: list[int] = []

    def alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):  # gone, or the pid is now another user's
            return False
        return True

    script = "import asyncio, os, pickle, sys\nfrom hands.brain.process import _child_of, brain_claude, spawn, start\nasync def main():\n    launch = pickle.load(sys.stdin.buffer)\n"
    script += "".join(f"    {line}\n" for line in started.strip().splitlines()) + "    os._exit(0)\nasyncio.run(main())\n"
    hands = await asyncio.create_subprocess_exec(sys.executable, "-c", script, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**os.environ, "TMPDIR": str(temp)})
    try:
        out, err = await asyncio.wait_for(hands.communicate(pickle.dumps(launch(tmp_path))), 30)
        assert hands.returncode == 0, err.decode()
        pids = [int(pid) for pid in out.split()]
        assert pids

        await until(lambda: not any(map(alive, pids)))
    finally:
        # Whatever failed, nothing it started outlives the test: killing a hands that hangs is itself its brain's hangup.
        if hands.returncode is None:
            hands.kill()
            await hands.wait()
        for pid in filter(alive, pids):
            os.kill(pid, signal.SIGKILL)
        shutil.rmtree(temp)


# A loop that ends as the brain starts: asyncio.run cancels every task at once, the launch's included, wherever it is.
# Ended a tick later each time, so one ends in the tick the brain is spawned in, which no step in time is sure to hit;
# then a tenth of a second later each time, so others end past the spawn: in the launch's own stop of what it spawned, or
# in the stop of the brain it started, as the daemon stops the brain it holds.
# Run in a process of its own, since a loop that never closes would take the test run down with it.
SHUT_DOWN = """
import asyncio, os, pickle, sys, time
from hands.brain.process import start

launch = pickle.load(sys.stdin.buffer)

async def run() -> None:
    brain = await start(launch, lambda _entry: None)
    await brain.stop()

async def ends(ticks: int, seconds: float) -> None:
    asyncio.create_task(run())
    for _ in range(ticks):
        await asyncio.sleep(0)
    await asyncio.sleep(seconds)
    raise RuntimeError(time.monotonic())

slowest = 0.0
for ticks, seconds in [*((ticks, 0.0) for ticks in range(40)), *((0, tenths / 10) for tenths in range(1, 16))]:
    try:
        asyncio.run(ends(ticks, seconds))
    except RuntimeError as ended:
        slowest = max(slowest, time.monotonic() - ended.args[0])
    try:
        os.waitpid(-1, os.WNOHANG)
        sys.exit(f"a fritter was left unreaped when the loop ended {ticks} ticks and {seconds}s into the launch")
    except ChildProcessError:
        pass
print(slowest)
"""


async def test_a_loop_ended_as_the_brain_starts_closes_and_leaves_nothing_it_started_running(tmp_path: Path, fake_claude: Path) -> None:
    """The daemon's shutdown cancels the brain's launch wherever it is, and must not then wait for ever on what it spawned.

    Python 3.12's asyncio subprocesses did: cancelled before the transport's own task first ran, they waited for an exit
    nothing would deliver."""
    temp = Path(tempfile.mkdtemp(dir="/tmp"))  # short: the sockets in it are held to the unix socket path limit
    hands = await asyncio.create_subprocess_exec(sys.executable, "-c", SHUT_DOWN, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**os.environ, "TMPDIR": str(temp)})
    try:
        out, err = await asyncio.wait_for(hands.communicate(pickle.dumps(launch(tmp_path))), 60)
    except TimeoutError:
        raise AssertionError("a loop ended as the brain started never finished closing") from None
    finally:
        if hands.returncode is None:
            hands.kill()
            await hands.wait()
        shutil.rmtree(temp)
    assert hands.returncode == 0, err.decode()[-2000:]
    assert float(out) < 1.0, f"the slowest loop took {out.decode().strip()}s to close"
    # The claude each fritter started ends with the terminal its fritter's end hangs up.
    await until(lambda: not running(tmp_path))


async def test_a_fritter_that_cannot_be_run_is_refused_with_what_its_terminal_showed(tmp_path: Path, fake_claude: Path, fritter: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unrunnable = tmp_path / "fritter"
    shutil.copy(fritter, unrunnable)
    unrunnable.chmod(0o644)
    monkeypatch.setattr(wrapper, "PACKAGED", unrunnable)
    with pytest.raises(Unstartable, match=r"(?s)fritter exited \(126\).*[Pp]ermission denied"):
        await start(launch(tmp_path), lambda _entry: None)


async def test_a_brain_that_never_turns_its_input_on_is_refused_with_what_it_showed_and_left_not_running(
    tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hands.brain.process.START_SECONDS", 1.0)
    fake_claude.write_text("#!/bin/sh\necho stuck on a screen\nsleep 30\n")
    with pytest.raises(Unstartable, match=r"(?s)had not turned its input on \(.*bracketed paste.*\) after 1s.*stuck on a screen"):
        await start(launch(tmp_path), lambda _entry: None)
    await until(lambda: not running(tmp_path))


async def test_a_conversation_the_brain_cannot_come_up_in_is_let_go_so_the_next_start_begins_anew(
    tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hands.brain.process.START_SECONDS", 1.0)
    # As Claude Code 2.1.289 does with a transcript it cannot read.
    fake_claude.write_text("#!/bin/sh\necho No conversation found with session ID: c1\n")
    brain = tmp_path / "brain"
    brain.mkdir()
    (brain / "conversation").write_text("c1\n")
    transcript(brain, "c1")
    recorded: list[Entry] = []
    with pytest.raises(Unstartable, match="No conversation found"):
        await start(replace(launch(tmp_path), conversation=conversation(brain, SessionId("n1"))), recorded.append)
    [launched] = events(recorded, "brain.launch")
    assert (launched.outcome, launched.facts["conversation"], launched.facts["let_go"]) == ("failed", "resumed", "c1")
    assert conversation(brain, SessionId("n2")) == Fresh(SessionId("n2"), None)


def test_a_brain_with_no_login_is_refused_naming_the_command(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert logged_in(tmp_path / "brain", "http://127.0.0.1:1", os.environ) == Account("claude.ai", "brain@example.com")
    monkeypatch.setenv("LOGGED_IN", "0")
    with pytest.raises(NotLoggedIn, match="`hands login` gives it one"):
        logged_in(tmp_path / "brain", "http://127.0.0.1:1", os.environ)


def logins(home: Path) -> list[tuple[str, dict[str, object]]]:
    """Each `hands login`'s event in the audit log of `home`, as its outcome and its facts."""
    events = (json.loads(line) for line in segment(home / "audit", 0).read_text().splitlines())
    return [(event["outcome"], event["facts"]) for event in events if event["type"] == "WideEvent" and event["event"] == "brain.login"]


def test_hands_login_sets_a_new_brain_home_up_with_its_settings_then_claude_codes_first_run_where_the_brain_runs(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not the brain's")
    # A hands shim ahead of the real claude would run the first run at this terminal as a session, under fritter.
    shim = tmp_path / "shim" / "claude"
    shim.parent.mkdir()
    shim.write_text(f"#!/bin/sh\n{MARK}\nexit 99\n")
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim.parent}:{os.environ['PATH']}")
    home = tmp_path / "home"
    assert main(["--home", str(home), "login"]) == 0
    brain = home / "brain"
    assert json.loads((brain / "settings.json").read_text()) == {"syncClaudeAiSkills": False, "syncClaudeAiPlugins": False, "permissions": {"defaultMode": "default"}}
    account_kept_out(brain)
    # Claude Code's own first run, on the brain's settings sources, in the directory the brain runs in, where its screens,
    # the trust of that directory among them, are answered once at this terminal; its settings there before it started,
    # so that it never syncs the account's skills or plugins; with no credential of this shell's beside it.
    assert json.loads((brain / "login.json").read_text()) == {"argv": ["--setting-sources", "user"], "cwd": str(workdir(brain)), "settings": True, "credentials": []}
    assert capsys.readouterr().out.splitlines()[0] == f"the brain at {brain} is logged in as brain@example.com (claude.ai)"
    assert logins(home) == [("ok", {"settings_written": True, "first_run": f"no {brain / '.claude.json'}", "account": {"type": "Account", "method": "claude.ai", "holder": "brain@example.com"}})]


def test_hands_login_on_a_brain_home_logs_it_in_again_leaving_its_settings_as_they_are(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    brain = tmp_path / "brain"
    said = b'{"permissions": {"allow": ["Bash(lit:*)"], "defaultMode": "acceptEdits"},\n "syncClaudeAiSkills": false, "syncClaudeAiPlugins": false}'
    onboard(brain, said)
    assert main(["--home", str(tmp_path), "login"]) == 0
    assert (brain / "settings.json").read_bytes() == said
    assert json.loads((brain / "login.json").read_text())["argv"] == ["auth", "login", "--claudeai"]
    assert capsys.readouterr().out.splitlines() == [f"the brain at {brain} is logged in as brain@example.com (claude.ai)", "a hands already running started its brain on the login before: restart it to start the brain on this one"]
    assert logins(tmp_path) == [("ok", {"settings_written": False, "first_run": None, "account": {"type": "Account", "method": "claude.ai", "holder": "brain@example.com"}})]


def test_hands_login_on_a_brain_home_claude_code_never_finished_its_first_run_on_runs_it(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    # What `claude auth status` leaves, asked by a `hands run` before any `hands login`: its state, and no onboarding.
    brain = tmp_path / "brain"
    brain.mkdir()
    (brain / ".claude.json").write_text("{}")
    assert main(["--home", str(tmp_path), "login"]) == 0
    assert json.loads((brain / "login.json").read_text())["argv"] == ["--setting-sources", "user"]
    account_kept_out(brain)
    assert logins(tmp_path) == [("ok", {"settings_written": True, "first_run": "its onboarding unfinished", "account": {"type": "Account", "method": "claude.ai", "holder": "brain@example.com"}})]


def test_hands_login_on_a_brain_home_whose_directory_claude_code_was_never_told_to_trust_runs_its_first_run(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    # Onboarded, then quit at the trust screen: the brain would start on that screen.
    onboard(tmp_path / "brain", trusted=False)
    assert main(["--home", str(tmp_path), "login"]) == 0
    assert json.loads((tmp_path / "brain" / "login.json").read_text())["argv"] == ["--setting-sources", "user"]
    assert logins(tmp_path)[0][1]["first_run"] == f"{(tmp_path / 'brain' / 'cwd').resolve()} untrusted"


def test_hands_login_on_a_brain_home_whose_state_claude_code_cannot_have_written_runs_its_first_run(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    onboard(tmp_path / "brain")
    (tmp_path / "brain" / ".claude.json").write_text("[]")
    assert main(["--home", str(tmp_path), "login"]) == 0
    assert json.loads((tmp_path / "brain" / "login.json").read_text())["argv"] == ["--setting-sources", "user"]
    assert str(logins(tmp_path)[0][1]["first_run"]).startswith(f"{tmp_path / 'brain' / '.claude.json'} unreadable: ")


def test_hands_login_whose_first_run_was_quit_before_its_last_screen_exits_1_saying_so(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    monkeypatch.setenv("LOGIN_UNANSWERED", "1")
    assert main(["--home", str(tmp_path), "login"]) == 1
    assert capsys.readouterr().err == f"hands login: the brain has not been through Claude Code's first screens (no {tmp_path / 'brain' / '.claude.json'}): `hands login` answers them\n"


def test_hands_login_on_a_brain_whose_settings_would_sync_its_accounts_skills_runs_no_claude_code(tmp_path: Path, fake_claude: Path, capsys: pytest.CaptureFixture[str]) -> None:
    onboard(tmp_path / "brain", b'{"permissions": {"defaultMode": "default"}}')
    assert main(["--home", str(tmp_path), "login"]) == 1
    assert not (tmp_path / "brain" / "login.json").exists()
    assert "its account's syncClaudeAiSkills and syncClaudeAiPlugins" in capsys.readouterr().err


def test_hands_login_after_a_first_run_that_failed_runs_it_again(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    monkeypatch.setenv("LOGIN_EXIT", "130")
    assert main(["--home", str(tmp_path), "login"]) == 1
    monkeypatch.delenv("LOGIN_EXIT")
    assert main(["--home", str(tmp_path), "login"]) == 0
    assert json.loads((tmp_path / "brain" / "login.json").read_text())["argv"] == ["--setting-sources", "user"]
    assert [outcome for outcome, _ in logins(tmp_path)] == ["failed", "ok"]


def test_hands_login_on_a_brain_path_that_is_no_directory_exits_1_saying_so(tmp_path: Path, fake_claude: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "brain").write_text("")
    assert main(["--home", str(tmp_path), "login"]) == 1
    assert capsys.readouterr().err.startswith(f"hands login: [Errno 17] File exists: '{tmp_path / 'brain'}'")


def test_hands_login_console_logs_a_brain_in_with_an_anthropic_console_key(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    onboard(tmp_path / "brain")
    assert main(["--home", str(tmp_path), "login", "--console"]) == 0
    assert json.loads((tmp_path / "brain" / "login.json").read_text())["argv"] == ["auth", "login", "--console"]
    assert capsys.readouterr().out.splitlines()[0] == f"the brain at {tmp_path / 'brain'} is logged in as /login managed key (api_key)"
    assert logins(tmp_path)[0][1]["account"] == {"type": "Account", "method": "api_key", "holder": "/login managed key"}


def test_a_brain_on_any_anthropic_login_starts(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_METHOD", "api_key")
    assert logged_in(tmp_path / "brain", "http://127.0.0.1:1", os.environ) == Account("api_key", "/login managed key")


def test_a_brain_reaching_claude_through_a_cloud_provider_is_refused_naming_it(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_PROVIDER", "bedrock")
    with pytest.raises(NotLoggedIn, match="through bedrock, past hands' proxy"):
        logged_in(tmp_path / "brain", "http://127.0.0.1:1", os.environ)


@pytest.mark.parametrize(("onboarded", "said"), [(False, "the brain's first run of Claude Code exited 3"), (True, "`claude auth login` for the brain exited 3")], ids=["first", "again"])
def test_hands_login_that_claude_code_fails_exits_1_saying_so(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], onboarded: bool, said: str) -> None:
    monkeypatch.setenv("LOGIN_EXIT", "3")
    if onboarded:
        onboard(tmp_path / "brain")
    assert main(["--home", str(tmp_path), "login"]) == 1
    assert capsys.readouterr().err == f"hands login: {said}\n"
    assert logins(tmp_path) == [("failed", {"settings_written": not onboarded})]


def test_the_brain_config_is_one_line_of_json_naming_only_hands() -> None:
    server = McpServer(url="http://127.0.0.1:9/mcp", token="t", runner=None)  # pyright: ignore[reportArgumentType]
    assert json.loads(server.config()) == {"mcpServers": {"hands": {"type": "http", "url": "http://127.0.0.1:9/mcp", "headers": {"Authorization": "Bearer t"}}}}


async def test_the_summariser_under_the_brain_asks_the_turn_as_a_side_question_within_its_time_and_says_a_failed_one() -> None:
    asked: list[tuple[str, Within]] = []

    async def ask(question: str, within: Within) -> str:
        asked.append((question, within))
        if "fail" in question:
            raise AsideFailed(Unanswered.TIMED_OUT, "no answer by its deadline")
        return "  The tests ran. "

    summarise = aside(ask, "Sum it up.", 0.1)
    assert await summarise("the tests ran") == "The tests ran."
    # The summary's own time is the side question's deadline, over the ones asked before it and its answer alike.
    assert asked == [("Sum it up.\n\nSummarize this:\n\nthe tests ran", Deadline(0.1))]
    with pytest.raises(SummaryFailed, match="no answer by its deadline"):
        await summarise("fail")


async def test_the_run_starts_the_brain_beside_hands_mcp_server_for_the_claude_variant_alone(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    wire = Wire(lambda _observed: None)
    api = VoiceConfig(llm=AnthropicBackend(base_url="https://api.anthropic.com", api_key="k", model="m"), voice=voices.DEFAULT)
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    refocus = Refocus(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=recorded.append), Home(tmp_path), recorded.append)

    async def unread() -> InFront:
        return FrontUnread("not read in this test")

    async with mind(api, [], [], lambda: "", unread, lambda: "screen", lambda: "held key", refocus, "http://127.0.0.1:1", wire, store, tmp_path / "audit", "hands recall", recorded.append, os.environ) as minded:
        assert isinstance(minded.llm, AnthropicLLMService) and minded.watches == () and minded.telling == Pushed()
        # An API model's context is noted as the user's words arrive; the brain's stage notes its own.
        assert [type(stage) for stage in minded.noting] == [Noting]
    claude = VoiceConfig(llm=ClaudeCodeBackend(model="claude-sonnet-5", config_dir=tmp_path / "brain", account=Account("claude.ai", "brain@example.com")), voice=voices.DEFAULT)
    async with mind(claude, [tool(echo)], [], lambda: "", unread, lambda: "screen", lambda: "held key", refocus, "http://127.0.0.1:1", wire, store, tmp_path / "audit", "hands recall", recorded.append, os.environ) as minded:
        assert isinstance(minded.llm, BrainStage) and minded.telling == Tailed() and minded.noting == ()
        assert [watch.name for watch in minded.watches] == ["the brain", "the brain's turns", "the brain's context"]
        # The brain alone is behind the proxy, so it alone is given its usage read off the wire.
        assert list(minded.llm._tools) == ["mcp__hands__echo", "mcp__hands__context_usage"]  # pyright: ignore[reportPrivateUsage]
        [launched] = events(recorded, "brain.launch")
        assert (launched.facts["cwd"], launched.facts["account"]) == (tmp_path / "brain" / "cwd", Account("claude.ai", "brain@example.com"))
        # The stage speaks from the wire while the brain runs, so a second one cannot join it.
        with pytest.raises(RuntimeError, match="joined the wire"), wire.joined(minded.llm):
            pass
    assert isinstance(ran := recorded[-1], WideEvent) and ran.event == "brain.run"
    # Gone with the brain: the wire forwards everything again.
    with wire.joined(minded.llm):
        pass
    # The next run's brain takes the conversation up again, while Claude Code keeps its transcript.
    transcript(tmp_path / "brain", str(launched.facts["session"]))
    async with mind(claude, [tool(echo)], [], lambda: "", unread, lambda: "screen", lambda: "held key", refocus, "http://127.0.0.1:1", wire, store, tmp_path / "audit", "hands recall", recorded.append, os.environ) as minded:
        # Its usage is heard under the session it resumed, not one begun for it.
        wire.observe(Sent("x", SessionId(str(launched.facts["session"])), MainTurn(None), {}))
        wire.observe(Heard("x", MessageStarted("msg_x", "claude-sonnet-5", {"input_tokens": 7, "output_tokens": 1})))
        assert isinstance(minded.llm, BrainStage)
        reported = await minded.llm._tools["mcp__hands__context_usage"].body()  # pyright: ignore[reportPrivateUsage]
        assert reported["in_context_tokens"] == 8
    [_, again] = events(recorded, "brain.launch")
    assert (again.facts["session"], again.facts["conversation"]) == (launched.facts["session"], "resumed")


async def test_each_variants_model_is_told_the_personality_the_run_was_configured_with(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[Entry] = []
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    refocus = Refocus(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=recorded.append), Home(tmp_path), recorded.append)

    async def unread() -> InFront:
        return FrontUnread("not read in this test")

    def minding(llm: AnthropicBackend | ClaudeCodeBackend):
        return mind(VoiceConfig(llm=llm, voice=voices.DEFAULT, personality="Dry and wry."), [], [], lambda: "", unread, lambda: "screen", lambda: "held key", refocus, "http://127.0.0.1:1", Wire(lambda _observed: None), store, tmp_path / "audit", "hands recall", recorded.append, os.environ)

    async with minding(AnthropicBackend(base_url="https://api.anthropic.com", api_key="k", model="m")) as minded:
        assert isinstance(minded.llm, AnthropicLLMService)
        assert "\n\nDry and wry.\n\n" in str(minded.llm._settings.system_instruction)  # pyright: ignore[reportPrivateUsage]
    launched: list[Launch] = []

    async def starting(launch: Launch, _record: object) -> Brain:
        launched.append(launch)
        raise Unstartable("seen")

    monkeypatch.setattr(run, "start_brain", starting)
    with pytest.raises(CannotStart, match="^seen$"):
        async with minding(ClaudeCodeBackend(model="claude-sonnet-5", config_dir=tmp_path / "brain", account=Account("claude.ai", "brain@example.com"))):
            pass
    [launch] = launched
    assert "\n\nDry and wry.\n\n" in launch.instruction


async def test_a_brain_that_cannot_start_refuses_the_run_saying_why(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    refocus = Refocus(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=recorded.append), Home(tmp_path), recorded.append)

    async def unread() -> InFront:
        return FrontUnread("not read in this test")

    claude = VoiceConfig(llm=ClaudeCodeBackend(model="claude-sonnet-5", config_dir=tmp_path / "brain", account=Account("claude.ai", "brain@example.com")), voice=voices.DEFAULT)
    with pytest.raises(CannotStart, match="^no claude on PATH"):
        async with mind(claude, [], [], lambda: "", unread, lambda: "screen", lambda: "held key", refocus, "http://127.0.0.1:1", Wire(lambda _observed: None), store, tmp_path / "audit", "hands recall", recorded.append, {"PATH": str(tmp_path)}):
            pass
    # The launch that failed is one event, saying why; no brain ran, so there is no run.
    [launched] = events(recorded, "brain.launch")
    assert launched.outcome == "failed" and launched.error == "Unstartable: no claude on PATH but hands' shims, so there is no Claude Code for hands to run as its own"
    assert launched.facts["account"] == Account("claude.ai", "brain@example.com") and "pid" not in launched.facts
    assert events(recorded, "brain.run") == []


async def test_a_hands_built_without_its_fritter_refuses_the_run_naming_the_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[Entry] = []
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    refocus = Refocus(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=recorded.append), Home(tmp_path), recorded.append)
    missing = tmp_path / "package" / "bin" / "fritter"
    monkeypatch.setattr(wrapper, "PACKAGED", missing)

    async def unread() -> InFront:
        return FrontUnread("not read in this test")

    claude = VoiceConfig(llm=ClaudeCodeBackend(model="claude-sonnet-5", config_dir=tmp_path / "brain", account=Account("claude.ai", "brain@example.com")), voice=voices.DEFAULT)
    # The words `hands install-fritter` refuses the same hands with.
    refused = f"hands' package carries no fritter at {missing}: install hands again, or in a checkout, `uv sync --reinstall-package hands`"
    with pytest.raises(CannotStart, match=f"^{re.escape(refused)}$"):
        async with mind(claude, [], [], lambda: "", unread, lambda: "screen", lambda: "held key", refocus, "http://127.0.0.1:1", Wire(lambda _observed: None), store, tmp_path / "audit", "hands recall", recorded.append, dict(os.environ)):
            pass
    [launched] = events(recorded, "brain.launch")
    assert launched.outcome == "failed" and launched.error == f"Unstartable: {refused}"
    assert launched.facts["account"] == Account("claude.ai", "brain@example.com") and "fritter" not in launched.facts and "pid" not in launched.facts
    assert events(recorded, "brain.run") == []
