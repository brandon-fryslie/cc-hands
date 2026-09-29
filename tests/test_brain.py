"""The brain: hands' tools over MCP as a Claude Code client reaches them, and the brain's process as hands drives it."""

import json
import os
import stat
import sys
from pathlib import Path

import aiohttp
import pytest

from hands.brain.mcp import McpServer, serve_mcp
from hands.brain.process import BUILTIN_TOOLS, SLIM, BrainGone, Launch, NotLoggedIn, command, environment, start
from hands.sessions.audit import BrainAnswered, BrainAsked, BrainExited, BrainLaunched, BrainReady, Called, Entry, McpConnected
from hands.daemon.run import mind
from hands.voice.pipeline import AnthropicBackend, ClaudeCodeBackend
from hands.voice.summary import SummaryFailed, summariser
from hands.voice.tools import Result, audited, tool


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
    async with aiohttp.ClientSession() as client, client.post(server.url, json=message) as reply:
        return reply.status, (await reply.json() if reply.status == 200 else None)


async def test_a_client_opens_lists_and_calls_the_tools_and_each_call_is_audited() -> None:
    recorded: list[Entry] = []
    server = await serve_mcp([audited(tool(echo), recorded.append)], recorded.append)
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
        assert recorded == [McpConnected({"name": "claude-code"}, "2025-06-18"), Called("echo", {"text": "hi", "times": 2}, {"said": "hi hi"})]
    finally:
        await server.close()


async def test_a_call_the_tool_cannot_answer_is_told_to_the_model_and_one_the_server_cannot_is_an_error() -> None:
    server = await serve_mcp([tool(echo), tool(broken)], lambda _: None)
    try:
        _, wrong = await rpc(server, {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "echo", "arguments": {"words": "hi"}}})
        assert wrong == {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "echo was called with the wrong arguments: missing a required argument: 'text'"}], "isError": True}}
        _, failed = await rpc(server, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "broken"}})
        assert failed == {"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "broken failed: RuntimeError: the transcript went away"}], "isError": True}}
        _, unknown = await rpc(server, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "resume"}})
        assert unknown == {"jsonrpc": "2.0", "id": 3, "error": {"code": -32602, "message": "no tool resume"}}
        _, unasked = await rpc(server, {"jsonrpc": "2.0", "id": 4, "method": "resources/list"})
        assert unasked == {"jsonrpc": "2.0", "id": 4, "error": {"code": -32601, "message": "no method resources/list"}}
        async with aiohttp.ClientSession() as client, client.get(server.url) as stream:
            assert stream.status == 405
    finally:
        await server.close()


def launch(tmp: Path) -> Launch:
    return Launch(tmp / "brain", tmp / "brain" / "cwd", "claude-sonnet-5", "You are hands.", "http://127.0.0.1:1", '{"mcpServers": {}}')


def test_the_brain_is_slim_strict_and_never_asks_and_runs_on_its_own_login_through_the_proxy(tmp_path: Path) -> None:
    argv = command(launch(tmp_path))
    assert argv[:2] == ["claude", "-p"]
    assert argv[argv.index("--tools") + 1] == ",".join(BUILTIN_TOOLS)
    assert argv[argv.index("--allowedTools") + 1] == ",".join((*BUILTIN_TOOLS, "mcp__hands"))
    assert {"--strict-mcp-config", "--include-partial-messages", "--verbose"} <= set(argv)
    assert [argv[argv.index(flag) + 1] for flag in ("--permission-mode", "--setting-sources", "--system-prompt")] == ["dontAsk", "user", "You are hands."]
    # Nothing positional: --tools would swallow it.
    assert argv[-1] != ",".join(BUILTIN_TOOLS)
    env = environment(tmp_path / "brain", "http://127.0.0.1:1", {"PATH": "/bin", "ANTHROPIC_API_KEY": "sk", "CLAUDE_CODE_OAUTH_TOKEN": "t", "ANTHROPIC_BASE_URL": "http://elsewhere"})
    assert env == {"PATH": "/bin", **SLIM, "CLAUDE_CONFIG_DIR": str(tmp_path / "brain"), "ANTHROPIC_BASE_URL": "http://127.0.0.1:1"}


INIT = {"type": "system", "subtype": "init", "session_id": "b1", "model": "claude-sonnet-5", "tools": ["Read", "mcp__hands__list_sessions"], "mcp_servers": [{"name": "hands", "status": "connected"}]}
RESULT = {"type": "result", "subtype": "success", "is_error": False, "num_turns": 1, "duration_ms": 812, "result": "Two."}


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A `claude` first on PATH that reports its login from LOGGED_IN, and as the brain answers each stdin line with RESULT."""
    script = tmp_path / "bin" / "claude"
    script.parent.mkdir()
    script.write_text(f"""#!{sys.executable}
import json, os, sys
if sys.argv[1:3] == ["auth", "status"]:
    print(json.dumps({{"loggedIn": os.environ["LOGGED_IN"] == "1"}}))
    sys.exit(0)
if "json" in sys.argv and "stream-json" not in sys.argv:
    turn = sys.stdin.read()
    print(json.dumps({{"type": "result", "is_error": turn == "fail", "result": "Summed: " + turn}}))
    sys.exit(0)
print(json.dumps({INIT!r}), flush=True)
print("not json", file=sys.stderr, flush=True)
for line in sys.stdin:
    if json.loads(line)["message"]["content"] == "die":
        sys.exit(3)
    print(json.dumps({{"type": "stream_event"}}), flush=True)
    print(json.dumps({RESULT!r}), flush=True)
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("LOGGED_IN", "1")
    return script


async def test_a_turn_is_written_to_stdin_and_ends_at_the_result_line_with_both_ends_in_the_log(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        assert await brain.ask("what is running?") == BrainAnswered("success", False, 1, 812)
        assert await brain.ask("and now?") == BrainAnswered("success", False, 1, 812)
    finally:
        await brain.stop()
    assert (tmp_path / "brain" / "cwd").is_dir()
    # The init line is read whenever it comes, which is not ordered against the first write to stdin.
    ready = BrainReady("b1", "claude-sonnet-5", ("Read", "mcp__hands__list_sessions"), {"hands": "connected"})
    assert ready in recorded
    assert [entry for entry in recorded if entry != ready] == [
        BrainLaunched(brain.pid, tmp_path / "brain", tmp_path / "brain" / "cwd", "claude-sonnet-5"),
        BrainAsked("what is running?"),
        BrainAnswered("success", False, 1, 812),
        BrainAsked("and now?"),
        BrainAnswered("success", False, 1, 812),
        BrainExited(-15, "not json"),
    ]


async def test_a_brain_that_dies_mid_turn_fails_the_turn_and_says_how_it_ended(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    with pytest.raises(BrainGone, match="exited"):
        await brain.ask("die")
    assert await brain.exited() == 3
    assert [entry for entry in recorded if isinstance(entry, BrainExited)] == [BrainExited(3, "not json")]


async def test_a_brain_with_no_login_is_never_started_and_the_error_names_the_command(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    recorded: list[Entry] = []
    with pytest.raises(NotLoggedIn, match=f"CLAUDE_CONFIG_DIR={tmp_path / 'brain'} claude auth login"):
        await start(launch(tmp_path), recorded.append)
    assert recorded == []


def test_the_brain_config_is_one_line_of_json_naming_only_hands() -> None:
    server = McpServer(url="http://127.0.0.1:9/mcp", runner=None)  # pyright: ignore[reportArgumentType]
    assert json.loads(server.config()) == {"mcpServers": {"hands": {"type": "http", "url": "http://127.0.0.1:9/mcp"}}}


async def test_the_summariser_on_the_brain_asks_a_one_shot_claude_on_the_same_login_and_says_a_failed_answer(tmp_path: Path, fake_claude: Path) -> None:
    summarise = summariser(ClaudeCodeBackend(base_url="https://api.anthropic.com", model="claude-sonnet-5", config_dir=tmp_path / "brain"), "Sum it up.", 200, 10.0)
    assert await summarise("the tests ran") == "Summed: the tests ran"
    with pytest.raises(SummaryFailed, match="claude -p failed"):
        await summarise("fail")


async def test_the_run_starts_the_brain_beside_hands_mcp_server_for_the_claude_variant_alone(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    async with mind(AnthropicBackend(base_url="https://api.anthropic.com", api_key="k", model="m"), [], "http://127.0.0.1:1", recorded.append) as watches:
        assert watches == ()
    claude = ClaudeCodeBackend(base_url="https://api.anthropic.com", model="claude-sonnet-5", config_dir=tmp_path / "brain")
    async with mind(claude, [tool(echo)], "http://127.0.0.1:1", recorded.append) as watches:
        assert [watch.name for watch in watches] == ["the brain"]
        [launched] = [entry for entry in recorded if isinstance(entry, BrainLaunched)]
        assert launched.cwd == tmp_path / "brain" / "cwd"
    assert isinstance(recorded[-1], BrainExited)


async def test_a_run_on_a_brain_with_no_login_stops_naming_the_command(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    claude = ClaudeCodeBackend(base_url="https://api.anthropic.com", model="claude-sonnet-5", config_dir=tmp_path / "brain")
    with pytest.raises(SystemExit, match="claude auth login"):
        async with mind(claude, [], "http://127.0.0.1:1", lambda _: None):
            pass
