"""The brain: hands' tools over MCP as a Claude Code client reaches them, and the brain's process as hands drives it."""

import asyncio
import json
from pathlib import Path

import aiohttp
import pytest

from hands.brain.mcp import McpServer, serve_mcp
from hands.brain.process import BUILTIN_TOOLS, SLIM, BrainGone, Launch, NotLoggedIn, command, environment, logged_in, start, workdir
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
    async with aiohttp.ClientSession() as client, client.post(server.url, json=message, headers={"Authorization": f"Bearer {server.token}"}) as reply:
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
        async with aiohttp.ClientSession() as client, client.get(server.url, headers={"Authorization": f"Bearer {server.token}"}) as stream:
            assert stream.status == 405
    finally:
        await server.close()


async def test_a_request_without_the_brains_token_reaches_no_tool() -> None:
    recorded: list[Entry] = []
    server = await serve_mcp([audited(tool(echo), recorded.append)], recorded.append)
    call = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "echo", "arguments": {"text": "hi"}}}
    try:
        async with aiohttp.ClientSession() as client:
            # What a web page can send with no preflight: a text/plain POST, with no token or the wrong one.
            for headers in ({"Content-Type": "text/plain"}, {"Authorization": "Bearer guessed"}):
                async with client.post(server.url, data=json.dumps(call), headers=headers) as reply:
                    assert reply.status == 401
        assert recorded == []
    finally:
        await server.close()


def launch(tmp: Path) -> Launch:
    return Launch(tmp / "brain", workdir(tmp / "brain"), "claude-sonnet-5", "You are hands.", "http://127.0.0.1:1", '{"mcpServers": {}}')


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


async def test_a_brain_that_dies_mid_turn_fails_the_turn_and_says_once_how_it_ended(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    with pytest.raises(BrainGone, match="output ended"):
        await brain.ask("die")
    assert await brain.exited() == 3
    with pytest.raises(BrainGone, match="before it was asked"):
        await brain.ask("anyone?")
    # The watch that saw it die and the stop at teardown both wait on the one exit.
    await brain.stop()
    assert [entry for entry in recorded if isinstance(entry, BrainExited)] == [BrainExited(3, "not json")]


async def test_an_asker_that_stops_waiting_leaves_the_turn_to_its_result_and_the_next_turn_gets_its_own(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        asked = asyncio.create_task(brain.ask("slow"))
        await asyncio.sleep(0.1)
        asked.cancel()
        assert await brain.ask("and now?") == BrainAnswered("success", False, 1, 812)
    finally:
        await brain.stop()
    turns = [entry for entry in recorded if isinstance(entry, BrainAsked | BrainAnswered)]
    assert turns == [BrainAsked("slow"), BrainAnswered("success", False, 1, 812), BrainAsked("and now?"), BrainAnswered("success", False, 1, 812)]


async def test_a_brain_whose_output_breaks_fails_the_turn_is_killed_and_says_why(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    with pytest.raises(BrainGone, match="output ended"):
        await brain.ask("flood")
    with pytest.raises(ValueError):
        await brain.exited()
    assert [entry.code for entry in recorded if isinstance(entry, BrainExited)] == [-9]


def test_a_brain_with_no_login_is_refused_naming_the_command(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    logged_in(tmp_path / "brain", "http://127.0.0.1:1")
    monkeypatch.setenv("LOGGED_IN", "0")
    with pytest.raises(NotLoggedIn, match=f"CLAUDE_CONFIG_DIR={tmp_path / 'brain'} claude auth login"):
        logged_in(tmp_path / "brain", "http://127.0.0.1:1")


def test_the_brain_config_is_one_line_of_json_naming_only_hands() -> None:
    server = McpServer(url="http://127.0.0.1:9/mcp", token="t", runner=None)  # pyright: ignore[reportArgumentType]
    assert json.loads(server.config()) == {"mcpServers": {"hands": {"type": "http", "url": "http://127.0.0.1:9/mcp", "headers": {"Authorization": "Bearer t"}}}}


async def test_the_summariser_on_the_brain_asks_a_one_shot_claude_on_the_same_login_and_says_a_failed_answer(tmp_path: Path, fake_claude: Path) -> None:
    summarise = summariser(ClaudeCodeBackend(model="claude-sonnet-5", config_dir=tmp_path / "brain"), "http://127.0.0.1:9", "Sum it up.", 200, 10.0)
    # In the brain's own empty directory, never the daemon's, and through hands' proxy, never straight to Anthropic's API.
    assert await summarise("the tests ran") == f"Summed: the tests ran in {(tmp_path / 'brain' / 'cwd').resolve()} via http://127.0.0.1:9"
    with pytest.raises(SummaryFailed, match="claude -p failed"):
        await summarise("fail")


async def test_the_run_starts_the_brain_beside_hands_mcp_server_for_the_claude_variant_alone(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    async with mind(AnthropicBackend(base_url="https://api.anthropic.com", api_key="k", model="m"), [], "http://127.0.0.1:1", recorded.append) as watches:
        assert watches == ()
    claude = ClaudeCodeBackend(model="claude-sonnet-5", config_dir=tmp_path / "brain")
    async with mind(claude, [tool(echo)], "http://127.0.0.1:1", recorded.append) as watches:
        assert [watch.name for watch in watches] == ["the brain"]
        [launched] = [entry for entry in recorded if isinstance(entry, BrainLaunched)]
        assert launched.cwd == tmp_path / "brain" / "cwd"
    assert isinstance(recorded[-1], BrainExited)
