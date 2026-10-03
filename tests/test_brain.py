"""The brain: hands' tools over MCP as a Claude Code client reaches them, the brain's process as hands drives it, and the
side questions hands asks of a Claude Code of their own."""

import asyncio
import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import aiohttp
import pytest
from loguru import logger

from hands.brain.mcp import McpServer, serve_mcp
from hands.brain.asides import AsideFailed, Asides, aside_command
from hands.brain.process import SLIM, Brain, BrainGone, Launch, NotLoggedIn, Station, Unstartable, Untaken, _listen, command, environment, logged_in, start, workdir  # pyright: ignore[reportPrivateUsage]
from hands.sessions.payload import Payload
from hands.sessions.audit import AsideAnswered, BrainAnswered, BrainAsked, BrainExited, BrainLaunched, BrainOffered, BrainRefused, Called, Entry, McpConnected
from pipecat.services.anthropic.llm import AnthropicLLMService

from hands.brain.stage import BrainStage
from hands.core.session import SessionId, pasted
from hands.core.wire import Exchanged, Fork, Garbled, MainTurn, Message, Reached, Sent, Streamed
from hands.core.wire import Text as Said
from hands.daemon.cli import main
from hands.daemon.run import mind
from hands.sessions.proxy import Wire
from hands.sessions.sentences import Sentences
from hands.sessions.wrapper import MARK
from hands.sessions.home import Home
from hands.sessions.registry import Sessions
from hands.voice.refocus import Refocus
from hands.voice.sentences import SummaryStore
from hands.voice.speech import Pushed, Tailed
from hands.voice.pipeline import AnthropicBackend, ClaudeCodeBackend, VoiceConfig
from hands.voice.summary import SummaryFailed, aside
from hands.voice.tools import Result, audited, tool
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


def station(tmp: Path) -> Station:
    return Station(tmp / "brain", workdir(tmp / "brain"), "claude-sonnet-5", "http://127.0.0.1:1")


def launch(tmp: Path, fritter: Path = Path("/nonexistent/fritter")) -> Launch:
    return Launch(station(tmp), "You are hands.", '{"mcpServers": {}}', SessionId("b1"), fritter)


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
    return Exchanged(exchange, SessionId(session), Fork(), "POST", "/v1/messages", 10, (), 0.0, 0.0, Reached(200, 0.0, 0.0, 10, Streamed(message)), False)


def test_the_brain_is_interactive_on_its_own_setup_beside_hands_server_and_its_own_login_through_the_proxy(tmp_path: Path) -> None:
    argv = command(launch(tmp_path), Path("/real/claude"), "http://127.0.0.1:7")
    # The real claude, interactive: no -p, and no pipe to speak over.
    assert argv[0] == "/real/claude" and "-p" not in argv and "--print" not in argv
    # Its tools, what it may do without asking, and its MCP servers are its config directory's: nothing here narrows them.
    assert not {"--tools", "--disallowedTools", "--permission-mode", "--strict-mcp-config", "--system-prompt", "--bare"} & set(argv)
    assert argv[argv.index("--allowedTools") + 1] == "mcp__hands"
    assert argv[argv.index("--mcp-config") + 1] == launch(tmp_path).mcp_config
    assert [argv[argv.index(flag) + 1] for flag in ("--setting-sources", "--append-system-prompt", "--session-id")] == ["user", "You are hands.", "b1"]
    assert json.loads(argv[argv.index("--settings") + 1]) == {"hooks": {
        event: [{"hooks": [{"type": "http", "url": f"http://127.0.0.1:7/{event}"}]}] for event in ("UserPromptSubmit", "Stop", "StopFailure", "PermissionRequest", "Elicitation")
    }}
    # A side question's Claude Code is the same slim one, closed whatever the brain's setup holds: no tools, no server.
    bare = aside_command(Path("/real/claude"), "claude-sonnet-5", SessionId("a1"), "what now?")
    assert bare[bare.index("--tools") + 1] == "" and bare[bare.index("--session-id") + 1] == "a1" and "--strict-mcp-config" in bare
    # Its options end before its prompt: --mcp-config takes every word up to the next option.
    assert bare[-2:] == ["--", "/btw what now?"]
    assert json.loads(bare[bare.index("--mcp-config") + 1]) == {"mcpServers": {}}
    assert not {"-p", "--print", "--settings", "--allowedTools", "--append-system-prompt"} & set(bare)
    env = environment(tmp_path / "brain", "http://127.0.0.1:1", {
        "PATH": "/bin",
        "ANTHROPIC_API_KEY": "sk",
        "CLAUDE_CODE_OAUTH_TOKEN": "t",
        "ANTHROPIC_BASE_URL": "http://elsewhere",
        # A daemon started inside a tapped session inherits the session's tap; the brain is not that session.
        "FRITTER_TAP": "http://127.0.0.1:40000",
        "HTTPS_PROXY": "http://127.0.0.1:40000",
        "NODE_EXTRA_CA_CERTS": "/tmp/fritter-1/trusted.pem",
        "FRITTER_OUTER_HTTPS_PROXY": "http://corp:3128",
    })
    assert env == {"PATH": "/bin", "HTTPS_PROXY": "http://corp:3128", **SLIM, "CLAUDE_CONFIG_DIR": str(tmp_path / "brain"), "ANTHROPIC_BASE_URL": "http://127.0.0.1:1"}
    # The account's claude.ai connectors stay out of every request, whatever the brain's own setup names.
    assert env["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"
    # No turn opens but the ones hands types: no background task and no scheduled prompt opens one of its own.
    assert env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == env["CLAUDE_CODE_DISABLE_CRON"] == "1"


async def test_a_turn_is_typed_behind_a_space_and_ends_at_its_stop_hook_with_both_ends_in_the_log(tmp_path: Path, fake_claude: Path, fritter: Path) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path, fritter), recorded.append)
    try:
        assert await brain.ask("what is running?") == BrainAnswered("p1", None)
        assert await brain.ask("/and now?") == BrainAnswered("p2", None)
    finally:
        await brain.stop()
    assert (tmp_path / "brain" / "cwd").is_dir()
    # Behind a space, so a turn that opens with a slash is the words it is and not a command.
    assert typed(tmp_path) == [["prompt", " what is running?"], ["prompt", " /and now?"]]
    exited = recorded[-1]
    assert isinstance(exited, BrainExited)
    assert recorded == [
        BrainLaunched(brain.pid, tmp_path / "brain", tmp_path / "brain" / "cwd", "claude-sonnet-5"),
        BrainAsked("what is running?"),
        BrainAnswered("p1", None),
        BrainAsked("/and now?"),
        BrainAnswered("p2", None),
        exited,
    ]


async def test_what_the_brains_setup_would_ask_about_is_refused_and_its_turn_still_ends(tmp_path: Path, fake_claude: Path, fritter: Path) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path, fritter), recorded.append)
    try:
        assert await asyncio.wait_for(brain.ask("write"), 10) == BrainAnswered("p1", None)
    finally:
        await brain.stop()
    assert recorded[1:5] == [BrainAsked("write"), BrainRefused("p1", "PermissionRequest", "Write"), BrainRefused("p1", "Elicitation", "probe"), BrainAnswered("p1", None)]


async def test_a_dialog_is_answered_no_and_said_between_turns_and_when_its_body_does_not_parse() -> None:
    hooks: asyncio.Queue[Payload] = asyncio.Queue()
    listener, url = await _listen(hooks)  # pyright: ignore[reportPrivateUsage]
    try:
        async with aiohttp.ClientSession() as client:
            async with client.post(f"{url}/Elicitation", data=b"not json") as reply:
                assert reply.status == 200 and (await reply.json())["hookSpecificOutput"]["action"] == "decline"
    finally:
        await listener.cleanup()
    recorded: list[Entry] = []
    brain = object.__new__(Brain)
    brain.session = SessionId("b1")
    brain._record = recorded.append  # pyright: ignore[reportPrivateUsage]
    brain._turn = None  # pyright: ignore[reportPrivateUsage]
    # An MCP server asking while it connects, with no turn in flight and so no prompt id.
    brain._hook(Payload({"hook_event_name": "Elicitation", "session_id": "b1", "mcp_server_name": "probe", "message": "Which?"}))  # pyright: ignore[reportPrivateUsage]
    assert recorded == [BrainRefused(None, "Elicitation", "probe")]


async def test_a_turn_the_api_fails_ends_at_its_stop_failure_hook_saying_what_failed_it(tmp_path: Path, fake_claude: Path, fritter: Path) -> None:
    brain = await start(launch(tmp_path, fritter), lambda _entry: None)
    try:
        assert await brain.ask("fail") == BrainAnswered("p1", "unknown: API Error: 400 refused")
        assert await brain.ask("and now?") == BrainAnswered("p2", None)
    finally:
        await brain.stop()


async def test_an_interrupt_is_escape_and_ends_the_turn_in_flight_and_the_next_turn_is_its_own(tmp_path: Path, fake_claude: Path, fritter: Path) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path, fritter), recorded.append)
    try:
        # With no turn in flight there is nothing to stop, and no Escape is pressed.
        brain.interrupt()
        waiting = asyncio.create_task(brain.ask("wait"))
        # As soon as it is typed, before Claude Code has said it took it: the Escape waits for that.
        await until(lambda: [" wait"] == [line[1] for line in typed(tmp_path) if line[0] == "prompt"])
        brain.interrupt()
        # Pressed once the turn is taken, not once it ends: the turn waiting on the Escape never ends without it.
        assert await asyncio.wait_for(waiting, 5) == BrainAnswered("p1", None)
        assert await brain.ask("and now?") == BrainAnswered("p2", None)
    finally:
        await brain.stop()
    # The Escape puts the stopped prompt back in the input, and Ctrl-C clears it before the next is typed.
    assert typed(tmp_path) == [["prompt", " wait"], ["escape", ""], ["ctrl_c", " wait"], ["prompt", " and now?"]]


async def test_no_stop_presses_ctrl_c_within_claude_codes_exit_window_of_the_last(tmp_path: Path, fake_claude: Path, fritter: Path) -> None:
    # A Ctrl-C that finds the input empty arms Claude Code's exit, which a second within 800ms takes.
    brain = await start(launch(tmp_path, fritter), lambda _entry: None)
    pressed: list[float] = []
    try:
        for stops in (1, 2):
            waiting = asyncio.create_task(brain.ask("wait"))
            await until(lambda: sum(line[0] == "prompt" for line in typed(tmp_path)) == stops)
            brain.interrupt()
            await until(lambda: sum(line[0] == "ctrl_c" for line in typed(tmp_path)) == stops)
            pressed.append(asyncio.get_running_loop().time())
            assert await asyncio.wait_for(waiting, 5) == BrainAnswered(f"p{stops}", None)
    finally:
        await brain.stop()
    assert pressed[1] - pressed[0] > 0.8


async def test_a_side_question_is_the_prompt_a_claude_code_of_its_own_opens_with_answered_from_the_wire_and_its_claude_code_ended(
    tmp_path: Path, fake_claude: Path
) -> None:
    recorded: list[Entry] = []
    asides = Asides(station(tmp_path), recorded.append)
    asked = asyncio.create_task(asides.ask("what did\n\tthe read \ud83d say?"))
    await until(lambda: len(typed(tmp_path)) == 1)
    [[_, question, first]] = typed(tmp_path)
    # Another session's side question, and this one's request that failed and is asked again, answer nothing.
    asides.hear(answered("elsewhere", "Not this."))
    asides.hear(replace(answered(first, "x"), reply=Reached(529, 0.0, 0.0, 10, Garbled("overloaded"))))
    await asyncio.sleep(0.1)
    assert not asked.done()
    asides.hear(answered(first, "It said four."))
    assert await asked == "It said four."
    # The command and its question whole, newline and all, with a tab as its spaces and half an emoji spelled out.
    assert question == "what did\n    the read \\ud83d say?"
    assert running(tmp_path) == []
    # The next question has a Claude Code of its own, under a session of its own, which carries nothing of the first.
    again = asyncio.create_task(asides.ask("and then?"))
    await until(lambda: len(typed(tmp_path)) == 2)
    second = typed(tmp_path)[1][2]
    assert second != first
    # A late answer to the first finds no question waiting on it.
    asides.hear(answered(first, "Too late."))
    asides.hear(answered(second, "Two."))
    assert await again == "Two."
    assert running(tmp_path) == []
    assert [(entry.question, entry.reply, entry.failed, entry.session) for entry in recorded if isinstance(entry, AsideAnswered)] == [
        ("what did\n\tthe read \ud83d say?", "It said four.", False, first),
        ("and then?", "Two.", False, second),
    ]


async def test_a_side_question_with_no_answer_fails_saying_why_and_leaves_no_claude_code_running(
    tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: list[Entry] = []
    asides = Asides(station(tmp_path), recorded.append)

    def sessions_asked(question: str) -> list[str]:
        return [session for kind, text, session in typed(tmp_path) if (kind, text) == ("btw", question)]

    # Only the question that is never answered is given half a second: nothing else here races a clock.
    with monkeypatch.context() as short:
        short.setattr("hands.brain.asides.ASIDE_SECONDS", 0.5)
        with pytest.raises(AsideFailed, match="no answer in 0s; its Claude Code showed"):
            await asides.ask("hold")
    # A reply that did not end in words: Claude Code shows words of its own for it, which are never the answer.
    silent = asyncio.create_task(asides.ask("silent"))
    await until(lambda: sessions_asked("silent") != [])
    asides.hear(answered(sessions_asked("silent")[0], "", stop="tool_use"))
    with pytest.raises(AsideFailed, match="ended 'tool_use'"):
        await silent
    with pytest.raises(AsideFailed, match=r"exited \(3\) before it answered; it showed:\n(.|\n)*bye"):
        await asides.ask("die")
    # An asker that stops waiting ends its Claude Code too; one that leaves before its turn never had one.
    leaving = asyncio.create_task(asides.ask("stay"))
    await until(lambda: sessions_asked("stay") != [])
    behind = asyncio.create_task(asides.ask("behind"))
    await asyncio.sleep(0.1)
    behind.cancel()
    with pytest.raises(asyncio.CancelledError):
        await behind
    leaving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leaving
    assert running(tmp_path) == [] and sessions_asked("behind") == []
    # A fault of hands' own is the line's reason, as itself: never an asker that left.
    async def fault(*_: object) -> None:
        raise RuntimeError("no thread")

    with monkeypatch.context() as broken:
        broken.setattr("hands.brain.asides.spawn", fault)
        with pytest.raises(RuntimeError, match="no thread"):
            await asides.ask("broken?")
    # With no claude on PATH there is no Claude Code to ask.
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    with pytest.raises(AsideFailed, match="no claude on PATH"):
        await asides.ask("anyone?")
    said = [entry for entry in recorded if isinstance(entry, AsideAnswered)]
    assert [(entry.question, entry.reply.split(";")[0].split(":")[0], entry.failed) for entry in said] == [
        ("hold", "no answer in 0s", True),
        ("silent", "the model's reply ended 'tool_use' with ''", True),
        ("die", "its Claude Code exited (3) before it answered", True),
        ("behind", "its asker stopped waiting", True),
        ("stay", "its asker stopped waiting", True),
        ("broken?", "RuntimeError('no thread')", True),
        ("anyone?", "no Claude Code to ask", True),
    ]
    # How long each waited its turn, and how long its Claude Code ran.
    [hold, _, _, behind_said, _, _, _] = said
    assert hold.waited < 0.5 <= hold.seconds
    assert behind_said.waited >= 0.1 and behind_said.seconds == 0


async def test_an_asker_told_to_leave_again_while_its_claude_code_is_ending_leaves_none_running(tmp_path: Path, fake_claude: Path) -> None:
    asides = Asides(station(tmp_path), lambda _entry: None)
    leaving = asyncio.create_task(asides.ask("stubborn"))
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
    tmp_path: Path, fake_claude: Path, fritter: Path
) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path, fritter), recorded.append)
    asides = Asides(station(tmp_path), recorded.append)
    try:
        stuck = asyncio.create_task(asides.ask("hold"))
        await until(lambda: any(line[0] == "btw" for line in typed(tmp_path)))
        assert await asyncio.wait_for(brain.ask("are you listening?"), 5) == BrainAnswered("p1", None)
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
    brain._hook(Payload({"hook_event_name": "StopFailure", "session_id": "b1", "prompt_id": "p1", "error": {"kind": "odd"}}))  # pyright: ignore[reportPrivateUsage]


def test_the_tools_a_brain_turn_offers_are_audited_when_they_change_and_a_turn_without_hands_tools_is_an_error() -> None:
    recorded: list[Entry] = []
    brain = object.__new__(Brain)
    brain.session = SessionId("b1")
    brain._record = recorded.append  # pyright: ignore[reportPrivateUsage]
    brain._offered = None  # pyright: ignore[reportPrivateUsage]
    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(message.record["message"]), level="ERROR")
    try:
        body = {"messages": [{"role": "user", "content": "hi"}], "tools": [{"name": "Read"}]}
        with_hands = {**body, "tools": [{"name": "Read"}, {"name": "mcp__hands__read_session"}]}
        brain.hear(Sent("x1", SessionId("b1"), MainTurn(None), with_hands))
        brain.hear(Sent("x2", SessionId("b1"), MainTurn(None), with_hands))
        brain.hear(Sent("x3", SessionId("elsewhere"), MainTurn(None), body))
        assert errors == []
        brain.hear(Sent("x4", SessionId("b1"), MainTurn(None), body))
    finally:
        logger.remove(sink)
    assert errors == ["the brain's turn went to the model without hands' tools: it did not connect to hands' MCP server (('Read',))"]
    # Said once for the first request and again only for the one that offered other tools; another session's are not the brain's.
    assert recorded == [BrainOffered(("Read", "mcp__hands__read_session")), BrainOffered(("Read",))]


async def test_a_brain_that_dies_mid_turn_fails_the_turn_and_says_once_how_it_ended(
    tmp_path: Path, fake_claude: Path, fritter: Path
) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path, fritter), recorded.append)
    with pytest.raises(BrainGone, match="exited"):
        await brain.ask("die")
    assert await brain.exited() == 3
    with pytest.raises(BrainGone, match="before it was asked"):
        await brain.ask("anyone?")
    # The watch that saw it die and the stop at teardown both wait on the one exit.
    await brain.stop()
    [exited] = [entry for entry in recorded if isinstance(entry, BrainExited)]
    assert exited.code == 3 and "bye" in exited.shown


async def test_a_turn_never_taken_fails_naming_the_setup_command_and_the_next_turn_is_its_own(
    tmp_path: Path, fake_claude: Path, fritter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hands.brain.process.TAKE_SECONDS", 0.3)
    brain = await start(launch(tmp_path, fritter), lambda _entry: None)
    try:
        with pytest.raises(Untaken, match=f"mkdir -p {tmp_path / 'brain' / 'cwd'} && cd {tmp_path / 'brain' / 'cwd'} && CLAUDE_CONFIG_DIR={tmp_path / 'brain'} claude"):
            await brain.ask("deaf")
        assert await brain.ask("and now?") == BrainAnswered("p2", None)
    finally:
        await brain.stop()


async def test_an_asker_that_stops_waiting_leaves_the_turn_to_its_stop_and_the_next_turn_gets_its_own(tmp_path: Path, fake_claude: Path, fritter: Path) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path, fritter), recorded.append)
    try:
        asked = asyncio.create_task(brain.ask("slow"))
        await asyncio.sleep(0.1)
        asked.cancel()
        assert await brain.ask("and now?") == BrainAnswered("p2", None)
    finally:
        await brain.stop()
    turns = [entry for entry in recorded if isinstance(entry, BrainAsked | BrainAnswered)]
    assert turns == [BrainAsked("slow"), BrainAnswered("p1", None), BrainAsked("and now?"), BrainAnswered("p2", None)]


# How a hands starts a Claude Code of its own, and the pids of what it started: its brain under fritter, or an aside's
# Claude Code run directly, as the leader of its own session.
BRAIN = """
brain = await start(launch, lambda _entry: None)
print(brain.pid, _child_of(brain.pid), flush=True)
"""
ASIDE = """
claude = await spawn(launch.station, [str(brain_claude()), "--session-id", "s1"])
print(claude.pid, flush=True)
"""


@pytest.mark.parametrize("started", [BRAIN, ASIDE], ids=["brain", "aside"])
async def test_a_hands_that_dies_without_stopping_its_claude_code_leaves_nothing_it_started_running(tmp_path: Path, fake_claude: Path, fritter: Path, started: str) -> None:
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
        out, err = await asyncio.wait_for(hands.communicate(pickle.dumps(launch(tmp_path, fritter))), 30)
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


async def test_a_fritter_that_cannot_be_run_is_refused_with_what_its_terminal_showed(tmp_path: Path, fake_claude: Path, fritter: Path) -> None:
    unrunnable = tmp_path / "fritter"
    shutil.copy(fritter, unrunnable)
    unrunnable.chmod(0o644)
    with pytest.raises(Unstartable, match=r"(?s)fritter exited \(126\).*[Pp]ermission denied"):
        await start(launch(tmp_path, unrunnable), lambda _entry: None)


async def test_a_brain_with_no_fritter_to_run_under_is_refused_naming_the_install(tmp_path: Path, fake_claude: Path) -> None:
    with pytest.raises(Unstartable, match="hands install-fritter"):
        await start(launch(tmp_path), lambda _entry: None)


def test_a_brain_with_no_login_is_refused_naming_the_command(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert logged_in(tmp_path / "brain", "http://127.0.0.1:1") == "brain@example.com"
    monkeypatch.setenv("LOGGED_IN", "0")
    with pytest.raises(NotLoggedIn, match=f"CLAUDE_CONFIG_DIR={tmp_path / 'brain'} claude"):
        logged_in(tmp_path / "brain", "http://127.0.0.1:1")


def test_hands_login_logs_the_brain_in_on_the_subscription_in_its_own_config_and_says_the_account(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not the brain's")
    # A hands shim ahead of the real claude would run the login at this terminal as a session, under fritter.
    shim = tmp_path / "shim" / "claude"
    shim.parent.mkdir()
    shim.write_text(f"#!/bin/sh\n{MARK}\nexit 99\n")
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim.parent}:{os.environ['PATH']}")
    assert main(["--home", str(tmp_path), "login"]) == 0
    # Claude Code's own login, run where the brain's login lives, with no credential of this shell's beside it.
    assert json.loads((tmp_path / "brain" / "login.json").read_text()) == {"argv": ["auth", "login", "--claudeai"], "credentials": []}
    assert capsys.readouterr().out.splitlines()[0] == f"the brain at {tmp_path / 'brain'} is logged in as brain@example.com"


def test_a_brain_logged_in_off_the_subscription_is_refused_naming_how_it_is(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_METHOD", "api_key")
    with pytest.raises(NotLoggedIn, match="logged in by api_key, not on the Claude subscription"):
        logged_in(tmp_path / "brain", "http://127.0.0.1:1")


def test_hands_login_that_claude_code_fails_exits_1_saying_so(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("LOGIN_EXIT", "3")
    assert main(["--home", str(tmp_path), "login"]) == 1
    assert capsys.readouterr().err == "hands login: `claude auth login` for the brain exited 3\n"


def test_the_brain_config_is_one_line_of_json_naming_only_hands() -> None:
    server = McpServer(url="http://127.0.0.1:9/mcp", token="t", runner=None)  # pyright: ignore[reportArgumentType]
    assert json.loads(server.config()) == {"mcpServers": {"hands": {"type": "http", "url": "http://127.0.0.1:9/mcp", "headers": {"Authorization": "Bearer t"}}}}


async def test_the_summariser_under_the_brain_asks_the_turn_as_a_side_question_and_says_a_failed_one() -> None:
    asked: list[str] = []

    async def ask(question: str) -> str:
        asked.append(question)
        if "fail" in question:
            raise AsideFailed("no answer in 120s")
        if "slow" in question:
            await asyncio.sleep(5)
        return "  The tests ran. "

    summarise = aside(ask, "Sum it up.", 0.1)
    assert await summarise("the tests ran") == "The tests ran."
    assert asked == ["Sum it up.\n\nSummarize this:\n\nthe tests ran"]
    with pytest.raises(SummaryFailed, match="no answer in 120s"):
        await summarise("fail")
    # The summary's own time holds, whatever the side question waited on: the ones asked before it, or its answer.
    with pytest.raises(SummaryFailed, match="no answer in 0s"):
        await summarise("slow")


async def test_the_run_starts_the_brain_beside_hands_mcp_server_for_the_claude_variant_alone(tmp_path: Path, fake_claude: Path, fritter: Path) -> None:
    recorded: list[Entry] = []
    wire = Wire(lambda _observed: None)
    api = VoiceConfig(llm=AnthropicBackend(base_url="https://api.anthropic.com", api_key="k", model="m"), whisper_model="w", voice=voices.DEFAULT)
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    refocus = Refocus(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=recorded.append), Home(tmp_path), recorded.append)
    async with mind(api, [], lambda: "", refocus, "http://127.0.0.1:1", wire, store, fritter, tmp_path / "audit", recorded.append) as minded:
        assert isinstance(minded.llm, AnthropicLLMService) and minded.watches == () and minded.telling == Pushed()
    claude = VoiceConfig(llm=ClaudeCodeBackend(model="claude-sonnet-5", config_dir=tmp_path / "brain", account="brain@example.com"), whisper_model="w", voice=voices.DEFAULT)
    async with mind(claude, [tool(echo)], lambda: "", refocus, "http://127.0.0.1:1", wire, store, fritter, tmp_path / "audit", recorded.append) as minded:
        assert isinstance(minded.llm, BrainStage) and minded.telling == Tailed()
        assert [watch.name for watch in minded.watches] == ["the brain", "the brain's turns", "the brain's context"]
        [launched] = [entry for entry in recorded if isinstance(entry, BrainLaunched)]
        assert launched.cwd == tmp_path / "brain" / "cwd"
        # The stage speaks from the wire while the brain runs, so a second one cannot join it.
        with pytest.raises(RuntimeError, match="joined the wire"), wire.joined(minded.llm):
            pass
    assert isinstance(recorded[-1], BrainExited)
    # Gone with the brain: the wire forwards everything again.
    with wire.joined(minded.llm):
        pass
