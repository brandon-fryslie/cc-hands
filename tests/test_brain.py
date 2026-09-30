"""The brain: hands' tools over MCP as a Claude Code client reaches them, and the brain's process as hands drives it."""

import asyncio
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import aiohttp
import pytest
from loguru import logger

from hands.brain.mcp import McpServer, serve_mcp
from hands.brain.process import BUILTIN_TOOLS, SLIM, Brain, BrainGone, ForkFailed, Launch, NotLoggedIn, Unstartable, Untaken, command, environment, logged_in, start, workdir
from hands.sessions.payload import Payload
from hands.sessions.audit import BrainAnswered, BrainAsked, BrainExited, BrainForked, BrainLaunched, Called, Entry, McpConnected
from pipecat.services.anthropic.llm import AnthropicLLMService

from hands.brain.stage import BrainStage
from hands.core.session import SessionId, pasted
from hands.core.wire import Exchanged, Fork, Garbled, MainTurn, Message, Reached, Sent, Streamed
from hands.core.wire import Text as Said
from hands.daemon.run import mind
from hands.sessions.proxy import Wire
from hands.sessions.registry import Sessions
from hands.sessions.sentences import Sentences
from hands.voice.sentences import SummaryStore
from hands.voice.speech import Pushed, Tailed
from hands.voice.pipeline import AnthropicBackend, ClaudeCodeBackend, VoiceConfig
from hands.voice.summary import SummaryFailed, aside
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


def launch(tmp: Path, fritter: Path = Path("/nonexistent/fritter")) -> Launch:
    return Launch(tmp / "brain", workdir(tmp / "brain"), "claude-sonnet-5", "You are hands.", "http://127.0.0.1:1", '{"mcpServers": {}}', SessionId("b1"), fritter)


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


def forked(session: str, question: str, exchange: str = "x1") -> Sent:
    """A fork's request on the wire, asking `question` as /btw's request does: after the history, in a message of its own."""
    body = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello."}, {"role": "user", "content": f"<system-reminder>{question}</system-reminder>"}]}
    return Sent(exchange, SessionId(session), Fork(), body)


def answered(session: str, words: str, stop: str = "end_turn", exchange: str = "x1") -> Exchanged:
    message = Message("m1", "claude-sonnet-5", (Said(words),) if words else (), stop, {})
    return Exchanged(exchange, SessionId(session), Fork(), "POST", "/v1/messages", 10, (), 0.0, 0.0, Reached(200, 0.0, 0.0, 10, Streamed(message)))


def test_the_brain_is_interactive_slim_strict_and_never_asks_and_runs_on_its_own_login_through_the_proxy(tmp_path: Path) -> None:
    argv = command(launch(tmp_path), Path("/real/claude"), "http://127.0.0.1:7")
    # The real claude, interactive: no -p, and no pipe to speak over.
    assert argv[0] == "/real/claude" and "-p" not in argv and "--print" not in argv
    assert argv[argv.index("--tools") + 1] == ",".join(BUILTIN_TOOLS)
    assert argv[argv.index("--allowedTools") + 1] == ",".join((*BUILTIN_TOOLS, "mcp__hands"))
    assert "--strict-mcp-config" in argv and "--system-prompt" not in argv
    assert [argv[argv.index(flag) + 1] for flag in ("--permission-mode", "--setting-sources", "--append-system-prompt", "--session-id")] == ["dontAsk", "user", "You are hands.", "b1"]
    assert json.loads(argv[argv.index("--settings") + 1]) == {"hooks": {
        event: [{"hooks": [{"type": "http", "url": f"http://127.0.0.1:7/{event}"}]}] for event in ("UserPromptSubmit", "Stop", "StopFailure")
    }}
    env = environment(tmp_path / "brain", "http://127.0.0.1:1", {
        "PATH": "/bin",
        "ANTHROPIC_API_KEY": "sk",
        "CLAUDE_CODE_OAUTH_TOKEN": "t",
        "ANTHROPIC_BASE_URL": "http://elsewhere",
        # A daemon started inside a tapped session inherits the session's tap; the brain is not that session.
        "FRITTER_TAP": "http://127.0.0.1:40000",
        "HANDS_API_URL": "",
        "_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL": "1",
    })
    assert env == {"PATH": "/bin", **SLIM, "CLAUDE_CONFIG_DIR": str(tmp_path / "brain"), "ANTHROPIC_BASE_URL": "http://127.0.0.1:1"}


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


async def test_a_side_question_is_btw_typed_answered_from_the_wire_and_dismissed(tmp_path: Path, fake_claude: Path, fritter: Path) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path, fritter), recorded.append)
    try:
        asked = asyncio.create_task(brain.fork("what did\n\tthe read say?"))
        await until(lambda: any(line[0] == "btw" for line in typed(tmp_path)))
        # Another session's side question, a fork of the brain's own that asks something else, and the brain's asking
        # that failed and is asked again, answer nothing.
        brain.hear(forked("elsewhere", "what did\n    the read say?", "x0"))
        brain.hear(answered("elsewhere", "Not this.", exchange="x0"))
        brain.hear(forked("b1", "predict the next prompt", "x2"))
        brain.hear(answered("b1", "Not this either.", exchange="x2"))
        brain.hear(forked("b1", "what did\n    the read say?", "x3"))
        brain.hear(replace(answered("b1", "x", exchange="x3"), reply=Reached(529, 0.0, 0.0, 10, Garbled("overloaded"))))
        brain.hear(forked("b1", "what did\n    the read say?", "x4"))
        brain.hear(answered("b1", "It said four.", exchange="x4"))
        assert await asked == "It said four."
        silent = asyncio.create_task(brain.fork("silent\\"))
        await until(lambda: sum(line[0] == "btw" for line in typed(tmp_path)) == 2)
        brain.hear(forked("b1", "silent\\ ", "x5"))
        brain.hear(answered("b1", "", stop="tool_use", exchange="x5"))
        with pytest.raises(ForkFailed, match="no words"):
            await silent
    finally:
        await brain.stop()
    # The command typed and its question pasted whole, newline and all, with a tab as its spaces and a closing backslash
    # kept from the Return; each answer dismissed before anything else is typed.
    assert typed(tmp_path) == [["btw", "what did\n    the read say?"], ["dismissed", ""], ["btw", "silent\\ "], ["dismissed", ""]]
    asides = [(entry.question, entry.reply, entry.failed) for entry in recorded if isinstance(entry, BrainForked)]
    assert asides == [("what did\n\tthe read say?", "It said four.", False), ("silent\\", "the brain answered the side question with no words (tool_use)", True)]


def test_text_passed_on_to_the_brain_is_typed_as_the_characters_it_shows() -> None:
    # What a terminal is told is dropped, line ends are newlines, a tab is its spaces, any other control is spelled out,
    # and a closing backslash is kept from the Return that sends it.
    assert pasted("\x1b[31mred\x1b[0m\r\nnext\rline\n\tgo\tthere\x07 C:\\") == "red\nnext\nline\n    go  there\\x07 C:\\ "
    assert pasted("plain\nwords") == "plain\nwords"


async def test_a_stop_goes_before_the_side_questions_waiting_to_type_and_returns_at_once(tmp_path: Path, fake_claude: Path, fritter: Path) -> None:
    brain = await start(launch(tmp_path, fritter), lambda _entry: None)
    try:
        waiting = asyncio.create_task(brain.ask("wait"))
        await until(lambda: any(line[0] == "prompt" for line in typed(tmp_path)))
        first = asyncio.create_task(brain.fork("first"))
        await until(lambda: any(line[0] == "btw" for line in typed(tmp_path)))
        second = asyncio.create_task(brain.fork("second"))
        await asyncio.sleep(0.05)
        # A barge-in does not wait on the input: the stage that calls this goes on with its frames.
        brain.interrupt()
        brain.hear(forked("b1", "first", "x1"))
        brain.hear(answered("b1", "One.", exchange="x1"))
        assert await first == "One."
        assert await asyncio.wait_for(waiting, 5) == BrainAnswered("p1", None)
        await until(lambda: sum(line[0] == "btw" for line in typed(tmp_path)) == 2)
        brain.hear(forked("b1", "second", "x2"))
        brain.hear(answered("b1", "Two.", exchange="x2"))
        assert await second == "Two."
    finally:
        await brain.stop()
    assert typed(tmp_path) == [
        ["prompt", " wait"],
        ["btw", "first"],
        ["dismissed", ""],
        ["escape", ""],
        ["ctrl_c", " wait"],
        ["btw", "second"],
        ["dismissed", ""],
    ]


async def test_a_turn_goes_before_the_side_questions_waiting_to_type(tmp_path: Path, fake_claude: Path, fritter: Path) -> None:
    brain = await start(launch(tmp_path, fritter), lambda _entry: None)
    try:
        first = asyncio.create_task(brain.fork("first"))
        await until(lambda: any(line[0] == "btw" for line in typed(tmp_path)))
        second = asyncio.create_task(brain.fork("second"))
        await asyncio.sleep(0.05)
        # The user's turn waits only for the side question on the screen, not for those queued behind it.
        asked = asyncio.create_task(brain.ask("are you listening?"))
        await asyncio.sleep(0.05)
        brain.hear(forked("b1", "first", "x1"))
        brain.hear(answered("b1", "One.", exchange="x1"))
        assert await first == "One."
        assert await asyncio.wait_for(asked, 5) == BrainAnswered("p1", None)
        await until(lambda: sum(line[0] == "btw" for line in typed(tmp_path)) == 2)
        brain.hear(forked("b1", "second", "x2"))
        brain.hear(answered("b1", "Two.", exchange="x2"))
        assert await second == "Two."
    finally:
        await brain.stop()
    assert [line[:2] for line in typed(tmp_path)] == [["btw", "first"], ["dismissed", ""], ["prompt", " are you listening?"], ["btw", "second"], ["dismissed", ""]]


def test_a_hook_with_a_field_that_does_not_parse_is_passed_over_and_hooks_are_still_heard() -> None:
    brain = object.__new__(Brain)
    brain.session = SessionId("b1")
    brain._turn = None  # pyright: ignore[reportPrivateUsage]
    # An error that is not text: the hook is logged and passed over, never raised out of the loop that hears hooks.
    brain._hook(Payload({"hook_event_name": "StopFailure", "session_id": "b1", "prompt_id": "p1", "error": {"kind": "odd"}}))  # pyright: ignore[reportPrivateUsage]


def test_a_turn_sent_without_hands_tools_is_an_error_and_one_with_them_is_not() -> None:
    brain = object.__new__(Brain)
    brain.session = SessionId("b1")
    brain._fork = None  # pyright: ignore[reportPrivateUsage]
    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(message.record["message"]), level="ERROR")
    try:
        body = {"messages": [{"role": "user", "content": "hi"}], "tools": [{"name": "Read"}]}
        brain.hear(Sent("x1", SessionId("b1"), MainTurn(), {**body, "tools": [{"name": "Read"}, {"name": "mcp__hands__read_session"}]}))
        brain.hear(Sent("x2", SessionId("elsewhere"), MainTurn(), body))
        assert errors == []
        brain.hear(Sent("x3", SessionId("b1"), MainTurn(), body))
    finally:
        logger.remove(sink)
    assert errors == ["the brain's turn went to the model without hands' tools: it did not connect to hands' MCP server (('Read',))"]


async def test_a_side_question_never_answered_fails_in_time_and_is_cancelled(
    tmp_path: Path, fake_claude: Path, fritter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hands.brain.process.FORK_SECONDS", 0.2)
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path, fritter), recorded.append)
    try:
        with pytest.raises(ForkFailed, match="no answer in 0s"):
            await brain.fork("hold")
        # A late answer finds no question waiting on it.
        brain.hear(answered("b1", "Too late."))
        assert await brain.ask("and now?") == BrainAnswered("p1", None)
    finally:
        await brain.stop()
    # Never answered, it is cancelled with Escape rather than dismissed with Return.
    assert typed(tmp_path) == [["btw", "hold"], ["cancelled", ""], ["prompt", " and now?"]]


async def test_a_brain_that_dies_mid_turn_fails_the_turn_and_its_side_question_and_says_once_how_it_ended(
    tmp_path: Path, fake_claude: Path, fritter: Path
) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path, fritter), recorded.append)
    with pytest.raises(BrainGone, match="exited"):
        await brain.ask("die")
    assert await brain.exited() == 3
    with pytest.raises(BrainGone, match="before it was asked"):
        await brain.ask("anyone?")
    with pytest.raises(BrainGone, match="before it was asked a side question"):
        await brain.fork("anyone?")
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


async def test_a_brain_with_no_fritter_to_run_under_is_refused_naming_the_install(tmp_path: Path, fake_claude: Path) -> None:
    with pytest.raises(Unstartable, match="hands install-fritter"):
        await start(launch(tmp_path), lambda _entry: None)


def test_a_brain_with_no_login_is_refused_naming_the_command(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    logged_in(tmp_path / "brain", "http://127.0.0.1:1")
    monkeypatch.setenv("LOGGED_IN", "0")
    with pytest.raises(NotLoggedIn, match=f"CLAUDE_CONFIG_DIR={tmp_path / 'brain'} claude"):
        logged_in(tmp_path / "brain", "http://127.0.0.1:1")


def test_the_brain_config_is_one_line_of_json_naming_only_hands() -> None:
    server = McpServer(url="http://127.0.0.1:9/mcp", token="t", runner=None)  # pyright: ignore[reportArgumentType]
    assert json.loads(server.config()) == {"mcpServers": {"hands": {"type": "http", "url": "http://127.0.0.1:9/mcp", "headers": {"Authorization": "Bearer t"}}}}


async def test_the_summariser_on_the_brain_types_the_turn_in_as_a_side_question_and_says_a_failed_one() -> None:
    asked: list[str] = []

    async def fork(question: str) -> str:
        asked.append(question)
        if "fail" in question:
            raise ForkFailed("no answer in 120s")
        if "slow" in question:
            await asyncio.sleep(5)
        return "  The tests ran. "

    summarise = aside(fork, "Sum it up.", 0.1)
    assert await summarise("the tests ran") == "The tests ran."
    assert asked == ["Sum it up.\n\nSummarize this:\n\nthe tests ran"]
    with pytest.raises(SummaryFailed, match="no answer in 120s"):
        await summarise("fail")
    # The summary's own time holds, whatever the side question waited on: the input, or the brain's answer.
    with pytest.raises(SummaryFailed, match="no answer in 0s"):
        await summarise("slow")


async def test_the_run_starts_the_brain_beside_hands_mcp_server_for_the_claude_variant_alone(tmp_path: Path, fake_claude: Path, fritter: Path) -> None:
    recorded: list[Entry] = []
    wire = Wire(lambda _observed: None)
    api = VoiceConfig(llm=AnthropicBackend(base_url="https://api.anthropic.com", api_key="k", model="m"), whisper_model="w", voice="v")
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=recorded.append)
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    async with mind(api, [], sessions, "http://127.0.0.1:1", wire, store, fritter, recorded.append) as minded:
        assert isinstance(minded.llm, AnthropicLLMService) and minded.watches == () and minded.telling == Pushed()
    claude = VoiceConfig(llm=ClaudeCodeBackend(model="claude-sonnet-5", config_dir=tmp_path / "brain"), whisper_model="w", voice="v")
    async with mind(claude, [tool(echo)], sessions, "http://127.0.0.1:1", wire, store, fritter, recorded.append) as minded:
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
