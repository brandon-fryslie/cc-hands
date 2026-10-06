"""The focus: the session the user's words go to when they name none, set in one call, held in the home, and never a lock."""

from collections.abc import Callable
import json
from datetime import UTC, datetime
from pathlib import Path

import aiohttp
import pytest

from hands.brain.mcp import CallSpans, serve_mcp
from hands.core.effects import Input, Text, Type
from hands.core.events import Joined
from hands.core.session import Membership, PromptText, SessionId
from hands.sessions.audit import AuditLog, Record, tail as audit_tail
from hands.sessions.focus import Unreadable, focused, set_focus
from hands.sessions.home import Home
from hands.sessions.registry import Sessions
from hands.sessions.sentences import Sentences
from hands.voice.briefing import as_sent
from hands.voice.narrator import Recounts
from hands.voice.player import Player
from hands.voice.refocus import Refocus
from hands.voice.sentences import SummaryStore
from hands.daemon.config import Config, OwnModel, Settings
from hands.voice.ptt import PushToTalk
from hands.spotify import Catalogue, Missing
from hands.voice.trigger import Triggers
from hands.voice.tool import Tool
from hands.voice.tools import audited, defaulting_to_focus, intermediary_tools, stay_silent_tool

HANDS = SessionId("s-hands")
LAWS = SessionId("s-laws")


async def two_sessions(tmp: Path, typed: list[Type[Input]]) -> Sessions:
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None, typist=typed.append)
    for id, project in ((HANDS, "cc-hands"), (LAWS, "laws")):
        await sessions.apply(Joined(Membership(id, 4242, Path("/code") / project, tmp / f"{id}.jsonl", tmp / f"{id}.sock"), "startup"))
    return sessions


def tools(sessions: Sessions, home: Home, record: Record = lambda _: None, acting: Callable[[], None] = lambda: None) -> dict[str, Tool]:
    """The tools as the daemon gives them, focus defaulting and each call its event in `record`."""
    given = intermediary_tools(sessions, SummaryStore(Sentences(home.root / "sentences.db")), home, Recounts(), Player(lambda _entry: None), Refocus(sessions, home, lambda _entry: None), PushToTalk(lambda _entry: None).switch, Triggers(), OwnModel(home, Settings(None, Config()), lambda _config: None), Catalogue(Missing("no credentials in tests")), acting)
    return {tool.name: audited(tool, record) for tool in given}


async def test_every_call_is_heard_acting_but_staying_silent(tmp_path: Path) -> None:
    acted: list[None] = []
    given = tools(await two_sessions(tmp_path, []), Home(tmp_path / "home"), acting=lambda: acted.append(None))
    await given["list_sessions"].body()
    await given["focus_session"].body(session=HANDS)
    await given["stay_silent"].body()
    assert len(acted) == 2


async def say(given: dict[str, Tool], text: str, **session: str) -> dict[str, object]:
    """One dictated turn, staged and sent, naming the session only where `session` does."""
    staged = await given["stage_draft"].body(text=text, resolutions=[], **session)
    assert "error" not in staged, staged
    return dict(await given["send_draft"].body(**session))


def typed_to(typed: list[Type[Input]]) -> list[tuple[str, Input]]:
    return [(effect.socket.stem, effect.input) for effect in typed]


async def test_unnamed_turns_reach_the_focus_and_one_call_moves_it_while_a_named_turn_reaches_the_one_named(tmp_path: Path) -> None:
    typed: list[Type[Input]] = []
    sessions = await two_sessions(tmp_path, typed)
    given = tools(sessions, Home(tmp_path / "home"))

    assert await given["focus_session"].body(session=HANDS) == {"readback": "Now on cc-hands."}
    for text in ("run the tests", "fix what failed", "commit it"):
        assert await say(given, text) == {"readback": "Sent the draft to cc-hands.", "focused_session": HANDS}
    assert await given["focus_session"].body(session=LAWS) == {"readback": "Now on laws."}
    await say(given, "read the new law")
    # Focused on laws, a turn that names cc-hands reaches cc-hands: the focus is a default, never a lock.
    assert await say(given, "and push it", session=HANDS) == {"readback": "Sent the draft to cc-hands."}

    assert typed_to(typed) == [
        (HANDS, Text(PromptText("run the tests"))),
        (HANDS, Text(PromptText("fix what failed"))),
        (HANDS, Text(PromptText("commit it"))),
        (LAWS, Text(PromptText("read the new law"))),
        (HANDS, Text(PromptText("and push it"))),
    ]


async def test_the_focus_outlives_a_restart_and_is_what_the_brain_is_told_with_every_request(tmp_path: Path) -> None:
    typed: list[Type[Input]] = []
    sessions = await two_sessions(tmp_path, typed)
    await tools(sessions, Home(tmp_path / "home"))["focus_session"].body(session=LAWS)

    # A daemon started again reads the home afresh.
    again = Home(tmp_path / "home")
    assert focused(again) == LAWS
    assert f'The focused session, the one the user\'s words go to when they name none, is "laws" (id {LAWS}).' in as_sent(sessions, again)
    assert (await tools(sessions, again)["list_sessions"].body())["focus"] == LAWS


async def test_with_no_focus_an_unnamed_turn_is_refused_and_nothing_is_typed(tmp_path: Path) -> None:
    typed: list[Type[Input]] = []
    sessions = await two_sessions(tmp_path, typed)
    given = tools(sessions, Home(tmp_path / "home"))

    assert "No session is focused." in as_sent(sessions, Home(tmp_path / "home"))
    assert await given["stage_draft"].body(text="run the tests", resolutions=[]) == {"error": "no session was named and none is focused: ask the user which session they mean"}
    assert typed == []


async def test_focusing_none_clears_it_and_a_session_that_is_not_running_cannot_be_focused(tmp_path: Path) -> None:
    sessions = await two_sessions(tmp_path, [])
    home = Home(tmp_path / "home")
    given = tools(sessions, home)

    await given["focus_session"].body(session=HANDS)
    assert await given["focus_session"].body(session="gone") == {"error": "no running session has the id 'gone'; take one from list_sessions"}
    assert focused(home) == HANDS
    assert await given["focus_session"].body(session="") == {"readback": "No session is focused now."}
    assert focused(home) is None
    await given["focus_session"].body(session=HANDS)
    # A null is the session left out, and focusing takes one: the call is refused, and the focus stays where it was.
    assert await given["focus_session"].body(session=None) == {"error": "focus_session was called with the wrong arguments: missing a required argument: 'session'"}
    assert focused(home) == HANDS


async def test_a_focused_session_that_has_stopped_running_is_said_so(tmp_path: Path) -> None:
    sessions = await two_sessions(tmp_path, [])
    home = Home(tmp_path / "home")
    set_focus(home, SessionId("ended"))
    assert "The focused session (id ended) is not running now." in as_sent(sessions, home)


async def test_a_focus_file_holding_no_session_id_is_refused_and_the_brain_is_told_it_cannot_be_read(tmp_path: Path) -> None:
    sessions = await two_sessions(tmp_path, [])
    home = Home(tmp_path / "home")
    home.root.mkdir()
    home.focus.write_text("../escape\n")
    assert isinstance(unreadable := focused(home), Unreadable) and "which is no session id" in unreadable.reason
    assert "Which session is focused cannot be read:" in as_sent(sessions, home)
    assert (await tools(sessions, home)["list_sessions"].body())["focus"] == {"cannot_read": unreadable.reason}
    result = await tools(sessions, home)["read_backlog"].body()
    assert "which one is focused cannot be read" in str(result["error"])


async def test_every_tool_that_acts_on_a_session_takes_the_focus_for_one_left_unnamed(tmp_path: Path) -> None:
    given = tools(await two_sessions(tmp_path, []), Home(tmp_path / "home"))
    defaulted = {name for name, tool in given.items() if "session" in tool.properties and "session" not in tool.required}
    assert defaulted == {
        "read_session", "read_turn", "tell_turn", "expand", "read_backlog", "read_ticket",
        "stage_draft", "amend_draft", "discard_draft", "send_draft", "send_command", "interrupt_session", "set_overlay",
    }  # fmt: skip
    assert all(str(given[name].properties["session"]["description"]).endswith("Empty for the focused session.") for name in defaulted)
    # focus_session's own empty means "focus none", never "the focus".
    assert given["focus_session"].required == ("session",)
    with pytest.raises(TypeError, match="stay_silent takes no session"):
        defaulting_to_focus(stay_silent_tool(), Home(tmp_path / "home"))


async def test_the_session_the_focus_stood_in_for_is_on_the_calls_event(tmp_path: Path) -> None:
    sessions = await two_sessions(tmp_path, [])
    home = Home(tmp_path / "home")
    path = tmp_path / "audit"
    record = AuditLog(path, clock=lambda: datetime.now(UTC)).record
    given = tools(sessions, home, record)

    await given["focus_session"].body(session=LAWS)
    await given["stage_draft"].body(text="run the tests", resolutions=[])

    called = [line["facts"] for line in map(json.loads, audit_tail(path, 1000)[0]) if line.get("event") == "tool.run"]
    assert [(facts["tool"], facts["called"]["arguments"], facts["called"]["result"]) for facts in called] == [
        ("focus_session", {"session": LAWS}, {"readback": "Now on laws."}),
        ("stage_draft", {"text": "run the tests", "resolutions": []}, {"says": "Draft for laws: run the tests", "focused_session": LAWS}),
    ]


async def test_a_call_with_arguments_the_tool_does_not_take_is_refused_as_the_tool_would_refuse_it(tmp_path: Path) -> None:
    home = Home(tmp_path / "home")
    sessions = await two_sessions(tmp_path, [])
    await tools(sessions, home)["focus_session"].body(session=LAWS)
    server = await serve_mcp(list(tools(sessions, home).values()), lambda _: None, CallSpans())
    try:
        call = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "stage_draft", "arguments": {"prompt": "run the tests"}}}
        async with aiohttp.ClientSession() as client, client.post(server.url, json=call, headers={"Authorization": f"Bearer {server.token}"}) as reply:
            refused = await reply.json()
        assert json.loads(refused["result"]["content"][0]["text"])["error"].startswith("stage_draft was called with the wrong arguments: ")
    finally:
        await server.close()

