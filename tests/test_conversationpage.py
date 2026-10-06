"""The conversation page reads the conversation off the audit log, and hands what is typed into it to the voice."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import count
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from hands.sessions.audit import AuditLog, Entry, Replied, Transcribed, segments
from hands.sessions.wide import WideEvent
from hands.voice import conversationpage
from hands.voice.conversationpage import Conversation, conversation_routes
from hands.voice.tool import tool
from hands.voice.tools import audited

KEY = "the-phones-key"
KEYED = {"Authorization": f"Bearer {KEY}"}
MORNING = datetime(2026, 10, 6, 9, 0, tzinfo=UTC)


@dataclass
class Page:
    client: TestClient[web.Request, web.Application]
    log: AuditLog
    # What the routes recorded, apart from the log the page reads.
    recorded: list[Entry]
    typed: list[str] = field(default_factory=list[str])

    def events(self, name: str) -> list[WideEvent]:
        return [entry for entry in self.recorded if isinstance(entry, WideEvent) and entry.event == name]


def a_log(directory: Path, segment_bytes: int = 32 * 1024 * 1024) -> AuditLog:
    minutes = count()
    return AuditLog(directory, clock=lambda: MORNING + timedelta(minutes=next(minutes)), segment_bytes=segment_bytes)


@pytest.fixture
async def page(tmp_path: Path) -> AsyncGenerator[Page]:
    log, recorded = a_log(tmp_path / "audit"), list[Entry]()
    typed: list[str] = []

    async def typing(text: str) -> None:
        typed.append(text)

    app = web.Application()
    app.add_routes(conversation_routes(Conversation(tmp_path / "audit"), typing, KEY, recorded.append))
    client = TestClient(TestServer(app))
    await client.start_server()
    yield Page(client, log, recorded, typed)
    await client.close()


async def set_trigger(trigger: str) -> dict[str, object]:
    return {"trigger": trigger}


async def test_the_page_is_served_to_anyone(page: Page) -> None:
    shown = await page.client.get("/conversation")
    assert shown.status == 200 and "Type to hands" in await shown.text()


async def test_the_conversation_is_what_the_user_and_hands_said_and_each_tool_hands_called(page: Page) -> None:
    page.log.record(Transcribed("switch to the wake word"))
    assert "trigger" in await audited(tool(set_trigger), page.log.record).body(trigger="wake word")
    page.log.record(Replied("Okay. Say Hey Jarvis, then what you want.", False))
    read = await page.client.get("/conversation/moments", headers=KEYED)
    assert read.status == 200
    body = await read.json()
    assert [(moment["kind"], moment["heading"], moment["text"]) for moment in body["moments"]] == [
        ("heard", "user", "switch to the wake word"),
        ("called", "called set_trigger", '{"trigger": "wake word"}\n{"trigger": "wake word"}'),
        ("said", "you", "Okay. Say Hey Jarvis, then what you want."),
    ]
    assert body["moments"][0]["at"] == MORNING.isoformat()
    assert body["changes"] == 3
    [event] = page.events("conversation.read")
    assert (event.outcome, event.counts["moments"], event.facts["seen"], event.facts["changes"]) == ("ok", 3, None, 3)


async def test_a_tool_call_that_failed_says_why(page: Page) -> None:
    async def refused() -> dict[str, object]:
        return {"error": "no such session"}

    await audited(tool(refused), page.log.record).body()
    body = await (await page.client.get("/conversation/moments", headers=KEYED)).json()
    assert [moment["text"] for moment in body["moments"]] == ["{}\nfailed: no such session"]


async def test_a_read_that_has_seen_the_conversation_waits_for_it_to_change(page: Page) -> None:
    page.log.record(Transcribed("what time is it"))
    first = await (await page.client.get("/conversation/moments", headers=KEYED)).json()
    waiting = asyncio.create_task(page.client.get("/conversation/moments", params={"seen": str(first["changes"])}, headers=KEYED))
    await asyncio.sleep(conversationpage.POLL_SECONDS * 3)
    assert not waiting.done()
    page.log.record(Replied("Nine o'clock.", False))
    body = await (await waiting).json()
    assert [moment["text"] for moment in body["moments"]] == ["what time is it", "Nine o'clock."]
    assert body["changes"] == first["changes"] + 1


async def test_a_read_whose_conversation_never_changes_is_answered_with_it_as_it_is(page: Page, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conversationpage, "WAIT_SECONDS", 0.3)
    read = await page.client.get("/conversation/moments", params={"seen": "0"}, headers=KEYED)
    assert (await read.json()) == {"changes": 0, "moments": []}
    [event] = page.events("conversation.read")
    assert isinstance(waited := event.facts["waited_ms"], float) and waited >= 300


async def test_the_conversation_is_read_on_across_the_segments_the_log_rolls_to(tmp_path: Path) -> None:
    log = a_log(tmp_path, segment_bytes=200)
    conversation = Conversation(tmp_path)
    rolled: set[int] = set()
    for said in ("first", "second", "third", "fourth"):
        log.record(Transcribed(said))
        rolled |= set(segments(tmp_path))
        _, moments = await conversation.caught_up()
    assert len(rolled) > 2
    assert [moment.text for moment in moments] == ["first", "second", "third", "fourth"]


@pytest.mark.parametrize("path", ["/conversation/moments", "/conversation/typed"])
async def test_a_request_without_the_phones_key_is_refused_and_says_why(page: Page, path: str) -> None:
    page.log.record(Transcribed("the secret plan"))
    refused = await page.client.request("GET" if path.endswith("moments") else "POST", path, json={"text": "hi"}, headers={"Authorization": "Bearer guessed"})
    assert refused.status == 401 and "the secret plan" not in await refused.text()
    [event] = page.events("conversation.read" if path.endswith("moments") else "conversation.typed")
    assert event.outcome == "failed" and "without the phone's key" in str(event.error)
    assert page.typed == []


async def test_a_read_that_says_what_it_has_seen_in_no_number_is_refused(page: Page) -> None:
    refused = await page.client.get("/conversation/moments", params={"seen": "lots"}, headers=KEYED)
    assert refused.status == 400
    [event] = page.events("conversation.read")
    assert event.outcome == "failed"


async def test_words_typed_are_handed_to_the_voice_and_counted(page: Page) -> None:
    sent = await page.client.post("/conversation/typed", json={"text": "  run the tests \n"}, headers=KEYED)
    assert sent.status == 202
    assert page.typed == ["run the tests"]
    [event] = page.events("conversation.typed")
    assert (event.outcome, event.facts["chars"]) == ("ok", 13)


@pytest.mark.parametrize("body", [b'{"text": "   "}', b'{"words": "hi"}', b'"hi"', b"hi", b'{"text": "\xff"}'])
async def test_words_typed_that_are_none_are_refused_and_handed_to_nobody(page: Page, body: bytes) -> None:
    refused = await page.client.post("/conversation/typed", data=body, headers={**KEYED, "Content-Type": "application/json"})
    assert refused.status == 400
    assert page.typed == []
    [event] = page.events("conversation.typed")
    assert event.outcome == "failed"
