"""The text Claude Code displays, posted by its MessageDisplay hook to a loopback route that takes nothing else."""

import json
import socket

import aiohttp
import pytest

from hands.core.events import Displayed, Event
from hands.core.session import PromptId, SessionId
from hands.sessions.hookconfig import DISPLAY_URL, plugin_hooks
from hands.sessions.hooks import parse_display
from hands.sessions.payload import Rejected
from hands.sessions.server import serve_display

SID = SessionId("9a07bdf1-5512-402c-a1ea-4b3d10f6a73d")

LINE = "3. The OS first checks local sources such as the hosts file and its own DNS cache.\n"
# As Claude Code 2.1.288 posted it, with the paths shortened.
DISPLAYED: dict[str, object] = {
    "session_id": SID,
    "transcript_path": "/x/9a07bdf1.jsonl",
    "cwd": "/x",
    "hook_event_name": "MessageDisplay",
    "prompt_id": "ec58f7d1-9dad-4171-877d-8026f2961d81",
    "turn_id": "cac59a10-6c1f-46c1-8047-05358855e6d4",
    "message_id": "a4e632ff-0000-0000-0000-000000000000",
    "index": 3,
    "final": False,
    "delta": LINE,
}


def test_a_displayed_batch_is_its_lines_in_the_turn_its_prompt_names() -> None:
    assert parse_display(json.dumps(DISPLAYED).encode(), at=7.0) == Displayed(SID, (PromptId("ec58f7d1-9dad-4171-877d-8026f2961d81"),), LINE, 7.0)


def test_the_display_route_takes_no_other_hook() -> None:
    with pytest.raises(Rejected, match="only MessageDisplay"):
        parse_display(json.dumps({**DISPLAYED, "hook_event_name": "PermissionRequest"}).encode(), at=7.0)


def test_the_plugin_posts_message_display_to_the_route_and_runs_no_process_for_it() -> None:
    hooks = plugin_hooks()["hooks"]
    assert isinstance(hooks, dict)
    assert hooks["MessageDisplay"] == [{"hooks": [{"type": "http", "url": DISPLAY_URL, "timeout": 2}]}]


class Applying:
    def __init__(self) -> None:
        self.applied: list[Event] = []

    def now(self) -> float:
        return 7.0

    async def apply(self, event: Event) -> None:
        self.applied.append(event)


async def test_the_route_applies_what_was_displayed_and_refuses_anything_else() -> None:
    sessions = Applying()
    port = _free_port()
    runner = await serve_display(sessions, "127.0.0.1", port, "/hands/display")  # pyright: ignore[reportArgumentType]  (only now() and apply() are asked)
    try:
        async with aiohttp.ClientSession() as client:
            async with client.post(f"http://127.0.0.1:{port}/hands/display", json=DISPLAYED) as taken:
                assert taken.status == 204
            async with client.post(f"http://127.0.0.1:{port}/hands/display", json={**DISPLAYED, "hook_event_name": "Stop"}) as refused:
                assert refused.status == 400
            async with client.post(f"http://127.0.0.1:{port}/hook", json=DISPLAYED) as elsewhere:
                assert elsewhere.status == 404
    finally:
        await runner.cleanup()
    assert sessions.applied == [parse_display(json.dumps(DISPLAYED).encode(), at=7.0)]


async def test_a_daemon_that_cannot_take_the_port_does_not_start() -> None:
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        port = held.getsockname()[1]
        with pytest.raises(RuntimeError, match=f"127.0.0.1:{port}"):
            await serve_display(Applying(), "127.0.0.1", port, "/hands/display")  # pyright: ignore[reportArgumentType]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
