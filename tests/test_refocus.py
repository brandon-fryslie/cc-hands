"""Once hands has told the user of a session's turn or its question, the focus moves to that session."""

import asyncio
from pathlib import Path

from pipecat.frames.frames import Frame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from conftest import running
from hands.core.session import SessionId
from hands.sessions.audit import Entry, Refocused, level
from hands.sessions.focus import focused, set_focus
from hands.sessions.home import Home
from hands.voice.refocus import Refocus
from hands.voice.speech import Told

ONE, OTHER = SessionId("one"), SessionId("other")


class Passed(FrameProcessor):
    """What leaves Refocus."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        await self.push_frame(frame, direction)


async def told(home: Home, session: SessionId) -> tuple[list[Entry], list[Frame]]:
    """A Told for `session` through Refocus, then a marker it passes on: what it recorded, and what it passed."""
    recorded: list[Entry] = []
    passed = Passed()
    async with running([Refocus(home, recorded.append), passed]) as run:
        await run.worker.queue_frame(Told(session))
        await run.worker.queue_frame(TTSSpeakFrame("marker"))
        async with asyncio.timeout(5.0):
            while not any(isinstance(frame, TTSSpeakFrame) for frame in passed.frames):
                await asyncio.sleep(0.01)
    return recorded, passed.frames


async def test_a_session_told_of_becomes_the_focus_whatever_was_focused_before(tmp_path: Path) -> None:
    home = Home(tmp_path)
    set_focus(home, OTHER)
    recorded, passed = await told(home, ONE)
    assert focused(home) == ONE
    assert [entry for entry in recorded if isinstance(entry, Refocused)] == [Refocused(ONE, None)]
    # Consumed where the focus moves: nothing past the model's stage has a use for it.
    assert not any(isinstance(frame, Told) for frame in passed)


async def test_a_focus_that_cannot_be_written_is_an_error_line_and_the_frames_go_on(tmp_path: Path) -> None:
    # A home that is a file: nothing can be written under it.
    (tmp_path / "home").write_text("")
    recorded, passed = await told(Home(tmp_path / "home"), ONE)
    [refocused] = [entry for entry in recorded if isinstance(entry, Refocused)]
    assert refocused.session == ONE and refocused.failed is not None and level(refocused) == "error"
    assert any(isinstance(frame, TTSSpeakFrame) for frame in passed)
