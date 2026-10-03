"""Every line hands says as written is written down at the speaker's door, and the brain is reminded of the last of them."""

import asyncio

from pipecat.frames.frames import Frame, LLMTextFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from conftest import running
from hands.sessions.audit import Entry, HandsSpoke
from hands.voice.ledger import Ledger


class Sink(FrameProcessor):
    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self.frames: list[TTSSpeakFrame | LLMTextFrame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSSpeakFrame | LLMTextFrame):
            self.frames.append(frame)
        await self.push_frame(frame, direction)


async def until(what: object) -> None:
    async with asyncio.timeout(5.0):
        while not what():  # pyright: ignore[reportCallIssue]
            await asyncio.sleep(0.01)


async def test_a_line_hands_says_as_written_is_kept_and_audited_and_still_said() -> None:
    recorded: list[Entry] = []
    ledger, sink = Ledger(recorded.append), Sink()
    async with running([ledger, sink]) as run:
        await run.worker.queue_frame(TTSSpeakFrame("cc-hands has a question for you."))
        await run.worker.queue_frame(LLMTextFrame("The brain's own words."))
        await until(lambda: len(sink.frames) == 2)
    # The brain's words pass through unrecorded: they are its history's already.
    assert ledger.lately() == ("cc-hands has a question for you.",)
    assert recorded == [HandsSpoke("cc-hands has a question for you.")]
    assert [frame.text for frame in sink.frames] == ["cc-hands has a question for you.", "The brain's own words."]


async def test_the_ledger_keeps_only_the_last_lines() -> None:
    ledger, sink = Ledger(lambda _: None, kept=2), Sink()
    async with running([ledger, sink]) as run:
        for line in ("one", "two", "three"):
            await run.worker.queue_frame(TTSSpeakFrame(line))
        await until(lambda: len(sink.frames) == 3)
    assert ledger.lately() == ("two", "three")
