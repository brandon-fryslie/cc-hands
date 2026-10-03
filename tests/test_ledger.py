"""Every line hands says as written is written down at the speaker's door, and the brain is reminded of the last of them."""

import asyncio
from collections.abc import Callable

from pipecat.frames.frames import Frame, LLMTextFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from conftest import running
from hands.sessions.audit import Entry, HandsSpoke
from hands.voice.ledger import Ledger
from hands.voice.system import ModelUnreachable, SystemChannel


class Sink(FrameProcessor):
    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self.frames: list[TTSSpeakFrame | LLMTextFrame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSSpeakFrame | LLMTextFrame):
            self.frames.append(frame)
        await self.push_frame(frame, direction)


async def until(what: Callable[[], bool]) -> None:
    async with asyncio.timeout(5.0):
        while not what():
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


async def test_a_line_said_long_enough_ago_is_no_longer_one_the_user_may_be_answering() -> None:
    now = [0.0]
    ledger, sink = Ledger(lambda _: None, clock=lambda: now[0], within=600.0), Sink()
    async with running([ledger, sink]) as run:
        await run.worker.queue_frame(TTSSpeakFrame("cc-hands has a question for you."))
        await until(lambda: len(sink.frames) == 1)
        now[0] = 300.0
        await run.worker.queue_frame(TTSSpeakFrame("20 seconds left to answer cc-hands."))
        await until(lambda: len(sink.frames) == 2)
    assert ledger.lately() == ("cc-hands has a question for you.", "20 seconds left to answer cc-hands.")
    now[0] = 700.0
    assert ledger.lately() == ("20 seconds left to answer cc-hands.",)


async def test_a_fault_the_system_channel_says_is_written_down_at_the_ledger_and_reaches_the_speaker() -> None:
    """In the daemon the channel queues at the ledger and reads usability from the TTS: two processors, not one."""
    recorded: list[Entry] = []
    ledger, tts = Ledger(recorded.append), Sink()

    async def notify(_text: str) -> bool:
        raise AssertionError("speech works, so nothing goes to the screen")

    async with running([ledger, tts]):
        await SystemChannel(ledger, tts, notify, recorded.append).say(ModelUnreachable())
        await until(lambda: len(tts.frames) == 1)
    assert ledger.lately() == ("The language model is unreachable.",)
    assert HandsSpoke("The language model is unreachable.") in recorded
