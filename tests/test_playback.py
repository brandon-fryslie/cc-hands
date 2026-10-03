"""Playback: where the speaker is, where readings were cut off, and what going back, skipping, and repeating say."""

import asyncio
from collections.abc import Callable
from functools import reduce

import pytest
from pipecat.frames.frames import AggregatedTextFrame, Frame, InterruptionFrame, TTSSpeakFrame, TTSTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.text.base_text_aggregator import AggregationType

from conftest import running
from hands.core.playback import (
    BOOKMARKS,
    Bookmark,
    LastOne,
    NothingCut,
    NothingSaid,
    Playback,
    Played,
    Replay,
    cut,
    finished,
    handed,
    repeat,
    resume,
    skip,
)
from hands.sessions.audit import CutOff, Entry
from hands.voice.player import Player
from hands.voice.speech import Pushed

# What happens at the speaker, as the taps report it: a sentence handed to it, one played to its end, the user cutting in.
Happening = str


def played(*happenings: Happening) -> Playback:
    """Playback after the happenings in order: a sentence's text is that sentence handed, "done" the next one finished,
    and "cut" a barge-in."""

    def step(playback: Playback, happening: Happening) -> Playback:
        match happening:
            case "done":
                return finished(playback)
            case "cut":
                return cut(playback)
            case sentence:
                return handed(playback, sentence)

    return reduce(step, happenings, Playback())


@pytest.mark.parametrize(
    ("happenings", "act", "replayed", "left"),
    [
        # Cut in on the second sentence: going back says it again from its start, and what followed it.
        (("A.", "B.", "C.", "done", "cut"), resume, Replay(("B.", "C.")), ()),
        # Cut in on the first: it is said again whole.
        (("A.", "B.", "cut"), resume, Replay(("A.", "B.")), ()),
        # Skipping goes on from the sentence after the one cut off.
        (("A.", "B.", "C.", "done", "cut"), skip, Replay(("C.",)), ()),
        (("A.", "B.", "done", "cut"), skip, LastOne(), ()),
        # Saying it again says the whole reading, cut off or not, and leaves the bookmark where it was.
        (("A.", "B.", "done", "cut"), repeat, Replay(("A.", "B.")), (Bookmark(("A.", "B."), 1),)),
        (("A.", "B.", "done", "done"), repeat, Replay(("A.", "B.")), ()),
        # A reading played to its end was not cut off, whenever the user speaks next.
        (("A.", "done", "cut"), resume, NothingCut(), ()),
        ((), resume, NothingCut(), ()),
        ((), skip, NothingCut(), ()),
        ((), repeat, NothingSaid(), ()),
        # Readings cut off one after another are gone back to latest first, the earlier still waiting.
        (("A.", "B.", "cut", "C.", "D.", "done", "cut"), resume, Replay(("D.",)), (Bookmark(("A.", "B."), 0),)),
        # A report of a sentence finishing after its reading was cut changes nothing.
        (("A.", "B.", "cut", "done"), resume, Replay(("A.", "B.")), ()),
    ],
)
def test_each_act_says_what_the_speaker_was_at(
    happenings: tuple[Happening, ...], act: Callable[[Playback], tuple[Playback, Played]], replayed: Played, left: tuple[Bookmark, ...]
) -> None:
    after, said = act(played(*happenings))
    assert said == replayed
    assert after.interrupted == left


def test_a_sentence_handed_after_a_reading_ended_starts_a_new_one_and_one_handed_before_continues_it() -> None:
    assert played("A.", "done", "B.").reading == ("B.",)
    assert played("A.", "B.", "done", "C.").reading == ("A.", "B.", "C.")
    # A reading cut off is over: what is said next is another.
    assert played("A.", "cut", "B.").reading == ("B.",)


def test_what_is_said_again_is_itself_a_reading_that_can_be_cut_and_gone_back_to() -> None:
    playback, _ = resume(played("A.", "B.", "C.", "cut"))
    again = reduce(handed, ("A.", "B.", "C."), playback)
    after, said = resume(cut(finished(again)))
    assert said == Replay(("B.", "C."))
    assert after.interrupted == ()


def test_the_stack_keeps_only_the_latest_readings_cut_off() -> None:
    playback = reduce(lambda playback, n: cut(handed(playback, f"{n}.")), range(BOOKMARKS + 3), Playback())
    assert len(playback.interrupted) == BOOKMARKS
    assert playback.interrupted[0] == Bookmark((f"{3}.",), 0)


def spoken(text: str) -> AggregatedTextFrame:
    """A sentence as Pipecat's TTS hands it on ahead of its audio."""
    frame = AggregatedTextFrame(text, AggregationType.SENTENCE)
    frame.will_be_spoken = True
    return frame


def ended(text: str) -> TTSTextFrame:
    """A sentence as the output transport lets it go once its audio is written."""
    frame = TTSTextFrame(text, AggregationType.SENTENCE)
    frame.will_be_spoken = True
    return frame


class Heard(FrameProcessor):
    """What reaches the end of the pipeline."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self.frames: list[Frame] = []
        self.arrived = asyncio.Event()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        self.arrived.set()
        await self.push_frame(frame, direction)

    async def until(self, count: int, kind: type[Frame]) -> list[Frame]:
        while len(seen := [frame for frame in self.frames if isinstance(frame, kind)]) < count:
            self.arrived.clear()
            await asyncio.wait_for(self.arrived.wait(), 2.0)
        return seen


async def test_the_player_reads_the_speaker_off_the_pipeline_and_says_the_sentence_cut_off_from_its_start() -> None:
    recorded: list[Entry] = []
    player = Player(recorded.append)
    heard = Heard()
    # The taps as the daemon stands them, with the speaker between them reduced to what it lets through, in order.
    async with running([player.handing, player.playing, heard]) as run:
        playing = asyncio.create_task(player.keep_playing(Pushed(), run.worker.queue_frame))
        try:
            for frame in (spoken("The parser is fixed."), spoken("Its tests pass."), spoken("It pushed."), ended("The parser is fixed.")):
                await run.worker.queue_frame(frame)
            await heard.until(1, TTSTextFrame)
            await run.worker.queue_frame(InterruptionFrame())
            await heard.until(1, InterruptionFrame)
            assert recorded == [CutOff("Its tests pass.", 1)]

            assert player.act(resume) == ("Its tests pass.", "It pushed.")
            said = await heard.until(2, TTSSpeakFrame)
            assert [frame.text for frame in said if isinstance(frame, TTSSpeakFrame)] == ["Its tests pass.", "It pushed."]
            # Nothing is left to go back to, and saying so is said.
            assert player.act(resume) == ("Nothing was cut off to go back to.",)
        finally:
            playing.cancel()


async def test_a_barge_in_on_a_quiet_speaker_is_a_line_that_cut_nothing_off() -> None:
    recorded: list[Entry] = []
    player = Player(recorded.append)
    heard = Heard()
    async with running([player.handing, player.playing, heard]) as run:
        await run.worker.queue_frame(InterruptionFrame())
        await heard.until(1, InterruptionFrame)
    assert recorded == [CutOff(None, 0)]
