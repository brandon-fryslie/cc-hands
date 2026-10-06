"""Playback: where the speaker is, where readings were cut off, and what going back, skipping, and repeating say."""

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from functools import reduce

import pytest
from pipecat.frames.frames import (
    AggregatedTextFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.text.base_text_aggregator import AggregationType

from conftest import Running, running
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
    queued,
    repeat,
    resume,
    skip,
    went_on,
)
from hands.sessions.audit import CutOff, Entry, WentOn
from hands.sessions.wide import WideEvent
from hands.voice.player import Player, said
from hands.voice.tools import audited, playback_tools
from hands.voice.trigger import Edge
from hands.voice.turnstop import HoldDiscarded

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
        # Saying it again says the whole reading, cut off or not, and one said whole no longer waits to be gone back to.
        (("A.", "B.", "done", "cut"), repeat, Replay(("A.", "B.")), ()),
        (("A.", "B.", "done", "done", "cut"), repeat, Replay(("A.", "B.")), ()),
        # The model said a word of its own before asking: what is said again is what the user cut in on, not that word.
        (("A.", "done", "cut", "Sure."), repeat, Replay(("A.",)), ()),
        # A reading played to its end was not cut off, whenever the user speaks next.
        (("A.", "done", "cut"), resume, NothingCut(), ()),
        ((), resume, NothingCut(), ()),
        ((), skip, NothingCut(), ()),
        ((), repeat, NothingSaid(), ()),
        # Cut in on the answer to a side question to say "go back": it goes back past that answer, which is left, to
        # the reading cut off before it.
        (("A.", "B.", "cut", "C.", "D.", "done", "cut"), resume, Replay(("A.", "B.")), ()),
        # The answer had played to its end: going back is to the reading cut off before it all the same.
        (("A.", "B.", "cut", "C.", "done", "cut"), resume, Replay(("A.", "B.")), ()),
        # Skipping skips what was cut in on.
        (("A.", "B.", "cut", "C.", "D.", "E.", "done", "cut"), skip, Replay(("E.",)), (Bookmark(("A.", "B."), 0),)),
        # Readings cut off one after another, each played to its end since, are gone back to latest first.
        (("A.", "cut", "B.", "cut", "C.", "done", "cut"), resume, Replay(("B.",)), (Bookmark(("A.",), 0),)),
        # A report of a sentence finishing after its reading was cut changes nothing.
        (("A.", "B.", "cut", "done"), resume, Replay(("A.", "B.")), ()),
        # Nobody cut in after all: the reading cut off goes on from the sentence it stopped on, and waits no more...
        (("A.", "B.", "C.", "done", "cut"), went_on, Replay(("B.", "C.")), ()),
        # ...and it is the reading just cut off that goes on, not the one cut off before it, which still waits.
        (("A.", "cut", "B.", "C.", "cut"), went_on, Replay(("B.", "C.")), (Bookmark(("A.",), 0),)),
    ],
)
def test_each_act_says_what_the_speaker_was_at(
    happenings: tuple[Happening, ...], act: Callable[[Playback], tuple[Playback, Played]], replayed: Played, left: tuple[Bookmark, ...]
) -> None:
    after, said = act(played(*happenings))
    assert said == replayed
    assert after.bookmarks == left


def test_a_sentence_handed_after_a_reading_ended_starts_a_new_one_and_one_handed_before_continues_it() -> None:
    assert played("A.", "done", "B.").reading == ("B.",)
    assert played("A.", "B.", "done", "C.").reading == ("A.", "B.", "C.")
    # A reading cut off is over: what is said next is another.
    assert played("A.", "cut", "B.").reading == ("B.",)


def test_what_is_said_again_is_itself_a_reading_that_can_be_cut_and_gone_back_to() -> None:
    playback, _ = resume(played("A.", "B.", "C.", "cut"))
    again = reduce(handed, ("A.", "B.", "C."), queued(playback, ("A.", "B.", "C.")))
    after, said = resume(cut(finished(again)))
    assert said == Replay(("B.", "C."))
    assert after.bookmarks == ()


def test_lines_said_again_while_a_reading_still_plays_go_on_from_it() -> None:
    # The model said a few words of its own before asking to go back: those still playing, the lines follow them, and
    # the words finishing count as the words, not as the first line.
    playback = finished(handed(queued(handed(Playback(), "Sure."), ("A.", "B.")), "A."))
    assert playback.reading == ("Sure.", "A.") and playback.played == 1
    assert resume(cut(playback))[1] == Replay(("A.", "B."))


def test_a_word_handed_before_the_lines_queued_takes_its_own_place_and_loses_none_of_them() -> None:
    # The lines were queued while the model's word was still on its way to the speaker.
    playback = cut(handed(handed(queued(Playback(), ("A.", "B.", "C.")), "Sure."), "A."))
    assert playback.stopped == Bookmark(("Sure.", "A.", "B.", "C."), 0)


def test_a_reading_said_again_and_cut_before_all_of_it_reached_the_speaker_is_gone_back_to_whole() -> None:
    # The barge-in drops what had not yet reached the speaker; the reading knew it from the start, and keeps it.
    said_again = handed(queued(Playback(), ("A.", "B.", "C.")), "A.")
    assert said_again.reading == ("A.",)
    assert resume(cut(said_again))[1] == Replay(("A.", "B.", "C."))
    assert repeat(cut(said_again))[1] == Replay(("A.", "B.", "C."))


def test_the_stack_keeps_only_the_latest_readings_cut_off() -> None:
    playback = reduce(lambda playback, n: cut(handed(playback, f"{n}.")), range(BOOKMARKS + 3), Playback())
    assert len(playback.interrupted) == BOOKMARKS
    assert playback.interrupted[0] == Bookmark(("2.",), 0)


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


@asynccontextmanager
async def stood(player: Player, opened: Edge = "held key") -> AsyncGenerator[tuple[Running, Heard]]:
    """The player as the daemon stands it: Whisper ahead of its lines, its lines ahead of the speaker, and its observer on
    Whisper, the speaker, and the output transport, each reduced to a processor that lets frames through in order; every
    turn opened by `opened`."""
    hearer, speaker, output, heard = IdentityFilter(), IdentityFilter(), IdentityFilter(), Heard()
    async with running([hearer, player.lines, speaker, output, heard], [player.watching(hearer, speaker, output, lambda: opened)]) as run:
        yield run, heard


async def test_the_player_reads_the_speaker_off_the_pipeline_and_says_the_sentence_cut_off_from_its_start() -> None:
    recorded: list[Entry] = []
    player = Player(recorded.append)
    async with stood(player) as (run, heard):
        for frame in (spoken("The parser is fixed."), spoken("Its tests pass."), spoken("It pushed."), ended("The parser is fixed.")):
            await run.worker.queue_frame(frame)
        await heard.until(1, TTSTextFrame)
        await run.worker.queue_frame(InterruptionFrame())
        await heard.until(1, InterruptionFrame)
        assert recorded == [CutOff("Its tests pass.", 1)]

        assert await player.act(resume) == ("Its tests pass.", "It pushed.")
        said = await heard.until(2, TTSSpeakFrame)
        assert [frame.text for frame in said if isinstance(frame, TTSSpeakFrame)] == ["Its tests pass.", "It pushed."]


@pytest.mark.parametrize(
    ("act", "line"),
    [
        (resume, "Nothing was cut off to go back to."),
        (skip, "Nothing was cut off to go back to."),
        (repeat, "I haven't said anything yet."),
    ],
)
async def test_an_act_with_nothing_to_say_again_says_why(act: Callable[[Playback], tuple[Playback, Played]], line: str) -> None:
    player = Player(lambda _: None)
    async with stood(player) as (_, heard):
        assert await player.act(act) == (line,)
        said = await heard.until(1, TTSSpeakFrame)
        assert [frame.text for frame in said if isinstance(frame, TTSSpeakFrame)] == [line]


async def test_each_playback_tool_answers_with_what_was_said_and_how_many_readings_still_wait() -> None:
    recorded: list[Entry] = []
    player = Player(recorded.append)
    tools = {tool.name: audited(tool, recorded.append) for tool in playback_tools(player)}
    async with stood(player) as (run, heard):
        # A barge-in overtakes frames still queued, so each waits for what is ahead of it to be heard.
        for frame, seen in ((spoken("A."), 1), (spoken("B."), 2), (InterruptionFrame(), 1), (spoken("C."), 3), (InterruptionFrame(), 2)):
            await run.worker.queue_frame(frame)
            await heard.until(seen, type(frame))
        recorded.clear()
        assert await tools["resume"].body() == {"said": ("A.", "B."), "waiting": 0}
        assert await tools["skip"].body() == {"said": ("Nothing was cut off to go back to.",), "waiting": 0}
        assert [(type(entry), entry.event) for entry in recorded if isinstance(entry, WideEvent)] == [(WideEvent, "tool.run")] * 2 and len(recorded) == 2


def test_skipping_the_last_sentence_says_so() -> None:
    assert said(skip(played("A.", "B.", "done", "cut"))[1]) == ("That was the last of it.",)


async def test_a_barge_in_on_a_quiet_speaker_is_a_line_that_cut_nothing_off() -> None:
    recorded: list[Entry] = []
    async with stood(Player(recorded.append)) as (run, heard):
        await run.worker.queue_frame(InterruptionFrame())
        await heard.until(1, InterruptionFrame)
    assert recorded == [CutOff(None, 0)]


def words(text: str) -> TranscriptionFrame:
    """What Whisper pushes for a hold it heard words in."""
    return TranscriptionFrame(text, "user", "2026-10-05T01:00:09Z")


async def barged_in(run: Running, heard: Heard, *during: Frame) -> None:
    """Hands reads three sentences and has played the first when a turn of the user's opens, cutting it off; `during` is
    what Whisper pushes before the turn ends. A system frame overtakes those still queued, so each waits for what is
    ahead of it to be heard."""
    for frame, seen in (
        (spoken("The parser is fixed."), 1), (spoken("Its tests pass."), 2), (spoken("It pushed."), 3), (ended("The parser is fixed."), 1),
        (UserStartedSpeakingFrame(), 1), (InterruptionFrame(), 1), *((each, n + 1) for n, each in enumerate(during)), (UserStoppedSpeakingFrame(), 1),
    ):
        await run.worker.queue_frame(frame)
        await heard.until(seen, type(frame))


@pytest.mark.parametrize("opened", ["engaged conversation", "wake word"])
async def test_a_barge_in_the_voice_opened_that_heard_nothing_goes_on_from_the_sentence_it_cut_off(opened: Edge) -> None:
    # Hands' own reply came back through the microphone, the desk's detector took it for the user, and Whisper found no
    # words in the turn it opened: nobody cut in, so the reading goes on where it stopped.
    recorded: list[Entry] = []
    player = Player(recorded.append)
    async with stood(player, opened) as (run, heard):
        await barged_in(run, heard)
        said = await heard.until(2, TTSSpeakFrame)
    assert [frame.text for frame in said if isinstance(frame, TTSSpeakFrame)] == ["Its tests pass.", "It pushed."]
    assert recorded == [CutOff("Its tests pass.", 1), WentOn("Its tests pass.", 0)]
    assert player.waiting == 0


@pytest.mark.parametrize(
    ("opened", "during"),
    [
        # A key or button held and let go with nothing said is the user stopping hands.
        ("held key", ()),
        ("phone button", ()),
        # The user said something: what they said is the turn, and the reading waits to be gone back to.
        ("engaged conversation", (words("Wait, which file?"),)),
        # Whisper failed on the hold, or it was thrown away: it may have held the user's words.
        ("engaged conversation", (ErrorFrame(error="Whisper could not transcribe hold 3"),)),
        ("engaged conversation", (HoldDiscarded(),)),
    ],
)
async def test_a_barge_in_that_heard_the_user_or_was_theirs_by_hand_leaves_the_reading_cut_off(opened: Edge, during: tuple[Frame, ...]) -> None:
    recorded: list[Entry] = []
    player = Player(recorded.append)
    async with stood(player, opened) as (run, heard):
        await barged_in(run, heard, *during)
        # Nothing is said after the turn: a frame queued behind it arrives with nothing said ahead of it.
        await run.worker.queue_frame(spoken("Next."))
        await heard.until(4, AggregatedTextFrame)
    assert not [frame for frame in heard.frames if isinstance(frame, TTSSpeakFrame)]
    assert recorded == [CutOff("Its tests pass.", 1)]
    assert player.waiting == 1


async def test_a_turn_the_voice_opened_on_a_quiet_speaker_says_nothing_again_whatever_waits() -> None:
    # A reading the user cut off by hand still waits to be gone back to; the voice then opens a turn with nothing playing,
    # and nothing said in it. It cut nothing off, so nothing goes on: the reading the user stopped stays stopped.
    recorded: list[Entry] = []
    player = Player(recorded.append)
    async with stood(player, "engaged conversation") as (run, heard):
        for frame, seen in ((spoken("A."), 1), (InterruptionFrame(), 1), (UserStartedSpeakingFrame(), 1), (InterruptionFrame(), 2), (UserStoppedSpeakingFrame(), 1)):
            await run.worker.queue_frame(frame)
            await heard.until(seen, type(frame))
        await run.worker.queue_frame(spoken("Next."))
        await heard.until(2, AggregatedTextFrame)
    assert not [frame for frame in heard.frames if isinstance(frame, TTSSpeakFrame)]
    assert recorded == [CutOff("A.", 1), CutOff(None, 1)]
