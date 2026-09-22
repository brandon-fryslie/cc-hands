"""What the latency observer logs, and how often: one line per utterance, not one per processor it crosses."""

import pytest
from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed

from hands.voice.latency import LatencyObserver


async def logged(*frames: Frame) -> list[str]:
    """The lines the observer writes as those frames cross it, in order."""
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(message.record["message"]), level="INFO", filter="hands")
    observer = LatencyObserver()
    try:
        for frame in frames:
            await observer.on_push_frame(FramePushed(source=None, destination=None, frame=frame, direction=None, timestamp=0))  # pyright: ignore[reportArgumentType]
    finally:
        logger.remove(sink)
    return lines


async def test_one_utterance_nobody_asked_for_is_one_line_however_often_it_is_announced() -> None:
    """A turn's summary was measured raising four started-speaking frames between one start and one stop, and
    four identical lines read as four utterances."""
    speaking = BotStartedSpeakingFrame()
    assert await logged(speaking, speaking, speaking, speaking) == ["latency: first audio, answering no user turn"]


async def test_the_next_utterance_is_its_own_line_once_the_speaker_has_stopped() -> None:
    lines = await logged(BotStartedSpeakingFrame(), BotStartedSpeakingFrame(), BotStoppedSpeakingFrame(), BotStartedSpeakingFrame())
    assert lines == ["latency: first audio, answering no user turn"] * 2


async def test_speech_that_answers_a_user_turn_is_measured_from_the_release_instead() -> None:
    lines = await logged(
        VADUserStartedSpeakingFrame(),
        VADUserStoppedSpeakingFrame(),
        TranscriptionFrame(text="hello", user_id="u", timestamp="t"),
        BotStartedSpeakingFrame(),
    )
    assert [line.split(" ")[1] for line in lines] == ["transcript", "first"]
    assert all("after key release" in line for line in lines)


@pytest.mark.parametrize("frame", [BotStartedSpeakingFrame(), TranscriptionFrame(text="x", user_id="u", timestamp="t")])
async def test_nothing_is_measured_from_a_release_that_never_happened(frame: Frame) -> None:
    """A transcript with no user turn open has no release to be late from, so it is not reported as instant."""
    assert all("after key release" not in line for line in await logged(frame))


async def test_a_narration_after_a_user_turn_is_still_measured_as_answering_nobody() -> None:
    """The turn a user opened has to close when the audio it was waiting for finishes. Left open, the first user
    turn of a session stays open for the rest of it, every later narration finds `first audio` already marked,
    and the line this observer exists for is written nowhere — while the numbers in docs/architecture.md come
    from exactly that line."""
    lines = await logged(
        VADUserStartedSpeakingFrame(),
        VADUserStoppedSpeakingFrame(),
        TranscriptionFrame(text="hello", user_id="u", timestamp="t"),
        BotStartedSpeakingFrame(),
        BotStoppedSpeakingFrame(),
        BotStartedSpeakingFrame(),
        BotStoppedSpeakingFrame(),
        BotStartedSpeakingFrame(),
    )
    assert lines[-2:] == ["latency: first audio, answering no user turn"] * 2


async def test_a_barge_in_keeps_the_turn_it_opened_while_the_speaker_is_stopping() -> None:
    """The user interrupting opens the next turn before the speaker's stop arrives, and that turn is waiting for
    its own first audio — so the stop that belongs to the utterance being cut off must not close it."""
    lines = await logged(
        VADUserStartedSpeakingFrame(),
        VADUserStoppedSpeakingFrame(),
        TranscriptionFrame(text="first", user_id="u", timestamp="t"),
        BotStartedSpeakingFrame(),
        VADUserStartedSpeakingFrame(),
        BotStoppedSpeakingFrame(),
        VADUserStoppedSpeakingFrame(),
        TranscriptionFrame(text="second", user_id="u", timestamp="t"),
    )
    assert lines[-1].startswith("latency: transcript") and "after key release" in lines[-1]
