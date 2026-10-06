"""What the latency observer logs, and how often: one line per utterance, not one per processor it crosses."""

import pytest
from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    LLMTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed

from hands.voice.latency import LatencyObserver
from hands.voice.mark import Mark
from hands.voice.turnstop import Hold, HoldDiscarded, TurnOpened, TurnResolved, Words


async def observed(*frames: Frame) -> tuple[list[str], list[Mark]]:
    """The lines the observer writes as those frames cross it, and the marks it tells, each in order."""
    lines: list[str] = []
    marks: list[Mark] = []
    sink = logger.add(lambda message: lines.append(message.record["message"]), level="INFO", filter="hands")
    observer = LatencyObserver(marks.append)
    try:
        for frame in frames:
            await observer.on_push_frame(FramePushed(source=None, destination=None, frame=frame, direction=None, timestamp=0))  # pyright: ignore[reportArgumentType]
    finally:
        logger.remove(sink)
    return lines, marks


async def logged(*frames: Frame) -> list[str]:
    lines, _ = await observed(*frames)
    return lines


def opened(number: int) -> TurnOpened:
    return TurnOpened(hold=Hold(number, "held key"))


def resolved(number: int) -> TurnResolved:
    return TurnResolved(hold=Hold(number, "held key"))


async def told(*frames: Frame) -> list[Mark]:
    _, marks = await observed(*frames)
    return marks


async def test_an_answered_turn_is_told_mark_by_mark_each_once_however_often_its_frames_cross() -> None:
    release = VADUserStoppedSpeakingFrame()
    transcript = Words(text="hello", user_id="u", timestamp="t")
    token = LLMTextFrame("Hi")
    speaking = BotStartedSpeakingFrame()
    ended = UserStoppedSpeakingFrame()
    marks = await told(opened(1), release, release, transcript, transcript, resolved(1), ended, ended, token, token, speaking, speaking)
    assert marks == ["released", "transcript", "first LLM token", "first audio"]


async def test_a_hold_thrown_away_is_told_as_discarded_once_and_never_as_released() -> None:
    discarded = HoldDiscarded()
    assert await told(opened(1), discarded, discarded, resolved(1), UserStoppedSpeakingFrame()) == ["discarded"]


async def test_a_turn_with_no_words_in_it_is_told_and_what_hands_says_next_is_no_answer_to_it() -> None:
    """Nothing is sent to the model for it, so the page would wait on a reply that is not coming; and left open, its
    window would time the next announcement as that reply."""
    ended = UserStoppedSpeakingFrame()
    lines, marks = await observed(opened(1), VADUserStoppedSpeakingFrame(), resolved(1), ended, ended, BotStartedSpeakingFrame())
    assert marks == ["released", "no words"]
    assert lines[-1] == "latency: first audio, answering no user turn"


@pytest.mark.parametrize(("took_over", "answered"), [(False, True), (True, False)])
async def test_a_wordless_turn_that_took_nothing_over_leaves_the_reply_on_its_way_timed_from_its_own_release(
    took_over: bool, answered: bool
) -> None:
    """A turn the voice opened that heard no words cut nothing off, so the reply still coming answers the turn before;
    one that took over cut that reply off, so what hands says next answers nothing."""
    asked: list[Frame] = [opened(1), UserStartedSpeakingFrame(), VADUserStoppedSpeakingFrame(), Words(text="hello", user_id="u", timestamp="t"), resolved(1), UserStoppedSpeakingFrame()]
    noise: list[Frame] = [opened(2), *([UserStartedSpeakingFrame()] if took_over else []), VADUserStoppedSpeakingFrame(), resolved(2), UserStoppedSpeakingFrame()]
    lines, marks = await observed(*asked, *noise, LLMTextFrame("Hi"), BotStartedSpeakingFrame())
    assert marks == ["released", "transcript", "released", "no words", *(["first LLM token", "first audio"] if answered else [])]
    assert (lines[-1] == "latency: first audio, answering no user turn") is not answered


async def test_another_hold_done_with_says_nothing_of_the_hold_still_being_transcribed() -> None:
    """A press thrown away while the hold before it is with Whisper is done with at once; the earlier hold's words are
    still to come."""
    marks = await told(
        opened(1),
        VADUserStoppedSpeakingFrame(),
        opened(2),
        HoldDiscarded(),
        resolved(2),
        Words(text="hello", user_id="u", timestamp="t"),
        resolved(1),
        UserStoppedSpeakingFrame(),
    )
    assert marks == ["released", "discarded", "transcript"]


async def test_a_hold_with_no_words_in_a_turn_that_has_some_is_still_answered_and_timed_from_its_release() -> None:
    """A press while Whisper is on the hold before joins that hold's turn. Its own silence sends nothing more, but the
    turn is sent with the earlier hold's words, and the reply is the answer to the release the user made last."""
    lines, marks = await observed(
        opened(1),
        VADUserStoppedSpeakingFrame(),
        opened(2),
        Words(text="hello", user_id="u", timestamp="t"),
        resolved(1),
        VADUserStoppedSpeakingFrame(),
        resolved(2),
        UserStoppedSpeakingFrame(),
        LLMTextFrame("Hi"),
        BotStartedSpeakingFrame(),
    )
    assert marks == ["released", "transcript", "released", "first LLM token", "first audio"]
    assert lines[-1].startswith("latency: first audio") and "after key release" in lines[-1]


async def test_a_press_thrown_away_after_a_turn_was_sent_says_nothing_of_the_reply_still_awaited() -> None:
    """The turn the thrown-away press opened ends with no words, and it released nothing: the window still open is the
    turn's before it."""
    lines, marks = await observed(
        opened(1),
        VADUserStoppedSpeakingFrame(),
        Words(text="hello", user_id="u", timestamp="t"),
        resolved(1),
        UserStoppedSpeakingFrame(),
        opened(2),
        HoldDiscarded(),
        resolved(2),
        UserStoppedSpeakingFrame(),
        BotStartedSpeakingFrame(),
    )
    assert marks == ["released", "transcript", "discarded", "first audio"]
    assert "after key release" in lines[-1]


async def test_a_stage_failing_while_a_release_waits_is_told_as_failed_and_never_as_no_words() -> None:
    """Whisper pushes its error and then says it is done with the hold, so the turn ends with no words in it: the page
    and the log would both say the user said nothing. What hands says next is the failure, and no answer."""
    error = ErrorFrame("Whisper could not transcribe hold 1")
    lines, marks = await observed(opened(1), VADUserStoppedSpeakingFrame(), error, error, resolved(1), UserStoppedSpeakingFrame(), BotStartedSpeakingFrame())
    assert marks == ["released", "failed"]
    assert lines[-1] == "latency: first audio, answering no user turn"


async def test_a_hold_whisper_failed_on_in_a_turn_that_has_words_fails_nothing_and_the_reply_is_timed() -> None:
    """The turn is sent with the other hold's words and answered. The error's frame is still crossing boundaries
    upstream after the turn has ended, and is the same failure it was."""
    error = ErrorFrame("Whisper could not transcribe hold 1")
    lines, marks = await observed(
        opened(1),
        VADUserStoppedSpeakingFrame(),
        opened(2),
        VADUserStoppedSpeakingFrame(),
        error,
        resolved(1),
        Words(text="hello", user_id="u", timestamp="t"),
        resolved(2),
        UserStoppedSpeakingFrame(),
        error,
        LLMTextFrame("Hi"),
        BotStartedSpeakingFrame(),
    )
    assert marks == ["released", "released", "transcript", "first LLM token", "first audio"]
    assert "after key release" in lines[-1]


async def test_an_error_with_no_release_waiting_says_nothing_of_the_next_turn() -> None:
    """A narration nobody asked for failing is no failure of the hold pressed after it."""
    marks = await told(ErrorFrame("no voice"), opened(1), VADUserStoppedSpeakingFrame(), resolved(1), UserStoppedSpeakingFrame())
    assert marks == ["released", "no words"]


async def test_the_model_failing_after_the_words_were_heard_is_told_as_failed() -> None:
    marks = await told(
        opened(1),
        VADUserStoppedSpeakingFrame(),
        Words(text="hello", user_id="u", timestamp="t"),
        resolved(1),
        UserStoppedSpeakingFrame(),
        ErrorFrame("LLM completion timeout"),
    )
    assert marks == ["released", "transcript", "failed"]


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
        Words(text="hello", user_id="u", timestamp="t"),
        BotStartedSpeakingFrame(),
    )
    assert [line.split(" ")[1] for line in lines] == ["transcript", "first"]
    assert all("after key release" in line for line in lines)


@pytest.mark.parametrize("frame", [BotStartedSpeakingFrame(), Words(text="x", user_id="u", timestamp="t")])
async def test_nothing_is_measured_from_a_release_that_never_happened(frame: Frame) -> None:
    """A transcript with no user turn open has no release to be late from, so it is not reported as instant."""
    assert all("after key release" not in line for line in await logged(frame))


async def test_a_hold_thrown_away_is_no_release_to_measure_a_narration_from() -> None:
    lines = await logged(VADUserStartedSpeakingFrame(), HoldDiscarded(), BotStartedSpeakingFrame())
    assert lines == ["latency: first audio, answering no user turn"]


async def test_a_narration_after_a_user_turn_is_still_measured_as_answering_nobody() -> None:
    """The turn a user opened has to close when the audio it was waiting for finishes. Left open, the first user
    turn of a session stays open for the rest of it, every later narration finds `first audio` already marked,
    and the line this observer exists for is written nowhere — while the numbers in docs/architecture.md come
    from exactly that line."""
    lines = await logged(
        VADUserStartedSpeakingFrame(),
        VADUserStoppedSpeakingFrame(),
        Words(text="hello", user_id="u", timestamp="t"),
        BotStartedSpeakingFrame(),
        BotStoppedSpeakingFrame(),
        BotStartedSpeakingFrame(),
        BotStoppedSpeakingFrame(),
        BotStartedSpeakingFrame(),
    )
    assert lines[-2:] == ["latency: first audio, answering no user turn"] * 2


async def test_a_barge_in_keeps_the_turn_it_opened_while_the_speaker_is_stopping() -> None:
    """The user interrupting releases the key before the speaker's stop arrives, and the window that release
    opened is waiting for its own first audio — so the stop belonging to the utterance being cut off leaves it
    open, and the next turn is still measured."""
    lines = await logged(
        VADUserStartedSpeakingFrame(),
        VADUserStoppedSpeakingFrame(),
        Words(text="first", user_id="u", timestamp="t"),
        BotStartedSpeakingFrame(),
        VADUserStartedSpeakingFrame(),
        BotStoppedSpeakingFrame(),
        VADUserStoppedSpeakingFrame(),
        Words(text="second", user_id="u", timestamp="t"),
    )
    assert lines[-1].startswith("latency: transcript") and "after key release" in lines[-1]


async def test_a_barge_in_over_a_still_pushing_utterance_keeps_its_own_measurement() -> None:
    """One utterance raises several started-speaking frames. A user barging in between two of them has a window
    opened by their release that those trailing frames would otherwise mark `first audio` on — before any reply
    exists — and the stop ending the cut-off utterance would then close the window and drop the reply's
    transcript, first token and first audio together. What marks first audio is the speaker going from silent to
    sounding, and a trailing push of an utterance already sounding starts nothing."""
    lines = await logged(
        BotStartedSpeakingFrame(),
        VADUserStartedSpeakingFrame(),
        BotStartedSpeakingFrame(),
        BotStoppedSpeakingFrame(),
        VADUserStoppedSpeakingFrame(),
        Words(text="wait", user_id="u", timestamp="t"),
        BotStartedSpeakingFrame(),
    )
    assert [line.split(" ")[1] for line in lines] == ["first", "transcript", "first"]
    assert all("after key release" in line for line in lines[1:])


async def test_an_announcement_that_starts_while_the_key_is_held_is_still_said() -> None:
    """The larger half of what this daemon says starts whenever a session stops, including mid-press, and its
    own time is what a Stop's audit line is subtracted from. A turn opened at the press and waiting for a
    release used to swallow that announcement: it was logged nowhere, and the trailing push of the very same
    utterance then marked `first audio` on the release when it came, destroying the reply's measurement."""
    lines = await logged(
        VADUserStartedSpeakingFrame(),
        BotStartedSpeakingFrame(),
        VADUserStoppedSpeakingFrame(),
        BotStartedSpeakingFrame(),
        BotStoppedSpeakingFrame(),
        Words(text="held", user_id="u", timestamp="t"),
        BotStartedSpeakingFrame(),
    )
    assert lines[0] == "latency: first audio, answering no user turn"
    # The reply's own audio is what `first audio` measures, never the announcement still playing at the release.
    assert [line.split(" ")[1] for line in lines[1:]] == ["transcript", "first"]
    assert all("after key release" in line for line in lines[1:])


async def test_one_release_is_one_window_however_often_it_crosses_a_boundary() -> None:
    """The stop frame is pushed once per processor boundary, as every other frame here is. A window opened on
    the frame rather than on the user falling silent would throw away the marks the first push's window took
    and restart the measurement from a later zero — so a milestone landing between two pushes is said twice,
    the second time timed from a moment the user had already finished speaking at."""
    release = VADUserStoppedSpeakingFrame()
    transcript = Words(text="once", user_id="u", timestamp="t")
    lines = await logged(VADUserStartedSpeakingFrame(), release, transcript, release, transcript, BotStartedSpeakingFrame())
    assert [line.split(" ")[1] for line in lines] == ["transcript", "first"]
