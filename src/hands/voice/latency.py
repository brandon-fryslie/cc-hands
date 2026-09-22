"""Voice-to-voice latency, measured where it is felt.

The number the spike exists to produce is the time from key release to the
first audio the speaker plays. This observer watches frames cross processor
boundaries and logs, per user turn, how long after the release the transcript
arrived, the first LLM token arrived, and the output transport started
speaking. Each milestone is logged as it happens, so a leg that fails still
leaves the legs before it on record. It performs no effect but logging.
"""

import time
from dataclasses import dataclass, field

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMTextFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

Mark = str


@dataclass
class TurnMarks:
    """Monotonic timestamps for one user turn, keyed by milestone."""

    pressed: float
    released: float | None = None
    marks: dict[Mark, float] = field(default_factory=lambda: dict[Mark, float]())

    def since_release(self, mark: Mark) -> float | None:
        at = self.marks.get(mark)
        if self.released is None or at is None:
            return None
        return at - self.released


FIRST_AUDIO: Mark = "first audio"

# The first arrival of each of these frame types after a release is a mark.
_MILESTONES: dict[type[Frame], Mark] = {
    TranscriptionFrame: "transcript",
    LLMTextFrame: "first LLM token",
    BotStartedSpeakingFrame: FIRST_AUDIO,
}


class LatencyObserver(BaseObserver):
    """Log each milestone's distance from the key release, per turn."""

    def __init__(self) -> None:
        # Pipecat's observer base takes untyped **kwargs; nothing is passed.
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self._turn: TurnMarks | None = None
        # Whether the speaker is already sounding, so one utterance is one line however often it is announced.
        self._speaking = False

    async def on_push_frame(self, data: FramePushed) -> None:
        now = time.monotonic()
        frame = data.frame
        if isinstance(frame, BotStoppedSpeakingFrame):
            self._speaking = False
            if self._turn is not None and FIRST_AUDIO in self._turn.marks:
                # The turn closes when the audio it was waiting for has finished, so the next utterance with no
                # user behind it is recognised as one. Left open, the first user turn of a session stays open for
                # the rest of it: every later narration finds `first audio` already marked and is logged nowhere,
                # which is the measurement this observer exists for. A barge-in opens its own turn before this
                # frame arrives and that turn has no first audio yet, so it is not closed here.
                self._turn = None
            return
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._turn = TurnMarks(pressed=now)
            return
        if isinstance(frame, BotStartedSpeakingFrame) and self._turn is None:
            # Speech nobody asked for out loud — a turn's summary, an announcement — has no release to measure
            # from, and is the larger half of what this daemon says. Logged with its own time and nothing else,
            # which is what a Stop's audit line is subtracted from to get how long a turn took to be heard.
            # What is logged once is the speaking, not the frame: an utterance raised four of these between one
            # start and one stop, measured live, and four identical lines are four utterances to a reader
            # [LAW:one-source-of-truth].
            if not self._speaking:
                logger.info("latency: first audio, answering no user turn")
            self._speaking = True
            return
        if self._turn is None:
            return
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._turn.released = now
            return
        mark = _MILESTONES.get(type(frame))
        if mark is not None and mark not in self._turn.marks:
            self._turn.marks[mark] = now
            logger.info(f"latency: {mark} {_fmt(self._turn.since_release(mark))} after key release")


def _fmt(seconds: float | None) -> str:
    return "before release" if seconds is None else f"{seconds * 1000:.0f} ms"
