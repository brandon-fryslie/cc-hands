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

FIRST_AUDIO: Mark = "first audio"

# The first arrival of each of these frame types in an open window is a mark. `first audio` is not among them:
# the speaker starting is a transition into sounding rather than a frame, and the frame announcing it is pushed
# once per processor boundary it crosses.
_MILESTONES: dict[type[Frame], Mark] = {
    TranscriptionFrame: "transcript",
    LLMTextFrame: "first LLM token",
}


@dataclass
class Window:
    """The stretch one turn's milestones are measured in: it opens on the key release and holds what has landed.

    A turn before its release is not represented here, because there is nothing for it to hold. The release is
    what every number in the window is a distance from, so a turn without one measures nothing, and the state
    that would carry it would carry only a timestamp nobody reads [LAW:types-are-the-program]. The release is
    therefore not a field that may be absent — it is the thing that brings the window into being.
    """

    released: float
    marks: dict[Mark, float] = field(default_factory=lambda: dict[Mark, float]())


def _take(window: Window, mark: Mark, at: float) -> None:
    """Log `mark`'s distance from the release, the first time it lands in this window.

    Demanding a `Window` rather than an optional one is the whole of the rule a mark is subject to: a milestone
    belongs to a turn once that turn has a release, so a caller holding no window has nothing to take and no
    distance to report [LAW:parse-dont-validate]. A transcript that arrived before the release did not answer
    it, and attributing one to the other would report a number measuring nothing.
    """
    if mark in window.marks:
        return
    window.marks[mark] = at
    logger.info(f"latency: {mark} {(at - window.released) * 1000:.0f} ms after key release")


class LatencyObserver(BaseObserver):
    """Log each milestone's distance from the key release, per turn."""

    def __init__(self) -> None:
        # Pipecat's observer base takes untyped **kwargs; nothing is passed.
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self._window: Window | None = None
        # Who is sounding, on each side of the conversation. Every frame below is pushed once per processor
        # boundary it crosses, so what these two record is the transition, and the transition is the event:
        # the speaker going from silent to sounding is the first audio, and the user falling silent is the
        # release [LAW:one-source-of-truth].
        self._speaking = False
        self._holding = False

    async def on_push_frame(self, data: FramePushed) -> None:
        now = time.monotonic()
        frame = data.frame
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._holding = True
            return
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._holding:
                # The release is what the window's numbers are measured from, so it is also what opens the
                # window — and what opens it is the user falling silent, not each push of the frame saying so.
                # Opened on the frame, the second push would throw away the marks the first push's window had
                # taken and restart the measurement from a later zero, logging a milestone twice and timing it
                # from a moment the user was already done speaking at.
                self._window = Window(released=now)
            self._holding = False
            return
        if isinstance(frame, BotStartedSpeakingFrame):
            if not self._speaking:
                self._first_audio(now)
            self._speaking = True
            return
        if isinstance(frame, BotStoppedSpeakingFrame):
            self._speaking = False
            if self._window is not None and FIRST_AUDIO in self._window.marks:
                # The window closes when the audio it was waiting for has finished, so the next utterance with
                # no user behind it is recognised as one. Left open, the first turn of a session stays open for
                # the rest of it: every later narration finds `first audio` already marked and is logged
                # nowhere, which is the measurement this observer exists for. A window that a barge-in's release
                # opened has no first audio yet, so the stop ending the utterance being cut off leaves it alone.
                self._window = None
            return
        mark = _MILESTONES.get(type(frame))
        if mark is not None and self._window is not None:
            _take(self._window, mark, now)

    def _first_audio(self, at: float) -> None:
        """Say when the speaker went from silent to sounding: its distance from a release, or that none asked for it.

        Speech nobody asked for out loud — a turn's summary, an announcement — has no release to measure from,
        and is the larger half of what this daemon says. Its own time is what a Stop's audit line is subtracted
        from to get how long a turn took to be heard, so it is said even while the user holds the key down.

        What is said once is the transition, never the frame: an utterance raised four started-speaking pushes
        between one start and one stop, measured live, and four identical lines are four utterances to a reader
        [LAW:one-source-of-truth]. That same transition is what keeps the trailing pushes of an utterance a user
        barged in over from marking `first audio` on the window that user's release then opens — the speaker was
        already sounding when they arrived, so they start nothing.
        """
        if self._window is None:
            logger.info("latency: first audio, answering no user turn")
            return
        _take(self._window, FIRST_AUDIO, at)
