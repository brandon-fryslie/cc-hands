"""What the user heard hands say in its own words: one ledger at the speaker, told to the brain on every request.

Every line hands says as written passes the one processor ahead of the speaker, whichever part of hands queued it: a
session's nudge or a deadline from the relay, a session gone from the narrator, a lane the brain's stage handed on, or
a failure the system channel queued past the model. The brain's own words are in its history; what hands said beside
them was nowhere the brain could see, so on 2026-09-30 and again on 2026-10-03 the user answered a question hands had
read out and the brain asked what they were talking about (hands-wire-6ic.b9j). Kept here, the last lines ride the tail
of every request the brain makes, so what the user says next can answer any of them.

[LAW:single-enforcer] the only place a line of hands' is written down as heard: a site that builds a TTSSpeakFrame
never records it, and a site that bypasses this processor is a bug, not a choice.
"""

from collections import deque

from pipecat.frames.frames import Frame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from hands.sessions.audit import HandsSpoke, Record

# How many of hands' lines the brain is reminded of: enough to cover what was said while it answered, and bounded,
# since every request carries them.
KEPT = 8


class Ledger(FrameProcessor):
    """Passes everything on to the speaker, and keeps the last lines hands said as written."""

    def __init__(self, record: Record, kept: int = KEPT) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._record = record
        self._lines: deque[str] = deque(maxlen=kept)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSSpeakFrame):
            self._lines.append(frame.text)
            # [LAW:nothing-unseen] every line of hands' the user heard, where it was heard.
            self._record(HandsSpoke(frame.text))
        await self.push_frame(frame, direction)

    def lately(self) -> tuple[str, ...]:
        """The last lines hands said as written, oldest first."""
        return tuple(self._lines)
