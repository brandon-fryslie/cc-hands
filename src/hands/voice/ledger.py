"""What the user heard hands say in its own words: one ledger at the speaker, told to the brain on every request.

Every line hands says as written passes the one processor ahead of the speaker, whichever part of hands queued it: a
session's nudge or a deadline from the relay, a session gone from the narrator, a lane the brain's stage handed on, or
a failure the system channel queued past the model. The brain's own words are in its history; what hands said beside
them was nowhere the brain could see, so on 2026-09-30 and again on 2026-10-03 the user answered a question hands had
read out and the brain asked what they were talking about (hands-wire-6ic.b9j). Kept here, the last lines ride the tail
of every request the brain makes, so what the user says next can answer any of them.

Written down at the door and not where the speaker sounded it: a line the user cut off is still one the brain is told
hands said, since the cost of telling it a line it need not have is a sentence, and the cost of missing one is the bug.

[LAW:single-enforcer] the only place a line of hands' is written down as said to the user: a site that builds a
TTSSpeakFrame never records it, and a site that bypasses this processor is a bug, not a choice. The system channel's
`Announced` records which way it routed a fault, speech or screen, and not what was said.
"""

import time
from collections import deque
from collections.abc import Callable

from pipecat.frames.frames import Frame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from hands.sessions.audit import HandsSpoke, Record

# How many of hands' lines the brain is reminded of: enough to cover what was said while it answered, and bounded,
# since every request carries them.
KEPT = 8
# How long a line stays one the user may be answering: a question read out at nine is not what "yes, go ahead" at
# half past eleven answers, and the sessions' own pending requests ride the same tail for as long as they wait.
WITHIN_SECONDS = 600.0


class Ledger(FrameProcessor):
    """Passes everything on to the speaker, and keeps the last lines hands said as written."""

    def __init__(self, record: Record, clock: Callable[[], float] = time.monotonic, kept: int = KEPT, within: float = WITHIN_SECONDS) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._record = record
        self._clock = clock
        self._within = within
        self._lines: deque[tuple[float, str]] = deque(maxlen=kept)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSSpeakFrame):
            self._lines.append((self._clock(), frame.text))
            # [LAW:nothing-unseen] every line of hands' the user heard, where it was heard.
            self._record(HandsSpoke(frame.text))
        await self.push_frame(frame, direction)

    def lately(self) -> tuple[str, ...]:
        """The last lines hands said as written in the last `within` seconds, oldest first."""
        since = self._clock() - self._within
        return tuple(text for at, text in self._lines if at >= since)
