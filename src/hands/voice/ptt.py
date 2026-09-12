"""Push-to-talk: the key is the voice activity detector.

Pipecat's turn machinery, and the segmented Whisper service that decides when
to transcribe, both run off VAD frames. So the key does not gate audio frames;
it *is* the VAD. `Gate` is the pure state (which position the key is in) and
`KeyVAD` is the analyzer that reports full voice confidence while the key is
held and none while it is up. Everything downstream is stock Pipecat: the VAD
turn strategies open and close the user turn, broadcast the interruption that
flushes queued speech on barge-in, and tell Whisper what to transcribe. While
the key is up, whatever the microphone hears, including the pipeline's own
speech, is silence to the pipeline.
"""

from dataclasses import dataclass
from typing import Literal

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams

# [LAW:types-are-the-program] the key has exactly two positions and a move is
# only a transition when the position changes; a repeated press or release is
# a no-op by construction, not by a guard at the call site.
Key = Literal["up", "down"]
Turn = Literal["start", "stop", "none"]

# A 20 ms analysis frame. The VAD counts frames against start_secs/stop_secs,
# so this is also the resolution of the turn boundary.
FRAME_SECS = 0.02

# [LAW:no-mode-explosion] the key is binary, so the thresholds are fixed, not
# tunable: any confidence above zero is speech, volume never vetoes the key,
# and the turn opens and closes within one analysis frame of the key.
KEY_VAD_PARAMS = VADParams(confidence=0.5, start_secs=FRAME_SECS, stop_secs=FRAME_SECS, min_volume=0.0)


@dataclass(frozen=True)
class Gate:
    """The pure push-to-talk state: which position the key is in."""

    key: Key = "up"

    def moved(self, to: Key) -> tuple["Gate", Turn]:
        """Return the gate after the key moves and the turn event that implies."""
        if to == self.key:
            return self, "none"
        return Gate(key=to), "start" if to == "down" else "stop"

    @property
    def confidence(self) -> float:
        """Voice confidence as the VAD sees it: all or nothing."""
        return 1.0 if self.key == "down" else 0.0


class KeyVAD(VADAnalyzer):
    """A voice activity detector whose only input is the push-to-talk key."""

    def __init__(self, *, sample_rate: int | None = None) -> None:
        super().__init__(sample_rate=sample_rate, params=KEY_VAD_PARAMS)
        self._gate = Gate()

    def move_key(self, to: Key) -> Turn:
        """Report a key position; the edge that reads the keyboard calls this."""
        self._gate, turn = self._gate.moved(to)
        return turn

    def num_frames_required(self) -> int:
        return int(self.sample_rate * FRAME_SECS)

    def voice_confidence(self, buffer: bytes) -> float:
        # [LAW:dataflow-not-control-flow] the audio is ignored on purpose: the
        # key is the signal, and the same call answers for every buffer.
        return self._gate.confidence
