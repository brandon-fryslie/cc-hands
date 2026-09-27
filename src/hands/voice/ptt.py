"""Push-to-talk: the key is the voice activity detector and the microphone mute.

Pipecat's turn machinery, and the segmented Whisper service that decides when
to transcribe, both run off VAD frames. So the key *is* the VAD: `KeyVAD`
reports full voice confidence while the key is held and none while it is up.
Whisper also keeps the last second of audio from before the turn opened, so
the key is also the mute: the microphone bytes are silence while the key is up
(applied where they are captured, in `hands.voice.microphone`). Audio frames
always flow; only their content and their voice confidence follow the key. Everything
downstream is stock Pipecat: the VAD turn strategies open and close the user
turn, broadcast the interruption that flushes queued speech on barge-in, and
tell Whisper what to transcribe. While the key is up, whatever the microphone
hears, including the pipeline's own speech, is silence to the pipeline.
"""

from dataclasses import dataclass
from typing import Literal

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.frames.frames import InputAudioRawFrame

from hands.voice.hold import Move

# [LAW:types-are-the-program] the key is up, down, or up with the turn it held thrown away: `dropped` hears and says
# nothing, like up, and tells Whisper not to transcribe what the turn recorded. The hold (`hands.voice.hold`) decides
# every move, so the gate never sees a press or a release that is not a transition.
Key = Literal["up", "down", "dropped"]

@dataclass(kw_only=True)
class KeyedAudio(InputAudioRawFrame):
    """Microphone audio, and the key position it was captured under.

    [LAW:no-ambient-temporal-coupling] the key travels with the audio, in capture order, so whatever reads it later
    reads the key as it was when this sound was recorded, never as it is by the time the frame arrives.
    """

    key: Key


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

    def after(self, move: Move) -> "Gate":
        """The gate once the hold has moved the turn."""
        match move:
            case "start":
                return Gate("down")
            case "stop":
                return Gate("up")
            case "drop":
                return Gate("dropped")

    @property
    def confidence(self) -> float:
        """Voice confidence as the VAD sees it: all or nothing."""
        return 1.0 if self.key == "down" else 0.0

    def audible(self, audio: bytes) -> bytes:
        """The microphone bytes as the pipeline hears them: intact or silence."""
        # [LAW:dataflow-not-control-flow] a frame of the same length always
        # goes out, so the VAD and Whisper see an unbroken stream; the key
        # only decides its content.
        return audio if self.key == "down" else bytes(len(audio))


class PushToTalk:
    """The one owner of the key position; the talk key's edge writes, the microphone, the VAD, and Whisper read."""

    # [LAW:no-shared-mutable-globals] three pipeline components consult the key,
    # so it lives here once with one writer, not once in each of them.
    def __init__(self) -> None:
        self._gate = Gate()

    def move(self, move: Move) -> None:
        """Report what the hold did to the turn; the edge that reads the keyboard calls this."""
        self._gate = self._gate.after(move)

    @property
    def gate(self) -> Gate:
        # Read from the audio callback thread too: one attribute load of a frozen value, whole either way.
        return self._gate


class KeyVAD(VADAnalyzer):
    """A voice activity detector whose only input is the push-to-talk key."""

    def __init__(self, key: PushToTalk, *, sample_rate: int | None = None) -> None:
        super().__init__(sample_rate=sample_rate, params=KEY_VAD_PARAMS)
        self._key = key

    def num_frames_required(self) -> int:
        return int(self.sample_rate * FRAME_SECS)

    def voice_confidence(self, buffer: bytes) -> float:
        # [LAW:dataflow-not-control-flow] the audio is ignored on purpose: the
        # key is the signal, and the same call answers for every buffer.
        return self._key.gate.confidence
