"""Push-to-talk: the key is the voice activity detector and the microphone mute.

Pipecat's turn machinery, and the segmented Whisper service that decides when
to transcribe, both run off VAD frames. So the key *is* the VAD: every
microphone frame carries the key it was captured under, and Whisper
(`hands.voice.whisper`) says the user started or stopped speaking where those
keys change. The key is also the mute: the microphone bytes are silence unless
the key is pressed (applied where they are captured, in
`hands.voice.microphone`), and a hold's audio begins at its press, so what was
said before the hold meant talk is kept. Audio frames always flow; only their
content and their key follow the key. The stock VAD turn strategies open and
close the user turn on those frames and broadcast the interruption that flushes
queued speech on barge-in. While the key is up, whatever the microphone hears,
including the pipeline's own speech, is silence to the pipeline.
"""

from dataclasses import dataclass
from typing import Literal

from pipecat.frames.frames import InputAudioRawFrame

from hands.voice.hold import Move

# [LAW:types-are-the-program] the key is up; arming, pressed but not yet meaning talk, which hears so the words said
# before it does are kept; down, a hold; or dropped, up with the hold thrown away, which tells Whisper not to transcribe
# what it recorded. The hold (`hands.voice.hold`) decides every move, so the gate never sees one that is not a
# transition.
Key = Literal["up", "arming", "down", "dropped"]

@dataclass(kw_only=True)
class KeyedAudio(InputAudioRawFrame):
    """Microphone audio, and the key position it was captured under.

    [LAW:no-ambient-temporal-coupling] the key travels with the audio, in capture order, so whatever reads it later
    reads the key as it was when this sound was recorded, never as it is by the time the frame arrives.
    """

    key: Key


@dataclass(frozen=True)
class Gate:
    """The pure push-to-talk state: which position the key is in."""

    key: Key = "up"

    def after(self, move: Move) -> "Gate":
        """The gate once the hold has moved the turn."""
        match move:
            case "arm":
                return Gate("arming")
            case "disarm":
                return Gate("up")
            case "start":
                return Gate("down")
            case "stop":
                return Gate("up")
            case "drop":
                return Gate("dropped")

    @property
    def turn_open(self) -> bool:
        """The key has been held past the hold: what the microphone hears now is the user's turn."""
        return self.key == "down"

    def audible(self, audio: bytes) -> bytes:
        """The microphone bytes as the pipeline hears them: intact while the key is pressed, silence otherwise."""
        # [LAW:dataflow-not-control-flow] a frame of the same length always
        # goes out, so Whisper sees an unbroken stream; the key
        # only decides its content.
        return audio if self.key in ("arming", "down") else bytes(len(audio))


class PushToTalk:
    """The one owner of the key position; the talk key's edge writes, the microphone reads and tags every frame with it."""

    # [LAW:no-shared-mutable-globals] the event loop writes the key and the capture thread reads it,
    # so it lives here once with one writer.
    def __init__(self) -> None:
        self._gate = Gate()

    def move(self, move: Move) -> None:
        """Report what the hold did to the turn; the edge that reads the keyboard calls this."""
        self._gate = self._gate.after(move)

    @property
    def gate(self) -> Gate:
        # Read from the audio callback thread too: one attribute load of a frozen value, whole either way.
        return self._gate

