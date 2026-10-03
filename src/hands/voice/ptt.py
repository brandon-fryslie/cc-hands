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

The key is pressed at a place: the desk, by the talk key, or the phone, by its page's button. The gate holds where the
last turn was opened, and that place is where hands is: the pipeline hears that place's microphone and answers on its
speaker, and the other place's moves cannot touch a turn that is not theirs.
"""

from dataclasses import dataclass
from typing import Literal

from pipecat.frames.frames import InputAudioRawFrame

from hands.core.place import Place
from hands.sessions.audit import Moved, Record
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
    """The pure push-to-talk state: which position the key is in, and the place hands is at."""

    key: Key = "up"
    place: Place = "desk"

    def after(self, move: Move, at: Place) -> "Gate":
        """The gate once a hold at `at` has moved the turn."""
        # [LAW:one-source-of-truth] the place is moved only by a turn opening there, or by the phone coming and going:
        # a Shift typed at the desk while the user talks on the phone arms nothing and ends nothing of theirs.
        match at == self.place, move:
            case True, _:
                return Gate(_key_after(move), self.place)
            case False, "start":
                return Gate("down", at)
            case False, _:
                return self

    def moved(self, to: Place) -> "Gate":
        """The gate once hands is at `to`: a hold open at the place it leaves is thrown away, not sent."""
        match to == self.place, self.key:
            case True, _:
                return self
            case False, "arming" | "down":
                return Gate("dropped", to)
            case False, "up" | "dropped":
                return Gate(self.key, to)

    def hears(self, place: Place) -> bool:
        """Whether the pipeline hears the microphone at `place`: only hands' own, so one stream of frames reaches Whisper."""
        return place == self.place

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


def _key_after(move: Move) -> Key:
    match move:
        case "arm":
            return "arming"
        case "disarm" | "stop":
            return "up"
        case "start":
            return "down"
        case "drop" | "expire":
            return "dropped"


class PushToTalk:
    """The one owner of the key position; the talk key's edge writes, the microphone reads and tags every frame with it."""

    # [LAW:no-shared-mutable-globals] the event loop writes the key and the capture thread reads it,
    # so it lives here once with one writer.
    def __init__(self, record: Record) -> None:
        self._gate = Gate()
        self._record = record

    def move(self, move: Move, at: Place) -> None:
        """Report what the hold at `at` did to the turn; the edge that reads the talk key or the phone's button calls this."""
        self._become(self._gate.after(move, at), "turn")

    def go(self, to: Place) -> None:
        """Report that hands is at `to` now: the phone came, or went."""
        self._become(self._gate.moved(to), "call")

    def _become(self, gate: Gate, by: Literal["turn", "call"]) -> None:
        before, self._gate = self._gate, gate
        # [LAW:nothing-unseen] the one writer of the gate says each move of the place, whichever edge made it.
        if gate.place != before.place:
            self._record(Moved(to=gate.place, by=by, dropped=gate.key == "dropped" and before.key != "dropped"))

    @property
    def gate(self) -> Gate:
        # Read from the audio callback thread too: one attribute load of a frozen value, whole either way.
        return self._gate

