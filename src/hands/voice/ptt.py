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

from hands.core.place import Modality, Place, modality_at
from hands.sessions.audit import Moved, Record
from hands.voice.hold import Move, TurnMove
from hands.voice.trigger import Edge, place_of

# [LAW:types-are-the-program] the key is up; listening, the desk heard between turns while engaged, so a turn the voice
# opens keeps the words said before the detector was sure of them; arming, pressed but not yet meaning talk, which hears
# so the words said before it does are kept; down, a hold; or dropped, up with the hold thrown away, which tells Whisper
# not to transcribe what it recorded. The edges decide every move, so the gate never sees one that is not a transition.
Key = Literal["up", "listening", "arming", "down", "dropped"]

@dataclass(kw_only=True)
class KeyedAudio(InputAudioRawFrame):
    """Microphone audio, and the key position it was captured under.

    [LAW:no-ambient-temporal-coupling] the key travels with the audio, in capture order, so whatever reads it later
    reads the key as it was when this sound was recorded, never as it is by the time the frame arrives.
    """

    key: Key


@dataclass(frozen=True)
class Gate:
    """The pure push-to-talk state: which position the key is in, the place hands is at, and whether the desk listens
    between turns."""

    key: Key = "up"
    place: Place = "desk"
    # [LAW:one-source-of-truth] engaged conversation is the desk's: at the phone, its button alone opens the microphone.
    listens: bool = False

    def took(self, move: Move, at: Place) -> Move | None:
        """What a hold at `at` did to the turn, as the gate takes it; None where it does nothing to it.

        [LAW:one-source-of-truth] the place is moved only by a turn opening there, or by the phone coming and going: a
        Shift typed at the desk while the user talks on the phone arms nothing and ends nothing of theirs. A turn opened
        at one place while a hold is open at the other is a second hand on a second key, which drops both, as a key
        pressed while the talk key is held drops the turn; and the end of a hold the gate has already thrown away, as a
        call came or went, ends nothing more. The desk's listening is taken wherever hands is.
        """
        match at == self.place, move, self.key:
            case _, "listen" | "deafen", _:
                return move
            case True, "stop" | "drop" | "expire", "dropped":
                return None
            case True, _, _:
                return move
            case False, "start", "arming" | "down":
                return "drop"
            case False, "start", "up" | "listening" | "dropped":
                return "start"
            case False, _, _:
                return None

    def after(self, move: Move, at: Place) -> "Gate":
        """The gate once a hold at `at` has moved the turn."""
        match self.took(move, at):
            case None:
                return self
            case "listen" | "deafen" as listening:
                listens = listening == "listen"
                return Gate(self._resting(self.place, listens), self.place, listens)
            case taken:
                return Gate(_key_after(taken, self._rest(at, self.listens)), at, self.listens)

    def moved(self, to: Place) -> "Gate":
        """The gate once hands is at `to`: a hold open at the place it leaves is thrown away, not sent."""
        match to == self.place, self.key:
            case True, _:
                return self
            case False, "arming" | "down":
                return Gate("dropped", to, self.listens)
            case False, "dropped":
                return Gate("dropped", to, self.listens)
            case False, "up" | "listening":
                return Gate(self._rest(to, self.listens), to, self.listens)

    def _resting(self, place: Place, listens: bool) -> Key:
        """The key once the desk starts or stops listening: a turn open is sent, a press arming is let go of, and one
        thrown away stays thrown away until the next opens."""
        match place, self.key:
            case "phone", _:
                return self.key
            case "desk", "dropped":
                return "dropped"
            case "desk", _:
                return self._rest(place, listens)

    @staticmethod
    def _rest(place: Place, listens: bool) -> Key:
        """The key between turns at `place`."""
        return "listening" if listens and place == "desk" else "up"

    def hears(self, place: Place) -> bool:
        """Whether the pipeline hears the microphone at `place`: only hands' own, so one stream of frames reaches Whisper."""
        return place == self.place

    @property
    def turn_open(self) -> bool:
        """The key has been held past the hold: what the microphone hears now is the user's turn."""
        return self.key == "down"

    def audible(self, audio: bytes) -> bytes:
        """The microphone bytes as the pipeline hears them: intact while the key is pressed or the desk listens, silence
        otherwise."""
        # [LAW:dataflow-not-control-flow] a frame of the same length always
        # goes out, so Whisper sees an unbroken stream; the key
        # only decides its content.
        return audio if self.key in ("listening", "arming", "down") else bytes(len(audio))


def _key_after(move: TurnMove, rest: Key) -> Key:
    match move:
        case "arm":
            return "arming"
        case "disarm" | "stop":
            return rest
        case "start":
            return "down"
        case "drop" | "expire":
            return "dropped"


class PushToTalk:
    """The one owner of the key position and of where the user is: the talk key's edge writes, the microphone reads and
    tags every frame with it; and whether the user can see a screen there, which the place they talk from sets and they
    can switch by voice."""

    # [LAW:no-shared-mutable-globals] the event loop writes the key and the capture thread reads it,
    # so it lives here once with one writer.
    def __init__(self, record: Record) -> None:
        self._gate = Gate()
        self._modality: Modality = modality_at(self._gate.place)
        # The edge that opened the last turn, as it opened it: a trigger switched since does not rewrite it.
        self._opened: Edge = "held key"
        self._record = record

    def move(self, move: Move, by: Edge) -> Move | None:
        """Report what the hold made by `by` did to the turn, and get back what the gate took it as, to cue and tell: the
        desk's edge and the phone's button call this."""
        at = place_of(by)
        taken = self._gate.took(move, at)
        if taken == "start":
            self._opened = by
        self._become(self._gate.after(move, at), "turn")
        return taken

    def go(self, to: Place) -> None:
        """Report that hands is at `to` now: the phone came, or went."""
        self._become(self._gate.moved(to), "call")

    def switch(self, to: Modality) -> None:
        """Report that the user asked to be taken as `to` from now on, until hands next moves between the desk and the phone."""
        self._modality = to

    def _become(self, gate: Gate, by: Literal["turn", "call"]) -> None:
        before, self._gate = self._gate, gate
        # [LAW:nothing-unseen] the one writer of the gate says each move of the place, whichever edge made it.
        if gate.place != before.place:
            # [LAW:one-source-of-truth] the way the user talks sets the modality, so a move sets it in the same write.
            self._modality = modality_at(gate.place)
            self._record(Moved(to=gate.place, by=by, dropped=gate.key == "dropped" and before.key != "dropped"))

    @property
    def gate(self) -> Gate:
        # Read from the audio callback thread too: one attribute load of a frozen value, whole either way.
        return self._gate

    @property
    def modality(self) -> Modality:
        return self._modality

    @property
    def opened(self) -> Edge:
        return self._opened

