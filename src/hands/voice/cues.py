"""What the user is shown and played as the talk key opens and closes a turn: a terminal line and a short tone.

The tone goes to the speaker ahead of anything else and is short and soft enough to leave the microphone open: it
is not the pipeline's speech, so it holds nothing shut, and a word said over it reaches Whisper whole.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np

from hands.voice.hold import Move

CUE_SECONDS = 0.06
# About -16 dBFS against replies that peak near full scale: heard over the room, never louder than the voice.
CUE_LEVEL = 0.16


@dataclass(frozen=True)
class Cue:
    """One edge of a turn: the line the terminal shows, and the pitches its tone glides between, one tone per pair."""

    line: str
    glides: tuple[tuple[float, float], ...]


# Rising opens, falling sends, and a low pair drops: three shapes told apart without looking.
OPENED = Cue("turn: started", ((660.0, 990.0),))
SENT = Cue("turn: ended", ((990.0, 660.0),))
DROPPED = Cue("turn: dropped", ((330.0, 330.0), (330.0, 330.0)))


def cues(move: Move) -> tuple[Cue, ...]:
    """What a move shows and plays: the turn's edges, and nothing of the microphone arming, which every Shift does."""
    match move:
        case "start":
            return (OPENED,)
        case "stop":
            return (SENT,)
        case "drop":
            return (DROPPED,)
        case "arm" | "disarm":
            return ()


@cache
def sound(cue: Cue, sample_rate: int, channels: int) -> bytes:
    """The cue's tones one after another as 16-bit PCM, each followed by a tone's length of silence."""
    n = round(sample_rate * CUE_SECONDS)
    seconds = n / sample_rate
    t = np.arange(n) / sample_rate
    # A raised-cosine envelope: a tone that starts or stops at full level clicks.
    envelope = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    tones: list[np.ndarray] = []
    for start, end in cue.glides:
        # The phase of a linear glide from `start` to `end` Hz over the tone.
        phase = 2 * np.pi * (start * t + (end - start) * t**2 / (2 * seconds))
        tones += [CUE_LEVEL * envelope * np.sin(phase), np.zeros(n)]
    mono = (np.concatenate(tones) * 32767).astype(np.int16)
    return np.repeat(mono, channels).tobytes()
