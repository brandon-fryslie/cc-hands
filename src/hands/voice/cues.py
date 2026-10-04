"""Cues: earcons, the short tones that say what hands is doing without a word, each with the terminal line it shows.

The talk key's edges are cued as the key moves: the tone goes to the speaker ahead of anything else and is short and
soft enough to leave the microphone open, so a word said over it reaches Whisper whole. The rest are cues for silence:
hands has received a turn, and hands is acting. They are owed as it happens and played once hands is not speaking, so
none is ever heard over its voice.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Protocol

import numpy as np
from loguru import logger

from hands.sessions.audit import Cued, Record
from hands.voice.hold import TURN_LIMIT_SECONDS, Move

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
EXPIRED = Cue(f"turn: dropped, open {TURN_LIMIT_SECONDS:.0f}s", DROPPED.glides)
# Steady pitches, apart from the key's glides: a high pair rising says the words reached the model, a single mid tone
# says hands, or the session in focus, did something.
RECEIVED = Cue("turn: received", ((1320.0, 1320.0), (1760.0, 1760.0)))
WORKING = Cue("working", ((880.0, 880.0),))


def cues(move: Move) -> tuple[Cue, ...]:
    """What a move shows and plays: the turn's edges, and nothing of the microphone arming, which every Shift does."""
    match move:
        case "start":
            return (OPENED,)
        case "stop":
            return (SENT,)
        case "drop":
            return (DROPPED,)
        case "expire":
            return (EXPIRED,)
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


class Quiet(Protocol):
    """A speaker that says when hands is not speaking, and plays a cue at once."""

    @property
    def quiet(self) -> asyncio.Event: ...

    def cue(self, cue: Cue) -> None: ...


class QuietCues:
    """The cues owed to silence, oldest first: anything may owe one, before or after the speaker exists, and
    `keep_cueing` is the one player of them."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        # [LAW:no-shared-mutable-globals] written only by `owe`, read only by `owed`.
        self._owed: asyncio.Queue[tuple[Cue, float]] = asyncio.Queue()
        self._clock = clock

    def owe(self, cue: Cue) -> None:
        self._owed.put_nowait((cue, self._clock()))

    async def owed(self) -> list[tuple[Cue, float]]:
        """Every cue owed now, waiting for one if none is."""
        return [await self._owed.get(), *self.drained()]

    def drained(self) -> list[tuple[Cue, float]]:
        """Every cue owed now, none if none is."""
        owed: list[tuple[Cue, float]] = []
        while not self._owed.empty():
            owed.append(self._owed.get_nowait())
        return owed

    def now(self) -> float:
        return self._clock()


async def keep_cueing(cues: QuietCues, speaker: Quiet, record: Record) -> None:
    """Play each cue owed once hands is not speaking, until cancelled. What is owed while it waits is played with it,
    each cue once however many times it was owed, so a burst of acts is one tone and not a rattle."""
    while True:
        owed = await cues.owed()
        await speaker.quiet.wait()
        owed += cues.drained()
        played = cues.now()
        folded: dict[Cue, list[float]] = {}
        for cue, at in owed:
            folded.setdefault(cue, []).append(at)
        for cue, ats in folded.items():
            logger.info(cue.line)
            speaker.cue(cue)
            # [LAW:nothing-unseen] which cue played, how many it stood for, and how long speech held it.
            record(Cued(cue.line, len(ats), played - ats[0]))
