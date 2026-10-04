"""Cues: earcons, the short tones that say what hands is doing without a word, each with the terminal line it shows.

The talk key's edges are cued as the key moves: the tone goes to the speaker ahead of anything else and is short and
soft enough to leave the microphone open, so a word said over it reaches Whisper whole. The rest are cues for silence:
hands has received a turn, and hands is acting. They are owed as it happens and played once neither hands nor the user
is speaking, so none is ever heard over a voice.
"""

import asyncio
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Protocol

import numpy as np
from loguru import logger

from hands.sessions.audit import Cued, Played, Record
from hands.voice.hold import TURN_LIMIT_SECONDS, Move

CUE_SECONDS = 0.06
# About -16 dBFS against replies that peak near full scale: heard over the room, never louder than the voice.
CUE_LEVEL = 0.16


@dataclass(frozen=True)
class Cue:
    """One earcon: the line the terminal shows, and the pitches its tone glides between, one tone per pair."""

    line: str
    glides: tuple[tuple[float, float], ...]


# Rising opens, falling sends, and a low pair drops: three shapes told apart without looking.
OPENED = Cue("turn: started", ((660.0, 990.0),))
SENT = Cue("turn: ended", ((990.0, 660.0),))
DROPPED = Cue("turn: dropped", ((330.0, 330.0), (330.0, 330.0)))
EXPIRED = Cue(f"turn: dropped, open {TURN_LIMIT_SECONDS:.0f}s", DROPPED.glides)
# The desk starting and stopping listening for every turn, under engaged conversation or the wake word: two rising tones
# as it starts, two falling as it stops. Two tones, so neither is taken for a turn opening or ending.
LISTENING = Cue("desk: listening", ((660.0, 990.0), (660.0, 990.0)))
DEAF = Cue("desk: not listening", ((990.0, 660.0), (990.0, 660.0)))


@dataclass(frozen=True)
class QuietCue:
    """A cue for silence, and how long after it plays it waits before playing again: what is owed meanwhile plays as
    one when that time is up."""

    cue: Cue
    spacing: float


# Above the key's pitches: a high pair stepping up says the words reached the model, which is heard once per turn; a
# single mid tone says hands, or the session in focus, did something, and is heard at most every two seconds however
# much it does, since a tone for every act is a rattle he stops hearing.
RECEIVED = QuietCue(Cue("turn: received", ((1320.0, 1320.0), (1760.0, 1760.0))), 0.0)
WORKING = QuietCue(Cue("working", ((880.0, 880.0),)), 2.0)


def cues(move: Move) -> tuple[Cue, ...]:
    """What a move shows and plays: the turn's edges, and the desk starting and stopping listening for every turn; and
    nothing of the microphone arming, which every Shift does."""
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
        case "listen":
            return (LISTENING,)
        case "deafen":
            return (DEAF,)


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
    """A speaker that says when nobody is speaking, and plays a cue at once, saying where it went."""

    @property
    def quiet(self) -> asyncio.Event: ...

    def cue(self, cue: Cue) -> Played: ...


class QuietCues:
    """The cues owed to silence, and their one player: anything may owe one, before or after the speaker exists."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        # [LAW:no-shared-mutable-globals] written only by `owe`, read only by `keep_playing`.
        self._owed: asyncio.Queue[tuple[QuietCue, float]] = asyncio.Queue()
        self._clock = clock

    def owe(self, cue: QuietCue) -> None:
        self._owed.put_nowait((cue, self._clock()))

    async def keep_playing(self, speaker: Quiet, record: Record) -> None:
        """Play each cue owed once nobody is speaking and its spacing is up, until cancelled. What is owed of one cue
        while it waits is played with it, once, so a burst of acts is one tone."""
        held: dict[QuietCue, list[float]] = {}
        last: dict[QuietCue, float] = {}

        def due(cue: QuietCue) -> float:
            return last.get(cue, -math.inf) + cue.spacing

        while True:
            soonest = min((due(cue) for cue in held), default=math.inf)
            self._hold(held, await self._owed_within(soonest - self._clock()))
            await speaker.quiet.wait()
            self._hold(held, self._drained())
            now = self._clock()
            for cue in [cue for cue in held if due(cue) <= now]:
                ats = held.pop(cue)
                last[cue] = now
                logger.info(cue.cue.line)
                played = speaker.cue(cue.cue)
                # [LAW:nothing-unseen] which cue played and where, how many it stood for, and how long it was held.
                record(Cued(cue.cue.line, len(ats), now - ats[0], played))

    async def _owed_within(self, seconds: float) -> list[tuple[QuietCue, float]]:
        """Every cue owed now, waiting up to `seconds` for one if none is."""
        if self._owed.empty() and seconds > 0:
            try:
                return [await asyncio.wait_for(self._owed.get(), None if math.isinf(seconds) else seconds), *self._drained()]
            except TimeoutError:
                return []
        return self._drained()

    def _drained(self) -> list[tuple[QuietCue, float]]:
        owed: list[tuple[QuietCue, float]] = []
        while not self._owed.empty():
            owed.append(self._owed.get_nowait())
        return owed

    @staticmethod
    def _hold(held: dict[QuietCue, list[float]], owed: list[tuple[QuietCue, float]]) -> None:
        for cue, at in owed:
            held.setdefault(cue, []).append(at)
