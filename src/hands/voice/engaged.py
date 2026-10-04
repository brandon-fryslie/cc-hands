"""Engaged conversation: one hold of the talk key engages hands, and from then on the user's voice opens each turn and
end-of-turn detection closes it, until another hold disengages it.

Two models listen to the desk's microphone, heard through the echo canceller as every buffer is: Silero's voice
activity detector says where speech starts and stops, and Smart Turn says whether a stop is the end of the turn or a
pause inside it, from how the speech ended rather than from how long the silence is. A pause it judges a thought still
going holds the turn open; if the silence then runs on to Smart Turn's own `stop_secs`, the turn ends there.

The edge moves the gate with the held key's own moves, so the gate, Whisper, and the cues take an engaged turn as they
take a held one: speech starting arms the microphone, speech confirmed opens the turn, a start that was only a noise
disarms it, and the end of the turn sends it. Engaging has the desk listen between turns, so Whisper keeps the second
before each turn opens and the words said while the detector made sure of them are in it; disengaging stops that.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, ExitStack
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.metrics.metrics import TurnMetricsData

from hands.sessions.wide import WideEvent, annotate, count, unit
from hands.voice.hold import TURN_LIMIT_SECONDS, Hold, Idle, Instant, KeyEvent, Move, step as hold_step


@dataclass(frozen=True)
class SpeechStarting:
    """The detector heard what may be speech, and is making sure."""


@dataclass(frozen=True)
class SpeechStarted:
    """The detector is sure: the user is speaking, since `at`."""

    at: Instant


@dataclass(frozen=True)
class SpeechStopped:
    """The detector hears no speech: a start that came to nothing, or speech that has stopped."""


@dataclass(frozen=True)
class TurnEnded:
    """The turn is over: Smart Turn judged the speech complete, or the silence after it ran past `stop_secs`."""


@dataclass(frozen=True)
class TurnTooLong:
    """`TURN_LIMIT_SECONDS` have passed since the turn opened at `opened_at`."""

    opened_at: Instant


Heard = SpeechStarting | SpeechStarted | SpeechStopped | TurnEnded | TurnTooLong
Event = KeyEvent | Heard


@dataclass(frozen=True)
class Disengaged:
    """Not engaged: whatever the room says opens nothing."""


@dataclass(frozen=True)
class Listening:
    """Engaged, and nobody speaking."""


@dataclass(frozen=True)
class Arming:
    """Engaged, with what may be speech starting: the microphone is open and the turn is not."""


@dataclass(frozen=True)
class Talking:
    """Engaged, with the user's turn open since `since`."""

    since: Instant


Phase = Disengaged | Listening | Arming | Talking


@dataclass(frozen=True)
class Engagement:
    """The talk key's hold, which engages and disengages, and where the conversation is."""

    hold: Hold = Idle()
    phase: Phase = Disengaged()

    @property
    def engaged(self) -> bool:
        return not isinstance(self.phase, Disengaged)


# What the edge does: a move of the gate, or asking Smart Turn whether the speech that just stopped ends the turn.
Act = Move | Literal["judge"]


def step(engagement: Engagement, event: Event) -> tuple[Engagement, tuple[Act, ...]]:
    """The engagement after `event`, and what the edge does."""
    match event:
        case SpeechStarting() | SpeechStarted() | SpeechStopped() | TurnEnded() | TurnTooLong():
            phase, acts = _heard(engagement.phase, event)
            return replace(engagement, phase=phase), acts
        case _:
            # [LAW:one-source-of-truth] a hold that would open a held key's turn is the one that engages or
            # disengages: Right Shift typed as Shift, or tapped, does neither.
            hold, moves = hold_step(engagement.hold, event)
            phase, acts = _toggled(engagement.phase) if "start" in moves else (engagement.phase, ())
            return Engagement(hold, phase), acts


def _toggled(phase: Phase) -> tuple[Phase, tuple[Act, ...]]:
    match phase:
        case Disengaged():
            return Listening(), ("listen",)
        case Listening() | Arming() | Talking():
            # The gate sends a turn open as the desk stops listening: disengaging ends the conversation, not the last
            # thing said in it.
            return Disengaged(), ("deafen",)


def _heard(phase: Phase, heard: Heard) -> tuple[Phase, tuple[Act, ...]]:
    match phase, heard:
        case Listening(), SpeechStarting():
            return Arming(), ("arm",)
        case Listening(), SpeechStarted(at=at):
            return Talking(at), ("arm", "start")
        case Arming(), SpeechStarted(at=at):
            return Talking(at), ("start",)
        case Arming(), SpeechStopped():
            return Listening(), ("disarm",)
        case Talking(), SpeechStopped():
            return phase, ("judge",)
        case Talking(), TurnEnded():
            return Listening(), ("stop",)
        # [LAW:no-ambient-temporal-coupling] a limit counts only for the turn it was set for.
        case Talking(since=since), TurnTooLong(opened_at=opened_at) if opened_at == since:
            return Listening(), ("expire",)
        case _:
            return phase, ()


def released(engagement: Engagement) -> tuple[Move, ...]:
    """What the gate is left with when the edge stops, switched to another trigger: the desk stops listening, and a turn
    open here is thrown away, since the next turn opens the new way."""
    match engagement.phase:
        case Disengaged():
            return ()
        case Listening() | Arming():
            return ("deafen",)
        case Talking():
            return ("drop", "deafen")


class Ears(Protocol):
    """What the edge asks of the models that listen: Silero's detector and Smart Turn, in one."""

    async def detect(self, audio: bytes) -> VADState: ...
    def heard(self, audio: bytes, speech: bool) -> bool: ...
    async def judge(self) -> tuple[bool, TurnMetricsData | None]: ...
    def clear(self) -> None: ...


class Models:
    """Silero's voice activity detector and Smart Turn v3, run locally on the CPU, both shipped inside Pipecat."""

    def __init__(self, sample_rate: int) -> None:
        self._vad = SileroVADAnalyzer(sample_rate=sample_rate)
        self._vad.set_sample_rate(sample_rate)
        self._turn = LocalSmartTurnAnalyzerV3(sample_rate=sample_rate)
        self._turn.set_sample_rate(sample_rate)

    async def detect(self, audio: bytes) -> VADState:
        return await self._vad.analyze_audio(audio)

    def heard(self, audio: bytes, speech: bool) -> bool:
        """Give Smart Turn the audio as it is heard; True once the silence after speech has run past `stop_secs`."""
        return self._turn.append_audio(audio, speech) == EndOfTurnState.COMPLETE

    async def judge(self) -> tuple[bool, TurnMetricsData | None]:
        state, metrics = await self._turn.analyze_end_of_turn()
        return state == EndOfTurnState.COMPLETE, metrics if isinstance(metrics, TurnMetricsData) else None

    def clear(self) -> None:
        self._turn.clear()


# Each move the edge makes is counted on its engagement's event, by the move's name.
_COUNTED: tuple[Move, ...] = ("listen", "arm", "disarm", "start", "stop", "expire", "deafen")


async def drive_engaged(
    tapped: Callable[[Callable[[KeyEvent], None]], AbstractAsyncContextManager[None]],
    overheard: Callable[[], AbstractAsyncContextManager[AsyncIterator[bytes]]],
    ears: Ears,
    on_move: Callable[[Move], Awaitable[None]],
    emit: Callable[[WideEvent], None],
    clock: Callable[[], Instant] = time.monotonic,
) -> None:
    """Hand every move the edge makes to `on_move`, until cancelled.

    One task steps the engagement, in the order things were heard, so the speech, the key, and Smart Turn's verdicts
    never race each other.
    """
    loop = asyncio.get_running_loop()
    events: asyncio.Queue[Event | bytes] = asyncio.Queue()

    async def overhearing() -> None:
        async with overheard() as heard:
            async for audio in heard:
                events.put_nowait(audio)

    engagement = Engagement()
    # The silence the last buffer was heard in, as the detector said it.
    detected = VADState.QUIET
    # [LAW:nothing-unseen] one event per engagement, open from the hold that engages to the one that disengages.
    engaged = ExitStack()
    try:
        with engaged:
            async with tapped(events.put_nowait), asyncio.TaskGroup() as group:
                group.create_task(overhearing(), name="the engaged edge's microphone")
                while True:
                    match await events.get():
                        case bytes() as audio:
                            state = await ears.detect(audio)
                            # Smart Turn hears speech only where the detector is sure of it, as Pipecat feeds it.
                            silent_too_long = ears.heard(audio, state == VADState.SPEAKING)
                            heard = [*_changed(detected, state, clock), *([TurnEnded()] if silent_too_long else [])]
                            detected = state
                        case event:
                            heard = [event]
                    for each in heard:
                        was = engagement
                        engagement, acts = step(engagement, each)
                        if engagement.engaged and not was.engaged:
                            engaged.enter_context(unit("trigger.engaged", emit, counts=(*_COUNTED, "held_open")))
                        for act in acts:
                            match act:
                                case "judge":
                                    if await _judged(ears, emit):
                                        events.put_nowait(TurnEnded())
                                    else:
                                        count(held_open=1)
                                case move:
                                    count(**{move: 1})
                                    # [LAW:no-ambient-temporal-coupling] Smart Turn starts each turn with no speech
                                    # of the last one in it, as Pipecat clears it at each turn's edges.
                                    if move in ("arm", "stop"):
                                        ears.clear()
                                    await on_move(move)
                        match engagement.phase:
                            case Talking(since=since) if not isinstance(was.phase, Talking):
                                loop.call_at(since + TURN_LIMIT_SECONDS, events.put_nowait, TurnTooLong(since))
                            case _:
                                pass
                        if was.engaged and not engagement.engaged:
                            engaged.close()
    finally:
        for move in released(engagement):
            await on_move(move)


def _changed(was: VADState, now: VADState, clock: Callable[[], Instant]) -> tuple[Heard, ...]:
    """What the detector's state moving from `was` to `now` says of the speech."""
    match was, now:
        case VADState.QUIET, VADState.STARTING:
            return (SpeechStarting(),)
        case VADState.QUIET | VADState.STARTING, VADState.SPEAKING:
            return (SpeechStarted(clock()),)
        case VADState.STARTING | VADState.SPEAKING | VADState.STOPPING, VADState.QUIET:
            return (SpeechStopped(),)
        case _:
            return ()


async def _judged(ears: Ears, emit: Callable[[WideEvent], None]) -> bool:
    """Ask Smart Turn whether the speech that just stopped ends the turn: its own unit of work, with its verdict."""
    with unit("trigger.judged", emit):
        complete, metrics = await ears.judge()
        # [LAW:nothing-unseen] the verdict, and how sure of it the model was; no metrics is a segment with no audio in it.
        annotate(complete=complete, probability=None if metrics is None else metrics.probability, inference_ms=None if metrics is None else metrics.e2e_processing_time_ms)
        return complete
