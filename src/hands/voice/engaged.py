"""Engaged conversation: one hold of the talk key engages hands, and from then on the user's voice opens each turn and
end-of-turn detection closes it, until another hold disengages it.

Two models listen to the desk's microphone, heard through the echo canceller as every buffer is: Silero's voice
activity detector says where speech starts and stops, a silence shorter than `STOP_SECS` being inside the speech, and
at each stop Smart Turn says whether it is the end of the turn or a pause inside it, from how the speech ended. A pause
it judges a thought still going holds the turn open; if the silence then runs on to Smart Turn's own `stop_secs`, the
turn ends there.

The edge moves the gate with the held key's own moves, so the gate, Whisper, and the cues take an engaged turn as they
take a held one: speech starting arms the microphone, speech confirmed opens the turn, a start that was only a noise
disarms it, and the end of the turn sends it. Engaging has the desk listen between turns, so Whisper keeps the second
before each turn opens and the words said while the detector made sure of them are in it; disengaging stops that.

The driver is shared with the wake word (`hands.voice.wake`), the other edge that opens turns without a hand: each is a
`Conversation`, its own step over the same engagement, and the driver hears for both.
"""

import asyncio
import time
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, ExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from typing import Literal, Protocol, get_args

from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState
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
    """The turn is over, `by` Smart Turn judging the speech complete, or by the silence after it running past
    `stop_secs`."""

    by: Literal["verdict", "silence"]


@dataclass(frozen=True)
class Woken:
    """The wake word was heard, at `at`."""

    at: Instant


@dataclass(frozen=True)
class TurnTooLong:
    """`TURN_LIMIT_SECONDS` have passed since the turn opened at `opened_at`."""

    opened_at: Instant


@dataclass(frozen=True)
class Begun:
    """The edge has begun: its trigger was switched to."""


Heard = SpeechStarting | SpeechStarted | SpeechStopped | Woken | TurnEnded | TurnTooLong
Event = KeyEvent | Heard | Begun


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


@dataclass(frozen=True)
class Woke:
    """The wake word opened the user's turn, since `since`, and what they ask has yet to start: a stop now is the end of
    the wake word, not of the turn (`hands.voice.wake`)."""

    since: Instant


Phase = Disengaged | Listening | Arming | Woke | Talking


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
        case Begun():
            return engagement, ()
        case SpeechStarting() | SpeechStarted() | SpeechStopped() | Woken() | TurnEnded() | TurnTooLong():
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
        case Listening():
            return Disengaged(), ("deafen",)
        case Arming():
            return Disengaged(), ("disarm", "deafen")
        case Woke() | Talking():
            # Disengaging ends the conversation, not the last thing said in it: the turn open is sent, and heard sent.
            return Disengaged(), ("stop", "deafen")


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
        case Talking(), SpeechStopped() | TurnEnded() | TurnTooLong():
            return in_turn(phase, heard)
        case _:
            return phase, ()


def in_turn(turn: Woke | Talking, heard: SpeechStopped | TurnEnded | TurnTooLong) -> tuple[Phase, tuple[Act, ...]]:
    """How a turn the voice opened ends, whatever opened it: Smart Turn's verdict on a stop, the silence after it, or the
    limit."""
    match heard:
        case SpeechStopped():
            return turn, ("judge",)
        case TurnEnded():
            return Listening(), ("stop",)
        # [LAW:no-ambient-temporal-coupling] a limit counts only for the turn it was set for.
        case TurnTooLong(opened_at=opened_at) if opened_at == turn.since:
            return Listening(), ("expire",)
        case TurnTooLong():
            return turn, ()


def released(engagement: Engagement) -> tuple[Move, ...]:
    """What the gate is left with when the edge stops, switched to another trigger: the desk stops listening, and a turn
    open here is thrown away, since the next turn opens the new way."""
    match engagement.phase:
        case Disengaged():
            return ()
        case Listening():
            return ("deafen",)
        case Arming():
            return ("disarm", "deafen")
        case Woke() | Talking():
            return ("drop", "deafen")


class Ears(Protocol):
    """What the edge asks of the models that listen: Silero's detector and Smart Turn, in one."""

    async def detect(self, audio: bytes) -> VADState: ...
    def heard(self, audio: bytes, speech: bool) -> bool: ...
    async def judge(self) -> tuple[bool, TurnMetricsData | None]: ...
    def clear(self) -> None: ...
    def afresh(self) -> None: ...


# The silence that ends speech: a pause between words is shorter, so it is inside the speech. At Pipecat's 0.2 s, tuned
# for Smart Turn, Smart Turn judged every such pause, and a phrase said before one read to it as complete, so each
# phrase was a turn (the real-model test in tests/test_engaged.py fails at 0.2). The price is that every turn ends this
# long after the speech does, before Smart Turn is asked.
STOP_SECS = 0.8


class Models:
    """Silero's voice activity detector and Smart Turn v3, run locally on the CPU, both shipped inside Pipecat."""

    def __init__(self, sample_rate: int) -> None:
        self._vad = SileroVADAnalyzer(sample_rate=sample_rate, params=VADParams(stop_secs=STOP_SECS))
        self._vad.set_sample_rate(sample_rate)
        self._turn = LocalSmartTurnAnalyzerV3(sample_rate=sample_rate)
        self._turn.set_sample_rate(sample_rate)
        # As Pipecat syncs it at each start: the speech Silero takes to be sure of is kept before the segment judged.
        self._turn.update_vad_start_secs(self._vad.params.start_secs)

    @property
    def vad_stop_secs(self) -> float:
        return self._vad.params.stop_secs

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

    def afresh(self) -> None:
        """Hear on as if nothing came before: Silero is quiet until sure of speech again, and Smart Turn holds none."""
        # Setting the detector's parameters puts it back to quiet, with no speech counted toward starting or stopping.
        self._vad.set_params(self._vad.params)
        self._turn.clear()

    async def close(self) -> None:
        """Shut down the thread each model runs on."""
        await self._vad.cleanup()
        await self._turn.cleanup()


@asynccontextmanager
async def loaded(sample_rate: int, emit: Callable[[WideEvent], None]) -> AsyncGenerator[Models]:
    """Both models, loaded off the loop as their own unit of work, for as long as the edge listens with them."""
    loading = asyncio.create_task(asyncio.to_thread(Models, sample_rate))
    try:
        with unit("trigger.loaded", emit):
            # [LAW:nothing-unseen] the pause the detector ends speech at, as it was loaded with it.
            annotate(vad_stop_secs=(await asyncio.shield(loading)).vad_stop_secs)
        yield loading.result()
    finally:
        # A switch away while they load waits the load out, so what it built is let go of too.
        await (await loading).close()


# [LAW:one-source-of-truth] each move the edge makes is counted on its engagement's event, by the move's name.
_COUNTED: tuple[str, ...] = tuple(name for literal in get_args(Move) for name in get_args(literal))


@dataclass(frozen=True)
class Conversation:
    """An edge that opens turns without a hand on the key and closes them by end-of-turn detection: how it steps, what
    it leaves the gate with when switched away from, and the event each of its engagements is, with what it counts."""

    event: str
    step: Callable[[Engagement, Event], tuple[Engagement, tuple[Act, ...]]]
    released: Callable[[Engagement], tuple[Move, ...]]
    counts: tuple[str, ...] = ()


ENGAGED = Conversation("trigger.engaged", step, released)


@asynccontextmanager
async def untapped(into: Callable[[KeyEvent], None]) -> AsyncGenerator[None]:
    """No talk key: for an edge the key has no part in."""
    yield


async def unwoken(audio: bytes) -> bool:
    """No wake word: for an edge no word opens."""
    return False


async def drive_engaged(
    tapped: Callable[[Callable[[KeyEvent], None]], AbstractAsyncContextManager[None]],
    overheard: Callable[[], AbstractAsyncContextManager[AsyncIterator[bytes]]],
    ears: Ears,
    on_move: Callable[[Move], Awaitable[None]],
    emit: Callable[[WideEvent], None],
    clock: Callable[[], Instant] = time.monotonic,
) -> None:
    """Engaged conversation: the talk key engages and disengages, and the voice opens each turn."""
    await drive(ENGAGED, tapped, overheard, ears, unwoken, on_move, emit, clock)


async def drive(
    conversation: Conversation,
    tapped: Callable[[Callable[[KeyEvent], None]], AbstractAsyncContextManager[None]],
    overheard: Callable[[], AbstractAsyncContextManager[AsyncIterator[bytes]]],
    ears: Ears,
    woken: Callable[[bytes], Awaitable[bool]],
    on_move: Callable[[Move], Awaitable[None]],
    emit: Callable[[WideEvent], None],
    clock: Callable[[], Instant] = time.monotonic,
) -> None:
    """Hand every move `conversation` makes to `on_move`, until cancelled.

    One task steps the engagement, in the order things were heard, so the speech, the key, the wake word, and Smart
    Turn's verdicts never race each other: a verdict is stepped as it is given, before any audio heard while the model
    thought, so it ends the turn it judged and no other.
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
    # [LAW:nothing-unseen] one event per engagement, open from what engages to what disengages.
    engaged = ExitStack()
    # The acts the last step made that the gate has yet to be handed: a switch away mid-step hands them on.
    owed = deque[Act]()
    with engaged:
        try:
            events.put_nowait(Begun())
            async with tapped(events.put_nowait), asyncio.TaskGroup() as group:
                group.create_task(overhearing(), name="the desk's microphone, overheard")
                while True:
                    match await events.get():
                        case bytes() as audio:
                            state = await ears.detect(audio)
                            # Smart Turn hears speech only where the detector is sure of it, as Pipecat feeds it, and
                            # only while engaged, so the room it hears disengaged is trimmed as silence is.
                            silent_too_long = ears.heard(audio, engagement.engaged and state == VADState.SPEAKING)
                            heard: deque[Event] = deque(
                                [*_changed(detected, state, clock), *([Woken(clock())] if await woken(audio) else []), *([TurnEnded("silence")] if silent_too_long else [])]
                            )
                            detected = state
                        case event:
                            heard = deque[Event]([event])
                    while heard:
                        each = heard.popleft()
                        was = engagement
                        engagement, acts = conversation.step(engagement, each)
                        if engagement.engaged and not was.engaged:
                            engaged.enter_context(unit(conversation.event, emit, counts=(*_COUNTED, "held_open", "ended_on_silence", *conversation.counts)))
                        match each, acts:
                            case TurnEnded(by="silence"), ("stop",):
                                count(ended_on_silence=1)
                            # [LAW:types-are-the-program] the wake word's own speech is no part of what is asked:
                            # heard afresh from where it opened the turn, what is asked starts as any speech does, and
                            # Smart Turn judges it alone.
                            case Woken(), ("arm", "start"):
                                ears.afresh()
                                detected = VADState.QUIET
                            case _:
                                pass
                        owed.extend(acts)
                        while owed:
                            match owed.popleft():
                                case "judge":
                                    if await _judged(ears, emit):
                                        heard.appendleft(TurnEnded("verdict"))
                                    else:
                                        count(held_open=1)
                                case move:
                                    count(**{move: 1})
                                    # [LAW:no-ambient-temporal-coupling] Smart Turn starts each turn with no speech
                                    # of the last one in it, as Pipecat clears it as each turn ends.
                                    if move in ("stop", "drop", "expire"):
                                        ears.clear()
                                    await on_move(move)
                        match engagement.phase, was.phase:
                            case Woke(since=since) | Talking(since=since), Woke(since=opened) | Talking(since=opened) if since == opened:
                                pass
                            case Woke(since=since) | Talking(since=since), _:
                                loop.call_at(since + TURN_LIMIT_SECONDS, events.put_nowait, TurnTooLong(since))
                            case _:
                                pass
                        if was.engaged and not engagement.engaged:
                            engaged.close()
        finally:
            # Inside the engagement's event, so what a switch away leaves the gate with is counted on it: what the last
            # step owed the gate, and then what is left open.
            left: list[Move] = [act for act in owed if act != "judge"]
            for move in (*left, *conversation.released(engagement)):
                count(**{move: 1})
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
