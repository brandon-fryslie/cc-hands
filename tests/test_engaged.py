"""Engaged conversation: one hold engages, the user's voice opens each turn, end-of-turn detection closes it, and
another hold disengages."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager

import pytest
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.metrics.metrics import TurnMetricsData

from hands.sessions.wide import WideEvent
from hands.voice.engaged import (
    Act,
    Engagement,
    Event,
    SpeechStarted,
    SpeechStarting,
    SpeechStopped,
    TurnEnded,
    TurnTooLong,
    drive_engaged,
    released,
    step,
)
from hands.voice.hold import KeyEvent, Move, Pressed, Released, Ripe, Typed

# Right Shift held alone past the hold, then let go: what engages, and what disengages.
HOLD: tuple[KeyEvent, ...] = (Pressed(1.0), Ripe(1.0), Released())


def acts(events: Sequence[Event]) -> tuple[Engagement, list[Act]]:
    engagement = Engagement()
    made: list[Act] = []
    for event in events:
        engagement, now = step(engagement, event)
        made.extend(now)
    return engagement, made


@pytest.mark.parametrize(
    ("events", "made"),
    [
        # Disengaged, the room opens nothing.
        ([SpeechStarting(), SpeechStarted(2.0), SpeechStopped(), TurnEnded()], []),
        # Engaged: speech arms, is confirmed and opens the turn, stops and is judged, and the verdict sends it.
        ([*HOLD, SpeechStarting(), SpeechStarted(2.0), SpeechStopped(), TurnEnded()], ["listen", "arm", "start", "judge", "stop"]),
        # A pause judged a thought still going holds the turn open; speech goes on in the same turn.
        ([*HOLD, SpeechStarting(), SpeechStarted(2.0), SpeechStopped(), SpeechStarting(), SpeechStarted(3.0), SpeechStopped(), TurnEnded()], ["listen", "arm", "start", "judge", "judge", "stop"]),
        # A noise that never became speech disarms, and opens no turn.
        ([*HOLD, SpeechStarting(), SpeechStopped()], ["listen", "arm", "disarm"]),
        # Engaged while already speaking: the turn opens at once.
        ([*HOLD, SpeechStarted(2.0), SpeechStopped(), TurnEnded()], ["listen", "arm", "start", "judge", "stop"]),
        # Turns follow one another with no key between them.
        ([*HOLD, SpeechStarted(2.0), TurnEnded(), SpeechStarted(4.0), TurnEnded(), SpeechStarted(6.0), TurnEnded()], ["listen", *["arm", "start", "stop"] * 3]),
        # A turn open past the limit is thrown away; a limit set for an earlier turn ends nothing.
        ([*HOLD, SpeechStarted(2.0), TurnTooLong(2.0)], ["listen", "arm", "start", "expire"]),
        ([*HOLD, SpeechStarted(2.0), TurnEnded(), SpeechStarted(4.0), TurnTooLong(2.0)], ["listen", "arm", "start", "stop", "arm", "start"]),
        # Disengaging stops the desk listening, which sends a turn open; after it, the room opens nothing.
        ([*HOLD, SpeechStarted(2.0), Pressed(5.0), Ripe(5.0), SpeechStarting(), SpeechStarted(6.0)], ["listen", "arm", "start", "deafen"]),
        ([*HOLD, SpeechStarting(), Pressed(5.0), Ripe(5.0)], ["listen", "arm", "deafen"]),
        # Right Shift typed as Shift, or tapped, neither engages nor disengages.
        ([Pressed(1.0), Typed(), Ripe(1.0), Released(), SpeechStarted(2.0)], []),
        ([Pressed(1.0), Released(), Ripe(1.0), SpeechStarted(2.0)], []),
        ([*HOLD, Pressed(5.0), Typed(), Ripe(5.0), Released(), SpeechStarted(6.0)], ["listen", "arm", "start"]),
    ],
)
def test_what_the_edge_does(events: list[Event], made: list[Act]) -> None:
    # "judge" is the edge asking Smart Turn, which the driver answers with TurnEnded or nothing.
    assert acts(events)[1] == made


def test_a_hold_engages_and_another_disengages() -> None:
    assert acts(HOLD)[0].engaged
    assert not acts([*HOLD, Pressed(5.0), Ripe(5.0), Released()])[0].engaged


@pytest.mark.parametrize(
    ("events", "left"),
    [
        ([], ()),
        ([*HOLD], ("deafen",)),
        ([*HOLD, SpeechStarting()], ("deafen",)),
        ([*HOLD, SpeechStarted(2.0)], ("drop", "deafen")),
    ],
)
def test_switching_away_throws_away_a_turn_open_here(events: list[Event], left: tuple[Move, ...]) -> None:
    assert released(acts(events)[0]) == left


class ScriptedEars:
    """The models, as a script: each buffer of audio names the detector's state, and the verdicts are given in turn."""

    def __init__(self, verdicts: Sequence[bool]) -> None:
        self._verdicts = list(verdicts)
        self.cleared = 0

    async def detect(self, audio: bytes) -> VADState:
        return {b"q": VADState.QUIET, b"s": VADState.STARTING, b"S": VADState.SPEAKING, b"x": VADState.QUIET}[audio]

    def heard(self, audio: bytes, speech: bool) -> bool:
        # b"x" is quiet run past Smart Turn's stop_secs.
        return audio == b"x"

    async def judge(self) -> tuple[bool, TurnMetricsData | None]:
        complete = self._verdicts.pop(0)
        return complete, TurnMetricsData(processor="test", is_complete=complete, probability=0.9 if complete else 0.1, e2e_processing_time_ms=40.0)

    def clear(self) -> None:
        self.cleared += 1


class Rig:
    """The engaged edge driven with a fake talk key and a fake microphone."""

    def __init__(self, verdicts: Sequence[bool]) -> None:
        self.ears = ScriptedEars(verdicts)
        self.keys: list[Callable[[KeyEvent], None]] = []
        self.audio: asyncio.Queue[bytes] = asyncio.Queue()
        self.made: list[Move] = []
        self.events: list[WideEvent] = []

    @asynccontextmanager
    async def tapped(self, into: Callable[[KeyEvent], None]) -> AsyncGenerator[None]:
        self.keys.append(into)
        yield

    @asynccontextmanager
    async def overheard(self) -> AsyncGenerator[AsyncIterator[bytes]]:
        async def buffers() -> AsyncIterator[bytes]:
            while True:
                yield await self.audio.get()

        yield buffers()

    async def on_move(self, move: Move) -> None:
        self.made.append(move)

    def start(self) -> asyncio.Task[None]:
        return asyncio.create_task(drive_engaged(self.tapped, self.overheard, self.ears, self.on_move, self.events.append))

    async def press(self) -> None:
        while not self.keys:
            await asyncio.sleep(0)
        for event in HOLD:
            self.keys[0](event)

    async def hear(self, *audio: bytes) -> None:
        for each in audio:
            self.audio.put_nowait(each)

    async def settle(self, made: int) -> None:
        async with asyncio.timeout(1.0):
            while len(self.made) < made:
                await asyncio.sleep(0.001)
        for _ in range(20):
            await asyncio.sleep(0)


async def stopped(driving: asyncio.Task[None]) -> None:
    driving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await driving


async def test_three_turns_with_no_key_between_them_then_a_disengage() -> None:
    rig = Rig(verdicts=[False, True, True, False])
    driving = rig.start()
    await rig.press()
    # A pause mid-thought held open, then the thought finished; a second turn; a third held open by its verdict, then
    # ended by the silence after it running on.
    await rig.hear(b"s", b"S", b"q", b"s", b"S", b"q")
    await rig.settle(4)
    await rig.hear(b"s", b"S", b"q")
    await rig.settle(7)
    await rig.hear(b"s", b"S", b"q", b"x")
    await rig.settle(10)
    await rig.press()
    await rig.hear(b"s", b"S", b"q")
    for _ in range(20):
        await asyncio.sleep(0)
    await stopped(driving)
    assert rig.made == ["listen", *["arm", "start", "stop"] * 3, "deafen"]
    engagement = [event for event in rig.events if event.event == "trigger.engaged"]
    assert len(engagement) == 1
    assert dict(engagement[0].counts) == {"listen": 1, "arm": 3, "disarm": 0, "start": 3, "stop": 3, "expire": 0, "deafen": 1, "held_open": 2}
    judged = [event for event in rig.events if event.event == "trigger.judged"]
    assert [event.facts["complete"] for event in judged] == [False, True, True, False]
    assert all(event.parent_id == engagement[0].span_id for event in judged)


async def test_a_switch_away_mid_turn_drops_it_and_ends_the_engagement_cancelled() -> None:
    rig = Rig(verdicts=[])
    driving = rig.start()
    await rig.press()
    await rig.hear(b"s", b"S")
    await rig.settle(3)
    await stopped(driving)
    assert rig.made == ["listen", "arm", "start", "drop", "deafen"]
    assert [(event.event, event.outcome) for event in rig.events] == [("trigger.engaged", "cancelled")]
