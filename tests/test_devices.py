"""Following the system's default audio devices, with the reopen and the saying replaced."""

import asyncio
from contextlib import suppress
from typing import Literal

import pytest

from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.voice.coreaudio import DefaultDevices
from hands.voice.devices import follow
from hands.voice.microphone import Devices
from hands.voice.system import AudioMoved, SystemFact, system_text

HEADSET = Devices(input="headset", output="headset")
BUILT_IN = Devices(input="MacBook Pro Microphone", output="MacBook Pro Speakers")
PATIENCE_SECS = 2.0


class Notices(asyncio.Event):
    """The default-device notices, which also tell when the follower has settled: waiting on them, none pending."""

    def __init__(self) -> None:
        super().__init__()
        self.settled = asyncio.Event()

    async def wait(self) -> Literal[True]:
        if not self.is_set():
            self.settled.set()
        return await super().wait()

    def set(self) -> None:
        self.settled.clear()
        super().set()


class Follower:
    def __init__(self, *opened: Devices) -> None:
        self.defaults = DefaultDevices(input=1, output=1)
        self.opened_on = self.defaults
        self.changes = Notices()
        self.opened = list(opened)
        self.on = HEADSET
        self.reopens = 0
        self.failing: Exception | None = None
        self.said: list[SystemFact] = []
        self.recorded: list[Entry] = []
        self.reopening = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.task = asyncio.create_task(follow(self.changes, self.current, lambda: self.opened_on, lambda: self.on, self.reopen, self.say, self.recorded.append))

    async def reopen(self) -> Devices:
        self.reopens += 1
        # The transport reads the defaults as PortAudio lists them, which is before it has finished reopening.
        self.opened_on = self.defaults
        self.reopening.set()
        await self.release.wait()
        self.reopening.clear()
        if self.failing is not None:
            raise self.failing
        self.on = self.opened.pop(0)
        return self.on

    async def current(self) -> DefaultDevices:
        return self.defaults

    async def say(self, fact: SystemFact) -> None:
        self.said.append(fact)

    def notice(self, defaults: DefaultDevices) -> None:
        self.defaults = defaults
        self.changes.set()

    async def until(self, event: asyncio.Event) -> None:
        """Return once `event` is set, however slowly the loop runs; raise what stopped the follower if it stops first."""
        waiting = asyncio.ensure_future(event.wait())
        try:
            async with asyncio.timeout(PATIENCE_SECS):
                await asyncio.wait({waiting, self.task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiting.cancel()
        if self.task.done():
            self.task.result()
            raise AssertionError("the follower returned, which it never does until cancelled")

    async def settled(self) -> None:
        await self.until(self.changes.settled)

    @property
    def moves(self) -> list[WideEvent]:
        return [entry for entry in self.recorded if isinstance(entry, WideEvent) and entry.event == "devices.moved"]

    async def stop(self) -> None:
        self.task.cancel()
        with suppress(asyncio.CancelledError):
            await self.task


async def test_a_change_of_defaults_reopens_the_transport_and_says_where_the_audio_went() -> None:
    follower = Follower(BUILT_IN)
    follower.release.clear()
    await follower.settled()
    # An unplugged headset changes the default input and output together, as two notices, one while the reopen runs.
    follower.notice(DefaultDevices(input=2, output=2))
    await follower.until(follower.reopening)
    follower.notice(DefaultDevices(input=2, output=2))
    follower.release.set()
    await follower.settled()
    await follower.stop()
    assert follower.reopens == 1
    assert follower.said == [AudioMoved(BUILT_IN)]
    [move] = follower.moves
    assert (move.outcome, move.facts) == ("ok", {"before": HEADSET, "defaults": DefaultDevices(input=2, output=2), "after": BUILT_IN})


async def test_a_change_that_comes_while_the_transport_reopens_is_followed_too() -> None:
    follower = Follower(HEADSET, BUILT_IN)
    follower.on = BUILT_IN
    follower.release.clear()
    await follower.settled()
    follower.notice(DefaultDevices(input=2, output=2))  # plugged in
    await follower.until(follower.reopening)
    follower.notice(DefaultDevices(input=1, output=1))  # and pulled out again before the first reopen finished
    follower.release.set()
    await follower.settled()
    await follower.stop()
    assert follower.said == [AudioMoved(HEADSET), AudioMoved(BUILT_IN)]
    assert [(move.facts["before"], move.facts["after"]) for move in follower.moves] == [(BUILT_IN, HEADSET), (HEADSET, BUILT_IN)]


def test_where_the_audio_went_is_said_by_the_devices_names() -> None:
    assert system_text(AudioMoved(BUILT_IN)) == "Audio moved: listening on MacBook Pro Microphone, speaking on MacBook Pro Speakers."


async def test_a_notice_that_leaves_the_defaults_where_they_were_reopens_nothing() -> None:
    follower = Follower()
    await follower.settled()
    follower.notice(follower.defaults)
    await follower.settled()
    await follower.stop()
    assert (follower.reopens, follower.said, follower.recorded) == (0, [], [])


async def test_a_headset_unplugged_before_the_pipeline_started_is_followed_once_it_has() -> None:
    follower = Follower(BUILT_IN)
    follower.defaults = DefaultDevices(input=2, output=2)  # moved while the models loaded, after PortAudio listed the old ones
    await follower.settled()
    await follower.stop()
    assert follower.said == [AudioMoved(BUILT_IN)]


async def test_a_follower_stopped_during_a_reopen_waits_for_it_to_finish() -> None:
    follower = Follower(BUILT_IN)
    follower.release.clear()
    await follower.settled()
    follower.notice(DefaultDevices(input=2, output=2))
    await follower.until(follower.reopening)
    follower.task.cancel()
    await asyncio.sleep(0.01)
    assert not follower.task.done()  # the reopen holds the streams, so the stop waits for it
    follower.release.set()
    await asyncio.wait({follower.task})
    assert follower.task.cancelled()
    assert follower.opened == []  # the reopen ran to its end
    assert follower.said == []  # and nothing was said by a follower told to stop
    [move] = follower.moves
    assert (move.outcome, "after" in move.facts) == ("cancelled", False)


async def test_a_reopen_that_fails_stops_the_follower_and_its_move_is_failed() -> None:
    follower = Follower(BUILT_IN)
    follower.failing = OSError("no default output device")
    await follower.settled()
    follower.notice(DefaultDevices(input=2, output=2))
    await asyncio.wait({follower.task})
    with pytest.raises(OSError):
        follower.task.result()
    [move] = follower.moves
    assert (move.outcome, move.error, move.facts) == ("failed", "OSError: no default output device", {"before": HEADSET, "defaults": DefaultDevices(input=2, output=2)})
    assert follower.said == []
