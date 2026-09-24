"""Following the system's default audio devices, with the reopen and the saying replaced."""

import asyncio

from hands.voice.coreaudio import DefaultDevices
from hands.voice.devices import follow
from hands.voice.microphone import Devices
from hands.voice.system import AudioMoved, SystemFact, system_text

HEADSET = Devices(input="headset", output="headset")
BUILT_IN = Devices(input="MacBook Pro Microphone", output="MacBook Pro Speakers")


class Follower:
    def __init__(self, *opened: Devices) -> None:
        self.defaults = DefaultDevices(input=1, output=1)
        self.opened_on = self.defaults
        self.changes = asyncio.Event()
        self.opened = list(opened)
        self.reopens = 0
        self.said: list[SystemFact] = []
        self.release = asyncio.Event()
        self.release.set()

    async def reopen(self) -> Devices:
        self.reopens += 1
        # The transport reads the defaults as PortAudio lists them, which is before it has finished reopening.
        self.opened_on = self.defaults
        await self.release.wait()
        return self.opened.pop(0)

    def follow(self) -> "asyncio.Task[None]":
        return asyncio.create_task(follow(self.changes, lambda: self.defaults, lambda: self.opened_on, self.reopen, self.say))

    async def say(self, fact: SystemFact) -> None:
        self.said.append(fact)


async def test_a_change_of_defaults_reopens_the_transport_and_says_where_the_audio_went() -> None:
    follower = Follower(BUILT_IN)
    following = follower.follow()
    await asyncio.sleep(0.01)
    # An unplugged headset changes the default input and output together, as two notices, one after the reopen began.
    follower.defaults = DefaultDevices(input=2, output=2)
    follower.changes.set()
    await asyncio.sleep(0)
    follower.changes.set()
    await asyncio.sleep(0.01)
    following.cancel()
    assert follower.reopens == 1
    assert follower.said == [AudioMoved(BUILT_IN)]


async def test_a_change_that_comes_while_the_transport_reopens_is_followed_too() -> None:
    follower = Follower(HEADSET, BUILT_IN)
    follower.release.clear()
    following = follower.follow()
    await asyncio.sleep(0.01)
    follower.defaults = DefaultDevices(input=2, output=2)  # plugged in
    follower.changes.set()
    await asyncio.sleep(0.01)
    follower.defaults = DefaultDevices(input=1, output=1)  # and pulled out again before the first reopen finished
    follower.changes.set()
    follower.release.set()
    await asyncio.sleep(0.01)
    following.cancel()
    assert follower.said == [AudioMoved(HEADSET), AudioMoved(BUILT_IN)]


def test_where_the_audio_went_is_said_by_the_devices_names() -> None:
    assert system_text(AudioMoved(BUILT_IN)) == "Audio moved: listening on MacBook Pro Microphone, speaking on MacBook Pro Speakers."


async def test_a_notice_that_leaves_the_defaults_where_they_were_reopens_nothing() -> None:
    follower = Follower()
    following = follower.follow()
    await asyncio.sleep(0.01)
    follower.changes.set()
    await asyncio.sleep(0.01)
    following.cancel()
    assert (follower.reopens, follower.said) == (0, [])


async def test_a_headset_unplugged_before_the_pipeline_started_is_followed_once_it_has() -> None:
    follower = Follower(BUILT_IN)
    follower.defaults = DefaultDevices(input=2, output=2)  # moved while the models loaded, after PortAudio listed the old ones
    following = follower.follow()
    await asyncio.sleep(0.01)
    following.cancel()
    assert follower.said == [AudioMoved(BUILT_IN)]
