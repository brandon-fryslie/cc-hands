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
        self.changes = asyncio.Event()
        self.opened = list(opened)
        self.reopens = 0
        self.said: list[SystemFact] = []
        self.release = asyncio.Event()
        self.release.set()

    async def reopen(self) -> Devices:
        self.reopens += 1
        await self.release.wait()
        return self.opened.pop(0)

    async def say(self, fact: SystemFact) -> None:
        self.said.append(fact)


async def test_a_change_of_defaults_reopens_the_transport_and_says_where_the_audio_went() -> None:
    follower = Follower(BUILT_IN)
    following = asyncio.create_task(follow(follower.changes, lambda: follower.defaults, follower.reopen, follower.say))
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
    following = asyncio.create_task(follow(follower.changes, lambda: follower.defaults, follower.reopen, follower.say))
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
    following = asyncio.create_task(follow(follower.changes, lambda: follower.defaults, follower.reopen, follower.say))
    await asyncio.sleep(0.01)
    follower.changes.set()
    await asyncio.sleep(0.01)
    following.cancel()
    assert (follower.reopens, follower.said) == (0, [])
