"""Keeping the voice on the system's default audio devices: an unplugged headset moves it, and it says where to."""

import asyncio
from collections.abc import Awaitable, Callable

from hands.voice.coreaudio import default_device_changes
from hands.voice.microphone import Devices
from hands.voice.system import AudioMoved, SystemFact


async def follow(changes: asyncio.Event, reopen: Callable[[], Awaitable[Devices]], say: Callable[[SystemFact], Awaitable[None]]) -> None:
    """Each time the defaults change, reopen on the new ones and say which they are; until cancelled."""
    while True:
        await changes.wait()
        # Cleared before the reopen, so a change that comes while it runs is followed by another; an unplugged
        # headset changes the input and the output within a millisecond, and both are usually taken by one wake.
        changes.clear()
        # [LAW:no-silent-failure] a reopen that fails raises out of here and stops the run, so launchd starts a
        # fresh daemon on whatever devices there are, rather than one that has silently gone deaf and mute.
        await say(AudioMoved(await reopen()))


async def follow_default_devices(started: asyncio.Event, reopen: Callable[[], Awaitable[Devices]], say: Callable[[SystemFact], Awaitable[None]]) -> None:
    """Follow the system's default devices once the pipeline has started, until cancelled."""
    # [LAW:no-ambient-temporal-coupling] the streams are Pipecat's to open and start until the pipeline has started.
    # A change that falls between Pipecat opening them and reporting the start, milliseconds, is the one not followed.
    await started.wait()
    loop = asyncio.get_running_loop()
    changes = asyncio.Event()
    with default_device_changes(lambda: loop.call_soon_threadsafe(changes.set)):
        await follow(changes, reopen, say)
