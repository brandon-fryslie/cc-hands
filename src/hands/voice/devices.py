"""Keeping the voice on the system's default audio devices: an unplugged headset moves it, and it says where to."""

import asyncio
from collections.abc import Awaitable, Callable

from hands.voice.coreaudio import DefaultDevices, default_device_changes, default_devices
from hands.voice.microphone import Devices
from hands.voice.system import AudioMoved, SystemFact


async def follow(
    changes: asyncio.Event,
    current: Callable[[], DefaultDevices],
    reopen: Callable[[], Awaitable[Devices]],
    say: Callable[[SystemFact], Awaitable[None]],
) -> None:
    """Each time the defaults move off the ones last opened on, reopen on the new ones and say which; until cancelled."""
    opened = current()
    while True:
        opened = await moved(changes, current, opened)
        # [LAW:no-silent-failure] a reopen that fails raises out of here and stops the run, so launchd starts a
        # fresh daemon on whatever devices there are, rather than one that has silently gone deaf and mute.
        await say(AudioMoved(await reopen()))


async def moved(changes: asyncio.Event, current: Callable[[], DefaultDevices], opened: DefaultDevices) -> DefaultDevices:
    """The defaults, once they are no longer the ones opened on.

    An unplugged headset changes the input and the output within a millisecond, as two notices; the second finds
    the defaults already what the first reopened on. A change while a reopen runs is read after it, and followed.
    """
    while (now := current()) == opened:
        await changes.wait()
        changes.clear()
    return now


async def follow_default_devices(started: asyncio.Event, reopen: Callable[[], Awaitable[Devices]], say: Callable[[SystemFact], Awaitable[None]]) -> None:
    """Follow the system's default devices once the pipeline has started, until cancelled."""
    # [LAW:no-ambient-temporal-coupling] the streams are Pipecat's to open and start until the pipeline has started.
    # A change that falls between Pipecat opening them and reporting the start, milliseconds, is the one not followed.
    await started.wait()
    loop = asyncio.get_running_loop()
    changes = asyncio.Event()
    with default_device_changes(lambda: loop.call_soon_threadsafe(changes.set)):
        await follow(changes, default_devices, reopen, say)
