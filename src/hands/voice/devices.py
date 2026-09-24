"""Keeping the voice on the system's default audio devices: an unplugged headset moves it, and it says where to."""

import asyncio
from collections.abc import Awaitable, Callable

from hands.voice.coreaudio import DefaultDevices, default_device_changes, default_devices
from hands.voice.microphone import Devices, KeyedAudioTransport
from hands.voice.system import AudioMoved, SystemFact


async def follow(
    changes: asyncio.Event,
    current: Callable[[], DefaultDevices],
    opened_on: Callable[[], DefaultDevices],
    reopen: Callable[[], Awaitable[Devices]],
    say: Callable[[SystemFact], Awaitable[None]],
) -> None:
    """Each time the defaults move off the ones the streams are open on, reopen on the new ones and say which; until cancelled."""
    while True:
        await moved(changes, current, opened_on())
        # [LAW:no-silent-failure] a reopen that fails raises out of here and stops the run, so launchd starts a
        # fresh daemon on whatever devices there are, rather than one that has silently gone deaf and mute.
        await say(AudioMoved(await reopen()))


async def moved(changes: asyncio.Event, current: Callable[[], DefaultDevices], opened: DefaultDevices) -> None:
    """Return once the defaults are no longer the ones opened on.

    An unplugged headset changes the input and the output within a millisecond, as two notices; the second finds
    the defaults already what the first reopened on. A change while a reopen runs is read after it, and followed.
    """
    while current() == opened:
        await changes.wait()
        changes.clear()


async def follow_default_devices(started: asyncio.Event, audio: KeyedAudioTransport, say: Callable[[SystemFact], Awaitable[None]]) -> None:
    """Follow the system's default devices once the pipeline has started, until cancelled."""
    # [LAW:no-ambient-temporal-coupling] the streams are Pipecat's to open and start until the pipeline has started.
    # A change before then is still followed: the defaults are compared with those read when PortAudio listed them.
    await started.wait()
    loop = asyncio.get_running_loop()
    changes = asyncio.Event()
    with default_device_changes(lambda: loop.call_soon_threadsafe(changes.set)):
        await follow(changes, default_devices, lambda: audio.opened_on, audio.reopen, say)
