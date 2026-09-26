"""Keeping the voice on the system's default audio devices: an unplugged headset moves it, and it says where to."""

import asyncio
from collections.abc import Awaitable, Callable

from hands.voice.coreaudio import DefaultDevices, default_device_changes, default_devices
from hands.voice.microphone import Devices, KeyedAudioTransport
from hands.voice.system import AudioMoved, SystemFact
from hands.voice.threads import off_loop


async def follow(
    changes: asyncio.Event,
    current: Callable[[], Awaitable[DefaultDevices]],
    opened_on: Callable[[], DefaultDevices],
    reopen: Callable[[], Awaitable[Devices]],
    say: Callable[[SystemFact], Awaitable[None]],
) -> None:
    """Each time the defaults move off the ones the streams are open on, reopen on the new ones and say which; until cancelled."""
    while True:
        await moved(changes, current, opened_on())
        # [LAW:no-silent-failure] a reopen that fails raises out of here and stops the run, which reads as down, so
        # the next run opens whatever devices there are, rather than one run going on silently deaf and mute.
        await say(AudioMoved(await finished(reopen())))


async def finished[T](work: Awaitable[T]) -> T:
    """The result of work that, once begun, is let finish even when the caller is cancelled."""
    # [LAW:no-ambient-temporal-coupling] a reopen holds the streams, part of it on a thread that cancelling cannot
    # stop. A follower stopped during one waits for it, within its deadline, so the pipeline's cleanup that comes next
    # finds the streams attached and closes them itself, instead of racing a reopen for them.
    running = asyncio.ensure_future(work)
    try:
        return await asyncio.shield(running)
    except asyncio.CancelledError:
        await asyncio.wait({running})
        raise


async def moved(changes: asyncio.Event, current: Callable[[], Awaitable[DefaultDevices]], opened: DefaultDevices) -> None:
    """Return once the defaults are no longer the ones opened on.

    An unplugged headset changes the input and the output within a millisecond, as two notices; the second finds
    the defaults already what the first reopened on. A change while a reopen runs is read after it, and followed.
    """
    while await current() == opened:
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
        # Read off the loop: CoreAudio answers under a lock it may be holding while it tears down a device that is gone.
        await follow(changes, lambda: off_loop(default_devices, "reading the default devices"), lambda: audio.opened_on, audio.reopen, say)
