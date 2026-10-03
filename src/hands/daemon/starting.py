"""How a run says "starting" through the slow steps of its start, the first of which imports Pipecat itself.

Nothing here imports Pipecat, so the one beater can beat through that import as it does through every later step.
"""

import asyncio
import signal
from collections.abc import Callable, Coroutine

from hands.sessions import heartbeat

# The signals that stop a run as the q key does: closing its terminal is how a run in a terminal is most often ended.
QUIT_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


async def start[T](prepare: Callable[[], Coroutine[object, object, T]], heart: heartbeat.Heart, live: Callable[[], int], quit_event: asyncio.Event) -> T | None:
    """What `prepare` makes, while the loop beats "starting"; None when told to stop first.

    [LAW:single-enforcer] one beater says "starting" for each step of the start. Its slow steps run off the loop:
    importing Pipecat takes seconds, saying what hands is missing asks `claude`, reading the configuration can wait on
    the user at a keychain prompt, and loading the models takes seconds. A start that waits reads as starting, and only
    a stuck loop as not responding.
    """
    preparing = asyncio.create_task(prepare())
    starting = asyncio.create_task(keep_beating(lambda: heart.beat("starting", None, live(), False), heart.period.total_seconds()))
    quitting = asyncio.create_task(quit_event.wait())
    try:
        await asyncio.wait({preparing, starting, quitting}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        # A stop does not wait for the models or the keychain: their threads are daemons, which the process exits without.
        for task in (preparing, starting, quitting):
            if not task.done():
                task.cancel()
    if starting.done() and not starting.cancelled():
        # [LAW:no-silent-failure] the heartbeat only ends by raising, and its error stops the run as the steady one does.
        starting.result()
    return preparing.result() if preparing.done() and not preparing.cancelled() else None


async def keep_beating(beat: Callable[[], None], period: float) -> None:
    """Write the heartbeat now and once a period after, until cancelled."""
    while True:
        beat()
        await asyncio.sleep(period)
