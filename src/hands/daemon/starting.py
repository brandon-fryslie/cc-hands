"""How a run says "starting" through the slow steps of its start, the first of which imports Pipecat itself.

Nothing here imports Pipecat, so the one beater can beat through that import as it does through every later step.
"""

import asyncio
import os
import signal
import sys
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, NoReturn

from hands.daemon.restart import RESTART_SIGNAL
from hands.sessions import heartbeat
from hands.sessions.audit import Record, Restarting
from hands.sessions.home import Home

# The signals that stop a run as the q key does: closing its terminal is how a run in a terminal is most often ended.
QUIT_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
# How a run that was told to stop ends: gone, or started again.
type Ending = Literal["quit", "restart"]
# How each signal that stops a run ends it: the restart signal, which the plugin's restart skill sends, stops it as the
# others do and then starts it again.
STOP_SIGNALS: dict[signal.Signals, Ending] = {**{number: "quit" for number in QUIT_SIGNALS}, RESTART_SIGNAL: "restart"}
# What the last heartbeat of a run says its pipeline is, by how the run ended: a restart is the next run starting, so
# the indicator never reads the moment between the two as hands having stopped, and posts nothing.
LAST_BEAT: dict[Ending, heartbeat.PipelineState] = {"quit": "stopped", "restart": "starting"}


@dataclass(frozen=True)
class Ended:
    """What a run that was told to stop knew last, which its last heartbeat says beside the pipeline's state."""

    last_audio_out: datetime | None
    live_sessions: int


def invocation(home: Home, *arguments: str) -> list[str]:
    """The command line that runs `hands --home <home> <arguments>` on this Python and this code, whichever `hands` is on PATH."""
    # -P, as the plugin's launcher runs Python: the working directory is kept off the path.
    return [sys.executable, "-P", "-m", "hands.daemon", "--home", str(home.root), *arguments]


def again(argv: list[str], record: Record) -> NoReturn:
    """Start the run again as `argv`, in this process: the same pid, terminal, and children, with the code on disk now.

    [LAW:one-source-of-truth] the pid stays the run's, so the menu-bar indicator watching it carries on, and the
    sessions a run lists are read again from the home, where they outlive any one run.
    """
    record(Restarting(os.getpid()))
    # exec replaces the process without running Python's exit: what is buffered for the terminal is written first.
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(argv[0], argv)


async def start[T](prepare: Callable[[], Coroutine[object, object, T]], heart: heartbeat.Heart, live: Callable[[], int], quit_event: asyncio.Event) -> T | None:
    """What `prepare` makes, while the loop beats "starting"; None when told to stop first.

    [LAW:single-enforcer] one beater says "starting" for each step of the start. Its slow steps run off the loop:
    importing Pipecat takes seconds, saying what hands is missing asks `claude`, reading the configuration can wait on
    the user at a keychain prompt, and loading the models takes seconds. A start that waits reads as starting, and only
    a stuck loop as not responding.
    """
    preparing = asyncio.create_task(prepare())
    starting = asyncio.create_task(keep_beating(lambda: heart.beat("starting", None, live(), listening=False, deaf=False), heart.period.total_seconds()))
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
