"""How a run says "starting" through the slow steps of its start, the first of which imports Pipecat itself.

Nothing here imports Pipecat, so the one beater can beat through that import as it does through every later step.
"""

import asyncio
import os
import signal
import sys
from collections.abc import Callable, Coroutine, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, NoReturn

from hands.daemon.restart import RESTART_SIGNAL
from hands.sessions import heartbeat
from hands.sessions.home import Home
from hands.sessions import wide
from hands.sessions.wide import Fact, WideEvent, begun

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


class CannotStart(Exception):
    """A start hands will not make: a setting, a key, or a grant it cannot start without. The message is the reason."""


class Start:
    """A start of hands, from `hands run` beginning to its pipeline reported started, or to what ended it first.

    Its one event, `hands.start`, is timed from the start's first moment, so its duration is how long hands took to be
    ready. Its facts say which run it is (`pid`, `restarted`, `after_crash`, and what became of the `previous_indicator`
    a restart replaced), which settings won, and what the run listens
    on, each added as the step that learns it is taken, so a start that ended first says how far it got. It ends ok when
    ready, failed with what raised (a CannotStart's reason, where it was refused), or cancelled, told to stop first.

    [LAW:nothing-unseen] no one body runs a start: it begins at the door, before the run's loop exists, and ends in the
    middle of the run. Its event is emitted as it ends, timed from the moment it began: held open as a unit over the run
    instead, every server and task the start makes would be inside it, and every unit of work they ran a part of the
    start's trace.
    """

    def __init__(self, restarted: bool) -> None:
        self._began = begun()
        self._facts: dict[str, Fact] = {"pid": os.getpid(), "restarted": restarted}
        self._ended = False

    def heard(self, **facts: Fact) -> None:
        """Add what a step of the start learned to its event."""
        if self._ended:
            # [LAW:no-silent-failure] a fact learned after the event was emitted would never reach the log.
            raise LookupError(f"the start has ended; {sorted(facts)} came too late for its event")
        self._facts.update(facts)

    def ended(self, emit: Callable[[WideEvent], None], raised: BaseException | None) -> None:
        """Emit the start's event: ready where nothing `raised`, cancelled where a CancelledError did, failed otherwise.

        [LAW:single-enforcer] the one place the start's event is emitted, and only once: a second end is a bug, refused.
        """
        if self._ended:
            raise RuntimeError("the start has already ended")
        self._ended = True
        # What raised is the caller's to raise on; the event only writes it down.
        wide.ended("hands.start", emit, self._began, raised, **self._facts)

    @contextmanager
    def ending(self, emit: Callable[[WideEvent], None]) -> Generator[None]:
        """Run the body, and end the start here where the body ends before it was ready: failed with what it raised, or
        cancelled where it returned, told to stop first. A start already ended is left as it ended."""
        try:
            yield
        except BaseException as error:
            if not self._ended:
                self.ended(emit, error)
            raise
        if not self._ended:
            self.ended(emit, asyncio.CancelledError())


def refuse(cannot: CannotStart, held: heartbeat.Heart | None) -> None:
    """Say why a start cannot be made where it was started, and, where the run holds the heartbeat, as its last; one
    refused before it holds one leaves the heartbeat to whatever wrote it. Its Start ends failed with the reason.

    [LAW:nothing-unseen] a start from a launcher whose terminal nobody watches is otherwise only gone: `hands status`, and
    the menu-bar indicator of a start refused after it was shown, read the reason from the heartbeat, and the log keeps
    it, as the start's event, after the next run replaces that.
    """
    reason = str(cannot)
    print(f"hands: {reason}", file=sys.stderr)
    match held:
        case heartbeat.Heart():
            held.beat(heartbeat.Refusal(reason), None, 0, listening=False, degraded=())
        case None:
            pass


@dataclass(frozen=True)
class Ended:
    """What a run that was told to stop knew last, which its last heartbeat says beside the pipeline's state."""

    last_audio_out: datetime | None
    live_sessions: int


def invocation(home: Home, *arguments: str) -> list[str]:
    """The command line that runs `hands --home <home> <arguments>` on this Python and this code, whichever `hands` is on PATH."""
    # -P: the working directory is kept off the path.
    return [sys.executable, "-P", "-m", "hands.daemon", "--home", str(home.root), *arguments]


def again(argv: list[str]) -> NoReturn:
    """Start the run again as `argv`, in this process: the same pid, terminal, and children, with the code on disk now.
    The start after it says it was restarted.

    [LAW:one-source-of-truth] the pid stays the run's, so the menu-bar indicator watching it waits out the start until
    the run after ends it for its own, and the sessions a run lists are read again from the home, where they outlive any one run.
    """
    # exec replaces the process without running Python's exit: what is buffered for the terminal is written first.
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(argv[0], argv)


async def start[T](prepare: Callable[[], Coroutine[object, object, T]], heart: heartbeat.Heart, live: Callable[[], int], degraded: Callable[[], tuple[heartbeat.Degradation, ...]], quit_event: asyncio.Event
) -> T | None:
    """What `prepare` makes, while the loop beats "starting"; None when told to stop first.

    [LAW:single-enforcer] one beater says "starting" for each step of the start. Its slow steps run off the loop:
    importing Pipecat takes seconds, saying what hands is missing asks `claude`, reading the configuration can wait on
    the user at a keychain prompt, and loading the models takes seconds. A start that waits reads as starting, and only
    a stuck loop as not responding.
    """
    preparing = asyncio.create_task(prepare())
    starting = asyncio.create_task(keep_beating(lambda: heart.beat("starting", None, live(), listening=False, degraded=degraded()), heart.period.total_seconds()))
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
