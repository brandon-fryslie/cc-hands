"""The trigger: the way a turn of the user's is opened at the desk, one choice the user changes by voice while hands runs.

[LAW:one-type-per-behavior] every way of opening a turn is one value of `Trigger`, never a flag of its own: a trigger
built later is one more value here, one more arm wherever a trigger is matched, and nothing else.
"""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Literal

from hands.core.place import Place

# Right Shift held alone for each turn (`hands.voice.hold`); or held once to engage, after which the user's voice opens
# each turn and end-of-turn detection closes it (`hands.voice.engaged`); or the wake word said, with no key at all
# (`hands.voice.wake`).
Trigger = Literal["held key", "engaged conversation", "wake word"]
# [LAW:domain-language] an edge is what moves the gate (docs/architecture.md, "The gate has one owner and several
# edges"): the desk's is the trigger in use, and the phone's is its page's talk button, there for every call.
Edge = Trigger | Literal["phone button"]


def place_of(edge: Edge) -> Place:
    """Where the user is when `edge` moves the gate."""
    match edge:
        case "held key" | "engaged conversation" | "wake word":
            return "desk"
        case "phone button":
            return "phone"


class Triggers:
    """The one owner of which trigger is in use: the brain's tools switch it, and the desk is driven by its edge."""

    # [LAW:no-shared-mutable-globals] the brain's tool writes and the desk's driver reads, both on the event loop.
    def __init__(self) -> None:
        self._in_use: Trigger = "held key"
        # Set once the trigger in use is replaced by another; the driver of the desk waits on it.
        self._switched = asyncio.Event()

    def choose(self, trigger: Trigger) -> Trigger:
        """Put `trigger` in use, and get back the one it replaced."""
        was, self._in_use = self._in_use, trigger
        if trigger != was:
            self._switched.set()
        return was

    async def drive(self, edge: Callable[[Trigger], Coroutine[object, object, None]]) -> None:
        """Run `edge` for the trigger in use until it is switched, then for the new one in its place, until cancelled.

        [LAW:no-ambient-temporal-coupling] the old edge is stopped before the new one starts, so the next turn opens the
        new way and no key or word is read by both. An edge that fails ends the drive with its failure.
        """
        while True:
            self._switched = switched = asyncio.Event()
            async with asyncio.TaskGroup() as group:
                running = group.create_task(edge(self._in_use))
                await switched.wait()
                running.cancel()

    @property
    def in_use(self) -> Trigger:
        return self._in_use


def described(trigger: Trigger) -> str:
    """What hands says of a trigger in use: its name, and how to talk under it."""
    match trigger:
        case "held key":
            return "The held key: hold Right Shift to talk, and let go to send."
        case "engaged conversation":
            return "Engaged conversation: hold Right Shift once to engage, then just talk; hands answers when you finish, and listens again. Hold it once more to disengage."
        case "wake word":
            return "The wake word: say Hey Jarvis, then what you want; hands answers when you finish. It cannot hear the wake word while it speaks."
