"""The trigger: the way a turn of the user's is opened at the desk, one choice the user changes by voice while hands runs.

[LAW:one-type-per-behavior] every way of opening a turn is one value of `Trigger`, never a flag of its own: a trigger
built later is one more value here, one more arm wherever a trigger is matched, and nothing else.
"""

from typing import Literal

from hands.core.place import Place

# Right Shift held alone for each turn (`hands.voice.hold`).
Trigger = Literal["held key"]
# [LAW:domain-language] an edge is what moves the gate (docs/architecture.md, "The gate has one owner and several
# edges"): the desk's is the trigger in use, and the phone's is its page's talk button, there for every call.
Edge = Trigger | Literal["phone button"]


def place_of(edge: Edge) -> Place:
    """Where the user is when `edge` moves the gate."""
    match edge:
        case "held key":
            return "desk"
        case "phone button":
            return "phone"


class Triggers:
    """The one owner of which trigger is in use: the brain's tools switch it, and the desk is driven by its edge."""

    # [LAW:no-shared-mutable-globals] the brain's tool writes and the desk's driver reads, both on the event loop.
    def __init__(self) -> None:
        self._in_use: Trigger = "held key"

    def choose(self, trigger: Trigger) -> Trigger:
        """Put `trigger` in use, and get back the one it replaced."""
        was, self._in_use = self._in_use, trigger
        return was

    @property
    def in_use(self) -> Trigger:
        return self._in_use


def described(trigger: Trigger) -> str:
    """What hands says of a trigger in use: its name, and how to talk under it."""
    match trigger:
        case "held key":
            return "The held key: hold Right Shift to talk, and let go to send."
