"""The trigger: the way a turn of the user's is opened at the desk, one choice the user changes by voice while hands runs.

[LAW:one-type-per-behavior] every way of opening a turn is one value of `Trigger`, never a flag of its own: a trigger
built later is one more value here, one more arm wherever a trigger is matched, and nothing else.
"""

from typing import Literal

# Right Shift held alone for each turn (`hands.voice.hold`).
Trigger = Literal["held key"]


class Triggers:
    """The one owner of which trigger is in use: the brain's tools switch it, and the talk key's edge reads it at every
    key event, so the next turn opens the way the user last chose."""

    # [LAW:no-shared-mutable-globals] the brain's tool writes and the talk key's edge reads, both on the event loop.
    def __init__(self) -> None:
        self._in_use: Trigger = "held key"

    def choose(self, trigger: Trigger) -> None:
        self._in_use = trigger

    @property
    def in_use(self) -> Trigger:
        return self._in_use


def described(trigger: Trigger) -> str:
    """What hands says of a trigger in use: its name, and how to talk under it."""
    match trigger:
        case "held key":
            return "The held key: hold Right Shift to talk, and let go to send."
