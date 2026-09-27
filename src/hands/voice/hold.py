"""The talk key: Right Shift held by itself. A pure machine from what the keyboard did to what the turn does.

Right Shift is also Shift, so the key cannot mean talk the moment it goes down. It means talk once it has been held
alone for `HOLD_SECONDS`; a quick tap is nothing, and a key pressed while it is held is typing: no turn starts, and a
turn already started is dropped, never sent. Right Shift itself is never swallowed, so Shift keeps working as Shift.
"""

from dataclasses import dataclass
from typing import Literal

HOLD_SECONDS = 0.3

Instant = float  # seconds on the monotonic clock


@dataclass(frozen=True)
class Pressed:
    """Right Shift went down."""

    at: Instant


@dataclass(frozen=True)
class Released:
    """Right Shift went up."""


@dataclass(frozen=True)
class Typed:
    """Any other key, a modifier included, went down or changed."""


@dataclass(frozen=True)
class Ripe:
    """`HOLD_SECONDS` have passed since the press made at `pressed_at`, whatever happened in between."""

    pressed_at: Instant


KeyEvent = Pressed | Released | Typed | Ripe


@dataclass(frozen=True)
class Idle:
    pass


@dataclass(frozen=True)
class Arming:
    """Right Shift is down alone, pressed at `since`, and not yet held long enough to mean talk."""

    since: Instant


@dataclass(frozen=True)
class Talking:
    pass


@dataclass(frozen=True)
class Typing:
    """Right Shift is down as Shift: its release ends nothing."""


Hold = Idle | Arming | Talking | Typing

# What the hold does to the turn: opens it, closes it and sends it, or closes it and throws it away.
Move = Literal["start", "stop", "drop"]


def step(hold: Hold, event: KeyEvent) -> tuple[Hold, tuple[Move, ...]]:
    """The hold after `event`, and what it does to the turn: most keystrokes do nothing to it."""
    match hold, event:
        case Idle(), Pressed(at=at):
            return Arming(at), ()
        # [LAW:no-ambient-temporal-coupling] a Ripe counts only for the press it was scheduled for: one left over
        # from an earlier press, released and pressed again since, names another instant and changes nothing.
        case Arming(since=since), Ripe(pressed_at=pressed_at) if pressed_at == since:
            return Talking(), ("start",)
        case Arming(), Typed():
            return Typing(), ()
        case Talking(), Typed():
            return Typing(), ("drop",)
        case Talking(), Released():
            return Idle(), ("stop",)
        case Arming() | Typing(), Released():
            return Idle(), ()
        case _:
            return hold, ()
