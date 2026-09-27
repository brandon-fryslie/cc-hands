"""The talk key: Right Shift held by itself. A pure machine from what the keyboard did to what the turn does.

Right Shift is also Shift, so the key cannot mean talk the moment it goes down. It means talk once it has been held
alone for `HOLD_SECONDS`; a quick tap is nothing, and a key pressed while it is held is typing: no turn starts, and a
turn already started is dropped, never sent. Right Shift itself is never swallowed, so Shift keeps working as Shift.
The microphone opens on the press all the same, so words said before the hold means talk are kept: they reach a turn
only if the hold opens one, and are thrown away if it turns out to be Shift.
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
    """Anything else the hand did: another key or modifier, a click, a scroll; or Right Shift pressed in a chord."""


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

# What the hold does: opens the microphone on a press, closes it on a press that was Shift after all, and to the turn:
# opens it, closes it and sends it, or closes it and throws it away.
Move = Literal["arm", "disarm", "start", "stop", "drop"]


def turn_lines(move: Move) -> tuple[str, ...]:
    """What the terminal says of a move: the turn's edges, and nothing of the microphone arming, which every Shift does."""
    match move:
        case "start":
            return ("turn: started",)
        case "stop":
            return ("turn: ended",)
        case "drop":
            return ("turn: dropped",)
        case "arm" | "disarm":
            return ()


def step(hold: Hold, event: KeyEvent) -> tuple[Hold, tuple[Move, ...]]:
    """The hold after `event`, and what it does: most keystrokes do nothing."""
    match hold, event:
        # A press is a fresh start whatever came before: one that finds the key already down proves a release went by
        # unseen, while macOS had the tap switched off, and a turn still open is dropped rather than sent.
        case Talking(), Pressed(at=at):
            return Arming(at), ("drop", "arm")
        case _, Pressed(at=at):
            return Arming(at), ("arm",)
        # [LAW:no-ambient-temporal-coupling] a Ripe counts only for the press it was scheduled for: one left over
        # from an earlier press, released and pressed again since, names another instant and changes nothing.
        case Arming(since=since), Ripe(pressed_at=pressed_at) if pressed_at == since:
            return Talking(), ("start",)
        case Arming(), Typed():
            return Typing(), ("disarm",)
        case Talking(), Typed():
            return Typing(), ("drop",)
        case Talking(), Released():
            return Idle(), ("stop",)
        case Arming(), Released():
            return Idle(), ("disarm",)
        case Typing(), Released():
            return Idle(), ()
        case _:
            return hold, ()
