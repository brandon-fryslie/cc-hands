"""What the menu-bar indicator shows for the daemon's verdict, and when it posts a notification. No AppKit here."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from hands.daemon.status import Down, NeverRan, Stopped, Unreadable, Unresponsive, Up, Verdict, describe

# Stopped and never ran are one light: in both, nothing is running and nothing went wrong on the way to that.
Light = Literal["up", "not responding", "down", "off", "unreadable"]

# [LAW:no-silent-failure] an unreadable heartbeat is as loud as a dead daemon: nothing can say hands is up.
TITLES: dict[Light, str] = {
    "up": "✋",
    "not responding": "⚠︎ hands stuck",
    "down": "⚠︎ hands down",
    "unreadable": "⚠︎ hands unreadable",
    "off": "✋ off",
}


def light(verdict: Verdict) -> Light:
    # [LAW:one-source-of-truth] the light follows status.judge's verdict, never the heartbeat's raw fields.
    match verdict:
        case Up():
            return "up"
        case Unresponsive():
            return "not responding"
        case Down():
            return "down"
        case NeverRan() | Stopped():
            return "off"
        case Unreadable():
            return "unreadable"


@dataclass(frozen=True)
class Shown:
    light: Light
    title: str  # the menu bar's text
    text: str  # the verdict in words, under the title
    notices: tuple[str, ...]  # notifications to post now


def show(before: Light | None, verdict: Verdict, now: datetime) -> Shown:
    """The indicator after this look, given the light it showed before (None on its first look)."""
    after = light(verdict)
    text = describe(verdict, now)
    # Only a departure from up is news; a daemon found already down at the first look is shown, not announced.
    leaving_up = before == "up" and after != "up"
    return Shown(after, TITLES[after], text, (text,) if leaving_up else ())
