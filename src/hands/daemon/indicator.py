"""What the menu-bar indicator shows for the daemon's verdict, and when it posts a notification. No AppKit here."""

from dataclasses import dataclass
from datetime import datetime, timedelta
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


# After a notice, how long another departure from up is shown and not posted: a daemon that crashes on every start
# comes up and goes down again once per launchd throttle interval, and a notice for each would bury the screen.
QUIET = timedelta(seconds=60)


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
    owed: bool  # hands left up while the quiet window was open, and has not come back since
    posted_at: datetime | None  # when a notice last went out, which opens the quiet window


def show(before: Shown | None, verdict: Verdict, now: datetime) -> Shown:
    """The indicator after this look, given what it showed at the look before (None on its first look)."""
    after = light(verdict)
    text = describe(verdict, now)
    match before:
        case None:
            # A daemon found already down at the first look is shown, not announced: only a departure from up is news.
            return Shown(after, TITLES[after], text, (), False, None)
        case Shown(light=was, owed=owed, posted_at=posted_at):
            # A departure held back by the quiet window is owed, not dropped: it goes out when the window closes,
            # unless hands has come back up by then and there is nothing left to tell.
            owing = after != "up" and (owed or was == "up")
            quiet = posted_at is not None and now - posted_at < QUIET
            notices = (text,) if owing and not quiet else ()
            return Shown(after, TITLES[after], text, notices, owing and not notices, now if notices else posted_at)
