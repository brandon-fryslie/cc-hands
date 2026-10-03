"""What the menu-bar indicator shows for the daemon's verdict, and when it posts a notification. No AppKit here."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from hands.sessions.heartbeat import Down, NeverRan, Status, Stopped, Unreadable, Unresponsive, Up, Verdict, describe

# Stopped and never ran are one light: in both, nothing is running and nothing went wrong on the way to that.
Light = Literal["up", "deaf", "not responding", "down", "off", "unreadable"]
# The lights of a running daemon: leaving one for a worse light is news, and up is the only one that is not a warning.
RUNNING: frozenset[Light] = frozenset({"up", "deaf"})

# [LAW:no-silent-failure] an unreadable heartbeat is as loud as a dead daemon: nothing can say hands is up.
TITLES: dict[Light, str] = {
    "up": "✋",
    "deaf": "⚠︎ hands can't hear",
    "not responding": "⚠︎ hands stuck",
    "down": "⚠︎ hands down",
    "unreadable": "⚠︎ hands unreadable",
    "off": "✋ off",
}
# Up with a turn open: the same light, so a turn is never news to the notices, only to the eye.
LISTENING = "✋ 🎙"


# After a notice, how long another departure from up is shown and not posted: a daemon whose loop stalls and recovers
# over and over leaves up each time, and a notice for each would bury the screen.
QUIET = timedelta(seconds=60)


def title(verdict: Verdict, light: Light) -> str:
    match verdict:
        # A press to talk with no microphone opens a turn that hears nothing, so it never shows as one.
        case Up(status=Status(listening=True, deaf=False)):
            return LISTENING
        case _:
            return TITLES[light]


def light(verdict: Verdict) -> Light:
    # [LAW:one-source-of-truth] the light follows heartbeat.judge's verdict; an up daemon's own word that it cannot hear splits up in two.
    match verdict:
        case Up(status=Status(deaf=True)):
            return "deaf"
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
    shown = title(verdict, after)
    text = describe(verdict, now)
    match before:
        case None:
            # A daemon found already down at the first look is shown, not announced: only a departure from a running light is news.
            return Shown(after, shown, text, (), False, None)
        case Shown(light=was, owed=owed, posted_at=posted_at):
            # A departure held back by the quiet window is owed, not dropped: it goes out when the window closes,
            # unless hands has come back up by then and there is nothing left to tell. A daemon that cannot hear is
            # still running, so its going down or getting stuck is a departure as much as up's.
            owing = after != "up" and (owed or (was in RUNNING and after != was))
            quiet = posted_at is not None and now - posted_at < QUIET
            notices = (text,) if owing and not quiet else ()
            return Shown(after, shown, text, notices, owing and not notices, now if notices else posted_at)


def finished(verdict: Verdict, orphaned: bool, run: int) -> bool:
    """Whether the indicator is done: the run that started it, pid `run`, is gone, and the heartbeat no longer says it is up."""
    # A run that has exited but is not yet reaped still holds its pid and reads as up; the look after says what became
    # of it. A heartbeat that says up for another pid is the next run's, which has an indicator of its own.
    return orphaned and not (isinstance(verdict, Up) and verdict.status.pid == run)


def last_words(seen: Shown) -> tuple[str, ...]:
    """What the indicator posts on its way out: a departure the quiet window was holding back goes out now or never."""
    return seen.notices or ((seen.text,) if seen.owed else ())
