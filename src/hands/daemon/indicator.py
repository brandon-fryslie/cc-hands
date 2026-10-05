"""What the menu-bar indicator shows for the daemon's verdict, and when it posts a notification. No AppKit here."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from hands.sessions.heartbeat import Degradation, Down, NeverRan, Refused, Status, Stopped, Unreadable, Unresponsive, Up, Verdict, describe

# Stopped and never ran are one light: in both, nothing is running and nothing went wrong on the way to that.
# Up is one light, degraded or not: whether it warns is whether it has any degradation.
Light = Literal["up", "not responding", "down", "refused", "off", "unreadable"]

# [LAW:no-silent-failure] an unreadable heartbeat is as loud as a dead daemon: nothing can say hands is up.
TITLES: dict[Light, str] = {
    "up": "✋",
    "not responding": "⚠︎ hands stuck",
    "down": "⚠︎ hands down",
    "refused": "⚠︎ hands refused to start",
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
        # A degraded daemon shows what is wrong, turn open or not: a press to talk with no microphone hears nothing.
        case Up(status=Status(degraded=(_, *_) as degraded)):
            return f"⚠︎ hands {', '.join(each.brief for each in degraded)}"
        case Up(status=Status(listening=True)):
            return LISTENING
        case _:
            return TITLES[light]


def degradations(verdict: Verdict) -> tuple[Degradation, ...]:
    """What is wrong with a daemon that is up; one that is not up says nothing of itself that holds now."""
    match verdict:
        case Up(status=status):
            return status.degraded
        case _:
            return ()


def light(verdict: Verdict) -> Light:
    # [LAW:one-source-of-truth] the light follows heartbeat.judge's verdict.
    match verdict:
        case Up():
            return "up"
        case Unresponsive():
            return "not responding"
        case Down():
            return "down"
        case Refused():
            return "refused"
        case NeverRan() | Stopped():
            return "off"
        case Unreadable():
            return "unreadable"


@dataclass(frozen=True)
class Shown:
    light: Light
    degraded: tuple[Degradation, ...]  # what was wrong with hands at this look, were it up
    title: str  # the menu bar's text
    text: str  # the verdict in words, under the title
    notices: tuple[str, ...]  # notifications to post now
    owed: bool  # hands left up while the quiet window was open, and has not come back since
    posted_at: datetime | None  # when a notice last went out, which opens the quiet window


def show(before: Shown | None, verdict: Verdict, now: datetime) -> Shown:
    """The indicator after this look, given what it showed at the look before (None on its first look)."""
    after = light(verdict)
    degraded = degradations(verdict)
    shown = title(verdict, after)
    text = describe(verdict, now)
    match before:
        case None:
            # A daemon found already down at the first look is shown, not announced: only leaving up, or a degradation
            # arriving, is news.
            return Shown(after, degraded, shown, text, (), False, None)
        case Shown(light=was, degraded=had, owed=owed, posted_at=posted_at):
            # A departure held back by the quiet window is owed, not dropped: it goes out when the window closes,
            # unless hands has come back up whole by then and there is nothing left to tell. A change is news when it
            # leaves up for a warning, or when a degradation the look before did not show arrives, from anywhere,
            # stuck included, or beside others.
            warning = after != "up" or bool(degraded)
            arrived = not set(degraded) <= set(had)
            owing = warning and (owed or arrived or (after != was and was == "up"))
            quiet = posted_at is not None and now - posted_at < QUIET
            notices = (text,) if owing and not quiet else ()
            return Shown(after, degraded, shown, text, notices, owing and not notices, now if notices else posted_at)


def finished(verdict: Verdict, orphaned: bool, run: int) -> bool:
    """Whether the indicator is done: the run that started it, pid `run`, is gone, and the heartbeat no longer says it is up."""
    # A run that has exited but is not yet reaped still holds its pid and reads as up; the look after says what became
    # of it. A heartbeat that says up for another pid is the next run's, which has an indicator of its own.
    return orphaned and not (isinstance(verdict, Up) and verdict.status.pid == run)


def last_words(seen: Shown) -> tuple[str, ...]:
    """What the indicator posts on its way out: a departure the quiet window was holding back goes out now or never."""
    return seen.notices or ((seen.text,) if seen.owed else ())
