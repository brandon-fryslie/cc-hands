"""What the menu-bar indicator shows for the daemon's verdict, and when it posts a notification. No AppKit here."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from hands.sessions.heartbeat import Degradation, Down, NeverRan, Refused, Status, Stopped, Unreadable, Unresponsive, Up, Verdict, describe

# Stopped and never ran are one light: in both, nothing is running and nothing went wrong on the way to that.
# Up is one light, degraded or not: whether it warns is whether it has any degradation.
Light = Literal["up", "not responding", "down", "refused", "off", "unreadable"]
# What a notice tells: the warning light hands left up for, or a degradation that arrived.
type News = Light | Degradation

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
        # A degraded daemon shows what is wrong, and a turn open beside it; a daemon that cannot hear never says it is listening.
        case Up(status=Status(degraded=(_, *_) as degraded, listening=listening)):
            return f"⚠︎ hands {', '.join(each.brief for each in degraded)}{' 🎙' if listening else ''}"
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


# tmux styles for the status-line segment: colour carries the urgency, and the title's ⚠︎ survives a monochrome line.
STYLES: dict[Light, str] = {
    "up": "fg=green",
    "not responding": "fg=yellow",
    "down": "fg=red,bold",
    "refused": "fg=red,bold",
    "unreadable": "fg=red,bold",
    "off": "fg=colour244",
}
# An up daemon that warns: up, but not well.
DEGRADED_STYLE = "fg=yellow"


def segment(verdict: Verdict) -> str:
    """The menu bar's title as a tmux status-line segment, styled by its light."""
    # [LAW:one-source-of-truth] one more rendering of the light and title the menu bar shows, so the two cannot disagree.
    shown = light(verdict)
    style = DEGRADED_STYLE if degradations(verdict) else STYLES[shown]
    # tmux reads `#` in a segment as the start of a style or format; doubled, it is the character.
    return f"#[{style}]{title(verdict, shown).replace('#', '##')}#[default]"


@dataclass(frozen=True)
class Shown:
    light: Light
    degraded: tuple[Degradation, ...]  # what was wrong with hands at this look, were it up
    title: str  # the menu bar's text
    text: str  # the verdict in words, under the title
    notices: tuple[str, ...]  # notifications to post now
    owed: frozenset[News]  # news the quiet window held back, which still holds
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
            return Shown(after, degraded, shown, text, (), frozenset(), None)
        case Shown(light=was, degraded=had, owed=owed, posted_at=posted_at):
            # News is leaving up for a warning, or a degradation the look before did not show arriving, from anywhere,
            # stuck included, or beside others. News held back by the quiet window is owed, not dropped: it goes out
            # when the window closes if it still holds then, a departure until hands is back up, a degradation until it clears.
            left: set[News] = {after} if after != "up" and was == "up" else set()
            owing = frozenset[News](news for news in left | (set(degraded) - set(had)) | owed if holds(news, after, degraded))
            quiet = posted_at is not None and now - posted_at < QUIET
            notices = (text,) if owing and not quiet else ()
            return Shown(after, degraded, shown, text, notices, frozenset() if notices else owing, now if notices else posted_at)


def holds(news: News, after: Light, degraded: tuple[Degradation, ...]) -> bool:
    match news:
        case Degradation():
            return news in degraded
        case _:
            return after != "up"


def finished(verdict: Verdict, orphaned: bool, run: int) -> bool:
    """Whether the indicator is done: the run that started it, pid `run`, is gone, and the heartbeat no longer says it is up."""
    # A run that has exited but is not yet reaped still holds its pid and reads as up; the look after says what became
    # of it. A heartbeat that says up for another pid is the next run's, which has an indicator of its own.
    return orphaned and not (isinstance(verdict, Up) and verdict.status.pid == run)


def last_words(seen: Shown) -> tuple[str, ...]:
    """What the indicator posts on its way out: a departure the quiet window was holding back goes out now or never."""
    return seen.notices or ((seen.text,) if seen.owed else ())
