"""What the user missed: what sessions finished and hands said while they were away, read back out of the audit log.

The log is the long memory [LAW:one-source-of-truth]: every turn the reducer tells is a Summarise it decided, every
session it says is gone a SessionGone, every occurrence a hook passes on a Tell, said or not, and everything hands says
unprompted an Announced line, so nothing here keeps a record of its own or decides again what the reducer decided; it
folds the lines of a window.
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from hands.core.session import SessionId
from hands.sessions.audit import backwards


@dataclass(frozen=True)
class LastSpoke:
    """The window opens where the user was last heard before the words being answered."""


@dataclass(frozen=True)
class Finished:
    """A session that finished turns in the window: how many, and the newest words one closed with (None for none)."""

    session: SessionId
    turns: int
    closing: str | None


@dataclass(frozen=True)
class Announcement:
    """Something hands said unprompted in the window, and how many times it said it."""

    text: str
    times: int


@dataclass(frozen=True)
class Happened:
    """What a session's hooks said happened in the window, one kind of it (hands.core.occurrences): how many times, and
    the newest, as the log holds it. Counted, so a morning of subagents is one line a session, not hundreds."""

    session: SessionId
    times: int
    newest: Mapping[str, object]


@dataclass(frozen=True)
class Missed:
    """What the window holds. `since` is when it opens, None when it reaches back to the start of the log. `unreadable`
    counts the lines in it that are not JSON: the fragment a write that failed partway leaves, ended by the next line."""

    since: datetime | None
    finished: tuple[Finished, ...]
    ended: tuple[SessionId, ...]
    occurred: tuple[Happened, ...]
    announced: tuple[Announcement, ...]
    unreadable: int


def missed(directory: Path, opening: datetime | LastSpoke) -> Missed:
    """What happened since `opening`, or since the user's words before the ones being answered.

    The words being answered are already the newest Transcribed line: the user's turn is written as it enters the
    model's context, before any tool it calls runs, so the user was last heard at the one before it.
    """
    since = None if isinstance(opening, LastSpoke) else opening
    heard = 0
    window: list[Mapping[str, object]] = []
    unreadable = 0
    # Newest first, so a window of minutes reads no further back than it opens.
    for line in backwards(directory):
        try:
            entry = cast(Mapping[str, object], json.loads(line))
        except json.JSONDecodeError:
            unreadable += 1
            continue
        at = datetime.fromisoformat(cast(str, entry["at"]))
        if since is not None and at < since:
            break
        if isinstance(opening, LastSpoke) and entry["type"] == "Transcribed":
            heard += 1
            if heard == 2:
                since = at
                break
        window.append(entry)
    return _fold(reversed(window), since, unreadable)


def _fold(entries: Iterable[Mapping[str, object]], since: datetime | None, unreadable: int) -> Missed:
    closings: dict[SessionId, list[str | None]] = {}
    ended: dict[SessionId, None] = {}
    occurred: dict[tuple[SessionId, object], Happened] = {}
    announced: dict[str, int] = {}
    for entry in entries:
        match entry:
            case {"type": "WideEvent", "event": "applied", "facts": {"effects": object() as effects}}:
                for performed in cast(list[object], effects):
                    match performed:
                        # A turn is told once however its telling went: one whose summary failed still finished.
                        case {"outcome": "ok" | "failed", "effect": {"type": "Summarise", "session": str(session), "closing": str() | None as closing}}:
                            closings.setdefault(SessionId(session), []).append(closing)
                        case {"outcome": "ok" | "failed", "effect": {"type": "SessionGone", "session": str(session)}}:
                            ended[SessionId(session)] = None
                        case {"outcome": "ok" | "failed", "effect": {"type": "Tell", "session": str(session), "occurrence": object() as occurrence}}:
                            newest = cast(Mapping[str, object], occurrence)
                            key = (SessionId(session), newest.get("type"))
                            occurred[key] = Happened(key[0], occurred[key].times + 1 if key in occurred else 1, newest)
                        case _:
                            pass
            case {"type": "Announced", "text": str(text)}:
                announced[text] = announced.get(text, 0) + 1
            case _:
                pass
    finished = tuple(Finished(session, len(said), next((words for words in reversed(said) if words), None)) for session, said in closings.items())
    return Missed(since, finished, tuple(ended), tuple(occurred.values()), tuple(Announcement(text, times) for text, times in announced.items()), unreadable)
