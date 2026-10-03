"""What the user missed: what sessions finished and hands said while they were away, read back out of the audit log.

The log is the long memory [LAW:one-source-of-truth]: every turn a session finishes is an Applied line, and everything
hands says unprompted an Announced one, so nothing here keeps a record of its own; it folds the lines of a window.
"""

import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

from hands.core.session import SessionId
from hands.sessions.audit import tail


@dataclass(frozen=True)
class Finished:
    """A session that finished turns in the window: how many, and the newest words one closed with (None for none)."""

    session: SessionId
    turns: int
    closing: str | None


@dataclass(frozen=True)
class Missed:
    """What the window holds. `since` is when it opens, None when it reaches back to the start of the log."""

    since: datetime | None
    finished: tuple[Finished, ...]
    ended: tuple[SessionId, ...]
    announced: tuple[str, ...]


def missed(directory: Path, now: datetime, minutes: int) -> Missed:
    """What happened in the last `minutes`, or with 0, since the user's words before the ones being answered.

    The words being answered are already the newest Transcribed line: the user's turn is written as it enters the
    model's context, before any tool it calls runs, so the user was last heard at the one before it.
    """
    lines, _ = tail(directory, sys.maxsize)
    since = now - timedelta(minutes=minutes) if minutes > 0 else None
    heard = 0
    window: list[Mapping[str, object]] = []
    for line in reversed(lines):
        entry = cast(Mapping[str, object], json.loads(line))
        at = datetime.fromisoformat(cast(str, entry["at"]))
        if since is not None and at < since:
            break
        if minutes <= 0 and entry["type"] == "Transcribed":
            heard += 1
            if heard == 2:
                since = at
                break
        window.append(entry)
    return _fold(reversed(window), since)


def _fold(entries: Iterable[Mapping[str, object]], since: datetime | None) -> Missed:
    # Keyed by the turn: a turn's reply is heard on the wire before its Stop, and a Stop blocked by a hook comes again,
    # each under the one prompt id, so a turn is counted once and keeps its newest words.
    turns: dict[tuple[SessionId, str], str | None] = {}
    ended: dict[SessionId, None] = {}
    announced: list[str] = []
    for entry in entries:
        match entry:
            case {"type": "Applied", "event": {"type": "Stopped" | "Closed", "session": str(session), "prompt": str(prompt), "closing": str() | None as closing}}:
                key = (SessionId(session), prompt)
                turns[key] = closing or turns.get(key)
            case {"type": "Applied", "event": {"type": "Ended", "session": str(session)}}:
                ended[SessionId(session)] = None
            case {"type": "Applied", "event": {"type": "Died" | "MovedOn", "membership": {"id": str(session)}}}:
                ended[SessionId(session)] = None
            case {"type": "Announced", "text": str(text)}:
                announced.append(text)
            case _:
                pass
    by_session: dict[SessionId, list[str | None]] = {}
    for (session, _), closing in turns.items():
        by_session.setdefault(session, []).append(closing)
    finished = tuple(Finished(session, len(closings), next((said for said in reversed(closings) if said), None)) for session, closings in by_session.items())
    return Missed(since, finished, tuple(ended), tuple(announced))
