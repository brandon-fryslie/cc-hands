"""What the user hears about a draft, built from what happened to it, never from the model repeating itself."""

import difflib
import re
from collections.abc import Iterable

from hands.core.drafts import (
    AwaitingPermission,
    DraftAmended,
    DraftDiscarded,
    DraftOutcome,
    DraftSent,
    DraftStaged,
    NothingStaged,
    OutsideTmux,
    SessionEnded,
    UnknownSession,
)
from hands.core.session import Resolution

_WORDS = re.compile(r"[^\s]+|\n")


def readback(outcome: DraftOutcome, name: str) -> str:
    """One spoken reply for the outcome; `name` is how the user knows the session."""
    match outcome:
        case DraftStaged(draft=draft, replaced=None):
            return f"Draft for {name}{_reading(draft.resolutions)}: {draft.text}"
        case DraftStaged(draft=draft):
            return f"New draft for {name}, replacing the last one{_reading(draft.resolutions)}: {draft.text}"
        case DraftAmended(before=before, after=after):
            # Speak what changed, not the whole draft again.
            added = [resolution for resolution in after.resolutions if resolution not in before.resolutions]
            return f"In the draft for {name}{_reading(added)}: {_amendment(before.text, after.text)}"
        case DraftDiscarded():
            return f"Discarded the draft for {name}."
        case DraftSent():
            return f"Sent to {name}."
        case UnknownSession(session=session):
            return f"There is no session {session}."
        case NothingStaged():
            return f"There is no draft for {name}."
        case SessionEnded():
            return f"{name} has ended, so nothing was sent."
        case OutsideTmux():
            return f"{name} is not running in tmux, so I cannot type into it."
        case AwaitingPermission(permission=permission):
            return f"{name} is waiting for permission to use {permission.tool}. Answer that first; the draft is still staged."


def _reading(resolutions: Iterable[Resolution]) -> str:
    return "".join(f", reading '{resolution.heard}' as {resolution.meant}" for resolution in resolutions)


def _amendment(before: str, after: str) -> str:
    match (_changes(before, after), before == after):
        case ([], True):
            return "nothing changed"
        case ([], False):
            return "only the spacing changed"
        case (changes, _):
            return "; ".join(changes)


def _changes(before: str, after: str) -> list[str]:
    # A line break is a word here, so moving text onto its own line is heard as a change.
    old, new = _WORDS.findall(before), _WORDS.findall(after)
    return [
        _change(_spoken(old[i1:i2]), _spoken(new[j1:j2]))
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes()
        if tag != "equal"
    ]


def _spoken(words: list[str]) -> str:
    return " ".join("a line break" if word == "\n" else word for word in words)


def _change(removed: str, added: str) -> str:
    match (removed, added):
        case ("", _):
            return f"added '{added}'"
        case (_, ""):
            return f"removed '{removed}'"
        case _:
            return f"'{removed}' is now '{added}'"
