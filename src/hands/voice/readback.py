"""What the user hears about a draft, a command, or an interrupt, built from what happened to it, never from the model repeating itself."""

import difflib
import re
from collections.abc import Iterable, Mapping

from hands.core.drafts import DraftAmended, DraftDiscarded, DraftOutcome, DraftStaged, NothingStaged
from hands.core.effects import Command, Key, NotTyped, Text, Typed
from hands.core.keyboard import KeyboardOutcome, NothingRunning
from hands.core.reach import AtItsDialog, SessionEnded, UnknownSession, Unwrapped
from hands.core.session import Mode, PermissionMode, Resolution, SessionId, UnknownMode
from hands.sessions.registry import Listing, Sessions

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
        case UnknownSession(session=session):
            return f"There is no session {session}."
        case NothingStaged():
            return f"There is no draft for {name}."
        case SessionEnded():
            return f"{name} has ended, so its draft cannot be staged, changed, or sent."
        case Typed():
            return f"Sent the draft to {name}."
        case NotTyped(input=Text(prompt=text), reason=reason):
            return f"The draft for {name} was not sent, and is no longer staged: {reason}. It said: {text}"
        case Unwrapped():
            return f"{name} was not started under fritter, so hands cannot type into it. The draft is still staged."
        case AtItsDialog():
            return f"{name} is waiting at a dialog, which would take the draft as its answer. The draft is still staged."


def keyboard_readback(outcome: KeyboardOutcome, name: str) -> str:
    """One spoken reply for what came of a command or an interrupt; `name` is how the user knows the session."""
    match outcome:
        case Typed(input=input):
            return f"Typed {_spoken_input(input)} into {name}."
        case NotTyped(input=input, reason=reason):
            return f"{_spoken_input(input)} was not typed into {name}: {reason}."
        case NothingRunning():
            return f"{name} is at its prompt, so there is nothing to interrupt."
        case UnknownSession(session=session):
            return f"There is no session {session}."
        case SessionEnded():
            return f"{name} has ended."
        case Unwrapped():
            return f"{name} was not started under fritter, so hands cannot type into it."
        case AtItsDialog():
            return f"{name} is waiting at a dialog, which would take the command as its answer. Nothing was sent."


def _spoken_input(input: Command | Key) -> str:
    match input:
        case Command():
            return input.typed
        case Key(key=key):
            return key.replace("_", " ").capitalize()


def spoken_title(listing: Listing) -> str:
    return f"{listing.title or 'untitled'} in {listing.session.membership.cwd.name}"


# Each mode as the footer of a session's own screen names it, so what is heard is what the user would read there:
# 2.1.281 shows the default mode as "manual mode on".
_MODES: Mapping[PermissionMode, str] = {
    "default": "manual mode",
    "acceptEdits": "accept edits mode",
    "plan": "plan mode",
    "auto": "auto mode",
    "dontAsk": "don't ask mode",
    "bypassPermissions": "bypass permissions mode",
}


def spoken_mode(mode: Mode) -> str:
    match mode:
        case UnknownMode(name=name):
            return f"a mode hands does not know, named {name}"
        case known:
            return _MODES[known]


def spoken_name(sessions: Sessions, session: SessionId) -> str:
    """How the user knows a session: its title and project, or its id when the registry has never heard of it."""
    listing = sessions.listing(session)
    return session if listing is None else spoken_title(listing)


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
