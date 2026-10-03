"""Which of what the sessions say reaches the ear: each session's overlay, and what it lets through."""

from typing import Literal

from hands.core.effects import Heard, Narrate, Note, Speak, WaitingForYou
from hands.core.session import SessionId

# How a session's own news reaches the user. "watched": the user asked to be told when it stops and waits for them.
# "normal": they did not, so its waiting is not said, though list_sessions still names it waiting. What a session asks
# (a permission, a question, a plan) is spoken whatever its overlay: it needs an answer.
Overlay = Literal["normal", "watched"]

# Not told until asked for: with several sessions, every one that stopped said aloud was noise (hands-announce-5md).
DEFAULT: Overlay = "normal"


def routed(heard: Heard, overlay: Overlay) -> bool:
    """Whether what a session said is passed on to the user, by that session's overlay."""
    match heard, overlay:
        case Speak(announcement=WaitingForYou()), "normal":
            return False
        case _:
            return True


def speaker(heard: Heard) -> SessionId:
    """The session that said it, whose overlay routes it."""
    match heard:
        case Speak(announcement=announcement):
            return announcement.session
        case Narrate(moment=moment):
            return moment.session
        case Note(fact=fact):
            return fact.session
