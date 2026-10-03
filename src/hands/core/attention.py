"""Which sessions' finished turns reach the ear unasked: each session's overlay, and how a turn is delivered by it."""

from typing import Literal, get_args

# How the user asked to hear a session's finished turns. "watched": each turn it finishes is told as it finishes.
# "normal": its turns are told only with spoken summaries on, or when the user asks for one. "muted": told only when the
# user asks, even with spoken summaries on. What a session asks (a permission, a question, a plan) is spoken whatever its
# overlay: it needs an answer, and one held unsaid would wait out its deadline and be refused.
Overlay = Literal["normal", "watched", "muted"]


def overlay_named(name: str) -> Overlay | None:
    """The overlay `name` names, or None where it names none."""
    # [LAW:single-enforcer] the one reading of an overlay's name, whether the user's tool call or the home's file wrote it.
    return next((overlay for overlay in get_args(Overlay) if overlay == name), None)


# Not told until asked for: with several sessions, every one that stopped said aloud was noise (hands-announce-5md).
DEFAULT: Overlay = "normal"

# How a finished turn's summary reached the user, and what decided it: told unasked because spoken summaries are on, or
# because the session is watched; held until the user asks for it because neither is so, or because the session is
# muted. One summary is made whichever it is; only who hears it when differs.
Delivery = Literal["summaries", "watched", "on request", "muted"]
