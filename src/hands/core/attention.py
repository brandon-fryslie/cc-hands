"""Which sessions' finished turns reach the ear unasked: each session's overlay, and how a turn is delivered by it."""

from typing import Literal

# How the user asked to hear a session's finished turns. "watched": each turn it finishes is told as it finishes.
# "normal": its turns are told only with spoken summaries on, or when the user asks for one. "muted": told only when the
# user asks, even with spoken summaries on. What a session asks (a permission, a question, a plan) is spoken whatever its
# overlay: it needs an answer, and one held unsaid would wait out its deadline and be refused.
Overlay = Literal["normal", "watched", "muted"]


# Not told until asked for: with several sessions, every one that stopped said aloud was noise (hands-announce-5md).
DEFAULT: Overlay = "normal"

# How a finished turn's summary reached the user, and what decided it: told unasked because spoken summaries are on, or
# because the session is watched; held until the user asks for it because neither is so, or because the session is
# muted. One summary is made whichever it is; only who hears it when differs.
Delivery = Literal["summaries", "watched", "on request", "muted"]


# How a session's progress reaches the user: "play", said as it happens; "note", left to the session listing, which
# the model reads when asked and says nothing of until then.
Route = Literal["play", "note"]


def progress_route(focused: bool, overlay: Overlay) -> Route:
    """The focused session is heard working; any other is noted, and a muted one is noted even when focused."""
    # [LAW:dataflow-not-control-flow] a table over the focus and the overlay, every pair a row the type checker holds to.
    match overlay, focused:
        case "muted", _:
            return "note"
        case "normal" | "watched", True:
            return "play"
        case "normal" | "watched", False:
            return "note"
