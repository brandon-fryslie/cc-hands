"""Which sessions' finished turns reach the ear unasked: each session's overlay, and how a turn is delivered by it."""

from typing import Literal

# Whether the user asked to hear a session's finished turns. "watched": each turn it finishes is told as it finishes.
# "normal": its turns are told only with spoken summaries on, or when the user asks for one. What a session asks
# (a permission, a question, a plan) is spoken whatever its overlay: it needs an answer.
Overlay = Literal["normal", "watched"]

# Not told until asked for: with several sessions, every one that stopped said aloud was noise (hands-announce-5md).
DEFAULT: Overlay = "normal"

# How a finished turn's summary reached the user: told unasked because spoken summaries are on, or because the session
# is watched, or held until the user asks for it. One summary is made whichever it is; only who hears it when differs.
Delivery = Literal["summaries", "watched", "on request"]
