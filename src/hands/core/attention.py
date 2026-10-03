"""What reaches the ear unasked: what the user set hands to say of its own accord, each session's overlay, and the
tables that route a finished turn, a session's progress, and a session ending by them.

[LAW:one-source-of-truth] one control: every kind of thing hands says unprompted has its level here, and quiet is one
setting of the same value, never a switch beside it. What a session asks (a permission, a question, a plan) is no kind
here: it needs an answer, and one held unsaid would wait out its deadline and be refused, so it is spoken whatever is set.
"""

from dataclasses import dataclass
from typing import Literal

# How the user asked to hear a session's finished turns. "watched": each turn it finishes is told as it finishes.
# "normal": its turns are told as finished turns are set to be. "muted": told only when the user asks.
Overlay = Literal["normal", "watched", "muted"]


# Not told until asked for: with several sessions, every one that stopped said aloud was noise (hands-announce-5md).
DEFAULT: Overlay = "normal"

# How much of a thing told unasked is said: "brief", the gist of it; "full", all of it worth hearing.
Amount = Literal["brief", "full"]

# A level: "off" is said only when asked for; otherwise how much of it is said, an Amount. Spelled flat, so its levels
# read off it as one set.
Level = Literal["off", "brief", "full"]

# Whether something is said as it happens: "on" or "off".
Switch = Literal["on", "off"]


@dataclass(frozen=True)
class Attention:
    """What hands says unprompted, by kind: each turn a session finishes, the focused session's progress as it works,
    and a session ending. Quiet holds all of it, whatever its level, until the user lets hands talk again, and leaves
    the levels as they were set, so talking again is back to them.

    The defaults are how hands spoke before any of it was set: every finished turn told was too much to sit through with
    several sessions (the user's decision, 2026-09-26), and the focused session is heard step by step.
    """

    finished: Level = "off"
    progress: Level = "full"
    ended: Switch = "on"
    quiet: Switch = "off"


# What decided how a finished turn reaches the user. Told as it finishes, how much of it, and why: finished turns are
# set to be told, or the session is watched. Withheld until the user asks for it (tell_turn, catch_up), and why: finished
# turns are off, the session is muted, or hands is quiet. One summary is made whichever it is; only who hears it when differs.
@dataclass(frozen=True)
class Spoken:
    amount: Amount
    why: Literal["finished", "watched"]


@dataclass(frozen=True)
class Withheld:
    why: Literal["off", "muted", "quiet"]


Delivery = Spoken | Withheld


def delivery(attention: Attention, overlay: Overlay) -> Delivery:
    """How a finished turn reaches the user: never unasked from a muted session or while quiet; a watched session's as
    finished turns are set to be told, and all of it when they are off; any other's as they are set."""
    # [LAW:dataflow-not-control-flow] a table over the settings, every combination a row the type checker holds to.
    match attention.quiet, overlay, attention.finished:
        case _, "muted", _:
            return Withheld("muted")
        case "on", _, _:
            return Withheld("quiet")
        case "off", "watched", "off":
            return Spoken("full", "watched")
        case "off", "watched", ("brief" | "full") as amount:
            return Spoken(amount, "watched")
        case "off", "normal", "off":
            return Withheld("off")
        case "off", "normal", ("brief" | "full") as amount:
            return Spoken(amount, "finished")


# How a session's progress, or its ending, reaches the user: said as it happens, and how much of it; or "note", left to
# the session listing and the audit log, which the model reads when asked and says nothing of until then.
Route = Amount | Literal["note"]


def progress_route(attention: Attention, focused: bool, overlay: Overlay) -> Route:
    """The focused session is heard working, as its progress is set to be; any other is noted, and a muted one is noted
    even when focused, as all of it is while quiet."""
    # [LAW:dataflow-not-control-flow] a table over the settings, the focus and the overlay, every row type-checked.
    match attention.quiet, overlay, focused, attention.progress:
        case "on", _, _, _:
            return "note"
        case "off", "muted", _, _:
            return "note"
        case "off", "normal" | "watched", False, _:
            return "note"
        case "off", "normal" | "watched", True, "off":
            return "note"
        case "off", "normal" | "watched", True, ("brief" | "full") as amount:
            return amount


def ended_route(attention: Attention) -> Route:
    """A session ending is said, unless that is off or hands is quiet; it is in the audit log either way, for catch_up."""
    match attention.quiet, attention.ended:
        case "off", "on":
            return "full"
        case "on", _:
            return "note"
        case _, "off":
            return "note"
