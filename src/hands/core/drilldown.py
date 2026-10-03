"""Drill-down: "more on that" opens a told turn's parts, one rung longer each time it is asked.

A rung is a way of telling a segment, and the ladder runs from the line it plays to longer and longer renderings of
the records it holds. The deepest rung is still a rendering for a model to put in its own words, never the records
as written: there is no verbatim rung to reach [LAW:types-are-the-program], so nothing asked for more is ever read out.
"""

from collections.abc import Callable
from dataclasses import dataclass

from hands.core.narration import Segment, opened
from hands.core.turn import Budget

# A way of telling one segment.
Rung = Callable[[Segment], str]

# Values, as every budget is, so the lengths that work are found by changing numbers. Each rung shows more of every step
# than the one above it, and more steps; none names the opening, which the headline already answered.
_STEPS = Budget(opening=0, said=300, input=100, result=150, steps=8, files=5, commits=3, changes=0)
_MORE = Budget(opening=0, said=800, input=200, result=400, steps=20, files=20, commits=10, changes=1500)
_MOST = Budget(opening=0, said=2000, input=400, result=1200, steps=60, files=60, commits=30, changes=6000)


def _line(segment: Segment) -> str:
    return segment.text


def _at(budget: Budget) -> Rung:
    return lambda segment: opened(segment, budget)


# The segment's own line first, which is what the top level of the tree says of it, then its records at each budget.
LADDER: tuple[Rung, ...] = (_line, *(_at(budget) for budget in (_STEPS, _MORE, _MOST)))


@dataclass(frozen=True)
class Drilled:
    """The parts opened, each by its topic's name, and whether asking again tells more of them."""

    told: tuple[tuple[str, str], ...]
    deeper: bool


def drill(parts: tuple[Segment, ...], depth: int) -> Drilled:
    """The parts told at rung `depth` of the ladder, or at its last for a depth past it: asked again at the bottom, the
    longest telling is told again rather than an error.

    `deeper` is whether the next rung tells any part differently, measured rather than read off the rung's number: a
    part with little in it is told whole at a short rung, and promising more of it would send the user asking for
    the same words again.
    """
    rung = min(depth, len(LADDER) - 1)
    below = min(rung + 1, len(LADDER) - 1)
    told = tuple((segment.topic.name, LADDER[rung](segment)) for segment in parts)
    return Drilled(told, any(LADDER[below](segment) != text for segment, (_, text) in zip(parts, told)))
