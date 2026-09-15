"""A finished turn as the summariser reads it: what the user asked, and each thing Claude said or used, in order."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Said:
    """Assistant text, as Claude wrote it for a screen."""

    text: str


@dataclass(frozen=True)
class Used:
    """A tool call and the result it got back. `purpose` is the description a tool call carries, when it has one."""

    tool: str
    purpose: str | None
    input: str
    result: str
    failed: bool


# [LAW:types-are-the-program] tool calls are content, not noise: a turn's results are mostly in what it used.
Step = Said | Used


@dataclass(frozen=True)
class Asked:
    """The user's own prompt, typed or sent through the SDK."""

    text: str


@dataclass(frozen=True)
class Notified:
    """A background task's notification, which Claude Code hands the session as its next prompt."""

    text: str


# [LAW:types-are-the-program] who opened the turn is a variant, so a notification is never reported as something the user asked.
Opening = Asked | Notified


@dataclass(frozen=True)
class Turn:
    opening: Opening
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class Budget:
    """How much of a turn the summariser is shown. Values, so the length that works is found by changing numbers."""

    opening: int  # characters of the prompt or notification that opened the turn
    said: int  # characters of each text block
    input: int  # characters of each tool input
    result: int  # characters of each tool result
    steps: int  # steps shown; the middle of a longer turn is left out, keeping how it started and how it ended


CUT = " ... (cut)"


def render(turn: Turn, budget: Budget) -> str:
    """The turn as the summariser's user message."""
    head = (budget.steps + 1) // 2
    tail = budget.steps - head
    steps = turn.steps
    if len(steps) > budget.steps:
        shown = [_step(step, budget) for step in steps[:head]]
        shown.append(f"({len(steps) - budget.steps} steps in the middle are left out)")
        shown.extend(_step(step, budget) for step in steps[len(steps) - tail :])
    else:
        shown = [_step(step, budget) for step in steps]
    return "\n\n".join([_opening(turn.opening, budget), *shown])


def _opening(opening: Opening, budget: Budget) -> str:
    match opening:
        case Asked(text=text):
            return f"The user asked:\n{_cut(text, budget.opening)}"
        case Notified(text=text):
            return f"A background task reported:\n{_cut(text, budget.opening)}"


def _step(step: Step, budget: Budget) -> str:
    match step:
        case Said(text=text):
            return f"Claude said:\n{_cut(text, budget.said)}"
        case Used(tool=tool, purpose=purpose, input=input, result=result, failed=failed):
            used = tool if purpose is None else f"{tool} ({purpose})"
            outcome = "Result (failed)" if failed else "Result"
            return f"Claude used {used}: {_cut(input, budget.input)}\n{outcome}: {_cut(result, budget.result)}"


def _cut(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + CUT
