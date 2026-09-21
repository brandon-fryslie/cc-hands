"""A turn as the summariser reads it: what opened it, and each thing Claude said or did, in order."""

from dataclasses import dataclass
from typing import NewType

# The uuid of the transcript record a step came from, so a spoken segment can name what it summarised.
# A step names no record in the two cases where there is none to name: a record that carries no uuid, and
# the closing reply a Stop hook hands over in the milliseconds before Claude Code writes its record.
Ref = NewType("Ref", str)


@dataclass(frozen=True)
class Committed:
    sha: str
    kind: str  # what Claude Code calls the commit it saw: `committed`, and whatever else it writes


@dataclass(frozen=True)
class Pushed:
    branch: str


@dataclass(frozen=True)
class Branched:
    ref: str
    action: str


@dataclass(frozen=True)
class PullRequested:
    number: int
    url: str
    action: str


# What a command did to the repository, as the result's `gitOperation` records it. One command can do several:
# 70 records in a 900-transcript sample commit and push, and 30 open a pull request and push [LAW:types-are-the-program].
GitChange = Committed | Pushed | Branched | PullRequested


@dataclass(frozen=True)
class Said:
    """Assistant text, as Claude wrote it for a screen."""

    ref: Ref | None
    text: str


@dataclass(frozen=True)
class Edited:
    """A file Claude changed: the hunks its result recorded, or the whole of what the file now holds for one
    Claude wrote from nothing, which has no hunks because there was nothing to diff it against."""

    ref: Ref | None
    path: str
    created: bool
    change: str


@dataclass(frozen=True)
class Ran:
    """A command, what it printed, whether it exited non-zero, and what it did to the repository."""

    ref: Ref | None
    command: str
    purpose: str | None
    failed: bool
    output: str
    git: tuple[GitChange, ...]


@dataclass(frozen=True)
class Tested:
    """A command whose output a test runner wrote. `passed` is None for a runner that does not count what passed."""

    ref: Ref | None
    runner: str
    passed: int | None
    failed: int
    failing: tuple[str, ...]


@dataclass(frozen=True)
class Looked:
    """A read, a search, or a fetch: what was asked for, and what came back."""

    ref: Ref | None
    tool: str
    target: str
    found: str


@dataclass(frozen=True)
class Planned:
    """One task of Claude's own plan, and what happened to it. `change` is the status Claude Code wrote, not a set hands fixes."""

    ref: Ref | None
    task: str
    change: str


@dataclass(frozen=True)
class Delegated:
    """A subagent Claude dispatched. `report` is None while it is still running: an async agent reports back as a notification, which opens a turn of its own."""

    ref: Ref | None
    agent: str | None
    description: str
    report: str | None


@dataclass(frozen=True)
class Question:
    question: str
    options: tuple[str, ...]
    answer: str | None


@dataclass(frozen=True)
class Questioned:
    """A question Claude put to the user through AskUserQuestion, with the answer when one came back."""

    ref: Ref | None
    questions: tuple[Question, ...]


@dataclass(frozen=True)
class Other:
    """A tool no recogniser claims: named and summarised from its input and result, never dropped [LAW:no-silent-failure]."""

    ref: Ref | None
    tool: str
    input: str
    result: str
    failed: bool


# [LAW:types-are-the-program] tool calls are content, not noise: a turn's results are mostly in what it used, and
# one variant per kind of result is what lets the summariser be shown a test run's counts instead of its scrollback.
Step = Said | Edited | Ran | Tested | Looked | Planned | Delegated | Questioned | Other


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
    input: int  # characters of each tool input, command, or target
    result: int  # characters of each tool result, patch, or output
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
        case Edited(path=path, created=created, change=change):
            return f"Claude {'wrote' if created else 'edited'} {path}:\n{_cut(change, budget.result)}"
        case Ran(command=command, purpose=purpose, failed=failed, output=output, git=git):
            ran = f"Claude ran {_cut(command, budget.input)}"
            did = "".join(f"\n{_git(change)}" for change in git)
            return f"{ran}{'' if purpose is None else f' ({purpose})'}\n{'Output (exit code not zero)' if failed else 'Output'}: {_cut(output, budget.result)}{did}"
        case Tested(runner=runner, passed=passed, failed=failed, failing=failing):
            counted = f"{failed} failed" + ("" if passed is None else f", {passed} passed")
            named = "".join(f"\n  {name}" for name in failing)
            return f"Claude ran the {runner} tests: {counted}{named}"
        case Looked(tool=tool, target=target, found=found):
            return f"Claude used {tool} on {_cut(target, budget.input)}\nFound: {_cut(found, budget.result)}"
        case Planned(task=task, change=change):
            return f"Claude's plan: {_cut(task, budget.input)} is {change}"
        case Delegated(agent=agent, description=description, report=report):
            to = "a subagent" if agent is None else f"the {agent} subagent"
            return f"Claude gave {to} this job: {_cut(description, budget.input)}\n" + (
                "It is still working." if report is None else f"It reported: {_cut(report, budget.result)}"
            )
        case Questioned(questions=questions):
            return "\n".join(_question(question, budget) for question in questions)
        case Other(tool=tool, input=input, result=result, failed=failed):
            return f"Claude used {tool}: {_cut(input, budget.input)}\n{'Result (failed)' if failed else 'Result'}: {_cut(result, budget.result)}"


def _question(question: Question, budget: Budget) -> str:
    asked = f"Claude asked the user: {_cut(question.question, budget.input)}"
    offered = "" if not question.options else "\nOptions: " + ", ".join(question.options)
    return f"{asked}{offered}" + ("\nUnanswered." if question.answer is None else f"\nThe user chose: {question.answer}")


def _git(change: GitChange) -> str:
    match change:
        case Committed(sha=sha, kind=kind):
            return f"It {kind} {sha}."
        case Pushed(branch=branch):
            return f"It pushed {branch}."
        case Branched(ref=ref, action=action):
            return f"It {action} {ref}."
        case PullRequested(number=number, url=url, action=action):
            return f"It {action} pull request {number}, {url}."


def _cut(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + CUT
