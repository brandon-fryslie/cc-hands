"""A turn as the summariser reads it: what opened it, and each thing Claude said or did, in order."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NewType

from hands.core.delta import Changed, Delta

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

    # pytest collects any class whose name starts with Test, and this is a step, not a test case. Said here,
    # about the one name that collides, rather than by turning class collection off for the whole repository.
    __test__ = False

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

    ref: Ref | None
    text: str


@dataclass(frozen=True)
class Notified:
    """A background task's notification, which Claude Code hands the session as its next prompt."""

    ref: Ref | None
    text: str


# [LAW:types-are-the-program] who opened the turn is a variant, so a notification is never reported as something the user asked.
Opening = Asked | Notified

# One thing that happened in a session: what opened a turn, or a step of the answer to it. A Turn holds the two
# apart because it is summarised as a whole, against its request. A reading of a session that nobody was
# listening to has no whole to summarise, and hands them over in the one order they make sense in — the order
# they happened [LAW:one-source-of-truth]. Both are put in words by the same function, so a request cannot be
# described one way in a turn and another in a reading.
Happening = Opening | Step


@dataclass(frozen=True)
class Answering:
    """This telling is the first of its turn, so its opening is the request the steps below answer."""


@dataclass(frozen=True)
class Continuing:
    """A later telling of a turn told once already, because another hook blocked its first Stop.

    `told` is how many steps went out in the earlier tellings. Carried so the opening can be shown as context
    instead of as the request: a small model handed the opening twice answers it twice, which was heard live on
    2026-09-21 as a second summary restating the first half of the turn.
    """

    told: int


# [LAW:types-are-the-program] whether the opening is still the question being answered is a fact about the
# telling, not a flag beside it, so the two readings of one opening are variants the render must both handle.
Standing = Answering | Continuing


@dataclass(frozen=True)
class Turn:
    """The part of a turn that is to be told now: its opening, the steps not yet told, and which telling this is."""

    opening: Opening
    steps: tuple[Step, ...]
    standing: Standing = Answering()


@dataclass(frozen=True)
class Budget:
    """How much of a turn the summariser is shown. Values, so the length that works is found by changing numbers."""

    opening: int  # characters of the prompt or notification that opened the turn
    said: int  # characters of each text block
    input: int  # characters of each tool input, command, or target
    result: int  # characters of each tool result, patch, or output
    steps: int  # steps shown; the middle of a longer turn is left out, keeping how it started and how it ended
    files: int  # files of the turn's delta named one by one; a formatter touches hundreds, where the count is the story
    commits: int  # commits of the turn's delta named one by one; a pull or a rebase brings hundreds, and the count is again the story
    changes: int  # characters of the patch between where the turn began and where it ended


CUT = " ... (cut)"


def render(turn: Turn, delta: Delta, budget: Budget) -> str:
    """The turn as the summariser's user message: what was asked, what was done, and what the repository says.

    The delta is given beside the steps rather than folded into them, because it is the result of all of them
    together and belongs to no one step — and because the file a `sed` changed has no step to belong to.
    """
    return "\n\n".join([_opening(turn, budget), body(turn.steps, delta, budget)])


def body(happenings: Sequence[Happening], delta: Delta, budget: Budget) -> str:
    """What happened, in order and cut to the budget, and what the repository says after it.

    Shared by the whole turn the summariser is shown and by the slice one segment of the narration opens into,
    so a segment can never be described by rules the turn it was cut from was not [LAW:one-source-of-truth].
    """
    return "\n\n".join([*_shown(happenings, budget), *_changed(delta, budget)])


def _shown(happenings: Sequence[Happening], budget: Budget) -> list[str]:
    """Each happening in words, where a run longer than the budget keeps how it started and how it ended."""
    if len(happenings) <= budget.steps:
        return [describe(happening, budget) for happening in happenings]
    head = (budget.steps + 1) // 2
    tail = budget.steps - head
    return [
        *(describe(happening, budget) for happening in happenings[:head]),
        f"({len(happenings) - budget.steps} steps in the middle are left out)",
        *(describe(happening, budget) for happening in happenings[len(happenings) - tail :]),
    ]


def _opening(turn: Turn, budget: Budget) -> str:
    """The opening, as the request these steps answer or as context for a turn that was reported once already.

    A continuing telling says so in the words around the opening rather than dropping it: the steps below only
    make sense against what was asked, and a model shown the request alone answers it instead of them.
    """
    said = describe(turn.opening, budget)
    match turn.standing:
        case Answering():
            return said
        case Continuing(told=told):
            before = "its first step" if told == 1 else f"its first {told} steps"
            return (
                f"This turn has already been reported once, up to and including {before}, and none of that"
                f" may be reported again. For context only, this is what opened it:\n{said}\n"
                "Report only what it did after that, below."
            )


def _changed(delta: Delta, budget: Budget) -> list[str]:
    """What the repository says the turn did, said after the steps because it is what they came to."""
    if not delta:
        return []
    told: list[str] = []
    if delta.files:
        named = [f"  {_counted(file)}" for file in delta.files[: budget.files]]
        if len(delta.files) > budget.files:
            named.append(f"  (and {len(delta.files) - budget.files} more files)")
        told.append("The repository is different, whether or not a step above says so:\n" + "\n".join(named))
    if delta.commits:
        made = [f"  {commit.sha} {commit.subject}" for commit in delta.commits[: budget.commits]]
        if len(delta.commits) > budget.commits:
            made.append(f"  (and {len(delta.commits) - budget.commits} more commits)")
        told.append(f"It made {len(delta.commits)} commit{'' if len(delta.commits) == 1 else 's'}:\n" + "\n".join(made))
    if delta.patch:
        told.append(f"What changed:\n{_cut(delta.patch, budget.changes)}")
    return told


def _counted(file: Changed) -> str:
    """A file and its size of change; git counts no lines for a file it reads as binary."""
    if file.added is None or file.removed is None:
        return f"{file.path} (binary)"
    return f"{file.path} +{file.added} -{file.removed}"


def describe(happening: Happening, budget: Budget) -> str:
    """One thing that happened, in words, cut to its budget. What a turn is rendered out of, and what a session is read back as."""
    match happening:
        case Asked(text=text):
            return f"The user asked:\n{_cut(text, budget.opening)}"
        case Notified(text=text):
            return f"A background task reported:\n{_cut(text, budget.opening)}"
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
            # A suite that fails whole names hundreds of tests, so the names are one more part cut to its budget.
            named = "".join(f"\n  {line}" for line in _cut("\n".join(failing), budget.result).splitlines())
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
