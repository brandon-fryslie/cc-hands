"""What one tool call did, recognised as the kind of step it is.

[LAW:parse-dont-validate] this is where a turn stops being JSON: a call and its result go in, a typed
`Step` comes out, and nothing downstream reads a transcript field again. A recogniser claims a call only
when the record carries what its step asserts — a failed edit has no patch, so it is not an `Edited` —
which is why failure needs no case of its own anywhere below.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from hands.core.testrun import report_of
from hands.core.turn import (
    Branched,
    Committed,
    Delegated,
    Edited,
    GitChange,
    Looked,
    Other,
    Planned,
    PullRequested,
    Pushed,
    Question,
    Questioned,
    Ran,
    Ref,
    Step,
    Tested,
)


@dataclass(frozen=True)
class Result:
    """What came back from a call: the text the model was shown, the record Claude Code wrote beside it when it
    wrote one, and whether the call ended in an error. The text is always there; the record is absent on every
    error and on results the harness handles itself, so no recogniser may require it."""

    text: str
    structured: Mapping[str, object] | None
    failed: bool


@dataclass(frozen=True)
class Call:
    """A tool call, named by the record it was written in, and the result it got back if one has been written yet."""

    ref: Ref | None
    tool: str
    input: Mapping[str, object]
    result: Result | None


Recogniser = Callable[[Call], Step | None]


def recognise(call: Call) -> Step:
    """The step this call is. A tool no recogniser claims is still named and summarised [LAW:no-silent-failure]."""
    recogniser = _RECOGNISERS.get(call.tool)
    step = None if recogniser is None else recogniser(call)
    return _other(call) if step is None else step


def _ran(call: Call) -> Step | None:
    command = _text(call.input.get("command"))
    if command is None:
        return None
    printed = _printed(call.result)
    failed = call.result is not None and call.result.failed
    structured = _structured(call)
    # [LAW:one-source-of-truth] whether a command touched the repository is what the record says, never what this
    # version can name of what it says: an operation written in a shape nobody here reads yet is still a change.
    operations = _mapping(structured.get("gitOperation")) if structured is not None else None
    report = report_of(printed)
    # A test run's result is its counts and its failing names; its scrollback is the least of it. It is claimed
    # only where those counts are the whole story: a command that failed while nothing is counted failing, or one
    # that changed the repository on its way, did more than run a suite, and only its own output says what.
    if report is not None and operations is None and not (failed and report.failed == 0):
        return Tested(call.ref, report.runner, report.passed, report.failed, report.failing)
    return Ran(call.ref, command, _text(call.input.get("description")), failed, printed, _changes(operations))


def _edited(call: Call) -> Step | None:
    structured = _structured(call)
    if structured is None:
        return None
    path = _text(structured.get("filePath"))
    created = structured.get("type") == "create"
    # Claude Code records an empty hunk list for a file written from nothing and puts its text in `content`,
    # so a create is read from what the file now holds rather than from a patch that was never written.
    written = structured.get("content")
    change = written if created and isinstance(written, str) else _patch(structured.get("structuredPatch"))
    if path is None or change is None:
        return None
    return Edited(call.ref, path, created, change)


# What a read, a search or a fetch was pointed at, whichever of these the tool calls it.
_TARGETS = ("file_path", "notebook_path", "pattern", "url", "query", "path")


def _looked(call: Call) -> Step | None:
    target = next((text for key in _TARGETS if (text := _text(call.input.get(key))) is not None), None)
    if target is None:
        return None
    return Looked(call.ref, call.tool, target, _printed(call.result))


def _planned(call: Call) -> Step | None:
    structured = _structured(call)
    if structured is None:
        return None
    task = _mapping(structured.get("task"))
    named = _text(task.get("subject")) if task is not None else None
    status = _mapping(structured.get("statusChange"))
    change = _text(status.get("to")) if status is not None else None
    identifier = _text(structured.get("taskId")) or (_text(task.get("id")) if task is not None else None)
    if named is None and identifier is None:
        return None
    return Planned(call.ref, named or f"task {identifier}", change or ("created" if task is not None else "changed"))


def _delegated(call: Call) -> Step | None:
    description = _text(call.input.get("description"))
    if description is None:
        return None
    structured = _structured(call)
    # An agent launched to run in the background reports back later as a notification, which opens a turn of its own.
    launched = structured is not None and structured.get("isAsync") is True
    report = None if launched or call.result is None else call.result.text
    return Delegated(call.ref, _text(call.input.get("subagent_type")), description, report)


def _questioned(call: Call) -> Step | None:
    asked = _sequence(call.input.get("questions"))
    if asked is None:
        return None
    structured = _structured(call)
    answers = _mapping(structured.get("answers")) if structured is not None else None
    questions = tuple(question for block in asked if (question := _question(block, answers)) is not None)
    return Questioned(call.ref, questions) if questions else None


def _question(block: object, answers: Mapping[str, object] | None) -> Question | None:
    fields = _mapping(block)
    text = _text(fields.get("question")) if fields is not None else None
    if fields is None or text is None:
        return None
    offered = _sequence(fields.get("options")) or ()
    labels = tuple(label for option in offered if (label := _label(option)) is not None)
    return Question(text, labels, _text(answers.get(text)) if answers is not None else None)


def _label(option: object) -> str | None:
    fields = _mapping(option)
    return None if fields is None else _text(fields.get("label"))


def _other(call: Call) -> Other:
    shown = {key: value for key, value in call.input.items() if key != "description"}
    match shown:
        case {"command": str() as command} if len(shown) == 1:
            text = command
        case _:
            text = json.dumps(shown, ensure_ascii=False)
    purpose = _text(call.input.get("description"))
    return Other(
        ref=call.ref,
        tool=call.tool if purpose is None else f"{call.tool} ({purpose})",
        input=text,
        result=_printed(call.result),
        failed=call.result is not None and call.result.failed,
    )


def _printed(result: Result | None) -> str:
    """What the call put on the screen: a command's streams where the record has them, and the model's own text otherwise."""
    if result is None:
        # The call has no result yet, or the turn stopped before one was written.
        return "(no result)"
    structured = result.structured
    if structured is not None and ("stdout" in structured or "stderr" in structured):
        printed = "\n".join(part for key in ("stdout", "stderr") if (part := _text(structured.get(key))) is not None)
        return printed if printed.strip() else "(no output)"
    return result.text


def _changes(operations: Mapping[str, object] | None) -> tuple[GitChange, ...]:
    """Of what the record says the command did to the repository, the operations this version can name."""
    if operations is None:
        return ()
    return tuple(change for name, value in operations.items() if (change := _change(name, _mapping(value))) is not None)


def _change(name: str, operation: Mapping[str, object] | None) -> GitChange | None:
    # The verb is the operation's own `kind` or `action` where it carries one, so a word this version does
    # not know is still spoken as what Claude Code called it.
    match name, operation:
        case "commit", {"sha": str() as sha}:
            return Committed(sha, _text(operation.get("kind")) or "committed")
        case "push", {"branch": str() as branch}:
            return Pushed(branch)
        case "branch", {"ref": str() as ref}:
            return Branched(ref, _text(operation.get("action")) or "moved to")
        case "pr", {"number": int() as number, "url": str() as url}:
            return PullRequested(number, url, _text(operation.get("action")) or "opened")
        case _:
            return None


def _patch(value: object) -> str | None:
    """The result's hunks written back as a diff, which is the form a model reads them in."""
    hunks = _sequence(value)
    if hunks is None:
        return None
    written: list[str] = []
    for hunk in hunks:
        fields = _mapping(hunk)
        lines = _sequence(fields.get("lines")) if fields is not None else None
        if fields is None or lines is None:
            continue
        head = f"@@ -{_number(fields.get('oldStart'))},{_number(fields.get('oldLines'))} +{_number(fields.get('newStart'))},{_number(fields.get('newLines'))} @@"
        written.append("\n".join([head, *(line for line in lines if isinstance(line, str))]))
    return "\n".join(written) if written else None


def _number(value: object) -> int:
    return value if isinstance(value, int) else 0


def _structured(call: Call) -> Mapping[str, object] | None:
    return call.result.structured if call.result is not None else None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _mapping(value: object) -> Mapping[str, object] | None:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else None


def _sequence(value: object) -> Sequence[object] | None:
    return cast(Sequence[object], value) if isinstance(value, Sequence) and not isinstance(value, str) else None


# [LAW:one-source-of-truth] the one place a tool name decides anything. Names read out of 900 transcripts:
# this Claude Code plans with the task tools and delegates with Agent, and has no TodoWrite or Task at all.
_RECOGNISERS: Mapping[str, Recogniser] = {
    "Bash": _ran,
    "Edit": _edited,
    "Write": _edited,
    "MultiEdit": _edited,
    "NotebookEdit": _edited,
    "Read": _looked,
    "Grep": _looked,
    "Glob": _looked,
    "WebFetch": _looked,
    "WebSearch": _looked,
    "TaskCreate": _planned,
    "TaskUpdate": _planned,
    "Agent": _delegated,
    "AskUserQuestion": _questioned,
}
