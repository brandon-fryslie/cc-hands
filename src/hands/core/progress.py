"""Progress while a session works: each tool call as it is made, and a burst of them said as one sentence.

A call is said by what it sets out to do, read off its input, and never by its result: the result is the turn's to
tell when it finishes, and a call is worth hearing about while it runs, before there is any result to tell.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePath

from hands.core.spoken import spoken_count


@dataclass(frozen=True)
class Work:
    """A kind of work a call does, and how several calls of it are said together.

    [LAW:one-type-per-behavior] the kinds differ only in their words, so they are values of one type, and a new kind
    is a new constant rather than new code.
    """

    # "{count}" stands for how many different things were done: "edit {count} files".
    several: str


EDITING = Work("edit {count} files")
READING = Work("read {count} files")
SEARCHING = Work("run {count} searches")
LOOKING_UP = Work("make {count} web lookups")
RUNNING = Work("run {count} commands")
DELEGATING = Work("start {count} subagents")
PLANNING = Work("update its plan")
USING = Work("use {count} tools")


@dataclass(frozen=True)
class Doing:
    """One call as it is made: its kind of work, and what it is said as when it is the only one of its kind."""

    work: Work
    alone: str


def doing(tool: str, input: Mapping[str, object]) -> Doing | None:
    """What a call of `tool` with this input sets out to do; None for a call that asks the user, which is spoken as it
    is asked and needs no progress besides."""
    return _SAID.get(tool, _used)(tool, input)


def said(doings: Sequence[Doing]) -> str:
    """A burst of calls as one spoken clause, each kind of work once, in the order it was first done: "edit ten files,
    then run the test suite". A kind done to one thing only is said as that thing."""
    # Ordered and deduplicated in one step: ten edits of one file are one thing done.
    grouped: dict[Work, list[str]] = {}
    for each in dict.fromkeys(doings):
        grouped.setdefault(each.work, []).append(each.alone)
    return ", then ".join(alone[0] if len(alone) == 1 else work.several.format(count=spoken_count(len(alone))) for work, alone in grouped.items())


def _ran(_tool: str, input: Mapping[str, object]) -> Doing:
    # Claude Code asks for a description of every command, in the imperative ("Run the test suite"), which is what
    # this sentence is made of; the command itself is code, and never read aloud.
    description = _text(input, "description")
    return Doing(RUNNING, "run a command" if description is None else _lowered(description))


def _edited(_tool: str, input: Mapping[str, object]) -> Doing:
    return Doing(EDITING, f"edit {_file(input)}")


def _read(_tool: str, input: Mapping[str, object]) -> Doing:
    return Doing(READING, f"read {_file(input)}")


def _searched(_tool: str, input: Mapping[str, object]) -> Doing:
    pattern = _text(input, "pattern")
    return Doing(SEARCHING, "search the code" if pattern is None else f"search for {pattern}")


def _looked_up(_tool: str, input: Mapping[str, object]) -> Doing:
    query = _text(input, "query")
    return Doing(LOOKING_UP, "look something up on the web" if query is None else f"search the web for {query}")


def _fetched(_tool: str, input: Mapping[str, object]) -> Doing:
    # A page by its site: a whole address read out is noise.
    host = _HOST.match(_text(input, "url") or "")
    return Doing(LOOKING_UP, "read a web page" if host is None else f"read a page on {host[1]}")


def _delegated(_tool: str, input: Mapping[str, object]) -> Doing:
    description = _text(input, "description")
    return Doing(DELEGATING, "start a subagent" if description is None else f"start a subagent to {_lowered(description)}")


def _planned(_tool: str, _input: Mapping[str, object]) -> Doing:
    return Doing(PLANNING, "update its plan")


def _asked(_tool: str, _input: Mapping[str, object]) -> None:
    return None


def _used(tool: str, _input: Mapping[str, object]) -> Doing:
    # An MCP tool is named mcp__<server>__<tool>: the tool's own name is the part a listener can follow.
    return Doing(USING, f"use {tool.rsplit('__', 1)[-1].replace('_', ' ')}")


# [LAW:one-type-per-behavior] one table keyed by tool name, as the step recognisers are; a tool not in it is used.
_SAID: Mapping[str, Callable[[str, Mapping[str, object]], Doing | None]] = {
    "Bash": _ran,
    "Edit": _edited,
    "MultiEdit": _edited,
    "Write": _edited,
    "NotebookEdit": _edited,
    "Read": _read,
    "Grep": _searched,
    "Glob": _searched,
    "WebSearch": _looked_up,
    "WebFetch": _fetched,
    "Task": _delegated,
    "Agent": _delegated,
    "TodoWrite": _planned,
    "TaskCreate": _planned,
    "TaskUpdate": _planned,
    # Each is a dialog its PermissionRequest hook has spoken already.
    "AskUserQuestion": _asked,
    "ExitPlanMode": _asked,
}


# The host of a URL: what follows the scheme, up to its port, path, query, or fragment.
_HOST = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://(?:[^@/]*@)?([^/:?#]+)")


def _file(input: Mapping[str, object]) -> str:
    """The file a call works on, by its name: the directories it is in are not worth a listener's time."""
    path = _text(input, "file_path") or _text(input, "notebook_path")
    return "a file" if path is None else PurePath(path).name


def _text(input: Mapping[str, object], key: str) -> str | None:
    value = input.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _lowered(phrase: str) -> str:
    """An imperative phrase as the middle of a sentence: "Run the tests" is "run the tests", and "PR" stays "PR"."""
    first, rest = phrase[:1], phrase[1:]
    return first.lower() + rest if rest[:1].islower() or not rest else phrase


# [LAW:no-ambient-temporal-coupling] a burst is over once the session has made no new call for SETTLE seconds, or once
# its first call has waited LONGEST, so a session that never pauses is still heard. Both are read on the reducer's own
# clock, at its tick, which is the one owner of when gathered progress is told.
SETTLE = 2.0
LONGEST = 8.0


@dataclass(frozen=True)
class Gathering:
    """Calls a running turn made that nobody has been told of yet, and when the first and the last of them were read."""

    doings: tuple[Doing, ...]
    first: float
    last: float

    def joined(self, doings: Sequence[Doing], at: float) -> "Gathering":
        return Gathering((*self.doings, *doings), self.first, at)

    def due(self) -> float:
        """When the burst is told: once it has settled, and no later than its first call's longest wait."""
        return min(self.last + SETTLE, self.first + LONGEST)
