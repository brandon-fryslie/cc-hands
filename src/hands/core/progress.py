"""Progress while a session works: each tool call as it is made, the text Claude writes as it writes it, and a burst
of them said as one sentence.

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
    # One call of it whose input does not say what it was done to.
    one: str


EDITING = Work("edit {count} files", "edit a file")
READING = Work("read {count} files", "read a file")
SEARCHING = Work("run {count} searches", "search the code")
LOOKING_UP = Work("make {count} web lookups", "look something up on the web")
RUNNING = Work("run {count} commands", "run a command")
DELEGATING = Work("start {count} subagents", "start a subagent")
PLANNING = Work("update its plan", "update its plan")
USING = Work("use {count} tools", "use a tool")
# Text Claude wrote, said by a summary of it, which every burst of it has: alone, it is never said as this work's.
EXPLAINING = Work("explain {count} things", "explain something")
# Text Claude wrote that could not be summarised: said to have been written, since what it says is not known, and never
# read out.
WRITING = Work("write {count} things", "write something")


@dataclass(frozen=True)
class Doing:
    """One call as it is made: its kind of work, and what it is said as when it is the only one of its kind; None when
    its input does not say what it was done to, and then no other call is the same thing done."""

    work: Work
    alone: str | None


def doing(tool: str, input: Mapping[str, object]) -> Doing | None:
    """What a call of `tool` with this input sets out to do; None for a call that asks the user, which is spoken as it
    is asked and needs no progress besides."""
    return _SAID.get(tool, _used)(tool, input)


def said(doings: Sequence[Doing]) -> str:
    """A burst of calls as one spoken clause, each kind of work once, in the order it was first done: "edit ten files,
    then run the test suite". A kind done to one thing only is said as that thing."""
    # Ordered and deduplicated in one step: ten edits of one file are one thing done, and eight commands that say
    # nothing of themselves are eight, each its own by where it stands.
    grouped: dict[Work, dict[str | int, str]] = {}
    for at, each in enumerate(doings):
        grouped.setdefault(each.work, {})[at if each.alone is None else each.alone] = each.work.one if each.alone is None else each.alone
    return ", then ".join(
        next(iter(things.values())) if len(things) == 1 else work.several.format(count=spoken_count(len(things))) for work, things in grouped.items()
    )


def _ran(_tool: str, input: Mapping[str, object]) -> Doing:
    # Claude Code asks for a description of every command, in the imperative ("Run the test suite"), which is what
    # this sentence is made of; the command itself is code, and never read aloud.
    description = _text(input, "description")
    return Doing(RUNNING, None if description is None else lowered(description))


def _edited(_tool: str, input: Mapping[str, object]) -> Doing:
    file = _file(input)
    return Doing(EDITING, None if file is None else f"edit {file}")


def _read(_tool: str, input: Mapping[str, object]) -> Doing:
    file = _file(input)
    return Doing(READING, None if file is None else f"read {file}")


def _searched(_tool: str, input: Mapping[str, object]) -> Doing:
    # A pattern is said only when it is words: a regular expression or a glob read out is code, and never heard as such.
    pattern = _text(input, "pattern")
    return Doing(SEARCHING, f"search for {pattern}" if pattern is not None and _WORDS.fullmatch(pattern) else None)


def _looked_up(_tool: str, input: Mapping[str, object]) -> Doing:
    query = _text(input, "query")
    return Doing(LOOKING_UP, None if query is None else f"search the web for {query}")


def _fetched(_tool: str, input: Mapping[str, object]) -> Doing:
    # A page by its site: a whole address read out is noise.
    host = _HOST.match(_text(input, "url") or "")
    return Doing(LOOKING_UP, None if host is None else f"read a page on {host[1]}")


def _delegated(_tool: str, input: Mapping[str, object]) -> Doing:
    description = _text(input, "description")
    return Doing(DELEGATING, None if description is None else f"start a subagent to {lowered(description)}")


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


# What a listener can follow: words, and the dashes, dots, and underscores names are spelled with.
_WORDS = re.compile(r"[\w][\w .'-]*")

# The host of a URL: what follows the scheme, up to its port, path, query, or fragment.
_HOST = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://(?:[^@/]*@)?([^/:?#]+)")


def _file(input: Mapping[str, object]) -> str | None:
    """The file a call works on, by its name: the directories it is in are not worth a listener's time."""
    path = _text(input, "file_path") or _text(input, "notebook_path")
    return None if path is None else PurePath(path).name


def _text(input: Mapping[str, object], key: str) -> str | None:
    value = input.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def explained(summary: str) -> Doing:
    """Text Claude wrote, said as its summary: a phrase in the imperative, as the middle of a sentence."""
    return Doing(EXPLAINING, lowered(summary.strip().rstrip(".!")))


def lowered(phrase: str) -> str:
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
    """Calls a running turn made and text it wrote that nobody has been told of yet, and when the first and the last of
    them came."""

    doings: tuple[Doing, ...]
    # Every line written since the burst began, as Claude Code displayed it; empty when it wrote none.
    written: str
    first: float
    last: float

    def joined(self, doings: Sequence[Doing], at: float) -> "Gathering":
        return Gathering((*self.doings, *doings), self.written, self.first, at)

    def wrote(self, text: str, at: float) -> "Gathering":
        return Gathering(self.doings, self.written + text, self.first, at)

    def due(self) -> float:
        """When the burst is told: once it has settled, and no later than its first call's longest wait."""
        return min(self.last + SETTLE, self.first + LONGEST)
