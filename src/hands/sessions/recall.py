"""What was said, sent, and answered, read back out of the audit log, for `hands recall` to put in front of the brain.

The log is the long memory [LAW:one-source-of-truth]: the user's words are its Transcribed lines, hands' words its
Replied lines, what was typed into a session its Typing lines (and a TypingFailed after one that never arrived), and how
a permission was answered the Reply that settled a PermissionRequested. Nothing here keeps a record of its own; it folds
the log's lines into moments, oldest first.
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import cast

from hands.sessions.audit import forwards

# How much of a permission's tool input a moment shows: enough to know the command, never a whole file being written.
INPUT_CHARS = 300


@dataclass(frozen=True)
class Moment:
    """One thing said, sent, or answered: when, who said it or where it went ("user", "you", "sent to billing",
    "allowed in billing"), and what."""

    at: datetime
    heading: str
    text: str


@dataclass(frozen=True)
class Recalled:
    """The newest moments that hold every word asked for, oldest first, and what the reading took: the log's lines, those
    that were not JSON, the time of the oldest line read (None for an empty log), the moments they held, and how many of
    those held every word."""

    moments: tuple[Moment, ...]
    lines: int
    unreadable: int
    since: datetime | None
    found: int
    matched: int


@dataclass(frozen=True)
class _Said:
    """A moment before its session is named: `verb` and `session` make its heading, or `verb` alone for none."""

    at: datetime
    verb: str
    session: str | None
    text: str


def recall(directory: Path, words: Iterable[str], most: int) -> Recalled:
    """The `most` newest moments in the log at `directory` whose heading or text holds every one of `words`, in any case."""
    wanted = tuple(word.casefold() for word in words)
    reading = _Reading()
    found = list(moments(reading.entries(forwards(directory))))
    matched = [moment for moment in found if all(word in f"{moment.heading} {moment.text}".casefold() for word in wanted)]
    return Recalled(tuple(matched[-most:] if most > 0 else ()), reading.lines, reading.unreadable, reading.since, len(found), len(matched))


class _Reading:
    """The log's lines as entries, one at a time, counting the lines, the ones that were not JSON, and the oldest time."""

    def __init__(self) -> None:
        self.lines = 0
        self.unreadable = 0
        self.since: datetime | None = None

    def entries(self, lines: Iterable[str]) -> Iterable[Mapping[str, object]]:
        for line in lines:
            self.lines += 1
            try:
                entry = cast(Mapping[str, object], json.loads(line))
            except json.JSONDecodeError:
                # The fragment a write that failed partway leaves, ended by the next line.
                self.unreadable += 1
                continue
            if self.since is None and isinstance(at := entry.get("at"), str):
                self.since = datetime.fromisoformat(at)
            yield entry


def moments(entries: Iterable[Mapping[str, object]]) -> list[Moment]:
    """The moments in a run of the log's lines, oldest first, each session under the last name the run gave it."""
    names: dict[str, str] = {}
    asked: dict[str, str] = {}
    said: list[_Said] = []
    # Where each send not yet seen to fail sits in `said`, by the Typing line's effect, which a TypingFailed repeats.
    sending: dict[str, int] = {}
    for entry in entries:
        match entry:
            # A session goes by the title hands gave it, the one it was found with, or the user's own /rename.
            case (
                {"type": "NameGiven", "session": str(session), "name": str(name)}
                | {"type": "Named", "session": str(session), "before": str(name)}
                | {"type": "NameWithheld", "session": str(session), "held": str(name)}
            ):
                names[session] = name
            case {"type": "Transcribed", "at": str(at), "text": str(text)}:
                said.append(_Said(datetime.fromisoformat(at), "user", None, text))
            case {"type": "Replied", "at": str(at), "text": str(text)}:
                said.append(_Said(datetime.fromisoformat(at), "you", None, text))
            case {"type": "Typing", "at": str(at), "effect": {"session": str(session), "input": object() as typed} as effect}:
                sending[json.dumps(effect, sort_keys=True)] = len(said)
                said.append(_Said(datetime.fromisoformat(at), "sent to", session, _typed(typed)))
            case {"type": "TypingFailed", "effect": object() as effect, "reason": str(reason)}:
                if (index := sending.pop(json.dumps(effect, sort_keys=True), None)) is not None:
                    said[index] = replace(said[index], verb="not sent to", text=f"{said[index].text}, because: {reason}")
            case {"type": "Applied", "event": {"type": "PermissionRequested", "request": str(request), "on": object() as on}}:
                asked[request] = _blocker(on)
            # A reply to a hook no request was seen for, as a Stop's is, answers no permission.
            case {"type": "Performed", "at": str(at), "effect": {"type": "Reply", "session": str(session), "request": str(request), "reply": object() as reply}} if request in asked:
                if (answer := _answer(reply)) is not None:
                    verb, told = answer
                    said.append(_Said(datetime.fromisoformat(at), f"{verb} in", session, f"{asked.pop(request)}{told}"))
            case _:
                pass
    return [Moment(each.at, each.verb if each.session is None else f"{each.verb} {names.get(each.session, each.session)}", each.text) for each in said]


def _typed(typed: object) -> str:
    match typed:
        case {"type": "Text", "prompt": str(prompt)}:
            return prompt
        case {"type": "Command", "name": str(name), "args": str(args)}:
            return f"/{name} {args}"
        case {"type": "Command", "name": str(name)}:
            return f"/{name}"
        case {"type": "Key", "key": str(key)}:
            return f"the {key} key"
        case _:
            raise ValueError(f"a Typing line holds an input recall cannot read: {typed}")


def _blocker(on: object) -> str:
    match on:
        case {"type": "Permission", "tool": str(tool), "input": object() as tool_input}:
            return f"{tool} {json.dumps(tool_input, ensure_ascii=False)[:INPUT_CHARS]}"
        case {"type": "Question", "asked": object() as asked}:
            return " ".join(str(question["question"]) for question in cast(list[Mapping[str, object]], asked))
        case {"type": "Plan", "text": str(text)}:
            return f"the plan: {text[:INPUT_CHARS]}"
        case _:
            raise ValueError(f"a PermissionRequested line holds a request recall cannot read: {on}")


def _answer(reply: object) -> tuple[str, str] | None:
    """How a reply to a permission request is said, and what it told the session; None for a Withdraw, which decided
    nothing: it left the request to the session's own dialog."""
    match reply:
        case {"type": "Allow"}:
            return "allowed", ""
        case {"type": "Approve"}:
            return "approved", ""
        # Only a question is answered with input, and what the user chose is its answers, one a question asked.
        case {"type": "AllowWith", "input": {"answers": object() as answers}}:
            return "answered", f", chose: {'; '.join(str(answer) for answer in cast(dict[str, object], answers).values())}"
        case {"type": "Deny", "message": str(message)}:
            return "denied", f", told: {message}"
        case {"type": "Withdraw"}:
            return None
        case _:
            raise ValueError(f"a Reply line holds a reply recall cannot read: {reply}")
