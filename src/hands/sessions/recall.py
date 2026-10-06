"""What was said, sent, answered, and called, read back out of the audit log: for `hands recall` to put in front of the
brain, and for the conversation page to show (`hands.voice.conversationpage`).

The log is the long memory [LAW:one-source-of-truth]: the user's words are its Transcribed lines, hands' words its
Replied lines, what was typed into a session its Typing lines (and a TypingFailed after one that never arrived), how
a permission was answered the Reply an applied event performed for a PermissionRequested, and each tool hands called its
`tool.run` event. Nothing here keeps a record of its own; it folds the log's lines into moments, oldest first.
"""

import json
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from hands.core.session import identifier
from hands.sessions.audit import forwards

# How much of a permission's tool input a moment shows: enough to know the command, never a whole file being written.
INPUT_CHARS = 300


# What a moment is: the user's words, hands' words, words typed into a session, a session's request answered, or a tool
# hands called.
Kind = Literal["heard", "said", "sent", "answered", "called"]


@dataclass(frozen=True)
class Moment:
    """One thing said, sent, answered, or called: when, what kind, who said it or where it went ("user", "you",
    "sent to billing", "allowed in billing", "called set_trigger"), and what."""

    at: datetime
    kind: Kind
    heading: str
    text: str


@dataclass(frozen=True)
class Recalled:
    """The newest moments that hold every word asked for, oldest first, and what the reading took: the log's lines, those
    that were not JSON objects, the time of the oldest line read (None for an empty log), the moments they held, and how many of
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
    kind: Kind
    verb: str
    session: str | None
    text: str


def recall(directory: Path, words: Iterable[str], most: int) -> Recalled:
    """The `most` newest moments in the log at `directory` whose heading or text holds every one of `words`, in any case."""
    wanted = tuple(word.casefold() for word in words)
    reading = Reading()
    # What hands called is no part of what was said: its moments are the conversation page's.
    found = [moment for moment in moments(reading.entries(forwards(directory))) if moment.kind != "called"]
    matched = [moment for moment in found if all(word in f"{moment.heading} {moment.text}".casefold() for word in wanted)]
    return Recalled(tuple(matched[-most:] if most > 0 else ()), reading.lines, reading.unreadable, reading.since, len(found), len(matched))


class Reading:
    """The log's lines as entries, one at a time, counting the lines, the ones that were not JSON objects, and the oldest time."""

    def __init__(self) -> None:
        self.lines = 0
        self.unreadable = 0
        self.since: datetime | None = None

    def entries(self, lines: Iterable[str]) -> Iterable[Mapping[str, object]]:
        for line in lines:
            self.lines += 1
            try:
                parsed: object = json.loads(line)
            except json.JSONDecodeError:
                # The fragment a write that failed partway leaves, ended by the next line.
                parsed = None
            if not isinstance(parsed, dict):
                # Every line the log writes is an object: anything else is no line of its.
                self.unreadable += 1
                continue
            entry = cast(Mapping[str, object], parsed)
            if self.since is None and isinstance(at := entry.get("at"), str):
                self.since = datetime.fromisoformat(at)
            yield entry


def moments(entries: Iterable[Mapping[str, object]]) -> list[Moment]:
    """The moments in a run of the log's lines, oldest first."""
    folded = Moments()
    for entry in entries:
        folded.take(entry)
    return folded.moments()


class Moments:
    """The `kept` newest moments of the log's lines taken so far (every one, for None), oldest first, each session as
    hands speaks it, by its project and the last name the run gave it; by its id when the run never saw it join.

    A send seen to fail later is said again where it was, and a session named or seen to join is spoken anew wherever it
    went: `changes` counts every moment added and every such change, so a reader holding the moments as they were at one
    count knows them as they are while it stands.
    """

    def __init__(self, kept: int | None = None) -> None:
        self._projects: dict[str, Path] = {}
        self._names: dict[str, str] = {}
        self._asked: dict[str, str] = {}
        self._said: deque[_Said] = deque(maxlen=kept)
        # How many moments have been taken, so the oldest still kept is moment `taken - len(said)`.
        self._taken = 0
        # The moment each send still kept and not yet seen to fail is, by the Typing line's effect, which a TypingFailed
        # repeats; oldest first, as they were taken.
        self._sending: dict[str, int] = {}
        self.changes = 0

    def take(self, entry: Mapping[str, object]) -> None:
        """Fold one of the log's lines in."""
        match entry:
            # A session goes by the title hands gave it, the one it was found with, or the user's own /rename.
            case (
                {"type": "WideEvent", "event": "hook", "facts": {"session": str(session), "name": {"type": "NameGiven", "name": str(name)} | {"type": "NameWithheld", "held": str(name)}}}
                | {"type": "WideEvent", "event": "name.judged", "facts": {"session": str(session), "before": str(name)}}
            ):
                self._changed(self._names, session, name)
            case {"type": "WideEvent", "event": "applied", "at": str(at), "facts": {"applied": object() as applied, "effects": object() as effects}}:
                match applied:
                    # Every event that carries a session's membership says where it works.
                    case {"membership": {"id": str(session), "cwd": str(cwd)}}:
                        self._changed(self._projects, session, Path(cwd))
                    case {"type": "PermissionRequested", "request": str(request), "on": object() as on}:
                        self._asked[request] = _blocker(on)
                    case _:
                        pass
                for performed in cast(list[object], effects):
                    match performed:
                        # A reply to a hook no request was seen for, as a Stop's is, answers no permission.
                        case {"outcome": "ok", "effect": {"type": "Reply", "session": str(session), "request": str(request), "reply": object() as reply}} if request in self._asked:
                            if (answer := _answer(reply)) is not None:
                                verb, told = answer
                                self._add(_Said(datetime.fromisoformat(at), "answered", f"{verb} in", session, f"{self._asked.pop(request)}{told}"))
                        case _:
                            pass
            case {"type": "WideEvent", "event": "tool.run", "at": str(at), "outcome": str(outcome), "error": object() as error, "facts": {"tool": str(tool), "called": {"arguments": object() as arguments, "result": object() as result}}}:
                self._add(_Said(datetime.fromisoformat(at), "called", f"called {tool}", None, _call(arguments, outcome, result, error)))
            case {"type": "Transcribed", "at": str(at), "text": str(text)}:
                self._add(_Said(datetime.fromisoformat(at), "heard", "user", None, text))
            # A turn hands ended without a word said nothing.
            case {"type": "Replied", "at": str(at), "text": str(text)} if text.strip():
                self._add(_Said(datetime.fromisoformat(at), "said", "you", None, text))
            case {"type": "Typing", "at": str(at), "effect": {"session": str(session), "input": object() as typed} as effect}:
                sending = json.dumps(effect, sort_keys=True)
                # Taken out first, so a send made again goes to the back, where the newest are.
                self._sending.pop(sending, None)
                self._sending[sending] = self._taken
                self._add(_Said(datetime.fromisoformat(at), "sent", "sent to", session, _typed(typed)))
            case {"type": "TypingFailed", "effect": object() as effect, "reason": str(reason)}:
                if (taken := self._sending.pop(json.dumps(effect, sort_keys=True), None)) is not None:
                    index = taken - (self._taken - len(self._said))
                    self._said[index] = replace(self._said[index], verb="not sent to", text=f"{self._said[index].text}, because: {reason}")
                    self.changes += 1
            case _:
                pass

    def moments(self) -> list[Moment]:
        """The moments kept, oldest first."""
        return [Moment(each.at, each.kind, each.verb if each.session is None else f"{each.verb} {_spoken(each.session, self._projects, self._names)}", each.text) for each in self._said]

    def _add(self, said: _Said) -> None:
        self._said.append(said)
        self._taken += 1
        self.changes += 1
        # A send no longer kept is said again nowhere when it fails.
        oldest = self._taken - len(self._said)
        while self._sending and next(iter(self._sending.values())) < oldest:
            del self._sending[next(iter(self._sending))]

    def _changed[V](self, known: dict[str, V], session: str, value: V) -> None:
        if known.get(session) != value:
            known[session] = value
            self.changes += 1


def _spoken(session: str, projects: Mapping[str, Path], names: Mapping[str, str]) -> str:
    match projects.get(session):
        case None:
            return names.get(session, session)
        case cwd:
            return identifier(cwd, names.get(session))


def _call(arguments: object, outcome: str, result: object, error: object) -> str:
    """What a tool was given, and what it handed back, why it failed, or that it was cut off before it could."""
    given = json.dumps(arguments, ensure_ascii=False)
    match outcome, error:
        case "cancelled", _:
            return f"{given}\ncancelled"
        case _, str(why):
            return f"{given}\nfailed: {why}"
        case _:
            return f"{given}\n{json.dumps(result, ensure_ascii=False)}"


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
