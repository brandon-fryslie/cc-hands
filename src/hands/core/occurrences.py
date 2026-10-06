"""What a session's hooks say happened in it that hands only passes on: nothing in the session moves for it, and the
user hears of it as they set its kind to be said (hands.core.attention), in the words made here.

Each is one hook of Claude Code's, parsed at the edge (hands.sessions.hooks); a /clear is the SessionStart it fires,
which hands also joins the session's new id on.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hands.core.attention import Amount, Attention, Level, Overlay, Route, occurrence_route

# How much of a detail a full telling says as written: past it, the line is cut and says so.
DETAIL_SHOWN = 300


@dataclass(frozen=True)
class AutoDenied:
    """Auto mode refused a tool call: PermissionDenied, which fires in auto mode alone, never for a dialog the user
    answered or a rule that matched."""

    tool: str
    # Why, as Claude Code words it: a classifier's rule in brackets, such as "[Data Exfiltration]", or a sentence.
    reason: str
    input: Mapping[str, object]


@dataclass(frozen=True)
class SubagentStarted:
    """The session started a subagent, or resumed one. `agent_type` is the agent's name, never empty: the hook's matcher
    keeps out Claude Code's own internal agents, which run under the empty name."""

    agent_type: str


@dataclass(frozen=True)
class SubagentStopped:
    """A subagent the session started finished responding. `closing` is its last text, as the hook carries it: None
    where it carried none, as for a subagent that handed its report back through a tool (SubagentHandback, seen on
    2.1.289), whose report is told with the turn it reports back to (hands.core.subagents)."""

    agent_type: str
    closing: str | None


@dataclass(frozen=True)
class TaskCompleted:
    """A task on the session's task list was marked completed, by an agent or a teammate finishing its turn."""

    subject: str
    description: str | None
    teammate: str | None


# Which configuration changed under a running session, as ConfigChange names it.
ConfigSource = Literal["user_settings", "project_settings", "local_settings", "policy_settings", "skills"]

_CONFIG: Mapping[ConfigSource, str] = {
    "user_settings": "user settings",
    "project_settings": "project settings",
    "local_settings": "local settings",
    "policy_settings": "managed policy",
    "skills": "skills",
}


@dataclass(frozen=True)
class ConfigChanged:
    source: ConfigSource
    path: Path | None


# What started a compaction: /compact, or the context filling.
CompactTrigger = Literal["manual", "auto"]


@dataclass(frozen=True)
class Compacting:
    """Claude Code is about to compact the session's context: asked with /compact, with what was typed after it, or on
    its own as the context filled."""

    trigger: CompactTrigger
    instructions: str | None


@dataclass(frozen=True)
class Cleared:
    """The user cleared the conversation with /clear: the same process goes on as a new session."""


Occurrence = AutoDenied | SubagentStarted | SubagentStopped | TaskCompleted | ConfigChanged | Compacting | Cleared


def route(attention: Attention, overlay: Overlay, occurrence: Occurrence) -> Route:
    """How `occurrence` reaches the user, as its kind is set to be said."""
    return occurrence_route(attention, overlay, _level(attention, occurrence))


def _level(attention: Attention, occurrence: Occurrence) -> Level:
    # [LAW:one-source-of-truth] each kind's level is the setting's own field, read off the variant.
    match occurrence:
        case AutoDenied():
            return attention.permission_denied
        case SubagentStarted():
            return attention.subagent_start
        case SubagentStopped():
            return attention.subagent_stop
        case TaskCompleted():
            return attention.task_completed
        case ConfigChanged():
            return attention.config_change
        case Compacting():
            return attention.pre_compact
        case Cleared():
            return attention.clear


def said(occurrence: Occurrence, name: str, amount: Amount) -> str:
    """The line hands says of `occurrence` in session `name`: brief is what happened, full adds what the hook says of it."""
    match amount:
        case "brief":
            return _line(occurrence, name)
        case "full":
            return " ".join((_line(occurrence, name), *_details(occurrence)))


def _line(occurrence: Occurrence, name: str) -> str:
    match occurrence:
        case AutoDenied(tool=tool):
            return f"Auto mode refused {name} a {tool} call."
        case SubagentStarted(agent_type=agent):
            return f"{name} started its {agent} subagent."
        case SubagentStopped(agent_type=agent):
            return f"{name}'s {agent} subagent finished."
        case TaskCompleted(subject=subject):
            return f"{name} completed a task: {subject}."
        case ConfigChanged(source=source):
            return f"{name}'s {_CONFIG[source]} changed."
        case Compacting():
            return f"{name} is compacting its context."
        case Cleared():
            return f"{name} was cleared."


def _details(occurrence: Occurrence) -> tuple[str, ...]:
    """What a full telling says past the line, each detail a sentence; none where the hook carries nothing more."""
    match occurrence:
        case AutoDenied(reason=reason, input=input):
            return (f"Why: {_bounded(reason.strip('[]'))}.", f"The call: {_bounded(json.dumps(dict(input), ensure_ascii=False))}")
        case SubagentStarted() | Cleared():
            return ()
        case SubagentStopped(closing=closing):
            return _maybe(closing, "It said: {}")
        case TaskCompleted(description=description, teammate=teammate):
            return (*_maybe(teammate, "{} completed it."), *_maybe(description, "{}"))
        case ConfigChanged(path=path):
            return _maybe(None if path is None else str(path), "The file: {}.")
        case Compacting(trigger="auto"):
            return ("Its context filled, so Claude Code is doing it on its own.",)
        case Compacting(instructions=instructions):
            return ("It was asked to, with /compact.", *_maybe(instructions, "Its instructions: {}"))


def _maybe(detail: str | None, sentence: str) -> tuple[str, ...]:
    """`detail` in its sentence, bounded; nothing where the hook carried none."""
    return () if detail is None else (sentence.format(_bounded(detail)),)


def _bounded(text: str) -> str:
    return text if len(text) <= DETAIL_SHOWN else f"{text[:DETAIL_SHOWN]}... (cut short)"
