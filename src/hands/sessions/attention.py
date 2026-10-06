"""What hands says unprompted, as the user set it: one file in the home, so it outlives the daemon and needs no restart.

The plugin's `/hands:attention` skill runs this module through the plugin's launcher, on the installed hands' own
interpreter, and it imports only the standard library and hands' data modules:

    hooks/python -m hands.sessions.attention [finished|progress|ended|quiet LEVEL]...

With no argument it says what is set. The daemon reads the file at every finished turn, every burst of progress, and
every session ending, and holds no copy [LAW:one-source-of-truth], so a change is heard from the next one on.
"""

import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from typing import cast, get_args, get_type_hints

from hands.core.attention import Attention, Level, Switch
from hands.sessions.files import replace_whole
from hands.sessions.home import Home, default_home
from hands.sessions.payload import Rejected

# [LAW:one-source-of-truth] each kind's levels are read off the type, so a kind added to Attention is one here as well.
_LEVELS: Mapping[str, tuple[str, ...]] = {name: get_args(hint) for name, hint in get_type_hints(Attention).items()}


def attention(home: Home) -> Attention:
    """What is set; a kind never set, or a file never written, is at its default."""
    try:
        written = home.attention.read_bytes()
    except FileNotFoundError:
        return Attention()
    try:
        read = json.loads(written)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Rejected(f"{home.attention} is not JSON: {error}") from error
    if not isinstance(read, dict):
        raise Rejected(f"{home.attention} says {written!r}, which is no setting of what hands says unprompted")
    return changed(Attention(), [(str(kind), str(level)) for kind, level in cast(dict[object, object], read).items()])


def changed(attention: Attention, changes: Sequence[tuple[str, str]]) -> Attention:
    """`attention` with each (kind, level) in `changes` set, in order, read as a person says it: "On" is on, and a kind
    with levels turned on is all of it."""
    # [LAW:parse-dont-validate] the one place a kind and its level are read, by the CLI, the voice tool, and the file.
    settings: dict[str, Level | Switch] = {}
    for kind, level in changes:
        kind, level = kind.strip().lower(), level.strip().lower()
        allowed = _LEVELS.get(kind)
        if allowed is None:
            raise Rejected(f"{kind!r} is no kind of thing hands says unprompted; it is one of {', '.join(_LEVELS)}")
        said = "full" if level == "on" and "full" in allowed else level
        if said not in allowed:
            raise Rejected(f"{level!r} is no level of {kind}; it is one of {', '.join(allowed)}")
        settings[kind] = said  # pyright: ignore[reportArgumentType]  (held to the kind's Literal just above)
    return replace(attention, **settings)


def set_attention(home: Home, to: Attention) -> None:
    # [LAW:no-ambient-temporal-coupling] replaced whole, so the daemon never reads half a file.
    replace_whole(home.attention, json.dumps(asdict(to)) + "\n", 0o644)


def asked(home: Home, changes: Sequence[tuple[str, str]]) -> Attention:
    """What is set once `changes` are made, as the CLI and the voice tool are both asked: none only reads what is set, and
    writes nothing, so a question never lands over a change made meanwhile [LAW:single-enforcer]."""
    if not changes:
        return attention(home)
    to = changed(_in_effect(home), changes)
    set_attention(home, to)
    return to


def _in_effect(home: Home) -> Attention:
    """What is set as the daemon acts on it: a file that cannot be read is in effect as the defaults, so a change is made
    to them and replaces it, and the readback, which says every setting, says what is set now. A file only a change can
    mend would leave the user unable to quiet hands by voice."""
    try:
        return attention(home)
    except Rejected:
        return Attention()


# Each hook kind as the readback names it, in the order it says them.
_HOOKS: Mapping[str, str] = {
    "permission_denied": "auto mode's refusals",
    "subagent_start": "subagents starting",
    "subagent_stop": "subagents finishing",
    "task_completed": "tasks completed",
    "config_change": "settings changing",
    "pre_compact": "compaction",
    "clear": "a session cleared",
}


_HOW: Mapping[str, str] = {"brief": "briefly", "full": "in full"}


def _hooks(attention: Attention) -> list[tuple[str, str]]:
    return [(kind, getattr(attention, kind)) for kind in _HOOKS]


def _listed(parts: Sequence[str]) -> str:
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"


def described(attention: Attention) -> str:
    """What is set, as a sentence a person can be read: each kind, then quiet, then what is spoken whatever is set."""
    finished = {
        "off": "Finished turns wait until you ask, except a watched session's.",
        "brief": "I tell each finished turn in a few words.",
        "full": "I tell each finished turn as it finishes.",
    }[attention.finished]
    progress = {
        "off": "I don't tell what the focused session is doing.",
        "brief": "I tell what the focused session says it is doing, without each step.",
        "full": "I tell each step the focused session takes.",
    }[attention.progress]
    ended = {"on": "I say when a session ends.", "off": "I don't say when a session ends."}[attention.ended]
    told = [f"{_HOOKS[kind]} {_HOW[level]}" for kind, level in _hooks(attention) if level != "off"]
    hooks = f"Of Claude Code's other events, I tell {_listed(told)}." if told else "I tell none of Claude Code's other events."
    quiet = {
        "on": "I'm keeping quiet for now: none of that is said until you let me talk again, and you can ask for any of it.",
        "off": "",
    }[attention.quiet]
    return " ".join(part for part in (finished, progress, ended, hooks, quiet, "What needs your answer is always said.") if part)


def main(argv: Sequence[str]) -> int:
    try:
        home = default_home(os.environ)
        if len(argv) % 2:
            print(f"hands attention: expected a kind and its level, in pairs, got {' '.join(argv)!r}", file=sys.stderr)
            return 2
        print(described(asked(home, list(zip(argv[::2], argv[1::2], strict=True)))))
    except (Rejected, OSError) as error:
        print(f"hands attention: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
