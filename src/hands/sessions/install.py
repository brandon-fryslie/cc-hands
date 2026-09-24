"""`hands install-hooks`: hands' hook entries merged into a Claude Code settings file, the same whatever it held before.

The entries installed are exactly the ones `hookconfig` declares; the installer keeps no list of its own. Every
entry that runs hands' shim is taken out and the declared ones put in, so a second run changes nothing, an event
hands no longer subscribes to loses its entry, and a moved venv or home, or a hand-wrapped shim, is replaced rather
than joined by a second. Nothing that does not run the shim is touched.
"""

import difflib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from hands.sessions.home import Home
from hands.sessions.files import replace_whole
from hands.sessions.hookconfig import hook_settings, runs_the_shim
from hands.sessions.payload import Rejected

JsonObject = dict[str, object]


def is_hands_entry(entry: object) -> bool:
    """Whether a hook entry is one of hands': a command that runs its shim."""
    match entry:
        case {"type": "command", "command": str(command)}:
            return runs_the_shim(command)
        case _:
            return False


def merged(settings: JsonObject, declared: Mapping[str, list[object]]) -> JsonObject:
    """The settings with every hands entry taken out and the declared groups put in; everything else as it was."""
    hooks = {event: _list(groups, event) for event, groups in _object(settings.get("hooks", {}), "hooks").items()}
    kept = {event: _without_hands(groups) for event, groups in hooks.items()}
    # An event left with nothing loses its key, unless it held nothing to begin with; that one was not ours to remove.
    events = {event: groups for event, groups in kept.items() if groups or not hooks[event]}
    for event, groups in declared.items():
        events[event] = [*events.get(event, []), *groups]
    return {**settings, "hooks": events}


def _without_hands(groups: list[object]) -> list[object]:
    kept: list[object] = []
    for group in groups:
        match group:
            case {"hooks": list()}:
                entries = cast(list[object], cast(JsonObject, group)["hooks"])
                theirs = [entry for entry in entries if not is_hands_entry(entry)]
                # A group that held only hands' entries goes with them; one that held anything else stays, less ours.
                if theirs or not entries:
                    kept.append({**cast(JsonObject, group), "hooks": theirs})
            case _:
                kept.append(group)
    return kept


@dataclass(frozen=True)
class Installed:
    path: Path  # the file written: the settings path with its symlinks resolved
    diff: str  # empty when the file already held exactly these entries


def install(settings: Path, python: Path, home: Home) -> Installed:
    """Merge hands' declared hook entries into the settings file, and say what changed."""
    # A settings file kept in a dotfiles repo is a symlink; the file it points to is replaced, never the link.
    path = settings.resolve()
    before = path.read_text(encoding="utf-8") if path.exists() else ""
    current = _object(_parse(before, path), str(path)) if before else {}
    # Claude Code runs a hook from the session's own directory, so the home the shim is given must not be relative.
    declared = cast(Mapping[str, list[object]], hook_settings(python, Home(home.root.resolve()))["hooks"])
    after = _encode(merged(current, declared))
    diff = _diff(before, after, str(path))
    if diff:
        replace_whole(path, after, path.stat().st_mode & 0o777 if path.exists() else 0o644)
    return Installed(path, diff)


def default_settings() -> Path:
    """The user settings file Claude Code reads: in CLAUDE_CONFIG_DIR when that is set, as Claude Code itself does."""
    return Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")).expanduser() / "settings.json"


def _diff(before: str, after: str, name: str) -> str:
    lines = difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), name, name)
    # A last line with no newline is marked as git and patch mark it, rather than run into the line after it.
    return "".join(line if line.endswith("\n") else f"{line}\n\\ No newline at end of file\n" for line in lines)


def _encode(settings: JsonObject) -> str:
    # Claude Code writes its settings with two-space indents and a closing newline; so does this, so that the
    # diff is the hooks and nothing else.
    return json.dumps(settings, indent=2, ensure_ascii=False) + "\n"


def _parse(text: str, path: Path) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        # [LAW:no-silent-failure] a settings file that does not parse is refused, never overwritten.
        raise Rejected(f"{path} is not JSON, so it is left as it is: {error}") from error


def _object(value: object, where: str) -> JsonObject:
    match value:
        case dict():
            return cast(JsonObject, value)
        case _:
            raise Rejected(f"{where} should be a JSON object, got {type(value).__name__}; nothing was changed")


def _list(value: object, event: str) -> list[object]:
    match value:
        case list():
            return cast(list[object], value)
        case _:
            raise Rejected(f"hooks.{event} should be a list, got {type(value).__name__}; nothing was changed")
