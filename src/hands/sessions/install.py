"""`hands install-hooks`: hands' hook entries merged into a Claude Code settings file, the same whatever it held before.

The entries installed are exactly the ones `hookconfig` declares; the installer keeps no list of its own. Every
entry that is hands' is taken out and the declared ones put in, so a second run changes nothing, an event hands no
longer subscribes to loses its entry, and a moved venv or home replaces the old command rather than adding a second.
Nothing that is not hands' is touched.
"""

import difflib
import json
import os
import shlex
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from hands.sessions.home import Home
from hands.sessions.hookconfig import SHIM_MODULE, hook_settings
from hands.sessions.payload import Rejected

JsonObject = dict[str, object]


def is_hands_entry(entry: object) -> bool:
    """Whether a hook entry is one hands installed: the shim, run as the single simple command hookconfig builds."""
    # [LAW:parse-dont-validate] the shape hookconfig produces, matched exactly: a wrapped or compound command, or one
    # with other arguments, is somebody else's, even if it mentions the shim.
    match entry:
        case {"type": "command", "command": str(command)}:
            try:
                argv = shlex.split(command)
            except ValueError:
                return False
            return len(argv) == 4 and argv[1:3] == ["-m", SHIM_MODULE]
        case _:
            return False


def merged(settings: JsonObject, declared: Mapping[str, list[object]]) -> JsonObject:
    """The settings with every hands entry taken out and the declared groups put in; everything else as it was."""
    hooks = _object(settings.get("hooks", {}), "hooks")
    kept = {event: _without_hands(event, groups) for event, groups in hooks.items()}
    events = {event: groups for event, groups in kept.items() if groups or not _list(hooks[event], event)}
    for event, groups in declared.items():
        events[event] = [*events.get(event, []), *groups]
    return {**settings, "hooks": events}


def _without_hands(event: str, groups: object) -> list[object]:
    kept: list[object] = []
    for group in _list(groups, event):
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
    before = path.read_text() if path.exists() else ""
    current = _object(_parse(before, path), str(path)) if before else {}
    declared = cast(Mapping[str, list[object]], hook_settings(python, home)["hooks"])
    after = _encode(merged(current, declared))
    diff = "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), str(path), str(path)))
    if diff:
        _replace(path, after)
    return Installed(path, diff)


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


def _replace(path: Path, text: str) -> None:
    """Replace the file whole, keeping its permissions: Claude Code reads the old settings or the new, never half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".hands")
    with os.fdopen(handle, "w") as out:
        out.write(text)
    os.chmod(temporary, mode)
    os.replace(temporary, path)
