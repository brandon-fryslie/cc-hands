"""The hook settings a Claude Code session runs hands with, and the one number every permission time comes from.

    <hands python> -m hands.sessions.hookconfig     # prints the settings JSON
    hands install-hooks                             # merges them into ~/.claude/settings.json

It is imported by the shim, so it holds only the standard library and hands' data modules.
"""

import json
import re
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path

from hands.sessions.home import Home, default_home

# [LAW:single-enforcer] Claude Code kills a PermissionRequest hook after this many seconds. The settings
# below declare it, and the shim's wait and the daemon's default-deny deadline are both derived from it.
PERMISSION_HOOK_TIMEOUT_SECONDS = 90

# A deny sent at the deadline must reach Claude Code before it kills the hook, or nothing denies it.
REPLY_MARGIN_SECONDS = 5
PERMISSION_DEADLINE_SECONDS = float(PERMISSION_HOOK_TIMEOUT_SECONDS - REPLY_MARGIN_SECONDS)

# Every other hook posts and returns, so a daemon slower than this is reported as unreachable.
POST_TIMEOUT_SECONDS = 2.0

# [LAW:one-source-of-truth] the module every hook command runs, and the installer's mark of an entry that is hands'.
SHIM_MODULE = "hands.sessions.shim"

SUBSCRIBED = ("SessionStart", "UserPromptSubmit", "Stop", "Notification", "PermissionRequest", "PostToolUse", "PostToolUseFailure", "SessionEnd")

# [LAW:dataflow-not-control-flow] what each hook declares beyond its command, as a table; the rest take Claude Code's defaults.
_DECLARED_TIMEOUTS: Mapping[str, int] = {"PermissionRequest": PERMISSION_HOOK_TIMEOUT_SECONDS}
# The notifications hands hears, by Claude Code's notification_type; the rest never spawn the shim.
_MATCHERS: Mapping[str, str] = {"Notification": "idle_prompt"}
# Fired for every tool call, so they run in the background and never hold the agent up. They are how the
# daemon learns that a tool it was asked about ran after all: its dialog was answered at the keyboard.
_IN_BACKGROUND = frozenset({"PostToolUse", "PostToolUseFailure"})


def post_timeout(event: str) -> float:
    """How long the shim waits on the daemon for this hook: a blocking hook as long as Claude Code lets it."""
    return float(_DECLARED_TIMEOUTS.get(event, POST_TIMEOUT_SECONDS))


def hook_settings(python: Path, home: Home) -> dict[str, object]:
    # A single simple command, so the hook's shell execs it and the shim's parent is the claude process.
    command = shlex.join([str(python), "-m", SHIM_MODULE, str(home.root)])
    return {"hooks": {event: [{**_matched(event), "hooks": [{"type": "command", "command": command, **_declared(event)}]}] for event in SUBSCRIBED}}


def runs_the_shim(command: str) -> bool:
    """Whether a hook command runs hands' shim, as hook_settings builds it or as any earlier or hand-edited version did."""
    # [LAW:one-source-of-truth] beside the builder, and keyed on the one thing every version of it shares: the shim's
    # module, named whole. It is found inside `sh -c '...'` and in `-mhands.sessions.shim` alike, so a change to how
    # the command is built, or a user's wrapping of it, never leaves an entry looking like somebody else's and doubled.
    return _SHIM_NAMED.search(command) is not None


# Named whole: not inside a longer dotted name, though it may be joined to its -m.
_SHIM_NAMED = re.compile(rf"(?:(?<=-m)|(?<![\w.])){re.escape(SHIM_MODULE)}(?![\w.])")


def _matched(event: str) -> dict[str, object]:
    matcher = _MATCHERS.get(event)
    return {} if matcher is None else {"matcher": matcher}


def _declared(event: str) -> dict[str, object]:
    timeout = _DECLARED_TIMEOUTS.get(event)
    return {**({} if timeout is None else {"timeout": timeout}), **({"async": True} if event in _IN_BACKGROUND else {})}


def main() -> None:
    print(json.dumps(hook_settings(Path(sys.executable), default_home()), indent=2))


if __name__ == "__main__":
    main()
