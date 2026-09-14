"""The hook settings a Claude Code session runs hands with, and the one number every permission time comes from.

    <hands python> -m hands.sessions.hookconfig     # prints the settings JSON

It is imported by the shim, so it holds only the standard library and hands' data modules.
"""

import json
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

SUBSCRIBED = ("SessionStart", "UserPromptSubmit", "Stop", "PermissionRequest", "SessionEnd")

# [LAW:dataflow-not-control-flow] the hooks that declare their own timeout, as a table; the rest take Claude Code's default.
_DECLARED_TIMEOUTS: Mapping[str, int] = {"PermissionRequest": PERMISSION_HOOK_TIMEOUT_SECONDS}


def post_timeout(event: str) -> float:
    """How long the shim waits on the daemon for this hook: a blocking hook as long as Claude Code lets it."""
    return float(_DECLARED_TIMEOUTS.get(event, POST_TIMEOUT_SECONDS))


def hook_settings(python: Path, home: Home) -> dict[str, object]:
    # A single simple command, so the hook's shell execs it and the shim's parent is the claude process.
    command = shlex.join([str(python), "-m", "hands.sessions.shim", str(home.root)])
    return {"hooks": {event: [{"hooks": [{"type": "command", "command": command, **_timeout(event)}]}] for event in SUBSCRIBED}}


def _timeout(event: str) -> dict[str, int]:
    declared = _DECLARED_TIMEOUTS.get(event)
    return {} if declared is None else {"timeout": declared}


def main() -> None:
    print(json.dumps(hook_settings(Path(sys.executable), default_home()), indent=2))


if __name__ == "__main__":
    main()
