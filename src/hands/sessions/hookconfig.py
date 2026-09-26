"""The hooks hands' Claude Code plugin installs, and the one number every permission time comes from.

    <hands python> -m hands.sessions.hookconfig > plugin/hooks/hooks.json     # regenerates the plugin's hook file

It is imported by the shim, so it holds only the standard library and hands' data modules.
"""

import json
from collections.abc import Mapping

# [LAW:single-enforcer] Claude Code kills a PermissionRequest hook after this many seconds. The hooks
# below declare it, and the shim's wait and the daemon's default-deny deadline are both derived from it.
PERMISSION_HOOK_TIMEOUT_SECONDS = 90

# A deny sent at the deadline must reach Claude Code before it kills the hook, or nothing denies it.
REPLY_MARGIN_SECONDS = 5
PERMISSION_DEADLINE_SECONDS = float(PERMISSION_HOOK_TIMEOUT_SECONDS - REPLY_MARGIN_SECONDS)

# Every other hook posts and returns, so a daemon slower than this is reported as unreachable.
POST_TIMEOUT_SECONDS = 2.0

# [LAW:one-source-of-truth] the module every hook runs, and the launcher, inside the plugin, that runs it under a
# Python new enough for hands. The plugin has no venv, so the launcher puts the plugin's own src on the path.
SHIM_MODULE = "hands.sessions.shim"
LAUNCHER = "hooks/python"
# Where the plugin's hook file lives, relative to the plugin root; Claude Code loads it from there unasked.
HOOKS_FILE = "hooks/hooks.json"
# The plugin, relative to the repository, which is its marketplace. A directory of its own, so an install copies the
# hooks and a link to src, never the repository's venv.
PLUGIN_DIR = "plugin"

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


def plugin_hooks() -> dict[str, object]:
    """The plugin's hooks.json: every subscribed event runs the shim through the plugin's launcher."""
    # Exec form (`args` set): Claude Code spawns the launcher itself, with no shell between, and the launcher execs
    # Python, so the shim is the process Claude Code spawned and its parent is the claude process whose pid it records.
    command = {"type": "command", "command": f"${{CLAUDE_PLUGIN_ROOT}}/{LAUNCHER}", "args": ["-m", SHIM_MODULE]}
    return {"hooks": {event: [{**_matched(event), "hooks": [{**command, **_declared(event)}]}] for event in SUBSCRIBED}}


def rendered() -> str:
    """hooks.json's text, byte for byte as it is checked in."""
    return json.dumps(plugin_hooks(), indent=2) + "\n"


def _matched(event: str) -> dict[str, object]:
    matcher = _MATCHERS.get(event)
    return {} if matcher is None else {"matcher": matcher}


def _declared(event: str) -> dict[str, object]:
    timeout = _DECLARED_TIMEOUTS.get(event)
    return {**({} if timeout is None else {"timeout": timeout}), **({"async": True} if event in _IN_BACKGROUND else {})}


def main() -> None:
    print(rendered(), end="")


if __name__ == "__main__":
    main()
