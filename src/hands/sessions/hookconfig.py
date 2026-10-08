"""The hooks hands' Claude Code plugin installs, and the one number every permission time comes from.

    <hands python> -m hands.sessions.hookconfig > src/hands/sessions/plugin/hooks/hooks.json     # regenerates the plugin's hook file

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

# [LAW:single-enforcer] how long the daemon holds a Stop's hook for the transcript to say whose Stop it is: past the
# second the reducer waits for that record (UNTOLD), and a tail read after it. Past this the hook is let go, and the
# Stop tells its turn once decided, with nothing holding Claude Code. The shim waits that much longer for a Stop.
STOP_HOLD_SECONDS = 1.5
STOP_POST_TIMEOUT_SECONDS = POST_TIMEOUT_SECONDS + STOP_HOLD_SECONDS

# [LAW:one-source-of-truth] the module every hook runs, and the launcher, inside the plugin, that runs it on the
# installed hands' own interpreter, which `hands plugin` writes (hands.sessions.marketplace).
SHIM_MODULE = "hands.sessions.shim"
LAUNCHER = "hooks/python"
# Where the plugin's hook file lives, relative to the plugin root; Claude Code loads it from there unasked.
HOOKS_FILE = "hooks/hooks.json"
# The plugin as Claude Code names it: its name in its .claude-plugin/plugin.json, at the marketplace's name in
# .claude-plugin/marketplace.json.
PLUGIN_ID = "hands@cc-hands"
# Where Claude Code adds that marketplace from: this repository, on GitHub.
MARKETPLACE = "brandon-fryslie/cc-hands"

# [LAW:one-source-of-truth] where MessageDisplay is posted: Claude Code dispatches it synchronously for every batch of
# lines it displays (2.1.270), so it is an HTTP hook, with no process spawned per batch. Its URL takes no variables
# (2.1.288), so the port is fixed here and the daemon serves it, on loopback alone: over TCP any local user can reach a
# route, so this one takes MessageDisplay and nothing else, and permissions stay on the home's unix socket.
DISPLAY_HOST = "127.0.0.1"
DISPLAY_PORT = 47615
DISPLAY_PATH = "/hands/display"
DISPLAY_URL = f"http://{DISPLAY_HOST}:{DISPLAY_PORT}{DISPLAY_PATH}"
# How long Claude Code waits on each displayed batch: it waits in the agent's path, so a daemon that is slow, down, or
# hung costs lines of narration and never the agent's speed (measured on 2.1.280).
DISPLAY_TIMEOUT_SECONDS = 2

# The hooks whose occurrences hands only passes on (hands.core.occurrences), as the user sets each to be said.
PASSED_ON = ("PermissionDenied", "SubagentStart", "SubagentStop", "TaskCompleted", "ConfigChange", "PreCompact")
SUBSCRIBED = ("SessionStart", "UserPromptSubmit", "Stop", "PermissionRequest", "PostToolUse", "PostToolUseFailure", "SessionEnd", *PASSED_ON)

# [LAW:dataflow-not-control-flow] what each hook declares beyond its command, as a table; the rest take Claude Code's defaults.
_DECLARED_TIMEOUTS: Mapping[str, int] = {"PermissionRequest": PERMISSION_HOOK_TIMEOUT_SECONDS}
# [LAW:dataflow-not-control-flow] how long the shim waits on the daemon for each hook that waits on it.
_WAITS: Mapping[str, float] = {**_DECLARED_TIMEOUTS, "Stop": STOP_POST_TIMEOUT_SECONDS}
# Run in the background, so they never hold the agent up: the tool hooks fire for every call, and are how the daemon
# learns that a tool it was asked about ran after all, its dialog answered at the keyboard; what is only passed on
# waits on nothing hands could say back.
_IN_BACKGROUND = frozenset({"PostToolUse", "PostToolUseFailure", *PASSED_ON})
# Claude Code's own internal agents (prompt suggestions, /btw) fire the subagent hooks under an empty agent type, which
# a matcher that matches no empty string keeps out: they are no subagent the session started.
_MATCHERS: Mapping[str, str] = {"SubagentStart": ".+", "SubagentStop": ".+"}


def post_timeout(event: str) -> float:
    """How long the shim waits on the daemon for this hook: a blocking hook as long as Claude Code lets it."""
    return float(_WAITS.get(event, POST_TIMEOUT_SECONDS))


def plugin_hooks() -> dict[str, object]:
    """The plugin's hooks.json: every subscribed event runs the shim through the plugin's launcher, and MessageDisplay
    is posted to the daemon."""
    # Exec form (`args` set): Claude Code spawns the launcher itself, with no shell between, and the launcher execs
    # Python, so the shim is the process Claude Code spawned and its parent is the claude process whose pid it records.
    command = {"type": "command", "command": f"${{CLAUDE_PLUGIN_ROOT}}/{LAUNCHER}", "args": ["-m", SHIM_MODULE]}
    display = {"type": "http", "url": DISPLAY_URL, "timeout": DISPLAY_TIMEOUT_SECONDS}
    return {"hooks": {**{event: [{**_matched(event), "hooks": [{**command, **declared(event)}]}] for event in SUBSCRIBED}, "MessageDisplay": [{"hooks": [display]}]}}


def _matched(event: str) -> dict[str, object]:
    matcher = _MATCHERS.get(event)
    return {} if matcher is None else {"matcher": matcher}


def rendered() -> str:
    """hooks.json's text, byte for byte as it is checked in."""
    return json.dumps(plugin_hooks(), indent=2) + "\n"


def declared(event: str) -> dict[str, object]:
    """What a hook declares beyond what runs it: how long Claude Code waits on it, and whether it runs in the background."""
    timeout = _DECLARED_TIMEOUTS.get(event)
    return {**({} if timeout is None else {"timeout": timeout}), **({"async": True} if event in _IN_BACKGROUND else {})}


def main() -> None:
    print(rendered(), end="")


if __name__ == "__main__":
    main()
