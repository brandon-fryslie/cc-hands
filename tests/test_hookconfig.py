"""The hook settings: one declared permission timeout, and every other permission time derived from it."""

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import cast

from hands.core.effects import Allow, Deny, Withdraw
from hands.sessions.home import Home
from hands.sessions.hookconfig import (
    PERMISSION_DEADLINE_SECONDS,
    PERMISSION_HOOK_TIMEOUT_SECONDS,
    POST_TIMEOUT_SECONDS,
    SUBSCRIBED,
    hook_settings,
    post_timeout,
)
from hands.sessions.hooks import hook_output


def test_every_subscribed_hook_runs_the_shim_the_permission_hook_waits_and_tool_hooks_run_in_the_background() -> None:
    home = Home(Path("/Users/me/my hands"))
    settings = hook_settings(Path("/venv/bin/python"), home)
    hooks = cast(dict[str, object], settings["hooks"])
    assert list(hooks) == list(SUBSCRIBED)
    command = "/venv/bin/python -m hands.sessions.shim '/Users/me/my hands'"
    assert shlex.split(command)[-1] == str(home.root)
    for event, entries in hooks.items():
        declared = {
            "PermissionRequest": {"timeout": PERMISSION_HOOK_TIMEOUT_SECONDS},
            "PostToolUse": {"async": True},
            "PostToolUseFailure": {"async": True},
        }.get(event, {})
        assert entries == [{"hooks": [{"type": "command", "command": command, **declared}]}]


def test_the_shim_waits_as_long_as_claude_code_lets_the_hook_and_the_daemon_denies_before_that() -> None:
    assert post_timeout("PermissionRequest") == PERMISSION_HOOK_TIMEOUT_SECONDS
    assert PERMISSION_DEADLINE_SECONDS < PERMISSION_HOOK_TIMEOUT_SECONDS
    assert post_timeout("Stop") == POST_TIMEOUT_SECONDS


def test_the_module_prints_settings_claude_code_can_load() -> None:
    printed = subprocess.run([sys.executable, "-m", "hands.sessions.hookconfig"], capture_output=True, text=True, check=True)
    assert set(json.loads(printed.stdout)["hooks"]) == set(SUBSCRIBED)


def test_a_reply_is_printed_in_the_shape_claude_code_reads() -> None:
    decided = {"hookEventName": "PermissionRequest"}
    assert hook_output(Allow()) == {"hookSpecificOutput": {**decided, "decision": {"behavior": "allow"}}}
    assert hook_output(Deny("not on main")) == {"hookSpecificOutput": {**decided, "decision": {"behavior": "deny", "message": "not on main"}}}
    assert hook_output(Withdraw()) is None
