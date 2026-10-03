"""The plugin's hooks: one declared permission timeout, every other permission time derived from it, and a hooks.json
that is hookconfig's table and nothing else."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

from hands.core.effects import AllowWith, Allow, Deny, Withdraw
from hands.core.reducer import UNTOLD
from hands.sessions.hookconfig import (
    HOOKS_FILE,
    LAUNCHER,
    PERMISSION_DEADLINE_SECONDS,
    PERMISSION_HOOK_TIMEOUT_SECONDS,
    PLUGIN_DIR,
    PLUGIN_ID,
    POST_TIMEOUT_SECONDS,
    STOP_HOLD_SECONDS,
    SUBSCRIBED,
    plugin_hooks,
    post_timeout,
    rendered,
)
from hands.sessions.hooks import hook_output

PLUGIN_ROOT = Path(__file__).resolve().parent.parent / PLUGIN_DIR


def test_every_subscribed_hook_runs_the_shim_the_permission_hook_waits_and_tool_hooks_run_in_the_background() -> None:
    hooks = cast(dict[str, object], plugin_hooks()["hooks"])
    # MessageDisplay is posted to the daemon with no shim: tests/test_display.py.
    assert list(hooks) == [*SUBSCRIBED, "MessageDisplay"]
    command = {"type": "command", "command": "${CLAUDE_PLUGIN_ROOT}/hooks/python", "args": ["-m", "hands.sessions.shim"]}
    for event in SUBSCRIBED:
        entries = hooks[event]
        declared = {
            "PermissionRequest": {"timeout": PERMISSION_HOOK_TIMEOUT_SECONDS},
            "PostToolUse": {"async": True},
            "PostToolUseFailure": {"async": True},
        }.get(event, {})
        assert entries == [{"hooks": [{**command, **declared}]}]


def test_the_checked_in_hooks_json_is_what_hookconfig_declares() -> None:
    # [LAW:one-source-of-truth] hooks.json is generated from hookconfig; a hand edit, or a change to hookconfig not
    # regenerated with `python -m hands.sessions.hookconfig > plugin/hooks/hooks.json`, fails here.
    assert (PLUGIN_ROOT / HOOKS_FILE).read_text(encoding="utf-8") == rendered()


def test_the_module_prints_the_hooks_json() -> None:
    printed = subprocess.run([sys.executable, "-m", "hands.sessions.hookconfig"], capture_output=True, text=True, check=True)
    assert printed.stdout == rendered()
    assert set(json.loads(printed.stdout)["hooks"]) == {*SUBSCRIBED, "MessageDisplay"}


def test_the_launcher_every_hook_names_is_in_the_plugin_and_runnable() -> None:
    assert os.access(PLUGIN_ROOT / LAUNCHER, os.X_OK)


def test_the_shim_waits_as_long_as_claude_code_lets_the_hook_and_the_daemon_denies_before_that() -> None:
    assert post_timeout("PermissionRequest") == PERMISSION_HOOK_TIMEOUT_SECONDS
    assert PERMISSION_DEADLINE_SECONDS < PERMISSION_HOOK_TIMEOUT_SECONDS
    # A Stop's hook is held past the window the reducer waits for the record that names it, and the shim waits the
    # hold longer than for any other hook.
    assert STOP_HOLD_SECONDS > UNTOLD / 1000
    assert post_timeout("Stop") == POST_TIMEOUT_SECONDS + STOP_HOLD_SECONDS
    assert post_timeout("Notification") == POST_TIMEOUT_SECONDS


def test_a_reply_is_printed_in_the_shape_claude_code_reads() -> None:
    decided = {"hookEventName": "PermissionRequest"}
    assert hook_output(Allow()) == {"hookSpecificOutput": {**decided, "decision": {"behavior": "allow"}}}
    assert hook_output(Deny("not on main")) == {"hookSpecificOutput": {**decided, "decision": {"behavior": "deny", "message": "not on main"}}}
    assert hook_output(Withdraw()) is None
    answered: dict[str, object] = {"questions": [{"question": "Which?", "options": []}], "answers": {"Which?": "this"}}
    assert hook_output(AllowWith(answered)) == {"hookSpecificOutput": {**decided, "decision": {"behavior": "allow", "updatedInput": answered}}}


def test_the_plugin_id_is_the_plugin_s_name_at_the_marketplace_s() -> None:
    marketplace = json.loads((PLUGIN_ROOT.parent / ".claude-plugin" / "marketplace.json").read_text())
    plugin = json.loads((PLUGIN_ROOT / ".claude-plugin" / "plugin.json").read_text())
    assert f"{plugin['name']}@{marketplace['name']}" == PLUGIN_ID
    assert [entry["source"] for entry in marketplace["plugins"] if entry["name"] == plugin["name"]] == [f"./{PLUGIN_DIR}"]
