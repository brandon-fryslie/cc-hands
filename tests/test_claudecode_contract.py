"""The contract of `ClaudeCode` (docs/testing.md): what any Claude Code answers, run against the fake the other tests use
and, where a real Claude Code is installed, against `Installed` running it. A fake that drifts from the real one fails
here first.

Only what needs no person and no network is asked of the real one: its login and listings under a config directory no
Claude Code has used, and its files.
"""

import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from claudecode_fake import Fake
from hands.sessions import wrapper
from hands.sessions.claudecode import ClaudeCode, Installed, Instance

REAL = wrapper.real_claude(os.environ.get("PATH", ""))
KEY = "sk-ant-api03-contract0not0a0real0key0AAAAAAAAAAAAAAAAAAAAAA"


def fake() -> ClaudeCode:
    return Fake()


def installed() -> ClaudeCode:
    if REAL is None:
        pytest.skip("no Claude Code is installed here to hold the fake to")
    return Installed()


@pytest.fixture(params=[fake, installed], ids=["fake", "installed"])
def claude_code(request: pytest.FixtureRequest) -> ClaudeCode:
    built: Callable[[], ClaudeCode] = request.param
    return built()


def unused(tmp_path: Path, **environment: str) -> Instance:
    """A Claude Code under a config directory no Claude Code has used, with no key unless one is named."""
    kept = {name: value for name, value in os.environ.items() if name != "ANTHROPIC_API_KEY"}
    return Instance(REAL or "claude", {**kept, "CLAUDE_CONFIG_DIR": str(tmp_path / "config"), **environment}, tmp_path)


def test_a_config_no_claude_code_has_used_holds_no_login(claude_code: ClaudeCode, tmp_path: Path) -> None:
    said = claude_code.auth_status(unused(tmp_path))
    status = json.loads(said.stdout)
    assert (said.exit, status["loggedIn"], status["authMethod"], status["apiProvider"]) == (1, False, "none", "firstParty")


def test_an_api_key_in_the_environment_is_a_login(claude_code: ClaudeCode, tmp_path: Path) -> None:
    said = claude_code.auth_status(unused(tmp_path, ANTHROPIC_API_KEY=KEY))
    status = json.loads(said.stdout)
    assert (said.exit, status["loggedIn"], status["authMethod"], status["apiKeySource"]) == (0, True, "api_key", "ANTHROPIC_API_KEY")


def test_a_config_no_claude_code_has_used_lists_no_plugins_and_no_marketplaces(claude_code: ClaudeCode, tmp_path: Path) -> None:
    claude = unused(tmp_path)
    plugins, marketplaces = claude_code.plugins(claude), claude_code.marketplaces(claude)
    assert (plugins.exit, json.loads(plugins.stdout), marketplaces.exit, json.loads(marketplaces.stdout)) == (0, [], 0, [])


def test_a_file_that_is_not_there_reads_as_none_is_created_once_and_replaced_whole(claude_code: ClaudeCode, tmp_path: Path) -> None:
    state = tmp_path / "config" / ".claude.json"
    assert claude_code.read(state) is None
    assert claude_code.create(state, "{}") is True
    assert claude_code.create(state, '{"again": true}') is False
    assert claude_code.read(state) == b"{}"
    claude_code.replace(state, '{"hasCompletedOnboarding": true}', 0o600)
    assert claude_code.read(state) == b'{"hasCompletedOnboarding": true}'
