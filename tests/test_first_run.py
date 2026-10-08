"""`hands first-run`, run at a terminal as install.sh runs it: what the person's own Claude Code asks only once is said
before Claude Code shows it, asked in the folder `hands smoke` starts its session in, and not asked again once answered.

`claude` is a stand-in that records each call, says whether it is logged in from a file beside its calls, or, as Claude
Code does, logged in on any API key in its environment, approved or not, and, started
with no arguments, shows its first screen and reads the answer from the terminal as Claude Code does: y answers every
question and logs in, writing what Claude Code records; anything else quits before the last question.
"""

import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from hands.sessions import firstrun
from hands.sessions.audit import segment
from hands.sessions.home import Home
from hands.sessions.payload import Rejected

CLAUDE = r"""#!/bin/bash
echo "claude $* in $(pwd -P)" >>"$ROOT/calls"
case "$1 $2" in
  "auth status")
    if [ -n "$ANTHROPIC_API_KEY" ]; then echo '{"loggedIn": true, "authMethod": "api_key", "apiKeySource": "ANTHROPIC_API_KEY"}'
    elif [ -e "$ROOT/console" ]; then echo '{"loggedIn": true, "authMethod": "api_key", "apiKeySource": "/login managed key"}'
    elif [ -e "$ROOT/logged-in" ]; then echo '{"loggedIn": true, "authMethod": "claude.ai"}'
    else echo '{"loggedIn": false}'; exit 1; fi ;;
  "auth login")
    [ ! -e "$ROOT/login-fails" ] || exit 1
    touch "$ROOT/logged-in" ;;
  " ")
    printf 'FIRST SCREEN: '
    read -r answer
    [ "$answer" = y ] || exit 0
    approved=$([ -n "$ANTHROPIC_API_KEY" ] && printf '"%s"' "${ANTHROPIC_API_KEY: -20}")
    printf '{"hasCompletedOnboarding": true, "projects": {"%s": {"hasTrustDialogAccepted": true}}, "customApiKeyResponses": {"approved": [%s], "rejected": []}}' "$(pwd -P)" "$approved" >"$CLAUDE_CONFIG_DIR/.claude.json"
    touch "$ROOT/logged-in" ;;
esac
"""


@pytest.fixture
def root() -> Iterator[Path]:
    root = Path(tempfile.mkdtemp(prefix="first-", dir="/tmp")).resolve()
    stand_in = root / "bin" / "claude"
    stand_in.parent.mkdir()
    stand_in.write_text(CLAUDE)
    stand_in.chmod(0o755)
    (root / "config").mkdir()
    yield root
    shutil.rmtree(root)


def home(root: Path) -> Home:
    return Home(root / "home")


def at_a_terminal(root: Path, typed_ahead: bytes, answer: bytes, key: str | None = None) -> tuple[int, str]:
    """Run `hands first-run` on a terminal holding typed_ahead before it starts, answer Claude Code's first screen with
    answer once it shows, and return its exit and everything the terminal showed."""
    controller, terminal = os.openpty()
    os.write(controller, typed_ahead)
    environment = {name: value for name, value in os.environ.items() if name != "ANTHROPIC_API_KEY"}
    environment |= {"PATH": f"{root / 'bin'}:/usr/bin:/bin", "ROOT": str(root), "HANDS_HOME": str(home(root).root), "CLAUDE_CONFIG_DIR": str(root / "config")}
    environment |= {} if key is None else {"ANTHROPIC_API_KEY": key}
    process = subprocess.Popen([sys.executable, "-m", "hands.daemon", "first-run"], env=environment, stdin=terminal, stdout=terminal, stderr=terminal, start_new_session=True)
    os.close(terminal)
    shown = b""
    answered = False
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        ready, _, _ = select.select([controller], [], [], 0.1)
        if not ready:
            if process.poll() is not None:
                break
            continue
        try:
            chunk = os.read(controller, 4096)
        except OSError:  # EIO: the last process holding the terminal let go of it
            break
        if not chunk:
            break
        shown += chunk
        if not answered and b"FIRST SCREEN: " in shown:
            os.write(controller, answer)
            answered = True
    exit = process.wait(timeout=10)
    os.close(controller)
    return exit, shown.decode().replace("\r\n", "\n")


def answered_before(root: Path, smoke: Path, keys: dict[str, list[str]] | None = None) -> None:
    """A Claude Code through its first run in smoke, with these answers on API keys, and logged in."""
    (root / "config" / ".claude.json").write_text(json.dumps({"hasCompletedOnboarding": True, "projects": {str(smoke): {"hasTrustDialogAccepted": True}}, "customApiKeyResponses": keys or {}}))
    (root / "logged-in").touch()


def events(root: Path) -> list[dict[str, Any]]:
    lines = (json.loads(line) for line in segment(home(root).audit, 0).read_text().splitlines())
    return [line for line in lines if line.get("event") == "claude.first_run"]


def calls(root: Path) -> list[str]:
    return (root / "calls").read_text().splitlines()


def test_a_fresh_claude_code_is_told_what_it_will_ask_then_asks_it_in_the_smoke_folder_and_a_key_typed_before_is_not_the_answer(root: Path) -> None:
    exit, shown = at_a_terminal(root, typed_ahead=b"n\n", answer=b"y\n")
    assert exit == 0, shown
    smoke = home(root).smoke.resolve()
    announced = f"Claude Code now starts in {home(root).smoke}, where `hands smoke` starts its session, and asks what it asks only once: a theme and a login; whether to trust {smoke}. Answer each, then type /exit"
    assert shown.index(announced) < shown.index("FIRST SCREEN: ")
    assert f"Claude Code is logged in and asks nothing first in {home(root).smoke}" in shown
    status = f"claude auth status in {Path.cwd().resolve()}"
    assert calls(root) == [status, f"claude  in {smoke}", status, status]
    # [LAW:nothing-unseen] its event: what was open before, what the first run exited with, and what was there after.
    [event] = events(root)
    facts = event["facts"]
    assert (event["outcome"], facts["first_run_exit"], "login_exit" in facts, facts["after"]["type"]) == ("ok", 0, False, "Ready")
    assert facts["before"]["unanswered"]["why"] == f"no {root / 'config' / '.claude.json'}"


def test_an_api_key_in_the_environment_is_named_among_the_questions(root: Path) -> None:
    exit, shown = at_a_terminal(root, typed_ahead=b"", answer=b"y\n", key="sk-ant-api03-0123456789abcdefghijklmn")
    assert exit == 0, shown
    assert "; whether to use the API key ANTHROPIC_API_KEY in your environment sets. Answer each, then type /exit" in shown


def test_a_claude_code_already_through_its_first_run_and_logged_in_is_asked_nothing(root: Path) -> None:
    answered_before(root, home(root).smoke.resolve())
    exit, shown = at_a_terminal(root, typed_ahead=b"", answer=b"y\n")
    assert exit == 0, shown
    assert "FIRST SCREEN" not in shown and "now starts" not in shown and "now logs in" not in shown
    assert [call.split(" in ")[0] for call in calls(root)] == ["claude auth status"] * 3
    [event] = events(root)
    assert (event["outcome"], "first_run_exit" in event["facts"], "login_exit" in event["facts"]) == ("ok", False, False)


def test_a_claude_code_through_its_first_run_but_logged_out_logs_in_with_its_own_login_alone(root: Path) -> None:
    answered_before(root, home(root).smoke.resolve())
    (root / "logged-in").unlink()
    exit, shown = at_a_terminal(root, typed_ahead=b"", answer=b"y\n")
    assert exit == 0, shown
    assert "FIRST SCREEN" not in shown and "Claude Code now logs in with its own login" in shown
    assert [call.split(" in ")[0] for call in calls(root)] == ["claude auth status", "claude auth status", "claude auth login", "claude auth status"]


def test_a_first_run_quit_before_its_last_question_fails_saying_a_second_run_asks_again_and_asks_no_login(root: Path) -> None:
    exit, shown = at_a_terminal(root, typed_ahead=b"", answer=b"q\n")
    assert exit == 1
    assert "hands first-run: Claude Code would first ask a theme and a login" in shown and "has no login" in shown
    assert shown.rstrip().endswith("running this again asks again")
    assert "claude auth login" not in "\n".join(calls(root))
    [event] = events(root)
    assert (event["outcome"], event["facts"]["first_run_exit"], "login_exit" in event["facts"]) == ("failed", 0, False)


def test_with_no_claude_code_to_ask_it_exits_2_and_asks_nothing(root: Path) -> None:
    (root / "bin" / "claude").unlink()
    exit, shown = at_a_terminal(root, typed_ahead=b"", answer=b"y\n")
    assert exit == 2
    assert "hands first-run: there is no Claude Code on this PATH" in shown and "now starts" not in shown


@pytest.mark.parametrize("onboarded", [False, True], ids=["first-run", "login-only"])
def test_with_no_terminal_to_ask_at_it_exits_2_and_never_starts_claude_code(root: Path, onboarded: bool) -> None:
    if onboarded:
        answered_before(root, home(root).smoke.resolve())
        (root / "logged-in").unlink()
    environment = {"PATH": f"{root / 'bin'}:/usr/bin:/bin", "ROOT": str(root), "HANDS_HOME": str(home(root).root), "CLAUDE_CONFIG_DIR": str(root / "config"), "HOME": str(root)}
    ran = subprocess.run([sys.executable, "-m", "hands.daemon", "first-run"], env=environment, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    assert ran.returncode == 2
    assert ran.stderr == "hands first-run: Claude Code asks its first-run questions and its login at a terminal, and this command's input is not one\n"
    assert [call.split(" in ")[0] for call in calls(root)] == ["claude auth status"]
    [event] = events(root)
    assert (event["outcome"], event["facts"]["terminal"]) == ("failed", False)


def test_an_empty_api_key_is_none_and_one_settings_sets_is_used_over_the_environments(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    key = "sk-ant-api03-0123456789abcdefghijklmn"
    assert firstrun.api_key(settings, {"ANTHROPIC_API_KEY": ""}) is None
    assert firstrun.api_key(settings, {"ANTHROPIC_API_KEY": key}) == firstrun.Key(key, "ANTHROPIC_API_KEY in your environment")
    # Claude Code puts settings.json's env over the process's, so an empty one there leaves it no key at all.
    settings.write_text(json.dumps({"env": {"ANTHROPIC_API_KEY": ""}}))
    assert firstrun.api_key(settings, {"ANTHROPIC_API_KEY": key}) is None
    settings.write_text(json.dumps({"env": {"ANTHROPIC_API_KEY": "sk-ant-api03-settings"}}))
    assert firstrun.api_key(settings, {"ANTHROPIC_API_KEY": key}) == firstrun.Key("sk-ant-api03-settings", str(settings))


def test_a_state_or_settings_hands_cannot_read_is_rejected_not_taken_for_a_first_run(tmp_path: Path) -> None:
    state = tmp_path / ".claude.json"
    state.mkdir()
    with pytest.raises(Rejected, match=f"^{state} unreadable: "):
        firstrun.recorded(state)
    settings = tmp_path / "settings.json"
    settings.mkdir()
    with pytest.raises(Rejected, match=f"^{settings} unreadable: "):
        firstrun.api_key(settings, {})


def test_the_state_is_in_the_config_directory_claude_config_dir_names_else_beside_it_in_the_home(tmp_path: Path) -> None:
    config = tmp_path / "cwd" / "relative"
    assert firstrun.state_of({"CLAUDE_CONFIG_DIR": "relative", "HOME": str(tmp_path)}, config) == config / ".claude.json"
    assert firstrun.state_of({"HOME": str(tmp_path)}, tmp_path / ".claude") == tmp_path / ".claude.json"


KEY = "sk-ant-api03-0123456789abcdefghijklmn"


def test_an_api_key_it_was_told_not_to_use_is_no_login_so_it_logs_in_with_its_own(root: Path) -> None:
    answered_before(root, home(root).smoke.resolve(), {"approved": [], "rejected": [KEY[-20:]]})
    (root / "logged-in").unlink()
    exit, shown = at_a_terminal(root, typed_ahead=b"", answer=b"y\n", key=KEY)
    assert exit == 0, shown
    assert "FIRST SCREEN" not in shown and "Claude Code now logs in with its own login" in shown
    assert [call.split(" in ")[0] for call in calls(root)] == ["claude auth status", "claude auth status", "claude auth login", "claude auth status"]


def test_an_api_key_it_was_told_to_use_is_its_login_and_nothing_is_asked(root: Path) -> None:
    answered_before(root, home(root).smoke.resolve(), {"approved": [KEY[-20:]], "rejected": []})
    (root / "logged-in").unlink()
    exit, shown = at_a_terminal(root, typed_ahead=b"", answer=b"y\n", key=KEY)
    assert exit == 0, shown
    assert "FIRST SCREEN" not in shown and "now logs in" not in shown


def test_a_console_login_is_a_login_though_it_is_an_api_key(root: Path) -> None:
    answered_before(root, home(root).smoke.resolve())
    (root / "logged-in").unlink()
    (root / "console").touch()
    exit, shown = at_a_terminal(root, typed_ahead=b"", answer=b"y\n")
    assert exit == 0, shown
    assert "now logs in" not in shown


def test_a_project_entry_this_cannot_read_is_no_trust_and_leaves_the_rest_read(tmp_path: Path) -> None:
    state = tmp_path / ".claude.json"
    state.write_text(json.dumps({"hasCompletedOnboarding": True, "projects": {"/elsewhere": [], str(tmp_path): {"hasTrustDialogAccepted": True}, "/odd": {"hasTrustDialogAccepted": "yes"}}}))
    assert firstrun.recorded(state) == firstrun.Recorded(True, frozenset({str(tmp_path)}), frozenset(), frozenset())
    assert firstrun.unanswered(firstrun.recorded(state), tmp_path / "below", None) is None
