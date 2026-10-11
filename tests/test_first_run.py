"""`hands first-run`, as install.sh runs it at a terminal: what the person's own Claude Code asks only once is said
before Claude Code shows it, asked in the folder `hands smoke` starts its session in, and not asked again once answered.

Claude Code is the fake built from recordings (claudecode_fake), with a person at its terminal who answers what each
test chooses. A `claude` on PATH is there only to be found: the fake is what answers for it.
"""

import json
import os
import select
import sys
from pathlib import Path
from typing import Any

import pytest

from claudecode_fake import Config, Fake, Person
from hands.daemon import cli
from hands.sessions import firstrun
from hands.sessions.audit import segment
from hands.sessions.home import Home
from hands.sessions.payload import Rejected

KEY = "sk-ant-api03-0123456789abcdefghijklmn"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A PATH with a `claude` to find, and the person's Claude Code under a config directory of the test's own."""
    root = tmp_path.resolve()
    found = root / "bin" / "claude"
    found.parent.mkdir()
    found.write_text("#!/bin/sh\n")
    found.chmod(0o755)
    monkeypatch.setenv("PATH", f"{root / 'bin'}:/usr/bin:/bin")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(root / "config"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return root


def home(root: Path) -> Home:
    return Home(root / "home")


def config(root: Path) -> Path:
    return root / "config"


def first_run(root: Path, fake: Fake, terminal: bool = True) -> int:
    return cli.first_run(fake, home(root), terminal, fake.told, cli.audit_log_of(home(root)).record)


def answered_before(root: Path, fake: Fake, keys: dict[str, list[str]] | None = None, login: bool = True) -> None:
    """A Claude Code through its first run in the smoke folder, with these answers on API keys, and logged in unless not `login`."""
    smoke = home(root).smoke.resolve()
    state = {"hasCompletedOnboarding": True, "projects": {str(smoke): {"hasTrustDialogAccepted": True}}, "customApiKeyResponses": keys or {}}
    fake.files[config(root) / ".claude.json"] = json.dumps(state).encode()
    fake.configs[config(root)] = Config(login="claude.ai" if login else None)


def events(root: Path) -> list[dict[str, Any]]:
    lines = (json.loads(line) for line in segment(home(root).audit, 0).read_text().splitlines())
    return [line for line in lines if line.get("event") == "claude.first_run"]


def asked(fake: Fake) -> list[str]:
    """What hands asked of Claude Code, in order, without what it told the person or what Claude Code showed them."""
    return [line for line in fake.transcript if not line.startswith(("told: ", "shown: "))]


def test_a_fresh_claude_code_is_told_what_it_will_ask_then_asks_it_in_the_smoke_folder(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = Fake()
    assert first_run(root, fake) == 0
    smoke = home(root).smoke.resolve()
    announced = f"told: Claude Code now starts in {home(root).smoke}, where `hands smoke` starts its session, and asks what it asks only once: a theme and a login; whether to trust {smoke}. Answer each, then type /exit"
    # Said before Claude Code shows its first screen, and the screens are the ones a fresh Claude Code shows, in order.
    started = fake.transcript.index(announced)
    assert fake.transcript[started + 1] == f"first run in {home(root).smoke}"
    assert [line for line in fake.transcript if line.startswith("shown: ")] == [f"shown: {screen}" for screen in screens("theme", "login", "notes", "trust")]
    assert asked(fake) == ["auth status", f"first run in {home(root).smoke}", "auth status", "auth status"]
    assert f"Claude Code is logged in and asks nothing first in {home(root).smoke}" in capsys.readouterr().out
    # [LAW:nothing-unseen] its event: what was open before, what the first run exited with, and what was there after.
    [event] = events(root)
    facts = event["facts"]
    assert (event["outcome"], facts["first_run_exit"], "login_exit" in facts, facts["after"]["type"], facts["terminal"]) == ("ok", 0, False, "Ready", True)
    assert facts["before"]["unanswered"]["why"] == f"no {config(root) / '.claude.json'}"


def screens(*names: str) -> list[str]:
    from claudecode_fake import SCREENS

    return [SCREENS[name] for name in names]


def test_an_api_key_in_the_environment_is_named_among_the_questions_and_asked_in_place_of_a_login(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    fake = Fake()
    assert first_run(root, fake) == 0
    [announced] = [line for line in fake.transcript if line.startswith("told: ")]
    assert announced.endswith("; whether to use the API key ANTHROPIC_API_KEY in your environment sets. Answer each, then type /exit")
    assert [line for line in fake.transcript if line.startswith("shown: ")] == [f"shown: {screen}" for screen in screens("theme", "api_key", "notes", "trust")]


def test_a_claude_code_already_through_its_first_run_and_logged_in_is_asked_nothing(root: Path) -> None:
    fake = Fake()
    answered_before(root, fake)
    assert first_run(root, fake) == 0
    assert asked(fake) == ["auth status"] * 3 and not [line for line in fake.transcript if line.startswith(("told: ", "shown: "))]
    [event] = events(root)
    assert (event["outcome"], "first_run_exit" in event["facts"], "login_exit" in event["facts"]) == ("ok", False, False)


def test_a_claude_code_through_its_first_run_but_logged_out_logs_in_with_its_own_login_alone(root: Path) -> None:
    fake = Fake()
    answered_before(root, fake, login=False)
    assert first_run(root, fake) == 0
    assert "told: Claude Code now logs in with its own login: it opens your browser, or prints a link to open, for your Claude account" in fake.transcript
    assert asked(fake) == ["auth status", "auth status", "auth login", "auth status"]


def test_a_first_run_quit_before_its_last_question_fails_saying_a_second_run_asks_again_and_asks_no_login(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = Fake(Person(first_run="quits"))
    assert first_run(root, fake) == 1
    said = capsys.readouterr().err
    assert "hands first-run: Claude Code would first ask a theme and a login" in said and "has no login" in said
    assert said.rstrip().endswith("running this again asks again")
    assert "auth login" not in asked(fake)
    [event] = events(root)
    assert (event["outcome"], event["facts"]["first_run_exit"], "login_exit" in event["facts"]) == ("failed", 0, False)


def test_a_login_abandoned_on_its_first_run_fails_as_a_run_quit_part_way(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = Fake(Person(login="abandons"))
    assert first_run(root, fake) == 1
    assert [line for line in fake.transcript if line.startswith("shown: ")] == [f"shown: {screen}" for screen in screens("theme", "login")]
    assert capsys.readouterr().err.rstrip().endswith("running this again asks again")


def test_with_no_claude_code_to_ask_it_exits_2_and_asks_nothing(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (root / "bin" / "claude").unlink()
    fake = Fake()
    assert first_run(root, fake) == 2
    assert "hands first-run: there is no Claude Code on this PATH" in capsys.readouterr().err and fake.transcript == []


def test_a_claude_code_that_cannot_be_started_exits_2(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = Fake(unrunnable={"first run"})
    assert first_run(root, fake) == 2
    assert "hands first-run: Claude Code could not be started to ask: `claude first run` could not be run" in capsys.readouterr().err


@pytest.mark.parametrize("onboarded", [False, True], ids=["first-run", "login-only"])
def test_with_no_terminal_to_ask_at_it_exits_2_and_never_starts_claude_code(root: Path, capsys: pytest.CaptureFixture[str], onboarded: bool) -> None:
    fake = Fake()
    if onboarded:
        answered_before(root, fake, login=False)
    assert first_run(root, fake, terminal=False) == 2
    assert capsys.readouterr().err == "hands first-run: Claude Code asks its first-run questions and its login at a terminal, and this command's input is not one\n"
    assert asked(fake) == ["auth status"]
    [event] = events(root)
    assert (event["outcome"], event["facts"]["terminal"]) == ("failed", False)


def test_an_api_key_it_was_told_not_to_use_is_no_login_so_it_logs_in_with_its_own(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    fake = Fake()
    answered_before(root, fake, {"approved": [], "rejected": [KEY[-20:]]}, login=False)
    assert first_run(root, fake) == 0
    assert asked(fake) == ["auth status", "auth status", "auth login", "auth status"]


def test_an_api_key_it_was_told_to_use_is_its_login_and_nothing_is_asked(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    fake = Fake()
    answered_before(root, fake, {"approved": [KEY[-20:]], "rejected": []}, login=False)
    assert first_run(root, fake) == 0
    assert asked(fake) == ["auth status"] * 3


def test_a_console_login_is_a_login_though_it_is_an_api_key(root: Path) -> None:
    fake = Fake()
    answered_before(root, fake)
    fake.configs[config(root)] = Config(login="console")
    assert first_run(root, fake) == 0
    assert "auth login" not in asked(fake)


def test_typed_ahead_keys_are_dropped_before_claude_code_asks(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # An n pressed during the install's earlier steps sits in the terminal; it must not reach Claude Code's question.
    controller, terminal = os.openpty()
    try:
        with open(terminal, "r", closefd=False) as stdin:
            monkeypatch.setattr(sys, "stdin", stdin)
            os.write(controller, b"n\n")
            cli.told("Claude Code now asks")
            assert select.select([terminal], [], [], 0.2)[0] == []
        assert capsys.readouterr().out == "Claude Code now asks\n"
    finally:
        os.close(controller)
        os.close(terminal)


def test_an_empty_api_key_is_none_and_one_settings_sets_is_used_over_the_environments(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    fake = Fake()
    assert firstrun.api_key(fake, settings, {"ANTHROPIC_API_KEY": ""}) is None
    assert firstrun.api_key(fake, settings, {"ANTHROPIC_API_KEY": KEY}) == firstrun.Key(KEY, "ANTHROPIC_API_KEY in your environment")
    # Claude Code puts settings.json's env over the process's, so an empty one there leaves it no key at all.
    fake.files[settings] = json.dumps({"env": {"ANTHROPIC_API_KEY": ""}}).encode()
    assert firstrun.api_key(fake, settings, {"ANTHROPIC_API_KEY": KEY}) is None
    fake.files[settings] = json.dumps({"env": {"ANTHROPIC_API_KEY": "sk-ant-api03-settings"}}).encode()
    assert firstrun.api_key(fake, settings, {"ANTHROPIC_API_KEY": KEY}) == firstrun.Key("sk-ant-api03-settings", str(settings))


def test_a_state_or_settings_hands_cannot_read_is_rejected_not_taken_for_a_first_run(tmp_path: Path) -> None:
    state, settings = tmp_path / ".claude.json", tmp_path / "settings.json"
    fake = Fake(unreadable={state, settings})
    with pytest.raises(Rejected, match=f"^{state} unreadable: "):
        firstrun.recorded(fake, state)
    with pytest.raises(Rejected, match=f"^{settings} unreadable: "):
        firstrun.api_key(fake, settings, {})


def test_the_state_is_in_the_config_directory_claude_config_dir_names_else_beside_it_in_the_home(tmp_path: Path) -> None:
    config = tmp_path / "cwd" / "relative"
    assert firstrun.state_of({"CLAUDE_CONFIG_DIR": "relative", "HOME": str(tmp_path)}, config) == config / ".claude.json"
    assert firstrun.state_of({"HOME": str(tmp_path)}, tmp_path / ".claude") == tmp_path / ".claude.json"


def test_a_project_entry_this_cannot_read_is_no_trust_and_leaves_the_rest_read(tmp_path: Path) -> None:
    state = tmp_path / ".claude.json"
    fake = Fake(files={state: json.dumps({"hasCompletedOnboarding": True, "projects": {"/elsewhere": [], str(tmp_path): {"hasTrustDialogAccepted": True}, "/odd": {"hasTrustDialogAccepted": "yes"}}}).encode()})
    assert firstrun.recorded(fake, state) == firstrun.Recorded(True, frozenset({str(tmp_path)}), frozenset(), frozenset())
    assert firstrun.unanswered(firstrun.recorded(fake, state), tmp_path / "below", None) is None


def test_the_recorded_states_read_as_claude_code_wrote_them() -> None:
    # The fake's states are a real Claude Code's: one started and quit records nothing of a first run; one through its
    # notes has finished its onboarding, with the key it was told to use; one through the trust question trusts /work.
    from claudecode_fake import STATES

    def read(name: str) -> firstrun.Recorded | firstrun.Blank:
        return firstrun.recorded_in(json.dumps(STATES[name]).encode(), Path("/config/.claude.json"))

    assert read("started") == firstrun.Recorded(False, frozenset(), frozenset(), frozenset())
    assert read("onboarded") == firstrun.Recorded(True, frozenset(), frozenset({"AAAAAAAAAAAAAAAAAAAA"}), frozenset())
    assert read("trusted") == firstrun.Recorded(True, frozenset({"/work"}), frozenset({"AAAAAAAAAAAAAAAAAAAA"}), frozenset())
