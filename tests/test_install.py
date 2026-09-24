"""`hands install-hooks`: merged into a settings file, idempotently, touching nothing that is not hands'."""

import json
import os
from pathlib import Path
from typing import cast

import pytest

from hands.daemon.cli import main
from hands.sessions.home import Home
from hands.sessions.hookconfig import SUBSCRIBED, hook_settings
from hands.sessions.install import install, is_hands_entry, merged
from hands.sessions.payload import Rejected

PYTHON = Path("/venv/bin/python")
HOME = Home(Path("/Users/me/.hands"))
SHIM = "/venv/bin/python -m hands.sessions.shim /Users/me/.hands"
THEIRS = {"type": "command", "command": "~/bin/notify-me"}


def hooks_of(settings: dict[str, object]) -> dict[str, object]:
    hooks = settings["hooks"]
    assert isinstance(hooks, dict)
    return cast(dict[str, object], hooks)


def declared() -> dict[str, list[object]]:
    hooks = hook_settings(PYTHON, HOME)["hooks"]
    assert isinstance(hooks, dict)
    return hooks  # pyright: ignore[reportUnknownVariableType]


@pytest.mark.parametrize(
    ("entry", "ours"),
    [
        ({"type": "command", "command": SHIM}, True),
        ({"type": "command", "command": SHIM, "timeout": 90, "async": True}, True),
        ({"type": "command", "command": "/old/venv/bin/python -m hands.sessions.shim '/Users/me/my hands'"}, True),
        ({"type": "command", "command": f"env X=1 {SHIM}"}, False),  # wrapped: its pid would not be the session's
        ({"type": "command", "command": f"{SHIM}; echo done"}, False),  # compound
        ({"type": "command", "command": f"{SHIM} --verbose"}, False),
        ({"type": "command", "command": "/venv/bin/python -m hands.sessions.shimmer /Users/me/.hands"}, False),
        ({"type": "command", "command": "python -m 'unbalanced"}, False),
        ({"type": "http", "url": "http://127.0.0.1:1/hook"}, False),
        ({"type": "command", "command": 42}, False),
        ("a string", False),
        (THEIRS, False),
    ],
)
def test_an_entry_is_hands_only_when_it_is_the_shim_run_as_hookconfig_builds_it(entry: object, ours: bool) -> None:
    assert is_hands_entry(entry) is ours


def test_installing_into_nothing_gives_exactly_the_declared_hooks() -> None:
    assert merged({}, declared()) == {"hooks": declared()}


def test_installing_twice_changes_nothing_the_second_time() -> None:
    once = merged({"model": "opus", "hooks": {"Stop": [{"hooks": [THEIRS]}]}}, declared())
    assert merged(once, declared()) == once


def test_what_is_not_hands_is_kept_where_it_was() -> None:
    settings: dict[str, object] = {
        "model": "opus",
        "hooks": {
            "PreToolUse": [{"matcher": "Bash", "hooks": [THEIRS]}],
            "Stop": [{"hooks": [THEIRS, {"type": "command", "command": SHIM}]}],  # theirs and ours in one group
            "Notification": [],
        },
    }
    after = merged(settings, declared())
    hooks = hooks_of(after)
    assert after["model"] == "opus"
    assert hooks["PreToolUse"] == [{"matcher": "Bash", "hooks": [THEIRS]}]
    assert hooks["Notification"] == []
    assert hooks["Stop"] == [{"hooks": [THEIRS]}, *declared()["Stop"]]
    assert list(hooks)[:3] == ["PreToolUse", "Stop", "Notification"]  # the file's own order, hands' new events after


def test_a_moved_venv_replaces_the_old_command_and_an_unsubscribed_event_loses_its_entry() -> None:
    old = {"type": "command", "command": "/old/venv/bin/python -m hands.sessions.shim /Users/me/.hands"}
    settings: dict[str, object] = {"hooks": {"Stop": [{"hooks": [old]}], "TeammateIdle": [{"hooks": [old]}]}}
    hooks = hooks_of(merged(settings, declared()))
    assert hooks["Stop"] == declared()["Stop"]
    assert "TeammateIdle" not in hooks
    assert set(hooks) == set(SUBSCRIBED)


def test_the_file_is_written_once_and_a_second_run_reports_nothing_to_do(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"model": "opus"}, indent=2) + "\n")
    settings.chmod(0o600)
    first = install(settings, PYTHON, HOME)
    assert first.diff.count("hands.sessions.shim") == len(SUBSCRIBED)
    assert json.loads(settings.read_text())["model"] == "opus"
    assert settings.stat().st_mode & 0o777 == 0o600
    second = install(settings, PYTHON, HOME)
    assert second.diff == ""
    assert [path.name for path in tmp_path.iterdir()] == ["settings.json"]  # nothing left of the replacement


def test_a_settings_file_that_is_a_symlink_stays_one(tmp_path: Path) -> None:
    kept = tmp_path / "dotfiles" / "settings.json"
    kept.parent.mkdir()
    kept.write_text("{}\n")
    link = tmp_path / "settings.json"
    link.symlink_to(kept)
    assert install(link, PYTHON, HOME).path == kept
    assert link.is_symlink()
    assert set(json.loads(kept.read_text())["hooks"]) == set(SUBSCRIBED)


@pytest.mark.parametrize(("text", "error"), [("{", "not JSON"), ("[]", "should be a JSON object"), ('{"hooks": []}', "should be a JSON object"), ('{"hooks": {"Stop": {}}}', "should be a list")])
def test_a_settings_file_it_cannot_read_is_refused_and_left_alone(text: str, error: str, tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(text)
    with pytest.raises(Rejected, match=error):
        install(settings, PYTHON, HOME)
    assert settings.read_text() == text


def test_the_command_prints_the_diff_then_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    settings = tmp_path / "claude" / "settings.json"
    assert main(["--home", str(tmp_path / "hands"), "install-hooks", "--settings", str(settings)]) == 0
    first = capsys.readouterr()
    assert "+" in first.out and "updated" in first.err
    assert main(["--home", str(tmp_path / "hands"), "install-hooks", "--settings", str(settings)]) == 0
    second = capsys.readouterr()
    assert second.out == "" and "already current" in second.err
    settings.write_text("{")
    assert main(["--home", str(tmp_path / "hands"), "install-hooks", "--settings", str(settings)]) == 2
    assert "not JSON" in capsys.readouterr().err
    assert os.path.getsize(settings) == 1
