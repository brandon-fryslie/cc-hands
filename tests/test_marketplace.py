"""`hands plugin`: the command hands' marketplace entry has Claude Code run, which prints the plugin, written for the
interpreter of the hands that ran it."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hands.daemon.cli import main
from hands.sessions.audit import segment
from hands.sessions.hookconfig import LAUNCHER
from hands.sessions import marketplace
from hands.sessions.home import Home
from hands.sessions.marketplace import PACKAGED, launcher, render


def packaged_files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_the_command_prints_the_plugin_once_written_and_the_same_one_after(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path / "home")
    assert main(["--home", str(home.root), "plugin"]) == 0
    printed = capsys.readouterr().out
    # Claude Code takes one line, an absolute path, as the plugin's directory.
    [line] = printed.splitlines()
    plugin = Path(line)
    assert plugin.is_absolute() and plugin.parent == home.plugins
    # Every file the package carries, and the launcher, which runs this hands' interpreter.
    assert packaged_files(plugin) == {**packaged_files(PACKAGED), LAUNCHER: launcher(sys.executable).encode()}
    assert os.access(plugin / LAUNCHER, os.X_OK)

    assert main(["--home", str(home.root), "plugin"]) == 0
    assert capsys.readouterr().out == printed
    # Nothing left beside it: no second copy, no half-staged one.
    assert list(home.plugins.iterdir()) == [plugin]
    # [LAW:nothing-unseen] each run's event: the interpreter, what it copied, what it printed, and whether it wrote it.
    events = [line for line in (json.loads(line) for line in segment(home.audit, 0).read_text().splitlines()) if line["type"] == "WideEvent"]
    facts = {"interpreter": sys.executable, "packaged": str(PACKAGED), "plugin": str(plugin)}
    assert [(event["event"], event["outcome"], event["facts"]) for event in events] == [
        ("plugin.render", "ok", {**facts, "written": True}),
        ("plugin.render", "ok", {**facts, "written": False}),
    ]


def test_a_config_the_daemon_rejects_still_renders_the_plugin(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path / "home")
    home.root.mkdir()
    home.config.write_text("[llm]\nbackend = 'local'\n")
    assert main(["--home", str(home.root), "plugin"]) == 0
    assert Path(capsys.readouterr().out.strip()).parent == home.plugins


def test_another_interpreter_is_another_plugin(tmp_path: Path) -> None:
    home = Home(tmp_path)
    here = render(home, sys.executable)
    there = render(home, "/Users/someone else/.local/share/uv/tools/hands/bin/python")
    assert here.plugin != there.plugin and there.written
    assert (there.plugin / LAUNCHER).read_text() == launcher("/Users/someone else/.local/share/uv/tools/hands/bin/python")


def test_a_launcher_whose_interpreter_is_gone_says_which_and_how_to_bring_the_plugin_back(tmp_path: Path) -> None:
    gone = tmp_path / "uninstalled hands" / "bin" / "python"
    plugin = render(Home(tmp_path / "home"), str(gone)).plugin
    ran = subprocess.run([plugin / LAUNCHER, "-m", "hands.sessions.shim"], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}, check=False)
    assert (ran.returncode, ran.stdout) == (1, "")
    assert ran.stderr == f"hands: the plugin runs {gone}, which is gone; install hands, then run `claude plugin update hands@cc-hands`\n"


def test_a_render_that_fails_leaves_nothing_staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)

    def full_disk(root: Path) -> str:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(marketplace, "digest", full_disk)
    with pytest.raises(OSError, match="No space left"):
        render(home, sys.executable)
    assert list(home.plugins.iterdir()) == []


def test_a_plugin_another_session_rendered_first_is_the_one_printed(tmp_path: Path) -> None:
    home = Home(tmp_path)
    first = render(home, sys.executable)
    (first.plugin / "marker").write_text("the one already there")
    # The same files, staged by a session that started alongside: it is not renamed over the one already there.
    again = render(home, sys.executable)
    assert (again.plugin, again.written) == (first.plugin, False)
    assert (first.plugin / "marker").read_text() == "the one already there"
    assert list(home.plugins.iterdir()) == [first.plugin]
