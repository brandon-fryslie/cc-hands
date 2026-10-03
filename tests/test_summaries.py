"""The summaries switch: off until turned on, kept in the home, and set by the plugin's skill under the plugin's own Python."""

import subprocess
from pathlib import Path

import pytest

from hands.sessions.hookconfig import LAUNCHER, PLUGIN_DIR
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.sessions.summaries import set_summaries, summaries

PLUGIN_ROOT = Path(__file__).resolve().parent.parent / PLUGIN_DIR


def test_summaries_are_off_until_turned_on(tmp_path: Path) -> None:
    home = Home(tmp_path / "home")
    assert summaries(home) == "off"
    set_summaries(home, "on")
    assert summaries(home) == "on"
    set_summaries(home, "off")
    assert summaries(home) == "off"


@pytest.mark.parametrize("written", [b"yes\n", b"", b"\xffon"])
def test_a_switch_that_says_neither_is_refused_rather_than_read_as_the_default(tmp_path: Path, written: bytes) -> None:
    home = Home(tmp_path)
    home.summaries.write_bytes(written)
    with pytest.raises(Rejected, match="neither on nor off"):
        summaries(home)


def switch(home: Home, cwd: Path, path: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    """`/hands:summaries` as the skill runs it: the plugin's launcher, in the session's directory, with no venv."""
    environment = {"HANDS_HOME": str(home.root), "PATH": path, "HOME": str(cwd)}
    return subprocess.run([PLUGIN_ROOT / LAUNCHER, "-m", "hands.sessions.summaries", *arguments], env=environment, cwd=cwd, capture_output=True, text=True)


def test_the_skill_turns_summaries_on_and_off_and_says_where_they_stand(tmp_path: Path, python312: str) -> None:
    home = Home(tmp_path / "home")
    asked = switch(home, tmp_path, python312)
    assert (asked.returncode, asked.stdout, asked.stderr) == (0, "Spoken turn summaries are off: only a watched session's turns are told as they finish, and any session's last turn when you ask for it.\n", "")
    on = switch(home, tmp_path, python312, "on")
    assert (on.returncode, on.stdout, on.stderr) == (0, "Spoken turn summaries are on: every turn a session finishes is told aloud.\n", "")
    assert summaries(home) == "on"
    assert switch(home, tmp_path, python312, "Off").returncode == 0
    assert summaries(home) == "off"


def test_the_skill_refuses_anything_but_on_or_off_and_leaves_the_switch_alone(tmp_path: Path, python312: str) -> None:
    home = Home(tmp_path / "home")
    set_summaries(home, "on")
    refused = switch(home, tmp_path, python312, "maybe")
    assert (refused.returncode, refused.stdout, refused.stderr) == (2, "", "hands summaries: expected on or off, got 'maybe'\n")
    assert summaries(home) == "on"
