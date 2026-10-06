"""`hands start-session`: `claude` started in a tmux window of the tmux session named for its folder, as from a terminal
outside any session, and the session that joined hands named. Driven through a tmux server of the test's own, with a
`claude` that joins as the plugin's first hook does."""

import json
import shutil
import subprocess
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from hands.brain.process import GIVEN, SLIM, environment
from hands.daemon.cli import main
from hands.daemon.startsession import SESSION_GIVEN, as_from_a_terminal, tmux_name
from hands.sessions.audit import segment
from hands.sessions.home import Home

TMUX = shutil.which("tmux")
needs_tmux = pytest.mark.skipif(TMUX is None, reason="no tmux to start sessions in")

# A `claude` that joins hands as the plugin's first hook does, writing its membership where HANDS_HOME's are kept, under
# fritter but in a folder holding `unwrapped`; it notes its arguments beside it, and stays running. In a folder holding
# `never-joins`, it ends before it joins. A window takes the environment of the tmux server it opens in, so what varies
# a start is its folder, never a variable.
CLAUDE = """#!/bin/sh
for argument; do echo "$argument"; done > "$PWD/claude-args"
[ -e "$PWD/never-joins" ] && exit 3
socket=/tmp/fritter.sock
[ -e "$PWD/unwrapped" ] && socket=
printf '{"pid": %d, "cwd": "%s", "transcript_path": "%s/t.jsonl", "fritter_socket": "%s"}' $$ "$PWD" "$PWD" "$socket" > "$HANDS_HOME/sessions/s$$.writing"
mv "$HANDS_HOME/sessions/s$$.writing" "$HANDS_HOME/sessions/s$$.json"
exec sleep 120
"""


@pytest.fixture
def terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path]:
    """A terminal whose PATH has the joining `claude` first and whose tmux is a server of the test's own; its bin."""
    assert TMUX is not None
    bin = tmp_path / "bin"
    bin.mkdir()
    (bin / "claude").write_text(CLAUDE)
    (bin / "claude").chmod(0o755)
    # The test's tmux server reads no tmux.conf of the user's.
    (bin / "tmux").write_text(f'#!/bin/sh\nexec {TMUX} -f /dev/null "$@"\n')
    (bin / "tmux").chmod(0o755)
    # A tmux socket's path has to fit a sockaddr_un, which a pytest tmp_path on macOS does not.
    sockets = Path(tempfile.mkdtemp(prefix="hands-tmux-", dir="/tmp"))
    monkeypatch.setenv("PATH", f"{bin}:/usr/bin:/bin")
    monkeypatch.setenv("TMUX_TMPDIR", str(sockets))
    # Never the user's own tmux server, which TMUX would name.
    monkeypatch.delenv("TMUX", raising=False)
    # The user's tmux server, already running, started from a terminal whose sessions report to another home.
    subprocess.run([TMUX, "-f", "/dev/null", "new-session", "-d", "-s", "theirs", "-e", "HANDS_HOME=/elsewhere", "sleep", "120"], check=True)
    subprocess.run([TMUX, "set-environment", "-g", "HANDS_HOME", "/elsewhere"], check=True)
    try:
        yield bin
    finally:
        # Each session of the test's own server ended by name, and the server with its last one.
        names = subprocess.run([TMUX, "list-sessions", "-F", "#{session_name}"], capture_output=True, text=True, check=False).stdout.split()
        for name in names:
            subprocess.run([TMUX, "kill-session", "-t", f"={name}"], capture_output=True, check=True)
        shutil.rmtree(sockets)


def home_in(tmp_path: Path) -> Home:
    home = Home(tmp_path / "home")
    home.memberships.mkdir(parents=True)
    return home


def started(home: Home) -> list[dict[str, Any]]:
    return [line for line in map(json.loads, segment(home.audit, 0).read_text().splitlines()) if line.get("event") == "session.start"]


def tmux(*arguments: str) -> str:
    assert TMUX is not None
    return subprocess.run([TMUX, *arguments], capture_output=True, text=True, check=True).stdout


@needs_tmux
def test_a_session_is_started_in_the_tmux_session_named_for_its_folder_and_named_once_it_joined(tmp_path: Path, terminal: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = home_in(tmp_path)
    folder = tmp_path / "billing.api"
    folder.mkdir()
    assert main(["--home", str(home.root), "start-session", str(folder), "--model", "opus"]) == 0
    assert main(["--home", str(home.root), "start-session", str(folder)]) == 0
    first, second = started(home)
    out = capsys.readouterr().out.splitlines()

    # The first start made the tmux session, named as tmux spells the folder; the second opened a window beside it.
    assert tmux_name(folder) == "billing_api"
    assert tmux("list-windows", "-t", "=billing_api", "-F", "#{pane_current_path}").split() == [str(folder.resolve())] * 2
    assert (first["outcome"], first["facts"]["made_tmux_session"], second["facts"]["made_tmux_session"]) == ("ok", True, False)
    # [LAW:nothing-unseen] which session joined, where, and on what model, on the start's event and in what it printed.
    for event, line in ((first, out[0]), (second, out[1])):
        assert event["facts"]["session"].startswith("s") and event["facts"]["session"] != (second if event is first else first)["facts"]["session"]
        assert line == f"Session {event['facts']['session']} joined hands, in tmux pane {event['facts']['pane']} of tmux session billing_api."
        assert (event["facts"]["tmux_session"], event["facts"]["under_fritter"], event["facts"]["folder"]) == ("billing_api", True, str(folder))
    assert (first["facts"]["model"], second["facts"]["model"]) == ("opus", None)
    # Only the model it was asked for, as one argument.
    assert (folder / "claude-args").read_text() == ""


@needs_tmux
def test_the_model_reaches_claude_as_one_argument(tmp_path: Path, terminal: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = home_in(tmp_path)
    folder = tmp_path / "work"
    folder.mkdir()
    assert main(["--home", str(home.root), "start-session", str(folder), "--model", "claude-opus-5-5"]) == 0
    capsys.readouterr()
    assert (folder / "claude-args").read_text() == "--model=claude-opus-5-5\n"


@needs_tmux
def test_a_session_not_under_fritter_and_one_that_ended_are_said_and_exit_one(tmp_path: Path, terminal: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = home_in(tmp_path)
    folder = tmp_path / "work"
    folder.mkdir()
    (folder / "unwrapped").touch()
    assert main(["--home", str(home.root), "start-session", str(folder)]) == 1
    assert "not under fritter, so hands cannot type into it" in capsys.readouterr().err
    (folder / "never-joins").touch()
    assert main(["--home", str(home.root), "start-session", str(folder)]) == 1
    assert "ended before it joined hands" in capsys.readouterr().err
    unfrittered, ended = started(home)
    assert (unfrittered["outcome"], unfrittered["facts"]["under_fritter"]) == ("failed", False)
    assert ended["outcome"] == "failed" and "session" not in ended["facts"] and ended["facts"]["made_tmux_session"] is False


def test_a_folder_that_is_not_there_is_said(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = home_in(tmp_path)
    assert main(["--home", str(home.root), "start-session", str(tmp_path / "nowhere")]) == 1
    assert f"there is no folder {tmp_path / 'nowhere'}" in capsys.readouterr().err
    assert started(home)[0]["outcome"] == "failed"


def test_what_the_brain_gives_its_own_claude_code_is_exactly_what_it_adds_to_hands_environment(tmp_path: Path) -> None:
    assert set(environment(tmp_path, "http://127.0.0.1:9", {"PATH": "/usr/bin"})) - {"PATH"} == set(GIVEN)


def test_a_session_started_from_the_brains_shell_is_the_users_not_the_brains(tmp_path: Path) -> None:
    home = Home(tmp_path)
    users = {"PATH": "/Users/me/.hands/bin:/usr/bin", "CLAUDE_CODE_USE_BEDROCK": "1", "TMUX": "/tmp/tmux-501/default,1,0"}
    # The brain's Bash: hands' environment, given what makes it the brain, inside a Claude Code under fritter.
    inside_brain = {**environment(home.brain, "http://127.0.0.1:9", users), **{name: "given" for name in SESSION_GIVEN}}
    assert as_from_a_terminal(inside_brain, home) == {**users, "HANDS_HOME": str(home.root)}
    # A user's own setup keeps its CLAUDE_CONFIG_DIR, and a setting that shares a name with what the brain is given.
    own = {**users, "CLAUDE_CONFIG_DIR": "/Users/me/.claude.work", **SLIM}
    assert as_from_a_terminal(own, home) == {**own, "HANDS_HOME": str(home.root)}
