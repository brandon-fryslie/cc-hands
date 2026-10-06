"""The `start_session` tool: `claude` started in a tmux window of the tmux session named for its folder, as from a terminal
outside any session, and the session that joined hands named. Driven through a tmux server of the test's own, with a
`claude` in the home's bin that joins as the plugin's first hook does, and the home's memberships as the registry."""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pytest

from hands.core.events import Joined
from hands.core.session import Membership, SessionId
from hands.sessions import startsession
from hands.sessions.audit import AuditLog, Entry, segment
from hands.sessions.home import Home
from hands.sessions.membership import parse_membership
from hands.sessions.overlays import Overlays
from hands.sessions.registry import Sessions
from hands.sessions.wide import WideEvent
from hands.sessions.startsession import SESSION_GIVEN, as_from_a_terminal, descends, tmux_name
from hands.sessions.terminals import Process
from hands.voice.tool import Result
from hands.voice.tools import Called, audited, list_sessions_tool, start_session_tool

TMUX = shutil.which("tmux")
needs_tmux = pytest.mark.skipif(TMUX is None, reason="no tmux to start sessions in")

# A `claude` that joins hands as the plugin's first hook does, writing its membership where HANDS_HOME's are kept, under
# fritter but in a folder holding `unwrapped`; it notes its arguments and environment beside it, and stays running. In a
# folder holding `never-joins`, it ends before it joins; in one holding `silent`, it asks a question and never joins. A
# window takes the environment of the tmux server it opens in, so what varies a start is its folder, never a variable.
CLAUDE = """#!/bin/sh
for argument; do echo "$argument"; done > "$PWD/claude-args"
env > "$PWD/claude-env"
[ -e "$PWD/never-joins" ] && exit 3
[ -e "$PWD/silent" ] && { echo "Do you trust the files in this folder?"; exec sleep 120; }
socket=/tmp/fritter.sock
[ -e "$PWD/unwrapped" ] && socket=
printf '{"pid": %d, "cwd": "%s", "transcript_path": "%s/t.jsonl", "fritter_socket": "%s"}' $$ "$PWD" "$PWD" "$socket" > "$HANDS_HOME/sessions/s$$.writing"
mv "$HANDS_HOME/sessions/s$$.writing" "$HANDS_HOME/sessions/s$$.json"
exec sleep 120
"""


@pytest.fixture
def terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path]:
    """A terminal whose tmux is a server of the test's own; its bin."""
    assert TMUX is not None
    bin = tmp_path / "bin"
    bin.mkdir()
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
    home.bin.mkdir(parents=True)
    (home.bin / "claude").write_text(CLAUDE)
    (home.bin / "claude").chmod(0o755)
    return home


def members(home: Home) -> list[Membership]:
    """The home's sessions as the registry holds them once each has written its membership."""
    return [parse_membership(SessionId(path.stem), path.read_bytes()) for path in sorted(home.memberships.glob("*.json"))]


def start(home: Home, folder: Path, model: str = "") -> dict[str, Any]:
    """The tool called as the model calls it, in the environment hands was started in."""
    start_session = start_session_tool(home, AuditLog(home.audit, clock=datetime.now).record, os.environ, lambda: members(home))

    async def called() -> Result:
        return await start_session.body(folder=str(folder), model=model)

    return dict(asyncio.run(called()))


def started(home: Home) -> list[dict[str, Any]]:
    return [line for line in map(json.loads, segment(home.audit, 0).read_text().splitlines()) if line.get("event") == "session.start"]


def tmux(*arguments: str) -> str:
    assert TMUX is not None
    return subprocess.run([TMUX, *arguments], capture_output=True, text=True, check=True).stdout


def environment_of(folder: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in (folder / "claude-env").read_text().splitlines() if "=" in line)


@needs_tmux
def test_a_session_is_started_in_the_tmux_session_named_for_its_folder_and_named_once_it_joined(tmp_path: Path, terminal: Path) -> None:
    home = home_in(tmp_path)
    folder = tmp_path / "billing.api"
    folder.mkdir()
    results = [start(home, folder, "opus"), start(home, folder)]
    first, second = started(home)

    # The first start made the tmux session, named as tmux spells the folder; the second opened a window beside it.
    assert tmux_name(folder) == "billing_api"
    assert tmux("list-windows", "-t", "=billing_api", "-F", "#{pane_current_path}").split() == [str(folder.resolve())] * 2
    assert (first["outcome"], first["facts"]["made_tmux_session"], second["facts"]["made_tmux_session"]) == ("ok", True, False)
    # [LAW:nothing-unseen] which session joined, where, and on what model, on the start's event and in what it returned.
    assert results[0]["session"] != results[1]["session"]
    for event, result in zip((first, second), results):
        assert result == {"session": event["facts"]["session"], "tmux_session": "billing_api", "pane": event["facts"]["pane"]}
        assert (event["facts"]["tmux_session"], event["facts"]["under_fritter"], event["facts"]["folder"]) == ("billing_api", True, str(folder))
    assert (first["facts"]["model"], second["facts"]["model"]) == ("opus", None)
    # Only the model it was asked for, as one argument; the window reports to this home, not the tmux server's.
    assert (folder / "claude-args").read_text() == ""
    assert environment_of(folder)["HANDS_HOME"] == str(home.root)


@needs_tmux
def test_a_started_session_lists_with_the_pane_it_was_started_in_and_follows_it_to_a_new_window(tmp_path: Path, terminal: Path) -> None:
    home = home_in(tmp_path)
    folder = tmp_path / "work"
    folder.mkdir()
    result = start(home, folder)
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    for member in members(home):
        asyncio.run(sessions.apply(Joined(member, "startup")))
    recorded: list[Entry] = []
    list_sessions = audited(list_sessions_tool(sessions, Overlays(home), home, os.environ), recorded.append)

    async def called() -> Result:
        return await list_sessions.body()

    def listed() -> object:
        """The session's pane, [LAW:nothing-unseen] as the call's event holds it."""
        asyncio.run(called())
        (event,) = [entry for entry in recorded[-1:] if isinstance(entry, WideEvent)]
        (entry,) = cast(dict[str, Any], cast(Called, event.facts["called"]).result)["sessions"]
        return entry["tmux"]

    socket = str(Path(os.environ["TMUX_TMPDIR"]) / f"tmux-{os.getuid()}" / "default")
    assert listed() == {"socket": socket, "pane": result["pane"], "session": "work", "window": 0}
    # Split beside a second pane, then broken out into a window of its own: the same pane, in its new window.
    tmux("split-window", "-d", "-t", result["pane"], "sleep", "120")
    tmux("break-pane", "-d", "-s", result["pane"])
    assert listed() == {"socket": socket, "pane": result["pane"], "session": "work", "window": 1}


@needs_tmux
def test_the_model_reaches_claude_as_one_argument(tmp_path: Path, terminal: Path) -> None:
    home = home_in(tmp_path)
    folder = tmp_path / "work"
    folder.mkdir()
    assert "session" in start(home, folder, "claude-opus-5-5")
    assert (folder / "claude-args").read_text() == "--model=claude-opus-5-5\n"


@needs_tmux
def test_a_session_not_under_fritter_and_one_that_ended_are_said(tmp_path: Path, terminal: Path) -> None:
    home = home_in(tmp_path)
    folder = tmp_path / "work"
    folder.mkdir()
    (folder / "unwrapped").touch()
    assert "not under fritter, so hands cannot type into it" in start(home, folder)["error"]
    (folder / "never-joins").touch()
    assert "ended before it joined hands" in start(home, folder)["error"]
    unfrittered, ended = started(home)
    assert (unfrittered["outcome"], unfrittered["facts"]["under_fritter"]) == ("failed", False)
    assert ended["outcome"] == "failed" and "session" not in ended["facts"] and ended["facts"]["made_tmux_session"] is False


@needs_tmux
def test_a_session_that_has_not_joined_in_time_is_said_with_what_its_pane_shows(tmp_path: Path, terminal: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(startsession, "JOIN_SECONDS", 1.0)
    home = home_in(tmp_path)
    folder = tmp_path / "work"
    folder.mkdir()
    (folder / "silent").touch()
    error = start(home, folder)["error"]
    assert "has not joined hands in 1 seconds, and is still running in tmux pane %" in error
    assert "Do you trust the files in this folder?" in error
    assert started(home)[0]["outcome"] == "failed"


@needs_tmux
def test_a_session_is_the_users_whatever_the_tmux_server_it_opens_in_was_started_inside(tmp_path: Path, terminal: Path) -> None:
    home = home_in(tmp_path)
    folder = tmp_path / "work"
    folder.mkdir()
    # The user's tmux server was started from a Claude Code session under fritter, tapped: each window would be given it.
    session = {name: "given" for name in SESSION_GIVEN}
    tap = {"FRITTER_TAP": "http://127.0.0.1:7", "HTTPS_PROXY": "http://127.0.0.1:7", "FRITTER_OUTER_HTTPS_PROXY": "http://proxy.corp:3128"}
    for name, value in {**session, **tap}.items():
        tmux("set-environment", "-g", name, value)
    assert "session" in start(home, folder)
    given = environment_of(folder)
    assert not set(given) & {*session, "FRITTER_TAP", "FRITTER_OUTER_HTTPS_PROXY"}
    assert given["HTTPS_PROXY"] == "http://proxy.corp:3128"


@needs_tmux
def test_two_sessions_started_in_one_folder_at_once_are_each_named_by_its_own_start(tmp_path: Path, terminal: Path) -> None:
    home = home_in(tmp_path)
    folder = tmp_path / "work"
    folder.mkdir()

    async def both() -> list[startsession.Started]:
        record = AuditLog(home.audit, clock=datetime.now).record
        return list(await asyncio.gather(*(startsession.start(home, record, folder, None, os.environ, lambda: members(home)) for _ in range(2))))

    first, second = asyncio.run(both())
    assert first.session != second.session
    for one in (first, second):
        assert tmux("display-message", "-p", "-t", one.pane, "#{pane_pid}").strip() == one.session.removeprefix("s")


def test_a_process_descends_from_itself_and_what_it_started_never_from_a_sibling() -> None:
    table = {pid: Process(pid, parent, 501, None) for pid, parent in {10: 1, 11: 10, 12: 11, 20: 1, 1: 0}.items()}
    assert [descends(pid, 10, table) for pid in (10, 11, 12, 20, 1, 99)] == [True, True, True, False, False, False]


def test_a_folder_that_is_not_there_is_a_symlink_loop_or_is_not_a_whole_path_is_said(tmp_path: Path) -> None:
    home = home_in(tmp_path)
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    for folder in (tmp_path / "nowhere", loop):
        assert f"there is no folder {folder}" in start(home, folder)["error"]
    # Never the daemon's own working directory.
    for folder in (Path("billing"), Path("")):
        assert f"{folder} is no absolute folder, nor one from ~" in start(home, folder)["error"]
    assert [event["outcome"] for event in started(home)] == ["failed"] * 4


def test_a_terminal_outside_any_session_has_nothing_of_one_and_keeps_the_users_own_setup() -> None:
    inside = {
        **{name: "given" for name in SESSION_GIVEN},
        # fritter's tap of the session hands was started in, and the proxy it replaced.
        "FRITTER_TAP": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9", "FRITTER_OUTER_HTTPS_PROXY": "http://proxy.lan:3128",
    }
    users = {"PATH": "/Users/me/.hands/bin:/usr/bin", "CLAUDE_CONFIG_DIR": "/Users/me/.claude.work", "ANTHROPIC_API_KEY": "sk-mine", "HANDS_HOME": "/elsewhere"}
    assert as_from_a_terminal({**inside, **users}) == {**users, "HTTPS_PROXY": "http://proxy.lan:3128"}
