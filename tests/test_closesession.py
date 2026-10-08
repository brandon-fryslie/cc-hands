"""The `close_session` tool: a session's claude ended, and the session gone from the registry, as the user asked: a session
they named whatever it is doing, and of the sessions asked about as done only those at their prompt. Each session is a
real process, and the registry is kept by the liveness sweep, as the daemon keeps it."""

import asyncio
import json
import os
import subprocess
import time
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pytest

from hands.core.events import Event, Joined, PermissionRequested, Prompted, StatusReported
from hands.core.session import Membership, Permission, PromptId, RequestId, SessionId
from hands.core.status import Busy, Idle, Report, Shell, Stamp, Status, Waiting
from hands.sessions import closesession
from hands.sessions.audit import AuditLog, segment
from hands.sessions.home import Home
from hands.sessions.liveness import keep_sweeping
from hands.sessions.membership import write_membership
from hands.sessions.registry import Sessions
from hands.voice.tool import Result
from hands.voice.tools import close_session_tool
from test_startsession import home_in, needs_tmux, start, terminal, tmux

__all__ = ["terminal"]  # the fixture, used by name


def registry() -> Sessions:
    return Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)


def run[T](home: Home, sessions: Sessions, body: Callable[[], Awaitable[T]]) -> T:
    """`body` run while the liveness sweep keeps `sessions`, as the daemon's does."""

    async def swept() -> T:
        sweeping = asyncio.create_task(keep_sweeping(home, sessions, 0.05))
        try:
            return await body()
        finally:
            sweeping.cancel()

    return asyncio.run(swept())


def close(home: Home, sessions: Sessions, ids: list[str], asked: str) -> Callable[[], Awaitable[Result]]:
    """The tool called as the model calls it."""
    close_session = close_session_tool(home, AuditLog(home.audit, clock=datetime.now).record, sessions)
    return lambda: close_session.body(sessions=ids, asked=asked)


def each(result: Result) -> list[Any]:
    """What a call said of each session it was given, in the order given."""
    return cast(list[Any], result["sessions"])


def closes(home: Home) -> list[dict[str, Any]]:
    return [line for line in map(json.loads, segment(home.audit, 0).read_text().splitlines()) if line.get("event") == "session.close"]


def running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@contextmanager
def processes(count: int, command: tuple[str, ...] = ("sleep", "120")) -> Generator[list[int]]:
    """`count` processes running `command`, each the test's grandchild, so that one ending is reaped by launchd as a
    claude is by the fritter that ran it, and is gone from the process table; their pids."""
    pids = [int(subprocess.run(["/bin/sh", "-c", '"$@" >/dev/null 2>&1 & echo $!', "sh", *command], capture_output=True, text=True, check=True).stdout) for _ in range(count)]
    try:
        yield pids
    finally:
        for pid in pids:
            with suppress(ProcessLookupError):
                os.kill(pid, 9)


def ended(pid: int) -> None:
    deadline = time.monotonic() + 5
    while running(pid):
        assert time.monotonic() < deadline, f"pid {pid} never ended"
        time.sleep(0.05)


def joined(home: Home, sessions: Sessions, tmp_path: Path, pid: int, name: str, *events: Callable[[SessionId], Event]) -> SessionId:
    """A session in its own folder whose claude is `pid`, as its first hook writes it, then `events` of it applied."""
    folder = tmp_path / name
    folder.mkdir()
    member = Membership(SessionId(name), pid, folder, folder / "t.jsonl")
    write_membership(home, member)
    for event in (Joined(member, "startup"), *(event(member.id) for event in events)):
        asyncio.run(sessions.apply(event))
    return member.id


def status(said: Status) -> Callable[[SessionId], Event]:
    return lambda session: StatusReported(session, Report(said, Stamp(1)), at=1.0)


def test_only_the_sessions_that_are_done_are_closed_of_those_asked_about_as_done(tmp_path: Path) -> None:
    home = home_in(tmp_path)
    sessions = registry()
    with processes(7) as (idle, busy, waiting, shelled, held, prompted, unasked):
        ids = {
            "idle": joined(home, sessions, tmp_path, idle, "idle", status(Idle())),
            "busy": joined(home, sessions, tmp_path, busy, "busy", status(Busy())),
            "waiting": joined(home, sessions, tmp_path, waiting, "waiting", status(Waiting("permission prompt"))),
            "shelled": joined(home, sessions, tmp_path, shelled, "shelled", status(Shell())),
            # At a dialog whose request is heard before the status that says it waits.
            "held": joined(home, sessions, tmp_path, held, "held", status(Idle()), lambda session: PermissionRequested(session, at=2.0, request=RequestId("r"), on=Permission("Bash", {}), mode="default", timeout=None)),
            # A prompt its hook opened a turn for, before the status that says it is busy is read.
            "prompted": joined(home, sessions, tmp_path, prompted, "prompted", status(Idle()), lambda session: Prompted(session, at=2.0, mode=None, prompt=PromptId("p1"))),
        }
        joined(home, sessions, tmp_path, unasked, "unasked", status(Idle()))

        results = dict(zip(ids, each(run(home, sessions, close(home, sessions, list(ids.values()), "done"))), strict=True))
        ended(idle)

        assert results["idle"] == {"closed": "idle"}
        assert {name: result["state"] for name, result in results.items() if name != "idle"} == {
            "busy": "working",
            "waiting": "waiting at a dialog: permission prompt",
            "shelled": "idle, with a shell command it started in the background still running",
            "held": "waiting for permission to use Bash",
            "prompted": "working",
        }
        assert all(results[name]["left_running"] == name for name in ("busy", "waiting", "shelled", "held", "prompted"))
        # Only the done one ended and left the registry; the rest, and the session nobody asked about, run on.
        assert [running(pid) for pid in (idle, busy, waiting, shelled, held, prompted, unasked)] == [False, True, True, True, True, True, True]
        assert sorted(sessions.live_ids()) == ["busy", "held", "prompted", "shelled", "unasked", "waiting"]
    # [LAW:nothing-unseen] each close says which session, why it was asked, whether it was done, and whether it was signalled.
    facts = {event["facts"]["session"]: event["facts"] for event in closes(home)}
    assert facts["idle"] == {"session": "idle", "asked": "done", "done": True, "signalled": True}
    assert facts["busy"] == {"session": "busy", "asked": "done", "done": False}


def test_a_named_session_is_closed_whatever_it_is_doing(tmp_path: Path) -> None:
    home = home_in(tmp_path)
    sessions = registry()
    with processes(2) as (busy, other):
        id = joined(home, sessions, tmp_path, busy, "billing", status(Busy()))
        joined(home, sessions, tmp_path, other, "docs", status(Idle()))
        assert each(run(home, sessions, close(home, sessions, [id], "named"))) == [{"closed": "billing"}]
        ended(busy)
        assert sessions.live_ids() == ["docs"] and running(other)
    assert closes(home)[0]["facts"] == {"session": "billing", "asked": "named", "done": False, "signalled": True}


def test_a_session_that_is_not_running_and_one_that_will_not_end_are_said(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(closesession, "END_SECONDS", 0.5)
    home = home_in(tmp_path)
    sessions = registry()
    assert each(run(home, sessions, close(home, sessions, ["nobody"], "named"))) == [{"error": "no session nobody is running"}]
    # A claude that does not end on SIGTERM, as one wedged would not.
    with processes(1, ("/bin/sh", "-c", "trap '' TERM; while :; do sleep 0.1; done")) as (stuck,):
        id = joined(home, sessions, tmp_path, stuck, "stuck", status(Idle()))
        time.sleep(0.2)  # the trap is set before the signal is sent
        assert each(run(home, sessions, close(home, sessions, [id], "named"))) == [{"error": "session stuck was told to end and is still running 0.5 seconds later"}]
        assert running(stuck) and sessions.live_ids() == ["stuck"]
    assert [event["outcome"] for event in closes(home)] == ["failed", "failed"]


def test_a_session_whose_claude_no_longer_holds_it_leaves_with_nothing_signalled(tmp_path: Path) -> None:
    home = home_in(tmp_path)
    sessions = registry()
    with processes(3) as (cleared, reused, unfiled):
        moved = joined(home, sessions, tmp_path, cleared, "moved", status(Idle()))
        # A /clear in it: the same claude holds a new session, written after the one it left.
        after = joined(home, sessions, tmp_path, cleared, "after", status(Idle()))
        os.utime(home.membership(after), (time.time() + 5, time.time() + 5))
        # A pid a later process took: the file naming it was written before that process started.
        taken = joined(home, sessions, tmp_path, reused, "taken", status(Idle()))
        os.utime(home.membership(taken), (time.time() - 60, time.time() - 60))
        # Its file gone, as its end hook removes it first.
        gone = joined(home, sessions, tmp_path, unfiled, "gone", status(Idle()))
        home.membership(gone).unlink()
        assert each(run(home, sessions, close(home, sessions, [moved, taken, gone], "named"))) == [{"closed": "moved"}, {"closed": "taken"}, {"closed": "gone"}]
        # No process was sent anything: the claude that moved on runs on in the session it holds.
        assert all(running(pid) for pid in (cleared, reused, unfiled))
        assert sessions.live_ids() == ["after"]
    assert [event["facts"]["signalled"] for event in closes(home)] == [False, False, False]


@needs_tmux
def test_a_window_hands_opened_closes_with_its_session_and_one_the_user_opened_stays(tmp_path: Path, terminal: Path) -> None:
    home = home_in(tmp_path)
    sessions = registry()
    folder = tmp_path / "work"
    folder.mkdir()
    opened = start(home, folder)
    # The user's own window: a shell they ran claude from, which takes their prompt again once it ends.
    mine = tmp_path / "mine"
    mine.mkdir()
    pane = tmux("new-window", "-d", "-P", "-F", "#{pane_id}", "-t", "=theirs:", "-c", str(mine), "-e", f"HANDS_HOME={home.root}", f"{home.bin / 'claude'}; exec sleep 120").strip()
    deadline = time.monotonic() + 5
    while len(list(home.memberships.glob("*.json"))) < 2:
        assert time.monotonic() < deadline, "the user's claude never joined"
        time.sleep(0.05)
    users = next(path.stem for path in home.memberships.glob("*.json") if path.stem != opened["session"])

    def panes() -> list[str]:
        return tmux("list-panes", "-a", "-F", "#{pane_id} #{pane_dead}").splitlines()

    async def both() -> list[Result]:
        # The sweep lists both sessions first, as the daemon's had before the user asked.
        while len(sessions.live_ids()) < 2:
            await asyncio.sleep(0.05)
        return each(await close(home, sessions, [opened["session"], users], "named")())

    assert {f"{opened['pane']} 0", f"{pane} 0"} <= set(panes())
    assert run(home, sessions, both) == [{"closed": "work"}, {"closed": "mine"}]
    # hands' window went with its session, and the tmux session it made with its last window; the user's stays, its
    # shell running on.
    deadline = time.monotonic() + 5
    while any(line.startswith(f"{opened['pane']} ") for line in panes()):
        assert time.monotonic() < deadline, f"hands' window outlived its session: {panes()}"
        time.sleep(0.05)
    assert f"{pane} 0" in panes()
    assert "work" not in tmux("list-sessions", "-F", "#{session_name}").split()
    assert sessions.live_ids() == []
