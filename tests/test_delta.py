"""What a turn changed in the repository it ran in, read from git without disturbing it."""

import asyncio
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hands.core.delta import Branched, Delta, PullRequested, Pushed
from hands.core.effects import SessionGone, Summarise
from hands.core.events import Closed, Ended, Joined, Prompted, StatusReported, Stopped, Taken
from hands.core import status
from hands.core.status import Report, Stamp
from hands.core.session import Membership, Opened, PromptId, RequestId, SessionId
from hands.sessions.audit import Entry
from hands.sessions.delta import HELD, MOST_COMMITS, MOST_LINES, Asked, Deltas, Mark
from hands.sessions.registry import Sessions
from hands.sessions.wide import WideEvent

# When hands heard a Stop, on the clock Claude Code stamps a status with.
STOP_HEARD = Stamp(1500)
STOP_REQUEST = RequestId("stop")

SID = SessionId("s1")


class Breaks:
    """A repository reader that fails at both ends, which neither end of a turn may be made to care about."""

    async def snapshot(self, session: SessionId, cwd: Path) -> None:
        raise RuntimeError("there is nowhere to put a scratch index")

    async def compare(self, session: SessionId, again: bool) -> None:
        raise RuntimeError("git is on fire")

    async def taken(self, session: SessionId) -> Delta:
        return Delta()


def attached(tmp_path: Path) -> Sessions:
    return Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _entry: None, changes=Breaks())


def git(repo: Path, *args: str) -> str:
    return subprocess.run(("git", "-C", str(repo), *args), capture_output=True, text=True, check=True).stdout.strip()


def repo(tmp_path: Path) -> Path:
    root = tmp_path / "work"
    root.mkdir()
    git(root.parent, "init", "-q", str(root))
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "Test")
    (root / "a.py").write_text("x = 1\n")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "first")
    return root


async def turn(root: Path, work: object = None, record: list[Entry] | None = None) -> Delta:
    """One turn: mark where the repository is, let `work` happen, then read what changed, writing its audit line to `record`."""
    deltas = Deltas(record=(record if record is not None else []).append, inherited=os.environ)
    await deltas.snapshot(SID, root)
    if callable(work):
        work()
    await deltas.compare(SID, again=False)
    return await deltas.taken(SID)


def readings(record: list[Entry]) -> list[WideEvent]:
    """The event of each reading in record, in the order they ended."""
    return [entry for entry in record if isinstance(entry, WideEvent) and entry.event == "delta.read"]


async def test_a_turn_whose_only_change_came_from_a_shell_command_names_the_file_it_changed(tmp_path: Path) -> None:
    """The done bar. A formatter, a code generator, or a `sed` leaves no `Edited` step to name what it did."""
    root = repo(tmp_path)
    delta = await turn(root, lambda: (root / "a.py").write_text("x = 1\ny = 2\n"))
    assert [(file.path, file.added, file.removed) for file in delta.files] == [("a.py", 1, 0)]
    assert "y = 2" in delta.patch
    assert delta  # there is something to tell


async def test_a_file_a_turn_created_is_named_even_though_git_was_never_told_about_it(tmp_path: Path) -> None:
    """A code generator writes files nobody adds. `git stash create` holds none of them, which is why this does not use it."""
    root = repo(tmp_path)
    delta = await turn(root, lambda: (root / "generated.py").write_text("# made by a generator\n"))
    assert [file.path for file in delta.files] == ["generated.py"]


async def test_a_file_the_repository_ignores_is_not_a_result(tmp_path: Path) -> None:
    root = repo(tmp_path)
    (root / ".gitignore").write_text("*.log\n")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "ignore logs")
    delta = await turn(root, lambda: (root / "noise.log").write_text("chatter\n"))
    assert delta.files == () and not delta


async def test_a_commit_the_turn_made_is_named_and_the_work_in_it_is_still_the_turn_s(tmp_path: Path) -> None:
    """Committed work is still what the turn did: the delta is against where the turn began, not against HEAD."""
    root = repo(tmp_path)

    def commit() -> None:
        (root / "a.py").write_text("x = 2\n")
        git(root, "add", "-A")
        git(root, "commit", "-qm", "tidy up")

    delta = await turn(root, commit)
    assert [commit.subject for commit in delta.commits] == ["tidy up"]
    assert [file.path for file in delta.files] == ["a.py"]


async def test_work_already_in_the_tree_before_the_turn_is_not_the_turn_s(tmp_path: Path) -> None:
    """The mark holds the working tree as it stood, so what was already changed is not reported as done now."""
    root = repo(tmp_path)
    (root / "a.py").write_text("x = 1\nleft over from before\n")
    delta = await turn(root)
    assert not delta and delta.files == ()


async def test_a_turn_that_changed_nothing_has_nothing_to_tell(tmp_path: Path) -> None:
    assert not await turn(repo(tmp_path))


async def test_reading_a_repository_leaves_it_exactly_as_it_was_found(tmp_path: Path) -> None:
    """[LAW:effects-at-boundaries] reading what a turn did must never be something the user has to undo.

    Read over a repository with work in progress, which is the state that has something to lose: a modified
    file, a staged file, and an untracked one. Nothing here may stage, stash, commit, or revert any of them.
    """
    root = repo(tmp_path)
    (root / "a.py").write_text("x = 1\nuncommitted\n")
    (root / "staged.py").write_text("half done\n")
    git(root, "add", "staged.py")
    (root / "scratch.py").write_text("not git's yet\n")
    before = (git(root, "status", "--porcelain"), git(root, "rev-parse", "HEAD"), git(root, "stash", "list"), git(root, "diff"), git(root, "diff", "--cached"))
    index = (root / ".git" / "index").read_bytes()

    await turn(root)

    assert (root / ".git" / "index").read_bytes() == index, "the repository's own index was written to"
    assert (git(root, "status", "--porcelain"), git(root, "rev-parse", "HEAD"), git(root, "stash", "list"), git(root, "diff"), git(root, "diff", "--cached")) == before
    assert (root / "a.py").read_text() == "x = 1\nuncommitted\n"
    assert (root / "scratch.py").read_text() == "not git's yet\n"


def published(tmp_path: Path) -> Path:
    """A repository with a remote it has pushed its trunk to, as one a session opens pull requests from."""
    root = repo(tmp_path)
    git(tmp_path, "init", "-q", "--bare", str(tmp_path / "remote.git"))
    git(root, "remote", "add", "origin", str(tmp_path / "remote.git"))
    git(root, "push", "-q", "-u", "origin", "HEAD")
    return root


def forge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: str) -> Path:
    """A `gh` on the PATH that answers every question with `answer` and writes down what it was asked."""
    bin = tmp_path / "bin"
    bin.mkdir()
    asked = tmp_path / "asked"
    gh = bin / "gh"
    gh.write_text(f"#!/bin/sh\necho \"$*\" >> {asked}\ncat <<'EOF'\n{answer}\nEOF\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin}{os.pathsep}{os.environ['PATH']}")
    return asked


def stamp(at: datetime) -> str:
    return at.strftime("%Y-%m-%dT%H:%M:%SZ")


async def test_a_push_no_step_recorded_is_read_from_the_remote_tracking_ref_git_logged_it_on(tmp_path: Path) -> None:
    """`git commit -m x && git push` carries no push operation, and the remote-tracking ref says it anyway."""
    root = published(tmp_path)

    def push() -> None:
        (root / "a.py").write_text("x = 2\n")
        git(root, "commit", "-qam", "tidy")
        git(root, "push", "-q")

    delta = await turn(root, push)
    assert delta.changes == (Pushed(git(root, "branch", "--show-current")),)


async def test_a_fetch_moves_the_same_ref_and_is_not_heard_as_a_push(tmp_path: Path) -> None:
    root = published(tmp_path)
    other = tmp_path / "other"
    git(tmp_path, "clone", "-q", str(tmp_path / "remote.git"), str(other))
    git(other, "config", "user.email", "t@example.com")
    git(other, "config", "user.name", "Test")
    git(other, "commit", "-q", "--allow-empty", "-m", "someone else")
    git(other, "push", "-q")
    delta = await turn(root, lambda: git(root, "fetch", "-q"))
    assert delta.changes == ()


async def test_a_branch_the_turn_made_is_read_from_the_branches_the_mark_did_not_hold(tmp_path: Path) -> None:
    """`checkout -b` inside a compound command is the one Claude Code nearly never records."""
    root = repo(tmp_path)
    delta = await turn(root, lambda: git(root, "checkout", "-q", "-b", "feature/narration"))
    assert delta.changes == (Branched("feature/narration", "created branch"),)
    assert delta  # a new branch alone is something to tell


async def test_a_pull_request_opened_from_a_branch_the_turn_pushed_is_read_from_the_forge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`gh pr create` writes nothing in the repository, so the forge is the one witness — and a pull request the
    branch had before the turn is not the turn's."""
    root = published(tmp_path)
    now, before = datetime.now(UTC), datetime.now(UTC) - timedelta(days=1)
    asked = forge(
        tmp_path,
        monkeypatch,
        f'[{{"number": 7, "url": "https://x/7", "createdAt": "{stamp(now)}"}}, {{"number": 3, "url": "https://x/3", "createdAt": "{stamp(before)}"}}]',
    )

    def open_one() -> None:
        git(root, "checkout", "-q", "-b", "fix")
        git(root, "push", "-q", "-u", "origin", "fix")

    delta = await turn(root, open_one)
    assert delta.changes == (Branched("fix", "created branch"), Pushed("fix"), PullRequested(7, "https://x/7", "created"))
    assert asked.read_text().split() == ["pr", "list", "--head", "fix", "--author", "@me", "--state", "all", "--json", "number,url,createdAt"]


async def test_the_forge_is_not_asked_about_a_turn_that_pushed_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = published(tmp_path)
    asked = forge(tmp_path, monkeypatch, "[]")

    def commit() -> None:
        (root / "a.py").write_text("x = 2\n")
        git(root, "commit", "-qam", "tidy")

    await turn(root, commit)
    assert not asked.exists()


async def test_a_forge_that_answers_in_a_shape_hands_does_not_read_costs_the_pull_request_and_not_the_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = published(tmp_path)
    forge(tmp_path, monkeypatch, "not json")
    delta = await turn(root, lambda: (git(root, "checkout", "-q", "-b", "fix"), git(root, "push", "-q", "-u", "origin", "fix")))
    assert delta.changes == (Branched("fix", "created branch"), Pushed("fix"))


def cloned(tmp_path: Path) -> Path:
    """A clone of a published repository, which knows the branch its remote's HEAD follows."""
    published(tmp_path)
    return clone(tmp_path, "clone")


def clone(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    git(tmp_path, "clone", "-q", str(tmp_path / "remote.git"), str(root))
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "Test")
    return root


def slow_forge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    bin = tmp_path / "bin"
    bin.mkdir()
    gh = bin / "gh"
    gh.write_text(f"#!/bin/sh\nsleep {seconds}\necho '[]'\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin}{os.pathsep}{os.environ['PATH']}")


async def test_a_branch_made_and_pushed_in_another_worktree_is_not_this_turns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Worktrees share every branch and remote-tracking ref; only the one checked out here is this session's."""
    root = published(tmp_path)
    asked = forge(tmp_path, monkeypatch, "[]")
    beside = tmp_path / "beside"

    def elsewhere() -> None:
        git(root, "worktree", "add", "-q", "-b", "theirs", str(beside))
        git(beside, "push", "-q", "-u", "origin", "theirs")

    delta = await turn(root, elsewhere)
    assert delta.changes == ()
    assert not asked.exists()


async def test_checking_out_a_branch_the_remote_already_had_is_not_making_one(tmp_path: Path) -> None:
    root = cloned(tmp_path)
    git(tmp_path / "work", "push", "-q", "origin", "HEAD:theirs")
    git(root, "fetch", "-q")
    delta = await turn(root, lambda: git(root, "checkout", "-q", "theirs"))
    assert delta.changes == ()


async def test_renaming_a_branch_is_not_making_one(tmp_path: Path) -> None:
    """A rename carries the log of the branch it renamed, and that branch was made before the turn began."""
    root = repo(tmp_path)
    git(root, "checkout", "-q", "-b", "old")
    time.sleep(1.1)  # git logs to the second
    delta = await turn(root, lambda: git(root, "branch", "-m", "old", "new"))
    assert delta.changes == ()


async def test_a_push_the_turn_followed_with_a_pull_is_still_a_push(tmp_path: Path) -> None:
    root = cloned(tmp_path)
    other = clone(tmp_path, "other")
    git(root, "checkout", "-q", "-b", "fix")

    def push_then_pull() -> None:
        git(root, "push", "-q", "-u", "origin", "fix")
        git(other, "fetch", "-q")
        git(other, "checkout", "-q", "fix")
        git(other, "commit", "-q", "--allow-empty", "-m", "someone else")
        git(other, "push", "-q")
        git(root, "pull", "-q")

    delta = await turn(root, push_then_pull)
    assert Pushed("fix") in delta.changes


async def test_one_branch_pushed_to_two_remotes_is_one_push_and_one_question(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = published(tmp_path)
    git(tmp_path, "init", "-q", "--bare", str(tmp_path / "fork.git"))
    git(root, "remote", "add", "fork", str(tmp_path / "fork.git"))
    asked = forge(tmp_path, monkeypatch, "[]")

    def push_twice() -> None:
        git(root, "checkout", "-q", "-b", "fix")
        git(root, "push", "-q", "origin", "fix")
        git(root, "push", "-q", "fork", "fix")

    delta = await turn(root, push_twice)
    assert delta.changes == (Branched("fix", "created branch"), Pushed("fix"))
    assert len(asked.read_text().splitlines()) == 1


async def test_the_forge_is_not_asked_about_a_push_to_the_default_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No pull request is opened from the branch the remote's HEAD follows, and it is where most pushes go."""
    root = cloned(tmp_path)
    asked = forge(tmp_path, monkeypatch, "[]")

    def push() -> None:
        git(root, "commit", "-q", "--allow-empty", "-m", "tidy")
        git(root, "push", "-q")

    record: list[Entry] = []
    delta = await turn(root, push, record)
    assert delta.changes == (Pushed(git(root, "branch", "--show-current")),)
    assert not asked.exists()
    assert [event.facts["forge"] for event in readings(record)] == ["unasked"]


async def test_a_slow_forge_costs_the_turn_its_pull_request_and_never_its_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The narrator waits for the whole reading at once, so the forge is asked beside the tree and inside that wait."""
    root = published(tmp_path)
    slow_forge(tmp_path, monkeypatch, 10)

    def commit_and_push() -> None:
        git(root, "checkout", "-q", "-b", "fix")
        git(root, "commit", "-q", "--allow-empty", "-m", "tidy")
        git(root, "push", "-q", "-u", "origin", "fix")

    record: list[Entry] = []
    # A narrator far less patient than the daemon's, so a forge budget not taken from its patience would outlast it.
    deltas = Deltas(record=record.append, inherited=os.environ, patience=1.0)
    await deltas.snapshot(SID, root)
    commit_and_push()
    await deltas.compare(SID, again=False)
    delta = await deltas.taken(SID)
    assert [commit.subject for commit in delta.commits] == ["tidy"]
    assert delta.changes == (Branched("fix", "created branch"), Pushed("fix"))
    [read] = readings(record)
    assert read.facts["forge"] == "unanswered" and read.duration_ms < 1000


async def test_a_forge_that_refuses_is_told_apart_from_a_slow_one_and_nothing_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote that is not GitHub, or a gh not signed in: gh says no at once, and that is not a failure of hands."""
    from loguru import logger

    root = published(tmp_path)
    bin = tmp_path / "bin"
    bin.mkdir()
    (bin / "gh").write_text("#!/bin/sh\necho 'none of the git remotes point to a known GitHub host' >&2\nexit 1\n")
    (bin / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin}{os.pathsep}{os.environ['PATH']}")
    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(str(message)), level="ERROR")
    record: list[Entry] = []
    try:
        delta = await turn(root, lambda: (git(root, "checkout", "-q", "-b", "fix"), git(root, "push", "-q", "-u", "origin", "fix")), record)
    finally:
        logger.remove(sink)
    assert delta.changes == (Branched("fix", "created branch"), Pushed("fix"))
    assert [event.facts["forge"] for event in readings(record)] == ["refused"]
    assert errors == []


async def test_a_machine_with_no_gh_has_no_forge_to_ask_and_nothing_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from loguru import logger

    root = published(tmp_path)
    git_dir = Path(subprocess.run(("which", "git"), capture_output=True, text=True, check=True).stdout.strip()).parent
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "git").symlink_to(git_dir / "git")
    monkeypatch.setenv("PATH", str(bare))
    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(str(message)), level="ERROR")
    record: list[Entry] = []
    try:
        delta = await turn(root, lambda: (git(root, "checkout", "-q", "-b", "fix"), git(root, "push", "-q", "-u", "origin", "fix")), record)
    finally:
        logger.remove(sink)
    assert delta.changes == (Branched("fix", "created branch"), Pushed("fix"))
    assert [event.facts["forge"] for event in readings(record)] == ["absent"]
    assert errors == []


async def test_a_pull_request_stamped_with_no_zone_is_not_told_and_costs_nothing_else(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = published(tmp_path)
    forge(tmp_path, monkeypatch, '[{"number": 7, "url": "https://x/7", "createdAt": "2099-01-01T00:00:00"}]')
    delta = await turn(root, lambda: (git(root, "checkout", "-q", "-b", "fix"), git(root, "push", "-q", "-u", "origin", "fix")))
    assert delta.changes == (Branched("fix", "created branch"), Pushed("fix"))


async def test_each_reading_is_one_audit_line_with_what_it_found_and_what_the_forge_cost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = published(tmp_path)
    forge(tmp_path, monkeypatch, f'[{{"number": 7, "url": "https://x/7", "createdAt": "{stamp(datetime.now(UTC))}"}}]')

    def open_one() -> None:
        git(root, "checkout", "-q", "-b", "fix")
        git(root, "commit", "-q", "--allow-empty", "-m", "tidy")
        git(root, "push", "-q", "-u", "origin", "fix")

    record: list[Entry] = []
    await turn(root, open_one, record)
    [read] = readings(record)
    assert (read.outcome, read.counts) == ("ok", {"commits": 1, "files": 0})
    assert (read.facts["session"], read.facts["reading"], read.facts["forge"]) == (SID, "read", "answered")
    assert read.facts["changes"] == (Branched("fix", "created branch"), Pushed("fix"), PullRequested(7, "https://x/7", "created"))


async def test_a_turn_outside_a_repository_is_one_unmarked_reading_that_counts_nothing_and_asks_no_forge(tmp_path: Path) -> None:
    record: list[Entry] = []
    await turn(tmp_path, None, record)
    [read] = readings(record)
    assert (read.outcome, read.facts["reading"], read.counts) == ("ok", "unmarked", {"commits": 0, "files": 0})
    assert "forge" not in read.facts and "index" not in read.facts


async def test_a_reading_says_which_index_its_snapshot_started_from_and_counts_zero_where_nothing_changed(tmp_path: Path) -> None:
    root = repo(tmp_path)
    record: list[Entry] = []
    await turn(root, None, record)
    [mark] = [entry for entry in record if isinstance(entry, WideEvent) and entry.event == "delta.mark"]
    [read] = readings(record)
    assert (mark.outcome, mark.facts) == ("ok", {"session": SID, "index": "copied", "marked": True})
    assert (read.outcome, read.counts) == ("ok", {"commits": 0, "files": 0})
    assert read.facts == {"session": SID, "reading": "read", "index": "copied", "changes": (), "forge": "unasked", "forge_seconds": 0.0}
    # Two units of work, neither opened inside the other: two traces.
    assert mark.trace_id != read.trace_id


async def test_a_repository_with_no_index_yet_is_snapshotted_from_an_empty_one_and_says_why(tmp_path: Path) -> None:
    root = tmp_path / "work"
    git(tmp_path, "init", "-q", str(root))
    (root / "a.py").write_text("x = 1\n")
    record: list[Entry] = []
    await turn(root, lambda: (root / "b.py").write_text("y = 2\n"), record)
    [mark] = [entry for entry in record if isinstance(entry, WideEvent) and entry.event == "delta.mark"]
    [read] = readings(record)
    assert mark.facts["index"] == read.facts["index"] == "empty"
    assert "No such file" in str(mark.facts["index_error"])
    assert read.counts == {"commits": 0, "files": 1}


async def test_a_session_whose_directory_is_gone_is_not_logged_as_git_failing_to_start(tmp_path: Path) -> None:
    from loguru import logger

    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(str(message)), level="ERROR")
    try:
        await turn(tmp_path / "gone")
    finally:
        logger.remove(sink)
    assert errors == []


async def test_a_session_that_works_outside_a_repository_is_told_by_its_steps_alone(tmp_path: Path) -> None:
    """[LAW:no-silent-failure] not every session is in a repository, and that is not a failure to report."""
    plain = tmp_path / "plain"
    plain.mkdir()
    assert not await turn(plain, lambda: (plain / "a.txt").write_text("hello\n"))


async def test_a_directory_that_is_not_there_is_told_by_its_steps_alone(tmp_path: Path) -> None:
    assert not await turn(tmp_path / "never-existed")


async def test_a_turn_stopping_with_no_mark_before_it_reads_nothing(tmp_path: Path) -> None:
    """The daemon started in the middle of a turn: there is no beginning to compare against, so there is no delta."""
    deltas = Deltas(record=lambda _entry: None, inherited=os.environ)
    await deltas.compare(SID, again=False)
    assert not await deltas.taken(SID)


async def test_a_delta_is_told_once_and_never_twice(tmp_path: Path) -> None:
    """A delta told is a delta spent; the next turn's is the next turn's."""
    root = repo(tmp_path)
    deltas = Deltas(record=lambda _entry: None, inherited=os.environ)
    await deltas.snapshot(SID, root)
    (root / "a.py").write_text("x = 3\n")
    await deltas.compare(SID, again=False)
    assert await deltas.taken(SID)
    assert not await deltas.taken(SID)


async def test_an_edit_that_keeps_a_files_size_and_second_is_still_told(tmp_path: Path) -> None:
    """Git trusts a file whose stat matches its index entry unless the file is as new as the index itself. Pinned
    here to the one second the commit, the index and the edit all share, which a fast turn lands in by chance."""
    root = repo(tmp_path)
    # The edit changes ctime, which a real turn inside the same second leaves in that second too.
    git(root, "config", "core.trustctime", "false")
    second = time.time() - 100
    os.utime(root / "a.py", (second, second))
    git(root, "update-index", "--refresh")
    os.utime(root / ".git" / "index", (second, second))
    deltas = Deltas(record=lambda _entry: None, inherited=os.environ)
    await deltas.snapshot(SID, root)
    (root / "a.py").write_text("x = 3\n")
    os.utime(root / "a.py", (second, second))
    await deltas.compare(SID, again=False)
    assert [file.path for file in (await deltas.taken(SID)).files] == ["a.py"]


async def test_a_new_turn_reads_against_its_own_beginning_and_not_the_one_before(tmp_path: Path) -> None:
    root = repo(tmp_path)
    deltas = Deltas(record=lambda _entry: None, inherited=os.environ)
    await deltas.snapshot(SID, root)
    (root / "a.py").write_text("first turn\n")
    await deltas.compare(SID, again=False)
    assert [file.path for file in (await deltas.taken(SID)).files] == ["a.py"]

    await deltas.snapshot(SID, root)
    (root / "b.py").write_text("second turn\n")
    await deltas.compare(SID, again=False)
    assert [file.path for file in (await deltas.taken(SID)).files] == ["b.py"]


async def test_a_turn_that_stops_again_tells_only_what_it_changed_after_its_first_stop(tmp_path: Path) -> None:
    """Another Stop hook blocked the first Stop and Claude went on, with no prompt to mark from: the second part is read
    against where the first part's reading found the repository, so each change is told once."""
    root = repo(tmp_path)
    deltas = Deltas(record=lambda _entry: None, inherited=os.environ)
    await deltas.snapshot(SID, root)
    (root / "a.py").write_text("first part\n")
    await deltas.compare(SID, again=False)
    assert [file.path for file in (await deltas.taken(SID)).files] == ["a.py"]

    (root / "b.py").write_text("second part\n")
    await deltas.compare(SID, again=True)
    assert [file.path for file in (await deltas.taken(SID)).files] == ["b.py"]


async def test_a_turn_no_prompt_marked_is_not_read_against_where_the_turn_before_ended(tmp_path: Path) -> None:
    """Between the two, the user may have edited by hand or pulled for hours: none of it is this turn's."""
    root = repo(tmp_path)
    deltas = Deltas(record=lambda _entry: None, inherited=os.environ)
    await deltas.snapshot(SID, root)
    (root / "a.py").write_text("first turn\n")
    await deltas.compare(SID, again=False)
    assert await deltas.taken(SID)

    (root / "b.py").write_text("by hand, or another session's\n")
    await deltas.compare(SID, again=False)
    assert not await deltas.taken(SID)


async def test_a_prompt_and_a_stop_through_the_daemon_read_what_the_turn_changed(tmp_path: Path) -> None:
    """End to end through the parts that decide it: the reducer emits, the registry performs, in that order.

    The turn here is what the ticket is for — a shell command changed a file and no step in the transcript
    says so — and the delta is ready before the turn is ever handed over to be summarised.
    """
    from hands.core.events import Joined, Prompted, Stopped
    from hands.core.session import Membership
    from hands.sessions.registry import Sessions

    root = repo(tmp_path)
    deltas = Deltas(record=lambda _entry: None, inherited=os.environ)
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _entry: None, changes=deltas)
    await sessions.apply(Joined(Membership(SID, pid=4242, cwd=root, transcript=tmp_path / "t.jsonl"), "startup"))

    await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p1")))
    subprocess.run(("sed", "-i", "", "s/x = 1/x = 99/", str(root / "a.py")), check=True)
    await sessions.apply(Stopped(SID, "Done.", mode=None, prompt=PromptId("p1"), again=False, heard=STOP_HEARD, request=STOP_REQUEST))

    # The story is queued, and by the time anyone takes it the delta is already read and waiting.
    story = await sessions.story()
    assert story.session == SID
    delta = await deltas.taken(SID)
    assert [(file.path, file.added, file.removed) for file in delta.files] == [("a.py", 1, 1)]
    assert "x = 99" in delta.patch


async def test_a_mark_that_would_cost_more_than_the_hook_can_afford_is_not_taken(tmp_path: Path) -> None:
    """A prompt's hook waits on this one, and the shim gives up after two seconds and says it cannot reach
    the daemon — on every prompt and every stop. A repository too slow to mark has its turn told without."""
    root = repo(tmp_path)
    deltas = Deltas(record=lambda _entry: None, inherited=os.environ, marking=0.0)
    await deltas.snapshot(SID, root)
    (root / "a.py").write_text("changed\n")
    await deltas.compare(SID, again=False)
    assert not await deltas.taken(SID)


async def test_a_stop_waits_for_none_of_the_reading_it_starts(tmp_path: Path) -> None:
    """The turn's telling is queued right after this, and a reading that is slow, that fails, or whose hook
    gives up must cost the turn its delta and never its telling."""
    root = repo(tmp_path)
    deltas = Deltas(record=lambda _entry: None, inherited=os.environ)
    await deltas.snapshot(SID, root)
    (root / "a.py").write_text("changed by something\n")
    start = time.perf_counter()
    await deltas.compare(SID, again=False)
    assert time.perf_counter() - start < 0.01, "the stop path waited for git"
    # And the reading still arrives, for whoever comes to take it.
    assert [file.path for file in (await deltas.taken(SID)).files] == ["a.py"]


# A loop that ends while a reading runs: asyncio.run cancels every task at once, the reading's included, wherever its
# git happens to be, from being spawned to being read. Run in a process of its own, since a loop that never closes
# would take the test run down with it.
SHUT_DOWN = """
import asyncio, os, sys, time
from pathlib import Path
from hands.sessions.delta import Deltas

root = Path(sys.argv[1])

async def ends(after: float) -> None:
    deltas = Deltas(record=lambda _entry: None, inherited=os.environ)
    await deltas.snapshot("s", root)
    (root / "a.py").write_text(f"x = {after}\\n")
    await deltas.compare("s", again=False)
    await asyncio.sleep(after)
    raise RuntimeError(time.monotonic())

slowest = 0.0
for step in range(30):
    try:
        asyncio.run(ends(step * 0.002))
    except RuntimeError as ended:
        slowest = max(slowest, time.monotonic() - ended.args[0])
    try:
        os.waitpid(-1, os.WNOHANG)
        sys.exit(f"a git was left unreaped when the loop ended {step * 2}ms into a reading")
    except ChildProcessError:
        pass
print(slowest)
"""


def test_a_loop_ended_mid_reading_kills_and_reaps_its_git_and_closes(tmp_path: Path) -> None:
    """The daemon's shutdown cancels a reading wherever it is, and must not then wait for ever on a git it spawned.

    Python 3.12's asyncio subprocesses did: cancelled while starting, they waited for an exit nothing would deliver."""
    root = repo(tmp_path)
    try:
        ran = subprocess.run((sys.executable, "-c", SHUT_DOWN, str(root)), capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise AssertionError("a loop ended mid-reading never finished closing") from None
    assert ran.returncode == 0, ran.stderr[-2000:]
    assert float(ran.stdout) < 1.0, f"the slowest loop took {ran.stdout.strip()}s to close"


async def test_two_turns_that_stop_before_either_is_told_keep_their_own_changes(tmp_path: Path) -> None:
    """The narrator summarises one turn at a time and takes seconds over each, so a session can stop twice
    before the first is told. Told the newer delta, the first turn would be given results it never had."""
    root = repo(tmp_path)
    deltas = Deltas(record=lambda _entry: None, inherited=os.environ)

    await deltas.snapshot(SID, root)
    (root / "first.py").write_text("turn one\n")
    await deltas.compare(SID, again=False)

    await deltas.snapshot(SID, root)
    (root / "second.py").write_text("turn two\n")
    await deltas.compare(SID, again=False)

    assert [file.path for file in (await deltas.taken(SID)).files] == ["first.py"]
    assert [file.path for file in (await deltas.taken(SID)).files] == ["second.py"]


async def test_a_turn_that_stops_with_nothing_to_read_still_takes_its_place_in_the_order(tmp_path: Path) -> None:
    """One reading is made for every turn that stops, so every telling takes exactly one and the two stay in
    step. A stop that reads nothing must still leave something to take, or every later turn is told the one
    before's changes."""
    root = repo(tmp_path)
    deltas = Deltas(record=lambda _entry: None, inherited=os.environ)
    await deltas.compare(SID, again=False)  # no mark: the daemon started in the middle of this turn

    await deltas.snapshot(SID, root)
    (root / "later.py").write_text("a later turn\n")
    await deltas.compare(SID, again=False)

    assert not await deltas.taken(SID)
    assert [file.path for file in (await deltas.taken(SID)).files] == ["later.py"]


async def test_a_reading_that_fails_outright_still_lets_the_turn_be_told(tmp_path: Path) -> None:
    """[LAW:no-silent-failure] the effect queued after the reading is the one that has the turn spoken at all.

    Without this the turn is never told and nothing says why: the user simply stops hearing about a session.
    """
    sessions = attached(tmp_path)
    await sessions.apply(Joined(Membership(SID, pid=4242, cwd=tmp_path, transcript=tmp_path / "t.jsonl"), "startup"))
    await sessions.apply(Stopped(SID, "Done.", mode=None, prompt=PromptId("p1"), again=False, heard=STOP_HEARD, request=STOP_REQUEST))
    story = await asyncio.wait_for(sessions.story(), 2.0)
    assert isinstance(story, Summarise) and story.session == SID


async def test_a_mark_that_fails_outright_still_lets_the_prompt_through(tmp_path: Path) -> None:
    """The same promise on the other side of the turn: a mark is taken while the user's prompt hook waits.

    Every way git itself can fail is already answered with None, but the reading needs a temporary file, and
    a TMPDIR that is full or read-only fails before git is ever run. Unguarded, that turns a best-effort read
    of a repository into an error on the prompt the user just typed [LAW:no-silent-failure].
    """
    sessions = attached(tmp_path)
    await sessions.apply(Joined(Membership(SID, pid=4242, cwd=tmp_path, transcript=tmp_path / "t.jsonl"), "startup"))
    await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p1")))
    # The mark is gone, the prompt is not: it went through, and the hook that said so was answered.
    assert [listing.session.turn for listing in sessions.live()] == [Opened(PromptId("p1"))]


async def test_a_repository_with_no_commit_yet_still_says_what_the_turn_did(tmp_path: Path) -> None:
    """`git init` and a first prompt: there is no commit to be on, so there is no `HEAD` to compare against.

    The tree is still a tree, and the first commit a turn makes is still reachable from where it ended.
    """
    root = tmp_path / "fresh"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "Test")

    def first() -> None:
        (root / "first.py").write_text("hello\n")
        git(root, "add", "-A")
        git(root, "commit", "-qm", "the very first commit")

    delta = await turn(root, first)
    assert [file.path for file in delta.files] == ["first.py"]
    assert [commit.subject for commit in delta.commits] == ["the very first commit"]


async def test_a_detached_head_is_read_like_any_other(tmp_path: Path) -> None:
    """What a bisect or a checkout of a bare sha leaves behind, which is a working state and not a broken one."""
    root = repo(tmp_path)
    git(root, "checkout", "-q", "--detach")
    delta = await turn(root, lambda: (root / "b.py").write_text("y = 2\n"))
    assert [file.path for file in delta.files] == ["b.py"]


class Slow(Deltas):
    """A repository whose tree takes everything the *mark* was given, which is what a slow one really does.

    The deadline is the only thing separating a repository with no commit from one git could not answer for,
    so a test of that has to spend the real deadline rather than hand one command a shorter one. Only the
    mark is slowed: a reading slowed too would end in the narrator's patience running out, and the turn would
    come back with no delta for a reason that has nothing to do with what this is testing.
    """

    marking = True

    async def _tree(self, root: Path, deadline: float) -> str | None:
        tree = await super()._tree(root, deadline)
        if self.marking:
            await asyncio.sleep(max(0.0, deadline - time.monotonic()) + 0.01)
            self.marking = False
        return tree


async def test_a_mark_that_ran_out_of_time_reading_where_it_stands_is_no_mark_at_all(tmp_path: Path) -> None:
    """Otherwise the history made before the turn is read as the history the turn made, and spoken that way.

    A repository with no commit yet and a repository git could not answer for both leave `rev-parse` saying
    nothing. A mark that takes the second for the first compares against no commit at all, so every commit
    ever made in that repository is reachable from where the turn ended and not from where it began.
    """
    root = repo(tmp_path)
    for n in range(3):
        (root / f"c{n}.py").write_text(f"n = {n}\n")
        git(root, "add", "-A")
        git(root, "commit", "-qm", f"made long before this turn {n}")

    deltas = Slow(record=lambda _entry: None, inherited=os.environ)
    await deltas.snapshot(SID, root)
    (root / "during.py").write_text("the turn's own work\n")
    await deltas.compare(SID, again=False)
    # No mark, so no delta: the turn is told by its steps alone, which is the one honest answer here.
    assert await deltas.taken(SID) == Delta()


def piled(root: Path, count: int) -> None:
    """A long history in one git invocation, because five hundred of them is half a minute of subprocesses."""
    branch = git(root, "symbolic-ref", "HEAD")
    stream = "".join(
        f"commit {branch}\ncommitter Test <t@example.com> {1700000000 + n} +0000\ndata {len(f'pulled {n}')}\npulled {n}\n"
        # The first of them says where the pile starts; each after it carries on from the one before.
        + (f"from {branch}^0\n" if n == 0 else "")
        for n in range(count)
    )
    subprocess.run(("git", "-C", str(root), "fast-import", "--quiet"), input=stream, text=True, check=True)


async def test_a_turn_that_pulled_a_history_keeps_no_more_of_it_than_it_could_ever_say(tmp_path: Path) -> None:
    """A pull or a rebase brings commits by the hundred, and every one of them was held, whole, until told."""
    root = repo(tmp_path)
    delta = await turn(root, lambda: piled(root, MOST_COMMITS + 5))
    assert len(delta.commits) == MOST_COMMITS


async def test_more_turns_than_can_be_held_lose_the_newest_deltas_and_never_the_pairing(tmp_path: Path) -> None:
    """The bound drops readings; nothing drops the tellings they belong to, which queue unbounded beside them.

    Dropped from the front, every telling from the first on would be handed the delta of the turn after its
    own — a listener can do something about changes they did not hear, and nothing about changes attributed
    to the wrong turn. So the turns past the bound are the ones told without a delta, and every turn that is
    handed one is handed its own.
    """
    root = repo(tmp_path)
    record: list[Entry] = []
    deltas = Deltas(record=record.append, inherited=os.environ)
    for n in range(HELD + 2):
        await deltas.snapshot(SID, root)
        (root / f"turn{n}.py").write_text(f"turn {n}\n")
        await deltas.compare(SID, again=False)

    told = [await deltas.taken(SID) for _ in range(HELD + 2)]
    assert [[file.path for file in delta.files] for delta in told[:HELD]] == [[f"turn{n}.py"] for n in range(HELD)]
    assert told[HELD:] == [Delta(), Delta()]
    assert sorted(str(event.facts["reading"]) for event in readings(record)) == ["dropped"] * 2 + ["read"] * HELD


class Exploding(Deltas):
    """A reading that raises, as a bug in it would."""

    async def _between(self, mark: Mark, deadline: float, forging: float) -> tuple[Delta, Mark | None, Asked]:
        raise RuntimeError("a bug")


class Endless(Deltas):
    """A reading that never finishes on its own, so the daemon's shutdown is what ends it."""

    async def _between(self, mark: Mark, deadline: float, forging: float) -> tuple[Delta, Mark | None, Asked]:
        await asyncio.Event().wait()
        raise AssertionError("never reached")


async def test_a_reading_that_raises_is_one_failed_audit_line(tmp_path: Path) -> None:
    root = repo(tmp_path)
    record: list[Entry] = []
    deltas = Exploding(record=record.append, inherited=os.environ)
    await deltas.snapshot(SID, root)
    await deltas.compare(SID, again=False)
    assert not await deltas.taken(SID)
    assert [event.outcome for event in readings(record)] == ["failed"]


async def test_a_reading_cancelled_by_the_shutdown_is_one_cancelled_audit_line(tmp_path: Path) -> None:
    root = repo(tmp_path)
    record: list[Entry] = []
    deltas = Endless(record=record.append, inherited=os.environ)
    await deltas.snapshot(SID, root)
    await deltas.compare(SID, again=False)
    [reading] = [task for task in asyncio.all_tasks() if task.get_name().startswith("what a turn of session")]
    await asyncio.sleep(0)
    reading.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reading
    assert [event.outcome for event in readings(record)] == ["cancelled"]


class Torn(Deltas):
    """A repository whose HEAD cannot be read in the moment the mark asks for it.

    A `git checkout` in the next terminal along, landing between two of the mark's own commands. There is
    time left on the clock, so nothing about the deadline says anything about this one way or the other.
    """

    marking = True

    async def _tree(self, root: Path, deadline: float) -> str | None:
        tree = await super()._tree(root, deadline)
        if self.marking:
            self.marking = False
            # A torn write, which is what a HEAD caught mid-rewrite is. A bogus-but-well-formed sha would
            # not do: `rev-parse --verify` answers a forty-character hex string without checking anything
            # is there. Only the mark is torn — git calls a directory with an unreadable HEAD no repository
            # at all, so tearing the reading too would end in an empty delta for that reason and not this one.
            (root / ".git" / "HEAD").write_text("a HEAD caught halfway through being rewritten\n")
        return tree


async def test_a_head_that_could_not_be_read_is_no_more_a_repository_with_no_commit_than_a_slow_one_is(tmp_path: Path) -> None:
    """Silence has several causes and only one of them means there is no commit, so unbornness is asked for.

    Inferred instead from a deadline with time left on it, a HEAD that merely could not be read is taken for
    a repository that has no commit to be on — and the whole history before the turn is told as the turn's.
    """
    root = repo(tmp_path)
    for n in range(3):
        git(root, "commit", "-q", "--allow-empty", "-m", f"made long before this turn {n}")
    stood = (root / ".git" / "HEAD").read_text()

    deltas = Torn(record=lambda _entry: None, inherited=os.environ)
    await deltas.snapshot(SID, root)
    (root / ".git" / "HEAD").write_text(stood)  # the checkout finished, and the repository reads again
    (root / "during.py").write_text("the turn's own work\n")
    await deltas.compare(SID, again=False)
    assert await deltas.taken(SID) == Delta()


async def test_a_turn_that_changed_more_lines_than_can_be_kept_is_told_by_its_files(tmp_path: Path) -> None:
    """git hands back a whole diff before a character of it is cut, and a generated file has no other bound."""
    root = repo(tmp_path)
    delta = await turn(root, lambda: (root / "generated.csv").write_text("n,x\n" * (MOST_LINES + 10)))
    assert [file.path for file in delta.files] == ["generated.csv"]
    assert not delta.patch


class Blind(Deltas):
    """A repository whose tree can be marked but not read again, which a slow enough `git add -A` does."""

    marking = True

    async def _tree(self, root: Path, deadline: float) -> str | None:
        if not self.marking:
            return None
        self.marking = False
        return await super()._tree(root, deadline)


async def test_a_commit_is_still_told_when_the_tree_it_left_behind_cannot_be_read(tmp_path: Path) -> None:
    """Two fast commands and one slow one, and the slow one held the fast ones' answer hostage.

    What a turn committed is the most narratable thing about it, and reading the tree is what runs out of a
    reading's deadline on a large repository — so the commits are read first and kept whatever the tree does.
    """
    root = repo(tmp_path)
    deltas = Blind(record=lambda _entry: None, inherited=os.environ)
    await deltas.snapshot(SID, root)
    (root / "b.py").write_text("y = 2\n")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "the one thing worth saying about this turn")
    await deltas.compare(SID, again=False)
    delta = await deltas.taken(SID)
    assert [commit.subject for commit in delta.commits] == ["the one thing worth saying about this turn"]


class Uncounted(Deltas):
    """A repository that will say its trees differ but not by how much.

    What a numstat running past the reading's deadline does — on exactly the diff whose size was the reason
    to ask how big it was.
    """

    async def _git(self, cwd: Path, *args: str, env: Mapping[str, str] | None = None, deadline: float) -> str | None:
        if args[:2] == ("diff", "--numstat"):
            return None
        return await super()._git(cwd, *args, env=env, deadline=deadline)


async def test_a_turn_whose_changes_could_not_be_counted_has_its_patch_left_unread(tmp_path: Path) -> None:
    """git saying nothing is not git saying nothing changed, and counted as the second the bound is not one.

    Every count is then zero, the guard that keeps a generated file out of the daemon's memory passes, and
    the whole diff is read after all. What the turn committed is known without counting anything, so that
    much is still told.
    """
    root = repo(tmp_path)
    deltas = Uncounted(record=lambda _entry: None, inherited=os.environ)
    await deltas.snapshot(SID, root)
    (root / "generated.csv").write_text("n,x\n" * (MOST_LINES + 10))
    git(root, "add", "-A")
    git(root, "commit", "-qm", "wrote the generated file")
    await deltas.compare(SID, again=False)

    delta = await deltas.taken(SID)
    assert not delta.patch and not delta.files
    assert [commit.subject for commit in delta.commits] == ["wrote the generated file"]


async def test_a_turn_claude_code_said_is_over_keeps_its_own_changes_when_the_next_prompt_lands_first(tmp_path: Path) -> None:
    """The next prompt's hook landed before the record of p1's interrupt was read. p1 is compared there, before p2 is
    marked, so p2 is told only what p2 changed."""
    root = repo(tmp_path)
    deltas = Deltas(record=lambda _entry: None, inherited=os.environ)
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _entry: None, changes=deltas)
    await sessions.apply(Joined(Membership(SID, pid=4242, cwd=root, transcript=tmp_path / "t.jsonl"), "startup"))
    await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p1")))
    await sessions.apply(Taken(SID, PromptId("p1"), None, 5.0))
    (root / "first.py").write_text("turn one\n")
    await sessions.apply(StatusReported(SID, Report(status.Idle(), Stamp(1)), at=4.0))
    await sessions.apply(Prompted(SID, at=5.0, mode=None, prompt=PromptId("p2")))
    (root / "second.py").write_text("turn two\n")
    await sessions.apply(Stopped(SID, "Done.", mode=None, prompt=PromptId("p2"), again=False, heard=STOP_HEARD, request=STOP_REQUEST))

    assert await asyncio.wait_for(sessions.story(), 2.0) == Summarise(SID, PromptId("p1"), None)
    assert [file.path for file in (await deltas.taken(SID)).files] == ["first.py"]
    assert await asyncio.wait_for(sessions.story(), 2.0) == Summarise(SID, PromptId("p2"), "Done.")
    assert [file.path for file in (await deltas.taken(SID)).files] == ["second.py"]


async def test_a_message_queued_behind_a_turn_is_told_only_what_its_own_turn_changed(tmp_path: Path) -> None:
    """hands-status-bpp.44l: no prompt of its own marks it, so p1's Stop does, while its hook holds Claude Code (2.1.282)."""
    root = repo(tmp_path)
    deltas = Deltas(record=lambda _entry: None, inherited=os.environ)
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _entry: None, changes=deltas)
    await sessions.apply(Joined(Membership(SID, pid=4242, cwd=root, transcript=tmp_path / "t.jsonl"), "startup"))
    await sessions.apply(StatusReported(SID, Report(status.Idle(), Stamp(1)), at=0.5))
    await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p1")))
    await sessions.apply(Taken(SID, PromptId("p1"), Stamp(2), 1.5))
    await sessions.apply(Prompted(SID, at=2.0, mode=None, prompt=PromptId("p1")))
    (root / "first.py").write_text("turn one\n")
    await sessions.apply(Stopped(SID, "One.", mode=None, prompt=PromptId("p1"), again=False, heard=STOP_HEARD, request=STOP_REQUEST))
    await sessions.apply(Taken(SID, PromptId("p2"), Stamp(4), 4.0))
    (root / "queued.py").write_text("the queued turn\n")
    await sessions.apply(Stopped(SID, "Two.", mode=None, prompt=PromptId("p2"), again=False, heard=STOP_HEARD, request=STOP_REQUEST))

    assert await asyncio.wait_for(sessions.story(), 2.0) == Summarise(SID, PromptId("p1"), "One.")
    assert [file.path for file in (await deltas.taken(SID)).files] == ["first.py"]
    assert await asyncio.wait_for(sessions.story(), 2.0) == Summarise(SID, PromptId("p2"), "Two.")
    assert [file.path for file in (await deltas.taken(SID)).files] == ["queued.py"]




class Marking:
    """A repository reader whose marks after the first take as long as the test says."""

    def __init__(self) -> None:
        self.marks, self.started, self.release = 0, asyncio.Event(), asyncio.Event()

    async def snapshot(self, session: SessionId, cwd: Path) -> None:
        self.marks += 1
        if self.marks > 1:
            self.started.set()
            await self.release.wait()

    async def compare(self, session: SessionId, again: bool) -> None: ...

    async def taken(self, session: SessionId) -> Delta:
        return Delta()


async def test_a_stop_whose_turn_the_wire_told_holds_claude_code_until_the_turn_queued_behind_it_is_marked(tmp_path: Path) -> None:
    """The reply on the wire tells the turn and starts marking the one queued behind it before the Stop fires. The Stop
    then ends nothing, and its hook is let go only once that mark is taken: Claude Code runs the queued turn after it."""
    repository = Marking()
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _entry: None, changes=repository)
    await sessions.apply(Joined(Membership(SID, pid=4242, cwd=tmp_path, transcript=tmp_path / "t.jsonl"), "startup"))
    await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p1")))
    await sessions.apply(Prompted(SID, at=2.0, mode=None, prompt=PromptId("p1")))
    sessions.hear(Closed(SID, PromptId("p1"), "Done."))
    await asyncio.wait_for(repository.started.wait(), 2.0)

    stopping = asyncio.create_task(sessions.stop(Stopped(SID, "Done.", mode=None, prompt=PromptId("p1"), again=False, heard=STOP_HEARD, request=STOP_REQUEST)))
    done, _ = await asyncio.wait({stopping}, timeout=0.2)
    assert not done, "the Stop's hook was let go while the queued turn was still being marked"
    repository.release.set()
    await asyncio.wait_for(stopping, 2.0)
    assert await asyncio.wait_for(sessions.story(), 2.0) == Summarise(SID, PromptId("p1"), "Done.")


class Reading:
    """A repository reader whose readings take as long as the test says."""

    def __init__(self) -> None:
        self.started, self.release = asyncio.Event(), asyncio.Event()

    async def snapshot(self, session: SessionId, cwd: Path) -> None: ...

    async def compare(self, session: SessionId, again: bool) -> None:
        self.started.set()
        await self.release.wait()

    async def taken(self, session: SessionId) -> Delta:
        return Delta()


async def test_a_session_is_told_gone_only_after_the_turn_it_finished_is_told(tmp_path: Path) -> None:
    repository = Reading()
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _entry: None, changes=repository)
    await sessions.apply(Joined(Membership(SID, pid=4242, cwd=tmp_path, transcript=tmp_path / "t.jsonl"), "startup"))
    await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p1")))
    sessions.hear(Closed(SID, PromptId("p1"), "Done."))
    await asyncio.wait_for(repository.started.wait(), 2.0)
    sessions.hear(Ended(SID, "other"))
    repository.release.set()
    assert await asyncio.wait_for(sessions.story(), 2.0) == Summarise(SID, PromptId("p1"), "Done.")
    assert await asyncio.wait_for(sessions.story(), 2.0) == SessionGone(SID)
