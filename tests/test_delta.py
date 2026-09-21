"""What a turn changed in the repository it ran in, read from git without disturbing it."""

import subprocess
import time
from pathlib import Path

from hands.core.delta import Delta
from hands.core.session import SessionId
from hands.sessions.delta import Deltas

SID = SessionId("s1")


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


async def turn(root: Path, work: object = None) -> Delta:
    """One turn: mark where the repository is, let `work` happen, then read what changed."""
    deltas = Deltas()
    await deltas.snapshot(SID, root)
    if callable(work):
        work()
    await deltas.compare(SID)
    return await deltas.taken(SID)


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


async def test_a_session_that_works_outside_a_repository_is_told_by_its_steps_alone(tmp_path: Path) -> None:
    """[LAW:no-silent-failure] not every session is in a repository, and that is not a failure to report."""
    plain = tmp_path / "plain"
    plain.mkdir()
    assert not await turn(plain, lambda: (plain / "a.txt").write_text("hello\n"))


async def test_a_directory_that_is_not_there_is_told_by_its_steps_alone(tmp_path: Path) -> None:
    assert not await turn(tmp_path / "never-existed")


async def test_a_turn_stopping_with_no_mark_before_it_reads_nothing(tmp_path: Path) -> None:
    """The daemon started in the middle of a turn: there is no beginning to compare against, so there is no delta."""
    deltas = Deltas()
    await deltas.compare(SID)
    assert not await deltas.taken(SID)


async def test_a_delta_is_told_once_and_never_twice(tmp_path: Path) -> None:
    """A delta told is a delta spent; the next turn's is the next turn's."""
    root = repo(tmp_path)
    deltas = Deltas()
    await deltas.snapshot(SID, root)
    (root / "a.py").write_text("x = 3\n")
    await deltas.compare(SID)
    assert await deltas.taken(SID)
    assert not await deltas.taken(SID)


async def test_a_new_turn_reads_against_its_own_beginning_and_not_the_one_before(tmp_path: Path) -> None:
    root = repo(tmp_path)
    deltas = Deltas()
    await deltas.snapshot(SID, root)
    (root / "a.py").write_text("first turn\n")
    await deltas.compare(SID)
    assert [file.path for file in (await deltas.taken(SID)).files] == ["a.py"]

    await deltas.snapshot(SID, root)
    (root / "b.py").write_text("second turn\n")
    await deltas.compare(SID)
    assert [file.path for file in (await deltas.taken(SID)).files] == ["b.py"]


async def test_a_prompt_and_a_stop_through_the_daemon_read_what_the_turn_changed(tmp_path: Path) -> None:
    """End to end through the parts that decide it: the reducer emits, the registry performs, in that order.

    The turn here is what the ticket is for — a shell command changed a file and no step in the transcript
    says so — and the delta is ready before the turn is ever handed over to be summarised.
    """
    from hands.core.events import Joined, Prompted, Stopped
    from hands.core.session import Membership
    from hands.sessions.registry import Sessions

    root = repo(tmp_path)
    deltas = Deltas()
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _entry: None, changes=deltas)
    await sessions.apply(Joined(Membership(SID, pid=4242, cwd=root, transcript=tmp_path / "t.jsonl"), "startup"))

    await sessions.apply(Prompted(SID, at=1.0))
    subprocess.run(("sed", "-i", "", "s/x = 1/x = 99/", str(root / "a.py")), check=True)
    await sessions.apply(Stopped(SID, "Done."))

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
    deltas = Deltas(marking=0.0)
    await deltas.snapshot(SID, root)
    (root / "a.py").write_text("changed\n")
    await deltas.compare(SID)
    assert not await deltas.taken(SID)


async def test_a_stop_waits_for_none_of_the_reading_it_starts(tmp_path: Path) -> None:
    """The turn's telling is queued right after this, and a reading that is slow, that fails, or whose hook
    gives up must cost the turn its delta and never its telling."""
    root = repo(tmp_path)
    deltas = Deltas()
    await deltas.snapshot(SID, root)
    (root / "a.py").write_text("changed by something\n")
    start = time.perf_counter()
    await deltas.compare(SID)
    assert time.perf_counter() - start < 0.01, "the stop path waited for git"
    # And the reading still arrives, for whoever comes to take it.
    assert [file.path for file in (await deltas.taken(SID)).files] == ["a.py"]


async def test_two_turns_that_stop_before_either_is_told_keep_their_own_changes(tmp_path: Path) -> None:
    """The narrator summarises one turn at a time and takes seconds over each, so a session can stop twice
    before the first is told. Told the newer delta, the first turn would be given results it never had."""
    root = repo(tmp_path)
    deltas = Deltas()

    await deltas.snapshot(SID, root)
    (root / "first.py").write_text("turn one\n")
    await deltas.compare(SID)

    await deltas.snapshot(SID, root)
    (root / "second.py").write_text("turn two\n")
    await deltas.compare(SID)

    assert [file.path for file in (await deltas.taken(SID)).files] == ["first.py"]
    assert [file.path for file in (await deltas.taken(SID)).files] == ["second.py"]


async def test_a_turn_that_stops_with_nothing_to_read_still_takes_its_place_in_the_order(tmp_path: Path) -> None:
    """One reading is made for every turn that stops, so every telling takes exactly one and the two stay in
    step. A stop that reads nothing must still leave something to take, or every later turn is told the one
    before's changes."""
    root = repo(tmp_path)
    deltas = Deltas()
    await deltas.compare(SID)  # no mark: the daemon started in the middle of this turn

    await deltas.snapshot(SID, root)
    (root / "later.py").write_text("a later turn\n")
    await deltas.compare(SID)

    assert not await deltas.taken(SID)
    assert [file.path for file in (await deltas.taken(SID)).files] == ["later.py"]


async def test_a_reading_that_fails_outright_still_lets_the_turn_be_told(tmp_path: Path) -> None:
    """[LAW:no-silent-failure] the effect queued after the reading is the one that has the turn spoken at all.

    Without this the turn is never told and nothing says why: the user simply stops hearing about a session.
    """
    import asyncio

    from hands.core.effects import Summarise
    from hands.core.events import Joined, Stopped
    from hands.core.session import Membership
    from hands.sessions.registry import Sessions

    class Breaks:
        async def snapshot(self, session: SessionId, cwd: Path) -> None: ...

        async def compare(self, session: SessionId) -> None:
            raise RuntimeError("git is on fire")

        async def taken(self, session: SessionId) -> Delta:
            return Delta()

    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _entry: None, changes=Breaks())
    await sessions.apply(Joined(Membership(SID, pid=4242, cwd=tmp_path, transcript=tmp_path / "t.jsonl"), "startup"))
    await sessions.apply(Stopped(SID, "Done."))
    story = await asyncio.wait_for(sessions.story(), 2.0)
    assert isinstance(story, Summarise) and story.session == SID


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
