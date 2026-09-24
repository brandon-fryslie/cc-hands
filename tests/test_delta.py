"""What a turn changed in the repository it ran in, read from git without disturbing it."""

import asyncio
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

from hands.core.delta import Delta
from hands.core.effects import Summarise
from hands.core.events import Joined, Prompted, Stopped, Taken
from hands.core.session import Membership, PromptId, SessionId, Submitted
from hands.sessions.delta import HELD, MOST_COMMITS, MOST_LINES, Deltas
from hands.sessions.registry import Sessions

SID = SessionId("s1")


class Breaks:
    """A repository reader that fails at both ends, which neither end of a turn may be made to care about."""

    async def snapshot(self, session: SessionId, cwd: Path) -> None:
        raise RuntimeError("there is nowhere to put a scratch index")

    async def compare(self, session: SessionId) -> None:
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

    await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=None))
    subprocess.run(("sed", "-i", "", "s/x = 1/x = 99/", str(root / "a.py")), check=True)
    await sessions.apply(Stopped(SID, "Done.", mode=None, prompt=None))

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
    sessions = attached(tmp_path)
    await sessions.apply(Joined(Membership(SID, pid=4242, cwd=tmp_path, transcript=tmp_path / "t.jsonl"), "startup"))
    await sessions.apply(Stopped(SID, "Done.", mode=None, prompt=None))
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
    assert [listing.session.state for listing in sessions.live()] == [Submitted(since=1.0)]


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

    deltas = Slow()
    await deltas.snapshot(SID, root)
    (root / "during.py").write_text("the turn's own work\n")
    await deltas.compare(SID)
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
    deltas = Deltas()
    for n in range(HELD + 2):
        await deltas.snapshot(SID, root)
        (root / f"turn{n}.py").write_text(f"turn {n}\n")
        await deltas.compare(SID)

    told = [await deltas.taken(SID) for _ in range(HELD + 2)]
    assert [[file.path for file in delta.files] for delta in told[:HELD]] == [[f"turn{n}.py"] for n in range(HELD)]
    assert told[HELD:] == [Delta(), Delta()]


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

    deltas = Torn()
    await deltas.snapshot(SID, root)
    (root / ".git" / "HEAD").write_text(stood)  # the checkout finished, and the repository reads again
    (root / "during.py").write_text("the turn's own work\n")
    await deltas.compare(SID)
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
    deltas = Blind()
    await deltas.snapshot(SID, root)
    (root / "b.py").write_text("y = 2\n")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "the one thing worth saying about this turn")
    await deltas.compare(SID)
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
    deltas = Uncounted()
    await deltas.snapshot(SID, root)
    (root / "generated.csv").write_text("n,x\n" * (MOST_LINES + 10))
    git(root, "add", "-A")
    git(root, "commit", "-qm", "wrote the generated file")
    await deltas.compare(SID)

    delta = await deltas.taken(SID)
    assert not delta.patch and not delta.files
    assert [commit.subject for commit in delta.commits] == ["wrote the generated file"]


async def test_a_turn_that_ended_unheard_before_the_next_prompt_keeps_its_own_changes(tmp_path: Path) -> None:
    """The next prompt's hook landed before the record of p1's interrupt was read. p1 is compared there, before p2 is
    marked, so p2 is told only what p2 changed."""
    root = repo(tmp_path)
    deltas = Deltas()
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _entry: None, changes=deltas)
    await sessions.apply(Joined(Membership(SID, pid=4242, cwd=root, transcript=tmp_path / "t.jsonl"), "startup"))
    await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p1")))
    await sessions.apply(Taken(SID, PromptId("p1")))
    (root / "first.py").write_text("turn one\n")
    await sessions.apply(Prompted(SID, at=5.0, mode=None, prompt=PromptId("p2")))
    (root / "second.py").write_text("turn two\n")
    await sessions.apply(Stopped(SID, "Done.", mode=None, prompt=PromptId("p2")))

    assert await asyncio.wait_for(sessions.story(), 2.0) == Summarise(SID, PromptId("p1"), None)
    assert [file.path for file in (await deltas.taken(SID)).files] == ["first.py"]
    assert await asyncio.wait_for(sessions.story(), 2.0) == Summarise(SID, PromptId("p2"), "Done.")
    assert [file.path for file in (await deltas.taken(SID)).files] == ["second.py"]
