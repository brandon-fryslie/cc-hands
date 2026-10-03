"""What a turn changed in the repository a session works in, read from git without disturbing it.

Every command here reads: nothing moves a ref, stages anything, or touches the working tree. The one mark it
leaves is in the object database, where writing a tree costs one unreferenced object per content git has not
stored already, and git's own housekeeping collects them. Content is what git names an object by, so a file
that does not change costs its size once however many turns read it: measured at 1.6 MB the first time two
hundred untracked files were seen, and nothing at all on the two readings after. That is the same price
`git stash create` charges, and the reason this does not use it is that a stash holds no untracked file, and
the new file a code generator wrote is exactly what a turn must be able to name [LAW:effects-at-boundaries].
"""

import asyncio
import json
import os
import tempfile
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from loguru import logger

from hands.core.delta import Branched, Changed, Commit, Delta, PullRequested, Pushed
from hands.core.session import SessionId
from hands.sessions.audit import DeltaRead, DeltaReadOutcome, Forge, Record
from hands.sessions.child import run
from hands.sessions.hookconfig import POST_TIMEOUT_SECONDS

# What a mark may spend, all its git commands together. It is taken while a prompt's hook waits on the daemon,
# and the shim gives up after POST_TIMEOUT_SECONDS and prints that it cannot reach the daemon — so a mark that
# costs more than the hook can afford is worse than no mark at all, and the half second is the rest of the
# round trip [LAW:carrying-cost]. A repository too slow for this has its turn told by its steps alone.
MARKING = POST_TIMEOUT_SECONDS - 0.5

# What reading a delta may spend. Nothing holds a hook open for this: it runs on its own once the turn has
# stopped, and the only thing waiting is a narrator about to spend seconds on a model.
READING = 20.0

# How long the narrator waits for a reading still going before it tells the turn without it.
PATIENCE = 3.0

# Readings held for a narrator that has not taken them. One per turn; a daemon whose voice never loaded has
# no narrator at all, and must not grow a patch per turn for one that is never coming.
HELD = 8

# The most of a patch kept in memory until it is told. The summariser's budget cuts it again and much smaller;
# this only stops a formatter's rewrite of a whole repository from sitting here until someone asks.
MOST = 40_000

# The most changed lines a turn can have and still have its patch read. Counted from the numstat, which is
# one line a file, before the diff itself is ever asked for: see _between.
MOST_LINES = 20_000

# What asking the forge which pull requests a pushed branch has may spend. It is asked beside the tree and not
# before it, and a narrator waits PATIENCE for the whole reading, so with this inside that a slow forge costs the
# turn its pull request and never its commit: the forge is a network away where git is a disk away.
FORGING = 2.0

# What git writes in a remote-tracking ref's log when a push moved it, where a fetch writes `fetch` or `pull`.
_PUSHED = "update by push"

# The most commits kept from one turn. A turn that makes them one at a time makes a handful; past this it
# pulled or rebased a history, and how many there were is the story where which ones they were is not.
MOST_COMMITS = 500


@dataclass(frozen=True)
class Mark:
    """Where a repository stood: the commit it was on, and a tree of everything in it git would keep."""

    root: Path
    # None only where a repository has no commit to be on, never where git could not say. A mark that cannot
    # tell those two apart has every commit in the repository read as the work of one turn: see _mark.
    head: str | None
    tree: str
    # Every local branch and remote-tracking ref, and the commit each names. None where git would not list them,
    # never empty for it: read as a repository with no branches, every branch it has is one the turn made.
    refs: Mapping[str, str] | None
    # When the mark was taken, on the wall clock a forge stamps a pull request with.
    at: datetime


# A mark being taken, or taken: None where it could not be.
Marking = asyncio.Future[Mark | None]


@dataclass(frozen=True)
class Asked:
    """Whether the forge was asked about a pushed branch's pull requests, and how long it took to answer or not."""

    forge: Forge
    seconds: float


UNASKED = Asked("unasked", 0.0)


class Changes(Protocol):
    """What the registry and the narrator need of a repository reader, so either can be given one that reads none."""

    async def snapshot(self, session: SessionId, cwd: Path) -> None: ...

    async def compare(self, session: SessionId, again: bool) -> None: ...

    async def taken(self, session: SessionId) -> Delta: ...


class NoChanges:
    """Reads no repository, so every turn is told by its steps alone. What a daemon with no git gets."""

    async def snapshot(self, session: SessionId, cwd: Path) -> None: ...

    async def compare(self, session: SessionId, again: bool) -> None: ...

    async def taken(self, session: SessionId) -> Delta:
        return Delta()


class Deltas:
    """Where each session's repository stood when its turn began, so what the turn changed can be read at its end."""

    def __init__(self, record: Record, marking: float = MARKING, reading: float = READING, patience: float = PATIENCE) -> None:
        self._record = record
        self._marking = marking
        self._reading = reading
        self._patience = patience
        # The mark each session's last snapshot took, or is taking: held from the moment the snapshot starts, so a
        # reading that needs it waits for it rather than mistaking one not taken yet for one that could not be.
        # Bounded by the sessions the registry itself keeps, which holds every session it has heard of.
        self._marks: dict[SessionId, Marking] = {}
        # Where each session's last reading found the repository: the mark of a turn going on after another Stop hook
        # blocked its Stop, which no prompt marked. Kept apart from _marks, so no other turn is read against it, and
        # replaced by the next reading. Bounded as _marks is.
        self._reached: dict[SessionId, Marking] = {}
        # One reading per turn that stopped, oldest first, so a session that stops twice while the narrator is
        # busy has each turn told with its own delta rather than the newer one told as both.
        self._readings: dict[SessionId, deque[asyncio.Future[Delta]]] = {}
        self._running: set[asyncio.Task[None]] = set()

    async def snapshot(self, session: SessionId, cwd: Path) -> None:
        """Mark where the repository stands, before the turn has had the chance to change anything in it.

        Awaited while the prompt's hook waits, which is what makes it a mark of the turn's beginning rather
        than of some moment inside it — and why all of it together is given less than that hook can afford.
        """
        marking: Marking = asyncio.get_running_loop().create_future()
        self._marks[session] = marking
        mark = None
        try:
            mark = await self._mark(cwd, time.monotonic() + self._marking)
        finally:
            # [LAW:no-silent-failure] answered however this ends, a hook that gave up included, so no reading waits on it for ever.
            marking.set_result(mark)
        if mark is None:
            # Not a repository, or one that cannot be read: the turn is told by its steps, which is most of it.
            logger.debug(f"nothing to compare a turn of session {session} against in {cwd}")

    async def compare(self, session: SessionId, again: bool) -> None:
        """Take the turn's place in the order and start reading what it changed. Waits for none of it.

        Started when the turn stops, not when its summary is made: summaries are made one at a time and take
        seconds, by which time the session may have begun another turn and changed more, and the delta would
        then hold two turns' work and be told as one [LAW:no-ambient-temporal-coupling].

        Nothing is awaited here, and that is the point: this runs while a Stop hook waits on the daemon, and
        the effect queued after it is the one that has the turn spoken at all. A reading that is slow, that
        fails, or whose hook gives up and has its handler cancelled must cost the turn its delta and never
        its telling [LAW:no-silent-failure].
        """
        held = self._readings.setdefault(session, deque())
        start = (self._reached if again else self._marks).pop(session, None)
        # [LAW:no-ambient-temporal-coupling] where this reading finds the repository is where the turn goes on from if
        # another Stop hook blocked its Stop: the part going on is read against it, so a file changed after the first
        # Stop is told with the second, however the reading and the work interleave.
        end: Marking = asyncio.get_running_loop().create_future()
        self._reached[session] = end
        if len(held) >= HELD:
            # Dropped from the back, never the front, and that is the whole of what keeps the two sides in
            # step. A telling takes the oldest reading, and tellings are not dropped alongside readings —
            # they queue unbounded — so evicting the front would hand every telling after it the delta of
            # the turn after its own, which is the one thing `taken` exists to prevent. Dropped from the
            # back, every turn that has a delta has its own [LAW:no-ambient-temporal-coupling].
            logger.warning(f"{HELD} deltas of session {session} are already waiting to be told, so this turn is told without one")
            end.set_result(None)
            self._record(DeltaRead(session, "dropped", 0, 0, (), UNASKED.forge, UNASKED.seconds, 0.0))
            return
        pending: asyncio.Future[Delta] = asyncio.get_running_loop().create_future()
        held.append(pending)
        if start is None:
            pending.set_result(Delta())
            end.set_result(None)
            self._record(DeltaRead(session, "unmarked", 0, 0, (), UNASKED.forge, UNASKED.seconds, 0.0))
            return
        task = asyncio.create_task(self._read(session, pending, start, end), name=f"what a turn of session {session} changed")
        # Held, because the loop keeps only a weak reference and would collect a task nobody is awaiting.
        self._running.add(task)
        task.add_done_callback(self._running.discard)

    async def taken(self, session: SessionId) -> Delta:
        """What the turn that stopped changed, waited for while its reading is still going. Taken once.

        Spent whatever becomes of the turn it belongs to, including a summary that fails. One reading is made
        for every turn that stops and one telling is made for every turn that stops, so each telling takes
        exactly one, and that is the whole of what keeps the two in step. Held back for a turn whose summary
        failed, this reading would be taken by the next turn's telling and that turn would be told the turn
        before's changes — and a listener can do something about changes they did not hear, and nothing about
        changes attributed to the wrong turn [LAW:no-ambient-temporal-coupling].
        """
        held = self._readings.get(session)
        if not held:
            return Delta()
        reading = held.popleft()
        try:
            return await asyncio.wait_for(asyncio.shield(reading), self._patience)
        except TimeoutError:
            logger.info(f"what a turn of session {session} changed is still being read, so the turn is told without it")
            return Delta()

    async def _read(self, session: SessionId, pending: asyncio.Future[Delta], start: Marking, end: Marking) -> None:
        """[LAW:no-silent-failure] whatever happens here, whoever is waiting is answered rather than left."""
        began = time.monotonic()
        delta, reached, asked = Delta(), None, UNASKED
        outcome: DeltaReadOutcome = "unmarked"
        try:
            mark = await start
            if mark is not None:
                delta, reached, asked = await self._between(mark, time.monotonic() + self._reading)
                outcome = "read"
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as error:
            outcome = "failed"
            logger.error(f"what a turn changed could not be read: {type(error).__name__}: {error}")
        finally:
            # Answered however this ends, a cancelled reading included, so the part going on never waits on it for ever.
            end.set_result(reached)
            # [LAW:nothing-unseen] one line for every reading however it ended, so a turn told without its delta can be
            # told apart from one that changed nothing, and a slow forge from a slow repository.
            self._record(
                DeltaRead(session, outcome, len(delta.commits), len(delta.files), delta.changes, asked.forge, asked.seconds, round(time.monotonic() - began, 3))
            )
        if not pending.done():
            pending.set_result(delta)

    async def _mark(self, cwd: Path, deadline: float) -> Mark | None:
        at = datetime.now(UTC)
        root = await self._git(cwd, "rev-parse", "--show-toplevel", deadline=deadline)
        if not root:
            return None
        tree = await self._tree(Path(root), deadline)
        if tree is None:
            return None
        head = await self._git(Path(root), "rev-parse", "--verify", "--quiet", "HEAD", deadline=deadline)
        if head is None and not await self._unborn(Path(root), deadline):
            # [LAW:parse-dont-validate] read as a repository with no commit, a mark that is really one git
            # could not answer for compares against no commit at all: every commit ever made in it is then
            # reachable from where the turn ended and not from where it began, and the turn is spoken as
            # having made all of them. What is not known is refused where a mark is built, so no reading
            # downstream can be handed one that does not know where it stands.
            logger.warning(f"where {root} stands could not be read, so the turn is told without its delta")
            return None
        return Mark(Path(root), head, tree, await self._refs(Path(root), deadline), at)

    async def _unborn(self, root: Path, deadline: float) -> bool:
        """Whether a repository that would not say where it stands has nowhere to stand yet.

        HEAD naming a branch that no commit is on is what every repository looks like between `git init` and
        its first commit, and asking for it is a positive answer where the absence of one is not: a HEAD
        being rewritten by a checkout in the next terminal along, a ref that cannot be read, and a deadline
        with nothing left on it all leave the same silence, and none of them means there is no commit.
        """
        return await self._git(root, "symbolic-ref", "--quiet", "HEAD", deadline=deadline) is not None

    async def _between(self, mark: Mark, deadline: float) -> tuple[Delta, Mark | None, Asked]:
        """What changed from the mark to where the repository stands now, that place as a mark, and what the forge was asked."""
        at = datetime.now(UTC)
        # Read before the tree, because they are two fast commands where the tree is the slow one: a turn
        # whose commit is the one thing worth saying about it should not lose that because `git add -A` took
        # longer than a reading is given, or failed for a reason that has nothing to do with the commit.
        head = await self._git(mark.root, "rev-parse", "HEAD", deadline=deadline)
        commits = await self._commits(mark, head, deadline)
        refs = await self._refs(mark.root, deadline)
        # Beside the tree and not before it: the forge is a network away where git is a disk away, and the narrator
        # gives up on the whole reading at once, so a forge read first spends the commit's time on a pull request.
        (files, patch, tree), (changes, asked) = await asyncio.gather(self._worked(mark, deadline), self._moved(mark, refs, deadline))
        # [LAW:parse-dont-validate] as in _mark: where the repository stands is known only if both were read, and a HEAD
        # that would not answer is no commit only where git says there is none yet.
        reached = None if tree is None or (head is None and not await self._unborn(mark.root, deadline)) else Mark(mark.root, head, tree, refs, at)
        return Delta(files, commits, changes, patch), reached, asked

    async def _worked(self, mark: Mark, deadline: float) -> tuple[tuple[Changed, ...], str, str | None]:
        """The files the turn left different, the patch between, and the tree it left them in: None where unreadable."""
        tree = await self._tree(mark.root, deadline)
        if tree is None or tree == mark.tree:
            # Unreadable, or the working tree came back to where it started — which a commit and nothing else does.
            return (), "", tree
        numstat = await self._git(mark.root, "diff", "--numstat", mark.tree, tree, deadline=deadline)
        if numstat is None:
            # [LAW:no-silent-failure] git not answering is not git saying nothing changed. Counted as the
            # second, the numbers that decide whether the patch can be kept are all zero and the bound they
            # are here to enforce is not enforced at all — on the diff most likely to have been what stopped
            # the counting. What the turn committed is known either way, so that much is still told.
            logger.info(f"what a turn changed in {mark.root} could not be counted, so its patch is not read")
            return (), "", tree
        files = _files(numstat)
        counted = sum((file.added or 0) + (file.removed or 0) for file in files)
        if counted > MOST_LINES:
            # git hands back a whole diff before a character of it is cut, so a turn that wrote a million-line
            # file inside the repository would have all of it here at once. The counts cost one line a file and
            # are already in hand, so they are what says no — and what is left, the files and their counts, is
            # all of a diff that size that would have survived the summariser's budget anyway.
            logger.info(f"a turn changed {counted} lines in {mark.root}, too many to keep the patch of, so its files are told instead")
            return files, "", tree
        patch = await self._git(mark.root, "diff", mark.tree, tree, deadline=deadline)
        return files, "" if patch is None else patch[:MOST], tree

    async def _commits(self, mark: Mark, head: str | None, deadline: float) -> tuple[Commit, ...]:
        if head is None or head == mark.head:
            return ()
        # What is reachable from where the turn ended and not from where it began, which is what the turn
        # added however it got there — a merge, a rebase, or an amend that rewrote the commit before it.
        listed = await self._git(
            mark.root, "log", f"--max-count={MOST_COMMITS}", "--format=%h%x1f%s", f"{mark.head}..{head}" if mark.head else head, deadline=deadline
        )
        return () if not listed else tuple(Commit(*line.split("\x1f", 1)) for line in listed.splitlines() if "\x1f" in line)

    async def _refs(self, root: Path, deadline: float) -> dict[str, str] | None:
        """Each local branch and remote-tracking ref and the commit it names. A symbolic ref names a ref, not a
        commit — `origin/HEAD` follows the remote's default branch — so none of them is one."""
        listed = await self._git(root, "for-each-ref", "--format=%(refname)%00%(objectname)%00%(symref)", "refs/heads", "refs/remotes", deadline=deadline)
        if listed is None:
            return None
        return {name: sha for line in listed.splitlines() for name, sha, symbolic in [line.split("\0")] if not symbolic}

    async def _moved(self, mark: Mark, refs: Mapping[str, str] | None, deadline: float) -> tuple[tuple[Pushed | Branched | PullRequested, ...], Asked]:
        """Whether the turn made the branch this worktree is on, whether it pushed it, and the pull requests opened from it.

        That branch alone. Every worktree of a repository, and the terminal beside it, share refs/heads and refs/remotes
        and their logs, so a branch made or pushed anywhere else is another session's work — and the branch checked
        out here is checked out nowhere else. Each is read from git's own log of how the ref moved and when, so a
        branch checked out or renamed is not one the turn made, and a push the turn followed with a pull is still a push.
        """
        if mark.refs is None or refs is None:
            return (), UNASKED
        branch = await self._git(mark.root, "symbolic-ref", "--quiet", "--short", "HEAD", deadline=deadline)
        if branch is None:
            # A detached HEAD is on no branch for the turn to have made or pushed.
            return (), UNASKED
        since = int(mark.at.timestamp())
        # `refs/remotes/<remote>/<branch>`, and a branch may hold slashes where a remote does not.
        tracking = [name for name in refs if name.startswith("refs/remotes/") and name.split("/", 3)[3:] == [branch]]
        # A ref the turn left where it found it was pushed nothing, whatever its log says.
        moved = [name for name in tracking if mark.refs.get(name) != refs[name]]
        made, pushes = await asyncio.gather(
            self._made(mark.root, mark.refs, branch, tracking, since, deadline), asyncio.gather(*(self._pushed(mark.root, name, mark.refs.get(name), deadline) for name in moved))
        )
        pushed = any(pushes)
        # [LAW:carrying-cost] no pull request is opened from a remote's default branch, and most pushes are to it: asking
        # the forge after each would spend a request a turn on an answer that is always no.
        opened, asked = await self._opened(mark, branch, deadline) if pushed and branch not in await self._defaults(mark.root, deadline) else ((), UNASKED)
        return (*([Branched(branch, "created branch")] if made else []), *([Pushed(branch)] if pushed else []), *opened), asked

    async def _made(self, root: Path, known: Mapping[str, str], branch: str, tracking: list[str], since: int, deadline: float) -> bool:
        """Whether git logged `branch` created since the mark, and from anything but the remote branch of its own name,
        which is a checkout of a branch that already was. A rename carries the log of the branch it renamed with it."""
        if f"refs/heads/{branch}" in known:
            return False
        logged = await self._git(root, "reflog", "show", "--date=unix", "--format=%gd%x1f%gs", f"refs/heads/{branch}", deadline=deadline)
        if not logged:
            return False
        when, said = _entry(logged.splitlines()[-1])
        source = said.removeprefix("branch: Created from ")
        return when >= since and source != said and source.removeprefix("refs/remotes/") not in {name.removeprefix("refs/remotes/") for name in tracking}

    async def _pushed(self, root: Path, name: str, was: str | None, deadline: float) -> bool:
        """Whether git logged a push to the remote-tracking ref `name` after it named `was`, where the mark found it.

        Walked back from the newest entry to the one that left the ref where the mark found it, and not by time: git
        logs to the second, and a push in the second before the mark is not the turn's. A fetch moves the same ref and
        logs `fetch`, and a push the turn followed with a pull is still in the walk.
        """
        logged = await self._git(root, "reflog", "show", "--format=%H%x1f%gs", name, deadline=deadline)
        if logged is None:
            return False
        for line in logged.splitlines():
            sha, _, said = line.partition("\x1f")
            if sha == was:
                return False
            if said == _PUSHED:
                return True
        return False

    async def _defaults(self, root: Path, deadline: float) -> set[str]:
        """The branch each remote's HEAD follows, where git knows it: a clone does, a repository that added its remote may not."""
        listed = await self._git(root, "for-each-ref", "--format=%(symref:lstrip=3)", "refs/remotes", deadline=deadline)
        return set() if listed is None else {line for line in listed.splitlines() if line}

    async def _opened(self, mark: Mark, branch: str, deadline: float) -> tuple[tuple[PullRequested, ...], Asked]:
        """The pull requests the forge says were opened from `branch` since the mark. Nothing on this machine records one."""
        began = time.monotonic()
        answer = await self._ask(
            f"gh pr list in {mark.root}",
            ("gh", "pr", "list", "--head", branch, "--state", "all", "--json", "number,url,createdAt"),
            cwd=mark.root,
            deadline=min(deadline, began + FORGING),
        )
        took = round(time.monotonic() - began, 3)
        if answer is None:
            return (), Asked("unanswered", took)
        # The forge stamps to the second, so a pull request opened in the second the mark was taken is still since it.
        since = mark.at.replace(microsecond=0)
        try:
            listed: object = json.loads(answer)
        except ValueError as error:
            logger.warning(f"gh said something about the pull requests of {branch} that is not JSON, so none is told: {error}")
            return (), Asked("unanswered", took)
        entries = cast(list[object], listed) if isinstance(listed, list) else [listed]
        return tuple(request for entry in entries if (request := _request(entry, since)) is not None), Asked("answered", took)

    async def _tree(self, root: Path, deadline: float) -> str | None:
        """Everything git would keep, as one tree object, through an index of this daemon's own.

        The repository's own index is copied rather than started from nothing, and only copied: it carries what
        git already knows about every file, so a snapshot re-reads what changed instead of every file there is.
        Measured on a repository of 20,000 files: 0.105 s copied against 1.409 s from nothing, and this is
        taken while a prompt's hook waits on it.
        """
        with tempfile.TemporaryDirectory(prefix="hands-index-") as scratch:
            index = Path(scratch) / "index"
            known = await self._git(root, "rev-parse", "--git-path", "index", deadline=deadline)
            if known is not None:
                try:
                    # With its mtime: git re-reads a file whose stat matches its entry only when the file is as new
                    # as the index, and a copy stamped now hides a same-size edit made in the second of the commit.
                    # Both come from one open file, so an index renamed into place mid-copy cannot lend the old
                    # entries its newer mtime; and only the mtime, which is the one fact git reads off the file.
                    with (Path(known) if Path(known).is_absolute() else root / known).open("rb") as real:
                        index.write_bytes(real.read())
                        written = os.fstat(real.fileno()).st_mtime_ns
                    os.utime(index, ns=(written, written))
                except OSError as error:
                    # Missing before a first commit, or being rewritten as this read it: start from nothing.
                    logger.debug(f"the index of {root} could not be copied, so the snapshot reads every file: {error}")
            env = {"GIT_INDEX_FILE": str(index)}
            if await self._git(root, "add", "-A", env=env, deadline=deadline) is None:
                return None
            return await self._git(root, "write-tree", env=env, deadline=deadline)

    async def _git(self, cwd: Path, *args: str, env: Mapping[str, str] | None = None, deadline: float) -> str | None:
        # `-C` and not the child's working directory, so a session whose directory is gone is git's own quiet refusal
        # and not a process that could not be started.
        argv = ("git", "--no-optional-locks", "-C", str(cwd), *args)
        return await self._ask(f"git {args[0]} in {cwd}", argv, env={"GIT_OPTIONAL_LOCKS": "0", **(env or {})}, deadline=deadline)

    async def _ask(self, what: str, argv: tuple[str, ...], *, env: Mapping[str, str] | None = None, cwd: Path | None = None, deadline: float) -> str | None:
        """What one command said, or None where it could not answer inside what is left of the deadline.

        [LAW:no-silent-failure] a repository that cannot be read leaves the turn told without its delta and
        says why in the log, rather than failing the summary of a turn that mostly happened elsewhere. The
        deadline is the whole reading's, not this command's: five commands that each take a second are as
        late as one that takes five, and it is the total a hook or a listener is waiting through.
        """
        left = deadline - time.monotonic()
        if left <= 0:
            logger.warning(f"there was no time left to run {what}, so the turn is told without it")
            return None
        try:
            ran = await run(*argv, timeout=left, cwd=cwd, env={**os.environ, **(env or {})})
        except TimeoutError:
            logger.error(f"{what} did not answer in {left:.1f}s, so the turn is told without it")
            return None
        except OSError as error:
            logger.error(f"cannot run {what}: {error}")
            return None
        if ran.returncode != 0:
            logger.debug(f"{what}: {ran.err.decode(errors='replace').strip()}")
            return None
        return ran.out.decode(errors="replace").strip()


def _request(entry: object, since: datetime) -> PullRequested | None:
    """One pull request as `gh pr list --json number,url,createdAt` writes it, where it was opened since `since`."""
    match entry:
        case {"number": int() as number, "url": str() as url, "createdAt": str() as created} if (opened := _instant(created)) is not None:
            return PullRequested(number, url, "created") if opened >= since else None
        case _:
            pass
    logger.warning(f"gh named a pull request in a shape hands does not read, so it is not told: {entry!r}")
    return None


def _instant(stamp: str) -> datetime | None:
    """A time the forge wrote, where it says which zone it is in: one that does not cannot be set against the mark."""
    try:
        instant = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return instant if instant.tzinfo is not None else None


def _entry(line: str) -> tuple[int, str]:
    """One line of `git reflog show --date=unix --format=%gd%x1f%gs`: when the ref moved, and what git said moved it."""
    selector, _, said = line.partition("\x1f")
    return int(selector[selector.rindex("@{") + 2 : -1]), said


def _files(numstat: str) -> tuple[Changed, ...]:
    """`git diff --numstat`: added, removed and path, tab separated, with a dash for each count of a binary file.

    Takes what git said and not whether git spoke: a command that could not answer is the caller's to answer
    for, and a helper that quietly turned one into an empty list is what let "nothing changed" be read off a
    reading that never happened [LAW:no-defensive-null-guards].
    """
    if not numstat:
        return ()
    counted: list[Changed] = []
    for line in numstat.splitlines():
        match line.split("\t"):
            case [added, removed, path]:
                counted.append(Changed(path, _number(added), _number(removed)))
            case _:
                logger.debug(f"a line of git's own numstat was not three fields, so it names no file: {line!r}")
    return tuple(counted)


def _number(count: str) -> int | None:
    return int(count) if count.isdigit() else None
