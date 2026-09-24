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
import os
import shutil
import tempfile
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from loguru import logger

from hands.core.delta import Changed, Commit, Delta
from hands.core.session import SessionId
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


# A mark being taken, or taken: None where it could not be.
Marking = asyncio.Future[Mark | None]


class Changes(Protocol):
    """What the registry and the narrator need of a repository reader, so either can be given one that reads none."""

    async def snapshot(self, session: SessionId, cwd: Path) -> None: ...

    async def compare(self, session: SessionId) -> None: ...

    async def taken(self, session: SessionId) -> Delta: ...


class NoChanges:
    """Reads no repository, so every turn is told by its steps alone. What a daemon with no git gets."""

    async def snapshot(self, session: SessionId, cwd: Path) -> None: ...

    async def compare(self, session: SessionId) -> None: ...

    async def taken(self, session: SessionId) -> Delta:
        return Delta()


class Deltas:
    """Where each session's repository stood when its turn began, so what the turn changed can be read at its end."""

    def __init__(self, marking: float = MARKING, reading: float = READING, patience: float = PATIENCE) -> None:
        self._marking = marking
        self._reading = reading
        self._patience = patience
        # The mark each session's last snapshot took, or is taking, or where its last turn compared ended: held from the moment it starts, so a
        # reading that needs it waits for it rather than mistaking one not taken yet for one that could not be.
        # Bounded by the sessions the registry itself keeps, which holds every session it has heard of.
        self._marks: dict[SessionId, Marking] = {}
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

    async def compare(self, session: SessionId) -> None:
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
        start = self._marks.pop(session, None)
        # Where the repository stands at the end of this turn is where the next one begins, whatever opens it: a
        # message queued behind this turn runs as soon as its Stop hook returns, and no prompt of its own marks it
        # (2.1.282). A prompt that does comes later, and marks again [LAW:one-source-of-truth].
        following: Marking = asyncio.get_running_loop().create_future()
        self._marks[session] = following
        if len(held) >= HELD:
            # Dropped from the back, never the front, and that is the whole of what keeps the two sides in
            # step. A telling takes the oldest reading, and tellings are not dropped alongside readings —
            # they queue unbounded — so evicting the front would hand every telling after it the delta of
            # the turn after its own, which is the one thing `taken` exists to prevent. Dropped from the
            # back, every turn that has a delta has its own [LAW:no-ambient-temporal-coupling].
            logger.warning(f"{HELD} deltas of session {session} are already waiting to be told, so this turn is told without one")
            following.set_result(None)
            return
        pending: asyncio.Future[Delta] = asyncio.get_running_loop().create_future()
        held.append(pending)
        if start is None:
            pending.set_result(Delta())
            following.set_result(None)
            return
        task = asyncio.create_task(self._read(pending, following, start), name=f"what a turn of session {session} changed")
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

    async def _read(self, pending: asyncio.Future[Delta], following: Marking, start: Marking) -> None:
        """[LAW:no-silent-failure] whatever happens here, whoever is waiting is answered rather than left: the telling
        with the delta, and the turn after with where this one ended."""
        delta, end = Delta(), None
        try:
            mark = await start
            if mark is not None:
                delta, end = await self._between(mark, time.monotonic() + self._reading)
        except Exception as error:
            logger.error(f"what a turn changed could not be read: {type(error).__name__}: {error}")
        if not pending.done():
            pending.set_result(delta)
        if not following.done():
            following.set_result(end)

    async def _mark(self, cwd: Path, deadline: float) -> Mark | None:
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
        return Mark(Path(root), head, tree)

    async def _unborn(self, root: Path, deadline: float) -> bool:
        """Whether a repository that would not say where it stands has nowhere to stand yet.

        HEAD naming a branch that no commit is on is what every repository looks like between `git init` and
        its first commit, and asking for it is a positive answer where the absence of one is not: a HEAD
        being rewritten by a checkout in the next terminal along, a ref that cannot be read, and a deadline
        with nothing left on it all leave the same silence, and none of them means there is no commit.
        """
        return await self._git(root, "symbolic-ref", "--quiet", "HEAD", deadline=deadline) is not None

    async def _between(self, mark: Mark, deadline: float) -> tuple[Delta, Mark | None]:
        """What changed from the mark to where the repository stands now, and where that is: None where git could not
        say, so a repository with no commit yet leaves no end to begin the next turn from."""
        # Read before the tree, because they are two fast commands where the tree is the slow one: a turn
        # whose commit is the one thing worth saying about it should not lose that because `git add -A` took
        # longer than a reading is given, or failed for a reason that has nothing to do with the commit.
        head = await self._git(mark.root, "rev-parse", "HEAD", deadline=deadline)
        commits = await self._commits(mark, head, deadline)
        tree = await self._tree(mark.root, deadline)
        end = None if head is None or tree is None else Mark(mark.root, head, tree)
        return await self._changed(mark, commits, tree, deadline), end

    async def _changed(self, mark: Mark, commits: tuple[Commit, ...], tree: str | None, deadline: float) -> Delta:
        """What the tree the turn left says it changed, with the commits it made."""
        if tree is None or tree == mark.tree:
            # Unreadable, or the working tree came back to where it started — which a commit and nothing else does.
            return Delta(commits=commits)
        numstat = await self._git(mark.root, "diff", "--numstat", mark.tree, tree, deadline=deadline)
        if numstat is None:
            # [LAW:no-silent-failure] git not answering is not git saying nothing changed. Counted as the
            # second, the numbers that decide whether the patch can be kept are all zero and the bound they
            # are here to enforce is not enforced at all — on the diff most likely to have been what stopped
            # the counting. What the turn committed is known either way, so that much is still told.
            logger.info(f"what a turn changed in {mark.root} could not be counted, so its patch is not read")
            return Delta(commits=commits)
        files = _files(numstat)
        counted = sum((file.added or 0) + (file.removed or 0) for file in files)
        if counted > MOST_LINES:
            # git hands back a whole diff before a character of it is cut, so a turn that wrote a million-line
            # file inside the repository would have all of it here at once. The counts cost one line a file and
            # are already in hand, so they are what says no — and what is left, the files and their counts, is
            # all of a diff that size that would have survived the summariser's budget anyway.
            logger.info(f"a turn changed {counted} lines in {mark.root}, too many to keep the patch of, so its files are told instead")
            return Delta(files, commits)
        patch = await self._git(mark.root, "diff", mark.tree, tree, deadline=deadline)
        return Delta(files, commits, "" if patch is None else patch[:MOST])

    async def _commits(self, mark: Mark, head: str | None, deadline: float) -> tuple[Commit, ...]:
        if head is None or head == mark.head:
            return ()
        # What is reachable from where the turn ended and not from where it began, which is what the turn
        # added however it got there — a merge, a rebase, or an amend that rewrote the commit before it.
        listed = await self._git(
            mark.root, "log", f"--max-count={MOST_COMMITS}", "--format=%h%x1f%s", f"{mark.head}..{head}" if mark.head else head, deadline=deadline
        )
        return () if not listed else tuple(Commit(*line.split("\x1f", 1)) for line in listed.splitlines() if "\x1f" in line)

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
                    shutil.copyfile(Path(known) if Path(known).is_absolute() else root / known, index)
                except OSError as error:
                    # Missing before a first commit, or being rewritten as this read it: start from nothing.
                    logger.debug(f"the index of {root} could not be copied, so the snapshot reads every file: {error}")
            env = {"GIT_INDEX_FILE": str(index)}
            if await self._git(root, "add", "-A", env=env, deadline=deadline) is None:
                return None
            return await self._git(root, "write-tree", env=env, deadline=deadline)

    async def _git(self, cwd: Path, *args: str, env: Mapping[str, str] | None = None, deadline: float) -> str | None:
        """What one git command said, or None where git could not answer inside what is left of the deadline.

        [LAW:no-silent-failure] a repository that cannot be read leaves the turn told without its delta and
        says why in the log, rather than failing the summary of a turn that mostly happened elsewhere. The
        deadline is the whole reading's, not this command's: five commands that each take a second are as
        late as one that takes five, and it is the total a hook or a listener is waiting through.
        """
        left = deadline - time.monotonic()
        if left <= 0:
            logger.warning(f"there was no time left to run git {args[0]} in {cwd}, so the turn is told without it")
            return None
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                "--no-optional-locks",
                "-C",
                str(cwd),
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", **(env or {})},
            )
        except OSError as error:
            logger.error(f"cannot run git in {cwd}: {error}")
            return None
        try:
            out, err = await asyncio.wait_for(process.communicate(), left)
        except TimeoutError:
            process.kill()
            # Reaped here rather than left to the loop, which would report it as a subprocess still running.
            await process.wait()
            logger.error(f"git {args[0]} in {cwd} did not answer in {left:.1f}s, so the turn is told without it")
            return None
        if process.returncode != 0:
            logger.debug(f"git {args[0]} in {cwd}: {err.decode(errors='replace').strip()}")
            return None
        return out.decode(errors="replace").strip()


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
