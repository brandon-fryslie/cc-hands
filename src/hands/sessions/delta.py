"""What a turn changed in the repository a session works in, read from git without disturbing it.

Every command here reads: nothing moves a ref, stages anything, or touches the working tree. The one mark it
leaves is in the object database, where writing a tree costs a few unreferenced objects that git's own
housekeeping collects — the same price `git stash create` charges, and the reason this does not use it is that
a stash holds no untracked file, and the new file a code generator wrote is exactly what a turn must be able
to name [LAW:effects-at-boundaries].
"""

import asyncio
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from loguru import logger

from hands.core.delta import Changed, Commit, Delta
from hands.core.session import SessionId

# How long one git command is given before the turn is told without its delta. Generous: the snapshot is taken
# while a prompt's hook waits, and a repository slow enough to pass this is one something else is wrong with.
PATIENCE = 20.0

# The most of a patch kept in memory until it is told. The summariser's budget cuts it again and much smaller;
# this only stops a formatter's rewrite of a whole repository from sitting here until someone asks.
MOST = 40_000


@dataclass(frozen=True)
class Mark:
    """Where a repository stood: the commit it was on, and a tree of everything in it git would keep."""

    root: Path
    head: str | None  # None before a repository's first commit, where there is no commit to be on
    tree: str


class Changes(Protocol):
    """What the registry and the narrator need of a repository reader, so either can be given one that reads none."""

    async def snapshot(self, session: SessionId, cwd: Path) -> None: ...

    async def compare(self, session: SessionId) -> None: ...

    def taken(self, session: SessionId) -> Delta: ...


class NoChanges:
    """Reads no repository, so every turn is told by its steps alone. What a daemon with no git gets."""

    async def snapshot(self, session: SessionId, cwd: Path) -> None: ...

    async def compare(self, session: SessionId) -> None: ...

    def taken(self, session: SessionId) -> Delta:
        return Delta()


class Deltas:
    """Where each session's repository stood when its turn began, so what the turn changed can be read at its end."""

    def __init__(self, patience: float = PATIENCE) -> None:
        self._patience = patience
        self._marks: dict[SessionId, Mark] = {}
        self._taken: dict[SessionId, Delta] = {}

    async def snapshot(self, session: SessionId, cwd: Path) -> None:
        """Mark where the repository stands, before the turn has had the chance to change anything in it."""
        self._marks.pop(session, None)
        mark = await self._mark(cwd)
        if mark is None:
            # Not a repository, or one that cannot be read: the turn is told by its steps, which is most of it.
            logger.debug(f"nothing to compare a turn of session {session} against in {cwd}")
            return
        self._marks[session] = mark

    async def compare(self, session: SessionId) -> None:
        """Read what the turn changed against the mark its start took, and hold it until it is told.

        Read when the turn stops, not when its summary is made: summaries are made one at a time and take
        seconds, by which time the session may have begun another turn and changed more, and the delta would
        then hold two turns' work and be told as one [LAW:no-ambient-temporal-coupling].
        """
        mark = self._marks.pop(session, None)
        if mark is None:
            return
        delta = await self._between(mark)
        if delta:
            self._taken[session] = delta

    def taken(self, session: SessionId) -> Delta:
        """What this session's last turn changed, and nothing twice: a delta told is a delta spent."""
        return self._taken.pop(session, Delta())

    async def _mark(self, cwd: Path) -> Mark | None:
        root = await self._git(cwd, "rev-parse", "--show-toplevel")
        if not root:
            return None
        tree = await self._tree(Path(root))
        return None if tree is None else Mark(Path(root), await self._git(Path(root), "rev-parse", "HEAD"), tree)

    async def _between(self, mark: Mark) -> Delta:
        tree = await self._tree(mark.root)
        if tree is None:
            return Delta()
        commits = await self._commits(mark)
        if tree == mark.tree:
            # The working tree came back to where it started, which a commit and nothing else does.
            return Delta(commits=commits)
        return Delta(_files(await self._git(mark.root, "diff", "--numstat", mark.tree, tree)), commits, await self._patch(mark.root, mark.tree, tree))

    async def _commits(self, mark: Mark) -> tuple[Commit, ...]:
        head = await self._git(mark.root, "rev-parse", "HEAD")
        if head is None or head == mark.head:
            return ()
        # What is reachable from where the turn ended and not from where it began, which is what the turn
        # added however it got there — a merge, a rebase, or an amend that rewrote the commit before it.
        listed = await self._git(mark.root, "log", "--format=%h%x1f%s", f"{mark.head}..{head}" if mark.head else head)
        return () if not listed else tuple(Commit(*line.split("\x1f", 1)) for line in listed.splitlines() if "\x1f" in line)

    async def _patch(self, root: Path, before: str, after: str) -> str:
        patch = await self._git(root, "diff", before, after)
        return "" if patch is None else patch[:MOST]

    async def _tree(self, root: Path) -> str | None:
        """Everything git would keep, as one tree object, through an index of this daemon's own.

        The repository's own index is copied rather than started from nothing, and only copied: it carries what
        git already knows about every file, so a snapshot re-reads what changed instead of every file there is.
        Measured on a repository of 20,000 files: 0.105 s copied against 1.409 s from nothing, and this is
        taken while a prompt's hook waits on it.
        """
        with tempfile.TemporaryDirectory(prefix="hands-index-") as scratch:
            index = Path(scratch) / "index"
            known = await self._git(root, "rev-parse", "--git-path", "index")
            if known is not None:
                try:
                    shutil.copyfile(Path(known) if Path(known).is_absolute() else root / known, index)
                except OSError as error:
                    # Missing before a first commit, or being rewritten as this read it: start from nothing.
                    logger.debug(f"the index of {root} could not be copied, so the snapshot reads every file: {error}")
            env = {"GIT_INDEX_FILE": str(index)}
            if await self._git(root, "add", "-A", env=env) is None:
                return None
            return await self._git(root, "write-tree", env=env)

    async def _git(self, cwd: Path, *args: str, env: Mapping[str, str] | None = None) -> str | None:
        """What one git command said, or None where git could not answer.

        [LAW:no-silent-failure] a repository that cannot be read leaves the turn told without its delta and
        says why in the log, rather than failing the summary of a turn that mostly happened elsewhere.
        """
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
            out, err = await asyncio.wait_for(process.communicate(), self._patience)
        except TimeoutError:
            process.kill()
            logger.error(f"git {args[0]} in {cwd} did not answer in {self._patience}s, so the turn is told without it")
            return None
        if process.returncode != 0:
            logger.debug(f"git {args[0]} in {cwd}: {err.decode(errors='replace').strip()}")
            return None
        return out.decode(errors="replace").strip()


def _files(numstat: str | None) -> tuple[Changed, ...]:
    """`git diff --numstat`: added, removed and path, tab separated, with a dash for each count of a binary file."""
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
