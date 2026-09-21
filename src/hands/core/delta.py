"""What a turn left behind in the repository it ran in, which its own records need not name.

A formatter, a code generator, or a `sed` inside a shell command changes files that no `Edited` step mentions,
and `git commit` inside one makes commits no step reports. [LAW:one-source-of-truth] what a turn did to a
repository is what the repository says, not what the transcript happened to record of it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Changed:
    """One file a turn left different, and by how much. The counts are None for a file git read as binary."""

    path: str
    added: int | None
    removed: int | None


@dataclass(frozen=True)
class Commit:
    """A commit that was not in the repository when the turn began."""

    sha: str
    subject: str


@dataclass(frozen=True)
class Delta:
    """The files a turn left different, the commits it made, and the patch between where it began and ended.

    Empty is the whole of what it has to say in three different cases — nothing changed, the session works
    outside a repository, and git could not be read — because all three are the same silence to a listener,
    who is told what happened rather than what did not. Which of the three it was is in the log.
    """

    files: tuple[Changed, ...] = ()
    commits: tuple[Commit, ...] = ()
    patch: str = ""

    def __bool__(self) -> bool:
        """Whether there is anything here to tell. The patch alone is never enough: it names no file."""
        return bool(self.files or self.commits)
