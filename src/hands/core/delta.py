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
class Committed:
    sha: str
    kind: str  # what Claude Code calls the commit it saw: `committed`, and whatever else it writes


@dataclass(frozen=True)
class Pushed:
    branch: str


@dataclass(frozen=True)
class Branched:
    ref: str
    action: str


@dataclass(frozen=True)
class PullRequested:
    number: int
    url: str
    action: str


# What a command did to the repository, as the result's `gitOperation` records it. One command can do several:
# 70 records in a 900-transcript sample commit and push, and 30 open a pull request and push [LAW:types-are-the-program].
# The same types say what the repository showed a turn did where no record named it, so a change both saw is one value.
GitChange = Committed | Pushed | Branched | PullRequested


@dataclass(frozen=True)
class Delta:
    """The files a turn left different, the commits it made, what else it did to the repository, and the patch
    between where it began and ended.

    Empty is the whole of what it has to say in three different cases — nothing changed, the session works
    outside a repository, and git could not be read — because all three are the same silence to a listener,
    who is told what happened rather than what did not. Which of the three it was is in the log.
    """

    files: tuple[Changed, ...] = ()
    commits: tuple[Commit, ...] = ()
    # A push, a branch, or a pull request changes no file and adds no local commit, and Claude Code writes no
    # `gitOperation` for one made inside a heredoc, a script, or a compound command it does not parse: on this
    # machine, two commands in five that push and nearly every `checkout -b`. So each is read where it leaves a mark:
    # a remote-tracking ref git logs as updated by push, a local branch that was not there, a pull request the
    # forge says was opened since. Never a commit, which `commits` already holds with its subject.
    changes: tuple[Pushed | Branched | PullRequested, ...] = ()
    patch: str = ""

    def __bool__(self) -> bool:
        """Whether there is anything here to tell.

        The patch counts, even alone. It names no file, and a patch with no files is git answering one
        question and not the next — but a repository that demonstrably moved is news, and telling it badly
        beats telling the listener nothing at all [LAW:no-silent-failure].
        """
        return bool(self.files or self.commits or self.changes or self.patch)
