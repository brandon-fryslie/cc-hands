"""A project's lit backlog, read whole from `lit export` and parsed once into the tickets, their tree, and their comments.

`lit export` is lit's versioned data-export primitive, and the one command that hands over descriptions in bulk.
It is read fresh at every call and kept nowhere [LAW:one-source-of-truth]: lit is where the backlog is.
"""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hands.core.sentences import Thing
from hands.sessions.child import run
from hands.sessions.payload import Payload, Rejected

# The largest backlog on this machine (links, 8.6 MB of export) takes 5 s; past this lit is stuck, not slow.
EXPORT_TIMEOUT_SECONDS = 30.0

# The export shape this parser was written against; another is refused rather than half read.
EXPORT_VERSION = 2

Status = Literal["open", "in_progress", "closed"]

# The id of the whole backlog, as a thing to be said, beside the ids lit gives its tickets.
BACKLOG = "backlog"


class Unread(Exception):
    """lit could not hand over a backlog in this directory: it failed, or refused, and the message says why."""


@dataclass(frozen=True)
class Untracked:
    """A directory lit was never set up in, or that is in no git repository: it has no backlog, which is an answer, not
    a failure."""

    project: Path


# The exit lit refuses with, and the two reasons it gives for having no workspace here: ErrWorkspaceNotInitialized
# (internal/store/workspace_initialized.go) where `lit init` never ran, and its outside_git_workspace refusal outside
# any git repository. Its reason is the line after the prefix; the line after that is its remediation.
REFUSED = 3
NO_WORKSPACE = (
    f"error (code={REFUSED}): repository not initialized with lit",
    f"error (code={REFUSED}): links requires running inside a git repository",
)
REMEDIATION = "remediation: "


def _no_workspace(returncode: int, err: str) -> bool:
    """Whether lit refused only because it has no workspace here: the whole of what it said is that reason and its
    remediation, so a failure joined onto the reason is never mistaken for it."""
    reason, *rest = err.strip().splitlines() or [""]
    return returncode == REFUSED and reason.startswith(NO_WORKSPACE) and all(line.startswith(REMEDIATION) for line in rest)


@dataclass(frozen=True)
class Ticket:
    id: str
    title: str
    description: str
    kind: str
    # None for an epic: lit exports no status for one, since an epic's state is its children's.
    status: Status | None

    @property
    def text(self) -> str:
        """What the ticket says, and so what its sentence is made of and keyed by."""
        return f"{self.title}\n\n{self.description}"


@dataclass(frozen=True)
class Comment:
    by: str
    at: str
    body: str


@dataclass(frozen=True)
class Backlog:
    """Every ticket lit holds for one project, in rank order, with each one's children in rank order."""

    tickets: Mapping[str, Ticket]
    children: Mapping[str, tuple[str, ...]]
    parent: Mapping[str, str]
    comments: Mapping[str, tuple[Comment, ...]]

    def done(self, id: str) -> bool:
        """Whether a ticket is finished: closed, or for an epic, every child done."""
        match self.tickets[id].status:
            case None:
                children = self.children.get(id, ())
                return bool(children) and all(self.done(child) for child in children)
            case status:
                return status == "closed"

    def open_children(self, id: str) -> tuple[str, ...]:
        return tuple(child for child in self.children.get(id, ()) if not self.done(child))

    def roots(self) -> tuple[str, ...]:
        """What the backlog is made of: every unfinished ticket under no unfinished parent, in rank order.

        Epics and loose tickets, and a follow-up filed under a ticket already closed, which stands on its own rather
        than vanishing with the parent the tree no longer shows.
        """
        return tuple(id for id in self.tickets if not self.done(id) and (id not in self.parent or self.done(self.parent[id])))

    def thing(self) -> Thing:
        """The backlog as a thing to be said: its unfinished roots, each over its unfinished children."""

        def of(id: str) -> Thing:
            return Thing(id, self.tickets[id].text, tuple(of(child) for child in self.open_children(id)))

        # The backlog has no text of its own: it is what it holds.
        return Thing(BACKLOG, "", tuple(of(id) for id in self.roots()))


async def read_backlog(project: Path) -> Backlog | Untracked:
    """The backlog lit holds for `project`, or that lit has no workspace there; raises Unread when lit cannot say, and
    Rejected when what it said does not parse."""
    try:
        ran = await run("lit", "export", timeout=EXPORT_TIMEOUT_SECONDS, cwd=project)
    except TimeoutError:
        raise Unread(f"lit export in {project} did not answer in {EXPORT_TIMEOUT_SECONDS:.0f}s") from None
    except OSError as error:
        raise Unread(f"cannot run lit in {project}: {error}") from error
    if ran.returncode != 0:
        err = ran.err.decode(errors="replace")
        # [LAW:parse-dont-validate] exception: lit exits 3 for this and for every other refusal alike, and prints its
        # reason only as English, so the sentence is what tells them apart until lit says it as data (links-error-output-ckk7).
        if _no_workspace(ran.returncode, err):
            return Untracked(project)
        raise Unread(f"lit export in {project} exited {ran.returncode}: {err.strip()[-300:]}")
    # Off the loop: the largest export takes tens of milliseconds to parse, which the voice pipeline would hear.
    return await asyncio.to_thread(parse_export, ran.out)


def parse_export(raw: bytes) -> Backlog:
    """`lit export`'s JSON, parsed once [LAW:parse-dont-validate]: each ticket, its place in the tree, and what was said on it."""
    export = Payload.parse(raw)
    if (version := export.integer("version")) != EXPORT_VERSION:
        raise Rejected(f"lit export is version {version}, and hands reads version {EXPORT_VERSION}")
    issues = sorted((Payload.of(entry, "an issue") for entry in export.items("issues")), key=lambda issue: issue.text("rank"))
    tickets = {issue.text("id"): _ticket(issue) for issue in issues}
    parent: dict[str, str] = {}
    for relation in (Payload.of(entry, "a relation") for entry in export.items("relations")):
        # A parent the export leaves out (deleted, or archived) is no parent here: its child stands as a root rather than vanishing.
        if relation.text("type") == "parent-child" and relation.text("src_id") in tickets and relation.text("dst_id") in tickets:
            parent[relation.text("src_id")] = relation.text("dst_id")
    children: dict[str, tuple[str, ...]] = {}
    for id in tickets:  # rank order, so each parent's children are too
        if id in parent:
            children[parent[id]] = (*children.get(parent[id], ()), id)
    comments: dict[str, tuple[Comment, ...]] = {}
    for comment in sorted((Payload.of(entry, "a comment") for entry in export.items("comments")), key=lambda comment: comment.text("created_at")):
        issue = comment.text("issue_id")
        comments[issue] = (*comments.get(issue, ()), Comment(comment.text("created_by"), comment.text("created_at"), comment.text("body")))
    return Backlog(tickets, children, parent, comments)


def _ticket(issue: Payload) -> Ticket:
    return Ticket(issue.text("id"), issue.text("title"), issue.text("description"), issue.text("issue_type"), _status(issue.optional_text("status")))


def _status(status: str | None) -> Status | None:
    match status:
        case None | "open" | "in_progress" | "closed":
            return status
        case other:
            raise Rejected(f"a ticket's status should be open, in_progress, or closed, got {other!r}")
