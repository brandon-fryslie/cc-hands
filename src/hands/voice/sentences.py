"""The summary store as the daemon holds it: what is said of a thing under its current content, and what is waiting to be said.

The tools read it and never wait on it: a thing with no sentence yet is served by its title or its request, and is
asked for, to be said off the voice path by `hands.voice.summarising`.
"""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hands.core.sentences import Digest, Due, Reckoning, Thing, reckon
from hands.core.session import SessionId
from hands.sessions.sentences import Sentences
from hands.voice.sentence_instruction import SENTENCE_INSTRUCTION


@dataclass(frozen=True)
class Backlog:
    """The backlog in a project, to be read fresh and every sentence it is missing said."""

    project: Path


@dataclass(frozen=True)
class Turns:
    """Finished turns of a session that have no sentence: each already keyed and rendered by the reading that found them."""

    session: SessionId
    due: tuple[Due, ...]


# [LAW:types-are-the-program] a backlog is read when its pass starts, since it changes; a finished turn never does, so
# it is handed over as the reading found it.
Wanted = Backlog | Turns


class SummaryStore:
    """What is said of a thing now, and what is waiting to be said, for the tools and the summarising task to share."""

    def __init__(self, sentences: Sentences) -> None:
        self._sentences = sentences
        self._wanted: asyncio.Queue[Wanted] = asyncio.Queue()
        # What is queued and not yet taken: a project, or a turn by its key.
        self._pending: set[Path | Digest] = set()

    def reckon(self, thing: Thing) -> Reckoning:
        # [LAW:one-source-of-truth] the instruction is the summariser's version: a sentence written under other words
        # has another key, so an edit to the instruction can never be served the old sentences.
        return reckon(thing, SENTENCE_INSTRUCTION, self._sentences.known)

    def known(self, digest: Digest) -> str | None:
        return self._sentences.known(digest)

    def want(self, project: Path) -> None:
        """Ask for the backlog in `project` to be said; a project already waiting is not queued twice."""
        if project not in self._pending:
            self._pending.add(project)
            self._wanted.put_nowait(Backlog(project))

    def want_turns(self, session: SessionId, due: Sequence[Due]) -> None:
        """Ask for these turns to be said; a turn already waiting is not queued twice."""
        fresh = tuple(turn for turn in due if turn.digest not in self._pending)
        if fresh:
            self._pending.update(turn.digest for turn in fresh)
            self._wanted.put_nowait(Turns(session, fresh))

    async def wanted(self) -> Wanted:
        wanted = await self._wanted.get()
        # Let go of as it is taken, so a read that lands while it is being said asks for another pass, which sees that read's content.
        match wanted:
            case Backlog(project=project):
                self._pending.discard(project)
            case Turns(due=due):
                self._pending.difference_update(turn.digest for turn in due)
        return wanted

    def keep(self, said: Mapping[Digest, str]) -> None:
        self._sentences.keep(said)
