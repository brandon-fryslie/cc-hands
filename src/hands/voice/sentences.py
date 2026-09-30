"""The summary store as the daemon holds it: what is said of a thing under its current content, and which backlogs are waiting to be said.

The tools read it and never wait on it: a thing with no sentence yet is served by its title, and its backlog is
asked for, to be said off the voice path by `hands.voice.backlog_summaries`.
"""

import asyncio
from collections.abc import Mapping
from pathlib import Path

from hands.core.sentences import Digest, Reckoning, Thing, reckon
from hands.sessions.sentences import Sentences
from hands.voice.sentence_instruction import SENTENCE_INSTRUCTION


class SummaryStore:
    """What is said of a thing now, and the backlogs waiting to be said, for the tools and the summarising task to share."""

    def __init__(self, sentences: Sentences) -> None:
        self._sentences = sentences
        self._wanted: asyncio.Queue[Path] = asyncio.Queue()
        self._pending: set[Path] = set()

    def reckon(self, thing: Thing) -> Reckoning:
        # [LAW:one-source-of-truth] the instruction is the summariser's version: a sentence written under other words
        # has another key, so an edit to the instruction can never be served the old sentences.
        return reckon(thing, SENTENCE_INSTRUCTION, self._sentences.known)

    def want(self, project: Path) -> None:
        """Ask for the backlog in `project` to be said; a project already waiting is not queued twice."""
        if project not in self._pending:
            self._pending.add(project)
            self._wanted.put_nowait(project)

    async def wanted(self) -> Path:
        project = await self._wanted.get()
        # Let go of as it is taken, so a read that lands while it is being said asks for another pass, which sees that read's content.
        self._pending.discard(project)
        return project

    def keep(self, said: Mapping[Digest, str]) -> None:
        self._sentences.keep(said)
