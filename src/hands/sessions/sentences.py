"""The summary store on disk: one table, a sentence for each key, written by the daemon alone and kept across runs.

A key is a digest of everything the sentence was made from (see `hands.core.sentences`), so a row is never updated
and never goes stale: changed content has a new key, and the old row is simply not asked for again.
"""

import sqlite3
from collections.abc import Mapping
from pathlib import Path

from hands.core.sentences import Digest


class Sentences:
    """The store: what has been said of a key, and the one way to say it."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Autocommit: each keep is on disk when it returns, so a sentence made is never made again after a crash.
        self._db = sqlite3.connect(path, isolation_level=None)
        self._db.execute("create table if not exists sentences (digest text primary key, sentence text not null, made_at text not null default current_timestamp)")

    def known(self, digest: Digest) -> str | None:
        row = self._db.execute("select sentence from sentences where digest = ?", (digest,)).fetchone()
        return None if row is None else str(row[0])

    def keep(self, said: Mapping[Digest, str]) -> None:
        # A key said twice keeps its first sentence, so what a reader was served does not change under them.
        self._db.executemany("insert or ignore into sentences (digest, sentence) values (?, ?)", said.items())
