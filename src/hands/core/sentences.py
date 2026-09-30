"""A sentence for any content-addressed thing, kept until the thing changes: the keys, which sentences are due, and the summariser's page.

A thing is its own text and the things under it. Its key is a digest of the summariser's version, its own text, and
the sentence already said of each thing under it, so a parent is due only once every part is said, and editing one
part changes the key of that part and of everything above it and of nothing else — and stops rising at the first
sentence that comes back word for word as it was. What is not the thing's content —
rank, status, timestamps — is not in the key: a rerank costs no sentence.
"""

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import NewType

Digest = NewType("Digest", str)


@dataclass(frozen=True)
class Thing:
    """Something to be said in one sentence: an id to name it by, its own text, and what sits under it."""

    id: str
    text: str
    parts: tuple["Thing", ...]


@dataclass(frozen=True)
class Due:
    """A thing whose parts are all said and which is not: its key, and everything the summariser is shown of it."""

    id: str
    digest: Digest
    text: str
    parts: tuple[tuple[str, str], ...]  # each part's id and its sentence


@dataclass(frozen=True)
class Reckoning:
    """What a tree of things has said of it so far, and which of the rest can be said next."""

    said: Mapping[str, str]  # a thing's id to its sentence, for every thing whose key has one
    due: tuple[Due, ...]
    waiting: int  # things not due yet, because a part of each is not said


def digest(version: str, text: str, parts: Sequence[tuple[str, str]]) -> Digest:
    """The key of a thing with this text whose parts have these sentences, under this summariser."""
    # Parts in id order, not rank order, so a rerank leaves the key where it was.
    return Digest(hashlib.sha256(json.dumps([version, text, sorted(parts)]).encode()).hexdigest())


def reckon(thing: Thing, version: str, known: Callable[[Digest], str | None]) -> Reckoning:
    """Every sentence the store already holds for this tree, and the things that can be summarised now."""
    said: dict[str, str] = {}
    due: list[Due] = []
    waiting = 0

    def visit(node: Thing) -> str | None:
        nonlocal waiting
        # Every part is visited, said or not, so a tree reckoned once finds everything due at every depth.
        parts = [(part.id, visit(part)) for part in node.parts]
        sentences = [(id, sentence) for id, sentence in parts if sentence is not None]
        if len(sentences) < len(parts):
            waiting += 1
            return None
        key = digest(version, node.text, sentences)
        sentence = known(key)
        if sentence is None:
            due.append(Due(node.id, key, node.text, tuple(sentences)))
        else:
            said[node.id] = sentence
        return sentence

    visit(thing)
    return Reckoning(said, tuple(due), waiting)


def page(batch: Sequence[Due], text_limit: int) -> str:
    """The summariser's message for a batch: each thing's text, cut to `text_limit` characters, and its parts' sentences."""
    return "\n\n".join(_item(due, text_limit) for due in batch)


def _item(due: Due, text_limit: int) -> str:
    text = due.text if len(due.text) <= text_limit else f"{due.text[:text_limit]} [cut]"
    parts = "".join(f"\n- {id}: {sentence}" for id, sentence in due.parts)
    return f"<item id={json.dumps(due.id)}>\n{text}" + (f"\n<parts>{parts}\n</parts>" if parts else "") + "\n</item>"


@dataclass(frozen=True)
class Answered:
    """What a summariser's reply said of a batch: the sentences it gave, by key, and what it got wrong."""

    said: Mapping[Digest, str]
    missing: tuple[str, ...]  # ids in the batch the reply gave no sentence for
    stray: tuple[str, ...]  # lines of the reply that are not one sentence for one id in the batch


_LINE = re.compile(r"(\S+): (.+)")


def answered(reply: str, batch: Sequence[Due]) -> Answered:
    """The reply read line by line as `id: sentence`; an id said twice is said by neither, since nothing picks which is right."""
    by_id = {due.id: due for due in batch}
    heard: dict[str, list[str]] = {}
    stray: list[str] = []
    for line in filter(None, (line.strip() for line in reply.splitlines())):
        match _LINE.fullmatch(line):
            case re.Match() as named if named.group(1) in by_id:
                heard.setdefault(named.group(1), []).append(named.group(2).strip())
            case _:
                stray.append(line)
    said = {by_id[id].digest: sentences[0] for id, sentences in heard.items() if len(sentences) == 1}
    stray.extend(f"{id}: {sentence}" for id, sentences in heard.items() if len(sentences) > 1 for sentence in sentences)
    missing = tuple(due.id for due in batch if due.digest not in said)
    return Answered(said, missing, tuple(stray))
