"""What hands is about to name its sessions: the turns waiting to be judged for a name, and each name waiting for its
session's next prompt, which is the one moment a hook can hand Claude Code a title.

Claude Code holds a session's name, and the latest one set wins, whoever set it: hands keeps no copy of it
[LAW:one-source-of-truth], only a name it has decided and Claude Code has not yet been given, and the name Claude Code
held when it was decided, so a name set since, by the user's /rename, is not overwritten by an older decision.
"""

import asyncio
from dataclasses import dataclass

from hands.core.session import Membership, SessionId


@dataclass(frozen=True)
class Finished:
    """A turn a session finished, as its Stop hook told it: what the name is judged from."""

    membership: Membership
    closing: str


@dataclass(frozen=True)
class NameGiven:
    """A name handed to Claude Code in the reply to a session's prompt, which sets the session's title."""

    name: str


@dataclass(frozen=True)
class NameWithheld:
    """A name hands decided and did not give at the session's prompt: its title is no longer the one the name was decided
    against, but `held`, set since by the user's /rename, or none at all."""

    name: str
    against: str | None
    held: str | None


@dataclass(frozen=True)
class NameUnread:
    """A name hands decided and did not give at the session's prompt, since its title could not be read: it may be one
    the user set, and the name is never given over that."""

    name: str
    against: str | None


@dataclass(frozen=True)
class Due:
    """A name hands decided for a session, waiting for its next prompt, and the name Claude Code held as it was decided."""

    name: str
    against: str | None


class Names:
    """The one owner of what hands means to name its sessions, between the hook server and the naming task."""

    def __init__(self) -> None:
        self._finished: asyncio.Queue[Finished] = asyncio.Queue()
        self._due: dict[SessionId, Due] = {}

    def finished(self, turn: Finished) -> None:
        """Have the session's name judged again, now that it has finished this turn."""
        self._finished.put_nowait(turn)

    async def next_finished(self) -> Finished:
        return await self._finished.get()

    def rename(self, session: SessionId, name: str, against: str | None) -> None:
        """Give the session this name at its next prompt, if Claude Code still holds `against` then; a later decision
        replaces one not yet given."""
        self._due[session] = Due(name, against)

    def current(self, session: SessionId, held: str | None) -> str | None:
        """The session's name as it will stand at its next prompt: one hands has decided, or else `held`, the one
        Claude Code holds."""
        due = self._due.get(session)
        return held if due is None else due.name

    def due(self, session: SessionId) -> Due | None:
        """The name to give the session at this prompt, handed over once; None when hands has none waiting for it."""
        return self._due.pop(session, None)
