"""What hands is about to name its sessions: the turns waiting to be judged for a name, and each name waiting for its
session's next prompt, which is the one moment a hook can hand Claude Code a title.

Claude Code holds a session's name, and the latest one set wins, whoever set it: hands keeps no copy of it
[LAW:one-source-of-truth], only a name it has decided and Claude Code has not yet been given.
"""

import asyncio
from dataclasses import dataclass

from hands.core.session import Membership, SessionId


@dataclass(frozen=True)
class Finished:
    """A turn a session finished, as its Stop hook told it: what the name is judged from."""

    membership: Membership
    closing: str


class Names:
    """The one owner of what hands means to name its sessions, between the hook server and the naming task."""

    def __init__(self) -> None:
        self._finished: asyncio.Queue[Finished] = asyncio.Queue()
        self._due: dict[SessionId, str] = {}

    def finished(self, turn: Finished) -> None:
        """Have the session's name judged again, now that it has finished this turn."""
        self._finished.put_nowait(turn)

    async def next_finished(self) -> Finished:
        return await self._finished.get()

    def rename(self, session: SessionId, name: str) -> None:
        """Give the session this name at its next prompt; a later decision replaces one not yet given."""
        self._due[session] = name

    def current(self, session: SessionId, held: str | None) -> str | None:
        """The session's name as it will stand at its next prompt: one hands has decided, or else `held`, the one
        Claude Code holds."""
        return self._due.get(session, held)

    def due(self, session: SessionId) -> str | None:
        """The name to give the session at this prompt, handed over once; None when hands has none waiting for it."""
        return self._due.pop(session, None)
