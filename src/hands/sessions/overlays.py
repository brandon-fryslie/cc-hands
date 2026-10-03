"""Each session's overlay, kept in the home, one file per session, so it outlives the daemon.

The daemon reads a session's file each time something it said is routed, and holds no copy [LAW:one-source-of-truth],
so a change is heard from the next thing the session says.
"""

import os
import tempfile
from dataclasses import dataclass

from hands.core.attention import DEFAULT, Overlay
from hands.core.session import SessionId
from hands.sessions.home import Home
from hands.sessions.payload import Rejected


@dataclass(frozen=True)
class Overlays:
    home: Home

    def of(self, session: SessionId) -> Overlay:
        """The session's overlay; the default where it was never set."""
        path = self.home.overlay(session)
        try:
            # Bytes, not text: a file edited by hand can hold anything, and whatever is not an overlay is refused alike.
            written = path.read_bytes().strip()
        except FileNotFoundError:
            return DEFAULT
        match written:
            case b"normal":
                return "normal"
            case b"watched":
                return "watched"
            case _:
                raise Rejected(f"{path} says {written!r}, which is no overlay")

    def set(self, session: SessionId, to: Overlay) -> None:
        path = self.home.overlay(session)
        path.parent.mkdir(parents=True, exist_ok=True)
        # [LAW:no-ambient-temporal-coupling] written beside and renamed into place, so the daemon never reads half a file;
        # each write stages under a name of its own, so two at once for one session cannot rename each other's away.
        descriptor, staging = tempfile.mkstemp(dir=path.parent, prefix=f".{session}.")
        with os.fdopen(descriptor, "w") as staged:
            staged.write(f"{to}\n")
        os.replace(staging, path)
