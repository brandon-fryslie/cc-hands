"""Where the daemon and the shims meet on disk."""

from dataclasses import dataclass
from pathlib import Path

from hands.core.session import SessionId


@dataclass(frozen=True)
class Home:
    root: Path

    @property
    def socket(self) -> Path:
        return self.root / "hands.sock"

    @property
    def status(self) -> Path:
        """The heartbeat, written by the daemon alone."""
        return self.root / "status.json"

    @property
    def audit(self) -> Path:
        """Everything the daemon did, heard, said, and failed at, one JSON line each, written by the daemon alone."""
        return self.root / "audit.jsonl"

    @property
    def daemon_log(self) -> Path:
        """What the daemon process prints, as launchd captures it."""
        return self.root / "daemon.log"

    @property
    def memberships(self) -> Path:
        """One file per session, written by its shim: the set of sessions hands is attached to."""
        return self.root / "sessions"

    def membership(self, session: SessionId) -> Path:
        return self.memberships / f"{session}.json"


def default_home() -> Home:
    return Home(Path.home() / ".hands")
