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
    def daemon_log(self) -> Path:
        """What the daemon process prints, as launchd captures it."""
        return self.root / "daemon.log"

    def membership(self, session: SessionId) -> Path:
        return self.root / "sessions" / f"{session}.json"


def default_home() -> Home:
    return Home(Path.home() / ".hands")
