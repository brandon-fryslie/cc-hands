"""Where the daemon and the shims meet on disk."""

import os
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
    def indicator_log(self) -> Path:
        """What the menu-bar indicator prints, as launchd captures it."""
        return self.root / "indicator.log"

    @property
    def memberships(self) -> Path:
        """One file per session, written by its shim: the set of sessions hands is attached to."""
        return self.root / "sessions"

    def membership(self, session: SessionId) -> Path:
        return self.memberships / f"{session}.json"


def default_home() -> Home:
    """HANDS_HOME, or ~/.hands: the hook shim, which Claude Code runs with no arguments of hands', finds it as the CLI does."""
    # [LAW:one-source-of-truth] the one place the default is named; the CLI's --home and the shim both start here.
    return Home(Path(os.environ.get("HANDS_HOME") or Path.home() / ".hands"))
