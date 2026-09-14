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

    def membership(self, session: SessionId) -> Path:
        return self.root / "sessions" / f"{session}.json"


def default_home() -> Home:
    return Home(Path.home() / ".hands")
