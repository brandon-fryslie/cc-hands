"""Where the daemon and the shims meet on disk."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hands.core.session import SessionId
from hands.sessions.payload import Rejected


@dataclass(frozen=True)
class Home:
    root: Path

    @property
    def socket(self) -> Path:
        return self.root / "hands.sock"

    @property
    def wire(self) -> Path:
        """Where each wrapped session's fritter sends a copy of every exchange the session has with the API."""
        return self.root / "wire.sock"

    @property
    def status(self) -> Path:
        """The heartbeat, written by the daemon alone."""
        return self.root / "status.json"

    @property
    def audit(self) -> Path:
        """Everything the daemon did, heard, said, and failed at, one JSON line each, written by the daemon alone: a
        segmented log, the directory of its segments."""
        return self.root / "audit"

    @property
    def attention(self) -> Path:
        """What hands says unprompted, written by the attention tool and `/hands:attention` alone."""
        return self.root / "attention.json"

    @property
    def config(self) -> Path:
        """The settings, read by the daemon once when it starts (hands.daemon.config), and written by the user alone."""
        return self.root / "config.toml"

    @property
    def voice(self) -> Path:
        """The voice hands speaks in, written by the daemon alone, when the user chooses one."""
        return self.root / "voice"

    @property
    def focus(self) -> Path:
        """The session the user is talking to when they name none, written by the daemon alone; absent for none."""
        return self.root / "focus"

    @property
    def sentences(self) -> Path:
        """The summary store: a sentence for each content-addressed thing, written by the daemon alone."""
        return self.root / "sentences.db"

    @property
    def memberships(self) -> Path:
        """One file per session, written by its shim: the set of sessions hands is attached to."""
        return self.root / "sessions"

    @property
    def bin(self) -> Path:
        """fritter, and the claude that runs every interactive session under it, written by `hands install-fritter`."""
        return self.root / "bin"

    @property
    def fritter(self) -> Path:
        """The fritter every interactive session runs under, the brain's included, built by `hands install-fritter`."""
        return self.bin / "fritter"

    @property
    def shim(self) -> Path:
        """The claude that runs every interactive session under fritter."""
        return self.bin / "claude"

    @property
    def brain(self) -> Path:
        """The brain's CLAUDE_CONFIG_DIR: its login, settings, and skills, and under it the empty directory it runs in."""
        return self.root / "brain"

    def membership(self, session: SessionId) -> Path:
        return self.memberships / f"{session}.json"

    @property
    def overlays(self) -> Path:
        """One file per session saying how its news reaches the user, written by the daemon alone."""
        return self.root / "overlays"

    def overlay(self, session: SessionId) -> Path:
        return self.overlays / session


def default_home(environment: Mapping[str, str]) -> Home:
    """HANDS_HOME, or ~/.hands: the hook shim, which Claude Code runs with no arguments of hands', finds it as the CLI does."""
    # [LAW:one-source-of-truth] the one place the default is named; the CLI's --home and the shim both start here.
    # HANDS_HOME is no setting but where the settings are: the one thing a process started by Claude Code, with no
    # arguments of hands', can be told.
    root = Path(environment.get("HANDS_HOME") or Path.home() / ".hands").expanduser()
    # [LAW:parse-dont-validate] a hook runs in its session's directory, so a relative home would be a different
    # home in every project; it is refused here, where every reader of the home gets it.
    if not root.is_absolute():
        raise Rejected(f"HANDS_HOME must be an absolute path, got {str(root)!r}")
    return Home(root)
