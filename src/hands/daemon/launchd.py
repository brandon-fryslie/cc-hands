"""The LaunchAgent that keeps the daemon up: launchd owns when hands runs and restarts it when it dies."""

import plistlib
from pathlib import Path

from hands.sessions.home import Home

LABEL = "hands.daemon"


def agent(python: Path, home: Home, path: str, label: str = LABEL) -> bytes:
    """The property list for `~/Library/LaunchAgents/<label>.plist`."""
    return plistlib.dumps(
        {
            "Label": label,
            "ProgramArguments": [str(python), "-m", "hands.daemon", "--home", str(home.root), "run"],
            # [LAW:single-enforcer] launchd is the one owner of the daemon being up: started at login,
            # started again whenever it exits, at most once per launchd's throttle interval.
            "RunAtLoad": True,
            "KeepAlive": True,
            # Audio is interactive work; launchd would otherwise throttle a background agent's CPU.
            "ProcessType": "Interactive",
            "StandardOutPath": str(home.daemon_log),
            "StandardErrorPath": str(home.daemon_log),
            # launchd starts agents with a bare PATH, without the tmux the daemon types through.
            "EnvironmentVariables": {"PATH": path},
        }
    )
