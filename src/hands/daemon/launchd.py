"""The LaunchAgents that keep hands' processes up: launchd owns when each runs and restarts it when it dies."""

import plistlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from hands.sessions.home import Home


@dataclass(frozen=True)
class Agent:
    label: str
    command: str  # the `hands` subcommand the agent runs
    log: Callable[[Home], Path]  # where launchd puts what the process prints


# [LAW:one-type-per-behavior] the daemon and the indicator are kept up the same way; only these values differ.
DAEMON = Agent("hands.daemon", "run", lambda home: home.daemon_log)
# Its own agent, never a child of the daemon: the surface that says the daemon died cannot be one that dies with it.
INDICATOR = Agent("hands.indicator", "indicator", lambda home: home.indicator_log)
AGENTS: Mapping[str, Agent] = {"daemon": DAEMON, "indicator": INDICATOR}


def agent(which: Agent, python: Path, home: Home) -> bytes:
    """The property list for `~/Library/LaunchAgents/<label>.plist`."""
    log = str(which.log(home))
    return plistlib.dumps(
        {
            "Label": which.label,
            "ProgramArguments": [str(python), "-m", "hands.daemon", "--home", str(home.root), which.command],
            # [LAW:single-enforcer] launchd is the one owner of each process being up: started at login,
            # started again whenever it exits, at most once per launchd's throttle interval.
            "RunAtLoad": True,
            "KeepAlive": True,
            # Audio and the menu bar are interactive work; launchd would otherwise throttle a background agent's CPU.
            "ProcessType": "Interactive",
            "StandardOutPath": log,
            "StandardErrorPath": log,
        }
    )
