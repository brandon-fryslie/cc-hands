"""`/hands:restart`: start the running daemon again, so it runs the code, prompt, and brain setup on disk now.

The plugin's skill runs this module under the plugin's own Python, which has no venv, so it imports only the standard
library and hands' data modules:

    hooks/python -m hands.daemon.restart

A change to hands is never taken up live: the brain's prompt is hands' source, given to the brain when it launches; the
brain's setup in its config directory is read by the brain's Claude Code when it launches; and a skill in the plugin is
loaded by the session that uses it (on /reload-plugins), never by the daemon. So asking always restarts. The daemon
stops as `q` stops it and starts again in the same process (hands.daemon.starting.again), and this waits until the
heartbeat says the new run's pipeline is running.
"""

import os
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from hands.sessions import heartbeat
from hands.sessions.home import Home, default_home
from hands.sessions.payload import Rejected

# [LAW:one-source-of-truth] the one signal both ends mean "restart" by: the daemon's handler and this sender.
RESTART_SIGNAL = signal.SIGUSR1
# How long the new run has to say its pipeline is running: its start loads the speech models. Well under the two
# minutes a Claude Code command is given by default, so the deadline is said rather than cut off.
BACK_WITHIN = timedelta(seconds=90)
# How often the heartbeat is looked at while waiting.
LOOK_SECONDS = 0.2


@dataclass(frozen=True)
class Restarted:
    status: heartbeat.Status
    took: timedelta


@dataclass(frozen=True)
class NotRunning:
    """Nothing was asked: hands is not running with its pipeline up, which is the only run whose restart is heard."""

    verdict: heartbeat.Verdict


@dataclass(frozen=True)
class NotBack:
    """hands was asked, and the heartbeat has not said a new run is running: it failed, or is still starting."""

    verdict: heartbeat.Verdict
    waited: timedelta


Outcome = Restarted | NotRunning | NotBack


def restart(home: Home, now: Callable[[], datetime], wait: Callable[[], None], within: timedelta = BACK_WITHIN) -> Outcome:
    """Ask the daemon to restart and wait for the run it starts again to be running."""
    asked = now()
    match heartbeat.look(home.status, asked):
        # [LAW:no-ambient-temporal-coupling] only a running pipeline is asked: by then the run has long since put in
        # its handler, which a run in its first moments has not, and the signal's default would end it.
        case heartbeat.Up(status=before) if before.pipeline == "running":
            pass
        case verdict:
            return NotRunning(verdict)
    try:
        os.kill(before.pid, RESTART_SIGNAL)
    except ProcessLookupError:
        # It ended between the look and the ask: what its heartbeat says now is why it was not restarted.
        return NotRunning(heartbeat.look(home.status, now()))
    while True:
        at = now()
        match heartbeat.look(home.status, at):
            # The new run's heartbeat is told from the old one's by when it started: the pid is the same.
            case heartbeat.Up(status=status) if status.started_at > before.started_at and status.pipeline == "running":
                return Restarted(status, at - asked)
            # Gone, or stopped, or unreadable: nothing more is coming, so it is said now rather than at the deadline.
            case heartbeat.Down() | heartbeat.Stopped() | heartbeat.Unreadable() | heartbeat.NeverRan() as verdict:
                return NotBack(verdict, at - asked)
            # The old run stopping, the new one starting, or a moment unheard between the two.
            case verdict if at - asked >= within:
                return NotBack(verdict, at - asked)
            case _:
                wait()


def said(outcome: Outcome, now: datetime) -> str:
    match outcome:
        case Restarted(status=status, took=took):
            return f"hands restarted: pid {status.pid} is running again after {took.total_seconds():.0f}s, with {heartbeat.live_sessions(status)}."
        case NotRunning(verdict=verdict):
            return f"hands was not restarted: {heartbeat.describe(verdict, now)}."
        case NotBack(verdict=verdict, waited=waited):
            return f"hands was asked to restart {waited.total_seconds():.0f}s ago and is not running again: {heartbeat.describe(verdict, now)}."


def main() -> int:
    try:
        home = default_home()
    except Rejected as error:
        print(f"hands restart: {error}", file=sys.stderr)
        return 2
    outcome = restart(home, lambda: datetime.now(UTC), lambda: time.sleep(LOOK_SECONDS))
    match outcome:
        case Restarted():
            out, code = sys.stdout, 0
        case NotRunning() | NotBack():
            out, code = sys.stderr, 1
    print(said(outcome, datetime.now(UTC)), file=out)
    return code


if __name__ == "__main__":
    sys.exit(main())
