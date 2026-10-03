"""A daemon with no voice: hands' own launch, restart, session sweep, and heartbeat, around a run that only lists sessions.

    python standin_daemon.py <home>

It stands in for `hands run`, which needs a microphone, the talk key's grant, and the speech models, so that a test
can restart a running daemon through the plugin, or by editing its settings, and see what the run after it knows.
"""

import asyncio
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from hands.daemon.cli import launch
from hands.daemon.config import edited, load
from hands.daemon.starting import Ended, again, keep_beating
from hands.sessions import heartbeat
from hands.sessions.audit import AuditLog
from hands.sessions.home import Home
from hands.sessions.liveness import sweep
from hands.sessions.registry import Sessions

PERIOD = timedelta(milliseconds=100)


def main() -> None:
    home = Home(Path(sys.argv[1]))
    heart = heartbeat.Heart(home.status, os.getpid(), datetime.now(UTC), PERIOD)
    heart.beat("starting", None, 0, listening=False, deaf=False)
    audit = AuditLog(home.audit, clock=lambda: datetime.now(UTC))
    settings = load(home)

    async def run(quit_event: asyncio.Event) -> Ended:
        sessions = Sessions(permission_deadline=60.0, clock=time.monotonic, record=audit.record)
        # As hands' run does before its models load: the sessions already running are listed from the home.
        await sweep(home, sessions, frozenset())
        beating = asyncio.create_task(keep_beating(lambda: heart.beat("running", None, sessions.live_count(), listening=False, deaf=False), PERIOD.total_seconds()))
        await quit_event.wait()
        beating.cancel()
        return Ended(None, sessions.live_count())

    match asyncio.run(launch(lambda: run, heart, lambda: edited(home, audit.record, lambda _settings: None, settings, period=0.05), audit.record)):
        case "quit":
            return
        case "restart":
            again([sys.executable, __file__, *sys.argv[1:]], audit.record)


if __name__ == "__main__":
    main()
