"""Which sessions are running, from their membership files and the OS: silence is never evidence."""

import asyncio
import time
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from loguru import logger

from hands.core.events import Attached, Died, MovedOn, Observed
from hands.core.session import Membership, SessionId
from hands.sessions.home import Home
from hands.sessions.membership import read_membership, remove_ended_membership
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions

# ps reports elapsed time in whole seconds, and it is read a moment after the clock is.
START_SLACK_SECONDS = 2.0


@dataclass(frozen=True)
class Recorded:
    membership: Membership
    written_at: float  # wall-clock seconds: the file's modification time


async def sweep(home: Home, sessions: Sessions) -> None:
    """Apply what the files and the process table say now: a running session is attached, an ended one ends and its file goes."""
    # [LAW:no-ambient-temporal-coupling] taken before the directory is read: a session joins only after its shim
    # has written its file, so one that joins while the sweep runs is never taken for a session whose file is gone.
    listed = sessions.live_members()
    records = recorded(home)
    started = await process_starts({record.membership.pid for record in records} | {membership.pid for membership in listed})
    for seen in observations(listed, records, started):
        await sessions.apply(seen)
        match seen:
            case Died(membership=membership) | MovedOn(membership=membership):
                remove_ended_membership(home, membership)
            case Attached():
                pass


async def keep_sweeping(home: Home, sessions: Sessions, period: float) -> None:
    """Sweep now and once a period after, until cancelled. The period is how late a closed pane is heard."""
    while True:
        await sweep(home, sessions)
        await asyncio.sleep(period)


def observations(listed: Collection[Membership], records: Collection[Recorded], started: Mapping[int, float]) -> list[Observed]:
    """What each file, and each listed session whose file is gone, says about its session."""
    # One process holds one session, so of the files naming a pid the newest is its session and the rest are over.
    newest = {record.membership.pid: record for record in sorted(records, key=lambda record: record.written_at)}
    on_file = {record.membership.id for record in records}
    return [
        *(_observed(record, newest[record.membership.pid] is record, started) for record in records),
        # A listed session whose file is gone ended without its end hook reaching the daemon: the shim removes the file first.
        *(MovedOn(membership) if membership.pid in started else Died(membership) for membership in listed if membership.id not in on_file),
    ]


def _observed(record: Recorded, holds_its_process: bool, started: Mapping[int, float]) -> Observed:
    start = started.get(record.membership.pid)
    # [LAW:types-are-the-program] a running pid is not enough: a process that started after the file was written
    # took the number of the one the file names, which is dead.
    alive = start is not None and start <= record.written_at + START_SLACK_SECONDS
    match (holds_its_process, alive):
        case (False, _):
            return MovedOn(record.membership)
        case (True, True):
            return Attached(record.membership)
        case (True, False):
            return Died(record.membership)


def recorded(home: Home) -> list[Recorded]:
    """Every membership file that parses; one that does not is reported and removed, since it names no session hands can reach."""
    records: list[Recorded] = []
    # A shim's staging file ends in .tmp until it is renamed into place.
    for path in sorted(home.memberships.glob("*.json")):
        session = SessionId(path.stem)
        try:
            written_at = path.stat().st_mtime
            records.append(Recorded(read_membership(home, session), written_at))
        except FileNotFoundError:
            # The session ended between the listing and the read.
            continue
        except Rejected as error:
            if not path.exists():
                continue
            # [LAW:no-silent-failure] said once, as it is removed, rather than every sweep.
            logger.error(f"removing the membership file {path}, which names no session hands can attach: {error}")
            path.unlink(missing_ok=True)
    return records


async def process_starts(pids: Collection[int]) -> dict[int, float]:
    """When each running pid started, in wall-clock seconds; a pid that is not running is absent."""
    if not pids:
        return {}
    ps = await asyncio.create_subprocess_exec(
        "ps", "-o", "pid=,etime=", "-p", ",".join(str(pid) for pid in sorted(pids)),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await ps.communicate()
    now = time.time()
    # ps exits 1, printing nothing, when none of the pids is running.
    if err or ps.returncode not in (0, 1):
        raise RuntimeError(f"ps exited {ps.returncode} asking which sessions are running: {err.decode(errors='replace').strip()}")
    starts: dict[int, float] = {}
    for line in out.decode().splitlines():
        pid, etime = line.split()
        starts[int(pid)] = now - elapsed_seconds(etime)
    return starts


def elapsed_seconds(etime: str) -> int:
    """ps's elapsed time, [[dd-]hh:]mm:ss, in seconds."""
    days, _, clock = etime.rpartition("-")
    seconds = 0
    for part in clock.split(":"):
        seconds = seconds * 60 + int(part)
    return (int(days) if days else 0) * 86400 + seconds
