"""Which sessions are running, from their membership files and the OS: silence is never evidence."""

import asyncio
import time
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from hands.core.events import Attached, Died, MovedOn, Observed
from hands.core.session import Membership, SessionId
from hands.sessions.home import Home
from hands.sessions.membership import parse_membership, remove_ended_membership
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions

# ps reports elapsed time in whole seconds, and it is read a moment after the clock is.
START_SLACK_SECONDS = 2.0


@dataclass(frozen=True)
class Recorded:
    membership: Membership
    written_at: float  # wall-clock seconds: the file's modification time


# Sessions listed while their files were missing at one sweep, which the next one may conclude have ended.
Unfiled = frozenset[SessionId]


async def sweep(home: Home, sessions: Sessions, unfiled_before: Unfiled) -> Unfiled:
    """Apply what the files and the process table say now: a running session is attached, an ended one ends and its file goes."""
    # [LAW:no-ambient-temporal-coupling] taken before the directory is read: a session joins only after its shim
    # has written its file, so one that joins while the sweep runs is never taken for a session whose file is gone.
    listed = sessions.live_members()
    records = recorded(home)
    started = await process_starts({record.membership.pid for record in records})
    seen_all, unfiled = observations(listed, records, started, unfiled_before)
    for seen in seen_all:
        await sessions.apply(seen)
        match seen:
            case Died(membership=membership) | MovedOn(membership=membership):
                remove_ended_membership(home, membership)
            case Attached():
                pass
    return unfiled


async def keep_sweeping(home: Home, sessions: Sessions, period: float) -> None:
    """Sweep now and once a period after, until cancelled. The period is how late a closed pane is heard."""
    unfiled: Unfiled = frozenset()
    while True:
        unfiled = await sweep(home, sessions, unfiled)
        await asyncio.sleep(period)


def observations(
    listed: Collection[Membership], records: Collection[Recorded], started: Mapping[int, float], unfiled_before: Unfiled
) -> tuple[list[Observed], Unfiled]:
    """What each file, and each listed session whose file stayed gone, says about its session; and which listed sessions have no file now."""
    running = [record for record in records if _running(record, started)]
    # One process holds one session: of the files naming a running process, the newest is the session it holds.
    holders = {record.membership.pid: record.membership for record in sorted(running, key=lambda record: record.written_at)}
    on_file = {record.membership.id for record in records}
    unfiled = [membership for membership in listed if membership.id not in on_file]
    return [
        # A file whose own process is not running holds nothing, even when a later session took its pid.
        *(_observed(record.membership, holders.get(record.membership.pid) if record in running else None) for record in records),
        # The shim removes a session's file before it posts SessionEnd, so a listed session with no file has ended. It is
        # judged only once its file was also gone a sweep ago, so the end hook, which lands in milliseconds, says how.
        *(_observed(membership, holders.get(membership.pid)) for membership in unfiled if membership.id in unfiled_before),
    ], frozenset(membership.id for membership in unfiled)


def _running(record: Recorded, started: Mapping[int, float]) -> bool:
    start = started.get(record.membership.pid)
    # [LAW:types-are-the-program] a running pid is not enough: a process that started after the file was written
    # took the number of the one the file names, which is dead.
    return start is not None and start <= record.written_at + START_SLACK_SECONDS


def _observed(membership: Membership, holder: Membership | None) -> Observed:
    match holder:
        case None:
            # No running process holds a session under this pid, so this session's process is gone.
            return Died(membership)
        case Membership(id=held) if held == membership.id:
            return Attached(membership)
        case Membership():
            return MovedOn(membership)


def recorded(home: Home) -> list[Recorded]:
    """Every membership file that parses; one that does not is reported and removed, since it names no session hands can reach."""
    records: list[Recorded] = []
    # A shim's staging file ends in .tmp until it is renamed into place.
    for path in sorted(home.memberships.glob("*.json")):
        try:
            written_at = path.stat().st_mtime
            raw = path.read_bytes()
        except FileNotFoundError:
            # The session ended between the listing and the read.
            continue
        try:
            records.append(Recorded(parse_membership(SessionId(path.stem), raw), written_at))
        except Rejected as error:
            _remove_unreadable(path, raw, error)
    return records


def _remove_unreadable(path: Path, raw: bytes, error: Rejected) -> None:
    try:
        # A shim that rewrote the file since it was read wrote a good one, which stays.
        if path.read_bytes() != raw:
            return
    except FileNotFoundError:
        return
    # [LAW:no-silent-failure] said once, as it is removed, rather than every sweep.
    logger.error(f"removing the membership file {path}, which names no session hands can attach: {error}")
    path.unlink(missing_ok=True)


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
