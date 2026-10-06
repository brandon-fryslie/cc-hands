"""Which sessions are running, from their membership files and the OS: silence is never evidence."""

import asyncio
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from hands.core.events import Attached, Died, MovedOn, Observed
from hands.core.session import Membership, SessionId
from hands.sessions.home import Home
from hands.sessions.membership import parse_membership, remove_ended_membership
from hands.sessions.payload import Rejected
from hands.sessions.processes import process_starts, still_running
from hands.sessions.registry import Sessions


@dataclass(frozen=True)
class Recorded:
    membership: Membership
    written_at: float  # wall-clock seconds: the file's modification time


@dataclass(frozen=True)
class Unreadable:
    """A membership file that does not parse, as it was read: it names no session hands can reach."""

    path: Path
    raw: bytes
    error: Rejected


# Sessions listed while their files were missing at one sweep, which the next one may conclude have ended.
Unfiled = frozenset[SessionId]


async def sweep(home: Home, sessions: Sessions, unfiled_before: Unfiled) -> Unfiled:
    """Apply what the files and the process table say now: a running session is attached, an ended one ends and its file goes."""
    # [LAW:no-ambient-temporal-coupling] taken before the directory is read: a session joins only after its shim
    # has written its file, so one that joins while the sweep runs is never taken for a session whose file is gone.
    listed = sessions.live_members()
    records, unreadable = recorded(home)
    for file in unreadable:
        _remove_unreadable(file)
    started = process_starts({record.membership.pid for record in records})
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
    """Sweep now and once a period after, until cancelled. The period is how late a closed terminal is heard."""
    unfiled: Unfiled = frozenset()
    while True:
        unfiled = await sweep(home, sessions, unfiled)
        await asyncio.sleep(period)


def observed(home: Home, membership: Membership) -> Observed:
    """What the files and the OS say of one session now, judged as a sweep judges it."""
    # [LAW:single-enforcer] the sweep's own judgement, of only the files naming the session's pid: they alone say which
    # session that process holds. Judged as a session already unfiled, so a file that is gone says so now.
    records, _ = recorded(home)
    seen, _ = observations([membership], [record for record in records if record.membership.pid == membership.pid], process_starts({membership.pid}), frozenset({membership.id}))
    return next(observation for observation in seen if observation.membership.id == membership.id)


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
    # A running pid is not enough: the process must have been running when the file was written.
    return still_running(record.membership.pid, record.written_at, started)


def _observed(membership: Membership, holder: Membership | None) -> Observed:
    match holder:
        case None:
            # No running process holds a session under this pid, so this session's process is gone.
            return Died(membership)
        case Membership(id=held) if held == membership.id:
            return Attached(membership)
        case Membership():
            return MovedOn(membership)


def recorded(home: Home) -> tuple[list[Recorded], list[Unreadable]]:
    """Every membership file that parses, and every one that does not; nothing is removed here."""
    records: list[Recorded] = []
    unreadable: list[Unreadable] = []
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
            unreadable.append(Unreadable(path, raw, error))
    return records, unreadable


def _remove_unreadable(file: Unreadable) -> None:
    try:
        # A shim that rewrote the file since it was read wrote a good one, which stays.
        if file.path.read_bytes() != file.raw:
            return
    except FileNotFoundError:
        return
    # [LAW:no-silent-failure] said once, as it is removed, rather than every sweep.
    logger.error(f"removing the membership file {file.path}, which names no session hands can attach: {file.error}")
    file.path.unlink(missing_ok=True)
