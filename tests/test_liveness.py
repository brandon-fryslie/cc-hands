"""The liveness sweep against real membership files and the real process table."""

import asyncio
import os
import subprocess
import time
from pathlib import Path

import pytest
from pipecat.frames.frames import TTSSpeakFrame

from hands.core.effects import SessionGone, Speak
from hands.core.events import Attached, Died, Joined, MovedOn, Observed
from hands.core.session import Gone, Idle, Membership, SessionId
from hands.sessions.home import Home
from hands.sessions.liveness import START_SLACK_SECONDS, Recorded, elapsed_seconds, observations, process_starts, recorded, sweep
from hands.sessions.membership import remove_ended_membership, write_membership
from hands.sessions.registry import Sessions
from hands.voice.speech import frame


def member(name: str, pid: int) -> Membership:
    return Membership(SessionId(name), pid=pid, pane=None, cwd=Path("/code") / name, transcript=Path("/nowhere") / f"{name}.jsonl")


def dead_pid() -> int:
    process = subprocess.Popen(["true"])
    process.wait()
    return process.pid


def sessions() -> Sessions:
    return Sessions(permission_deadline=60.0, clock=time.monotonic)


@pytest.mark.parametrize(("etime", "seconds"), [("00:07", 7), ("12:34", 754), ("01:02:03", 3723), ("2-01:02:03", 176523), ("10-00:00:00", 864000)])
def test_ps_elapsed_time_reads_as_seconds(etime: str, seconds: int) -> None:
    assert elapsed_seconds(etime) == seconds


@pytest.mark.parametrize(
    ("started", "seen"),
    [
        ({4242: 100.0}, Attached),  # started long before its file was written
        ({4242: 1000.0 + START_SLACK_SECONDS}, Attached),  # ps rounds, so the edge is alive
        ({4242: 1000.0 + START_SLACK_SECONDS + 1}, Died),  # a later process took the dead one's pid
        ({}, Died),  # not running at all
    ],
)
def test_a_file_names_a_running_session_only_if_its_process_started_before_the_file(started: dict[int, float], seen: type) -> None:
    record = Recorded(member("a", 4242), written_at=1000.0)
    assert observations([], [record], started) == [seen(record.membership)]


@pytest.mark.parametrize("names", [("a", "b"), ("b", "a")])
@pytest.mark.parametrize("alive", [True, False])
def test_of_the_files_naming_one_process_the_newest_is_its_session_whatever_the_ids(names: tuple[str, str], alive: bool) -> None:
    old, new = Recorded(member(names[0], 4242), written_at=1000.0), Recorded(member(names[1], 4242), written_at=2000.0)
    started = {4242: 500.0} if alive else {}
    newest: Observed = Attached(new.membership) if alive else Died(new.membership)
    for records in ([old, new], [new, old]):
        assert set(observations([], records, started)) == {MovedOn(old.membership), newest}


def test_a_listed_session_whose_file_is_gone_moved_on_if_its_process_runs_and_died_if_not() -> None:
    running, dead, on_file = member("running", 4242), member("dead", 5353), member("on_file", 6464)
    records = [Recorded(on_file, written_at=1000.0)]
    assert observations([running, dead, on_file], records, {4242: 1.0, 6464: 1.0}) == [Attached(on_file), MovedOn(running), Died(dead)]


async def test_ps_says_when_a_running_process_started_and_leaves_out_a_dead_one() -> None:
    gone = dead_pid()
    starts = await process_starts({os.getpid(), gone})
    assert set(starts) == {os.getpid()}
    assert starts[os.getpid()] <= time.time()


async def test_no_pids_asks_ps_nothing() -> None:
    assert await process_starts(set()) == {}


def test_an_unreadable_file_is_removed_and_a_staging_file_is_not_read(tmp_path: Path) -> None:
    home = Home(tmp_path)
    write_membership(home, member("good", os.getpid()))
    home.membership(SessionId("bad")).write_text("{not json")
    (home.memberships / "half.tmp").write_text("{")
    assert [record.membership.id for record in recorded(home)] == ["good"]
    assert not home.membership(SessionId("bad")).exists()
    assert (home.memberships / "half.tmp").exists()


def test_a_dead_sessions_file_is_kept_when_a_new_process_has_rewritten_it(tmp_path: Path) -> None:
    home = Home(tmp_path)
    write_membership(home, member("a", os.getpid()))
    remove_ended_membership(home, member("a", dead_pid()))
    assert home.membership(SessionId("a")).exists()
    remove_ended_membership(home, member("a", os.getpid()))
    assert not home.membership(SessionId("a")).exists()


async def test_a_sweep_attaches_the_running_ends_the_dead_and_the_reused_and_speaks_each_death_once(tmp_path: Path) -> None:
    home = Home(tmp_path)
    running, dead, reused = member("running", os.getpid()), member("dead", dead_pid()), member("reused", os.getppid())
    for membership in (running, dead, reused):
        write_membership(home, membership)
    # The file names the parent pid but was written before that process started: the pid was taken by someone else.
    long_ago = time.time() - 10 * 365 * 86400
    os.utime(home.membership(reused.id), (long_ago, long_ago))
    registry = sessions()

    await sweep(home, registry)
    assert [listing.session.membership.id for listing in registry.live()] == [running.id]
    assert registry.listing(dead.id).session.state == Gone()  # pyright: ignore[reportOptionalMemberAccess]
    heard = {await registry.heard(), await registry.heard()}
    assert heard == {Speak(SessionGone(dead.id)), Speak(SessionGone(reused.id))}
    assert sorted(path.stem for path in home.memberships.glob("*.json")) == [running.id]

    await sweep(home, registry)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(registry.heard(), 0.1)


@pytest.mark.parametrize("stale", ["0-cleared", "z-cleared"])
async def test_a_restart_lists_the_session_a_cleared_process_holds_now_not_the_one_whose_end_was_lost(tmp_path: Path, stale: str) -> None:
    home = Home(tmp_path)
    write_membership(home, member(stale, os.getpid()))
    long_ago = time.time() - 60
    os.utime(home.membership(SessionId(stale)), (long_ago, long_ago))
    write_membership(home, member("m-current", os.getpid()))
    registry = sessions()
    await sweep(home, registry)
    assert [listing.session.membership.id for listing in registry.live()] == ["m-current"]
    assert sorted(path.stem for path in home.memberships.glob("*.json")) == ["m-current"]
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(registry.heard(), 0.1)


async def test_a_session_whose_end_hook_was_lost_after_its_file_went_is_ended_by_the_next_sweep(tmp_path: Path) -> None:
    home = Home(tmp_path)
    registry = sessions()
    await registry.apply(Joined(member("closed", dead_pid()), "startup"))
    await sweep(home, registry)
    assert registry.live() == []
    assert await registry.heard() == Speak(SessionGone(SessionId("closed")))


async def test_a_restarted_daemon_lists_the_sessions_the_last_one_did(tmp_path: Path) -> None:
    home = Home(tmp_path)
    for name, pid in (("one", os.getpid()), ("two", os.getppid())):
        write_membership(home, member(name, pid))
    before, after = sessions(), sessions()
    await sweep(home, before)
    await sweep(home, after)
    assert [listing.session for listing in after.live()] == [listing.session for listing in before.live()]
    assert {listing.session.state for listing in after.live()} == {Idle()}
    assert len(after.live()) == 2


def test_a_death_is_spoken_as_written_by_the_sessions_name() -> None:
    spoken = frame(Speak(SessionGone(SessionId("s1"))), names=lambda _: "cc-hands")
    assert isinstance(spoken, TTSSpeakFrame)
    assert spoken.text == "The session cc-hands is gone."
