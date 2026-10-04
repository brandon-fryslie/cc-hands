"""The heartbeat file and what `hands status` concludes from it, with no daemon running."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from conftest import unedited
from hands.daemon import indicator
from hands.sessions import heartbeat
from hands.daemon.cli import Run, launch, main
from hands.daemon.starting import Ended, Start, keep_beating
from hands.threads import off_loop
from hands.sessions.home import Home
from hands.sessions.payload import Rejected

NOW = datetime(2026, 9, 14, 12, 0, 0, tzinfo=UTC)
BEAT = timedelta(seconds=2)


def beat(**changes: object) -> heartbeat.Status:
    fields: dict[str, object] = {
        "pid": 4242,
        "started_at": NOW - timedelta(minutes=5, seconds=3),
        "written_at": NOW - timedelta(seconds=1),
        "heartbeat": BEAT,
        "pipeline": "running",
        "last_audio_out": NOW - timedelta(seconds=12),
        "live_sessions": 2,
        "listening": False,
        "deaf": False,
        **changes,
    }
    return heartbeat.Status(**fields)  # pyright: ignore[reportArgumentType]


def test_a_heartbeat_reads_back_as_it_was_written(tmp_path: Path) -> None:
    written = beat(last_audio_out=None)
    heartbeat.write(tmp_path / "status.json", written)
    assert heartbeat.read(tmp_path / "status.json") == written
    assert [path.name for path in tmp_path.iterdir()] == ["status.json"]  # nothing left of the replacement


def test_a_refused_start_reads_back_with_its_reason(tmp_path: Path) -> None:
    written = beat(pipeline=heartbeat.Refusal("OPENAI_API_KEY is not set"), last_audio_out=None, live_sessions=0)
    heartbeat.write(tmp_path / "status.json", written)
    assert heartbeat.read(tmp_path / "status.json") == written


def test_a_heartbeat_from_before_turns_were_written_reads_as_no_turn_open(tmp_path: Path) -> None:
    written = beat(listening=False)
    old = {key: value for key, value in json.loads(heartbeat.encode(written)).items() if key != "listening"}
    (tmp_path / "status.json").write_text(json.dumps(old))
    assert heartbeat.read(tmp_path / "status.json") == written


def test_a_heartbeat_from_before_hearing_was_written_reads_as_able_to_hear(tmp_path: Path) -> None:
    written = beat(deaf=False)
    old = {key: value for key, value in json.loads(heartbeat.encode(written)).items() if key != "deaf"}
    (tmp_path / "status.json").write_text(json.dumps(old))
    assert heartbeat.read(tmp_path / "status.json") == written


def test_every_heartbeat_of_a_run_repeats_what_the_heart_fixed(tmp_path: Path) -> None:
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=NOW, period=BEAT)
    heart.beat("starting", None, 0, listening=False, deaf=True)
    first = heartbeat.read(heart.path)
    heart.beat("running", NOW, 2, listening=True, deaf=False)
    second = heartbeat.read(heart.path)
    assert first is not None and second is not None
    assert (first.pid, first.started_at, first.heartbeat, first.pipeline, first.live_sessions) == (4242, NOW, BEAT, "starting", 0)
    assert (second.pid, second.started_at, second.heartbeat, second.pipeline, second.last_audio_out) == (4242, NOW, BEAT, "running", NOW)
    assert (first.listening, second.listening) == (False, True)
    assert (first.deaf, second.deaf) == (True, False)
    assert second.written_at >= first.written_at


async def test_a_start_still_importing_pipecat_beats_starting_until_the_run_it_loads_begins(tmp_path: Path) -> None:
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=NOW, period=timedelta(milliseconds=10))
    imported = threading.Event()
    ran: list[asyncio.Event] = []

    async def run(quit_event: asyncio.Event) -> Ended:
        ran.append(quit_event)
        return Ended(None, 0)

    def load() -> Run:
        imported.wait()
        return run

    launched = asyncio.create_task(launch(load, heart, unedited, lambda _entry: None, Start(restarted=False, after_crash=False)))
    # The import finishes only once the start has said "starting" three times while it waited.
    beats: set[datetime] = set()
    while len(beats) < 3:
        status = heartbeat.read(heart.path)
        if status is not None and status.pipeline == "starting":
            beats.add(status.written_at)
        await asyncio.sleep(0.005)
    imported.set()
    await launched
    assert len(ran) == 1 and not ran[0].is_set()


async def test_a_stop_during_the_pipecat_import_ends_the_run_as_stopped(tmp_path: Path) -> None:
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=NOW, period=timedelta(milliseconds=10))
    never = threading.Event()

    def load() -> Run:
        never.wait()
        raise AssertionError("the import never finished")

    launched = asyncio.create_task(launch(load, heart, unedited, lambda _entry: None, Start(restarted=False, after_crash=False)))
    while heartbeat.read(heart.path) is None:
        await asyncio.sleep(0.005)
    os.kill(os.getpid(), signal.SIGTERM)
    await launched
    status = heartbeat.read(heart.path)
    assert status is not None and status.pipeline == "stopped"
    never.set()


def test_no_file_is_a_daemon_that_never_ran(tmp_path: Path) -> None:
    assert heartbeat.read(tmp_path / "status.json") is None


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        (b"{", "not JSON"),
        (b'{"pid": 1}', "started_at"),
        (heartbeat.encode(beat()).replace("running", "dancing").encode(), "pipeline should be"),
        (heartbeat.encode(beat()).replace("+00:00", "").encode(), "carries its zone"),
        (heartbeat.encode(beat(pid=2**63)).encode(), "not a process id"),
        (heartbeat.encode(beat(pid=2**31)).encode(), "not a process id"),
        (heartbeat.encode(beat(pid=100_000)).encode(), "not a process id"),  # past macOS's PID_MAX, which ps refuses
        (heartbeat.encode(beat(pid=0)).encode(), "not a process id"),  # kill(0, 0) asks after our own process group
        (heartbeat.encode(beat(pid=-1)).encode(), "not a process id"),
        (heartbeat.encode(beat()).replace('"heartbeat_ms": 2000', '"heartbeat_ms": 100000000000000000000').encode(), "not a heartbeat period"),
        (heartbeat.encode(beat(heartbeat=timedelta(0))).encode(), "not a heartbeat period"),
        (heartbeat.encode(beat(pipeline=heartbeat.Refusal("OPENAI_API_KEY is not set"))).replace('"refusal"', '"reason"').encode(), "says why"),
        (heartbeat.encode(beat(pipeline=heartbeat.Refusal("OPENAI_API_KEY is not set"))).replace('"refused"', '"stopped"').encode(), "only of a refused pipeline"),
    ],
)
def test_a_heartbeat_that_does_not_parse_is_refused(raw: bytes, error: str) -> None:
    with pytest.raises(Rejected, match=error):
        heartbeat.parse(raw)


def test_the_verdict_follows_the_pid_and_the_heartbeat_age(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    assert heartbeat.judge(path, None, NOW, alive=False) == heartbeat.NeverRan(path)
    assert heartbeat.judge(path, beat(), NOW, alive=True) == heartbeat.Up(beat())
    assert heartbeat.judge(path, beat(), NOW, alive=False) == heartbeat.Down(beat())
    late = beat(written_at=NOW - BEAT * heartbeat.MISSED_BEATS - timedelta(seconds=1))
    assert heartbeat.judge(path, late, NOW, alive=True) == heartbeat.Unresponsive(late)


@pytest.mark.parametrize("alive", [False, True])
def test_a_daemon_whose_last_heartbeat_said_it_refused_to_start_is_refused_whoever_holds_its_pid_now(alive: bool, tmp_path: Path) -> None:
    refused = beat(pipeline=heartbeat.Refusal("OPENAI_API_KEY is not set"))
    assert heartbeat.judge(tmp_path, refused, NOW, alive=alive) == heartbeat.Refused(heartbeat.Refusal("OPENAI_API_KEY is not set"), refused.written_at)


@pytest.mark.parametrize("alive", [False, True])
def test_a_daemon_whose_last_heartbeat_said_stopped_is_stopped_whoever_holds_its_pid_now(alive: bool, tmp_path: Path) -> None:
    # Alive is a daemon still cleaning up, or another process that got the pid; neither is a hang.
    stopped = beat(pipeline="stopped", written_at=NOW - timedelta(hours=1))
    assert heartbeat.judge(tmp_path, stopped, NOW, alive=alive) == heartbeat.Stopped(stopped)


def test_each_verdict_is_said_plainly(tmp_path: Path) -> None:
    assert heartbeat.describe(heartbeat.Up(beat()), NOW) == (
        "hands is up: pid 4242, up 5m 3s, pipeline running, last audio out 12s ago, 2 live sessions"
    )
    assert heartbeat.describe(heartbeat.Up(beat(last_audio_out=None, live_sessions=1, started_at=NOW - timedelta(hours=2))), NOW) == (
        "hands is up: pid 4242, up 2h 0m 0s, pipeline running, last audio out never, 1 live session"
    )
    assert heartbeat.describe(heartbeat.Down(beat()), NOW) == "hands is down: its process, pid 4242, is gone; its last heartbeat was 1s ago"
    assert heartbeat.describe(heartbeat.Unresponsive(beat(written_at=NOW - timedelta(minutes=3))), NOW) == (
        "hands is not responding: pid 4242 is running, pipeline running, but its last heartbeat was 3m 0s ago"
    )
    assert heartbeat.describe(heartbeat.Stopped(beat(pipeline="stopped")), NOW) == "hands is stopped: pid 4242 finished its pipeline 1s ago"
    assert heartbeat.describe(heartbeat.Refused(heartbeat.Refusal("OPENAI_API_KEY is not set"), beat().written_at), NOW) == "hands refused to start 1s ago: OPENAI_API_KEY is not set"
    assert heartbeat.describe(heartbeat.NeverRan(tmp_path), NOW) == f"hands has not run: there is no heartbeat at {tmp_path}"
    assert heartbeat.describe(heartbeat.Unreadable(tmp_path, "not JSON"), NOW) == (
        f"hands is unknown: its heartbeat at {tmp_path} cannot be read: not JSON"
    )


def test_looking_at_the_heartbeat_judges_it_against_the_process_table(dead_pid: Callable[[], int], tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    now = datetime.now(UTC)
    assert heartbeat.look(path, now) == heartbeat.NeverRan(path)
    heartbeat.write(path, beat(pid=os.getpid(), started_at=now, written_at=now))
    assert isinstance(heartbeat.look(path, now), heartbeat.Up)
    heartbeat.write(path, beat(pid=dead_pid(), started_at=now, written_at=now))
    assert isinstance(heartbeat.look(path, now), heartbeat.Down)


@pytest.mark.parametrize(
    "raw", [b"{", heartbeat.encode(beat(pid=2**63)).encode(), heartbeat.encode(beat()).replace('"heartbeat_ms": 2000', '"heartbeat_ms": 1e400').encode()]
)
def test_a_heartbeat_that_cannot_be_read_is_its_own_verdict_and_never_raises(raw: bytes, tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    path.write_bytes(raw)
    assert isinstance(heartbeat.look(path, NOW), heartbeat.Unreadable)
    path.unlink()
    path.mkdir()  # a read that fails in the OS, not in the parse
    assert isinstance(heartbeat.look(path, NOW), heartbeat.Unreadable)


def test_hands_status_exits_zero_only_when_the_daemon_is_up(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path)
    assert main(["--home", str(tmp_path), "status"]) == 1
    assert "has not run" in capsys.readouterr().out
    heartbeat.write(home.status, beat(pid=os.getpid(), started_at=datetime.now(UTC), written_at=datetime.now(UTC)))
    assert main(["--home", str(tmp_path), "status"]) == 0
    assert capsys.readouterr().out.startswith(f"hands is up: pid {os.getpid()}")
    home.status.write_text("{")
    assert main(["--home", str(tmp_path), "status"]) == 2
    assert "cannot be read" in capsys.readouterr().err
    heartbeat.write(home.status, beat(pid=2**63))
    assert main(["--home", str(tmp_path), "status"]) == 2
    assert "not a process id" in capsys.readouterr().err


def test_a_relative_hands_home_is_refused_but_never_stands_in_the_way_of_an_explicit_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HANDS_HOME", "relhome")
    assert main(["--home", str(tmp_path), "status"]) == 1
    assert "has not run" in capsys.readouterr().out
    assert main(["status"]) == 2
    assert "HANDS_HOME must be an absolute path, got 'relhome'" in capsys.readouterr().err


def test_the_daemon_is_running_only_if_its_pid_is_held_by_the_process_that_started_then(dead_pid: Callable[[], int]) -> None:
    now = datetime.now(UTC)
    assert heartbeat.running(beat(pid=os.getpid(), started_at=now))
    assert heartbeat.running(beat(pid=1, started_at=now))  # launchd, root's: seen, and it started before now
    assert not heartbeat.running(beat(pid=dead_pid(), started_at=now))
    # This process holds the pid, but it started long after the heartbeat's daemon did: the number was reused.
    assert not heartbeat.running(beat(pid=os.getpid(), started_at=now - timedelta(days=3)))


def test_a_heartbeat_whose_pid_went_to_a_later_process_reads_as_down(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    now = datetime.now(UTC)
    # After a reboot: the old heartbeat, still on disk, names a pid some new process now holds.
    heartbeat.write(path, beat(pid=os.getpid(), started_at=now - timedelta(days=3), written_at=now - timedelta(days=3)))
    assert isinstance(heartbeat.look(path, now), heartbeat.Down)


async def test_the_heartbeat_is_rewritten_every_period() -> None:
    beats: list[None] = []
    beating = asyncio.create_task(keep_beating(lambda: beats.append(None), 0.01))
    await asyncio.sleep(0.055)
    beating.cancel()
    assert 3 <= len(beats) <= 6


async def test_work_off_the_loop_returns_its_result_or_raises_its_error() -> None:
    assert await off_loop(lambda: 42, "answer") == 42
    with pytest.raises(ValueError, match="no models"):
        await off_loop(lambda: (_ for _ in ()).throw(ValueError("no models")), "failure")


async def test_a_process_exits_without_waiting_for_work_left_running_off_the_loop() -> None:
    script = (
        "import asyncio, time\n"
        "from hands.threads import off_loop\n"
        "async def main():\n"
        "    work = asyncio.create_task(off_loop(lambda: time.sleep(30), 'slow'))\n"
        "    await asyncio.sleep(0.1)\n"
        "    work.cancel()\n"
        "    print('cancelled', flush=True)\n"
        "asyncio.run(main())\n"
    )
    child = await asyncio.create_subprocess_exec(sys.executable, "-c", script, stdout=subprocess.PIPE)
    said = child.stdout
    assert said is not None
    # Startup and the Pipecat import get their own generous bound, so a hang there fails here and not as a slow
    # exit; only the interval from the cancel to the exit is held to the budget.
    try:
        assert await asyncio.wait_for(said.readline(), 30) == b"cancelled\n"
        assert await asyncio.wait_for(child.wait(), 5) == 0
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()


def test_each_verdict_has_its_own_light_and_the_broken_ones_warn(tmp_path: Path) -> None:
    verdicts: list[heartbeat.Verdict] = [
        heartbeat.Up(beat()),
        heartbeat.Up(beat(deaf=True)),
        heartbeat.Unresponsive(beat()),
        heartbeat.Down(beat()),
        heartbeat.Refused(heartbeat.Refusal("OPENAI_API_KEY is not set"), beat().written_at),
        heartbeat.NeverRan(tmp_path),
        heartbeat.Unreadable(tmp_path, "not JSON"),
    ]
    shown = [indicator.show(None, verdict, NOW) for verdict in verdicts]
    assert [seen.light for seen in shown] == ["up", "deaf", "not responding", "down", "refused", "off", "unreadable"]
    assert len({seen.title for seen in shown}) == len(shown)
    assert [seen.title.startswith("⚠︎") for seen in shown] == [False, True, True, True, True, False, True]
    assert indicator.show(None, heartbeat.Stopped(beat(pipeline="stopped")), NOW).light == "off"
    assert [seen.text for seen in shown] == [heartbeat.describe(verdict, NOW) for verdict in verdicts]


def test_an_open_turn_shows_in_the_menu_bar_and_is_never_a_notice() -> None:
    idle = indicator.show(None, heartbeat.Up(beat()), NOW)
    talking = indicator.show(idle, heartbeat.Up(beat(listening=True)), NOW)
    back = indicator.show(talking, heartbeat.Up(beat()), NOW)
    assert (idle.title, talking.title, back.title) == ("✋", "✋ 🎙", "✋")
    assert talking.light == "up" and talking.notices == back.notices == ()
    # A stuck daemon's last word was a turn open, and stuck is what shows.
    assert indicator.show(talking, heartbeat.Unresponsive(beat(listening=True)), NOW).title == "⚠︎ hands stuck"


def test_a_daemon_that_cannot_hear_shows_its_own_light_and_says_so_in_words() -> None:
    deaf = indicator.show(None, heartbeat.Up(beat(deaf=True, listening=True)), NOW)
    assert (deaf.light, deaf.title) == ("deaf", "⚠︎ hands can't hear")
    assert deaf.text.startswith("hands is up but cannot hear, as there is no microphone: pid 4242")


def test_losing_the_microphone_is_announced_and_so_is_a_deaf_daemon_going_down() -> None:
    up, deaf, down = heartbeat.Up(beat()), heartbeat.Up(beat(deaf=True)), heartbeat.Down(beat())
    at = [NOW + timedelta(seconds=seconds) for seconds in (0, 100, 110, 200, 300, 400)]
    looks = list(zip([up, deaf, deaf, up, deaf, down], at))
    assert shown_over(looks) == [(), (heartbeat.describe(deaf, at[1]),), (), (), (heartbeat.describe(deaf, at[4]),), (heartbeat.describe(down, at[5]),)]


def test_a_start_refused_after_it_beat_starting_is_announced_with_its_reason() -> None:
    # Started from a launcher whose terminal nobody watches, the notice is where the reason reaches the user.
    starting, refused = heartbeat.Up(beat(pipeline="starting")), heartbeat.Refused(heartbeat.Refusal("OPENAI_API_KEY is not set"), beat().written_at)
    at = [NOW, NOW + timedelta(seconds=5)]
    assert shown_over(list(zip([starting, refused], at))) == [(), (heartbeat.describe(refused, at[1]),)]


def test_a_stuck_daemon_that_recovers_unable_to_hear_says_so() -> None:
    up, stuck, deaf = heartbeat.Up(beat()), heartbeat.Unresponsive(beat()), heartbeat.Up(beat(deaf=True))
    at = [NOW + timedelta(seconds=seconds) for seconds in (0, 100, 200)]
    assert shown_over(list(zip([up, stuck, deaf], at))) == [(), (heartbeat.describe(stuck, at[1]),), (heartbeat.describe(deaf, at[2]),)]


def test_a_deaf_daemon_is_read_from_the_heartbeat_by_hands_status(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path)
    heartbeat.write(home.status, beat(pid=os.getpid(), started_at=datetime.now(UTC), written_at=datetime.now(UTC), deaf=True))
    assert main(["--home", str(tmp_path), "status"]) == 0
    assert capsys.readouterr().out.startswith(f"hands is up but cannot hear, as there is no microphone: pid {os.getpid()}")


def shown_over(looks: Sequence[tuple[heartbeat.Verdict, datetime]]) -> list[tuple[str, ...]]:
    before: indicator.Shown | None = None
    posted: list[tuple[str, ...]] = []
    for verdict, at in looks:
        before = indicator.show(before, verdict, at)
        posted.append(before.notices)
    return posted


def test_a_notification_is_posted_when_the_verdict_leaves_up_and_only_then(tmp_path: Path) -> None:
    down, up = heartbeat.Down(beat()), heartbeat.Up(beat())
    looks: list[heartbeat.Verdict] = [down, up, up, down, down, heartbeat.Unreadable(tmp_path, "x")]
    assert shown_over([(verdict, NOW) for verdict in looks]) == [(), (), (), (heartbeat.describe(down, NOW),), (), ()]


def test_a_daemon_that_keeps_stalling_is_announced_once_a_quiet_window(tmp_path: Path) -> None:
    stuck, up = heartbeat.Unresponsive(beat()), heartbeat.Up(beat())
    # A loop that stalls for a few seconds, recovers, and stalls again every ten.
    looks = [(verdict, NOW + timedelta(seconds=10 * cycle)) for cycle in range(8) for verdict in (up, stuck)]
    posted = [at for (_, at), notices in zip(looks, shown_over(looks)) if notices]
    assert posted == [NOW, NOW + timedelta(seconds=60)]


def test_a_death_inside_the_quiet_window_is_announced_when_the_window_closes_if_hands_is_still_down() -> None:
    down, up = heartbeat.Down(beat()), heartbeat.Up(beat())
    at = [NOW + timedelta(seconds=seconds) for seconds in (0, 1, 10, 40, 50, 61, 70)]
    looks = list(zip([up, down, up, down, down, down, down], at))
    assert [bool(notices) for notices in shown_over(looks)] == [False, True, False, False, False, True, False]


def test_a_death_owed_inside_the_quiet_window_is_dropped_if_hands_comes_back_before_it_closes() -> None:
    down, up = heartbeat.Down(beat()), heartbeat.Up(beat())
    at = [NOW + timedelta(seconds=seconds) for seconds in (0, 1, 10, 40, 55, 70)]
    looks = list(zip([up, down, up, down, up, up], at))
    assert not any(shown_over(looks)[2:])


def test_the_indicator_finishes_once_the_run_that_started_it_is_gone_and_the_heartbeat_no_longer_says_it_is_up(tmp_path: Path) -> None:
    run = beat()
    looks: list[heartbeat.Verdict] = [
        heartbeat.Up(run),
        heartbeat.Up(beat(pid=run.pid + 1)),
        heartbeat.Unresponsive(run),
        heartbeat.Down(run),
        heartbeat.Stopped(beat(pipeline="stopped")),
        heartbeat.Unreadable(tmp_path, "x"),
        heartbeat.NeverRan(tmp_path),
    ]
    # A hung run is still its parent: the indicator stays to show it stuck.
    assert not any(indicator.finished(verdict, orphaned=False, run=run.pid) for verdict in looks)
    # Orphaned while its run reads up is a run that has exited and not yet been reaped: one more look. Up under another
    # pid is the next run, which has an indicator of its own.
    assert [indicator.finished(verdict, orphaned=True, run=run.pid) for verdict in looks] == [False, True, True, True, True, True, True]


def test_a_departure_the_quiet_window_held_back_goes_out_as_the_indicator_finishes(tmp_path: Path) -> None:
    stuck, down, up = heartbeat.Unresponsive(beat()), heartbeat.Down(beat()), heartbeat.Up(beat())
    before = None
    for verdict, seconds in [(up, 0), (stuck, 1), (up, 5), (down, 20)]:
        before = indicator.show(before, verdict, NOW + timedelta(seconds=seconds))
    # Killed twenty seconds after a stall was announced: the quiet window holds the death back, and the way out says it.
    assert before.notices == () and before.owed
    assert indicator.last_words(before) == (heartbeat.describe(down, NOW + timedelta(seconds=20)),)
    # A departure already posted is not posted twice, and a light that is up has nothing to say.
    assert indicator.last_words(indicator.show(None, down, NOW)) == ()
    assert indicator.last_words(indicator.show(before, up, NOW + timedelta(seconds=21))) == ()
