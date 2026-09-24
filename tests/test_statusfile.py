"""Claude Code's status of a session, read from the bytes of files it wrote (2.1.282), and heard each time it is set.

The fixtures are status files copied as a live session wrote them, with only the remote-control session id blanked.
"""

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from loguru import logger
from hands.core.session import Idle as Resting, Membership, Session, SessionId
from hands.core.status import Busy, Idle, Report, Shell, Stamp, Status, Unknown, UnknownReason, Waiting
from hands.sessions.payload import Rejected
from hands.sessions.statusfile import Statuses, parse_report, status_file

FIXTURES = Path(__file__).parent / "fixtures" / "status"


def written(name: str) -> bytes:
    return (FIXTURES / f"{name}.json").read_bytes()


def edited(name: str, **fields: object) -> bytes:
    return json.dumps({**json.loads(written(name)), **fields}).encode()


def member_of(raw: bytes, config: Path = Path("/config")) -> Membership:
    record = json.loads(raw)
    session = SessionId(record["sessionId"])
    return Membership(session, pid=record["pid"], cwd=Path(record["cwd"]), transcript=config / "projects" / "-work" / f"{session}.jsonl")


@pytest.mark.parametrize(
    ("name", "status"),
    [
        ("idle", Idle()),
        ("busy", Busy()),
        ("waiting-permission", Waiting("permission prompt")),
        ("waiting-input", Waiting("input needed")),
    ],
)
def test_each_status_a_session_wrote_is_read_as_itself_with_its_stamp(name: str, status: Status) -> None:
    raw = written(name)
    assert parse_report(member_of(raw), raw) == Report(status, Stamp(json.loads(raw)["statusUpdatedAt"]))


def test_the_shell_status_seen_on_2_1_280_is_read_as_shell() -> None:
    raw = edited("idle", status="shell")
    assert parse_report(member_of(raw), raw).status == Shell()


@pytest.mark.parametrize(
    ("fields", "status"),
    [
        ({"status": "dreaming"}, Unknown("dreaming")),
        ({"status": "waiting", "waitingFor": "a coffee"}, Waiting(UnknownReason("a coffee"))),
    ],
)
def test_a_status_or_reason_this_version_does_not_know_is_unknown_never_idle(fields: dict[str, object], status: Status) -> None:
    raw = edited("idle", **fields)
    assert parse_report(member_of(raw), raw).status == status


@pytest.mark.parametrize(
    "fields",
    [
        {"pid": 4242},  # another process's file
        {"sessionId": "another-session"},  # the process has moved on to another session
        {"status": None},
        {"statusUpdatedAt": "soon"},
        {"status": "waiting", "waitingFor": None},
    ],
)
def test_a_file_that_is_not_this_sessions_status_is_refused(fields: dict[str, object]) -> None:
    raw = written("idle")
    with pytest.raises(Rejected):
        parse_report(member_of(raw), edited("idle", **fields))


def test_the_file_is_found_in_the_config_directory_the_session_keeps_its_transcript_in(tmp_path: Path) -> None:
    raw = written("busy")
    member = member_of(raw, tmp_path / "claude-other")
    assert status_file(member) == tmp_path / "claude-other" / "sessions" / f"{member.pid}.json"


class Live:
    """A live session whose status file lives under tmp_path, rewritten as Claude Code rewrites it, and whose registry
    entry keeps each report it is handed, as the reducer does."""

    def __init__(self, tmp_path: Path) -> None:
        self.session = Session(member_of(written("idle"), tmp_path), Resting(), mode=None, turn=None)
        status_file(self.member).parent.mkdir(parents=True, exist_ok=True)

    @property
    def member(self) -> Membership:
        return self.session.membership

    def sets(self, name: str, stamp: int, **fields: object) -> Report:
        raw = edited(name, statusUpdatedAt=stamp, **fields)
        status_file(self.member).write_bytes(raw)
        return parse_report(self.member, raw)

    def writes(self, raw: bytes) -> None:
        status_file(self.member).write_bytes(raw)

    def heard(self, statuses: Statuses) -> list[Report]:
        heard = list(statuses.read([self.member.id], lambda _: self.session))
        assert {reported.session for reported in heard} <= {self.member.id}
        reports = [reported.report for reported in heard]
        for report in reports:
            self.session = replace(self.session, report=report)
        return reports


def test_each_time_the_stamp_moves_the_status_is_heard_once(tmp_path: Path) -> None:
    live, statuses = Live(tmp_path), Statuses(clock=lambda: 5.0)
    busy = live.sets("busy", 1000)
    assert live.heard(statuses) == [busy]
    assert live.heard(statuses) == []
    idle = live.sets("idle", 2000)
    assert live.heard(statuses) == [idle]


def test_a_status_set_again_to_what_it_was_is_heard_because_its_stamp_moved(tmp_path: Path) -> None:
    # An idle, busy, idle between two reads leaves the file idle as it was, with a later stamp.
    live, statuses = Live(tmp_path), Statuses(clock=lambda: 5.0)
    live.sets("idle", 1000)
    live.heard(statuses)
    again = live.sets("idle", 3000)
    assert live.heard(statuses) == [again]


def test_a_report_the_registry_does_not_hold_is_heard_again(tmp_path: Path) -> None:
    live, statuses = Live(tmp_path), Statuses(clock=lambda: 5.0)
    idle = live.sets("idle", 1000)
    live.heard(statuses)
    live.session = replace(live.session, report=None)
    assert live.heard(statuses) == [idle]


def test_a_session_with_no_status_file_is_heard_once_it_has_one(tmp_path: Path) -> None:
    live, statuses = Live(tmp_path), Statuses(clock=lambda: 5.0)
    assert live.heard(statuses) == []
    assert live.heard(statuses) == []
    idle = live.sets("idle", 1000)
    assert live.heard(statuses) == [idle]


def test_a_transcript_outside_any_config_directory_is_refused_not_raised(tmp_path: Path) -> None:
    session = Session(replace(member_of(written("idle")), transcript=Path("/s.jsonl")), Resting(), mode=None, turn=None)
    assert list(Statuses(clock=lambda: 5.0).read([session.membership.id], lambda _: session)) == []


def test_an_unreadable_status_file_is_refused_not_raised(tmp_path: Path) -> None:
    live = Live(tmp_path)
    status_file(live.member).mkdir()
    assert live.heard(Statuses(clock=lambda: 5.0)) == []


def logged(read: Callable[[], object]) -> list[str]:
    warnings = list[str]()
    sink = logger.add(lambda message: warnings.append(message.record["message"]), level="WARNING", filter="hands")
    try:
        read()
    finally:
        logger.remove(sink)
    return warnings


def test_why_a_status_cannot_be_read_is_said_once_not_every_read(tmp_path: Path) -> None:
    live, statuses = Live(tmp_path), Statuses(clock=lambda: 5.0)
    warnings = logged(lambda: [live.heard(statuses) for _ in range(3)])
    assert len(warnings) == 1 and "keeps no status" in warnings[0]


def test_a_read_that_fails_once_neither_hears_the_status_again_nor_goes_unsaid(tmp_path: Path) -> None:
    live, statuses = Live(tmp_path), Statuses(clock=lambda: 5.0)
    live.sets("busy", 1000)
    live.heard(statuses)
    live.writes(written("busy")[:40])  # caught half rewritten
    warnings = logged(lambda: live.heard(statuses))
    assert len(warnings) == 1 and "refused" in warnings[0]
    live.sets("busy", 1000)
    assert live.heard(statuses) == []


def test_a_status_this_version_does_not_know_is_passed_on_and_said(tmp_path: Path) -> None:
    live, statuses = Live(tmp_path), Statuses(clock=lambda: 5.0)
    live.sets("idle", 1000, status="dreaming")
    heard: list[Report] = []
    warnings = logged(lambda: heard.extend(live.heard(statuses)))
    assert [report.status for report in heard] == [Unknown("dreaming")]
    assert len(warnings) == 1 and "'dreaming'" in warnings[0]


def test_a_file_that_is_refused_is_heard_again_once_it_is_the_sessions(tmp_path: Path) -> None:
    live, statuses = Live(tmp_path), Statuses(clock=lambda: 5.0)
    live.writes(edited("idle", statusUpdatedAt=1000, sessionId="another-session"))
    assert live.heard(statuses) == []
    busy = live.sets("busy", 2000)
    assert live.heard(statuses) == [busy]


def test_each_session_is_read_only_once_the_report_before_it_is_applied(tmp_path: Path) -> None:
    """A hook applied while one session's report is (its turn's Compare awaits git) must not meet another's read before it."""
    one, two = Live(tmp_path / "one"), Live(tmp_path / "two")
    two.session = replace(two.session, membership=replace(two.member, id=SessionId("two"), pid=two.member.pid + 1))
    status_file(two.member).parent.mkdir(parents=True, exist_ok=True)
    one.sets("idle", 1000)
    two.writes(edited("idle", statusUpdatedAt=1000, pid=two.member.pid, sessionId="two"))
    held = {one.member.id: one.session, two.member.id: two.session}
    reading = Statuses(clock=lambda: 5.0).read(list(held), held.get)
    assert next(reading).session == one.member.id
    # Session two's prompt is applied in the meantime: Claude Code set busy before its hook ran.
    busy = parse_report(two.member, edited("busy", statusUpdatedAt=2000, pid=two.member.pid, sessionId="two"))
    two.writes(edited("busy", statusUpdatedAt=2000, pid=two.member.pid, sessionId="two"))
    assert [reported.report for reported in reading] == [busy]


def test_a_status_is_stamped_with_when_it_was_read(tmp_path: Path) -> None:
    live = Live(tmp_path)
    live.sets("idle", 1000)
    assert [reported.at for reported in Statuses(clock=lambda: 42.0).read([live.member.id], lambda _: live.session)] == [42.0]


def test_a_session_that_ended_while_the_one_before_was_applied_is_not_read(tmp_path: Path) -> None:
    live = Live(tmp_path)
    live.sets("idle", 1000)
    assert list(Statuses(clock=lambda: 5.0).read([live.member.id], lambda _: None)) == []
