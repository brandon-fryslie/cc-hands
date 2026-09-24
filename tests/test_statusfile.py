"""Claude Code's status of a session, read from the bytes of files it wrote (2.1.282), and heard each time it is set.

The fixtures are status files copied as a live session wrote them, with only the remote-control session id blanked.
"""

import json
from pathlib import Path

import pytest
from loguru import logger
from hands.core.events import StatusReported
from hands.core.session import Membership, SessionId
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


class Session:
    """A member whose status file lives under tmp_path, rewritten as Claude Code rewrites it."""

    def __init__(self, tmp_path: Path) -> None:
        self.member = member_of(written("idle"), tmp_path)
        status_file(self.member).parent.mkdir(parents=True)

    def sets(self, name: str, stamp: int) -> Report:
        raw = edited(name, statusUpdatedAt=stamp)
        status_file(self.member).write_bytes(raw)
        return parse_report(self.member, raw)


def test_each_time_the_stamp_moves_the_status_is_heard_once(tmp_path: Path) -> None:
    session, statuses = Session(tmp_path), Statuses()
    busy = session.sets("busy", 1000)
    assert statuses.read([session.member]) == [StatusReported(session.member.id, busy)]
    assert statuses.read([session.member]) == []
    idle = session.sets("idle", 2000)
    assert statuses.read([session.member]) == [StatusReported(session.member.id, idle)]


def test_a_status_set_again_to_what_it_was_is_heard_because_its_stamp_moved(tmp_path: Path) -> None:
    # An idle, busy, idle between two reads leaves the file idle as it was, with a later stamp.
    session, statuses = Session(tmp_path), Statuses()
    session.sets("idle", 1000)
    statuses.read([session.member])
    again = session.sets("idle", 3000)
    assert statuses.read([session.member]) == [StatusReported(session.member.id, again)]


def test_a_session_with_no_status_file_is_heard_once_it_has_one(tmp_path: Path) -> None:
    member, statuses = member_of(written("idle"), tmp_path), Statuses()
    assert statuses.read([member]) == []
    assert statuses.read([member]) == []
    session = Session(tmp_path)
    idle = session.sets("idle", 1000)
    assert statuses.read([member]) == [StatusReported(member.id, idle)]


def test_why_a_status_cannot_be_read_is_said_once_not_every_read(tmp_path: Path) -> None:
    member, statuses, warnings = member_of(written("idle"), tmp_path), Statuses(), list[str]()
    sink = logger.add(lambda message: warnings.append(message.record["message"]), level="WARNING", filter="hands")
    try:
        for _ in range(3):
            statuses.read([member])
    finally:
        logger.remove(sink)
    assert len(warnings) == 1 and "keeps no status" in warnings[0]


def test_a_status_this_version_does_not_know_is_passed_on_and_said(tmp_path: Path) -> None:
    session, statuses, warnings = Session(tmp_path), Statuses(), list[str]()
    status_file(session.member).write_bytes(edited("idle", status="dreaming", statusUpdatedAt=1000))
    sink = logger.add(lambda message: warnings.append(message.record["message"]), level="WARNING", filter="hands")
    try:
        assert [reported.report.status for reported in statuses.read([session.member])] == [Unknown("dreaming")]
    finally:
        logger.remove(sink)
    assert len(warnings) == 1 and "'dreaming'" in warnings[0]


def test_a_file_that_is_refused_is_heard_again_once_it_is_the_sessions(tmp_path: Path) -> None:
    session, statuses = Session(tmp_path), Statuses()
    status_file(session.member).write_bytes(edited("idle", sessionId="another-session"))
    assert statuses.read([session.member]) == []
    busy = session.sets("busy", 1000)
    assert statuses.read([session.member]) == [StatusReported(session.member.id, busy)]


def test_a_session_that_left_and_came_back_is_heard_afresh(tmp_path: Path) -> None:
    session, statuses = Session(tmp_path), Statuses()
    idle = session.sets("idle", 1000)
    statuses.read([session.member])
    statuses.read([])
    assert statuses.read([session.member]) == [StatusReported(session.member.id, idle)]
