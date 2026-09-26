"""Reading a membership file back: what it must refuse, and what it must not invent."""

import json
from pathlib import Path

import pytest

from hands.core.session import Membership, SessionId
from hands.sessions.membership import parse_membership
from hands.sessions.payload import Rejected

SID = SessionId("0f1e2d3c-aaaa-bbbb-cccc-000000000001")
RECORD = {"pid": 4242, "cwd": "/code/a", "transcript_path": "/nowhere/t.jsonl"}


def parsed(**extra: object) -> Membership:
    return parse_membership(SID, json.dumps({**RECORD, **extra}).encode())


def test_a_recorded_address_is_where_to_type_into_the_session() -> None:
    assert parsed(fritter_socket="/tmp/fritter-abc/session.sock").fritter == Path("/tmp/fritter-abc/session.sock")


@pytest.mark.parametrize("record", [{}, {"fritter_socket": ""}], ids=["missing", "empty"])
def test_no_address_is_no_address_however_it_was_left_out(record: dict[str, object]) -> None:
    # [LAW:parse-dont-validate] Path("") is PosixPath("."), which reads as a real address
    # all the way to the connect, where hands would dial a directory. An exported-but-empty
    # variable is how a shell hands on a value it does not have, so it reaches here.
    assert parsed(**record).fritter is None


def test_a_pid_no_process_could_have_is_refused_before_anything_asks_the_os() -> None:
    with pytest.raises(Rejected, match="is not a process id"):
        parse_membership(SID, json.dumps({**RECORD, "pid": 0}).encode())
