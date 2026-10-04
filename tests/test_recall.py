"""hands recall reads what was said, sent, and answered back out of the audit log, for the brain to answer from."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hands.core.effects import Allow, Command, Deny, Key, Reply, Text, Type, Withdraw
from hands.core.events import PermissionRequested
from hands.core.session import CommandName, Permission, Plan, PromptText, RequestId, SessionId
from hands.daemon import cli
from hands.sessions.audit import Applied, AuditLog, Entry, NameGiven, Performed, Replied, Transcribed, Typing, segment
from hands.sessions.home import Home
from hands.sessions.recall import Moment, recall

MORNING = datetime(2026, 10, 3, 9, 0, tzinfo=UTC)
BILLING = SessionId("6f1c2d3e-0000-4000-8000-000000000001")


def written(home: Home, entries: list[Entry]) -> list[datetime]:
    """Each entry on the log a minute after the one before it; the times they were written at."""
    times = [MORNING + timedelta(minutes=minute) for minute in range(len(entries))]
    clock = iter(times)
    log = AuditLog(home.audit, clock=lambda: next(clock))
    for entry in entries:
        log.record(entry)
    return times


def typed(prompt: str) -> Typing:
    return Typing(Type(BILLING, Path("/tmp/fritter/session.sock"), 42, Text(PromptText(prompt))))


def asked(request: str, on: Permission | Plan) -> Applied:
    return Applied(PermissionRequested(BILLING, 1.0, RequestId(request), on, "default"))


def answered(request: str, reply: Allow | Deny | Withdraw) -> Performed:
    return Performed(Reply(BILLING, RequestId(request), reply))


def test_a_sent_draft_is_recalled_by_a_word_in_it_under_the_session_it_went_to(tmp_path: Path) -> None:
    home = Home(tmp_path)
    times = written(
        home,
        [
            Transcribed("tell billing to drop the token helper and read the keychain"),
            NameGiven(BILLING, "billing"),
            Replied("Draft for billing: drop the token helper.", interrupted=False),
            typed("Drop the token-helper; read the token from the keychain."),
            Transcribed("what's the weather"),
        ],
    )
    found = recall(home.audit, ["TOKEN", "helper"], 20)
    assert found.moments == (
        Moment(times[0], "user", "tell billing to drop the token helper and read the keychain"),
        Moment(times[2], "you", "Draft for billing: drop the token helper."),
        Moment(times[3], "sent to billing", "Drop the token-helper; read the token from the keychain."),
    )
    assert (found.lines, found.unreadable, found.found, found.matched) == (5, 0, 4, 3)


def test_the_heading_is_searched_too_and_only_the_newest_are_kept(tmp_path: Path) -> None:
    home = Home(tmp_path)
    times = written(home, [typed("first"), typed("second"), typed("third"), Transcribed("not sent")])
    # No name given yet: the session goes by its id.
    assert [moment.text for moment in recall(home.audit, ["sent", "6f1c2d3e"], 2).moments] == ["second", "third"]
    assert recall(home.audit, [], 1).moments == (Moment(times[3], "user", "not sent"),)
    assert recall(home.audit, ["nowhere"], 20).moments == ()


def test_what_was_typed_reads_as_it_was_typed(tmp_path: Path) -> None:
    home = Home(tmp_path)
    for input in (Command(CommandName("compact"), None), Command(CommandName("rc"), PromptText("on")), Key("escape")):
        AuditLog(home.audit, clock=lambda: MORNING).record(Typing(Type(BILLING, Path("/s"), 1, input)))
    assert [moment.text for moment in recall(home.audit, [], 20).moments] == ["/compact", "/rc on", "the escape key"]


def test_a_permission_answered_is_recalled_with_what_it_asked_and_a_withdrawn_one_is_not(tmp_path: Path) -> None:
    home = Home(tmp_path)
    times = written(
        home,
        [
            NameGiven(BILLING, "billing"),
            asked("r1", Permission("Bash", {"command": "rm src/token_helper.py"})),
            answered("r1", Allow()),
            asked("r2", Permission("Write", {"file_path": "/x"})),
            answered("r2", Deny("Nobody answered in time.")),
            asked("r3", Plan("Move the token into the keychain.")),
            answered("r3", Withdraw()),
            # A reply to a hook no request was seen for, as a Stop's is, is no answer to recall.
            answered("stop", Allow()),
        ],
    )
    assert recall(home.audit, [], 20).moments == (
        Moment(times[2], "allowed in billing", 'Bash {"command": "rm src/token_helper.py"}'),
        Moment(times[4], "denied in billing", 'Write {"file_path": "/x"}, told: Nobody answered in time.'),
    )


def test_a_torn_line_is_counted_and_the_lines_after_it_are_read(tmp_path: Path) -> None:
    home = Home(tmp_path)
    AuditLog(home.audit, clock=lambda: MORNING).record(Transcribed("before"))
    with segment(home.audit, 0).open("a", encoding="utf-8") as torn:
        torn.write('{"at": "2026-10-03T00:00:00.000+00:00", "level": "in\n')
    AuditLog(home.audit, clock=lambda: MORNING).record(Transcribed("after"))
    found = recall(home.audit, [], 20)
    assert [moment.text for moment in found.moments] == ["before", "after"]
    assert (found.lines, found.unreadable) == (3, 1)


def test_hands_recall_prints_one_line_a_moment_in_local_time_and_records_what_it_took(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path)
    times = written(home, [Transcribed("drop the\ntoken helper"), Transcribed("something else")])
    assert cli.main(["--home", str(tmp_path), "recall", "token"]) == 0
    assert capsys.readouterr().out == f"{times[0].astimezone():%a %d %b %H:%M} user: drop the token helper\n"
    # [LAW:nothing-unseen] the recall's own event, last on the log it read, with what it was asked and what it found.
    [event] = [line for line in (json.loads(line) for line in segment(home.audit, 0).read_text().splitlines()) if line["type"] == "WideEvent"]
    assert (event["event"], event["outcome"], event["facts"]) == ("memory.recall", "ok", {"words": ["token"], "most": 20})
    assert event["counts"] == {"lines": 2, "unreadable": 0, "moments": 2, "matched": 1, "printed": 1}


def test_hands_recall_on_a_home_with_no_log_says_so_in_its_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--home", str(tmp_path), "recall"]) == 0
    assert capsys.readouterr().out == ""
    [event] = [json.loads(line) for line in segment(Home(tmp_path).audit, 0).read_text().splitlines()]
    assert event["counts"] == {"lines": 0, "unreadable": 0, "moments": 0, "matched": 0, "printed": 0}
