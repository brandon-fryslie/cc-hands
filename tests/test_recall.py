"""hands recall reads what was said, sent, and answered back out of the audit log, for the brain to answer from."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import get_args

import pytest

from hands.core.trace import Span
from hands.core.effects import Allow, AllowWith, Approve, Command, Deny, HookReply, Input, Key, Reply, Text, Type, Withdraw
from hands.core.events import Attached, Event, PermissionRequested, Tick
from hands.core.session import AskedQuestion, Blocker, CommandName, Membership, Option, Permission, Plan, PromptText, Question, RequestId, SessionId
from hands.daemon import cli
from hands.sessions.audit import AuditLog, Entry, Replied, Transcribed, Typing, TypingFailed, forwards, segment
from hands.sessions.home import Home
from hands.sessions.names import Finished, NameGiven, Names, NameWithheld
from hands.sessions.recall import Moment, Moments, recall
from hands.sessions.registry import Performed
from hands.sessions.wide import WideEvent
from hands.voice.naming import judge
from hands.voice.tool import tool
from hands.voice.tools import audited

MORNING = datetime(2026, 10, 3, 9, 0, tzinfo=UTC)
BILLING = SessionId("6f1c2d3e-0000-4000-8000-000000000001")


def written(home: Home, entries: Sequence[Entry]) -> list[datetime]:
    """Each entry on the log a minute after the one before it; the times they were written at."""
    times = [MORNING + timedelta(minutes=minute) for minute in range(len(entries))]
    clock = iter(times)
    log = AuditLog(home.audit, clock=lambda: next(clock))
    for entry in entries:
        log.record(entry)
    return times


# The span of the tool call that typed it.
SPAN = Span("a" * 32, "b" * 16, None)


def typed(prompt: str) -> Typing:
    return Typing(Type(BILLING, Path("/tmp/fritter/session.sock"), 42, Text(PromptText(prompt))), SPAN)


def applied(event: Event, *effects: Reply) -> WideEvent:
    """The event of the registry applying `event`, each of its effects performed."""
    return WideEvent("applied", "e" * 32, "f" * 16, None, MORNING, 1.0, "ok", None, (), {}, {"applied": event, "effects": tuple(Performed(effect, "ok", 0.5, None) for effect in effects)})


def asked(request: str, on: Blocker) -> WideEvent:
    return applied(PermissionRequested(BILLING, 1.0, RequestId(request), on, "default"))


def answered(request: str, reply: HookReply) -> WideEvent:
    return applied(Tick(2.0), Reply(BILLING, RequestId(request), reply))


def prompted(name: NameGiven | NameWithheld) -> WideEvent:
    """The event of a prompt's hook, whose reply gave the session a name hands decided, or withheld it."""
    return WideEvent("hook", "c" * 32, "d" * 16, None, MORNING, 1.0, "ok", None, (), {}, {"hook": "UserPromptSubmit", "session": BILLING, "name": name})


def test_a_sent_draft_is_recalled_by_a_word_in_it_under_the_session_it_went_to(tmp_path: Path) -> None:
    home = Home(tmp_path)
    times = written(
        home,
        [
            Transcribed("tell billing to drop the token helper and read the keychain"),
            prompted(NameGiven("billing")),
            Replied("Draft for billing: drop the token helper.", interrupted=False),
            typed("Drop the token-helper; read the token from the keychain."),
            Transcribed("what's the weather"),
        ],
    )
    found = recall(home.audit, ["TOKEN", "helper"], 20)
    assert found.moments == (
        Moment(times[0], "heard", "user", "tell billing to drop the token helper and read the keychain"),
        Moment(times[2], "said", "you", "Draft for billing: drop the token helper."),
        Moment(times[3], "sent", "sent to billing", "Drop the token-helper; read the token from the keychain."),
    )
    assert (found.lines, found.unreadable, found.since, found.found, found.matched) == (5, 0, times[0], 4, 3)


def test_the_heading_is_searched_too_and_only_the_newest_are_kept(tmp_path: Path) -> None:
    home = Home(tmp_path)
    times = written(home, [typed("first"), typed("second"), typed("third"), Transcribed("not sent")])
    # No name given yet: the session goes by its id.
    assert [moment.text for moment in recall(home.audit, ["sent", "6f1c2d3e"], 2).moments] == ["second", "third"]
    assert recall(home.audit, [], 1).moments == (Moment(times[3], "heard", "user", "not sent"),)
    assert recall(home.audit, ["nowhere"], 20).moments == ()


def test_what_was_typed_reads_as_it_was_typed(tmp_path: Path) -> None:
    home = Home(tmp_path)
    for input in (Command(CommandName("compact"), None), Command(CommandName("rc"), PromptText("on")), Key("escape")):
        AuditLog(home.audit, clock=lambda: MORNING).record(Typing(Type(BILLING, Path("/s"), 1, input), SPAN))
    assert [moment.text for moment in recall(home.audit, [], 20).moments] == ["/compact", "/rc on", "the escape key"]


def test_a_permission_answered_is_recalled_with_what_it_asked_and_a_withdrawn_one_is_not(tmp_path: Path) -> None:
    home = Home(tmp_path)
    times = written(
        home,
        [
            prompted(NameGiven("billing")),
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
        Moment(times[2], "answered", "allowed in billing", 'Bash {"command": "rm src/token_helper.py"}'),
        Moment(times[4], "answered", "denied in billing", 'Write {"file_path": "/x"}, told: Nobody answered in time.'),
    )


def test_a_torn_line_is_counted_and_the_lines_after_it_are_read(tmp_path: Path) -> None:
    home = Home(tmp_path)
    AuditLog(home.audit, clock=lambda: MORNING).record(Transcribed("before"))
    with segment(home.audit, 0).open("a", encoding="utf-8") as torn:
        torn.write('{"at": "2026-10-03T00:00:00.000+00:00", "level": "in\n')
        # JSON, but no line the log writes.
        torn.write("[1]\n")
    AuditLog(home.audit, clock=lambda: MORNING).record(Transcribed("after"))
    found = recall(home.audit, [], 20)
    assert [moment.text for moment in found.moments] == ["before", "after"]
    assert (found.lines, found.unreadable) == (4, 2)


async def test_a_session_goes_by_the_name_its_judging_found_it_with(tmp_path: Path) -> None:
    home = Home(tmp_path)
    transcript = tmp_path / "billing.jsonl"
    transcript.write_text(json.dumps({"type": "custom-title", "customTitle": "billing", "sessionId": BILLING}, separators=(",", ":")) + "\n")
    log = AuditLog(home.audit, clock=lambda: MORNING)

    async def keeps(_text: str) -> str:
        return "billing"

    await judge(Finished(Membership(BILLING, pid=1, cwd=tmp_path, transcript=transcript), "done"), Names(), (), keeps, log.record)
    log.record(typed("Drop the token-helper."))
    assert [moment.heading for moment in recall(home.audit, [], 20).moments] == ["sent to billing"]


def test_hands_recall_prints_one_line_a_moment_in_local_time_and_records_what_it_took(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path)
    times = written(home, [Transcribed("drop the\ntoken helper"), Transcribed("something else")])
    assert cli.main(["--home", str(tmp_path), "recall", "token"]) == 0
    assert capsys.readouterr().out == f"The log reaches back to {times[0].astimezone():%a %d %b %H:%M}.\n{times[0].astimezone():%a %d %b %H:%M} user: drop the token helper\n"
    # [LAW:nothing-unseen] the recall's own event, last on the log it read, with what it was asked and what it found.
    [event] = [line for line in (json.loads(line) for line in segment(home.audit, 0).read_text().splitlines()) if line.get("event") == "memory.recall"]
    assert (event["event"], event["outcome"], event["facts"]) == ("memory.recall", "ok", {"words": ["token"], "most": 20, "since": times[0].isoformat(timespec="milliseconds")})
    assert event["counts"] == {"lines": 2, "unreadable": 0, "moments": 2, "matched": 1, "printed": 1}


def test_hands_recall_on_a_home_with_no_log_says_so_in_its_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--home", str(tmp_path), "recall"]) == 0
    assert capsys.readouterr().out == "The log is empty.\n"
    # The recall's own event, read before the command's was written.
    [event, _] = [json.loads(line) for line in segment(Home(tmp_path).audit, 0).read_text().splitlines()]
    assert event["counts"] == {"lines": 0, "unreadable": 0, "moments": 0, "matched": 0, "printed": 0}


def test_a_send_goes_by_the_name_its_session_was_given_after_it_and_one_that_failed_says_so(tmp_path: Path) -> None:
    home = Home(tmp_path)
    failed = typed("second")
    times = written(
        home,
        [
            typed("first"),
            failed,
            TypingFailed(failed.effect, "cannot talk to the fritter"),
            # The name a turn's end decided is given in the reply to the prompt after it; the user's /rename outranks it.
            prompted(NameGiven("billing")),
            prompted(NameWithheld("billing work", against="billing", held="payments")),
        ],
    )
    assert recall(home.audit, [], 20).moments == (
        Moment(times[0], "sent", "sent to payments", "first"),
        Moment(times[1], "sent", "not sent to payments", "second, because: cannot talk to the fritter"),
    )


def test_an_answered_question_is_recalled_with_what_the_user_chose(tmp_path: Path) -> None:
    home = Home(tmp_path)
    options = tuple(Option(f"option {number}", "a long description " * 20) for number in range(4))
    question = Question((AskedQuestion("Which database?", options, several=False), AskedQuestion("Which ORM?", options, several=False)), {"questions": ["a long echoed input " * 40]})
    times = written(
        home,
        [
            prompted(NameGiven("billing")),
            asked("q", question),
            answered("q", AllowWith({**question.input, "answers": {"Which database?": "Postgres", "Which ORM?": "SQLAlchemy"}})),
        ],
    )
    assert recall(home.audit, [], 20).moments == (Moment(times[2], "answered", "answered in billing", "Which database? Which ORM?, chose: Postgres; SQLAlchemy"),)


# Every variant of what a Typing, PermissionRequested, or Reply line can hold, so a variant added to a union fails here
# rather than in a recall that meets its first line.
INPUTS: tuple[Input, ...] = (Text(PromptText("p")), Command(CommandName("c"), None), Key("escape"))
BLOCKERS: tuple[Blocker, ...] = (Permission("Bash", {}), Question((), {}), Plan("p"))
REPLIES: tuple[HookReply, ...] = (Allow(), AllowWith({"answers": {}}), Approve("default"), Deny("no"), Withdraw())


def test_recall_reads_every_input_request_and_reply_the_log_can_hold(tmp_path: Path) -> None:
    assert {type(each) for each in INPUTS} == set(get_args(Input))
    assert {type(each) for each in BLOCKERS} == set(get_args(Blocker))
    assert {type(each) for each in REPLIES} == set(get_args(HookReply))
    home = Home(tmp_path)
    written(
        home,
        [*(Typing(Type(BILLING, Path("/s"), 1, input), SPAN) for input in INPUTS)]
        + [line for number, (on, reply) in enumerate(zip(BLOCKERS * 2, REPLIES)) for line in (asked(f"r{number}", on), answered(f"r{number}", reply))],
    )
    # The Withdraw decided nothing, so it is the one reply with no moment.
    assert len(recall(home.audit, [], 20).moments) == len(INPUTS) + len(REPLIES) - 1


def test_a_session_hands_saw_join_goes_by_its_project_as_hands_speaks_it_and_a_silent_reply_is_no_moment(tmp_path: Path) -> None:
    home = Home(tmp_path)
    times = written(
        home,
        [
            applied(Attached(Membership(BILLING, 42, Path("/code/home-infra"), Path("/t.jsonl")))),
            typed("plan the atlantis wait"),
            Replied("", interrupted=False),
            prompted(NameGiven("atlantis plan wait")),
        ],
    )
    assert recall(home.audit, ["home-infra"], 20).moments == (Moment(times[1], "sent", "sent to home-infra, atlantis plan wait", "plan the atlantis wait"),)
    assert recall(home.audit, [], 20).found == 1


async def test_a_tool_hands_called_is_a_moment_of_the_conversation_and_no_part_of_what_is_recalled(tmp_path: Path) -> None:
    home = Home(tmp_path)

    async def set_trigger(trigger: str) -> dict[str, object]:
        return {"trigger": trigger}

    clock = iter(MORNING + timedelta(minutes=minute) for minute in range(3))
    log = AuditLog(home.audit, clock=lambda: next(clock))
    log.record(Transcribed("switch to the wake word"))
    await audited(tool(set_trigger), log.record).body(trigger="wake word")
    log.record(Replied("Okay.", False))
    assert [moment.kind for moment in folded(home, Moments()).moments()] == ["heard", "called", "said"]
    assert [moment.kind for moment in folded(home, Moments(2)).moments()] == ["called", "said"]
    assert [moment.text for moment in recall(home.audit, [], 20).moments] == ["switch to the wake word", "Okay."]


def test_a_send_seen_to_fail_is_a_change_to_the_moment_it_was(tmp_path: Path) -> None:
    home = Home(tmp_path)
    written(home, [typed("first"), TypingFailed(typed("first").effect, "cannot talk to the fritter")])
    taken = folded(home, Moments())
    assert taken.changes == 2
    assert [moment.heading for moment in taken.moments()] == [f"not sent to {BILLING}"]


def test_only_the_newest_kept_are_held_and_a_send_no_longer_kept_fails_unseen(tmp_path: Path) -> None:
    home = Home(tmp_path)
    written(home, [typed("first"), typed("second"), typed("third"), TypingFailed(typed("first").effect, "gone")])
    taken = folded(home, Moments(2))
    assert [moment.heading for moment in taken.moments()] == [f"sent to {BILLING}"] * 2
    assert taken.changes == 3


def test_a_session_named_after_its_send_is_a_change_and_one_named_again_the_same_is_not(tmp_path: Path) -> None:
    home = Home(tmp_path)
    written(home, [typed("first")])
    before = folded(home, Moments()).changes
    written(home, [prompted(NameGiven("payments")), prompted(NameGiven("payments"))])
    taken = folded(home, Moments())
    assert taken.changes == before + 1
    assert [moment.heading for moment in taken.moments()] == ["sent to payments"]


def folded(home: Home, moments: Moments) -> Moments:
    """`moments` with every line of `home`'s log taken."""
    for line in forwards(home.audit):
        moments.take(json.loads(line))
    return moments
