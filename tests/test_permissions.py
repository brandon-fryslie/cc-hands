"""Answering a permission as a table: registry before, answer, and what came of it. No I/O."""

from pathlib import Path

import pytest

from hands.core.effects import Allow, AllowWith, Answers, Approve, Decision, Deny, KeepPlanning, Reply
from hands.core.permissions import Answer, Answered, NotWaiting, Unfit, answer
from hands.core.session import AskedQuestion, Blocked, Blocker, Gone, Idle, Membership, Option, Permission, Plan, Question, Registry, RequestId, Session, SessionId, SessionState, Working

ONE = Membership(SessionId("s1"), pid=1, cwd=Path("/code/a"), transcript=Path("/t/s1.jsonl"))
TWO = Membership(SessionId("s2"), pid=2, cwd=Path("/code/b"), transcript=Path("/t/s2.jsonl"))
BASH = Permission(tool="Bash", input={"command": "rm -r build"})
REQUEST = RequestId("r1")
ASKED = {
    "questions": [
        {"question": "Which database?", "header": "DB", "options": [{"label": "Postgres", "description": "relational"}, {"label": "SQLite", "description": "a file"}], "multiSelect": False},
        {"question": "Which checks?", "header": "CI", "options": [{"label": "lint", "description": ""}, {"label": "test", "description": ""}], "multiSelect": True},
    ]
}
QUESTION = Question(
    (
        AskedQuestion("Which database?", (Option("Postgres", "relational"), Option("SQLite", "a file")), several=False),
        AskedQuestion("Which checks?", (Option("lint", ""), Option("test", "")), several=True),
    ),
    ASKED,
)
PLAN = Plan("1. Create hello.txt.\n2. Write hi into it.")


def registry(*sessions: Session) -> Registry:
    return Registry(permission_deadline=60.0, sessions={s.membership.id: s for s in sessions}, drafts={})


@pytest.mark.parametrize("decision", [Allow(), Deny("use git clean instead")])
def test_an_answer_replies_to_the_waiting_hook_and_the_turn_carries_on(decision: Allow | Deny) -> None:
    before = registry(Session(ONE, Blocked(on=BASH, request=REQUEST, deadline=61.0, warned=True)), Session(TWO, Idle()))
    assert answer(before, Answer(REQUEST, decision, at=20.0)) == (
        registry(Session(ONE, Working(since=20.0)), Session(TWO, Idle())),
        Answered(ONE.id, BASH, decision),
        [Reply(ONE.id, REQUEST, decision)],
    )


@pytest.mark.parametrize(
    "state",
    [Idle(), Working(since=1.0), Gone(), Blocked(on=BASH, request=RequestId("another"), deadline=61.0, warned=False)],
)
def test_an_answer_to_a_request_nobody_waits_on_changes_nothing(state: SessionState) -> None:
    before = registry(Session(ONE, state))
    assert answer(before, Answer(REQUEST, Allow(), at=20.0)) == (before, NotWaiting(REQUEST), [])


def test_a_request_is_answered_once() -> None:
    once, _, _ = answer(registry(Session(ONE, Blocked(on=BASH, request=REQUEST, deadline=61.0, warned=False))), Answer(REQUEST, Allow(), at=2.0))
    assert answer(once, Answer(REQUEST, Deny("no"), at=3.0)) == (once, NotWaiting(REQUEST), [])


def waiting_on(on: Blocker) -> Registry:
    return registry(Session(ONE, Blocked(on=on, request=REQUEST, deadline=61.0, warned=False)))


def test_answers_reach_the_question_keyed_by_what_each_asked_with_its_input_kept() -> None:
    chosen = Answers(("SQLite", "lint, test"))
    assert answer(waiting_on(QUESTION), Answer(REQUEST, chosen, at=20.0)) == (
        registry(Session(ONE, Working(since=20.0))),
        Answered(ONE.id, QUESTION, chosen),
        [Reply(ONE.id, REQUEST, AllowWith({**ASKED, "answers": {"Which database?": "SQLite", "Which checks?": "lint, test"}}))],
    )


def test_a_question_can_be_refused_as_a_permission_is() -> None:
    assert answer(waiting_on(QUESTION), Answer(REQUEST, Deny("ask me later"), at=20.0))[2] == [Reply(ONE.id, REQUEST, Deny("ask me later"))]


@pytest.mark.parametrize(
    ("on", "decision"),
    [
        (QUESTION, Answers(("SQLite",))),
        (QUESTION, Answers(("SQLite", "lint", "test"))),
        # Allowed with no answers, AskUserQuestion runs as unanswered.
        (QUESTION, Allow()),
        (BASH, Answers(("yes",))),
        # Allowed as a permission, ExitPlanMode leaves plan mode for a mode nobody chose.
        (PLAN, Allow()),
        (PLAN, Answers(("yes",))),
        (BASH, Approve("acceptEdits")),
        (QUESTION, Approve("default")),
        # Feedback on a plan, sent against the wrong request, must not refuse a tool with it.
        (BASH, KeepPlanning("split step 2")),
        (QUESTION, KeepPlanning("split step 2")),
    ],
)
def test_a_decision_that_does_not_answer_what_was_asked_sends_nothing_and_the_session_still_waits(on: Blocker, decision: Decision) -> None:
    before = waiting_on(on)
    assert answer(before, Answer(REQUEST, decision, at=20.0)) == (before, Unfit(REQUEST, on, decision), [])


@pytest.mark.parametrize(
    ("decision", "reply"),
    [
        (Approve("resume"), Approve("resume")),
        (Approve("acceptEdits"), Approve("acceptEdits")),
        (KeepPlanning("split step 2 in two"), Deny("split step 2 in two")),
        (Deny("not now"), Deny("not now")),
    ],
)
def test_a_plan_is_approved_for_the_mode_chosen_or_sent_back_to_planning(decision: Decision, reply: Approve | Deny) -> None:
    assert answer(waiting_on(PLAN), Answer(REQUEST, decision, at=20.0)) == (
        registry(Session(ONE, Working(since=20.0))),
        Answered(ONE.id, PLAN, decision),
        [Reply(ONE.id, REQUEST, reply)],
    )
