"""Answering a permission as a table: registry before, answer, and what came of it. No I/O."""

from pathlib import Path

import pytest

from hands.core.effects import Allow, AllowWith, Answers, Approve, Decision, Deny, KeepPlanning, Reply
from hands.core.permissions import Answer, Answered, NotWaiting, Unfit, answer, heard
from hands.core.session import AskedQuestion, Blocker, Gone, Held, Idle, Membership, Option, Permission, Plan, Question, Registry, RequestId, Running, Session, SessionId
from hands.core import status
from hands.core.status import Busy, Going, Stamp, Waiting

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


def running(going: Going = Busy()) -> Running:
    return Running(going, Stamp(1), idled=Stamp(1))


AT_DIALOG = running(Waiting("permission prompt"))


IDLE = Idle(status.Idle(), Stamp(1), after=None)


def registry(*sessions: Session) -> Registry:
    return Registry(permission_deadline=60.0, sessions={s.membership.id: s for s in sessions}, drafts={})


@pytest.mark.parametrize("decision", [Allow(), Deny("use git clean instead")])
def test_an_answer_replies_to_the_waiting_hook_and_the_turn_carries_on(decision: Allow | Deny) -> None:
    before = registry(Session(ONE, AT_DIALOG, mode=None, dialog=Held(on=BASH, request=REQUEST, deadline=61.0, warned=True, continues=None)), Session(TWO, IDLE, mode=None))
    assert answer(before, Answer(REQUEST, decision)) == (
        registry(Session(ONE, AT_DIALOG, mode=None), Session(TWO, IDLE, mode=None)),
        Answered(ONE.id, BASH, decision),
        [Reply(ONE.id, REQUEST, decision)],
    )


@pytest.mark.parametrize(
    "session",
    [
        Session(ONE, IDLE, mode=None),
        Session(ONE, running(), mode=None),
        Gone(ONE),
        Session(ONE, AT_DIALOG, mode=None, dialog=Held(on=BASH, request=RequestId("another"), deadline=61.0, warned=False, continues=None)),
    ],
)
def test_an_answer_to_a_request_nobody_waits_on_changes_nothing(session: Session) -> None:
    before = registry(session)
    assert answer(before, Answer(REQUEST, Allow())) == (before, NotWaiting(REQUEST), [])


def test_a_request_is_answered_once() -> None:
    once, _, _ = answer(registry(Session(ONE, AT_DIALOG, mode=None, dialog=Held(on=BASH, request=REQUEST, deadline=61.0, warned=False, continues=None))), Answer(REQUEST, Allow()))
    assert answer(once, Answer(REQUEST, Deny("no"))) == (once, NotWaiting(REQUEST), [])


def waiting_on(on: Blocker) -> Registry:
    return registry(Session(ONE, AT_DIALOG, mode=None, dialog=Held(on=on, request=REQUEST, deadline=61.0, warned=False, continues=None)))


def test_answers_reach_the_question_keyed_by_what_each_asked_with_its_input_kept() -> None:
    chosen = Answers(("SQLite", "lint, test"))
    assert answer(waiting_on(QUESTION), Answer(REQUEST, chosen)) == (
        registry(Session(ONE, AT_DIALOG, mode=None)),
        Answered(ONE.id, QUESTION, chosen),
        [Reply(ONE.id, REQUEST, AllowWith({**ASKED, "answers": {"Which database?": "SQLite", "Which checks?": "lint, test"}}))],
    )


def test_a_question_can_be_refused_as_a_permission_is() -> None:
    assert answer(waiting_on(QUESTION), Answer(REQUEST, Deny("ask me later")))[2] == [Reply(ONE.id, REQUEST, Deny("ask me later"))]


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
    assert answer(before, Answer(REQUEST, decision)) == (before, Unfit(REQUEST, on, decision), [])


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
    assert answer(waiting_on(PLAN), Answer(REQUEST, decision)) == (
        registry(Session(ONE, AT_DIALOG, mode=None)),
        Answered(ONE.id, PLAN, decision),
        [Reply(ONE.id, REQUEST, reply)],
    )


@pytest.mark.parametrize("words", ["Yes.", "yeah", "Yep!", "Okay, go ahead.", "Sure, do it.", "Go for it.", "All right.", "Yes please", "Sounds good.", "That’s fine."])
def test_a_plain_yes_spoken_to_a_permission_allows_it(words: str) -> None:
    assert heard(words) == Allow()


@pytest.mark.parametrize("words", ["No.", "Don't.", "Yes, but put it in the other file.", "Wait.", "Go ahead and delete everything else too.", "Please.", "", "Not yet.", "Yes 2."])
def test_anything_else_spoken_to_a_permission_refuses_it_carrying_the_words(words: str) -> None:
    # A yes with more said beside it is the brain's to read: it may want something other than what was asked.
    assert heard(words) == Deny(f'The user was asked whether to allow this, and answered by voice, so it did not run. Do what they said, and tell them aloud that it did not run and what that leaves undone. They said: "{words}"')
