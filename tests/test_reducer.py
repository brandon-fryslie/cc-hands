"""The session lifecycle as a table: registry before, event, registry after, effects. No I/O."""

from dataclasses import replace
from pathlib import Path

import pytest

from hands.core.effects import (
    AfterEnd,
    Audit,
    Deny,
    Effect,
    Narrate,
    Asking,
    DeadlineNear,
    Expired,
    ModeChanged,
    Note,
    Reply,
    SessionGone,
    Speak,
    Compare,
    Snapshot,
    Summarise,
    Unregistered,
    WaitingForYou,
    Withdraw,
)
from hands.core.events import Abandoned, Attached, Died, Ended, EndReason, MovedOn, Event, Interrupted, Continued, Taken, Joined, PermissionRequested, Prompted, SessionEvent, StartSource, StatusReported, Stopped, Tick, ToolFinished, Waited
from hands.core.reducer import EXPIRED_MESSAGE, IDLE_NUDGE_SECONDS, WARNING_LEAD_SECONDS, reduce
from hands.core.session import (
    AskedQuestion,
    AtDialog,
    Option,
    Plan,
    PlanApproved,
    Question,
    Blocked,
    Gone,
    Idle,
    Membership,
    Mode,
    Permission,
    PromptId,
    Registry,
    RequestId,
    Session,
    SessionId,
    SessionState,
    Submitted,
    UnknownMode,
    Working,
)
from hands.core.status import Busy, Report, Stamp

TIMEOUT = 60.0
ONE = Membership(SessionId("s1"), pid=4242, cwd=Path("/code/a"), transcript=Path("/t/s1.jsonl"))
TWO = Membership(SessionId("s2"), pid=5353, cwd=Path("/code/b"), transcript=Path("/t/s2.jsonl"))
BASH = Permission(tool="Bash", input={"command": "ls"})

LIVE: list[SessionState] = [
    Idle(),
    Submitted(since=1.0),
    Working(since=1.0),
    Blocked(on=BASH, request=RequestId("r0"), deadline=61.0, warned=False),
]
SESSION_EVENTS: list[SessionEvent] = [
    Prompted(ONE.id, at=5.0, mode=None, prompt=None),
    Stopped(ONE.id, None, mode=None, prompt=None),
    PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), on=BASH, mode=None),
    ToolFinished(ONE.id, at=5.0, call=BASH, mode=None),
    Ended(ONE.id, "prompt_input_exit"),
    Waited(ONE.id),
    StatusReported(ONE.id, Report(Busy(), Stamp(1000))),
]


def registry(*sessions: Session) -> Registry:
    return Registry(permission_deadline=TIMEOUT, sessions={s.membership.id: s for s in sessions}, drafts={})


def holding(state: SessionState) -> Registry:
    return registry(Session(ONE, state, mode=None, turn=None))


def test_a_start_registers_the_session_idle() -> None:
    assert reduce(registry(), Joined(ONE, "startup")) == (holding(Idle()), [])


@pytest.mark.parametrize("before", LIVE)
@pytest.mark.parametrize(
    ("event", "after"),
    [
        (Stopped(ONE.id, None, mode=None, prompt=None), Idle()),
        (
            PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), on=BASH, mode=None),
            Blocked(on=BASH, request=RequestId("r1"), deadline=5.0 + TIMEOUT, warned=False),
        ),
        (Ended(ONE.id, "prompt_input_exit"), Gone()),
    ],
)
def test_every_state_takes_each_event_to_its_state(before: SessionState, event: Event, after: SessionState) -> None:
    assert reduce(holding(before), event)[0] == holding(after)


@pytest.mark.parametrize("before", LIVE)
def test_claude_codes_status_is_kept_as_it_said_it_and_moves_no_state(before: SessionState) -> None:
    report = Report(Busy(), Stamp(1000))
    assert reduce(holding(before), StatusReported(ONE.id, report)) == (registry(Session(ONE, before, mode=None, turn=None, report=report)), [])


@pytest.mark.parametrize(("before", "after"), [(Idle(), Submitted(since=5.0)), (Submitted(since=1.0), Submitted(since=5.0)), (Working(since=1.0), Working(since=5.0)), (LIVE[3], Working(since=5.0))])
def test_a_prompt_from_the_prompt_is_only_sent_and_one_inside_a_turn_is_in_it(before: SessionState, after: SessionState) -> None:
    assert reduce(holding(before), Prompted(ONE.id, at=5.0, mode=None, prompt=TURN))[0] == registry(Session(ONE, after, mode=None, turn=TURN))


def test_a_prompt_whose_hook_names_no_turn_is_working_on_the_hooks_word() -> None:
    """No record can ever be matched to it, so waiting for one would leave the session not started for its whole turn."""
    assert reduce(holding(Idle()), Prompted(ONE.id, at=5.0, mode=None, prompt=None)) == (holding(Working(since=5.0)), [Snapshot(ONE.id, ONE.cwd)])


def test_a_prompt_queued_into_the_turn_a_sent_one_opened_says_that_one_was_taken() -> None:
    """A queued prompt's hook carries the running turn's id (2.1.281): the turn is running, though its record is not read yet."""
    assert reduce(in_turn(Submitted(since=1.0)), Prompted(ONE.id, at=5.0, mode=None, prompt=TURN)) == (in_turn(Working(since=1.0)), [])


WAITING = Blocked(on=BASH, request=RequestId("r0"), deadline=61.0, warned=False)


@pytest.mark.parametrize("before", [Idle(), Working(since=1.0)])
@pytest.mark.parametrize("event", [Prompted(ONE.id, at=5.0, mode=None, prompt=None), Ended(ONE.id, "prompt_input_exit"), Joined(ONE, "startup")])
def test_moving_between_states_that_wait_on_nothing_asks_for_nothing(before: SessionState, event: Event) -> None:
    # A prompt at the prompt marks the repository it is about to change, which asks the user nothing and is
    # not spoken. A prompt arriving mid-turn opens no turn and so marks nothing: see the test below.
    marked = [Snapshot(ONE.id, ONE.cwd)] if isinstance(event, Prompted) and isinstance(before, Idle) else []
    assert reduce(holding(before), event)[1] == marked


@pytest.mark.parametrize("before", [Idle(), Working(since=1.0)])
def test_a_permission_request_is_handed_to_the_intermediary(before: SessionState) -> None:
    event = PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), on=BASH, mode=None)
    assert reduce(holding(before), event)[1] == [Narrate(Asking(ONE.id, RequestId("r1"), BASH))]


@pytest.mark.parametrize(
    "event", [Prompted(ONE.id, at=5.0, mode=None, prompt=None), Ended(ONE.id, "prompt_input_exit"), *(Joined(ONE, source) for source in ("startup", "resume", "clear"))]
)
def test_a_session_that_moves_on_while_waiting_lets_its_hook_go_undecided(event: Event) -> None:
    # Most often the user answered the dialog at the keyboard; a voice reply after that would decide nothing.
    assert reduce(holding(WAITING), event)[1] == [Reply(ONE.id, RequestId("r0"), Withdraw())]


@pytest.mark.parametrize("before", [Working(since=1.0), WAITING])
def test_a_prompt_inside_a_running_turn_leaves_the_mark_where_that_turn_began(before: SessionState) -> None:
    """A turn opens from the prompt and nowhere else, so a prompt that lands inside one opens nothing.

    Claude Code sends the hook for a queued prompt, and a Stop that the daemon never heard leaves a session
    working as far as the registry knows. Marked again at either, the turn is compared against the middle of
    its own work: everything it changed before that second prompt is missing from the one telling that names
    it, which is exactly the result no tool record would name either.
    """
    assert Snapshot(ONE.id, ONE.cwd) not in reduce(holding(before), Prompted(ONE.id, at=5.0, mode=None, prompt=None))[1]


@pytest.mark.parametrize("before", [Idle(), Working(since=1.0)])
def test_a_finished_turn_is_summarised_from_the_session_transcript(before: SessionState) -> None:
    assert reduce(holding(before), Stopped(ONE.id, "Done.", mode=None, prompt=None)) == (holding(Idle()), [Compare(ONE.id), Summarise(ONE.id, None, "Done.")])


def test_a_turn_that_finishes_while_waiting_lets_the_hook_go_and_is_summarised() -> None:
    assert reduce(holding(WAITING), Stopped(ONE.id, None, mode=None, prompt=None))[1] == [Reply(ONE.id, RequestId("r0"), Withdraw()), Compare(ONE.id), Summarise(ONE.id, None, None)]


def test_a_turn_is_compared_before_it_is_handed_over_to_be_summarised() -> None:
    """Effects are performed in the order they are given, and this order is the whole of why it is two effects.

    A summary is made one at a time and takes seconds. Read the repository when the summary is made rather
    than when the turn stopped, and it holds whatever the next turn has since started doing.
    """
    effects = reduce(holding(Working(since=1.0)), Stopped(ONE.id, None, mode=None, prompt=None))[1]
    assert effects.index(Compare(ONE.id)) < effects.index(Summarise(ONE.id, None, None))


def test_a_prompt_marks_where_the_repository_stands_before_the_turn_can_change_it() -> None:
    """The mark is what the turn's changes are read against, so it is taken as the turn opens, not during it."""
    assert reduce(holding(Idle()), Prompted(ONE.id, at=5.0, mode=None, prompt=None))[1] == [Snapshot(ONE.id, ONE.cwd)]


def test_a_second_request_while_waiting_lets_the_first_go_and_asks_the_second() -> None:
    edit = Permission(tool="Edit", input={"file_path": "a.py"})
    after, effects = reduce(holding(WAITING), PermissionRequested(ONE.id, at=7.0, request=RequestId("r1"), on=edit, mode=None))
    assert after == holding(Blocked(on=edit, request=RequestId("r1"), deadline=7.0 + TIMEOUT, warned=False))
    assert effects == [Reply(ONE.id, RequestId("r0"), Withdraw()), Narrate(Asking(ONE.id, RequestId("r1"), edit))]


def test_before_the_warning_window_a_tick_changes_nothing() -> None:
    assert reduce(holding(WAITING), Tick(at=61.0 - WARNING_LEAD_SECONDS - 0.5)) == (holding(WAITING), [])


def test_the_warning_is_spoken_once_as_the_deadline_nears() -> None:
    warned, effects = reduce(holding(WAITING), Tick(at=53.0))
    assert warned == holding(replace(WAITING, warned=True))
    assert effects == [Speak(DeadlineNear(ONE.id, BASH, remaining=8.0))]
    assert reduce(warned, Tick(at=54.0)) == (warned, [])


@pytest.mark.parametrize("warned", [True, False])
def test_at_the_deadline_the_request_is_denied_and_said_to_be(warned: bool) -> None:
    after, effects = reduce(holding(replace(WAITING, warned=warned)), Tick(at=61.0))
    assert after == holding(Working(since=61.0))
    assert effects == [Reply(ONE.id, RequestId("r0"), Deny(EXPIRED_MESSAGE)), Speak(Expired(ONE.id, BASH))]


def test_ticking_through_a_whole_wait_warns_exactly_once_then_denies_once() -> None:
    state = holding(Idle())
    state, asked = reduce(state, PermissionRequested(ONE.id, at=0.0, request=RequestId("r"), on=BASH, mode=None))
    heard: list[Effect] = [*asked]
    for second in range(1, int(TIMEOUT) + 5):
        state, effects = reduce(state, Tick(at=float(second)))
        heard += effects
    assert heard == [
        Narrate(Asking(ONE.id, RequestId("r"), BASH)),
        Speak(DeadlineNear(ONE.id, BASH, remaining=WARNING_LEAD_SECONDS)),
        Reply(ONE.id, RequestId("r"), Deny(EXPIRED_MESSAGE)),
        Speak(Expired(ONE.id, BASH)),
    ]


def test_a_tick_moves_only_sessions_waiting_on_a_deadline() -> None:
    before = registry(Session(ONE, WAITING, mode=None, turn=None), Session(TWO, Working(since=0.0), mode=None, turn=None))
    after, _ = reduce(before, Tick(at=100.0))
    assert after == registry(Session(ONE, Working(since=100.0), mode=None, turn=None), Session(TWO, Working(since=0.0), mode=None, turn=None))


@pytest.mark.parametrize("before", LIVE)
def test_a_compacted_session_keeps_its_state_and_takes_the_new_membership(before: SessionState) -> None:
    moved = replace(ONE, pid=7777)
    assert reduce(holding(before), Joined(moved, "compact")) == (registry(Session(moved, before, mode=None, turn=None)), [])


@pytest.mark.parametrize("before", [*LIVE, Gone()])
@pytest.mark.parametrize("source", ["startup", "resume", "clear"])
def test_any_start_but_compaction_is_at_the_prompt(before: SessionState, source: StartSource) -> None:
    # a session resumed after a crash never sent the Stop or SessionEnd the registry is still waiting for
    assert reduce(holding(before), Joined(ONE, source))[0] == holding(Idle())


@pytest.mark.parametrize("before", [*LIVE, Idle(due=5.0)])
def test_a_compacted_session_keeps_what_claude_code_last_said_of_it(before: SessionState) -> None:
    # Compaction keeps its process, whose status file it goes on writing: a report dropped here would not be set again.
    report = Report(Busy(), Stamp(1000))
    held = registry(Session(ONE, before, mode=None, turn=None, report=report))
    assert reduce(held, Joined(ONE, "compact"))[0].sessions[ONE.id].report == report


def test_an_ended_session_compacting_is_idle() -> None:
    assert reduce(holding(Gone()), Joined(ONE, "compact")) == (holding(Idle()), [])


@pytest.mark.parametrize("event", SESSION_EVENTS)
def test_an_ended_session_ignores_late_hooks_and_audits_them(event: SessionEvent) -> None:
    assert reduce(holding(Gone()), event) == (holding(Gone()), [Audit(AfterEnd(event)), *let_go(event)])


@pytest.mark.parametrize("event", SESSION_EVENTS)
def test_an_event_for_a_session_that_never_joined_changes_nothing_and_is_audited(event: SessionEvent) -> None:
    before = registry(Session(TWO, Idle(), mode=None, turn=None))
    assert reduce(before, event) == (before, [Audit(Unregistered(event)), *let_go(event)])


def let_go(event: SessionEvent) -> list[Effect]:
    # A permission hook the registry cannot block on is released at once instead of hanging.
    return [Reply(ONE.id, RequestId("r1"), Withdraw())] if isinstance(event, PermissionRequested) else []


def test_the_tool_a_session_waits_on_finishing_means_its_dialog_was_answered_at_the_keyboard() -> None:
    after, effects = reduce(holding(WAITING), ToolFinished(ONE.id, at=20.0, call=BASH, mode=None))
    assert (after, effects) == (holding(Working(since=20.0)), [Reply(ONE.id, RequestId("r0"), Withdraw())])


def test_a_question_answered_at_the_keyboard_comes_back_with_its_answers_and_still_releases_the_wait() -> None:
    asked = {"questions": [{"question": "Which?", "options": [{"label": "this"}]}]}
    question = Question((AskedQuestion("Which?", (Option("this", None),), several=False),), asked)
    answered = Question(question.asked, {**asked, "answers": {"Which?": "this"}})
    waiting = Blocked(on=question, request=RequestId("r0"), deadline=65.0, warned=False)
    after, effects = reduce(holding(waiting), ToolFinished(ONE.id, at=20.0, call=answered, mode=None))
    assert (after, effects) == (holding(Working(since=20.0)), [Reply(ONE.id, RequestId("r0"), Withdraw())])


@pytest.mark.parametrize("before", [Blocked(on=Plan("the plan"), request=RequestId("r0"), deadline=65.0, warned=False), AtDialog(Plan("the plan"))])
def test_a_plan_approved_at_the_keyboard_releases_the_wait(before: SessionState) -> None:
    after, _ = reduce(holding(before), ToolFinished(ONE.id, at=20.0, call=PlanApproved(), mode=None))
    assert after == holding(Working(since=20.0))


@pytest.mark.parametrize("before", [WAITING, Idle(), Working(since=1.0)])
def test_any_other_tool_finishing_changes_nothing(before: SessionState) -> None:
    other = Permission(tool="Bash", input={"command": "ls -la"})
    expected = holding(before)
    assert reduce(holding(before), ToolFinished(ONE.id, at=20.0, call=other, mode=None)) == (expected, [])


def test_a_hook_that_went_away_ends_the_wait_with_nothing_to_reply_to() -> None:
    assert reduce(holding(WAITING), Abandoned(ONE.id, RequestId("r0"), at=30.0)) == (holding(Working(since=30.0)), [])


@pytest.mark.parametrize("before", [replace(WAITING, request=RequestId("newer")), Idle(), Gone()])
def test_an_abandoned_request_the_session_no_longer_waits_on_changes_nothing(before: SessionState) -> None:
    assert reduce(holding(before), Abandoned(ONE.id, RequestId("r0"), at=30.0)) == (holding(before), [])


def test_an_event_moves_only_its_own_session() -> None:
    before = registry(Session(ONE, Idle(), mode=None, turn=None), Session(TWO, Idle(), mode=None, turn=None))
    after, _ = reduce(before, Prompted(TWO.id, at=2.0, mode=None, prompt=None))
    assert after == registry(Session(ONE, Idle(), mode=None, turn=None), Session(TWO, Working(since=2.0), mode=None, turn=None))


def test_live_is_every_session_that_has_not_ended() -> None:
    both = registry(Session(ONE, Blocked(on=BASH, request=RequestId("r"), deadline=9.0, warned=False), mode=None, turn=None), Session(TWO, Gone(), mode=None, turn=None))
    assert both.live() == [Session(ONE, Blocked(on=BASH, request=RequestId("r"), deadline=9.0, warned=False), mode=None, turn=None)]


def test_a_file_for_a_session_never_heard_of_attaches_it_at_the_prompt() -> None:
    assert reduce(registry(), Attached(ONE)) == (holding(Idle()), [])


@pytest.mark.parametrize("before", [*LIVE, Gone()])
def test_a_file_for_a_session_already_known_changes_nothing(before: SessionState) -> None:
    moved = replace(ONE, cwd=Path("/code/elsewhere"))
    assert reduce(holding(before), Attached(moved)) == (holding(before), [])


@pytest.mark.parametrize("ended", [Died(ONE), MovedOn(ONE)])
def test_a_session_this_run_never_listed_ending_says_nothing(ended: Event) -> None:
    assert reduce(registry(), ended) == (registry(), [])


@pytest.mark.parametrize("before", LIVE)
def test_a_live_session_whose_process_died_is_gone_and_spoken_and_its_hook_let_go(before: SessionState) -> None:
    released = [Reply(ONE.id, before.request, Withdraw())] if isinstance(before, Blocked) else []
    assert reduce(holding(before), Died(ONE)) == (holding(Gone()), [*released, SessionGone(ONE.id)])


def test_a_session_already_gone_dying_again_says_nothing() -> None:
    assert reduce(holding(Gone()), Died(ONE)) == (holding(Gone()), [])


def test_a_dead_process_that_is_no_longer_the_sessions_changes_nothing() -> None:
    resumed = holding(Working(since=1.0))
    assert reduce(resumed, Died(replace(ONE, pid=ONE.pid + 1))) == (resumed, [])


@pytest.mark.parametrize("before", LIVE)
def test_a_session_whose_process_moved_on_is_gone_silently_and_its_hook_let_go(before: SessionState) -> None:
    released = [Reply(ONE.id, before.request, Withdraw())] if isinstance(before, Blocked) else []
    assert reduce(holding(before), MovedOn(ONE)) == (holding(Gone()), released)


def test_a_file_on_a_pid_another_session_holds_attaches_beside_it_the_sweep_decides_which_is_over() -> None:
    both = reduce(holding(Idle()), Attached(replace(TWO, pid=ONE.pid)))[0]
    assert [session.membership.id for session in both.live()] == [ONE.id, TWO.id]


@pytest.mark.parametrize("before", LIVE)
def test_a_session_whose_terminal_closed_is_gone_and_spoken(before: SessionState) -> None:
    released = [Reply(ONE.id, before.request, Withdraw())] if isinstance(before, Blocked) else []
    assert reduce(holding(before), Ended(ONE.id, "other")) == (holding(Gone()), [*released, SessionGone(ONE.id)])


@pytest.mark.parametrize("reason", ["clear", "resume", "logout", "prompt_input_exit", "bypass_permissions_disabled"])
def test_a_session_ended_at_the_keyboard_is_not_spoken(reason: EndReason) -> None:
    assert reduce(holding(Idle()), Ended(ONE.id, reason)) == (holding(Gone()), [])


def test_a_closed_terminal_after_the_sweep_found_the_session_dead_says_nothing_more() -> None:
    event = Ended(ONE.id, "other")
    assert reduce(holding(Gone()), event) == (holding(Gone()), [Audit(AfterEnd(event))])


NUDGE = Speak(WaitingForYou(ONE.id))


def test_a_session_left_at_its_prompt_is_said_to_be_waiting() -> None:
    assert reduce(holding(Idle()), Waited(ONE.id)) == (holding(Idle(nudged=True)), [NUDGE])


def test_an_idle_notification_that_lands_after_the_prompt_it_raced_leaves_the_turn_working() -> None:
    assert reduce(holding(Working(since=5.0)), Waited(ONE.id)) == (holding(Working(since=5.0)), [])


def test_a_nudged_session_prompted_again_marks_the_repository_its_turn_starts_from() -> None:
    assert reduce(holding(Idle(nudged=True)), Prompted(ONE.id, at=5.0, mode=None, prompt=None)) == (holding(Working(since=5.0)), [Snapshot(ONE.id, ONE.cwd)])


def test_one_idle_period_is_nudged_once() -> None:
    nudged = holding(Idle(nudged=True))
    assert reduce(nudged, Waited(ONE.id)) == (nudged, [])


def test_a_session_waiting_on_a_permission_is_not_nudged_and_keeps_its_hook() -> None:
    assert reduce(holding(WAITING), Waited(ONE.id)) == (holding(WAITING), [])


@pytest.mark.parametrize("opened", [Prompted(ONE.id, at=5.0, mode=None, prompt=None), Joined(ONE, "compact"), Joined(ONE, "resume")])
def test_a_session_prompted_again_and_left_again_is_nudged_again(opened: Event) -> None:
    heard: list[Effect] = []
    state = holding(Idle())
    for event in [Waited(ONE.id), Waited(ONE.id), opened, Stopped(ONE.id, None, mode=None, prompt=None), Waited(ONE.id), Waited(ONE.id)]:
        state, effects = reduce(state, event)
        heard += [effect for effect in effects if isinstance(effect, Speak)]
    assert heard == [NUDGE, NUDGE]


REPORTING: list[SessionEvent] = [
    Prompted(ONE.id, at=5.0, mode="plan", prompt=None),
    Stopped(ONE.id, None, mode="plan", prompt=None),
    PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), on=BASH, mode="plan"),
    ToolFinished(ONE.id, at=5.0, call=BASH, mode="plan"),
]


def moded(state: SessionState, mode: Mode | None) -> Registry:
    return registry(Session(ONE, state, mode=mode, turn=None))


@pytest.mark.parametrize("before", LIVE)
@pytest.mark.parametrize("event", REPORTING)
def test_every_hook_that_reports_a_mode_sets_the_sessions_mode(before: SessionState, event: SessionEvent) -> None:
    after, effects = reduce(moded(before, "default"), event)
    assert after.sessions[ONE.id].mode == "plan"
    assert Note(ModeChanged(ONE.id, "plan")) in effects


@pytest.mark.parametrize("event", [Waited(ONE.id), Stopped(ONE.id, None, mode=None, prompt=None)])
def test_a_hook_that_reports_no_mode_keeps_the_one_last_reported(event: SessionEvent) -> None:
    after, effects = reduce(moded(Working(since=1.0), "acceptEdits"), event)
    assert after.sessions[ONE.id].mode == "acceptEdits"
    assert not [effect for effect in effects if isinstance(effect, Note)]


def test_a_mode_reported_again_unchanged_is_not_noted() -> None:
    _, effects = reduce(moded(Idle(), "plan"), Prompted(ONE.id, at=5.0, mode="plan", prompt=None))
    assert not [effect for effect in effects if isinstance(effect, Note)]


def test_a_mode_reported_for_the_first_time_is_noted_because_a_resumed_session_may_be_in_another_mode() -> None:
    after, effects = reduce(moded(Idle(), None), Prompted(ONE.id, at=5.0, mode="auto", prompt=None))
    assert after.sessions[ONE.id].mode == "auto"
    assert Note(ModeChanged(ONE.id, "auto")) in effects


def test_a_request_is_narrated_after_the_mode_it_was_asked_in_is_noted() -> None:
    _, effects = reduce(moded(Working(since=1.0), "default"), PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), on=BASH, mode="acceptEdits"))
    assert effects == [Note(ModeChanged(ONE.id, "acceptEdits")), Narrate(Asking(ONE.id, RequestId("r1"), BASH))]


def test_a_mode_hands_does_not_know_is_noted_like_any_other() -> None:
    _, effects = reduce(moded(Idle(), "default"), Prompted(ONE.id, at=5.0, mode=UnknownMode("ultraplan"), prompt=None))
    assert Note(ModeChanged(ONE.id, UnknownMode("ultraplan"))) in effects


@pytest.mark.parametrize("before", [*LIVE, Gone()])
def test_compaction_keeps_the_mode_because_it_keeps_the_process(before: SessionState) -> None:
    assert reduce(moded(before, "acceptEdits"), Joined(replace(ONE, pid=7777), "compact"))[0].sessions[ONE.id].mode == "acceptEdits"


@pytest.mark.parametrize("source", ["startup", "resume", "clear"])
def test_any_other_start_has_not_reported_a_mode(source: StartSource) -> None:
    assert reduce(moded(Idle(), "acceptEdits"), Joined(ONE, source))[0].sessions[ONE.id].mode is None


def test_a_session_keeps_its_mode_through_a_deadline_an_abandoned_hook_and_its_end() -> None:
    waiting = Blocked(on=BASH, request=RequestId("r0"), deadline=61.0, warned=False)
    assert reduce(moded(waiting, "plan"), Tick(at=100.0))[0].sessions[ONE.id].mode == "plan"
    assert reduce(moded(waiting, "plan"), Abandoned(ONE.id, RequestId("r0"), at=9.0))[0].sessions[ONE.id].mode == "plan"
    assert reduce(moded(Idle(), "plan"), Died(ONE))[0].sessions[ONE.id].mode == "plan"
    assert reduce(moded(Idle(), "plan"), Ended(ONE.id, "other"))[0].sessions[ONE.id].mode == "plan"


# An interrupt fires no hook (2.1.281): the transcript's record of it names the prompt of the turn it stopped.
TURN = PromptId("p1")
NEXT = PromptId("p2")


def in_turn(state: SessionState, turn: PromptId | None = TURN) -> Registry:
    return registry(Session(ONE, state, mode=None, turn=turn))


def test_a_prompt_names_the_turn_it_opens() -> None:
    after, _ = reduce(in_turn(Idle(), turn=None), Prompted(ONE.id, at=5.0, mode=None, prompt=TURN))
    assert after == in_turn(Submitted(since=5.0))


def test_a_prompt_claude_code_took_is_working_from_when_it_was_sent() -> None:
    assert reduce(in_turn(Submitted(since=5.0)), Taken(ONE.id, TURN, opens=True)) == (in_turn(Working(since=5.0)), [])


def test_a_prompt_cancelled_while_its_hooks_ran_never_leaves_the_session_working() -> None:
    """As seen live on 2.1.281: Escape during UserPromptSubmit puts the prompt back in the box, and nothing is written
    or sent for it after — no Stop, no record, and no idle_prompt in 90 s. Resent, it is a prompt of its own."""
    state, effects = reduce(in_turn(Idle(), turn=None), Prompted(ONE.id, at=5.0, mode=None, prompt=TURN))
    assert state == in_turn(Submitted(since=5.0))
    state, effects = reduce(state, Prompted(ONE.id, at=9.0, mode=None, prompt=NEXT))
    assert (state, effects) == (in_turn(Submitted(since=9.0, over=TURN), turn=NEXT), [Snapshot(ONE.id, ONE.cwd)])
    # The resent prompt's record, read with none of the cancelled one's before it, says that one never ran.
    assert reduce(state, Taken(ONE.id, NEXT, opens=True)) == (in_turn(Working(since=9.0), turn=NEXT), [])


def test_a_prompt_taken_and_ended_before_any_of_it_was_read_is_told_as_itself_when_its_record_is() -> None:
    """hands-keyboard-gxr.90g: p1 is taken, runs, and is interrupted, and p2's hook is applied, all inside one tail
    period, so p2 finds p1 still sent. p1's records, read after, are the proof it ran: it is ended and told as itself,
    read against the mark p2 set aside, and p2 stays sent until its own record is read."""
    state, _ = reduce(in_turn(Idle(), turn=None), Prompted(ONE.id, at=5.0, mode=None, prompt=TURN))
    state, effects = reduce(state, Prompted(ONE.id, at=6.0, mode=None, prompt=NEXT))
    assert effects == [Snapshot(ONE.id, ONE.cwd)]
    state, effects = reduce(state, Taken(ONE.id, TURN, opens=True))
    assert effects == [Compare(ONE.id, "set_aside"), Summarise(ONE.id, TURN, None)]
    assert state == registry(Session(ONE, Submitted(since=6.0), mode=None, turn=NEXT, ended=frozenset({TURN})))
    # The rest of p1's records end nothing: not its interrupt, and not a Stop hook that lands late.
    assert reduce(state, Interrupted(ONE.id, TURN, at=7.0)) == (state, [])
    assert reduce(state, Stopped(ONE.id, "Done.", mode=None, prompt=TURN)) == (state, [])
    state, effects = reduce(state, Taken(ONE.id, NEXT, opens=True))
    assert (state.sessions[ONE.id].state, effects) == (Working(since=6.0), [])
    assert reduce(state, Stopped(ONE.id, "Next.", mode=None, prompt=NEXT))[1] == [Compare(ONE.id), Summarise(ONE.id, NEXT, "Next.")]


def test_the_stop_of_a_prompt_sent_over_applied_after_the_next_prompt_ends_that_one_and_not_the_next() -> None:
    """Each hook posts from its own process, so p1's Stop can land after p2's prompt hook. It ends p1, told as itself
    against the mark p2 set aside; p2 stays sent, and p1's records read later end nothing."""
    state, _ = reduce(in_turn(Idle(), turn=None), Prompted(ONE.id, at=5.0, mode=None, prompt=TURN))
    state, _ = reduce(state, Prompted(ONE.id, at=6.0, mode=None, prompt=NEXT))
    state, effects = reduce(state, Stopped(ONE.id, "Done.", mode=None, prompt=TURN))
    assert effects == [Compare(ONE.id, "set_aside"), Summarise(ONE.id, TURN, "Done.")]
    assert state == registry(Session(ONE, Submitted(since=6.0), mode=None, turn=NEXT, ended=frozenset({TURN})))
    assert reduce(state, Taken(ONE.id, TURN, opens=True)) == (state, [])


def test_a_turn_told_as_sent_over_keeps_the_turn_ended_unheard_before_it_ended() -> None:
    """p0 is ended unheard by p1's prompt; p2 is sent over p1, and p1's record read. p0's Stop, landing later still,
    ends nothing: not p2, which is only sent."""
    state, _ = reduce(in_turn(Working(since=1.0), turn=PromptId("p0")), Prompted(ONE.id, at=5.0, mode=None, prompt=TURN))
    state, _ = reduce(state, Prompted(ONE.id, at=6.0, mode=None, prompt=NEXT))
    state, _ = reduce(state, Taken(ONE.id, TURN, opens=True))
    assert state.sessions[ONE.id].ended == {PromptId("p0"), TURN}
    assert reduce(state, Stopped(ONE.id, None, mode=None, prompt=PromptId("p0"))) == (state, [])


@pytest.mark.parametrize("over", [None, NEXT])
def test_a_record_of_another_prompt_than_the_one_sent_over_read_while_sent_ends_nothing(over: PromptId | None) -> None:
    """A transcript read for the first time holds every turn before the daemon attached: only the prompt a sent one
    was sent over, and only that, can be ended by its record."""
    sent = registry(Session(ONE, Submitted(since=6.0, over=over), mode=None, turn=PromptId("p3")))
    assert reduce(sent, Taken(ONE.id, TURN, opens=True)) == (sent, [])


def test_a_prompt_read_as_taken_before_its_hook_was_applied_is_working_when_the_hook_lands() -> None:
    """The shim gives up after its timeout and Claude Code takes the prompt; the daemon applies the hook afterwards."""
    state, effects = reduce(in_turn(Idle(), turn=None), Taken(ONE.id, TURN, opens=True))
    assert (state, effects) == (in_turn(Idle()), [])
    assert reduce(state, Prompted(ONE.id, at=5.0, mode=None, prompt=TURN)) == (in_turn(Working(since=5.0)), [Snapshot(ONE.id, ONE.cwd)])


def test_a_stop_is_told_as_the_turn_its_hook_names() -> None:
    assert reduce(in_turn(Working(since=1.0)), Stopped(ONE.id, "Done.", mode=None, prompt=TURN))[1] == [Compare(ONE.id), Summarise(ONE.id, TURN, "Done.")]


def test_the_stop_of_a_turn_a_later_prompt_ended_applied_after_it_ends_nothing() -> None:
    """It was told when the prompt ended it; applied late, it would idle the turn now running and spend its mark."""
    state, _ = reduce(in_turn(Working(since=1.0)), Prompted(ONE.id, at=9.0, mode=None, prompt=NEXT))
    after, effects = reduce(state, Stopped(ONE.id, "Done.", mode="plan", prompt=TURN))
    assert effects == [Note(ModeChanged(ONE.id, "plan"))] and after.sessions[ONE.id].state == Submitted(since=9.0)


def test_a_stop_naming_an_id_not_heard_of_yet_ends_its_turn() -> None:
    """Only a turn known to have ended is skipped: a flush's id the tail has not read yet is still the turn stopping."""
    after, effects = reduce(in_turn(Working(since=1.0)), Stopped(ONE.id, "Done.", mode=None, prompt=PromptId("q")))
    assert after.sessions[ONE.id].state == Idle() and effects == [Compare(ONE.id), Summarise(ONE.id, PromptId("q"), "Done.")]


def test_the_stop_of_a_turn_gone_on_under_a_flushed_id_ends_it_before_claude_answered_under_it() -> None:
    state, _ = reduce(in_turn(Working(since=1.0)), Taken(ONE.id, PromptId("q"), opens=False))
    assert reduce(state, Stopped(ONE.id, "Done.", mode=None, prompt=PromptId("q")))[1] == [Compare(ONE.id), Summarise(ONE.id, PromptId("q"), "Done.")]


def test_a_new_turns_first_record_read_while_the_turn_before_runs_ends_that_one_and_its_late_hook_is_working() -> None:
    """A record that opens a turn is proof the one running is over, its Stop or its interrupt not heard. The new turn's
    hook, late past the shim's timeout, then finds its prompt taken, and its turn marked."""
    state, effects = reduce(in_turn(Working(since=1.0)), Taken(ONE.id, NEXT, opens=True))
    assert effects == [Compare(ONE.id), Summarise(ONE.id, TURN, None)]
    assert state.sessions[ONE.id].state == Idle() and state.sessions[ONE.id].turn == NEXT
    state, effects = reduce(state, Prompted(ONE.id, at=9.0, mode=None, prompt=NEXT))
    assert (state.sessions[ONE.id].state, effects) == (Working(since=9.0), [Snapshot(ONE.id, ONE.cwd)])
    assert reduce(state, Stopped(ONE.id, None, mode=None, prompt=TURN))[1] == []


@pytest.mark.parametrize("state", [Working(since=1.0), WAITING, AtDialog(BASH)])
def test_a_prompt_naming_another_turn_than_the_running_one_ends_that_one_unheard(state: SessionState) -> None:
    """A queued prompt's hook carries the running turn's id (2.1.281), so this one was sent from the prompt: the turn
    before it ended with its Stop or its interrupt not read yet. It is compared before the new turn is marked."""
    withdrawn = [Reply(ONE.id, RequestId("r0"), Withdraw())] if isinstance(state, Blocked) else []
    assert reduce(in_turn(state), Prompted(ONE.id, at=9.0, mode=None, prompt=NEXT)) == (
        registry(Session(ONE, Submitted(since=9.0), mode=None, turn=NEXT, ended=frozenset({TURN}))),
        [*withdrawn, Compare(ONE.id), Summarise(ONE.id, TURN, None), Snapshot(ONE.id, ONE.cwd)],
    )


def test_the_interrupt_of_a_turn_ended_unheard_read_after_the_next_prompt_moves_nothing() -> None:
    state, _ = reduce(in_turn(Working(since=1.0)), Prompted(ONE.id, at=9.0, mode=None, prompt=NEXT))
    assert reduce(state, Interrupted(ONE.id, TURN, at=10.0)) == (state, [])
    assert reduce(state, Taken(ONE.id, NEXT, opens=True))[0].sessions[ONE.id].state == Working(since=9.0)


def test_a_message_queued_under_a_flushed_id_before_claude_answers_under_it_is_queued_into_the_turn() -> None:
    """The flush's records carry its new id at once, and Claude's first answer under it — the Continued — comes seconds
    later. A message queued in between carries the new id (2.1.281), and is queued, not the turn ended."""
    flushed = PromptId("q")
    state, effects = reduce(in_turn(Working(since=1.0)), Taken(ONE.id, flushed, opens=False))
    assert effects == [] and state.sessions[ONE.id].taken == {flushed}
    state, effects = reduce(state, Prompted(ONE.id, at=9.0, mode=None, prompt=flushed))
    assert effects == [] and state.sessions[ONE.id].state == Working(since=9.0) and state.sessions[ONE.id].turn == TURN
    assert reduce(state, Prompted(ONE.id, at=12.0, mode=None, prompt=NEXT))[1] == [Compare(ONE.id), Summarise(ONE.id, TURN, None), Snapshot(ONE.id, ONE.cwd)]
    assert reduce(state, Interrupted(ONE.id, TURN, at=12.0))[0].sessions[ONE.id].state == Idle(due=12.0 + IDLE_NUDGE_SECONDS)


def test_a_turn_a_prompt_opens_has_gone_on_under_no_other_id_yet() -> None:
    state, _ = reduce(in_turn(Working(since=1.0)), Taken(ONE.id, PromptId("q"), opens=False))
    assert reduce(state, Prompted(ONE.id, at=9.0, mode=None, prompt=NEXT))[0].sessions[ONE.id].taken == frozenset()


@pytest.mark.parametrize(("turn", "prompt"), [(TURN, TURN), (TURN, None), (None, NEXT)])
def test_a_prompt_that_cannot_be_told_from_one_queued_into_the_running_turn_is_queued_into_it(turn: PromptId | None, prompt: PromptId | None) -> None:
    """The running turn's own id is a queued prompt; with no id on either side, nothing says which turn it is."""
    assert reduce(in_turn(Working(since=1.0), turn=turn), Prompted(ONE.id, at=9.0, mode=None, prompt=prompt))[1] == []


@pytest.mark.parametrize("session", [in_turn(Submitted(since=5.0), turn=NEXT), in_turn(Working(since=5.0))])
def test_a_prompt_taken_that_is_not_the_one_sent_moves_nothing(session: Registry) -> None:
    """The first record of a queued prompt's turn, or one read after its turn ended."""
    assert reduce(session, Taken(ONE.id, TURN, opens=True)) == (session, [])


@pytest.mark.parametrize("event", [
    Stopped(ONE.id, None, mode=None, prompt=None),
    PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), on=BASH, mode=None),
    ToolFinished(ONE.id, at=5.0, call=BASH, mode=None),
    Waited(ONE.id),
])
def test_only_a_prompt_moves_the_turn(event: SessionEvent) -> None:
    """A background subagent's hooks carry the prompt_id of the turn that started it, long after that turn is over."""
    assert reduce(in_turn(Working(since=1.0)), event)[0].sessions[ONE.id].turn == TURN


def test_an_interrupted_turn_leaves_the_session_idle_and_is_told() -> None:
    assert reduce(in_turn(Working(since=1.0)), Interrupted(ONE.id, TURN, at=10.0)) == (in_turn(Idle(due=10.0 + IDLE_NUDGE_SECONDS)), [Compare(ONE.id), Summarise(ONE.id, TURN, None)])


@pytest.mark.parametrize("state", [WAITING, AtDialog(on=BASH)])
def test_a_turn_interrupted_at_its_dialog_lets_the_hook_go(state: SessionState) -> None:
    after, effects = reduce(in_turn(state), Interrupted(ONE.id, TURN, at=10.0))
    assert after == in_turn(Idle(due=10.0 + IDLE_NUDGE_SECONDS))
    assert effects == [*([Reply(ONE.id, RequestId("r0"), Withdraw())] if isinstance(state, Blocked) else []), Compare(ONE.id), Summarise(ONE.id, TURN, None)]


def test_an_interrupt_read_after_the_next_prompt_leaves_the_next_turn_running() -> None:
    """The tail reads the record a moment after it is written, by which time the user may have typed again."""
    running = in_turn(Working(since=9.0), turn=NEXT)
    assert reduce(running, Interrupted(ONE.id, TURN, at=10.0)) == (running, [])


@pytest.mark.parametrize("state", [Idle(), Idle(nudged=True), Gone()])
def test_an_interrupt_of_a_turn_already_over_changes_nothing(state: SessionState) -> None:
    before = in_turn(state)
    assert reduce(before, Interrupted(ONE.id, TURN, at=10.0)) == (before, [])


def test_a_prompt_that_names_no_turn_is_not_ended_by_the_interrupt_of_the_one_before() -> None:
    after, _ = reduce(in_turn(Idle()), Prompted(ONE.id, at=5.0, mode=None, prompt=None))
    assert reduce(after, Interrupted(ONE.id, TURN, at=10.0)) == (after, [])


def test_an_interrupt_the_registry_never_heard_open_changes_nothing() -> None:
    before = in_turn(Working(since=1.0), turn=None)
    assert reduce(before, Interrupted(ONE.id, TURN, at=10.0)) == (before, [])


def test_a_session_left_idle_after_an_interrupt_is_nudged_once_by_the_clock() -> None:
    """Claude Code sends no idle_prompt after an interrupt (2.1.281), so the nudge comes when it would have."""
    heard: list[tuple[Event, Effect]] = []
    state = in_turn(Working(since=1.0))
    for event in [Interrupted(ONE.id, TURN, at=10.0), Tick(69.0), Tick(70.0), Tick(71.0), Waited(ONE.id)]:
        state, effects = reduce(state, event)
        heard += [(event, effect) for effect in effects if isinstance(effect, Speak)]
    assert heard == [(Tick(70.0), NUDGE)]


def test_an_idle_prompt_before_the_clock_is_the_one_nudge() -> None:
    heard: list[Effect] = []
    state = in_turn(Idle(due=70.0))
    for event in [Waited(ONE.id), Tick(80.0)]:
        state, effects = reduce(state, event)
        heard += [effect for effect in effects if isinstance(effect, Speak)]
    assert heard == [NUDGE]


def test_a_prompt_before_the_clock_leaves_nothing_to_nudge() -> None:
    state, _ = reduce(in_turn(Idle(due=70.0)), Prompted(ONE.id, at=20.0, mode=None, prompt=NEXT))
    assert reduce(state, Tick(80.0))[1] == []


def test_an_idle_period_after_a_stop_waits_for_claude_codes_idle_prompt_rather_than_the_clock() -> None:
    """idle_prompt knows what no clock does: it does not come while the user is typing at the prompt."""
    assert reduce(in_turn(Idle()), Tick(10_000.0)) == (in_turn(Idle()), [])


def test_compaction_keeps_the_turn_it_happened_in() -> None:
    assert reduce(in_turn(Working(since=1.0)), Joined(ONE, "compact"))[0] == in_turn(Working(since=1.0))


def test_compaction_after_an_interrupt_keeps_the_nudge_hands_is_timing() -> None:
    assert reduce(in_turn(Idle(due=70.0)), Joined(ONE, "compact"))[0].sessions[ONE.id].state == Idle(due=70.0)


def test_a_queued_prompt_does_not_let_the_interrupt_that_flushes_it_end_the_turn_it_opens() -> None:
    """As seen live on 2.1.281: a prompt queued while a tool ran fires UserPromptSubmit at once, with the running turn's
    prompt_id. Escape then flushes it, and the interrupt record names the queued prompt's own new id, whose turn is
    already running. That turn's Stop, not the interrupt, is what ends it."""
    state = in_turn(Working(since=1.0))
    for event in [Prompted(ONE.id, at=2.0, mode=None, prompt=TURN), Interrupted(ONE.id, NEXT, at=3.0)]:
        state, _ = reduce(state, event)
    assert isinstance(state.sessions[ONE.id].state, Working)


def test_the_turn_a_queued_prompt_goes_on_as_is_ended_by_the_escape_that_stops_it() -> None:
    """No hook names the queued prompt's id: the transcript does, once Claude answers under it."""
    state = in_turn(Working(since=1.0))
    for event in [Interrupted(ONE.id, NEXT, at=3.0), Continued(ONE.id, was=TURN, now=NEXT)]:
        state, _ = reduce(state, event)
    assert state == in_turn(Working(since=1.0), turn=NEXT)
    assert reduce(state, Interrupted(ONE.id, NEXT, at=9.0))[0] == in_turn(Idle(due=9.0 + IDLE_NUDGE_SECONDS), turn=NEXT)


@pytest.mark.parametrize("session", [in_turn(Working(since=1.0), turn=PromptId("p3")), in_turn(Idle())])
def test_a_turn_read_to_have_gone_on_after_it_ended_moves_nothing(session: Registry) -> None:
    assert reduce(session, Continued(ONE.id, was=TURN, now=NEXT)) == (session, [])
