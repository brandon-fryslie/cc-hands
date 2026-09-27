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
from hands.core.reducer import EXPIRED_MESSAGE, IDLE_NUDGE_SECONDS, UNTOLD_SECONDS, WARNING_LEAD_SECONDS, reduce
from hands.core.session import (
    AskedQuestion,
    Dialog,
    Gone,
    Held,
    Idle,
    LetGo,
    Membership,
    Mode,
    Opened,
    Option,
    Permission,
    Plan,
    PlanApproved,
    PromptId,
    Question,
    Registry,
    RequestId,
    Running,
    Session,
    SessionId,
    SessionState,
    Told,
    Turn,
    Unanswered,
    UnknownMode,
    Unreported,
    Untold,
)
from hands.core.status import Busy, Report, Shell, Stamp, Status, Unknown, Waiting
from hands.core import status

TIMEOUT = 60.0
ONE = Membership(SessionId("s1"), pid=4242, cwd=Path("/code/a"), transcript=Path("/t/s1.jsonl"))
TWO = Membership(SessionId("s2"), pid=5353, cwd=Path("/code/b"), transcript=Path("/t/s2.jsonl"))
BASH = Permission(tool="Bash", input={"command": "ls"})
TURN = PromptId("p1")
NEXT = PromptId("p2")

IDLE = Idle(Stamp(900), due=70.0, after=None)
BUSY = Running(Busy(), Stamp(1000), idled=Stamp(1000))
AT_DIALOG = Running(Waiting("permission prompt"), Stamp(1100), idled=Stamp(1000))
HELD = Held(on=BASH, request=RequestId("r0"), deadline=61.0, warned=False)

LIVE: list[SessionState] = [Unreported(), IDLE, BUSY, AT_DIALOG]
SESSION_EVENTS: list[SessionEvent] = [
    Prompted(ONE.id, at=5.0, mode=None, prompt=TURN),
    Stopped(ONE.id, None, mode=None, prompt=TURN, again=False),
    PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), on=BASH, mode=None),
    ToolFinished(ONE.id, at=5.0, call=BASH, mode=None),
    Ended(ONE.id, "prompt_input_exit"),
    Waited(ONE.id),
    StatusReported(ONE.id, Report(Busy(), Stamp(1000)), at=5.0),
]
# What hooks and the transcript say: which turn it is, what it did, and which request it waits on.
HEARD: list[SessionEvent] = [
    Prompted(ONE.id, at=5.0, mode=None, prompt=NEXT),
    Stopped(ONE.id, "Done.", mode=None, prompt=TURN, again=False),
    Stopped(ONE.id, "Again.", mode=None, prompt=TURN, again=True),
    Taken(ONE.id, NEXT, Stamp(5000), at=5.0),
    Interrupted(ONE.id, TURN, at=5.0),
    Continued(ONE.id, was=TURN, now=NEXT),
    PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), on=BASH, mode=None),
    ToolFinished(ONE.id, at=5.0, call=BASH, mode=None),
]


def registry(*sessions: Session) -> Registry:
    return Registry(permission_deadline=TIMEOUT, sessions={s.membership.id: s for s in sessions}, drafts={})


def holding(state: SessionState, turn: Turn = Told(), dialog: Dialog | None = None) -> Registry:
    return registry(Session(ONE, state, mode=None, turn=turn, dialog=dialog))


def in_turn(state: SessionState = BUSY, dialog: Dialog | None = None, turn: PromptId = TURN) -> Registry:
    return holding(state, Opened(turn), dialog)


def said(status: Status, stamp: int = 2000, at: float = 10.0) -> StatusReported:
    return StatusReported(ONE.id, Report(status, Stamp(stamp)), at=at)


def test_a_start_registers_the_session_with_no_status_read_yet() -> None:
    assert reduce(registry(), Joined(ONE, "startup")) == (holding(Unreported()), [])


@pytest.mark.parametrize("before", LIVE)
@pytest.mark.parametrize("turn", [Told(TURN), Opened(TURN), Untold(TURN, frozenset(), by=99.0, asking=False)])
@pytest.mark.parametrize("event", HEARD)
def test_no_hook_or_record_moves_a_session_between_running_and_not(before: SessionState, turn: Turn, event: SessionEvent) -> None:
    """[LAW:one-source-of-truth] hooks and records say which turn it is and what it did, never whether it runs."""
    assert reduce(holding(before, turn), event)[0].sessions[ONE.id].state == before


@pytest.mark.parametrize(
    ("before", "event", "after"),
    [
        (Unreported(), said(Busy(), 3000), Running(Busy(), Stamp(3000), idled=Stamp(3000))),
        # Running since the idle before it: a record written after that idle is of this run.
        (IDLE, said(Busy(), 3000), Running(Busy(), Stamp(3000), idled=IDLE.stamp)),
        (IDLE, said(Shell(), 3000), Running(Shell(), Stamp(3000), idled=IDLE.stamp)),
        (BUSY, said(Waiting("permission prompt"), 3000), Running(Waiting("permission prompt"), Stamp(3000), idled=BUSY.idled)),
        (AT_DIALOG, said(Busy(), 3000), Running(Busy(), Stamp(3000), idled=AT_DIALOG.idled)),
        (BUSY, said(Unknown("dreaming"), 3000), Running(Unknown("dreaming"), Stamp(3000), idled=BUSY.idled)),
        (BUSY, said(status.Idle(), 3000, at=10.0), Idle(Stamp(3000), due=10.0 + IDLE_NUDGE_SECONDS, after=None)),
        (Unreported(), said(status.Idle(), 3000, at=10.0), Idle(Stamp(3000), due=None, after=None)),
        # Set idle again, or idle, busy and idle between two reads: the same idle period, still timed and nudged as it was.
        (replace(IDLE, nudged=True), said(status.Idle(), 3000, at=10.0), replace(IDLE, stamp=Stamp(3000), nudged=True)),
    ],
)
def test_a_status_read_moves_the_session_to_what_claude_code_says_it_is_doing(before: SessionState, event: StatusReported, after: SessionState) -> None:
    assert reduce(holding(before), event) == (holding(after), [])


@pytest.mark.parametrize("before", LIVE)
def test_an_end_is_gone_from_any_state(before: SessionState) -> None:
    assert reduce(holding(before), Ended(ONE.id, "prompt_input_exit"))[0] == holding(Gone())


def test_a_prompt_opens_a_turn_and_one_inside_an_open_turn_is_queued_in_it() -> None:
    """One inside a turn is queued behind it, as 2.1.282 runs a message sent while a turn runs."""
    opened, _ = reduce(holding(BUSY), Prompted(ONE.id, at=5.0, mode=None, prompt=TURN))
    assert opened == in_turn()
    assert reduce(opened, Prompted(ONE.id, at=6.0, mode=None, prompt=TURN))[0] == holding(BUSY, Opened(TURN, queued=True))


@pytest.mark.parametrize("before", [IDLE, BUSY])
@pytest.mark.parametrize("event", [Prompted(ONE.id, at=5.0, mode=None, prompt=TURN), Ended(ONE.id, "prompt_input_exit"), Joined(ONE, "startup")])
def test_moving_between_states_that_wait_on_nothing_asks_for_nothing(before: SessionState, event: Event) -> None:
    # A prompt with no turn open marks the repository it is about to change, which asks the user nothing and is
    # not spoken. A prompt arriving mid-turn opens no turn and so marks nothing: see the test below.
    marked = [Snapshot(ONE.id, ONE.cwd)] if isinstance(event, Prompted) else []
    assert reduce(holding(before), event)[1] == marked


@pytest.mark.parametrize("before", LIVE)
def test_a_permission_request_is_handed_to_the_intermediary(before: SessionState) -> None:
    """Held whatever the status last read: a session that joins on its request has had none read yet."""
    event = PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), on=BASH, mode=None)
    assert reduce(holding(before), event) == (holding(before, dialog=replace(HELD, request=RequestId("r1"), deadline=5.0 + TIMEOUT)), [Narrate(Asking(ONE.id, RequestId("r1"), BASH))])


@pytest.mark.parametrize(
    "event",
    [
        Prompted(ONE.id, at=5.0, mode=None, prompt=TURN),
        Ended(ONE.id, "prompt_input_exit"),
        said(status.Idle()),
        *(Joined(ONE, source) for source in ("startup", "resume", "clear")),
    ],
)
def test_a_session_that_moves_on_while_waiting_lets_its_hook_go_undecided(event: Event) -> None:
    # Most often the user answered the dialog at the keyboard; a voice reply after that would decide nothing.
    assert reduce(in_turn(AT_DIALOG, HELD), event)[1] == [Reply(ONE.id, RequestId("r0"), Withdraw())]


@pytest.mark.parametrize("before", [in_turn(), in_turn(AT_DIALOG, HELD), in_turn(IDLE)])
def test_a_prompt_inside_an_open_turn_leaves_the_mark_where_that_turn_began(before: Registry) -> None:
    """A turn opens from a prompt with none open, so a prompt that lands inside one opens nothing.

    Claude Code sends the hook for a queued prompt, and a Stop that the daemon never heard leaves a turn open as far as
    the registry knows. Marked again at either, the turn is compared against the middle of its own work: everything it
    changed before that second prompt is missing from the one telling that names it.
    """
    assert Snapshot(ONE.id, ONE.cwd) not in reduce(before, Prompted(ONE.id, at=5.0, mode=None, prompt=TURN))[1]


def test_a_prompt_heard_after_the_busy_it_set_was_read_opens_and_marks_its_turn() -> None:
    """Claude Code sets busy before a prompt's hooks run (2.1.282), and the status can be read before the hook lands."""
    assert reduce(holding(BUSY, Told(TURN)), Prompted(ONE.id, at=5.0, mode=None, prompt=NEXT)) == (in_turn(turn=NEXT), [Snapshot(ONE.id, ONE.cwd)])


@pytest.mark.parametrize("before", [holding(IDLE), holding(BUSY), in_turn()])
def test_a_finished_turn_is_summarised_from_the_session_transcript(before: Registry) -> None:
    """The turn it names, whether hands had it open or never heard it open. It runs until Claude Code says it is idle."""
    state = before.sessions[ONE.id].state
    assert reduce(before, Stopped(ONE.id, "Done.", mode=None, prompt=TURN, again=False)) == (holding(state, Told(TURN)), [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, "Done.")])


def test_a_turn_that_stops_while_a_hook_is_held_is_told_and_the_hook_let_go_once_claude_code_says_idle() -> None:
    state, effects = reduce(in_turn(AT_DIALOG, HELD), Stopped(ONE.id, None, mode=None, prompt=TURN, again=False))
    assert effects == [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, None)]
    assert reduce(state, said(status.Idle()))[1] == [Reply(ONE.id, RequestId("r0"), Withdraw())]


def test_a_turn_is_compared_before_it_is_handed_over_to_be_summarised() -> None:
    """Effects are performed in the order they are given, and this order is the whole of why it is two effects.

    A summary is made one at a time and takes seconds. Read the repository when the summary is made rather
    than when the turn stopped, and it holds whatever the next turn has since started doing.
    """
    effects = reduce(in_turn(), Stopped(ONE.id, None, mode=None, prompt=TURN, again=False))[1]
    assert effects.index(Compare(ONE.id, again=False)) < effects.index(Summarise(ONE.id, TURN, None))


def test_a_prompt_marks_where_the_repository_stands_before_the_turn_can_change_it() -> None:
    """The mark is what the turn's changes are read against, so it is taken as the turn opens, not during it."""
    assert reduce(holding(IDLE), Prompted(ONE.id, at=5.0, mode=None, prompt=TURN))[1] == [Snapshot(ONE.id, ONE.cwd)]


def test_a_turn_that_goes_on_after_another_stop_hook_blocked_its_stop_runs_until_claude_code_says_idle() -> None:
    """Seen live on 2.1.282 (hands-status-bpp.55k): the file says busy between the blocked Stop and the one after it."""
    heard: list[Effect] = []
    state = in_turn()
    for event in [Stopped(ONE.id, "First.", mode=None, prompt=TURN, again=False), Stopped(ONE.id, "Second.", mode=None, prompt=TURN, again=True)]:
        state, effects = reduce(state, event)
        heard += effects
        assert state.sessions[ONE.id].state == BUSY
    assert heard == [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, "First."), Compare(ONE.id, again=True), Summarise(ONE.id, TURN, "Second.")]
    assert reduce(state, said(status.Idle())) == (holding(Idle(Stamp(2000), due=10.0 + IDLE_NUDGE_SECONDS, after=TURN), Told(TURN)), [])


def test_a_second_request_while_waiting_lets_the_first_go_and_asks_the_second() -> None:
    edit = Permission(tool="Edit", input={"file_path": "a.py"})
    after, effects = reduce(holding(AT_DIALOG, dialog=HELD), PermissionRequested(ONE.id, at=7.0, request=RequestId("r1"), on=edit, mode=None))
    assert after == holding(AT_DIALOG, dialog=Held(on=edit, request=RequestId("r1"), deadline=7.0 + TIMEOUT, warned=False))
    assert effects == [Reply(ONE.id, RequestId("r0"), Withdraw()), Narrate(Asking(ONE.id, RequestId("r1"), edit))]


def test_before_the_warning_window_a_tick_changes_nothing() -> None:
    assert reduce(holding(AT_DIALOG, dialog=HELD), Tick(at=61.0 - WARNING_LEAD_SECONDS - 0.5)) == (holding(AT_DIALOG, dialog=HELD), [])


def test_the_warning_is_spoken_once_as_the_deadline_nears() -> None:
    warned, effects = reduce(holding(AT_DIALOG, dialog=HELD), Tick(at=53.0))
    assert warned == holding(AT_DIALOG, dialog=replace(HELD, warned=True))
    assert effects == [Speak(DeadlineNear(ONE.id, BASH, remaining=8.0))]
    assert reduce(warned, Tick(at=54.0)) == (warned, [])


@pytest.mark.parametrize("warned", [True, False])
def test_at_the_deadline_the_request_is_denied_and_said_to_be(warned: bool) -> None:
    after, effects = reduce(holding(AT_DIALOG, dialog=replace(HELD, warned=warned)), Tick(at=61.0))
    assert after == holding(AT_DIALOG)
    assert effects == [Reply(ONE.id, RequestId("r0"), Deny(EXPIRED_MESSAGE)), Speak(Expired(ONE.id, BASH))]


def test_ticking_through_a_whole_wait_warns_exactly_once_then_denies_once() -> None:
    state = holding(BUSY)
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
    before = registry(Session(ONE, AT_DIALOG, mode=None, dialog=HELD), Session(TWO, BUSY, mode=None))
    after, _ = reduce(before, Tick(at=100.0))
    assert after == registry(Session(ONE, AT_DIALOG, mode=None), Session(TWO, BUSY, mode=None))


@pytest.mark.parametrize(
    "before",
    [
        Session(ONE, IDLE, mode="plan", turn=Untold(TURN, frozenset({NEXT}), by=11.0, asking=True)),
        Session(ONE, replace(IDLE, nudged=True), mode=None, turn=Told(TURN)),
        Session(ONE, BUSY, mode=None, turn=Opened(TURN, frozenset({NEXT}), queued=True)),
        Session(ONE, AT_DIALOG, mode="acceptEdits", turn=Opened(TURN), dialog=HELD),
        Session(ONE, AT_DIALOG, mode=None, turn=Opened(TURN), dialog=LetGo(BASH)),
        Session(ONE, Unreported(), mode=None),
    ],
)
def test_a_compacted_session_is_kept_whole_under_its_new_membership(before: Session) -> None:
    """Compaction starts a session again in the middle of what it was doing, in the same process, whose status file it
    goes on writing: a report, a turn, or a told turn dropped here would not come again."""
    moved = replace(ONE, pid=7777)
    assert reduce(registry(before), Joined(moved, "compact")) == (registry(replace(before, membership=moved)), [])


@pytest.mark.parametrize("before", [*LIVE, Gone()])
@pytest.mark.parametrize("source", ["startup", "resume", "clear"])
def test_any_start_but_compaction_waits_for_its_status_to_be_read(before: SessionState, source: StartSource) -> None:
    # a session resumed after a crash never sent the Stop or SessionEnd the registry is still waiting for
    assert reduce(holding(before, Opened(TURN)), Joined(ONE, source))[0] == holding(Unreported())


def test_an_ended_session_compacting_waits_for_its_status_to_be_read() -> None:
    assert reduce(holding(Gone()), Joined(ONE, "compact")) == (holding(Unreported()), [])


@pytest.mark.parametrize("event", SESSION_EVENTS)
def test_an_ended_session_ignores_late_hooks_and_audits_them(event: SessionEvent) -> None:
    assert reduce(holding(Gone()), event) == (holding(Gone()), [Audit(AfterEnd(event)), *let_go(event)])


@pytest.mark.parametrize("event", SESSION_EVENTS)
def test_an_event_for_a_session_that_never_joined_changes_nothing_and_is_audited(event: SessionEvent) -> None:
    before = registry(Session(TWO, IDLE, mode=None))
    assert reduce(before, event) == (before, [Audit(Unregistered(event)), *let_go(event)])


def let_go(event: SessionEvent) -> list[Effect]:
    # A permission hook the registry cannot block on is released at once instead of hanging.
    return [Reply(ONE.id, RequestId("r1"), Withdraw())] if isinstance(event, PermissionRequested) else []


def test_the_tool_a_session_waits_on_finishing_means_its_dialog_was_answered_at_the_keyboard() -> None:
    after, effects = reduce(holding(AT_DIALOG, dialog=HELD), ToolFinished(ONE.id, at=20.0, call=BASH, mode=None))
    assert (after, effects) == (holding(AT_DIALOG), [Reply(ONE.id, RequestId("r0"), Withdraw())])


def test_a_question_answered_at_the_keyboard_comes_back_with_its_answers_and_still_releases_the_wait() -> None:
    asked = {"questions": [{"question": "Which?", "options": [{"label": "this"}]}]}
    question = Question((AskedQuestion("Which?", (Option("this", None),), several=False),), asked)
    answered = Question(question.asked, {**asked, "answers": {"Which?": "this"}})
    waiting = Held(on=question, request=RequestId("r0"), deadline=65.0, warned=False)
    after, effects = reduce(holding(AT_DIALOG, dialog=waiting), ToolFinished(ONE.id, at=20.0, call=answered, mode=None))
    assert (after, effects) == (holding(AT_DIALOG), [Reply(ONE.id, RequestId("r0"), Withdraw())])


@pytest.mark.parametrize("dialog", [Held(on=Plan("the plan"), request=RequestId("r0"), deadline=65.0, warned=False), LetGo(Plan("the plan"))])
def test_a_plan_approved_at_the_keyboard_releases_the_wait(dialog: Dialog) -> None:
    after, _ = reduce(holding(AT_DIALOG, dialog=dialog), ToolFinished(ONE.id, at=20.0, call=PlanApproved(), mode=None))
    assert after == holding(AT_DIALOG)


@pytest.mark.parametrize("before", [holding(AT_DIALOG, dialog=HELD), holding(IDLE), holding(BUSY)])
def test_any_other_tool_finishing_changes_nothing(before: Registry) -> None:
    other = Permission(tool="Bash", input={"command": "ls -la"})
    assert reduce(before, ToolFinished(ONE.id, at=20.0, call=other, mode=None)) == (before, [])


def test_a_hook_that_went_away_ends_the_wait_with_nothing_to_reply_to() -> None:
    assert reduce(holding(AT_DIALOG, dialog=HELD), Abandoned(ONE.id, RequestId("r0"), at=30.0)) == (holding(AT_DIALOG), [])


@pytest.mark.parametrize("before", [holding(AT_DIALOG, dialog=replace(HELD, request=RequestId("newer"))), holding(IDLE), holding(Gone())])
def test_an_abandoned_request_the_session_no_longer_waits_on_changes_nothing(before: Registry) -> None:
    assert reduce(before, Abandoned(ONE.id, RequestId("r0"), at=30.0)) == (before, [])


def test_an_event_moves_only_its_own_session() -> None:
    before = registry(Session(ONE, IDLE, mode=None), Session(TWO, IDLE, mode=None))
    after, _ = reduce(before, Prompted(TWO.id, at=2.0, mode=None, prompt=TURN))
    assert after == registry(Session(ONE, IDLE, mode=None), Session(TWO, IDLE, mode=None, turn=Opened(TURN)))


def test_live_is_every_session_that_has_not_ended() -> None:
    both = registry(Session(ONE, AT_DIALOG, mode=None, dialog=HELD), Session(TWO, Gone(), mode=None))
    assert both.live() == [Session(ONE, AT_DIALOG, mode=None, dialog=HELD)]


def test_a_file_for_a_session_never_heard_of_attaches_it_to_be_read() -> None:
    assert reduce(registry(), Attached(ONE)) == (holding(Unreported()), [])


@pytest.mark.parametrize("before", [*LIVE, Gone()])
def test_a_file_for_a_session_already_known_changes_nothing(before: SessionState) -> None:
    moved = replace(ONE, cwd=Path("/code/elsewhere"))
    assert reduce(holding(before), Attached(moved)) == (holding(before), [])


@pytest.mark.parametrize("ended", [Died(ONE), MovedOn(ONE)])
def test_a_session_this_run_never_listed_ending_says_nothing(ended: Event) -> None:
    assert reduce(registry(), ended) == (registry(), [])


@pytest.mark.parametrize("dialog", [None, HELD])
@pytest.mark.parametrize("before", LIVE)
def test_a_live_session_whose_process_died_is_gone_and_spoken_and_its_hook_let_go(before: SessionState, dialog: Dialog | None) -> None:
    released = [] if dialog is None else [Reply(ONE.id, HELD.request, Withdraw())]
    assert reduce(holding(before, dialog=dialog), Died(ONE)) == (holding(Gone()), [*released, SessionGone(ONE.id)])


def test_a_session_already_gone_dying_again_says_nothing() -> None:
    assert reduce(holding(Gone()), Died(ONE)) == (holding(Gone()), [])


def test_a_dead_process_that_is_no_longer_the_sessions_changes_nothing() -> None:
    resumed = holding(BUSY)
    assert reduce(resumed, Died(replace(ONE, pid=ONE.pid + 1))) == (resumed, [])


@pytest.mark.parametrize("dialog", [None, HELD])
@pytest.mark.parametrize("before", LIVE)
def test_a_session_whose_process_moved_on_is_gone_silently_and_its_hook_let_go(before: SessionState, dialog: Dialog | None) -> None:
    released = [] if dialog is None else [Reply(ONE.id, HELD.request, Withdraw())]
    assert reduce(holding(before, dialog=dialog), MovedOn(ONE)) == (holding(Gone()), released)


def test_a_file_on_a_pid_another_session_holds_attaches_beside_it_the_sweep_decides_which_is_over() -> None:
    both = reduce(holding(IDLE), Attached(replace(TWO, pid=ONE.pid)))[0]
    assert [session.membership.id for session in both.live()] == [ONE.id, TWO.id]


@pytest.mark.parametrize("dialog", [None, HELD])
@pytest.mark.parametrize("before", LIVE)
def test_a_session_whose_terminal_closed_is_gone_and_spoken(before: SessionState, dialog: Dialog | None) -> None:
    released = [] if dialog is None else [Reply(ONE.id, HELD.request, Withdraw())]
    assert reduce(holding(before, dialog=dialog), Ended(ONE.id, "other")) == (holding(Gone()), [*released, SessionGone(ONE.id)])


@pytest.mark.parametrize("reason", ["clear", "resume", "logout", "prompt_input_exit", "bypass_permissions_disabled"])
def test_a_session_ended_at_the_keyboard_is_not_spoken(reason: EndReason) -> None:
    assert reduce(holding(IDLE), Ended(ONE.id, reason)) == (holding(Gone()), [])


def test_a_closed_terminal_after_the_sweep_found_the_session_dead_says_nothing_more() -> None:
    event = Ended(ONE.id, "other")
    assert reduce(holding(Gone()), event) == (holding(Gone()), [Audit(AfterEnd(event))])


NUDGE = Speak(WaitingForYou(ONE.id, asking=False))
WENT_IDLE = said(status.Idle(), 2000, at=10.0)


def test_a_session_left_at_its_prompt_is_said_to_be_waiting() -> None:
    assert reduce(holding(IDLE), Waited(ONE.id)) == (holding(replace(IDLE, nudged=True)), [NUDGE])


def nudged(state: Registry, *events: Event) -> list[Effect]:
    """What is said once the session has sat idle, after the events."""
    for event in events:
        state, _ = reduce(state, event)
    return [effect for effect in reduce(state, Waited(ONE.id))[1] if isinstance(effect, Speak)]


@pytest.mark.parametrize(
    ("closing", "asking"),
    [
        ("Fixed the refresh test. Want me to look at the other flaky ones?", True),
        # No question mark: an offer is asked all the same.
        ("The dev build should not be on that Mac. Say the word and I'll remove it.", True),
        # A question the reply went on to answer asks the listener nothing.
        ("Why did it fail?\n\nThe mock froze the clock. All twelve tests pass now.", False),
        ("Done.", False),
        (None, False),
    ],
)
def test_a_turn_that_stopped_on_a_question_is_nudged_as_having_one(closing: str | None, asking: bool) -> None:
    """Read by the narration's own reading of a text for questions, so the nudge never promises one the telling does not ask."""
    assert nudged(in_turn(), Stopped(ONE.id, closing, mode=None, prompt=TURN, again=False), WENT_IDLE) == [Speak(WaitingForYou(ONE.id, asking=asking))]


QUESTION = Question((AskedQuestion("Merge now?", (Option("merge", None), Option("wait", None)), several=False),), {})
ASKING = Held(on=QUESTION, request=RequestId("q0"), deadline=65.0, warned=False)
ESCAPED = Abandoned(ONE.id, RequestId("q0"), at=8.0)
STOPPED = Stopped(ONE.id, "Done.", mode=None, prompt=TURN, again=False)


@pytest.mark.parametrize("dialog", [ASKING, LetGo(QUESTION)])
def test_a_turn_that_ended_with_its_dialog_question_unanswered_is_nudged_as_having_one(dialog: Dialog) -> None:
    """The telling counts an unanswered `AskUserQuestion` as waiting, so the nudge does too: interrupted at the
    dialog, Claude Code sets the session idle with no Stop and no closing reply to read."""
    assert nudged(in_turn(AT_DIALOG, dialog), WENT_IDLE) == [Speak(WaitingForYou(ONE.id, asking=True))]


@pytest.mark.parametrize(
    "ended",
    [
        # The interrupt Claude Code sets idle for, with no Stop.
        [WENT_IDLE],
        # The same, with the Stop an Escape can fire after the idle, applied after it or before it was read.
        [WENT_IDLE, STOPPED],
        [STOPPED, WENT_IDLE],
        [WENT_IDLE, Interrupted(ONE.id, TURN, at=11.0)],
    ],
)
def test_a_turn_escaped_at_its_dialog_question_is_nudged_as_having_one_however_it_ends(ended: list[Event]) -> None:
    """The telling counts a dialog that nothing but the interruption followed as waiting, and so does the nudge,
    though the Escape killed the dialog's hook before the turn ended."""
    assert nudged(in_turn(AT_DIALOG, ASKING), ESCAPED, *ended) == [Speak(WaitingForYou(ONE.id, asking=True))]


def test_a_question_left_to_its_dialog_at_the_voice_deadline_and_then_escaped_is_still_the_nudge() -> None:
    assert nudged(in_turn(AT_DIALOG, ASKING), Tick(at=65.0), WENT_IDLE) == [Speak(WaitingForYou(ONE.id, asking=True))]


@pytest.mark.parametrize(
    "after",
    [
        # Claude went on past the escaped dialog and ran something, as the telling sees in the steps after it.
        [ESCAPED, ToolFinished(ONE.id, at=9.0, call=BASH, mode=None)],
        [ESCAPED, PermissionRequested(ONE.id, 9.0, RequestId("r1"), BASH, None), ToolFinished(ONE.id, at=9.5, call=BASH, mode=None)],
        # A message typed or queued after it answered it.
        [ESCAPED, Prompted(ONE.id, at=9.0, mode=None, prompt=TURN)],
        [ESCAPED, Continued(ONE.id, was=TURN, now=PromptId("p2"))],
        # Answered at the keyboard: the question's tool ran, before its hook was let go or after.
        [ToolFinished(ONE.id, at=9.0, call=QUESTION, mode=None)],
        [ToolFinished(ONE.id, at=9.0, call=QUESTION, mode=None), ESCAPED],
    ],
)
def test_a_dialog_question_answered_or_gone_past_is_not_what_the_turn_waits_on(after: list[Event]) -> None:
    assert nudged(in_turn(AT_DIALOG, ASKING), *after, WENT_IDLE) == [NUDGE]


def test_an_escaped_question_is_what_the_turn_waits_on_until_it_does_something_else() -> None:
    assert reduce(in_turn(AT_DIALOG, ASKING), ESCAPED)[0] == in_turn(AT_DIALOG, Unanswered())


def test_a_permission_escaped_at_its_dialog_asks_the_listener_nothing() -> None:
    assert nudged(in_turn(AT_DIALOG, replace(HELD, request=RequestId("q0"))), ESCAPED, WENT_IDLE) == [NUDGE]


def test_a_question_is_still_the_nudge_when_hands_times_it_itself() -> None:
    assert reduce(holding(IDLE, Told(TURN, asking=True)), Tick(at=70.0)) == (holding(replace(IDLE, nudged=True), Told(TURN, asking=True)), [Speak(WaitingForYou(ONE.id, asking=True))])


@pytest.mark.parametrize("before", [holding(BUSY), in_turn(IDLE)])
def test_an_idle_notification_that_lands_after_the_prompt_it_raced_nudges_nothing(before: Registry) -> None:
    """Running, or at its prompt with a turn opened that the status is yet to say runs."""
    assert reduce(before, Waited(ONE.id)) == (before, [])


def test_a_nudged_session_prompted_again_marks_the_repository_its_turn_starts_from() -> None:
    nudged = replace(IDLE, nudged=True)
    assert reduce(holding(nudged), Prompted(ONE.id, at=5.0, mode=None, prompt=TURN)) == (in_turn(nudged), [Snapshot(ONE.id, ONE.cwd)])


def test_one_idle_period_is_nudged_once() -> None:
    nudged = holding(replace(IDLE, nudged=True))
    assert reduce(nudged, Waited(ONE.id)) == (nudged, [])


def test_a_session_waiting_on_a_permission_is_not_nudged_and_keeps_its_hook() -> None:
    assert reduce(holding(AT_DIALOG, dialog=HELD), Waited(ONE.id)) == (holding(AT_DIALOG, dialog=HELD), [])


@pytest.mark.parametrize("opened", [Prompted(ONE.id, at=5.0, mode=None, prompt=TURN), Joined(ONE, "compact"), Joined(ONE, "resume")])
def test_a_session_prompted_again_and_left_again_is_nudged_again(opened: Event) -> None:
    heard: list[Effect] = []
    state = holding(IDLE)
    for event in [Waited(ONE.id), Waited(ONE.id), opened, said(Busy(), 3000), Stopped(ONE.id, None, mode=None, prompt=TURN, again=False), said(status.Idle(), 4000), Waited(ONE.id), Waited(ONE.id)]:
        state, effects = reduce(state, event)
        heard += [effect for effect in effects if isinstance(effect, Speak)]
    assert heard == [NUDGE, NUDGE]


REPORTING: list[SessionEvent] = [
    Prompted(ONE.id, at=5.0, mode="plan", prompt=TURN),
    Stopped(ONE.id, None, mode="plan", prompt=TURN, again=False),
    PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), on=BASH, mode="plan"),
    ToolFinished(ONE.id, at=5.0, call=BASH, mode="plan"),
]


def moded(state: SessionState, mode: Mode | None, dialog: Dialog | None = None) -> Registry:
    return registry(Session(ONE, state, mode=mode, dialog=dialog))


@pytest.mark.parametrize("before", LIVE)
@pytest.mark.parametrize("event", REPORTING)
def test_every_hook_that_reports_a_mode_sets_the_sessions_mode(before: SessionState, event: SessionEvent) -> None:
    after, effects = reduce(moded(before, "default"), event)
    assert after.sessions[ONE.id].mode == "plan"
    assert Note(ModeChanged(ONE.id, "plan")) in effects


@pytest.mark.parametrize("event", [Waited(ONE.id), Stopped(ONE.id, None, mode=None, prompt=TURN, again=False)])
def test_a_hook_that_reports_no_mode_keeps_the_one_last_reported(event: SessionEvent) -> None:
    after, effects = reduce(moded(BUSY, "acceptEdits"), event)
    assert after.sessions[ONE.id].mode == "acceptEdits"
    assert not [effect for effect in effects if isinstance(effect, Note)]


def test_a_mode_reported_again_unchanged_is_not_noted() -> None:
    _, effects = reduce(moded(IDLE, "plan"), Prompted(ONE.id, at=5.0, mode="plan", prompt=TURN))
    assert not [effect for effect in effects if isinstance(effect, Note)]


def test_a_mode_reported_for_the_first_time_is_noted_because_a_resumed_session_may_be_in_another_mode() -> None:
    after, effects = reduce(moded(IDLE, None), Prompted(ONE.id, at=5.0, mode="auto", prompt=TURN))
    assert after.sessions[ONE.id].mode == "auto"
    assert Note(ModeChanged(ONE.id, "auto")) in effects


def test_a_request_is_narrated_after_the_mode_it_was_asked_in_is_noted() -> None:
    _, effects = reduce(moded(BUSY, "default"), PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), on=BASH, mode="acceptEdits"))
    assert effects == [Note(ModeChanged(ONE.id, "acceptEdits")), Narrate(Asking(ONE.id, RequestId("r1"), BASH))]


def test_a_mode_hands_does_not_know_is_noted_like_any_other() -> None:
    _, effects = reduce(moded(IDLE, "default"), Prompted(ONE.id, at=5.0, mode=UnknownMode("ultraplan"), prompt=TURN))
    assert Note(ModeChanged(ONE.id, UnknownMode("ultraplan"))) in effects


@pytest.mark.parametrize("before", LIVE)
def test_compaction_keeps_the_mode_because_it_keeps_the_process(before: SessionState) -> None:
    assert reduce(moded(before, "acceptEdits"), Joined(replace(ONE, pid=7777), "compact"))[0].sessions[ONE.id].mode == "acceptEdits"


@pytest.mark.parametrize("source", ["startup", "resume", "clear"])
def test_any_other_start_has_not_reported_a_mode(source: StartSource) -> None:
    assert reduce(moded(IDLE, "acceptEdits"), Joined(ONE, source))[0].sessions[ONE.id].mode is None


def test_a_session_keeps_its_mode_through_a_deadline_an_abandoned_hook_and_its_end() -> None:
    assert reduce(moded(AT_DIALOG, "plan", HELD), Tick(at=100.0))[0].sessions[ONE.id].mode == "plan"
    assert reduce(moded(AT_DIALOG, "plan", HELD), Abandoned(ONE.id, RequestId("r0"), at=9.0))[0].sessions[ONE.id].mode == "plan"
    assert reduce(moded(IDLE, "plan"), Died(ONE))[0].sessions[ONE.id].mode == "plan"
    assert reduce(moded(IDLE, "plan"), Ended(ONE.id, "other"))[0].sessions[ONE.id].mode == "plan"


def test_a_prompt_names_the_turn_it_opens() -> None:
    after, _ = reduce(holding(IDLE), Prompted(ONE.id, at=5.0, mode=None, prompt=TURN))
    assert after == in_turn(IDLE)


@pytest.mark.parametrize("before", [in_turn(), in_turn(IDLE), holding(IDLE, Told(TURN))])
def test_a_record_of_a_turn_hands_already_named_moves_nothing(before: Registry) -> None:
    """The first record of a prompt's turn, read while it runs, before its busy is read, or after it was told."""
    assert reduce(before, Taken(ONE.id, TURN, Stamp(5000), 9.0)) == (before, [])


def test_a_prompt_cancelled_while_its_hooks_ran_is_over_when_claude_code_says_idle() -> None:
    """As seen live on 2.1.282: Escape during UserPromptSubmit puts the prompt back in the box and sets idle ~70 ms
    later, with no Stop and no record. Resent, it is a prompt of its own."""
    state, tellings = told([Prompted(ONE.id, at=5.0, mode=None, prompt=TURN), said(status.Idle(), at=5.1)], holding(IDLE))
    assert state == holding(Idle(Stamp(2000), due=5.1 + IDLE_NUDGE_SECONDS, after=TURN), Untold(TURN, frozenset(), by=5.1 + UNTOLD_SECONDS, asking=False))
    state, tellings = told([Prompted(ONE.id, at=9.0, mode=None, prompt=NEXT), Taken(ONE.id, NEXT, None, 9.0)], state)
    # Told as itself, which says nothing of a prompt that never ran: no record carries its id.
    assert (state.sessions[ONE.id].turn, tellings) == (Opened(NEXT), [*TOLD, Snapshot(ONE.id, ONE.cwd)])


def test_a_prompt_taken_and_stopped_before_any_of_it_was_read_is_told_as_itself() -> None:
    """hands-keyboard-gxr.90g: p1 is taken and interrupted inside one tail period, so Claude Code's idle finds it still
    unread. Its record and its interrupt, read after, tell it once, as itself; the next prompt opens its own turn."""
    state, tellings = told([Prompted(ONE.id, at=5.0, mode=None, prompt=TURN), said(status.Idle()), Taken(ONE.id, TURN, None, 9.0)], holding(IDLE))
    assert (state.sessions[ONE.id].turn, tellings) == (Untold(TURN, frozenset(), by=10.0 + UNTOLD_SECONDS, asking=False), [Snapshot(ONE.id, ONE.cwd)])
    state, tellings = told([INTERRUPT, STOP, PROMPT, Tick(20.0)], state)
    assert (state.sessions[ONE.id].turn, tellings) == (Opened(NEXT), [*TOLD, Snapshot(ONE.id, ONE.cwd)])


def test_a_stop_is_told_as_the_turn_its_hook_names() -> None:
    assert reduce(in_turn(), Stopped(ONE.id, "Done.", mode=None, prompt=TURN, again=False))[1] == [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, "Done.")]


def test_the_stop_of_a_turn_over_before_the_next_prompt_ends_nothing() -> None:
    """Applied late, it would end the turn now open and spend its mark."""
    after, effects = reduce(in_turn(turn=NEXT), Stopped(ONE.id, "Done.", mode="plan", prompt=TURN, again=False))
    assert effects == [Note(ModeChanged(ONE.id, "plan"))] and after.sessions[ONE.id].turn == Opened(NEXT)


@pytest.mark.parametrize(("turn", "again"), [(TURN, False), (None, False), (PromptId("q"), True)])
@pytest.mark.parametrize("state", [IDLE, BUSY])
def test_a_stop_with_no_turn_open_tells_the_turn_it_names(state: SessionState, turn: PromptId | None, again: bool) -> None:
    """The turn a session was in when hands attached to it, one it was never heard to open, or the last one stopping
    again after another Stop hook blocked its Stop, which Claude Code stops under the same id."""
    after, effects = reduce(holding(state, Told(turn)), Stopped(ONE.id, "Done.", mode=None, prompt=PromptId("q"), again=again))
    assert after == holding(state, Told(PromptId("q")))
    # The turn stopping again is read against where its last Stop's reading ended: no prompt marked where it went on.
    assert effects == [Compare(ONE.id, again), Summarise(ONE.id, PromptId("q"), "Done.")]


def test_the_stop_of_a_turn_already_told_ends_nothing_and_keeps_the_idle_period_it_lands_in() -> None:
    """An interrupted turn's Stop can land after it was told: told again, it would be heard twice."""
    assert reduce(holding(IDLE, Told(TURN)), STOP) == (holding(IDLE, Told(TURN)), [])


def test_a_stop_naming_an_id_the_open_turn_does_not_go_by_leaves_its_end_to_claude_code() -> None:
    """A flush's id the tail has not read yet: the turn is ended by the idle Claude Code sets after the Stop."""
    state, tellings = told([Stopped(ONE.id, "Done.", mode=None, prompt=PromptId("q"), again=False)])
    assert (state, tellings) == (in_turn(), [])
    state, tellings = told([said(status.Idle()), Tick(20.0)], state)
    assert (state.sessions[ONE.id].turn, tellings) == (Told(TURN), TOLD)


def test_the_stop_of_a_turn_gone_on_under_a_flushed_id_ends_it_before_claude_answered_under_it() -> None:
    state, _ = reduce(in_turn(), Taken(ONE.id, PromptId("q"), None, 9.0))
    assert reduce(state, Stopped(ONE.id, "Done.", mode=None, prompt=PromptId("q"), again=False))[1] == [Compare(ONE.id, again=False), Summarise(ONE.id, PromptId("q"), "Done.")]


@pytest.mark.parametrize("before", [in_turn(), in_turn(AT_DIALOG, HELD), in_turn(AT_DIALOG, LetGo(BASH))])
@pytest.mark.parametrize("event", [Prompted(ONE.id, at=9.0, mode=None, prompt=NEXT), Taken(ONE.id, NEXT, None, 9.0)])
def test_a_prompt_or_record_naming_another_turn_while_one_is_open_ends_nothing(before: Registry, event: Event) -> None:
    """[LAW:one-source-of-truth] only Claude Code's idle or the turn's Stop ends it. A queued message's hook carries the
    open turn's id (2.1.281), so another one means an idle hands has not read yet: the id joins the turn's, and its
    Stop ends it."""
    after, effects = reduce(before, event)
    turn = after.sessions[ONE.id].turn
    assert isinstance(turn, Opened) and (turn.turn, turn.others) == (TURN, {NEXT})
    assert not any(isinstance(effect, Compare | Summarise | Snapshot) for effect in effects)
    assert reduce(after, Stopped(ONE.id, "Next.", mode=None, prompt=NEXT, again=False))[0].sessions[ONE.id].turn == Told(TURN, frozenset({NEXT}))


def test_a_message_queued_under_a_flushed_id_before_claude_answers_under_it_is_queued_into_the_turn() -> None:
    """The flush's records carry its new id at once, and Claude's first answer under it — the Continued — comes seconds
    later. A message queued in between carries the new id (2.1.281), and is queued, not the turn ended."""
    flushed = PromptId("q")
    state, effects = reduce(in_turn(), Taken(ONE.id, flushed, None, 9.0))
    assert effects == [] and state.sessions[ONE.id].turn == Opened(TURN, frozenset({flushed}))
    state, effects = reduce(state, Prompted(ONE.id, at=9.0, mode=None, prompt=flushed))
    assert effects == [] and state.sessions[ONE.id].turn == Opened(TURN, frozenset({flushed}), queued=True)


def test_a_turn_a_prompt_opens_has_gone_on_under_no_other_id_yet() -> None:
    state, _ = told([Taken(ONE.id, PromptId("q"), None, 9.0), said(status.Idle()), PROMPT])
    assert state.sessions[ONE.id].turn == Opened(NEXT)


def test_a_prompt_that_cannot_be_told_from_one_queued_into_the_open_turn_is_queued_into_it() -> None:
    """The open turn's own id is a queued prompt's."""
    assert reduce(in_turn(), Prompted(ONE.id, at=9.0, mode=None, prompt=TURN))[1] == []


@pytest.mark.parametrize("event", [
    PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), on=BASH, mode=None),
    ToolFinished(ONE.id, at=5.0, call=BASH, mode=None),
    Waited(ONE.id),
])
def test_only_a_prompt_or_a_record_moves_the_turn(event: SessionEvent) -> None:
    """A background subagent's hooks carry the prompt_id of the turn that started it, long after that turn is over."""
    assert reduce(in_turn(), event)[0].sessions[ONE.id].turn == Opened(TURN)


@pytest.mark.parametrize("before", [in_turn(), in_turn(AT_DIALOG, HELD), in_turn(AT_DIALOG, LetGo(BASH))])
def test_the_record_of_an_interrupt_ends_nothing(before: Registry) -> None:
    """[LAW:one-source-of-truth] Claude Code's idle, set ~100 ms before the record is written (2.1.282), ends the turn."""
    assert reduce(before, Interrupted(ONE.id, TURN, at=10.0)) == (before, [])


def test_an_interrupt_read_after_the_next_prompt_leaves_the_next_turn_open() -> None:
    """The tail reads the record a moment after it is written, by which time the user may have typed again."""
    running = in_turn(turn=NEXT)
    assert reduce(running, Interrupted(ONE.id, TURN, at=10.0)) == (running, [])


@pytest.mark.parametrize("before", [holding(IDLE, Told(TURN)), holding(replace(IDLE, nudged=True), Told(TURN)), holding(Gone(), Told(TURN))])
def test_an_interrupt_of_a_turn_already_over_changes_nothing(before: Registry) -> None:
    assert reduce(before, Interrupted(ONE.id, TURN, at=10.0)) == (before, [])


def test_a_prompt_opened_after_a_turn_was_told_is_not_ended_by_the_interrupt_of_that_one() -> None:
    after, _ = reduce(holding(IDLE, Told(TURN)), Prompted(ONE.id, at=5.0, mode=None, prompt=NEXT))
    assert reduce(after, Interrupted(ONE.id, TURN, at=10.0)) == (after, [])


def test_an_interrupt_the_registry_never_heard_open_changes_nothing() -> None:
    before = holding(BUSY)
    assert reduce(before, Interrupted(ONE.id, TURN, at=10.0)) == (before, [])


def test_a_session_left_idle_after_an_interrupt_is_nudged_once_by_the_clock() -> None:
    """Claude Code sends no idle_prompt after an interrupt (2.1.281), so the nudge comes when it would have."""
    heard: list[tuple[Event, Effect]] = []
    state = in_turn()
    for event in [said(status.Idle(), at=10.0), INTERRUPT, Tick(69.0), Tick(70.0), Tick(71.0), Waited(ONE.id)]:
        state, effects = reduce(state, event)
        heard += [(event, effect) for effect in effects if isinstance(effect, Speak)]
    assert heard == [(Tick(70.0), NUDGE)]


def test_an_idle_prompt_before_the_clock_is_the_one_nudge() -> None:
    heard: list[Effect] = []
    state = holding(IDLE)
    for event in [Waited(ONE.id), Tick(80.0)]:
        state, effects = reduce(state, event)
        heard += [effect for effect in effects if isinstance(effect, Speak)]
    assert heard == [NUDGE]


def test_a_prompt_before_the_clock_leaves_nothing_to_nudge() -> None:
    state, _ = reduce(holding(IDLE), Prompted(ONE.id, at=20.0, mode=None, prompt=NEXT))
    assert reduce(state, Tick(80.0))[1] == []


def test_a_queued_prompt_does_not_let_the_interrupt_that_flushes_it_end_the_turn_it_opens() -> None:
    """As seen live on 2.1.281: a prompt queued while a tool ran fires UserPromptSubmit at once, with the running turn's
    prompt_id. Escape then flushes it, and the interrupt record names the queued prompt's own new id, whose turn is
    already running. That turn's Stop, not the interrupt, is what ends it."""
    state = in_turn()
    for event in [Prompted(ONE.id, at=2.0, mode=None, prompt=TURN), Interrupted(ONE.id, NEXT, at=3.0)]:
        state, _ = reduce(state, event)
    assert state == holding(BUSY, Opened(TURN, queued=True))


def test_the_turn_a_queued_prompt_goes_on_as_is_ended_by_the_escape_that_stops_it() -> None:
    """No hook names the queued prompt's id: the transcript does, once Claude answers under it."""
    state = in_turn()
    for event in [Interrupted(ONE.id, NEXT, at=3.0), Continued(ONE.id, was=TURN, now=NEXT)]:
        state, _ = reduce(state, event)
    assert state == in_turn(turn=NEXT)
    assert reduce(state, said(status.Idle(), at=9.0))[0].sessions[ONE.id].turn == Untold(NEXT, frozenset(), 9.0 + UNTOLD_SECONDS, asking=False)


@pytest.mark.parametrize("session", [in_turn(turn=PromptId("p3")), holding(IDLE, Told(TURN))])
def test_a_turn_read_to_have_gone_on_after_it_ended_moves_nothing(session: Registry) -> None:
    assert reduce(session, Continued(ONE.id, was=TURN, now=NEXT)) == (session, [])


def told(events: list[Event], start: Registry | None = None) -> tuple[Registry, list[Effect]]:
    """The registry after the events, and every Compare, Summarise, and Snapshot they called for, in order."""
    state, tellings = start or in_turn(), list[Effect]()
    for event in events:
        state, effects = reduce(state, event)
        tellings += [effect for effect in effects if isinstance(effect, Compare | Summarise | Snapshot)]
    return state, tellings


TOLD = [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, None)]
STOP = Stopped(ONE.id, "done", mode=None, prompt=TURN, again=False)
INTERRUPT = Interrupted(ONE.id, TURN, at=10.2)
PROMPT = Prompted(ONE.id, at=10.5, mode=None, prompt=NEXT)


@pytest.mark.parametrize("before", [in_turn(), in_turn(AT_DIALOG, HELD), in_turn(AT_DIALOG, LetGo(BASH))])
def test_an_open_turn_claude_code_says_is_idle_is_over_at_once_and_nudged_on_the_clock(before: Registry) -> None:
    """However it was stopped: no Stop, no record, and no idle_prompt follow a double Escape before the flushed message
    is answered. The telling waits for the transcript to say how it ended."""
    after, effects = reduce(before, said(status.Idle()))
    assert after == holding(Idle(Stamp(2000), due=10.0 + IDLE_NUDGE_SECONDS, after=TURN), Untold(TURN, frozenset(), 10.0 + UNTOLD_SECONDS, asking=False))
    assert effects == ([Reply(ONE.id, RequestId("r0"), Withdraw())] if before.sessions[ONE.id].dialog == HELD else [])


@pytest.mark.parametrize("state", [replace(IDLE, after=TURN), replace(IDLE, after=TURN, nudged=True)])
def test_an_idle_said_of_a_session_with_no_turn_heard_since_it_went_idle_moves_nothing_but_its_stamp(state: Idle) -> None:
    after, effects = reduce(holding(state, Told(TURN)), said(status.Idle()))
    assert (after, effects) == (holding(replace(state, stamp=Stamp(2000)), Told(TURN)), [])


@pytest.mark.parametrize("status_", [Busy(), Waiting("permission prompt")])
def test_busy_or_waiting_said_of_an_open_turn_leaves_it_open(status_: Busy | Waiting) -> None:
    after, effects = reduce(in_turn(), said(status_))
    assert (after, effects) == (in_turn(Running(status_, Stamp(2000), idled=BUSY.idled)), [])


def test_the_interrupt_record_written_after_claude_code_said_idle_is_when_the_turn_is_told() -> None:
    """Escape mid-tool: idle is set ~100 ms before the interrupt record is written (2.1.282)."""
    state, tellings = told([said(status.Idle()), INTERRUPT, Tick(20.0)])
    assert tellings == TOLD
    assert state == holding(Idle(Stamp(2000), due=10.0 + IDLE_NUDGE_SECONDS, after=TURN), Told(TURN))


def test_an_interrupt_record_of_another_turn_leaves_the_untold_one_waiting_for_its_own() -> None:
    """A record read again from the start of a transcript is not the one the turn waits for."""
    state, tellings = told([said(status.Idle()), Interrupted(ONE.id, NEXT, at=10.1)])
    assert tellings == []
    state, effects = reduce(state, INTERRUPT)
    assert [effect for effect in effects if isinstance(effect, Compare | Summarise)] == TOLD


def test_a_stop_that_fires_after_claude_code_said_idle_tells_the_turn_with_its_closing_reply() -> None:
    state, tellings = told([said(status.Idle()), Stopped(ONE.id, "done", mode="plan", prompt=TURN, again=False), Tick(20.0)])
    assert tellings == [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, "done")]
    assert (state.sessions[ONE.id].state, state.sessions[ONE.id].mode) == (Idle(Stamp(2000), due=10.0 + IDLE_NUDGE_SECONDS, after=TURN), "plan")


@pytest.mark.parametrize("ending", [STOP, INTERRUPT])
def test_a_turn_its_stop_or_interrupt_ended_before_claude_code_said_idle_is_told_once(ending: Event) -> None:
    _, tellings = told([ending, said(status.Idle()), Tick(20.0)])
    assert [telling for telling in tellings if isinstance(telling, Summarise)] == [Summarise(ONE.id, TURN, ending.closing if isinstance(ending, Stopped) else None)]


def test_a_turn_nothing_says_how_it_ended_is_told_at_its_deadline_with_what_was_read() -> None:
    state, tellings = told([said(status.Idle(), at=10.0), Tick(10.5)])
    assert tellings == []
    state, tellings = told([Tick(10.0 + UNTOLD_SECONDS), INTERRUPT, STOP, Tick(30.0)], state)
    assert tellings == TOLD
    assert state.sessions[ONE.id].turn == Told(TURN)


def test_a_turn_left_untold_is_told_before_the_next_prompt_marks_its_own() -> None:
    state, tellings = told([said(status.Idle()), PROMPT, INTERRUPT, Tick(20.0)])
    assert tellings == [*TOLD, Snapshot(ONE.id, ONE.cwd)]
    assert state.sessions[ONE.id].turn == Opened(NEXT)


def test_the_record_of_the_turn_left_untold_waits_with_it_for_how_it_ended() -> None:
    """Read after Claude Code said idle, as a prompt taken and stopped inside one tail period is: it names the turn, and
    tells nothing before the record of how it ended."""
    _, tellings = told([said(status.Idle()), Taken(ONE.id, TURN, None, 9.0), Tick(10.5), INTERRUPT])
    assert tellings == TOLD


@pytest.mark.parametrize("event", [Joined(ONE, "compact"), Waited(ONE.id)])
def test_a_turn_left_untold_is_still_told_after_compaction_or_an_idle_prompt(event: Event) -> None:
    _, tellings = told([said(status.Idle()), event, Tick(20.0)])
    assert tellings == TOLD


def test_a_double_escape_before_claude_answers_a_flushed_message_leaves_the_session_idle_told_and_nudged() -> None:
    """hands-keyboard-gxr.07g: the first Escape flushes the queued message, whose id is taken seconds before Claude
    answers under it; the second stops the turn with no record and no hook. Only the status says so."""
    heard: list[Effect] = []
    state = in_turn()
    for event in [Taken(ONE.id, NEXT, None, 9.0), said(status.Idle(), at=10.0), Tick(11.0), Tick(70.0), Continued(ONE.id, was=TURN, now=NEXT), Interrupted(ONE.id, NEXT, at=71.0)]:
        state, effects = reduce(state, event)
        heard += [effect for effect in effects if isinstance(effect, Summarise | Speak)]
    assert heard == [Summarise(ONE.id, TURN, None), NUDGE]
    assert state.sessions[ONE.id].state == Idle(Stamp(2000), due=10.0 + IDLE_NUDGE_SECONDS, after=TURN, nudged=True)


def test_a_single_escape_that_flushes_a_queued_message_leaves_the_turn_running_on_to_its_stop() -> None:
    """Measured on 2.1.282: the flush sets busy again, not idle, and Claude answers under the flushed id."""
    state, tellings = told([Taken(ONE.id, NEXT, None, 9.0), said(Busy()), Continued(ONE.id, was=TURN, now=NEXT), Stopped(ONE.id, "done", mode=None, prompt=NEXT, again=False), said(status.Idle(), 3000, at=20.0), Tick(30.0)])
    assert tellings == [Compare(ONE.id, again=False), Summarise(ONE.id, NEXT, "done")]
    assert isinstance(state.sessions[ONE.id].state, Idle)


# Claude Code set the session running at 1000, after its last idle, and last set its status at 1100.
RUNNING = Running(Busy(), Stamp(1100), idled=Stamp(1000))


def taken(prompt: PromptId, written: Stamp | None, at: float = 12.0) -> Taken:
    """The first record under an id, as the tail reads it."""
    return Taken(ONE.id, prompt, written, at)


QUEUED = Prompted(ONE.id, at=5.0, mode=None, prompt=TURN)


def test_a_message_queued_while_a_turn_ran_is_its_own_turn_named_from_when_it_is_taken() -> None:
    """hands-status-bpp.44l, seen live on 2.1.282: the queued message's hook carries the running turn's id, and once that
    turn's Stop lands Claude Code runs it under a new id no hook names. It is marked while that Stop's hook holds Claude
    Code, after the turn before is handed over to be told."""
    state, tellings = told([QUEUED, STOP, taken(NEXT, Stamp(1500))], in_turn(RUNNING))
    assert state == in_turn(RUNNING, turn=NEXT)
    assert tellings == [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, "done"), Snapshot(ONE.id, ONE.cwd)]
    state, tellings = told([Stopped(ONE.id, "two", mode=None, prompt=NEXT, again=False), said(status.Idle(), at=14.0), Tick(20.0)], state)
    assert tellings == [Compare(ONE.id, again=False), Summarise(ONE.id, NEXT, "two")]
    assert state == holding(Idle(Stamp(2000), due=14.0 + IDLE_NUDGE_SECONDS, after=NEXT), Told(NEXT))


@pytest.mark.parametrize(("written", "opens"), [(Stamp(2003), True), (Stamp(2000), True), (Stamp(1990), False), (None, False)])
def test_a_prompt_taken_at_the_prompt_opens_a_turn_only_if_written_since_claude_code_last_said_idle(written: Stamp | None, opens: bool) -> None:
    """Claude Code sets idle for ~3 ms between a turn and the one queued behind it (2.1.282), and a read can land on it.
    An idle set after the record was written is that turn over, before any of it was read."""
    state, _ = told([STOP, said(status.Idle()), taken(NEXT, written)], in_turn(RUNNING))
    assert isinstance(state.sessions[ONE.id].turn, Opened) == opens


@pytest.mark.parametrize(("written", "opens"), [(Stamp(1500), True), (Stamp(1000), True), (Stamp(990), False), (None, False)])
def test_a_prompt_taken_while_running_with_no_turn_open_opens_one_only_if_written_since_the_idle_before_the_run(written: Stamp | None, opens: bool) -> None:
    """A record written before the idle the run began from is of a turn over by then."""
    state, _ = told([taken(NEXT, written)], holding(RUNNING, Told(TURN)))
    assert isinstance(state.sessions[ONE.id].turn, Opened) == opens


def test_a_queued_turn_whose_stop_lands_before_its_record_is_read_is_told_once() -> None:
    state, tellings = told([QUEUED, STOP, Stopped(ONE.id, "two", mode=None, prompt=NEXT, again=False), taken(NEXT, Stamp(1500)), said(status.Idle(), at=14.0), Tick(20.0)], in_turn(RUNNING))
    assert tellings == [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, "done"), Snapshot(ONE.id, ONE.cwd), Compare(ONE.id, again=False), Summarise(ONE.id, NEXT, "two")]
    assert state.sessions[ONE.id].turn == Told(NEXT)


def test_a_bang_command_is_running_from_its_status_and_claudes_answer_to_it_is_told_as_itself() -> None:
    """Seen live on 2.1.282 and 2.1.283: Claude Code is busy while the command runs, writes its record under a new id
    stamped before that busy, Claude answers it, and a Stop names it."""
    bang = PromptId("bang")
    at_prompt = holding(Idle(Stamp(2000), due=70.0, after=TURN), Told(TURN))
    state, tellings = told([said(Busy(), 3000, at=20.0)], at_prompt)
    assert (state.sessions[ONE.id].state, tellings) == (Running(Busy(), Stamp(3000), idled=Stamp(2000)), [])
    state, tellings = told([taken(bang, Stamp(2900), at=24.0)], state)
    assert (state.sessions[ONE.id].turn, tellings) == (Opened(bang), [])
    state, tellings = told([Stopped(ONE.id, "It slept.", mode=None, prompt=bang, again=False)], state)
    assert tellings == [Compare(ONE.id, again=False), Summarise(ONE.id, bang, "It slept.")]


def test_a_command_claude_code_runs_at_the_prompt_is_running_while_it_runs_and_told_once_it_is_over() -> None:
    """Seen live on 2.1.282: /compact is busy for its whole run and writes records under a new id, with no Stop. What is
    told of it is what happened in it, which for a command is nothing, and the narrator says nothing of that."""
    at_prompt = holding(Idle(Stamp(2000), due=70.0, after=TURN), Told(TURN))
    compact = PromptId("compact")
    state, _ = told([said(Busy(), 3000, at=20.0), taken(compact, Stamp(3001), at=20.1), Joined(ONE, "compact")], at_prompt)
    assert state == holding(Running(Busy(), Stamp(3000), idled=Stamp(2000)), Opened(compact))
    state, tellings = told([said(status.Idle(), 9000, at=30.0), Tick(40.0)], state)
    assert (state.sessions[ONE.id].state, tellings) == (Idle(Stamp(9000), due=30.0 + IDLE_NUDGE_SECONDS, after=compact), [Compare(ONE.id, again=False), Summarise(ONE.id, compact, None)])


def test_a_stop_with_nothing_queued_behind_it_marks_nothing() -> None:
    """A message 2.1.281 took into the running turn is in that turn, and waits behind nothing: a mark taken at its Stop
    would be read, however much later, as the start of the next turn no prompt marks."""
    state, tellings = told([QUEUED, taken(NEXT, Stamp(1200)), Stopped(ONE.id, "done", mode=None, prompt=NEXT, again=False)], in_turn(RUNNING))
    assert tellings == [Compare(ONE.id, again=False), Summarise(ONE.id, NEXT, "done")]
    assert state.sessions[ONE.id].turn == Told(TURN, frozenset({NEXT}))


def test_a_prompt_taken_before_any_status_is_read_opens_nothing() -> None:
    """A transcript read from its start, as the daemon attaching to a session reads it: history, not a turn."""
    attached = holding(Unreported())
    assert reduce(attached, taken(NEXT, Stamp(1500))) == (attached, [])


def test_attached_mid_turn_it_is_running_and_the_turn_running_and_the_one_queued_behind_it_open_and_nothing_before_them() -> None:
    """hands-status-hjr: a session attached while busy is running from its first status read. Records written before
    Claude Code set it are history; the queued turn's is written after the running one's Stop."""
    state, _ = told([said(Busy(), 3000, at=1.0), taken(PromptId("old"), Stamp(1500))], holding(Unreported()))
    assert state == holding(Running(Busy(), Stamp(3000), idled=Stamp(3000)))
    state, _ = told([taken(TURN, Stamp(3003), at=2.0), Stopped(ONE.id, "done", mode=None, prompt=TURN, again=False), taken(NEXT, Stamp(9000), at=20.0)], state)
    assert state == holding(Running(Busy(), Stamp(3000), idled=Stamp(3000)), Opened(NEXT))


@pytest.mark.parametrize("stop", [Stopped(ONE.id, "other", mode=None, prompt=NEXT, again=False), Stopped(ONE.id, "done", mode=None, prompt=TURN, again=False)])
def test_a_stop_while_a_telling_waits_tells_that_turn_and_takes_no_name_from_it(stop: Stopped) -> None:
    state, _ = told([said(status.Idle()), stop])
    assert state.sessions[ONE.id].turn == Told(TURN)


@pytest.mark.parametrize("end", [Ended(ONE.id, "other"), Ended(ONE.id, "prompt_input_exit"), Died(ONE), MovedOn(ONE)])
def test_a_turn_left_untold_is_told_before_its_session_is_said_to_be_gone(end: Event) -> None:
    state, _ = told([said(status.Idle())])
    state, effects = reduce(state, end)
    tellings = [effect for effect in effects if isinstance(effect, Compare | Summarise | SessionGone)]
    assert tellings[:2] == TOLD and all(isinstance(e, SessionGone) for e in tellings[2:])
    assert reduce(state, Tick(20.0))[1] == []


@pytest.mark.parametrize("source", ["compact", "resume"])
def test_a_restart_while_a_telling_waits_keeps_the_turns_late_stop_ending_nothing_but_the_telling(source: StartSource) -> None:
    _, tellings = told([said(status.Idle()), Joined(ONE, source), STOP, Tick(20.0)])
    assert tellings == [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, "done")]


def test_a_compaction_between_a_turns_telling_and_its_late_stop_tells_it_once() -> None:
    """hands-status-hjr: compaction used to drop the told turn's name, so its late Stop told it again."""
    _, tellings = told([said(status.Idle()), INTERRUPT, Joined(ONE, "compact"), STOP, Tick(20.0)])
    assert tellings == TOLD


@pytest.mark.parametrize("event", [INTERRUPT, taken(NEXT, Stamp(9000)), Continued(ONE.id, was=TURN, now=NEXT)])
def test_a_late_record_of_a_session_gone_is_not_audited_as_after_its_end(event: Event) -> None:
    state, _ = told([said(status.Idle()), Ended(ONE.id, "prompt_input_exit")])
    assert reduce(state, event) == (state, [])


def test_a_session_first_read_at_its_prompt_is_nudged_by_idle_prompt_alone() -> None:
    """Attached after a restart, or just started: it went idle before hands followed it, so no turn's end times a nudge."""
    state, effects = reduce(holding(Unreported()), WENT_IDLE)
    assert (state, effects) == (holding(Idle(Stamp(2000), due=None, after=None)), [])
    assert reduce(state, Tick(10.0 + 100 * IDLE_NUDGE_SECONDS))[1] == []
    assert nudged(state) == [NUDGE]


def test_a_turn_told_before_its_idle_is_read_starts_an_idle_period_nudged_again() -> None:
    """Its busy fell between two reads: the prompt and the Stop are heard, then an idle with a new stamp."""
    after_nudge = holding(replace(IDLE, after=TURN, nudged=True), Told(TURN))
    state, _ = told([Prompted(ONE.id, at=100.0, mode=None, prompt=NEXT), Stopped(ONE.id, "Done.", mode=None, prompt=NEXT, again=False), said(status.Idle(), 3000, at=100.1)], after_nudge)
    assert state.sessions[ONE.id].state == Idle(Stamp(3000), due=100.1 + IDLE_NUDGE_SECONDS, after=NEXT)
    assert nudged(state) == [NUDGE]


def test_a_turn_heard_and_ended_inside_one_idle_read_starts_an_idle_period_nudged_again() -> None:
    """A prompt cancelled during its hooks after a nudge: Claude Code's busy is never read, and the next idle is new."""
    state, _ = told([Prompted(ONE.id, at=100.0, mode=None, prompt=NEXT), said(status.Idle(), at=100.1)], holding(replace(IDLE, nudged=True), Told(TURN)))
    assert state.sessions[ONE.id].state == Idle(Stamp(2000), due=100.1 + IDLE_NUDGE_SECONDS, after=NEXT)
    assert nudged(state) == [NUDGE]
