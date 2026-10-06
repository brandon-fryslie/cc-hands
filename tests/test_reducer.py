"""The session lifecycle as a table: registry before, event, registry after, effects. No I/O."""

from dataclasses import replace
from pathlib import Path

import pytest

from hands.core.effects import (
    AfterEnd,
    Holding,
    Unmatched,
    Unclosed,
    Unsettled,
    Audit,
    Deny,
    Effect,
    Narrate,
    Asking,
    DeadlineNear,
    Expired,
    Heard,
    ModeChanged,
    Note,
    Reply,
    SessionGone,
    Speak,
    Compare,
    Snapshot,
    Summarise,
    Unregistered,
    Withdraw,
)
from hands.core.events import Abandoned, Attached, Closed, Died, Ended, EndReason, MovedOn, Event, Interrupted, Continued, Taken, Joined, Read, PermissionRequested, Prompted, Progressed, SessionEvent, StartSource, StatusReported, Stopped, Tick, ToolFinished
from hands.core import progress
from hands.core.reducer import EXPIRED_MESSAGE, UNTOLD, WARNING_LEAD_SECONDS, reduce
from hands.core.session import (
    AskedQuestion,
    Dialog,
    Gone,
    Held,
    Idle,
    Known,
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
    UnknownMode,
    Unreported,
    Untold,
)
from hands.core.status import Busy, Report, Shell, Stamp, Status, Unknown, Waiting
from hands.core import status

# When hands heard a Stop, on the clock Claude Code stamps a status with.
STOP_HEARD = Stamp(1500)
STOP_REQUEST = RequestId("stop")

TIMEOUT = 60.0
ONE = Membership(SessionId("s1"), pid=4242, cwd=Path("/code/a"), transcript=Path("/t/s1.jsonl"))
# A Stop's hook let go, once what deciding it called for is done.
LET_STOP = Reply(ONE.id, STOP_REQUEST, Withdraw())
TWO = Membership(SessionId("s2"), pid=5353, cwd=Path("/code/b"), transcript=Path("/t/s2.jsonl"))
BASH = Permission(tool="Bash", input={"command": "ls"})
TURN = PromptId("p1")
NEXT = PromptId("p2")

IDLE = Idle(status.Idle(), Stamp(900), after=None)
BUSY = Running(Busy(), Stamp(1000), idled=Stamp(1000))
AT_DIALOG = Running(Waiting("permission prompt"), Stamp(1100), idled=Stamp(1000))
HELD = Held(on=BASH, request=RequestId("r0"), deadline=61.0, warned=False)

LIVE: list[SessionState] = [Unreported(), IDLE, BUSY, AT_DIALOG]
SESSION_EVENTS: list[SessionEvent] = [
    Prompted(ONE.id, at=5.0, mode=None, prompt=TURN),
    Stopped(ONE.id, None, mode=None, prompt=TURN, again=False, heard=STOP_HEARD, request=STOP_REQUEST),
    Closed(ONE.id, TURN, "Done."),
    PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), on=BASH, mode=None),
    ToolFinished(ONE.id, at=5.0, call=BASH, mode=None),
    Ended(ONE.id, "prompt_input_exit"),
    StatusReported(ONE.id, Report(Busy(), Stamp(1000)), at=5.0),
]
# What hooks and the transcript say: which turn it is, what it did, and which request it waits on.
HEARD: list[SessionEvent] = [
    Prompted(ONE.id, at=5.0, mode=None, prompt=NEXT),
    Stopped(ONE.id, "Done.", mode=None, prompt=TURN, again=False, heard=STOP_HEARD, request=STOP_REQUEST),
    Stopped(ONE.id, "Again.", mode=None, prompt=TURN, again=True, heard=STOP_HEARD, request=STOP_REQUEST),
    Closed(ONE.id, TURN, "Done."),
    Taken(ONE.id, NEXT, Stamp(5000), at=5.0),
    Interrupted(ONE.id, TURN, at=5.0),
    Continued(ONE.id, was=TURN, now=NEXT),
    PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), on=BASH, mode=None),
    ToolFinished(ONE.id, at=5.0, call=BASH, mode=None),
    Progressed(ONE.id, (TURN,), (progress.Doing(progress.RUNNING, "run the test suite"),), at=5.0),
]


def registry(*sessions: Known) -> Registry:
    return Registry(permission_deadline=TIMEOUT, sessions={s.membership.id: s for s in sessions}, drafts={})


GONE = Registry(permission_deadline=TIMEOUT, sessions={ONE.id: Gone(ONE)}, drafts={})


def live(registry: Registry) -> Session:
    """Session one as the registry holds it, which the test expects to be live."""
    match registry.sessions[ONE.id]:
        case Session() as session:
            return session
        case Gone():
            raise AssertionError("session one ended")


def holding(state: SessionState, turn: Turn = Told(), dialog: Dialog | None = None, earlier: frozenset[PromptId] = frozenset()) -> Registry:
    return registry(Session(ONE, state, mode=None, turn=turn, dialog=dialog, earlier=earlier))


def in_turn(state: SessionState = BUSY, dialog: Dialog | None = None, turn: PromptId = TURN) -> Registry:
    return holding(state, Opened(turn), dialog)


def said(status: Status, stamp: int = 2000, at: float = 10.0) -> StatusReported:
    return StatusReported(ONE.id, Report(status, Stamp(stamp)), at=at)


def test_a_start_registers_the_session_with_no_status_read_yet() -> None:
    assert reduce(registry(), Joined(ONE, "startup")) == (holding(Unreported()), [])


@pytest.mark.parametrize("before", LIVE)
@pytest.mark.parametrize("turn", [Told(TURN), Opened(TURN), Untold(TURN, frozenset(), by=Stamp(99))])
@pytest.mark.parametrize("event", HEARD)
def test_no_hook_or_record_moves_a_session_between_running_and_not(before: SessionState, turn: Turn, event: SessionEvent) -> None:
    """[LAW:one-source-of-truth] hooks and records say which turn it is and what it did, never whether it runs."""
    assert live(reduce(holding(before, turn), event)[0]).state == before


@pytest.mark.parametrize(
    ("before", "event", "after"),
    [
        (Unreported(), said(Busy(), 3000), Running(Busy(), Stamp(3000), idled=None)),
        # Running since the idle before it: a record written after that idle is of this run.
        (IDLE, said(Busy(), 3000), Running(Busy(), Stamp(3000), idled=IDLE.stamp)),
        # At its prompt, with a background shell command running: no turn runs (2.1.289).
        (BUSY, said(Shell(), 3000, at=10.0), Idle(Shell(), Stamp(3000), after=None)),
        (IDLE, said(Shell(), 3000), replace(IDLE, status=Shell(), stamp=Stamp(3000))),
        # Its background shell over: the same idle period, and the notification turn after it runs from that idle.
        (replace(IDLE, status=Shell()), said(status.Idle(), 3000), replace(IDLE, stamp=Stamp(3000))),
        (replace(IDLE, status=Shell()), said(Busy(), 3000), Running(Busy(), Stamp(3000), idled=IDLE.stamp)),
        (BUSY, said(Waiting("permission prompt"), 3000), Running(Waiting("permission prompt"), Stamp(3000), idled=BUSY.idled)),
        (AT_DIALOG, said(Busy(), 3000), Running(Busy(), Stamp(3000), idled=AT_DIALOG.idled)),
        (BUSY, said(Unknown("dreaming"), 3000), Running(Unknown("dreaming"), Stamp(3000), idled=BUSY.idled)),
        (BUSY, said(status.Idle(), 3000, at=10.0), Idle(status.Idle(), Stamp(3000), after=None)),
        (Unreported(), said(status.Idle(), 3000, at=10.0), Idle(status.Idle(), Stamp(3000), after=None)),
        # Set idle again, or idle, busy and idle between two reads: the same idle period.
        (IDLE, said(status.Idle(), 3000, at=10.0), replace(IDLE, stamp=Stamp(3000))),
    ],
)
def test_a_status_read_moves_the_session_to_what_claude_code_says_it_is_doing(before: SessionState, event: StatusReported, after: SessionState) -> None:
    assert reduce(holding(before), event) == (holding(after), [])


@pytest.mark.parametrize("before", LIVE)
def test_an_end_is_gone_from_any_state(before: SessionState) -> None:
    assert reduce(holding(before), Ended(ONE.id, "prompt_input_exit"))[0] == GONE


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
    assert [effect for effect in reduce(in_turn(AT_DIALOG, HELD), event)[1] if isinstance(effect, Reply)] == [Reply(ONE.id, RequestId("r0"), Withdraw())]


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
    assert reduce(holding(BUSY, Told(TURN)), Prompted(ONE.id, at=5.0, mode=None, prompt=NEXT)) == (holding(BUSY, Opened(NEXT), earlier=frozenset({TURN})), [Snapshot(ONE.id, ONE.cwd)])


@pytest.mark.parametrize("before", [holding(IDLE), holding(BUSY), in_turn()])
def test_a_finished_turn_is_summarised_from_the_session_transcript(before: Registry) -> None:
    """The turn it names, whether hands had it open or never heard it open. It runs until Claude Code says it is idle."""
    state = live(before).state
    assert reduce(before, Stopped(ONE.id, "Done.", mode=None, prompt=TURN, again=False, heard=STOP_HEARD, request=STOP_REQUEST)) == (holding(state, Told(TURN)), [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, "Done."), LET_STOP])


def test_a_turn_that_stops_while_a_hook_is_held_is_told_and_the_hook_let_go_once_claude_code_says_idle() -> None:
    state, effects = reduce(in_turn(AT_DIALOG, HELD), Stopped(ONE.id, None, mode=None, prompt=TURN, again=False, heard=STOP_HEARD, request=STOP_REQUEST))
    assert effects == [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, None), LET_STOP]
    assert reduce(state, said(status.Idle()))[1] == [Reply(ONE.id, RequestId("r0"), Withdraw())]


def test_a_turn_is_compared_before_it_is_handed_over_to_be_summarised() -> None:
    """Effects are performed in the order they are given, and this order is the whole of why it is two effects.

    A summary is made one at a time and takes seconds. Read the repository when the summary is made rather
    than when the turn stopped, and it holds whatever the next turn has since started doing.
    """
    effects = reduce(in_turn(), Stopped(ONE.id, None, mode=None, prompt=TURN, again=False, heard=STOP_HEARD, request=STOP_REQUEST))[1]
    assert effects.index(Compare(ONE.id, again=False)) < effects.index(Summarise(ONE.id, TURN, None))


def test_a_prompt_marks_where_the_repository_stands_before_the_turn_can_change_it() -> None:
    """The mark is what the turn's changes are read against, so it is taken as the turn opens, not during it."""
    assert reduce(holding(IDLE), Prompted(ONE.id, at=5.0, mode=None, prompt=TURN))[1] == [Snapshot(ONE.id, ONE.cwd)]


def test_a_turn_that_goes_on_after_another_stop_hook_blocked_its_stop_runs_until_claude_code_says_idle() -> None:
    """Seen live on 2.1.282 (hands-status-bpp.55k): the file says busy between the blocked Stop and the one after it."""
    heard: list[Effect] = []
    state = in_turn()
    for event in [Stopped(ONE.id, "First.", mode=None, prompt=TURN, again=False, heard=STOP_HEARD, request=STOP_REQUEST), Stopped(ONE.id, "Second.", mode=None, prompt=TURN, again=True, heard=STOP_HEARD, request=STOP_REQUEST)]:
        state, effects = reduce(state, event)
        heard += effects
        assert live(state).state == BUSY
    assert heard == [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, "First."), LET_STOP, Compare(ONE.id, again=True), Summarise(ONE.id, TURN, "Second."), LET_STOP]
    assert reduce(state, said(status.Idle())) == (holding(Idle(status.Idle(), Stamp(2000), after=TURN), Told(TURN)), [])


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
    assert effects == [Speak(DeadlineNear(ONE.id, RequestId("r0"), BASH, remaining=8.0))]
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
        Speak(DeadlineNear(ONE.id, RequestId("r"), BASH, remaining=WARNING_LEAD_SECONDS)),
        Reply(ONE.id, RequestId("r"), Deny(EXPIRED_MESSAGE)),
        Speak(Expired(ONE.id, BASH)),
    ]


def test_a_request_heard_again_keeps_its_deadline_and_is_warned_of_once() -> None:
    asked = PermissionRequested(ONE.id, at=0.0, request=RequestId("r"), on=BASH, mode=None)
    state, _ = reduce(holding(AT_DIALOG), asked)
    state, warned = reduce(state, Tick(at=51.0))
    again, effects = reduce(state, replace(asked, at=52.0))
    assert (again, effects) == (state, [])
    assert warned == [Speak(DeadlineNear(ONE.id, RequestId("r"), BASH, remaining=9.0))]
    assert reduce(again, Tick(at=53.0)) == (again, [])


SAID_TWICE_FROM: list[Registry] = [
    holding(IDLE),
    holding(BUSY),
    in_turn(),
    holding(IDLE, Untold(TURN, frozenset(), Stamp(5000))),
    holding(IDLE, Told(TURN, frozenset())),
    in_turn(AT_DIALOG, HELD),
    in_turn(AT_DIALOG, replace(HELD, warned=True)),
    in_turn(AT_DIALOG, LetGo(BASH)),
]


@pytest.mark.parametrize("before", SAID_TWICE_FROM)
@pytest.mark.parametrize(
    "event",
    [
        *HEARD,
        *SESSION_EVENTS,
        PermissionRequested(ONE.id, at=5.0, request=HELD.request, on=BASH, mode=None),
        Died(ONE),
        Ended(ONE.id, "other"),
        said(status.Idle()),
        Read(ONE.id, Stamp(99_999)),
        Tick(at=55.0),
        Tick(at=61.0),
    ],
)
def test_the_same_event_heard_twice_is_said_once(before: Registry, event: Event) -> None:
    # A transition is said, never a state, so the second hearing finds nothing new to say. A second Summarise asks the
    # tail only for what it has not told, by record, which a turn told has none of (see tests/test_narrator.py).
    once, _ = reduce(before, event)
    _, again = reduce(once, event)
    assert [effect for effect in again if isinstance(effect, Heard | SessionGone)] == []


def test_a_tick_moves_only_sessions_waiting_on_a_deadline() -> None:
    before = registry(Session(ONE, AT_DIALOG, mode=None, dialog=HELD), Session(TWO, BUSY, mode=None))
    after, _ = reduce(before, Tick(at=100.0))
    assert after == registry(Session(ONE, AT_DIALOG, mode=None), Session(TWO, BUSY, mode=None))


@pytest.mark.parametrize(
    "before",
    [
        Session(ONE, IDLE, mode="plan", turn=Untold(TURN, frozenset({NEXT}), by=Stamp(11))),
        Session(ONE, IDLE, mode=None, turn=Told(TURN)),
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


@pytest.mark.parametrize("before", [*[holding(state, Opened(TURN)) for state in LIVE], GONE])
@pytest.mark.parametrize("source", ["startup", "resume", "clear", "fork"])
def test_any_start_but_compaction_waits_for_its_status_to_be_read(before: Registry, source: StartSource) -> None:
    # a session resumed after a crash never sent the Stop or SessionEnd the registry is still waiting for
    assert reduce(before, Joined(ONE, source))[0] == holding(Unreported())


def test_a_fork_joins_as_a_new_session_beside_the_one_it_was_forked_from() -> None:
    parent = Session(ONE, IDLE, mode=None)
    assert reduce(registry(parent), Joined(TWO, "fork"))[0] == registry(parent, Session(TWO, Unreported(), mode=None))


def test_an_ended_session_compacting_waits_for_its_status_to_be_read() -> None:
    assert reduce(GONE, Joined(ONE, "compact")) == (holding(Unreported()), [])


@pytest.mark.parametrize("event", SESSION_EVENTS)
def test_an_ended_session_ignores_late_hooks_and_audits_them(event: SessionEvent) -> None:
    assert reduce(GONE, event) == (GONE, [Audit(AfterEnd(event)), *let_go(event)])


@pytest.mark.parametrize("event", SESSION_EVENTS)
def test_an_event_for_a_session_that_never_joined_changes_nothing_and_is_audited(event: SessionEvent) -> None:
    before = registry(Session(TWO, IDLE, mode=None))
    assert reduce(before, event) == (before, [Audit(Unregistered(event)), *let_go(event)])


def let_go(event: SessionEvent) -> list[Effect]:
    # A permission or Stop hook the registry cannot block on is released at once instead of hanging.
    match event:
        case PermissionRequested(request=request) | Stopped(request=request):
            return [Reply(ONE.id, request, Withdraw())]
        case _:
            return []


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


@pytest.mark.parametrize("before", [holding(AT_DIALOG, dialog=replace(HELD, request=RequestId("newer"))), holding(IDLE), GONE])
def test_an_abandoned_request_the_session_no_longer_waits_on_changes_nothing(before: Registry) -> None:
    assert reduce(before, Abandoned(ONE.id, RequestId("r0"), at=30.0)) == (before, [])


def test_an_event_moves_only_its_own_session() -> None:
    before = registry(Session(ONE, IDLE, mode=None), Session(TWO, IDLE, mode=None))
    after, _ = reduce(before, Prompted(TWO.id, at=2.0, mode=None, prompt=TURN))
    assert after == registry(Session(ONE, IDLE, mode=None), Session(TWO, IDLE, mode=None, turn=Opened(TURN)))


def test_live_is_every_session_that_has_not_ended() -> None:
    both = registry(Session(ONE, AT_DIALOG, mode=None, dialog=HELD), Gone(TWO))
    assert both.live() == [Session(ONE, AT_DIALOG, mode=None, dialog=HELD)]


def test_a_file_for_a_session_never_heard_of_attaches_it_to_be_read() -> None:
    assert reduce(registry(), Attached(ONE)) == (holding(Unreported()), [])


@pytest.mark.parametrize("before", [*[holding(state) for state in LIVE], GONE])
def test_a_file_for_a_session_already_known_changes_nothing(before: Registry) -> None:
    moved = replace(ONE, cwd=Path("/code/elsewhere"))
    assert reduce(before, Attached(moved)) == (before, [])


@pytest.mark.parametrize("ended", [Died(ONE), MovedOn(ONE)])
def test_a_session_this_run_never_listed_ending_says_nothing(ended: Event) -> None:
    assert reduce(registry(), ended) == (registry(), [])


@pytest.mark.parametrize("dialog", [None, HELD])
@pytest.mark.parametrize("before", LIVE)
def test_a_live_session_whose_process_died_is_gone_and_spoken_and_its_hook_let_go(before: SessionState, dialog: Dialog | None) -> None:
    released = [] if dialog is None else [Reply(ONE.id, HELD.request, Withdraw())]
    assert reduce(holding(before, dialog=dialog), Died(ONE)) == (GONE, [*released, SessionGone(ONE.id)])


def test_a_session_already_gone_dying_again_says_nothing() -> None:
    assert reduce(GONE, Died(ONE)) == (GONE, [])


def test_a_dead_process_that_is_no_longer_the_sessions_changes_nothing() -> None:
    resumed = holding(BUSY)
    assert reduce(resumed, Died(replace(ONE, pid=ONE.pid + 1))) == (resumed, [])


@pytest.mark.parametrize("dialog", [None, HELD])
@pytest.mark.parametrize("before", LIVE)
def test_a_session_whose_process_moved_on_is_gone_silently_and_its_hook_let_go(before: SessionState, dialog: Dialog | None) -> None:
    released = [] if dialog is None else [Reply(ONE.id, HELD.request, Withdraw())]
    assert reduce(holding(before, dialog=dialog), MovedOn(ONE)) == (GONE, released)


def test_a_file_on_a_pid_another_session_holds_attaches_beside_it_the_sweep_decides_which_is_over() -> None:
    both = reduce(holding(IDLE), Attached(replace(TWO, pid=ONE.pid)))[0]
    assert [session.membership.id for session in both.live()] == [ONE.id, TWO.id]


@pytest.mark.parametrize("dialog", [None, HELD])
@pytest.mark.parametrize("before", LIVE)
def test_a_session_whose_terminal_closed_is_gone_and_spoken(before: SessionState, dialog: Dialog | None) -> None:
    released = [] if dialog is None else [Reply(ONE.id, HELD.request, Withdraw())]
    assert reduce(holding(before, dialog=dialog), Ended(ONE.id, "other")) == (GONE, [*released, SessionGone(ONE.id)])


@pytest.mark.parametrize("reason", ["clear", "resume", "logout", "prompt_input_exit", "bypass_permissions_disabled"])
def test_a_session_ended_at_the_keyboard_is_not_spoken(reason: EndReason) -> None:
    assert reduce(holding(IDLE), Ended(ONE.id, reason)) == (GONE, [])


def test_a_closed_terminal_after_the_sweep_found_the_session_dead_says_nothing_more() -> None:
    event = Ended(ONE.id, "other")
    assert reduce(GONE, event) == (GONE, [Audit(AfterEnd(event))])


WENT_IDLE = said(status.Idle(), 2000, at=10.0)


def test_a_session_at_its_prompt_prompted_again_marks_the_repository_its_turn_starts_from() -> None:
    assert reduce(holding(IDLE), Prompted(ONE.id, at=5.0, mode=None, prompt=TURN)) == (in_turn(IDLE), [Snapshot(ONE.id, ONE.cwd)])


REPORTING: list[SessionEvent] = [
    Prompted(ONE.id, at=5.0, mode="plan", prompt=TURN),
    Stopped(ONE.id, None, mode="plan", prompt=TURN, again=False, heard=STOP_HEARD, request=STOP_REQUEST),
    PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), on=BASH, mode="plan"),
    ToolFinished(ONE.id, at=5.0, call=BASH, mode="plan"),
]


def moded(state: SessionState, mode: Mode | None, dialog: Dialog | None = None) -> Registry:
    return registry(Session(ONE, state, mode=mode, dialog=dialog))


@pytest.mark.parametrize("before", LIVE)
@pytest.mark.parametrize("event", REPORTING)
def test_every_hook_that_reports_a_mode_sets_the_sessions_mode(before: SessionState, event: SessionEvent) -> None:
    after, effects = reduce(moded(before, "default"), event)
    assert live(after).mode == "plan"
    assert Note(ModeChanged(ONE.id, "plan")) in effects


def test_a_hook_that_reports_no_mode_keeps_the_one_last_reported() -> None:
    after, effects = reduce(moded(BUSY, "acceptEdits"), Stopped(ONE.id, None, mode=None, prompt=TURN, again=False, heard=STOP_HEARD, request=STOP_REQUEST))
    assert live(after).mode == "acceptEdits"
    assert not [effect for effect in effects if isinstance(effect, Note)]


def test_a_mode_reported_again_unchanged_is_not_noted() -> None:
    _, effects = reduce(moded(IDLE, "plan"), Prompted(ONE.id, at=5.0, mode="plan", prompt=TURN))
    assert not [effect for effect in effects if isinstance(effect, Note)]


def test_a_mode_reported_for_the_first_time_is_noted_because_a_resumed_session_may_be_in_another_mode() -> None:
    after, effects = reduce(moded(IDLE, None), Prompted(ONE.id, at=5.0, mode="auto", prompt=TURN))
    assert live(after).mode == "auto"
    assert Note(ModeChanged(ONE.id, "auto")) in effects


def test_a_request_is_narrated_after_the_mode_it_was_asked_in_is_noted() -> None:
    _, effects = reduce(moded(BUSY, "default"), PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), on=BASH, mode="acceptEdits"))
    assert effects == [Note(ModeChanged(ONE.id, "acceptEdits")), Narrate(Asking(ONE.id, RequestId("r1"), BASH))]


def test_a_mode_hands_does_not_know_is_noted_like_any_other() -> None:
    _, effects = reduce(moded(IDLE, "default"), Prompted(ONE.id, at=5.0, mode=UnknownMode("ultraplan"), prompt=TURN))
    assert Note(ModeChanged(ONE.id, UnknownMode("ultraplan"))) in effects


@pytest.mark.parametrize("before", LIVE)
def test_compaction_keeps_the_mode_because_it_keeps_the_process(before: SessionState) -> None:
    assert live(reduce(moded(before, "acceptEdits"), Joined(replace(ONE, pid=7777), "compact"))[0]).mode == "acceptEdits"


@pytest.mark.parametrize("source", ["startup", "resume", "clear"])
def test_any_other_start_has_not_reported_a_mode(source: StartSource) -> None:
    assert live(reduce(moded(IDLE, "acceptEdits"), Joined(ONE, source))[0]).mode is None


def test_a_session_keeps_its_mode_through_a_deadline_and_an_abandoned_hook() -> None:
    assert live(reduce(moded(AT_DIALOG, "plan", HELD), Tick(at=100.0))[0]).mode == "plan"
    assert live(reduce(moded(AT_DIALOG, "plan", HELD), Abandoned(ONE.id, RequestId("r0"), at=9.0))[0]).mode == "plan"


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
    assert state == holding(Idle(status.Idle(), Stamp(2000), after=TURN), Untold(TURN, frozenset(), by=WINDOW))
    state, tellings = told([Prompted(ONE.id, at=9.0, mode=None, prompt=NEXT), Taken(ONE.id, NEXT, None, 9.0)], state)
    # Told as itself, which says nothing of a prompt that never ran: no record carries its id.
    assert (live(state).turn, tellings) == (Opened(NEXT), [*TOLD, Snapshot(ONE.id, ONE.cwd)])


def test_a_prompt_taken_and_stopped_before_any_of_it_was_read_is_told_as_itself() -> None:
    """hands-keyboard-gxr.90g: p1 is taken and interrupted inside one tail period, so Claude Code's idle finds it still
    unread. Its record and its interrupt, read after, tell it once, as itself; the next prompt opens its own turn."""
    state, tellings = told([Prompted(ONE.id, at=5.0, mode=None, prompt=TURN), said(status.Idle()), Taken(ONE.id, TURN, None, 9.0)], holding(IDLE))
    assert (live(state).turn, tellings) == (Untold(TURN, frozenset(), by=WINDOW), [Snapshot(ONE.id, ONE.cwd)])
    state, tellings = told([INTERRUPT, STOP, PROMPT, Tick(20.0)], state)
    assert (live(state).turn, tellings) == (Opened(NEXT), [*TOLD, Snapshot(ONE.id, ONE.cwd)])


def test_a_stop_is_told_as_the_turn_its_hook_names() -> None:
    assert reduce(in_turn(), Stopped(ONE.id, "Done.", mode=None, prompt=TURN, again=False, heard=STOP_HEARD, request=STOP_REQUEST))[1] == [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, "Done."), LET_STOP]


def test_the_stop_of_a_turn_over_before_the_next_prompt_ends_nothing() -> None:
    """Applied late, it would end the turn now open and spend its mark."""
    late = Stopped(ONE.id, "Done.", mode="plan", prompt=TURN, again=False, heard=STOP_HEARD, request=STOP_REQUEST)
    before, _ = reduce(in_turn(), Prompted(ONE.id, at=5.0, mode=None, prompt=NEXT))
    after, effects = reduce(before, late)
    assert effects == [Note(ModeChanged(ONE.id, "plan")), Audit(Unmatched(ONE.id, late.prompt)), LET_STOP] and live(after).turn == Opened(NEXT)


@pytest.mark.parametrize(
    ("turn", "again", "earlier"),
    [(TURN, False, frozenset({TURN})), (TURN, True, frozenset({TURN})), (None, False, frozenset[PromptId]()), (PromptId("q"), True, frozenset[PromptId]())],
)
@pytest.mark.parametrize("state", [IDLE, BUSY])
def test_a_stop_with_no_turn_open_tells_the_turn_it_names(state: SessionState, turn: PromptId | None, again: bool, earlier: frozenset[PromptId]) -> None:
    """The turn a session was in when hands attached to it, one it was never heard to open, or the last one stopping
    again after another Stop hook blocked its Stop, which Claude Code stops under the same id."""
    after, effects = reduce(holding(state, Told(turn)), Stopped(ONE.id, "Done.", mode=None, prompt=PromptId("q"), again=again, heard=STOP_HEARD, request=STOP_REQUEST))
    assert after == holding(state, Told(PromptId("q")), earlier=earlier)
    # Only the turn hands told before is read against where its last Stop's reading ended: no prompt marked where it went
    # on. One stopping again whose first Stop hands never heard is told whole, and the last reading is another turn's.
    assert effects == [Compare(ONE.id, again=turn == PromptId("q")), Summarise(ONE.id, PromptId("q"), "Done."), LET_STOP]


def test_the_stop_of_a_turn_already_told_ends_nothing_and_keeps_the_idle_period_it_lands_in() -> None:
    """An interrupted turn's Stop can land after it was told: told again, it would be heard twice."""
    assert reduce(holding(IDLE, Told(TURN)), STOP) == (holding(IDLE, Told(TURN)), [Audit(Unmatched(ONE.id, STOP.prompt)), LET_STOP])


def test_a_stop_naming_an_id_no_record_read_through_it_names_is_audited_and_leaves_the_turn_to_its_own_end() -> None:
    """A flush's id whose record the tail never hands on: the turn is ended by the idle Claude Code sets after the Stop,
    and told once the transcript is read through, without the reply; the Stop, read through too, is a line."""
    state, tellings = told([Stopped(ONE.id, "Done.", mode=None, prompt=PromptId("q"), again=False, heard=STOP_HEARD, request=STOP_REQUEST)])
    assert (live(state).turn, tellings) == (Opened(TURN), [])
    state, effects = reduce(*told([said(status.Idle())], state)[:1], Read(ONE.id, WINDOW))
    assert [e for e in effects if isinstance(e, Audit | Compare | Summarise)] == [*TOLD, Audit(Unmatched(ONE.id, PromptId("q")))]
    assert (live(state).turn, live(state).unnamed) == (Told(TURN), ())


def test_the_stop_of_a_turn_gone_on_under_a_flushed_id_ends_it_before_claude_answered_under_it() -> None:
    state, _ = reduce(in_turn(), Taken(ONE.id, PromptId("q"), None, 9.0))
    assert reduce(state, Stopped(ONE.id, "Done.", mode=None, prompt=PromptId("q"), again=False, heard=STOP_HEARD, request=STOP_REQUEST))[1] == [Compare(ONE.id, again=False), Summarise(ONE.id, PromptId("q"), "Done."), LET_STOP]


@pytest.mark.parametrize("before", [in_turn(), in_turn(AT_DIALOG, HELD), in_turn(AT_DIALOG, LetGo(BASH))])
def test_a_record_naming_another_turn_while_one_is_open_ends_nothing(before: Registry) -> None:
    """A flush's id: the turn goes on under it, and its Stop ends it."""
    after, effects = reduce(before, Taken(ONE.id, NEXT, None, 9.0))
    turn = live(after).turn
    assert isinstance(turn, Opened) and (turn.turn, turn.others) == (TURN, {NEXT})
    assert not any(isinstance(effect, Compare | Summarise | Snapshot) for effect in effects)
    assert live(reduce(after, Stopped(ONE.id, "Next.", mode=None, prompt=NEXT, again=False, heard=STOP_HEARD, request=STOP_REQUEST))[0]).turn == Told(TURN, frozenset({NEXT}))


def test_a_prompts_record_read_before_its_own_late_hook_opens_the_turn_and_the_session_runs_it_throughout() -> None:
    """hands-status-tlo.tmp: the shim gave up on a slow daemon, so Claude Code wrote the prompt's record before hands
    heard its hook. The record opens the turn, the status says it runs, and the hook heard late opens nothing more."""
    at_prompt = holding(Idle(status.Idle(), Stamp(900), after=TURN), Told(TURN))
    state, tellings = told([Taken(ONE.id, NEXT, Stamp(1500), 5.0), said(Busy(), 1400, at=5.1), Prompted(ONE.id, at=7.0, mode=None, prompt=NEXT)], at_prompt)
    assert (live(state).turn.turn, isinstance(live(state).state, Running), tellings) == (NEXT, True, [])
    _, tellings = told([Stopped(ONE.id, "done", mode=None, prompt=NEXT, again=False, heard=STOP_HEARD, request=STOP_REQUEST)], state)
    assert [telling for telling in tellings if isinstance(telling, Summarise)] == [Summarise(ONE.id, NEXT, "done")]


def test_an_escape_and_a_new_prompt_inside_one_status_read_open_two_turns() -> None:
    """hands-status-tlo.tmp, seen live on 2.1.283: Claude Code set idle at the Escape and busy for the next prompt 93 ms
    later, so no status read saw the idle. The prompt's hook carries an id of its own, which Claude Code gives only at
    its prompt: the turn before is told as itself, compared before the new one is marked, and the new one opens."""
    state, tellings = told([Prompted(ONE.id, at=9.0, mode=None, prompt=NEXT), said(Busy(), 2100), INTERRUPT, Taken(ONE.id, NEXT, Stamp(2090), 9.1)])
    assert tellings == [*TOLD, Snapshot(ONE.id, ONE.cwd)]
    assert live(state).turn == Opened(NEXT) and isinstance(live(state).state, Running)
    _, tellings = told([Stopped(ONE.id, "three", mode=None, prompt=NEXT, again=False, heard=STOP_HEARD, request=STOP_REQUEST)], state)
    assert tellings == [Compare(ONE.id, again=False), Summarise(ONE.id, NEXT, "three")]


def test_a_message_queued_under_a_flushed_id_before_claude_answers_under_it_is_queued_into_the_turn() -> None:
    """The flush's records carry its new id at once, and Claude's first answer under it — the Continued — comes seconds
    later. A message queued in between carries the new id (2.1.281), and is queued, not the turn ended."""
    flushed = PromptId("q")
    state, effects = reduce(in_turn(), Taken(ONE.id, flushed, None, 9.0))
    assert effects == [] and live(state).turn == Opened(TURN, frozenset({flushed}))
    state, effects = reduce(state, Prompted(ONE.id, at=9.0, mode=None, prompt=flushed))
    assert effects == [] and live(state).turn == Opened(TURN, frozenset({flushed}), queued=True)


def test_a_turn_a_prompt_opens_has_gone_on_under_no_other_id_yet() -> None:
    state, _ = told([Taken(ONE.id, PromptId("q"), None, 9.0), said(status.Idle()), PROMPT])
    assert live(state).turn == Opened(NEXT)


def test_a_prompt_that_cannot_be_told_from_one_queued_into_the_open_turn_is_queued_into_it() -> None:
    """The open turn's own id is a queued prompt's."""
    assert reduce(in_turn(), Prompted(ONE.id, at=9.0, mode=None, prompt=TURN))[1] == []


@pytest.mark.parametrize("event", [
    PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), on=BASH, mode=None),
    ToolFinished(ONE.id, at=5.0, call=BASH, mode=None),
])
def test_only_a_prompt_or_a_record_moves_the_turn(event: SessionEvent) -> None:
    """A background subagent's hooks carry the prompt_id of the turn that started it, long after that turn is over."""
    assert live(reduce(in_turn(), event)[0]).turn == Opened(TURN)


@pytest.mark.parametrize("before", [in_turn(), in_turn(AT_DIALOG, HELD), in_turn(AT_DIALOG, LetGo(BASH))])
def test_the_record_of_an_interrupt_of_the_turn_claude_is_answering_tells_it_whether_or_not_its_idle_was_read(before: Registry) -> None:
    """Read before the idle it follows, or with that idle never read; the idle read after it moves only the state."""
    state, tellings = told([Interrupted(ONE.id, TURN, at=10.0)], before)
    assert (live(state).turn, tellings) == (Told(TURN), TOLD)
    state, tellings = told([said(status.Idle()), Read(ONE.id, WINDOW)], state)
    assert (live(state).turn, tellings) == (Told(TURN), [])


def test_the_record_of_an_interrupt_under_an_id_the_turn_went_on_under_is_what_claude_answers_next() -> None:
    """A flush names the queued message it sends on, not the turn Claude was answering: the turn goes on."""
    state, _ = reduce(in_turn(), Taken(ONE.id, NEXT, None, 9.0))
    assert reduce(state, Interrupted(ONE.id, NEXT, at=9.1)) == (state, [])


def test_an_interrupt_read_after_the_next_prompt_leaves_the_next_turn_open() -> None:
    """The tail reads the record a moment after it is written, by which time the user may have typed again."""
    running = in_turn(turn=NEXT)
    assert reduce(running, Interrupted(ONE.id, TURN, at=10.0)) == (running, [])


@pytest.mark.parametrize("before", [holding(IDLE, Told(TURN)), GONE])
def test_an_interrupt_of_a_turn_already_over_changes_nothing(before: Registry) -> None:
    assert reduce(before, Interrupted(ONE.id, TURN, at=10.0)) == (before, [])


def test_a_prompt_opened_after_a_turn_was_told_is_not_ended_by_the_interrupt_of_that_one() -> None:
    after, _ = reduce(holding(IDLE, Told(TURN)), Prompted(ONE.id, at=5.0, mode=None, prompt=NEXT))
    assert reduce(after, Interrupted(ONE.id, TURN, at=10.0)) == (after, [])


def test_an_interrupt_the_registry_never_heard_open_changes_nothing() -> None:
    before = holding(BUSY)
    assert reduce(before, Interrupted(ONE.id, TURN, at=10.0)) == (before, [])


def test_a_session_left_at_its_prompt_says_nothing_however_long_it_waits() -> None:
    """A finished turn is told through its summary or not at all: the clock adds nothing of its own."""
    state, _ = reduce(in_turn(), said(status.Idle(), at=10.0))
    assert reduce(state, Tick(10.0 + 3600.0))[1] == []


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
    assert state == holding(BUSY, Opened(NEXT, frozenset({TURN})))
    assert live(reduce(state, said(status.Idle(), at=9.0))[0]).turn == Untold(NEXT, frozenset({TURN}), WINDOW)


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
# Where a turn said idle at said()'s stamp waits to be told until the transcript is read through.
WINDOW = Stamp(2000 + UNTOLD)
STOP = Stopped(ONE.id, "done", mode=None, prompt=TURN, again=False, heard=STOP_HEARD, request=STOP_REQUEST)
INTERRUPT = Interrupted(ONE.id, TURN, at=10.2)
PROMPT = Prompted(ONE.id, at=10.5, mode=None, prompt=NEXT)


@pytest.mark.parametrize("at", [status.Idle(), Shell()])
@pytest.mark.parametrize("before", [in_turn(), in_turn(AT_DIALOG, HELD), in_turn(AT_DIALOG, LetGo(BASH))])
def test_an_open_turn_claude_code_says_is_idle_is_over_at_once(before: Registry, at: status.AtPrompt) -> None:
    """However it was stopped: no Stop, no record, and no idle_prompt follow a double Escape before the flushed message
    is answered. The telling waits for the transcript to say how it ended. A shell command it left running in the
    background keeps no turn open."""
    after, effects = reduce(before, said(at))
    assert after == holding(Idle(at, Stamp(2000), after=TURN), Untold(TURN, frozenset(), WINDOW))
    assert effects == ([Reply(ONE.id, RequestId("r0"), Withdraw())] if live(before).dialog == HELD else [])


@pytest.mark.parametrize("state", [replace(IDLE, after=TURN), replace(IDLE, after=TURN)])
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
    assert state == holding(Idle(status.Idle(), Stamp(2000), after=TURN), Told(TURN))


def test_an_interrupt_record_of_another_turn_leaves_the_untold_one_waiting_for_its_own() -> None:
    """A record read again from the start of a transcript is not the one the turn waits for."""
    state, tellings = told([said(status.Idle()), Interrupted(ONE.id, NEXT, at=10.1)])
    assert tellings == []
    state, effects = reduce(state, INTERRUPT)
    assert [effect for effect in effects if isinstance(effect, Compare | Summarise)] == TOLD


def test_a_stop_that_fires_after_claude_code_said_idle_tells_the_turn_with_its_closing_reply() -> None:
    state, tellings = told([said(status.Idle()), Stopped(ONE.id, "done", mode="plan", prompt=TURN, again=False, heard=STOP_HEARD, request=STOP_REQUEST), Tick(20.0)])
    assert tellings == [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, "done")]
    assert (live(state).state, live(state).mode) == (Idle(status.Idle(), Stamp(2000), after=TURN), "plan")


@pytest.mark.parametrize("ending", [STOP, INTERRUPT])
def test_a_turn_its_stop_or_interrupt_ended_before_claude_code_said_idle_is_told_once(ending: Event) -> None:
    _, tellings = told([ending, said(status.Idle()), Tick(20.0)])
    assert [telling for telling in tellings if isinstance(telling, Summarise)] == [Summarise(ONE.id, TURN, ending.closing if isinstance(ending, Stopped) else None)]


def test_a_turn_nothing_says_how_it_ended_is_told_once_its_transcript_is_read_past_its_window() -> None:
    state, tellings = told([said(status.Idle(), at=10.0), Read(ONE.id, Stamp(WINDOW - 1)), Tick(30.0)])
    assert tellings == []
    state, tellings = told([Read(ONE.id, WINDOW), INTERRUPT, STOP, Read(ONE.id, Stamp(WINDOW + 5000))], state)
    assert tellings == TOLD
    assert live(state).turn == Told(TURN)


def test_an_interrupt_record_read_long_after_the_idle_still_reaches_the_telling_of_its_turn() -> None:
    """hands-status-tlo.tmp: however late hands reads it, the record reaches the telling, since the telling waits on how
    far the transcript was read, on Claude Code's clock, and never on hands' own."""
    state, tellings = told([said(status.Idle(), at=10.0), Tick(10.0 + 30.0)])
    assert tellings == []
    _, tellings = told([INTERRUPT, Read(ONE.id, Stamp(WINDOW + 30_000))], state)
    assert tellings == TOLD


def test_a_turn_left_untold_is_told_before_the_next_prompt_marks_its_own() -> None:
    state, tellings = told([said(status.Idle()), PROMPT, INTERRUPT, Tick(20.0)])
    assert tellings == [*TOLD, Snapshot(ONE.id, ONE.cwd)]
    assert live(state).turn == Opened(NEXT)


def test_the_record_of_the_turn_left_untold_waits_with_it_for_how_it_ended() -> None:
    """Read after Claude Code said idle, as a prompt taken and stopped inside one tail period is: it names the turn, and
    tells nothing before the record of how it ended."""
    _, tellings = told([said(status.Idle()), Taken(ONE.id, TURN, None, 9.0), Tick(10.5), INTERRUPT])
    assert tellings == TOLD


def test_a_turn_left_untold_is_still_told_after_compaction() -> None:
    _, tellings = told([said(status.Idle()), Joined(ONE, "compact"), Read(ONE.id, WINDOW)])
    assert tellings == TOLD


def test_a_double_escape_before_claude_answers_a_flushed_message_leaves_the_session_idle_and_told() -> None:
    """hands-keyboard-gxr.07g: the first Escape flushes the queued message, whose id is taken seconds before Claude
    answers under it; the second stops the turn with no record and no hook. Only the status says so."""
    heard: list[Effect] = []
    state = in_turn()
    for event in [Taken(ONE.id, NEXT, None, 9.0), said(status.Idle(), at=10.0), Read(ONE.id, WINDOW), Tick(70.0), Continued(ONE.id, was=TURN, now=NEXT), Interrupted(ONE.id, NEXT, at=71.0)]:
        state, effects = reduce(state, event)
        heard += [effect for effect in effects if isinstance(effect, Summarise | Speak)]
    assert heard == [Summarise(ONE.id, TURN, None)]
    assert live(state).state == Idle(status.Idle(), Stamp(2000), after=TURN)


def test_a_single_escape_that_flushes_a_queued_message_leaves_the_turn_running_on_to_its_stop() -> None:
    """Measured on 2.1.282: the flush sets busy again, not idle, and Claude answers under the flushed id."""
    state, tellings = told([Taken(ONE.id, NEXT, None, 9.0), said(Busy()), Continued(ONE.id, was=TURN, now=NEXT), Stopped(ONE.id, "done", mode=None, prompt=NEXT, again=False, heard=STOP_HEARD, request=STOP_REQUEST), said(status.Idle(), 3000, at=20.0), Tick(30.0)])
    assert tellings == [Compare(ONE.id, again=False), Summarise(ONE.id, NEXT, "done")]
    assert isinstance(live(state).state, Idle)


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
    assert state == holding(RUNNING, Opened(NEXT), earlier=frozenset({TURN}))
    assert tellings == [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, "done"), Snapshot(ONE.id, ONE.cwd)]
    state, tellings = told([Stopped(ONE.id, "two", mode=None, prompt=NEXT, again=False, heard=STOP_HEARD, request=STOP_REQUEST), said(status.Idle(), at=14.0), Tick(20.0)], state)
    assert tellings == [Compare(ONE.id, again=False), Summarise(ONE.id, NEXT, "two")]
    assert state == holding(Idle(status.Idle(), Stamp(2000), after=NEXT), Told(NEXT), earlier=frozenset({TURN}))


@pytest.mark.parametrize(("written", "opens"), [(Stamp(2003), True), (Stamp(2000), True), (Stamp(1990), False), (None, False)])
def test_a_prompt_taken_at_the_prompt_opens_a_turn_only_if_written_since_claude_code_last_said_idle(written: Stamp | None, opens: bool) -> None:
    """Claude Code sets idle for ~3 ms between a turn and the one queued behind it (2.1.282), and a read can land on it.
    An idle set after the record was written is that turn over, before any of it was read."""
    state, _ = told([STOP, said(status.Idle()), taken(NEXT, written)], in_turn(RUNNING))
    assert isinstance(live(state).turn, Opened) == opens


@pytest.mark.parametrize(("written", "opens"), [(Stamp(1500), True), (Stamp(1000), True), (Stamp(990), False), (None, False)])
def test_a_prompt_taken_while_running_with_no_turn_open_opens_one_only_if_written_since_the_idle_before_the_run(written: Stamp | None, opens: bool) -> None:
    """A record written before the idle the run began from is of a turn over by then."""
    state, _ = told([taken(NEXT, written)], holding(RUNNING, Told(TURN)))
    assert isinstance(live(state).turn, Opened) == opens


def test_a_queued_turn_whose_stop_lands_before_its_record_is_read_is_told_once() -> None:
    state, tellings = told([QUEUED, STOP, Stopped(ONE.id, "two", mode=None, prompt=NEXT, again=False, heard=STOP_HEARD, request=STOP_REQUEST), taken(NEXT, Stamp(1500)), said(status.Idle(), at=14.0), Tick(20.0)], in_turn(RUNNING))
    assert tellings == [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, "done"), Snapshot(ONE.id, ONE.cwd), Compare(ONE.id, again=False), Summarise(ONE.id, NEXT, "two")]
    assert live(state).turn == Told(NEXT)


def test_a_bang_command_is_running_from_its_status_and_claudes_answer_to_it_is_told_as_itself() -> None:
    """Seen live on 2.1.282 and 2.1.283: Claude Code is busy while the command runs, writes its record under a new id
    stamped before that busy, Claude answers it, and a Stop names it."""
    bang = PromptId("bang")
    at_prompt = holding(Idle(status.Idle(), Stamp(2000), after=TURN), Told(TURN))
    state, tellings = told([said(Busy(), 3000, at=20.0)], at_prompt)
    assert (live(state).state, tellings) == (Running(Busy(), Stamp(3000), idled=Stamp(2000)), [])
    state, tellings = told([taken(bang, Stamp(2900), at=24.0)], state)
    assert (live(state).turn, tellings) == (Opened(bang), [])
    state, tellings = told([Stopped(ONE.id, "It slept.", mode=None, prompt=bang, again=False, heard=STOP_HEARD, request=STOP_REQUEST)], state)
    assert tellings == [Compare(ONE.id, again=False), Summarise(ONE.id, bang, "It slept.")]


def test_a_command_claude_code_runs_at_the_prompt_is_running_while_it_runs_and_told_once_it_is_over() -> None:
    """Seen live on 2.1.282: /compact is busy for its whole run and writes records under a new id, with no Stop. What is
    told of it is what happened in it, which for a command is nothing, and the narrator says nothing of that."""
    at_prompt = holding(Idle(status.Idle(), Stamp(2000), after=TURN), Told(TURN))
    compact = PromptId("compact")
    state, _ = told([said(Busy(), 3000, at=20.0), taken(compact, Stamp(3001), at=20.1), Joined(ONE, "compact")], at_prompt)
    assert state == holding(Running(Busy(), Stamp(3000), idled=Stamp(2000)), Opened(compact), earlier=frozenset({TURN}))
    state, tellings = told([said(status.Idle(), 9000, at=30.0), Read(ONE.id, Stamp(9000 + UNTOLD))], state)
    assert (live(state).state, tellings) == (Idle(status.Idle(), Stamp(9000), after=compact), [Compare(ONE.id, again=False), Summarise(ONE.id, compact, None)])


def test_a_stop_with_nothing_queued_behind_it_marks_nothing() -> None:
    """A message 2.1.281 took into the running turn is in that turn, and waits behind nothing: a mark taken at its Stop
    would be read, however much later, as the start of the next turn no prompt marks."""
    state, tellings = told([QUEUED, taken(NEXT, Stamp(1200)), Stopped(ONE.id, "done", mode=None, prompt=NEXT, again=False, heard=STOP_HEARD, request=STOP_REQUEST)], in_turn(RUNNING))
    assert tellings == [Compare(ONE.id, again=False), Summarise(ONE.id, NEXT, "done")]
    assert live(state).turn == Told(TURN, frozenset({NEXT}))


def test_a_prompt_taken_before_any_status_is_read_opens_nothing() -> None:
    """No status says whether a turn of the session runs, so no record opens one."""
    attached = holding(Unreported())
    assert reduce(attached, taken(NEXT, Stamp(1500))) == (attached, [])


def test_followed_mid_turn_it_works_under_its_own_prompt_once_its_record_is_read_and_its_stop_ends_it() -> None:
    """hands-status-tlo.egc: a daemon restarted while the session runs. Its first status read says running, set after the
    turn's record was written (busy again after a dialog), and the tail hands on the record of the turn its transcript
    is in, and nothing from before it. The queued turn's record is written after the running one's Stop."""
    state, _ = told([said(Busy(), 3000, at=1.0), taken(TURN, Stamp(2000), at=1.1)], holding(Unreported()))
    assert state == holding(Running(Busy(), Stamp(3000), idled=None), Opened(TURN))
    state, tellings = told([Stopped(ONE.id, "done", mode=None, prompt=TURN, again=False, heard=STOP_HEARD, request=STOP_REQUEST)], state)
    assert (live(state).turn, tellings) == (Told(TURN), [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, "done")])
    state, _ = told([taken(NEXT, Stamp(9000), at=20.0)], state)
    assert state == holding(Running(Busy(), Stamp(3000), idled=None), Opened(NEXT), earlier=frozenset({TURN}))


@pytest.mark.parametrize("between", [[], [said(Busy(), 2100, at=10.5)], [said(Busy(), 2100, at=10.5), said(status.Idle(), 2600, at=11.0)]])
def test_a_stop_under_an_id_no_record_has_named_while_a_telling_waits_is_held_until_the_transcript_is_read_through_it(between: list[Event]) -> None:
    """Whether it is the waiting turn's, gone on under a queued message, or a turn after it, only the transcript says:
    however the status was sampled between, it tells nothing while held, and read through without a record naming it,
    it is a line, and the waiting turn is told by its own end."""
    before, _ = told([said(status.Idle()), *between])
    other = Stopped(ONE.id, "other", mode=None, prompt=NEXT, again=False, heard=Stamp(2010), request=STOP_REQUEST)
    held, effects = reduce(before, other)
    # Its hook is held with it: Claude Code goes on only once it is decided.
    assert (live(held).turn, effects) == (live(before).turn, [Audit(Holding(ONE.id, NEXT))])
    after, effects = reduce(held, Read(ONE.id, Stamp(2010 + UNTOLD)))
    assert [e for e in effects if isinstance(e, Audit | Compare | Summarise | Reply)] == [*TOLD, Audit(Unmatched(ONE.id, NEXT)), LET_STOP]
    assert live(after).unnamed == ()


def test_a_late_stop_of_a_turn_told_before_the_last_one_ends_nothing_and_is_audited() -> None:
    """p1's Stop delayed past p2's prompt, which tells p1, and past p2's own Stop: told again, p1 would be heard twice."""
    before, tellings = told([PROMPT, Stopped(ONE.id, "two", mode=None, prompt=NEXT, again=False, heard=STOP_HEARD, request=STOP_REQUEST)])
    assert tellings == [*TOLD, Snapshot(ONE.id, ONE.cwd), Compare(ONE.id, again=False), Summarise(ONE.id, NEXT, "two")]
    assert reduce(before, STOP) == (before, [Audit(Unmatched(ONE.id, STOP.prompt)), LET_STOP])


@pytest.mark.parametrize(("events", "closing"), [
    ([Continued(ONE.id, was=TURN, now=NEXT), Stopped(ONE.id, "two", mode=None, prompt=NEXT, again=False, heard=STOP_HEARD, request=STOP_REQUEST)], "two"),
    ([Stopped(ONE.id, "two", mode=None, prompt=NEXT, again=False, heard=STOP_HEARD, request=STOP_REQUEST), Continued(ONE.id, was=TURN, now=NEXT), Read(ONE.id, WINDOW)], "two"),
])
def test_a_turn_that_went_on_under_a_queued_message_is_told_once_under_that_message_s_id(events: list[Event], closing: str | None) -> None:
    """The Continued record can be read after the idle that ended the turn: the turn waiting to be told takes the new id.
    Its Stop heard before that record is held until the record names it, and tells the turn once, with its reply."""
    _, tellings = told([said(status.Idle()), *events])
    assert tellings == [Compare(ONE.id, again=False), Summarise(ONE.id, NEXT, closing)]


def test_a_stop_heard_before_the_record_of_the_turn_after_the_waiting_one_tells_that_turn_with_its_reply() -> None:
    """A message queued behind p1 is taken once p1 stops, and its turn can stop before its record is read: the waiting
    turn is told as itself, and the Stop tells the turn the record opened, never the one waiting."""
    _, tellings = told([said(status.Idle()), Stopped(ONE.id, "two", mode=None, prompt=NEXT, again=False, heard=Stamp(2500), request=STOP_REQUEST), taken(NEXT, Stamp(2100)), Read(ONE.id, Stamp(9000))])
    assert tellings == [*TOLD, Compare(ONE.id, again=False), Summarise(ONE.id, NEXT, "two")]


def test_a_stop_under_an_unnamed_id_in_a_session_with_no_status_read_is_not_held() -> None:
    """Its transcript is read only once a status is, so no record would come to name it: held, Claude Code would wait
    on its hook for nothing."""
    before = holding(Unreported(), Opened(TURN))
    stop = Stopped(ONE.id, "two", mode=None, prompt=NEXT, again=False, heard=STOP_HEARD, request=STOP_REQUEST)
    assert reduce(before, stop) == (before, [Audit(Unmatched(ONE.id, NEXT)), LET_STOP])


def test_a_held_stop_is_let_go_only_once_the_turn_it_tells_is_compared_and_the_one_queued_behind_it_marked() -> None:
    """Claude Code runs a queued message once the Stop hook returns: held until the record names it, the hook still
    returns after the mark, so the queued turn has changed nothing when it is taken."""
    queued, _ = reduce(in_turn(), Prompted(ONE.id, at=5.0, mode=None, prompt=TURN))
    held, _ = reduce(queued, Stopped(ONE.id, "two", mode=None, prompt=NEXT, again=False, heard=STOP_HEARD, request=STOP_REQUEST))
    _, effects = reduce(held, Continued(ONE.id, was=TURN, now=NEXT))
    assert effects == [Compare(ONE.id, again=False), Summarise(ONE.id, NEXT, "two"), Snapshot(ONE.id, ONE.cwd), LET_STOP]


def test_a_held_stop_and_the_stop_again_after_it_each_tell_their_part_once_the_record_names_them() -> None:
    """Another Stop hook blocked the first, so Claude went on under the same id: heard together before the record,
    they are told as they would have been heard after it."""
    first = Stopped(ONE.id, "First.", mode=None, prompt=NEXT, again=False, heard=STOP_HEARD, request=RequestId("first"))
    again = Stopped(ONE.id, "Second.", mode=None, prompt=NEXT, again=True, heard=STOP_HEARD, request=RequestId("again"))
    state, effects = told([first, again])
    assert effects == []
    after, effects = reduce(state, Continued(ONE.id, was=TURN, now=NEXT))
    assert effects == [
        Compare(ONE.id, again=False), Summarise(ONE.id, NEXT, "First."), Reply(ONE.id, first.request, Withdraw()),
        Compare(ONE.id, again=True), Summarise(ONE.id, NEXT, "Second."), Reply(ONE.id, again.request, Withdraw()),
    ]
    assert live(after).unnamed == ()


def test_a_held_stop_tells_the_waiting_turn_once_a_record_names_it() -> None:
    state, _ = told([said(status.Idle()), Stopped(ONE.id, "Fixed it. Want me to look at the others?", mode=None, prompt=NEXT, again=False, heard=Stamp(2010), request=STOP_REQUEST)])
    after, _ = reduce(state, Continued(ONE.id, was=TURN, now=NEXT))
    assert live(after).turn == Told(NEXT, frozenset({TURN}))


@pytest.mark.parametrize("end", [Ended(ONE.id, "other"), Died(ONE), Joined(ONE, "resume"), Joined(ONE, "startup")])
def test_a_stop_still_held_when_its_session_ends_or_starts_again_is_a_line_and_its_hook_let_go(end: Event) -> None:
    """The transcript is read afresh as history from there, so no record read after names it."""
    state, _ = told([said(status.Idle()), Stopped(ONE.id, "two", mode=None, prompt=NEXT, again=False, heard=Stamp(2500), request=STOP_REQUEST)])
    after, effects = reduce(state, end)
    assert [e for e in effects if isinstance(e, Audit | Reply)] == [Audit(Unsettled(ONE.id, NEXT)), LET_STOP]
    assert isinstance(after.sessions[ONE.id], Gone) or live(after).unnamed == ()


def test_a_stop_held_across_a_compaction_is_still_held() -> None:
    """The same process goes on writing the same transcript, so the record that names it can still be read."""
    state, _ = told([said(status.Idle()), Stopped(ONE.id, "two", mode=None, prompt=NEXT, again=False, heard=Stamp(2500), request=STOP_REQUEST)])
    after, effects = reduce(state, Joined(ONE, "compact"))
    assert (live(after).unnamed, [e for e in effects if isinstance(e, Audit | Reply)]) == (live(state).unnamed, [])


def test_a_late_stop_again_of_a_turn_told_before_the_one_waiting_ends_nothing() -> None:
    """Stopping again reopens only the last turn it names: an older turn's, heard late, would tell it twice."""
    state, _ = told([PROMPT, said(Busy(), 2100, at=11.0), said(status.Idle(), 3000, at=12.0)])
    again = Stopped(ONE.id, "done", mode=None, prompt=TURN, again=True, heard=STOP_HEARD, request=STOP_REQUEST)
    assert reduce(state, again) == (state, [Audit(Unmatched(ONE.id, again.prompt)), LET_STOP])


@pytest.mark.parametrize("source", ["resume", "startup"])
def test_a_turn_told_before_a_restart_is_not_told_again_by_its_late_stop(source: StartSource) -> None:
    state, _ = told([STOP, Joined(ONE, source)])
    assert reduce(state, STOP) == (state, [Audit(Unmatched(ONE.id, STOP.prompt)), LET_STOP])


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


@pytest.mark.parametrize("between", [[], [Joined(ONE, "compact")]])
@pytest.mark.parametrize(("own", "closing"), [(Stopped(ONE.id, "two", mode=None, prompt=NEXT, again=False, heard=STOP_HEARD, request=STOP_REQUEST), "two"), (Interrupted(ONE.id, NEXT, at=12.2), None)])
def test_a_late_stop_of_an_older_turn_leaves_the_turn_waiting_to_be_told_for_its_own_end(between: list[Event], own: Event, closing: str | None) -> None:
    """hands-status-tlo.ypl: p1's Stop is delayed past p2's prompt, which tells p1, and past the idle that ends p2. It is
    not p2's, so p2 waits on for its own Stop or record, compaction or not."""
    state, tellings = told([PROMPT, said(Busy(), 2000, at=11.0), said(status.Idle(), 3000, at=12.0), *between, STOP])
    assert tellings == [*TOLD, Snapshot(ONE.id, ONE.cwd)]
    assert isinstance(live(state).turn, Untold)
    assert told([own, Tick(20.0)], state)[1] == [Compare(ONE.id, again=False), Summarise(ONE.id, NEXT, closing)]


@pytest.mark.parametrize("event", [INTERRUPT, taken(NEXT, Stamp(9000)), Continued(ONE.id, was=TURN, now=NEXT)])
def test_a_late_record_of_a_session_gone_is_not_audited_as_after_its_end(event: Event) -> None:
    state, _ = told([said(status.Idle()), Ended(ONE.id, "prompt_input_exit")])
    assert reduce(state, event) == (state, [])


def test_a_session_first_read_at_its_prompt_is_idle_after_no_turn() -> None:
    """Attached after a restart, or just started: it went idle before hands followed it."""
    assert reduce(holding(Unreported()), WENT_IDLE) == (holding(Idle(status.Idle(), Stamp(2000), after=None)), [])


def test_a_turn_told_before_its_idle_is_read_starts_a_new_idle_period() -> None:
    """Its busy fell between two reads: the prompt and the Stop are heard, then an idle with a new stamp."""
    before = holding(replace(IDLE, after=TURN), Told(TURN))
    state, _ = told([Prompted(ONE.id, at=100.0, mode=None, prompt=NEXT), Stopped(ONE.id, "Done.", mode=None, prompt=NEXT, again=False, heard=STOP_HEARD, request=STOP_REQUEST), said(status.Idle(), 3000, at=100.1)], before)
    assert live(state).state == Idle(status.Idle(), Stamp(3000), after=NEXT)


def test_a_turn_heard_and_ended_inside_one_idle_read_starts_a_new_idle_period() -> None:
    """A prompt cancelled during its hooks: Claude Code's busy is never read, and the next idle is new."""
    state, _ = told([Prompted(ONE.id, at=100.0, mode=None, prompt=NEXT), said(status.Idle(), at=100.1)], holding(IDLE, Told(TURN)))
    assert live(state).state == Idle(status.Idle(), Stamp(2000), after=NEXT)


def test_a_turn_gone_on_under_a_queued_prompt_still_goes_by_the_id_it_went_on_from() -> None:
    """A hook sent before Claude Code moved on carries the old id: it is in the turn, and ends nothing."""
    state, _ = told([Taken(ONE.id, NEXT, None, 9.0), Continued(ONE.id, was=TURN, now=NEXT), Prompted(ONE.id, at=9.5, mode=None, prompt=TURN)])
    assert live(state).turn == Opened(NEXT, frozenset({NEXT, TURN}), queued=True)


# ── A turn told from the wire ───────────────────────────────────────────────────────────────────────────────────────


def stop(closing: str, again: bool = False) -> Stopped:
    return Stopped(ONE.id, closing, mode=None, prompt=TURN, again=again, heard=STOP_HEARD, request=STOP_REQUEST)


def through(*events: Event) -> tuple[Registry, list[Effect]]:
    """The registry after the events, from a turn open, and everything they called for."""
    state, heard = in_turn(), list[Effect]()
    for event in events:
        state, effects = reduce(state, event)
        heard += effects
    return state, heard


def test_the_reply_that_closes_a_turn_on_the_wire_tells_it_and_its_stop_then_only_lets_its_hook_go() -> None:
    state, heard = through(Closed(ONE.id, TURN, "Done."), stop("Done."))
    assert heard == [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, "Done."), Audit(Unmatched(ONE.id, TURN)), LET_STOP]
    assert live(state).turn == Told(TURN)


def test_a_turn_whose_stop_came_first_is_not_told_again_by_its_reply_on_the_wire() -> None:
    state, heard = through(stop("Done."), Closed(ONE.id, TURN, "Done."))
    assert heard == [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, "Done."), LET_STOP, Audit(Unclosed(ONE.id, TURN))]
    assert live(state).turn == Told(TURN)


@pytest.mark.parametrize("wire_first", [True, False])
def test_a_turn_that_goes_on_after_a_blocked_stop_has_each_of_its_endings_told_once(wire_first: bool) -> None:
    """Seen live on 2.1.285 (/tmp/stopprobe): a blocked Stop's turn goes on under the same prompt id, and its Stop fires again."""

    def ending(closing: str, again: bool) -> list[Event]:
        heard: list[Event] = [Closed(ONE.id, TURN, closing), stop(closing, again)]
        return heard if wire_first else heard[::-1]

    _, heard = through(*ending("done", again=False), *ending("BANANA", again=True))
    assert [effect for effect in heard if isinstance(effect, Summarise)] == [Summarise(ONE.id, TURN, "done"), Summarise(ONE.id, TURN, "BANANA")]
    assert [effect for effect in heard if isinstance(effect, Compare)] == [Compare(ONE.id, again=False), Compare(ONE.id, again=True)]


def test_an_open_turn_stopping_again_is_read_against_its_own_prompt_s_mark() -> None:
    """Its first Stop never reached hands, so no reading of it left off anywhere: where the last one did is another turn's."""
    _, heard = through(stop("Second.", again=True))
    assert heard == [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, "Second."), LET_STOP]


def test_a_reply_on_the_wire_of_a_turn_told_already_ends_nothing() -> None:
    """Heard late, after its turn was told with what the transcript held; or a second end_turn under the same id, which
    only a Stop's again says is the turn gone on."""
    state = holding(IDLE, Untold(TURN, frozenset(), by=Stamp(2000)))
    told, _ = reduce(state, Read(ONE.id, Stamp(3000)))
    assert reduce(told, Closed(ONE.id, TURN, "Done.")) == (told, [Audit(Unclosed(ONE.id, TURN))])


def test_a_reply_on_the_wire_tells_a_turn_claude_code_already_said_is_over() -> None:
    state = holding(IDLE, Untold(TURN, frozenset(), by=Stamp(2000)))
    after, effects = reduce(state, Closed(ONE.id, TURN, "Done."))
    assert effects == [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, "Done.")]
    assert live(after).turn == Told(TURN)


def test_a_reply_on_the_wire_marks_the_turn_queued_behind_the_one_it_tells() -> None:
    state = holding(BUSY, Opened(TURN, queued=True))
    assert reduce(state, Closed(ONE.id, TURN, "Done."))[1] == [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, "Done."), Snapshot(ONE.id, ONE.cwd)]


def test_a_reply_on_the_wire_that_asks_something_leaves_the_turn_asking() -> None:
    after, _ = reduce(in_turn(), Closed(ONE.id, TURN, "Should I push it?"))
    assert live(after).turn == Told(TURN)


def test_a_reply_on_the_wire_for_a_turn_the_session_is_not_in_is_a_line() -> None:
    assert reduce(in_turn(), Closed(ONE.id, NEXT, "Done.")) == (in_turn(), [Audit(Unclosed(ONE.id, NEXT))])


def test_a_message_queued_between_the_wire_telling_a_turn_and_its_stop_is_marked_and_the_turn_told_once() -> None:
    """Claude Code files it under the told turn's id, as it does one queued behind a running turn, and runs it once
    that turn's Stop hook returns: marked while its own prompt hook holds Claude Code."""
    state, heard = through(Closed(ONE.id, TURN, "Done."), Prompted(ONE.id, at=6.0, mode=None, prompt=TURN), stop("Done."))
    assert [effect for effect in heard if isinstance(effect, Compare | Summarise | Snapshot)] == [Compare(ONE.id, again=False), Summarise(ONE.id, TURN, "Done."), Snapshot(ONE.id, ONE.cwd)]
    assert live(state).turn == Told(TURN)
