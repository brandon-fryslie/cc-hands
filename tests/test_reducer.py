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
    PermissionAsked,
    PermissionDeadlineNear,
    PermissionExpired,
    Reply,
    SessionGone,
    Speak,
    Summarise,
    Unregistered,
    Withdraw,
)
from hands.core.events import Abandoned, Attached, Died, Ended, EndReason, MovedOn, Event, Joined, PermissionRequested, Prompted, SessionEvent, StartSource, Stopped, Tick, ToolFinished
from hands.core.reducer import EXPIRED_MESSAGE, WARNING_LEAD_SECONDS, reduce
from hands.core.session import (
    Blocked,
    Gone,
    Idle,
    Membership,
    Permission,
    Registry,
    RequestId,
    Session,
    SessionId,
    SessionState,
    Working,
)

TIMEOUT = 60.0
ONE = Membership(SessionId("s1"), pid=4242, cwd=Path("/code/a"), transcript=Path("/t/s1.jsonl"))
TWO = Membership(SessionId("s2"), pid=5353, cwd=Path("/code/b"), transcript=Path("/t/s2.jsonl"))
BASH = Permission(tool="Bash", input={"command": "ls"})

LIVE: list[SessionState] = [
    Idle(),
    Working(since=1.0),
    Blocked(on=BASH, request=RequestId("r0"), deadline=61.0, warned=False),
]
SESSION_EVENTS: list[SessionEvent] = [
    Prompted(ONE.id, at=5.0),
    Stopped(ONE.id),
    PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), permission=BASH),
    ToolFinished(ONE.id, at=5.0, call=BASH),
    Ended(ONE.id, "prompt_input_exit"),
]


def registry(*sessions: Session) -> Registry:
    return Registry(permission_deadline=TIMEOUT, sessions={s.membership.id: s for s in sessions}, drafts={})


def holding(state: SessionState) -> Registry:
    return registry(Session(ONE, state))


def test_a_start_registers_the_session_idle() -> None:
    assert reduce(registry(), Joined(ONE, "startup")) == (holding(Idle()), [])


@pytest.mark.parametrize("before", LIVE)
@pytest.mark.parametrize(
    ("event", "after"),
    [
        (Prompted(ONE.id, at=5.0), Working(since=5.0)),
        (Stopped(ONE.id), Idle()),
        (
            PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), permission=BASH),
            Blocked(on=BASH, request=RequestId("r1"), deadline=5.0 + TIMEOUT, warned=False),
        ),
        (Ended(ONE.id, "prompt_input_exit"), Gone()),
    ],
)
def test_every_state_takes_each_event_to_its_state(before: SessionState, event: Event, after: SessionState) -> None:
    assert reduce(holding(before), event)[0] == holding(after)


WAITING = Blocked(on=BASH, request=RequestId("r0"), deadline=61.0, warned=False)


@pytest.mark.parametrize("before", [Idle(), Working(since=1.0)])
@pytest.mark.parametrize("event", [Prompted(ONE.id, at=5.0), Ended(ONE.id, "prompt_input_exit"), Joined(ONE, "startup")])
def test_moving_between_states_that_wait_on_nothing_asks_for_nothing(before: SessionState, event: Event) -> None:
    assert reduce(holding(before), event)[1] == []


@pytest.mark.parametrize("before", [Idle(), Working(since=1.0)])
def test_a_permission_request_is_handed_to_the_intermediary(before: SessionState) -> None:
    event = PermissionRequested(ONE.id, at=5.0, request=RequestId("r1"), permission=BASH)
    assert reduce(holding(before), event)[1] == [Narrate(PermissionAsked(ONE.id, RequestId("r1"), BASH))]


@pytest.mark.parametrize(
    "event", [Prompted(ONE.id, at=5.0), Ended(ONE.id, "prompt_input_exit"), *(Joined(ONE, source) for source in ("startup", "resume", "clear"))]
)
def test_a_session_that_moves_on_while_waiting_lets_its_hook_go_undecided(event: Event) -> None:
    # Most often the user answered the dialog at the keyboard; a voice reply after that would decide nothing.
    assert reduce(holding(WAITING), event)[1] == [Reply(ONE.id, RequestId("r0"), Withdraw())]


@pytest.mark.parametrize("before", [Idle(), Working(since=1.0)])
def test_a_finished_turn_is_summarised_from_the_session_transcript(before: SessionState) -> None:
    assert reduce(holding(before), Stopped(ONE.id)) == (holding(Idle()), [Summarise(ONE.id, ONE.transcript)])


def test_a_turn_that_finishes_while_waiting_lets_the_hook_go_and_is_summarised() -> None:
    assert reduce(holding(WAITING), Stopped(ONE.id))[1] == [Reply(ONE.id, RequestId("r0"), Withdraw()), Summarise(ONE.id, ONE.transcript)]


def test_a_second_request_while_waiting_lets_the_first_go_and_asks_the_second() -> None:
    edit = Permission(tool="Edit", input={"file_path": "a.py"})
    after, effects = reduce(holding(WAITING), PermissionRequested(ONE.id, at=7.0, request=RequestId("r1"), permission=edit))
    assert after == holding(Blocked(on=edit, request=RequestId("r1"), deadline=7.0 + TIMEOUT, warned=False))
    assert effects == [Reply(ONE.id, RequestId("r0"), Withdraw()), Narrate(PermissionAsked(ONE.id, RequestId("r1"), edit))]


def test_before_the_warning_window_a_tick_changes_nothing() -> None:
    assert reduce(holding(WAITING), Tick(at=61.0 - WARNING_LEAD_SECONDS - 0.5)) == (holding(WAITING), [])


def test_the_warning_is_spoken_once_as_the_deadline_nears() -> None:
    warned, effects = reduce(holding(WAITING), Tick(at=53.0))
    assert warned == holding(replace(WAITING, warned=True))
    assert effects == [Speak(PermissionDeadlineNear(ONE.id, BASH, remaining=8.0))]
    assert reduce(warned, Tick(at=54.0)) == (warned, [])


@pytest.mark.parametrize("warned", [True, False])
def test_at_the_deadline_the_request_is_denied_and_said_to_be(warned: bool) -> None:
    after, effects = reduce(holding(replace(WAITING, warned=warned)), Tick(at=61.0))
    assert after == holding(Working(since=61.0))
    assert effects == [Reply(ONE.id, RequestId("r0"), Deny(EXPIRED_MESSAGE)), Speak(PermissionExpired(ONE.id, BASH))]


def test_ticking_through_a_whole_wait_warns_exactly_once_then_denies_once() -> None:
    state = holding(Idle())
    state, asked = reduce(state, PermissionRequested(ONE.id, at=0.0, request=RequestId("r"), permission=BASH))
    heard: list[Effect] = [*asked]
    for second in range(1, int(TIMEOUT) + 5):
        state, effects = reduce(state, Tick(at=float(second)))
        heard += effects
    assert heard == [
        Narrate(PermissionAsked(ONE.id, RequestId("r"), BASH)),
        Speak(PermissionDeadlineNear(ONE.id, BASH, remaining=WARNING_LEAD_SECONDS)),
        Reply(ONE.id, RequestId("r"), Deny(EXPIRED_MESSAGE)),
        Speak(PermissionExpired(ONE.id, BASH)),
    ]


def test_a_tick_moves_only_sessions_waiting_on_a_deadline() -> None:
    before = registry(Session(ONE, WAITING), Session(TWO, Working(since=0.0)))
    after, _ = reduce(before, Tick(at=100.0))
    assert after == registry(Session(ONE, Working(since=100.0)), Session(TWO, Working(since=0.0)))


@pytest.mark.parametrize("before", LIVE)
def test_a_compacted_session_keeps_its_state_and_takes_the_new_membership(before: SessionState) -> None:
    moved = replace(ONE, pid=7777)
    assert reduce(holding(before), Joined(moved, "compact")) == (registry(Session(moved, before)), [])


@pytest.mark.parametrize("before", [*LIVE, Gone()])
@pytest.mark.parametrize("source", ["startup", "resume", "clear"])
def test_any_start_but_compaction_is_at_the_prompt(before: SessionState, source: StartSource) -> None:
    # a session resumed after a crash never sent the Stop or SessionEnd the registry is still waiting for
    assert reduce(holding(before), Joined(ONE, source))[0] == holding(Idle())


def test_an_ended_session_compacting_is_idle() -> None:
    assert reduce(holding(Gone()), Joined(ONE, "compact")) == (holding(Idle()), [])


@pytest.mark.parametrize("event", SESSION_EVENTS)
def test_an_ended_session_ignores_late_hooks_and_audits_them(event: SessionEvent) -> None:
    assert reduce(holding(Gone()), event) == (holding(Gone()), [Audit(AfterEnd(event)), *let_go(event)])


@pytest.mark.parametrize("event", SESSION_EVENTS)
def test_an_event_for_a_session_that_never_joined_changes_nothing_and_is_audited(event: SessionEvent) -> None:
    before = registry(Session(TWO, Idle()))
    assert reduce(before, event) == (before, [Audit(Unregistered(event)), *let_go(event)])


def let_go(event: SessionEvent) -> list[Effect]:
    # A permission hook the registry cannot block on is released at once instead of hanging.
    return [Reply(ONE.id, RequestId("r1"), Withdraw())] if isinstance(event, PermissionRequested) else []


def test_the_tool_a_session_waits_on_finishing_means_its_dialog_was_answered_at_the_keyboard() -> None:
    after, effects = reduce(holding(WAITING), ToolFinished(ONE.id, at=20.0, call=BASH))
    assert (after, effects) == (holding(Working(since=20.0)), [Reply(ONE.id, RequestId("r0"), Withdraw())])


@pytest.mark.parametrize("before", [WAITING, Idle(), Working(since=1.0)])
def test_any_other_tool_finishing_changes_nothing(before: SessionState) -> None:
    other = Permission(tool="Bash", input={"command": "ls -la"})
    expected = holding(before)
    assert reduce(holding(before), ToolFinished(ONE.id, at=20.0, call=other)) == (expected, [])


def test_a_hook_that_went_away_ends_the_wait_with_nothing_to_reply_to() -> None:
    assert reduce(holding(WAITING), Abandoned(ONE.id, RequestId("r0"), at=30.0)) == (holding(Working(since=30.0)), [])


@pytest.mark.parametrize("before", [replace(WAITING, request=RequestId("newer")), Idle(), Gone()])
def test_an_abandoned_request_the_session_no_longer_waits_on_changes_nothing(before: SessionState) -> None:
    assert reduce(holding(before), Abandoned(ONE.id, RequestId("r0"), at=30.0)) == (holding(before), [])


def test_an_event_moves_only_its_own_session() -> None:
    before = registry(Session(ONE, Idle()), Session(TWO, Idle()))
    after, _ = reduce(before, Prompted(TWO.id, at=2.0))
    assert after == registry(Session(ONE, Idle()), Session(TWO, Working(since=2.0)))


def test_live_is_every_session_that_has_not_ended() -> None:
    both = registry(Session(ONE, Blocked(on=BASH, request=RequestId("r"), deadline=9.0, warned=False)), Session(TWO, Gone()))
    assert both.live() == [Session(ONE, Blocked(on=BASH, request=RequestId("r"), deadline=9.0, warned=False))]


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
    assert reduce(holding(before), Died(ONE)) == (holding(Gone()), [*released, Speak(SessionGone(ONE.id))])


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
    assert reduce(holding(before), Ended(ONE.id, "other")) == (holding(Gone()), [*released, Speak(SessionGone(ONE.id))])


@pytest.mark.parametrize("reason", ["clear", "resume", "logout", "prompt_input_exit", "bypass_permissions_disabled"])
def test_a_session_ended_at_the_keyboard_is_not_spoken(reason: EndReason) -> None:
    assert reduce(holding(Idle()), Ended(ONE.id, reason)) == (holding(Gone()), [])


def test_a_closed_terminal_after_the_sweep_found_the_session_dead_says_nothing_more() -> None:
    event = Ended(ONE.id, "other")
    assert reduce(holding(Gone()), event) == (holding(Gone()), [Audit(AfterEnd(event))])
