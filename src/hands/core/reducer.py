"""The session lifecycle as one pure function."""

from collections.abc import Callable
from dataclasses import replace

from hands.core.effects import (
    AfterEnd,
    Audit,
    Deny,
    Effect,
    HookReply,
    ModeChanged,
    Narrate,
    Note,
    Asking,
    DeadlineNear,
    Expired,
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
from hands.core.events import (
    Abandoned,
    Attached,
    Died,
    Ended,
    MovedOn,
    Event,
    Interrupted,
    Joined,
    PermissionRequested,
    Prompted,
    SessionEvent,
    StartSource,
    Stopped,
    Tick,
    ToolFinished,
    Waited,
)
from hands.core.session import AtDialog, Blocked, Blocker, Gone, Idle, Instant, Membership, Mode, Permission, Plan, PlanApproved, PromptId, Question, FinishedCall, Registry, RequestId, Session, SessionId, SessionState, UnknownMode, Working

# How long before a permission's deadline the one warning is spoken.
WARNING_LEAD_SECONDS = 10.0

# How long a session sits at its prompt before it is said to be waiting: Claude Code's own idle_prompt came 61 s after
# a Stop (2.1.281), so a nudge hands times itself comes when that one would have.
IDLE_NUDGE_SECONDS = 60.0

# What the agent reads when nobody answered in time.
EXPIRED_MESSAGE = (
    "Nobody answered this permission request by voice before its deadline, so hands denied it. "
    "Do not retry it until the user asks."
)


def reduce(registry: Registry, event: Event) -> tuple[Registry, list[Effect]]:
    """One event in; the next registry and the effects it calls for out. No I/O."""
    # [LAW:effects-at-boundaries] time arrives inside the event and the deadline
    # inside the registry, so a deadline is arithmetic on values, never a clock read.
    match event:
        case Joined(membership=membership, source=source):
            return _join(registry, _started(membership, source, registry.sessions.get(membership.id)))
        case Attached(membership=membership) if membership.id not in registry.sessions:
            # A membership file says nothing of the mode; the session's next hook will.
            return registry.put(Session(membership, Idle(), mode=None, turn=None)), []
        case Attached():
            # [LAW:no-ambient-temporal-coupling] a session the registry already knows was heard from its hooks, which
            # know more than its file: a sweep that read the file before a hook landed never overwrites it.
            return registry, []
        case Died(membership=membership):
            return _ended_unheard(registry, membership, [SessionGone(membership.id)])
        case MovedOn(membership=membership):
            # The user moved the process on at the keyboard, so there is nothing to tell them.
            return _ended_unheard(registry, membership, [])
        case Prompted(session=session, at=at):
            # Marked as the turn opens, so what it changes is read against a repository it has not touched yet.
            # A turn opens from the prompt and nowhere else, so only a session sitting at one is marked: a
            # prompt that lands inside a running turn — one that was queued, or one whose Stop nobody heard —
            # would otherwise move the mark into the middle of the work it is there to measure, and the turn
            # would be told only what it did after that [LAW:no-ambient-temporal-coupling].
            return _enter(
                registry,
                event,
                lambda _: Working(since=at),
                lambda was: [Snapshot(session, was.membership.cwd)] if isinstance(was.state, Idle) else [],
            )
        case Stopped(session=session, closing=closing):
            # Compared before the turn is handed over to be summarised, never after: see Compare.
            return _enter(registry, event, lambda _: Idle(), lambda _was: [Compare(session), Summarise(session, closing)])
        case Interrupted(session=session, prompt=prompt, at=at) if _in_turn(registry.sessions.get(session), prompt):
            # Told as a stopped turn is: what it did before it was stopped is still what it did, and the telling says
            # it was stopped because the transcript's record of the interrupt is one of its steps. No hook carried this,
            # so there is no closing reply to stand in for one. Claude Code sends no idle_prompt after an interrupt
            # (2.1.281, none in 126 s), so the nudge is timed here, from when the interrupt was read.
            return _enter(registry, event, lambda _: Idle(due=at + IDLE_NUDGE_SECONDS), lambda _was: [Compare(session), Summarise(session, None)])
        case Interrupted():
            # A turn that already ended — its Stop was heard, or the next prompt has opened another — or one this
            # registry never heard open: there is nothing left running for the interrupt to stop.
            return registry, []
        case Waited():
            return _enter(registry, event, _waited)
        case PermissionRequested(at=at, request=request, on=on):
            deadline = at + registry.permission_deadline
            return _enter(registry, event, lambda _: Blocked(on=on, request=request, deadline=deadline, warned=False))
        case ToolFinished(at=at, call=call):
            return _enter(registry, event, lambda state: _finished(state, call, at))
        case Ended(session=session, reason="other") if session in registry.sessions and not isinstance(registry.sessions[session].state, Gone):
            # Nobody ended it at the keyboard: its terminal closed. Spoken, as a process found dead is.
            after, effects = _enter(registry, event, lambda _: Gone())
            return after, [*effects, SessionGone(session)]
        case Ended():
            # /exit, Ctrl-C, /clear, /resume, and logging out are the user's own doing, at the keyboard.
            return _enter(registry, event, lambda _: Gone())
        case Abandoned(session=session, request=request, at=at):
            return _abandoned(registry, session, request, at), []
        case Tick(at=at):
            return _ticked(registry, at)


def _started(membership: Membership, source: StartSource, previous: Session | None) -> Session:
    # Compaction starts a session again in the middle of a turn, so what it was
    # doing carries over. Every other start sits at the prompt, whatever the
    # registry last heard: a session resumed after a crash was never told it stopped.
    # SessionStart carries no permission_mode, so only compaction, which keeps its process, keeps the one it had;
    # a session started or resumed in a new process may have been given any mode.
    match (source, previous):
        case ("compact", Session(state=Working() | Blocked() | AtDialog()) as previous):
            return replace(previous, membership=membership)
        case ("compact", Session(state=Idle(due=due), mode=mode)):
            # A new idle period, which a nudge hands was timing for still has to come from hands.
            return Session(membership, Idle(due=due), mode, turn=None)
        case ("compact", Session(mode=mode)):
            return Session(membership, Idle(), mode, turn=None)
        case _:
            return Session(membership, Idle(), mode=None, turn=None)


def _join(registry: Registry, session: Session) -> tuple[Registry, list[Effect]]:
    previous = registry.sessions.get(session.membership.id)
    return registry.put(session), _transition(session.membership.id, None if previous is None else previous.state, session.state)


def _ended_unheard(registry: Registry, membership: Membership, said: list[Effect]) -> tuple[Registry, list[Effect]]:
    """A session the sweep found over, though no end hook said so; `said` is what the user hears about it."""
    match registry.sessions.get(membership.id):
        case None | Session(state=Gone()):
            # A session this run never listed ended while the daemon was down, or before a reboot: the user was not told
            # of it here, so there is nothing to take back.
            return registry, []
        case Session(membership=held) if held.pid != membership.pid:
            # Started again in a new process since the sweep looked; what it saw ending is not this session.
            return registry, []
        case Session(state=before) as was:
            return registry.put(replace(was, membership=membership, state=Gone())), [*_transition(membership.id, before, Gone()), *said]


def _enter(
    registry: Registry,
    event: SessionEvent,
    next: Callable[[SessionState], SessionState],
    also: Callable[[Session], list[Effect]] = lambda _: [],
) -> tuple[Registry, list[Effect]]:
    """The event moves a live session to its next state; `also` is what the move calls for beyond the transition.

    `also` is given the session as it stood before the move, not after: what a move calls for can depend on
    where it moved from, and a state already overwritten cannot be asked [LAW:no-ambient-temporal-coupling].
    """
    match registry.sessions.get(event.session):
        case None:
            # [LAW:no-silent-failure] an event for a session that never joined is a record, not a drop.
            return registry, [Audit(Unregistered(event)), *_unwaited(event)]
        case Session(state=Gone()):
            # Ended is final until the session starts again; a hook that lands late cannot revive it.
            return registry, [Audit(AfterEnd(event)), *_unwaited(event)]
        case Session(membership=membership, state=before, mode=held, turn=turn) as was:
            after, reported = next(before), _reported(event)
            # [LAW:dataflow-not-control-flow] every hook that carries a mode sets it, so a mode changed at the keyboard
            # is heard at the session's next hook, whatever that hook moves the session to.
            mode = held if reported is None else reported
            # The mode is noted before the transition's effects, so a request it narrates is explained knowing the mode it was asked in.
            return registry.put(Session(membership, after, mode, _turn(event, turn))), [
                *_remoded(membership.id, held, mode),
                *_transition(membership.id, before, after),
                *also(was),
            ]


def _reported(event: SessionEvent) -> Mode | None:
    """The mode the event's hook reported; None from the hooks that carry none."""
    match event:
        case Prompted(mode=mode) | Stopped(mode=mode) | PermissionRequested(mode=mode) | ToolFinished(mode=mode):
            return mode
        case Interrupted() | Waited() | Ended():
            return None


def _turn(event: SessionEvent, held: PromptId | None) -> PromptId | None:
    """The turn the session is in after the event: the one a prompt opens, or the one it was in."""
    match event:
        case Prompted(prompt=prompt):
            # Even a prompt that names no turn opens one, so an interrupt read late for the turn before it matches nothing.
            return prompt
        case _:
            return held


def _in_turn(session: Session | None, prompt: PromptId) -> bool:
    """Whether the session is in the middle of the turn this prompt opened."""
    match session:
        case Session(state=Working() | Blocked() | AtDialog(), turn=turn):
            return turn == prompt
        case _:
            return False


def _remoded(session: SessionId, before: Mode | None, after: Mode | None) -> list[Effect]:
    match after:
        case str() | UnknownMode() if after != before:
            # Noted from no mode too: a session resumed in a new process may be in another mode than the model was
            # last told. The model is told, and says nothing: a mode the user set at the keyboard is not news to
            # them, and one an approved plan set was said in the approval's readback.
            return [Note(ModeChanged(session, after))]
        case _:
            return []


def _unwaited(event: SessionEvent) -> list[Effect]:
    # A permission hook from a session this registry cannot block, such as one that started before the
    # daemon did, is let go at once rather than left to hang until Claude Code kills it.
    match event:
        case PermissionRequested(session=session, request=request):
            return [Reply(session, request, Withdraw())]
        case _:
            return []


def _finished(state: SessionState, call: FinishedCall, at: Instant) -> SessionState:
    match state:
        case Blocked(on=asked) | AtDialog(on=asked) if _same_call(asked, call):
            # The tool the session was waiting to run has run, so its dialog was answered at the keyboard.
            return Working(since=at)
        case _:
            return state


def _same_call(asked: Blocker, call: FinishedCall) -> bool:
    match (asked, call):
        case (Question(asked=questions), Question(asked=answered)):
            # A question answered at the keyboard comes back with the answers added to its input: it is the same
            # call when it asks the same questions.
            return questions == answered
        case (Plan(), PlanApproved()):
            # A session has one plan up at a time, and ExitPlanMode running is that plan approved.
            return True
        case _:
            return asked == call


def _waited(state: SessionState) -> SessionState:
    match state:
        case Idle():
            return Idle(nudged=True)
        case _:
            # [LAW:no-ambient-temporal-coupling] each hook posts from its own process, so an idle_prompt sent as the
            # user typed can land after the prompt it raced: a working session stays working, and is not nudged.
            # A blocked one is already asked about aloud, and its deadline is spoken.
            return state


def _abandoned(registry: Registry, session: SessionId, request: RequestId, at: Instant) -> Registry:
    match registry.sessions.get(session):
        case Session(state=Blocked(request=held)) as was if held == request:
            # No hook waits for a reply, so there is nothing to withdraw, answer, or deny.
            return registry.put(replace(was, state=Working(since=at)))
        case _:
            return registry


def _transition(session: SessionId, before: SessionState | None, after: SessionState) -> list[Effect]:
    # [LAW:dataflow-not-control-flow] every change of state passes through here, so no event
    # can leave a hook waiting or a request unspoken by taking a path that forgot to.
    match (before, after):
        case (Blocked(request=held), Blocked(request=asked)) if held == asked:
            # Compaction kept the session waiting on the same request.
            return []
        case (Blocked(request=held), Blocked(request=asked, on=on)):
            return [Reply(session, held, Withdraw()), Narrate(Asking(session, asked, on))]
        case (Blocked(request=held), _):
            # The session moved on without a voice answer: the user answered its dialog at the keyboard
            # and the tool ran, or the turn went on. The waiting hook is let go, deciding nothing.
            return [Reply(session, held, Withdraw())]
        case (_, Blocked(request=asked, on=on)):
            return [Narrate(Asking(session, asked, on))]
        case (Idle(nudged=False), Idle(nudged=True)):
            return [Speak(WaitingForYou(session))]
        case _:
            return []


def _expiry(on: Blocker, at: Instant) -> tuple[SessionState, HookReply]:
    """Where a request left unanswered at its deadline leaves the session, and what its hook is told."""
    match on:
        case Permission():
            # [LAW:no-silent-failure] silence never approves: an unanswered request is denied, and said to be.
            return Working(since=at), Deny(EXPIRED_MESSAGE)
        case Question() | Plan():
            # Silence cannot answer a question or judge a plan, so there is nothing to refuse: it is left to its
            # dialog, where the user may be answering it at the keyboard, rather than closed under them.
            return AtDialog(on), Withdraw()


def _ticked(registry: Registry, at: Instant) -> tuple[Registry, list[Effect]]:
    after, effects = registry, list[Effect]()
    for session in registry.sessions.values():
        state, due = _deadline(session.membership.id, session.state, at)
        after = after.put(replace(session, state=state))
        effects += due
    return after, effects


def _deadline(session: SessionId, state: SessionState, at: Instant) -> tuple[SessionState, list[Effect]]:
    match state:
        case Blocked(on=on, request=request, deadline=deadline) if at >= deadline:
            after, reply = _expiry(on, at)
            return after, [Reply(session, request, reply), Speak(Expired(session, on))]
        case Blocked(on=on, deadline=deadline, warned=False) if at >= deadline - WARNING_LEAD_SECONDS:
            return replace(state, warned=True), [Speak(DeadlineNear(session, on, remaining=deadline - at))]
        case Idle(nudged=False, due=float() as due) if at >= due:
            # Nudged as an idle_prompt nudges, through the one place a nudge is said.
            return Idle(nudged=True), _transition(session, state, Idle(nudged=True))
        case _:
            return state, []
