"""The session lifecycle as one pure function."""

from collections.abc import Callable
from dataclasses import replace

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
    Unregistered,
    Withdraw,
)
from hands.core.events import (
    Abandoned,
    Attached,
    Died,
    Ended,
    MovedOn,
    Event,
    Joined,
    PermissionRequested,
    Prompted,
    SessionEvent,
    StartSource,
    Stopped,
    Tick,
    ToolFinished,
)
from hands.core.session import Blocked, Gone, Idle, Instant, Membership, Permission, Registry, RequestId, Session, SessionId, SessionState, Working

# How long before a permission's deadline the one warning is spoken.
WARNING_LEAD_SECONDS = 10.0

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
            return _join(registry, membership, _started(source, registry.sessions.get(membership.id)))
        case Attached(membership=membership) if membership.id not in registry.sessions:
            return registry.put(Session(membership, Idle())), []
        case Attached():
            # [LAW:no-ambient-temporal-coupling] a session the registry already knows was heard from its hooks, which
            # know more than its file: a sweep that read the file before a hook landed never overwrites it.
            return registry, []
        case Died(membership=membership):
            return _ended_unheard(registry, membership, [Speak(SessionGone(membership.id))])
        case MovedOn(membership=membership):
            # The user moved the process on at the keyboard, so there is nothing to tell them.
            return _ended_unheard(registry, membership, [])
        case Prompted(at=at):
            return _enter(registry, event, lambda _: Working(since=at))
        case Stopped():
            return _enter(registry, event, lambda _: Idle())
        case PermissionRequested(at=at, request=request, permission=permission):
            deadline = at + registry.permission_deadline
            return _enter(registry, event, lambda _: Blocked(on=permission, request=request, deadline=deadline, warned=False))
        case ToolFinished(at=at, call=call):
            return _enter(registry, event, lambda state: _finished(state, call, at))
        case Ended(session=session, reason="other") if session in registry.sessions and not isinstance(registry.sessions[session].state, Gone):
            # Nobody ended it at the keyboard: its pane or terminal closed. Spoken, as a process found dead is.
            after, effects = _enter(registry, event, lambda _: Gone())
            return after, [*effects, Speak(SessionGone(session))]
        case Ended():
            # /exit, Ctrl-C, /clear, /resume, and logging out are the user's own doing, at the keyboard.
            return _enter(registry, event, lambda _: Gone())
        case Abandoned(session=session, request=request, at=at):
            return _abandoned(registry, session, request, at), []
        case Tick(at=at):
            return _ticked(registry, at)


def _started(source: StartSource, previous: Session | None) -> SessionState:
    # Compaction starts a session again in the middle of a turn, so what it was
    # doing carries over. Every other start sits at the prompt, whatever the
    # registry last heard: a session resumed after a crash was never told it stopped.
    match (source, previous):
        case ("compact", Session(state=Working() | Blocked() as state)):
            return state
        case _:
            return Idle()


def _join(registry: Registry, membership: Membership, state: SessionState) -> tuple[Registry, list[Effect]]:
    previous = registry.sessions.get(membership.id)
    return registry.put(Session(membership, state)), _transition(membership.id, None if previous is None else previous.state, state)


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
        case Session(state=before):
            return registry.put(Session(membership, Gone())), [*_transition(membership.id, before, Gone()), *said]


def _enter(registry: Registry, event: SessionEvent, next: Callable[[SessionState], SessionState]) -> tuple[Registry, list[Effect]]:
    match registry.sessions.get(event.session):
        case None:
            # [LAW:no-silent-failure] an event for a session that never joined is a record, not a drop.
            return registry, [Audit(Unregistered(event)), *_unwaited(event)]
        case Session(state=Gone()):
            # Ended is final until the session starts again; a hook that lands late cannot revive it.
            return registry, [Audit(AfterEnd(event)), *_unwaited(event)]
        case Session(membership=membership, state=before):
            after = next(before)
            return registry.put(Session(membership, after)), _transition(membership.id, before, after)


def _unwaited(event: SessionEvent) -> list[Effect]:
    # A permission hook from a session this registry cannot block, such as one that started before the
    # daemon did, is let go at once rather than left to hang until Claude Code kills it.
    match event:
        case PermissionRequested(session=session, request=request):
            return [Reply(session, request, Withdraw())]
        case _:
            return []


def _finished(state: SessionState, call: Permission, at: Instant) -> SessionState:
    match state:
        case Blocked(on=asked) if asked == call:
            # The tool the session was waiting to run has run, so its dialog was answered at the keyboard.
            return Working(since=at)
        case _:
            return state


def _abandoned(registry: Registry, session: SessionId, request: RequestId, at: Instant) -> Registry:
    match registry.sessions.get(session):
        case Session(membership=membership, state=Blocked(request=held)) if held == request:
            # No hook waits for a reply, so there is nothing to withdraw, answer, or deny.
            return registry.put(Session(membership, Working(since=at)))
        case _:
            return registry


def _transition(session: SessionId, before: SessionState | None, after: SessionState) -> list[Effect]:
    # [LAW:dataflow-not-control-flow] every change of state passes through here, so no event
    # can leave a hook waiting or a request unspoken by taking a path that forgot to.
    match (before, after):
        case (Blocked(request=held), Blocked(request=asked)) if held == asked:
            # Compaction kept the session waiting on the same request.
            return []
        case (Blocked(request=held), Blocked(request=asked, on=permission)):
            return [Reply(session, held, Withdraw()), Narrate(PermissionAsked(session, asked, permission))]
        case (Blocked(request=held), _):
            # The session moved on without a voice answer: the user answered its dialog at the keyboard
            # and the tool ran, or the turn went on. The waiting hook is let go, deciding nothing.
            return [Reply(session, held, Withdraw())]
        case (_, Blocked(request=asked, on=permission)):
            return [Narrate(PermissionAsked(session, asked, permission))]
        case _:
            return []


def _ticked(registry: Registry, at: Instant) -> tuple[Registry, list[Effect]]:
    after, effects = registry, list[Effect]()
    for session in registry.sessions.values():
        state, due = _deadline(session.membership.id, session.state, at)
        after = after.put(Session(session.membership, state))
        effects += due
    return after, effects


def _deadline(session: SessionId, state: SessionState, at: Instant) -> tuple[SessionState, list[Effect]]:
    match state:
        case Blocked(on=permission, request=request, deadline=deadline) if at >= deadline:
            # [LAW:no-silent-failure] silence never approves: an unanswered request is denied, and said to be.
            return Working(since=at), [Reply(session, request, Deny(EXPIRED_MESSAGE)), Speak(PermissionExpired(session, permission))]
        case Blocked(on=permission, deadline=deadline, warned=False) if at >= deadline - WARNING_LEAD_SECONDS:
            return replace(state, warned=True), [Speak(PermissionDeadlineNear(session, permission, remaining=deadline - at))]
        case _:
            return state, []
