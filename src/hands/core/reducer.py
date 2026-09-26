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
    Continued,
    Taken,
    Joined,
    PermissionRequested,
    Prompted,
    SessionEvent,
    StartSource,
    StatusReported,
    Stopped,
    Tick,
    ToolFinished,
    Waited,
)
from hands.core.session import AtDialog, Blocked, Blocker, Gone, Idle, Instant, Membership, Mode, Permission, Plan, PlanApproved, PromptId, Question, FinishedCall, Registry, RequestId, Session, SessionId, SessionState, Submitted, UnknownMode, Untold, Working
from hands.core import status
from hands.core.narration import reported_and_asked
from hands.core.status import Report, Stamp

# How long before a permission's deadline the one warning is spoken.
WARNING_LEAD_SECONDS = 10.0

# How long a session sits at its prompt before it is said to be waiting: Claude Code's own idle_prompt came 61 s after
# a Stop (2.1.281), so a nudge hands times itself comes when that one would have.
IDLE_NUDGE_SECONDS = 60.0

# How long a turn Claude Code said went idle waits to be told for the record of how it ended: an interrupt's is written
# ~100 ms after the idle (2.1.282) and read within the tail's 0.1 s; some ways of stopping a turn write none.
UNTOLD_SECONDS = 1.0

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
        case Prompted(at=at):
            # A turn opens at the prompt: a prompt heard while a turn runs is in that turn, as a queued message's hook names
            # it (2.1.281), and marks nothing, or the mark would move into the middle of the work it is there to measure
            # [LAW:no-ambient-temporal-coupling]. That message's own turn opens when it is taken: see _opens.
            return _enter(registry, event, lambda state: _prompted(state, at), _marked)
        case Taken(session=session, prompt=prompt) if (held := _running(registry, session)) is not None:
            # The record names the turn Claude Code runs; it ends none. Only whether it is the one sent says anything.
            return _enter(registry, event, lambda state: _taken(state, _names(held, prompt)))
        case Taken(session=session, at=at) if _opens(registry.sessions.get(session), event):
            # A turn no hook opened: named, so its Stop ends it. Not marked here, where a mark could land after Claude has
            # begun changing the repository: a message queued behind a turn was marked while that turn's Stop hook held
            # Claude Code (see _following), and one nothing was queued for, as a `!` command's answer, is told by its steps.
            return _enter(registry, event, lambda _: Working(since=at))
        case Taken():
            # Read after its turn ended, before any status was read, or of one Claude Code said was over since: nothing to move.
            return registry, []
        case Stopped(session=session, closing=closing, prompt=stopped, again=again) if _ends(registry.sessions.get(session), event):
            # Compared before the turn is handed over to be summarised, never after: see Compare.
            return _enter(registry, event, lambda state: Idle(asking=_asking(closing, state)), lambda was: [Compare(session, again), Summarise(session, stopped, closing), *_following(was)])
        case Stopped():
            # The Stop of a turn already over: told now if its telling waited for it (see _untold), and otherwise one
            # applied after the next turn's prompt, which ending would idle that turn and spend its mark.
            return _enter(registry, event, lambda state: state)
        case Interrupted(session=session, prompt=prompt) if (held := registry.sessions.get(session)) is not None and _awaiting(held) and _names(held, prompt):
            # The record of how the turn Claude Code said is over ended: what its telling waits for.
            return _enter(registry, event, lambda state: state)
        case Interrupted():
            # [LAW:one-source-of-truth] the transcript says how a turn ended, never that it did: Claude Code's idle ends
            # it, ~100 ms before this record is written (2.1.282), so there is nothing left running for it to stop.
            return registry, []
        case Continued(session=session, was=was) if (held := _running(registry, session)) is not None and _names(held, was):
            # Still working, now on the queued message: the turn goes by its id, so the Stop that ends it names it.
            return _enter(registry, event, lambda state: state)
        case Continued():
            # Read after the turn it went on in had ended, or of one this registry never heard open: nothing to move.
            return registry, []
        case Waited():
            return _enter(registry, event, _waited)
        case StatusReported(session=session, report=Report(status=status.Idle()), at=at) if _running(registry, session) is not None:
            # [LAW:one-source-of-truth] Claude Code says the turn is over, however it was stopped, so it is: idle, and
            # nudged on hands' clock (Claude Code sent no idle_prompt in 75 s after some, 2.1.282). A prompt still sent was
            # cancelled by an Escape during its hooks, or taken and stopped before its record was read; either way it is
            # left untold, and told as itself if it ran. [LAW:no-ambient-temporal-coupling] the status read is the status
            # now, with no stamp to compare: Claude Code sets idle only once a Stop's hooks have returned, and the shim
            # waits for the Stop to be applied, so a stopped turn is normally ended by its Stop; and it sets busy before
            # a prompt's hooks run, so no idle read after a prompt is applied is one from before it. The turn is told
            # once the transcript says how it ended: see _untold.
            return _enter(registry, event, lambda state: Idle(due=at + IDLE_NUDGE_SECONDS, asking=_asking(None, state)))
        case StatusReported():
            # Kept as Claude Code said it, for what asks what the session is doing: a session not running already is
            # where an idle leaves it, and busy or waiting say nothing a hook has not.
            return _enter(registry, event, lambda state: state)
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
        case ("compact", Session(state=Submitted() | Working() | Blocked() | AtDialog()) as previous):
            return replace(previous, membership=membership)
        case ("compact", Session(state=Idle(due=due, asking=asking), mode=mode, report=report, idled=idled, untold=untold)):
            # A new idle period, which a nudge hands was timing for still has to come from hands, about the question
            # still unanswered: compacting the context answers nothing.
            return Session(membership, Idle(due=due, asking=asking), mode, turn=None, report=report, idled=idled, untold=untold)
        case ("compact", Session(mode=mode, report=report, idled=idled, untold=untold)):
            return Session(membership, Idle(), mode, turn=None, report=report, idled=idled, untold=untold)
        case (_, Session(untold=untold)):
            # A turn ended before the restart is still told.
            return Session(membership, Idle(), mode=None, turn=None, untold=untold)
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
            # A turn ended and not told yet is told before the session is said to be gone: it happened first.
            return registry.put(replace(was, membership=membership, state=Gone(), untold=None)), [*_transition(membership.id, before, Gone()), *_telling(was, None), *said]


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
        case Session(membership=membership, state=before, mode=held) as was:
            after, reported = next(before), _reported(event)
            # [LAW:dataflow-not-control-flow] every hook that carries a mode sets it, so a mode changed at the keyboard
            # is heard at the session's next hook, whatever that hook moves the session to.
            mode = held if reported is None else reported
            # The mode is noted before the transition's effects, so a request it narrates is explained knowing the mode it was asked in.
            turn, taken = _named(event, was)
            untold, telling = _untold(event, was)
            # A turn left untold is told before what the event calls for, so before a prompt marks the next one.
            report = _report(event, was)
            queued = _queued(event, was, after)
            return registry.put(replace(was, state=after, mode=mode, turn=turn, taken=taken, report=report, idled=_idled(report, was), queued=queued, untold=untold)), [
                *_remoded(membership.id, held, mode),
                *_transition(membership.id, before, after),
                *telling,
                *also(was),
            ]


def _reported(event: SessionEvent) -> Mode | None:
    """The mode the event's hook reported; None from the hooks that carry none."""
    match event:
        case Prompted(mode=mode) | Stopped(mode=mode) | PermissionRequested(mode=mode) | ToolFinished(mode=mode):
            return mode
        case Taken() | Interrupted() | Continued() | Waited() | StatusReported() | Ended():
            return None


def _report(event: SessionEvent, was: Session) -> Report | None:
    """What Claude Code says the session is doing after the event: only its own report changes it."""
    match event:
        case StatusReported(report=report):
            return report
        case _:
            return was.report


def _idled(report: Report | None, was: Session) -> Stamp | None:
    """When Claude Code last set the session idle, with the report the event leaves it."""
    match report:
        case Report(status=status.Idle(), stamp=stamp):
            return stamp
        case Report(stamp=stamp) if was.report is None:
            # The first status read, with no idle read yet, as when hands attaches mid-turn: a record written before Claude
            # Code set it is of a turn begun by then, and one queued behind the running turn is written after it.
            return stamp
        case _:
            return was.idled


def _queued(event: SessionEvent, was: Session, after: SessionState) -> bool:
    """Whether a message the user sent while the turn ran still waits behind it after the event."""
    match event:
        case Prompted() if _busy(was.state):
            # Its hook fires as it is sent, under the running turn's id (2.1.282).
            return True
        case Taken(prompt=prompt) if not _names(was, prompt):
            # Taken into the running turn, as 2.1.281 took a queued message, or opening a turn of its own: either way
            # nothing waits behind the turn now running.
            return False
        case _:
            return was.queued and _busy(after)


def _following(was: Session) -> list[Effect]:
    """The turn queued behind the one a Stop ends is marked while the Stop hook holds Claude Code, which runs it only once
    that hook returns: so before it can have changed anything [LAW:no-ambient-temporal-coupling]. After the Summarise,
    so a mark that fails or is cut short never costs the turn before its telling."""
    match was:
        case Session(queued=True, membership=membership):
            return [Snapshot(membership.id, membership.cwd)]
        case _:
            return []


def _untold(event: SessionEvent, was: Session) -> tuple[Untold | None, list[Effect]]:
    """The turn left untold after the event, and the telling of the one before if the event is what it waited for."""
    match event:
        case StatusReported(report=Report(status=status.Idle()), at=at) if _busy(was.state):
            return Untold(was.turn, at + UNTOLD_SECONDS), _telling(was, None)
        case Stopped(closing=closing) if _awaiting(was):
            # Its Stop fired after Claude Code set idle, as an Escape's can: told with the reply it carries.
            return None, _telling(was, closing)
        case Interrupted(prompt=prompt) if _awaiting(was) and _names(was, prompt):
            return None, _telling(was, None)
        case Ended() | Prompted() | Taken():
            # Told before a session is said to be gone, and before the turn after it opens, so it is compared against its
            # own mark: in the order they happened. A Taken reaches here only opening a turn, or inside one.
            return None, _telling(was, None)
        case _:
            return was.untold, []


def _telling(was: Session, closing: str | None) -> list[Effect]:
    match was.untold:
        case None:
            return []
        case Untold(turn=turn):
            return [Compare(was.membership.id, again=False), Summarise(was.membership.id, turn, closing)]


def _named(event: SessionEvent, was: Session) -> tuple[PromptId | None, frozenset[PromptId]]:
    """The turn the session is in after the event, and the other ids it has gone on under."""
    match event:
        case Prompted(prompt=prompt) | Taken(prompt=prompt) if isinstance(was.state, Idle):
            return prompt, frozenset()
        case Stopped(prompt=prompt) if isinstance(was.state, Idle) and _ends(was, event):
            # A turn hands never had running, as a queued one whose Stop is applied before its record is read: named as
            # told, so that record, read after, opens nothing.
            return prompt, frozenset()
        case Prompted(prompt=prompt) | Taken(prompt=prompt) if _busy(was.state) and not _names(was, prompt):
            # A flush's id, taken seconds before Claude answers under it, which a message queued in between carries
            # (2.1.281); or a turn's opened before the idle of the one running was read, whose Stop then ends it.
            return was.turn, was.taken | {prompt}
        case Continued(now=now):
            return now, was.taken
        case _:
            return was.turn, was.taken


def _running(registry: Registry, session: SessionId) -> Session | None:
    """The session, if Claude Code has it busy with a turn."""
    held = registry.sessions.get(session)
    return held if held is not None and _busy(held.state) else None


def _busy(state: SessionState) -> bool:
    """Whether Claude Code has the session busy with a turn, as far as hands has heard: sent, working, or at a dialog."""
    return isinstance(state, Submitted | Working | Blocked | AtDialog)


def _names(session: Session, prompt: PromptId) -> bool:
    """Whether the id is one the session's turn goes by."""
    return prompt == session.turn or prompt in session.taken


def _opens(session: Session | None, taken: Taken) -> bool:
    """Whether a prompt taken at the prompt opens a turn no hook opened: under an id the session does not go by
    already, written since Claude Code last said the session is idle."""
    match session:
        case Session(state=Idle(), idled=int() as idled) if not _names(session, taken.prompt):
            # [LAW:one-source-of-truth] Claude Code's own two clocks, never the order hands read them in: an idle set after
            # the record was written is that turn over, stopped before any of it was read. It sets idle for ~3 ms between
            # a turn and the message queued behind it (2.1.282), which a read can land on, and which came first.
            return taken.written is not None and taken.written >= idled
        case _:
            # Running already, or with no idle read since hands began following it: a transcript read from its start.
            return False


def _awaiting(session: Session | None) -> bool:
    """Whether the session is idle with a turn Claude Code ended waiting to be told: see Untold."""
    return session is not None and isinstance(session.state, Idle) and session.untold is not None


def _ends(session: Session | None, stop: Stopped) -> bool:
    """Whether a Stop ends a turn: the busy one it names, or, at the prompt, one hands never had running, such as the turn
    a session was in when it was attached, or the last turn stopping again after another Stop hook blocked its Stop. Any
    other Stop of the last turn ends nothing at the prompt: that turn was told. The telling of a turn stopping again holds
    only what was not told before."""
    match session:
        case Session(state=state) if _busy(state):
            return _names(session, stop.prompt)
        case Session(state=Idle(), untold=None):
            return session.turn is None or not _names(session, stop.prompt) or stop.again
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


def _marked(was: Session) -> list[Effect]:
    """A turn opened from the prompt is marked, so what it changes is read against a repository it has not touched yet."""
    match was.state:
        case Idle():
            return [Snapshot(was.membership.id, was.membership.cwd)]
        case _:
            return []


def _prompted(state: SessionState, at: Instant) -> SessionState:
    match state:
        case Idle():
            # [LAW:types-are-the-program] not working yet: Claude Code takes a prompt only once its hooks finish, and an
            # Escape before then cancels it, so only the record of its turn can make it Working.
            return Submitted(since=at)
        case Submitted(since=since):
            # Queued into the turn it opened, so that one was taken, whether or not its record has been read yet.
            return Working(since=since)
        case _:
            # Inside a running turn it is in that turn already, and a dialog it was typed past was answered.
            return Working(since=at)


def _taken(state: SessionState, named: bool) -> SessionState:
    match state:
        case Submitted(since=since) if named:
            # Working from when it was sent: the hooks it waited on are part of its turn.
            return Working(since=since)
        case _:
            return state


def _asking(closing: str | None, state: SessionState) -> bool:
    """Whether a turn that ended in `state`, on the reply `closing`, left the listener something to answer: the
    two things `open_questions` counts in the telling, so the nudge and the telling agree on what asking is.

    `open_questions` itself cannot be called here, because it reads the turn's `Questioned` steps, which exist
    only in the transcript the tail reads, and nothing the reducer is handed carries them; making it the one
    function would take the narrator handing its finding back as an event, after the summariser has answered.
    So each half is read from the reducer's own record of the same fact. The closing text is the Stop's reply,
    read by the narration's own `reported_and_asked` [LAW:one-source-of-truth]. An unanswered `AskUserQuestion`
    is a turn that ended still at its dialog: answered at the keyboard or by voice, the tool ran and the session
    was working again before it stopped. The two differ where the dialog was escaped: the turn ended on it and is
    told as waiting on it, but the Escape closed the dialog's hook, which moved the session off the question first.
    """
    match state:
        case Blocked(on=Question()) | AtDialog(on=Question()):
            return True
        case _:
            return closing is not None and bool(reported_and_asked(closing)[1])


def _waited(state: SessionState) -> SessionState:
    match state:
        case Idle():
            return replace(state, nudged=True, due=None)
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
        case (Idle(nudged=False), Idle(nudged=True, asking=asking)):
            return [Speak(WaitingForYou(session, asking))]
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
        # Nothing said how it ended by its deadline, as nothing does for a turn stopped with no record: told
        # with what was read.
        settled = session.untold is not None and at >= session.untold.by
        after = after.put(replace(session, state=state, untold=None if settled else session.untold))
        effects += [*(_telling(session, None) if settled else []), *due]
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
            nudged = replace(state, nudged=True, due=None)
            return nudged, _transition(session, state, nudged)
        case _:
            return state, []
