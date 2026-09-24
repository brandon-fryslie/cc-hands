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
from hands.core.status import Report

# How long before a permission's deadline the one warning is spoken.
WARNING_LEAD_SECONDS = 10.0

# How long a session sits at its prompt before it is said to be waiting: Claude Code's own idle_prompt came 61 s after
# a Stop (2.1.281), so a nudge hands times itself comes when that one would have.
IDLE_NUDGE_SECONDS = 60.0

# How long a turn Claude Code said went idle waits to be told for the record of how it ended: an interrupt's is written
# ~100 ms after the idle (2.1.282) and read within the tail's 0.1 s; after a double Escape none ever comes.
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
        case Prompted(session=session, at=at, prompt=prompt):
            # A turn opens from the prompt and nowhere else, so a prompt queued into a running turn marks nothing: it
            # would otherwise move the mark into the middle of the work it is there to measure, and the turn would be
            # told only what it did after that [LAW:no-ambient-temporal-coupling]. One that names another turn than
            # the running one was sent from the prompt, and opens its own: see _opens.
            return _enter(
                registry,
                event,
                lambda state: _prompted(state, registry.sessions[session].turn, prompt, _opens(registry.sessions[session], prompt), at),
                lambda was: _opened(was, _opens(was, prompt)),
            )
        case Taken(session=session, prompt=prompt) if (sent := _sent_over(registry.sessions.get(session), prompt)) is not None:
            # [LAW:no-ambient-temporal-coupling] the prompt the one now sent was sent over did run: taken, and ended by a
            # Stop or an interrupt, all before the tail read any of it. Its record is the positive knowledge that it ran.
            return _ran_unread(registry, event, sent, prompt, None)
        case Taken(session=session, prompt=prompt, opens=opens):
            match registry.sessions.get(session):
                case Session(state=Submitted(since=since), turn=turn) if turn == prompt:
                    # Working from when it was sent: the hooks it waited on are part of its turn.
                    return _enter(registry, event, lambda _: Working(since=since))
                case Session(state=Idle()):
                    # [LAW:no-ambient-temporal-coupling] read before its own hook was applied, as a daemon too slow for the
                    # shim's timeout lets happen: kept as the turn, so that hook finds its prompt already taken.
                    return _enter(registry, event, lambda state: state)
                case Session(state=Working() | Blocked() | AtDialog(), turn=turn) if prompt != turn and opens:
                    # A turn of its own opened from the prompt, read before its hook was applied: the one running is over,
                    # its Stop or its interrupt not heard. It is ended and told here, and the new turn kept as the turn,
                    # so the late hook finds it taken, as it would have from the prompt.
                    return _enter(registry, event, lambda _: Idle(), lambda was: [Compare(session), Summarise(session, was.turn, None)])
                case Session(state=Working() | Blocked() | AtDialog(), turn=turn, taken=taken) as was if prompt != turn:
                    # The turn going on under a flushed message's id, which Claude has not answered under yet: a message
                    # queued now carries this id, and is queued into this turn, not opening another.
                    return registry.put(replace(was, taken=taken | {prompt})), []
                case _:
                    # A turn under way going on under a queued prompt, or one read after it ended: nothing to move.
                    return registry, []
        case Stopped(prompt=stopped) if _ended_already(registry.sessions.get(event.session), stopped):
            # [LAW:no-ambient-temporal-coupling] the Stop of a turn a later prompt or turn already ended, applied after
            # it: it was told there, or is told now if its telling waited for it (see _untold), and ending the turn now
            # running would idle it and spend its mark.
            return _enter(registry, event, lambda state: state)
        case Stopped(session=session, closing=closing, prompt=str() as stopped) if (sent := _sent_over(registry.sessions.get(session), stopped)) is not None:
            # Its hook posts from its own process, so it can land after the next prompt's: it ends the turn it names, as
            # that turn's record would have, and not the prompt sent after it.
            return _ran_unread(registry, event, sent, stopped, closing)
        case Stopped(session=session, closing=closing, prompt=stopped):
            # Compared before the turn is handed over to be summarised, never after: see Compare.
            return _enter(registry, event, lambda _: Idle(), lambda _was: [Compare(session), Summarise(session, stopped, closing)])
        case Interrupted(session=session, prompt=prompt, at=at) if _in_turn(registry.sessions.get(session), prompt):
            # Told as a stopped turn is: what it did before it was stopped is still what it did, and the telling says
            # it was stopped because the transcript's record of the interrupt is one of its steps. No hook carried this,
            # so there is no closing reply to stand in for one. Claude Code sends no idle_prompt after an interrupt
            # (2.1.281, none in 126 s), so the nudge is timed here, from when the interrupt was read.
            return _enter(registry, event, lambda _: Idle(due=at + IDLE_NUDGE_SECONDS), lambda _was: [Compare(session), Summarise(session, prompt, None)])
        case Interrupted(session=session, prompt=prompt) if _ended_already(held := registry.sessions.get(session), prompt) and held is not None and held.untold is not None:
            # The record of how a turn Claude Code already said is over ended: what its telling waits for, if it does.
            return _enter(registry, event, lambda state: state)
        case Interrupted():
            # A turn that already ended — its Stop was heard, or the next prompt has opened another — or one this
            # registry never heard open: there is nothing left running for the interrupt to stop.
            return registry, []
        case Continued(session=session, was=was) if _in_turn(registry.sessions.get(session), was):
            # Still working, now on the queued message: the turn goes by its id, so the Escape that stops it is heard.
            return _enter(registry, event, lambda state: state)
        case Continued():
            # Read after the turn it went on in had ended, or of one this registry never heard open: nothing to move.
            return registry, []
        case Waited():
            return _enter(registry, event, _waited)
        case StatusReported(session=session, report=Report(status=status.Idle()), at=at) if _running_in(registry.sessions.get(session)):
            # [LAW:one-source-of-truth] Claude Code says the turn is over, however it was stopped, so it is: idle, and
            # nudged on hands' clock, as an interrupted turn is (no idle_prompt in 75 s after a double Escape, 2.1.282).
            # [LAW:no-ambient-temporal-coupling] the status read is the status now, with no stamp to compare: Claude
            # Code sets idle only once a Stop's hooks have returned, and the shim waits for the Stop to be applied, so a
            # stopped turn is normally ended by its Stop; and it sets busy before a prompt's hooks run, so no idle read
            # after a prompt is applied is one from before it. The turn is told once the transcript says how it ended:
            # see _untold. What else ends it later ends nothing: see _named.
            return _enter(registry, event, lambda _: Idle(due=at + IDLE_NUDGE_SECONDS))
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
        case ("compact", Session(state=Idle(due=due), mode=mode, report=report, ended=ended, untold=untold)):
            # A new idle period, which a nudge hands was timing for still has to come from hands.
            return Session(membership, Idle(due=due), mode, turn=None, ended=ended, report=report, untold=untold)
        case ("compact", Session(mode=mode, report=report, ended=ended, untold=untold)):
            return Session(membership, Idle(), mode, turn=None, ended=ended, report=report, untold=untold)
        case (_, Session(ended=ended, untold=untold)):
            # A turn ended before the restart is still told, and its late Stop or interrupt still ends nothing.
            return Session(membership, Idle(), mode=None, turn=None, ended=ended, untold=untold)
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
            turn, taken, ended = _named(event, was)
            untold, telling = _untold(event, was)
            # A turn left untold is told before what the event calls for, so before a prompt marks the next one.
            return registry.put(replace(was, state=after, mode=mode, turn=turn, taken=taken, ended=ended, report=_report(event, was), untold=untold)), [
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


def _untold(event: SessionEvent, was: Session) -> tuple[Untold | None, list[Effect]]:
    """The turn left untold after the event, and the telling of the one before if the event is what it waited for."""
    match event:
        case StatusReported(report=Report(status=status.Idle()), at=at) if _running(was.state):
            return Untold(was.turn, at + UNTOLD_SECONDS), _telling(was, None)
        case Stopped(closing=closing, prompt=prompt) if _ended_already(was, prompt):
            # Its Stop fired after Claude Code set idle, as an Escape's can: told with the reply it carries.
            return None, _telling(was, closing)
        case Interrupted(prompt=prompt) if _ended_already(was, prompt):
            return None, _telling(was, None)
        case Ended():
            # Told before the session is said to be gone, in the order they happened.
            return None, _telling(was, None)
        case Prompted(prompt=prompt) | Taken(prompt=prompt) if prompt not in was.ended:
            # A turn after it opens: it is told before the new turn is marked, so it is compared against its own mark.
            return None, _telling(was, None)
        case _:
            return was.untold, []


def _telling(was: Session, closing: str | None) -> list[Effect]:
    match was.untold:
        case None:
            return []
        case Untold(turn=turn):
            return [Compare(was.membership.id), Summarise(was.membership.id, turn, closing)]


def _named(event: SessionEvent, was: Session) -> tuple[PromptId | None, frozenset[PromptId], frozenset[PromptId]]:
    """The turn the session is in after the event, the other ids it has gone on under, and those of the last turn ended
    before its Stop was heard."""
    over = frozenset({was.turn} if was.turn is not None else set()) | was.taken
    match event:
        case Prompted(prompt=prompt) if _opens(was, prompt) and _running(was.state):
            # Opened from the prompt with a turn still running: that one ended unheard, and its late Stop ends nothing.
            return prompt, frozenset(), over
        case Taken(prompt=prompt, opens=True) if _running(was.state):
            return prompt, frozenset(), over
        case Taken(prompt=prompt) if isinstance(was.state, Idle):
            return prompt, was.taken, was.ended
        case Prompted(prompt=prompt) if _opens(was, prompt):
            # Even a prompt that names no turn opens one, so an interrupt read late for the turn before it matches nothing.
            return prompt, frozenset(), was.ended
        case Prompted(prompt=prompt) if was.turn is None:
            # Queued into a turn that was opened with no id: the hook names the id Claude Code runs it under.
            return prompt, was.taken, was.ended
        case Continued(now=now):
            return now, was.taken, was.ended
        case StatusReported(report=Report(status=status.Idle())) if _running(was.state):
            # Ended by Claude Code's word: its Stop, its interrupt, and its flushed ids read after this end nothing.
            return was.turn, was.taken, over
        case Taken(prompt=prompt) | Stopped(prompt=str() as prompt) if _sent_over(was, prompt) is not None:
            # Told now: what else is read of it later ends nothing, and nor does the late Stop of a turn ended before it.
            return was.turn, was.taken, was.ended | {prompt}
        case _:
            # A prompt queued into the running turn leaves it named as it was, so its interrupt is still heard.
            return was.turn, was.taken, was.ended


def _running(state: SessionState) -> bool:
    return isinstance(state, Working | Blocked | AtDialog)


def _running_in(session: Session | None) -> bool:
    return session is not None and _running(session.state)


def _ended_already(session: Session | None, prompt: PromptId | None) -> bool:
    """Whether a Stop or an interrupt is the late end of a turn hands already ended: one it names among those a later
    prompt, a later turn, or Claude Code's idle showed over, or, whatever it names, the one an idle session's telling
    waits for. Only what is known to have ended is: a Stop naming an id not heard of yet ends its turn as any Stop does."""
    return session is not None and (prompt in session.ended or (session.untold is not None and isinstance(session.state, Idle)))


def _sent_over(session: Session | None, prompt: PromptId) -> Submitted | None:
    """The prompt sent after the one this id names, where the session holds one: see Submitted.over."""
    match session:
        case Session(state=Submitted(over=over) as sent) if prompt == over:
            return sent
        case _:
            return None


def _ran_unread(registry: Registry, event: SessionEvent, sent: Submitted, prompt: PromptId, closing: str | None) -> tuple[Registry, list[Effect]]:
    """A prompt a later one was sent over ran and ended before any of it was read: told as itself, with what it changed
    read against the mark the later prompt set aside, and the later prompt left sent."""
    session = event.session
    return _enter(registry, event, lambda _: Submitted(since=sent.since), lambda _: [Compare(session, "set_aside"), Summarise(session, prompt, closing)])


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


def _opens(was: Session, prompt: PromptId | None) -> bool:
    """Whether a prompt opens a turn rather than being queued into the one its id names.

    A queued prompt's hook carries the id of the turn it is queued into, the one that turn went on under after a
    flush included (2.1.281), so a prompt naming another id opens its own even inside a running turn: that turn ended,
    with neither its Stop nor its interrupt heard yet. A prompt with no id can be matched to no turn, so inside one it
    is taken to be queued into it.
    """
    match was.state:
        case Idle():
            # From the prompt, whatever it names: one whose record was read before this hook landed opened a turn too.
            return True
        case Submitted():
            return prompt is None or prompt != was.turn
        case Working() | Blocked() | AtDialog():
            return prompt is not None and was.turn is not None and prompt != was.turn and prompt not in was.taken
        case Gone():
            return False


def _opened(was: Session, opens: bool) -> list[Effect]:
    """What a prompt calls for beyond its move, given the session as it stood before: a turn it opens is marked, so what
    the turn changes is read against a repository it has not touched yet."""
    session, mark = was.membership.id, Snapshot(was.membership.id, was.membership.cwd)
    match was.state:
        case _ if not opens:
            return []
        case Working() | Blocked() | AtDialog():
            # [LAW:no-ambient-temporal-coupling] the turn it finds running ended unheard, so it is ended here, as its
            # Stop or its interrupt would have ended it: compared before the new mark replaces its own, and told as
            # itself. The record of how it ended, read after this, finds another turn open and moves nothing.
            return [Compare(session), Summarise(session, was.turn, None), mark]
        case _:
            return [mark]


def _prompted(state: SessionState, turn: PromptId | None, prompt: PromptId | None, opens: bool, at: Instant) -> SessionState:
    match (state, opens):
        case (Submitted(since=since), False):
            # Queued into the turn it opened, so that one was taken, whether or not its record has been read yet.
            return Working(since=since)
        case (Submitted(), True) if prompt is not None and prompt != turn:
            # Sent over one still sent: that one was cancelled, or ran unread. See Submitted.over.
            return Submitted(since=at, over=turn)
        case (_, True) if prompt is not None and prompt != turn:
            # [LAW:types-are-the-program] not working yet: Claude Code takes a prompt only once its hooks finish, and an
            # Escape before then cancels it with nothing to say so, so only the record of its turn can make it Working.
            return Submitted(since=at)
        case _:
            # Inside a running turn it is in that turn already; a prompt read as taken before its hook landed was taken;
            # and a prompt with no id can never be matched to its record, so it is taken on the hook's word, as every
            # prompt was before the record could be read.
            return Working(since=at)


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
        # Nothing said how it ended by its deadline, as after a double Escape before a flushed message is answered: told
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
            return Idle(nudged=True), _transition(session, state, Idle(nudged=True))
        case _:
            return state, []
