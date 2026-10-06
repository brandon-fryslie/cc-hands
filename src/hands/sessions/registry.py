"""The one owner of the session registry."""

import asyncio
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Literal, get_args

from loguru import logger

from hands.core import drafts, keyboard
from hands.core.drafts import DraftOutcome, DraftRequest
from hands.core.effects import AfterEnd, Audit, AuditRecord, Compare, Decision, Effect, Heard, HookReply, Input, Narrate, Note, NotTyped, Progress, Reply, Repository, SessionGone, Snapshot, Speak, Story, Summarise, Tell, Type, Typed, Holding, Unclosed, Unmatched, Unregistered, Unsettled, Withdraw
from hands.core.events import Abandoned, Event, PermissionRequested, Stopped, Tick, ToolFinished
from hands.core.keyboard import KeyboardOutcome, KeyboardRequest
from hands.core.permissions import Answer, Outcome, answer
from hands.core.reducer import reduce
from hands.core.session import Gone, Instant, Known, Membership, Registry, RequestId, Session, SessionId, status_stamp
from hands.core.status import Stamp
from hands.core.tmux import Keyboard, NotInTmux, PaneUnread
from hands.sessions.audit import Record, Typing, TypingFailed
from hands.sessions import wide
from hands.sessions.wide import Begun, annotate, begun, continuing, count, here, since, unit
from hands.sessions.clock import stamp_now
from hands.sessions.hookconfig import STOP_HOLD_SECONDS
from hands.sessions.delta import Changes, NoChanges
from hands.sessions.names import Names
from hands.sessions.payload import Rejected
from hands.sessions.transcript import session_name
from hands.sessions.typing import Untyped, type_into


# How a Stop's hook ended: decided inside the hold, or let go at it with Claude Code going on undecided.
StopHeld = Literal["decided", "let go"]

# Every kind of effect the core decides, each counted on the event of what called for it, 0 for one it did not.
EFFECT_KINDS = tuple(kind.__name__ for kind in get_args(Effect))


@dataclass(frozen=True)
class Performed:
    """An effect an applied event called for and how its performing went: "ok"; "failed", with why; "cancelled" partway;
    or "not performed", because one of its session's effects decided before it failed first, or the daemon stopped
    before its turn. `duration_ms` is how long it took, None for one never begun."""

    effect: Effect
    outcome: wide.Outcome | Literal["not performed"]
    duration_ms: float | None
    error: str | None


async def unread(pids: Sequence[int]) -> list[Keyboard]:
    """No pane read for any of `pids`: what a Sessions told of no tmux answers."""
    return [PaneUnread("this daemon reads no tmux panes")] * len(pids)


@dataclass(frozen=True)
class Listing[S: Known]:
    session: S
    name: str | None  # the session's name as it stands (Names.current), absent until it has one


class Sessions:
    """Applies events, draft and keyboard requests, and answers to waiting sessions through the core, performs their effects, and answers who is running."""

    def __init__(
        self,
        permission_deadline: float,
        clock: Callable[[], Instant],
        record: Record,
        changes: Changes | None = None,
        typist: Callable[[Type[Input]], Awaitable[None]] = partial(type_into, {}),
        keyboards: Callable[[Sequence[int]], Awaitable[list[Keyboard]]] = unread,
        stamp: Callable[[], Stamp] = stamp_now,
        stop_hold: float = STOP_HOLD_SECONDS,
        names: Names | None = None,
    ) -> None:
        # [LAW:no-shared-mutable-globals] the registry is replaced only here, one event or request at a time.
        self._registry = Registry(permission_deadline=permission_deadline, sessions={}, drafts={})
        # [LAW:effects-at-boundaries] the one clock: hooks, answers, and ticks are all stamped from it.
        self._clock = clock
        # [LAW:one-source-of-truth] the wall clock Claude Code stamps its statuses and records with: a Stop is heard on
        # it and a transcript read through on it, and the two are compared.
        self._stamp = stamp
        self._stop_hold = stop_hold
        # [LAW:single-enforcer] every event and every effect passes through here, so here is where each becomes a wide event.
        self._record = record
        # What a turn did to the repository it ran in. A daemon given none tells every turn by its steps alone.
        self._changes = changes or NoChanges()
        # What types a Type into its session, or raises Untyped.
        self._typist = typist
        # The tmux pane whose keys reach each of a list of processes, which a session's writer is chosen by. A daemon
        # given none reads none, so nothing it is not told to type into is typed into.
        self._keyboards = keyboards
        # The names hands decided and has not yet given, which every listing names its session by. A daemon given none
        # has decided none.
        self._names = names or Names()
        # A blocking hook's connection waits on its future; only a Reply effect resolves one, until shutdown lets them all go.
        self._waiting: dict[RequestId, asyncio.Future[HookReply]] = {}
        # Set once, at shutdown: from then on a permission hook is let go as soon as it asks.
        self._released = False
        # The waiting hooks shutdown let go, each marked as its future is resolved: a Stop decided by the reducer just
        # before shutdown began was decided, though shutdown has since released everything.
        self._let_go: set[RequestId] = set()
        self._heard: asyncio.Queue[Heard] = asyncio.Queue()
        # Apart from what is heard: a summary takes seconds of model time, which must not hold up a permission request.
        self._story: asyncio.Queue[Story] = asyncio.Queue()
        # Each session's latest performing still under way: what the next of that session's waits on.
        self._performing: dict[SessionId, asyncio.Task[None]] = {}
        # What was heard where nothing waits on it, still being performed.
        self._hearing: set[asyncio.Future[None]] = set()

    def now(self) -> Instant:
        return self._clock()

    def stamp(self) -> Stamp:
        return self._stamp()

    async def apply(self, event: Event) -> None:
        # Shielded: the event is in the registry already, so a caller that gives up never costs it its effects.
        await asyncio.shield(self._decided(event))

    def hear(self, event: Event) -> None:
        """Apply an event heard where nothing waits on it: decided now, in the order it was heard, and performed after."""
        hearing = self._decided(event)
        self._hearing.add(hearing)
        hearing.add_done_callback(self._performed)

    def _performed(self, hearing: asyncio.Future[None]) -> None:
        self._hearing.discard(hearing)
        if not hearing.cancelled() and (error := hearing.exception()) is not None:
            # [LAW:no-silent-failure] nothing awaits it to raise to, so it is said here; an effect that failed is named on
            # its applied event as well.
            logger.error(f"performing what was heard failed: {type(error).__name__}: {error}")

    def _decided(self, event: Event) -> asyncio.Future[None]:
        """The event reduced now, and its effects scheduled, under the event of its applying."""
        before = self._registry
        self._registry, effects = reduce(before, event)
        performed = _unperformed(effects)
        # Begun now, as it is decided, in the context of whatever unit of work applied it: inside a hook post's, it is that
        # post's child.
        applying = begun()
        # [LAW:no-ambient-temporal-coupling] the effects are scheduled as they are decided, before the applying opens its
        # unit, so they are started inside its span: what an effect opens, such as a repository mark, is its child, and
        # the applying is timed from before any of them ran.
        with continuing(applying.span):
            scheduled = self._scheduled(performed)
        if _quiet(before, self._registry, effects):
            return scheduled
        return asyncio.ensure_future(self._applied(event, performed, scheduled, applying))

    async def _applied(self, applied: Event | Answer, performed: list[Performed], performing: Awaitable[None], applying: Begun) -> None:
        # [LAW:nothing-unseen] one event for each event or answer the registry applies, open until every effect it
        # called for is performed: what was applied, each effect with its outcome and time, and the effects counted by
        # kind. [LAW:one-source-of-truth] it is the one record of them; no line beside it repeats an effect.
        with unit("applied", self._record, counts=EFFECT_KINDS, began=applying):
            annotate(applied=applied)
            try:
                await performing
            finally:
                annotate(effects=tuple(performed))
                count(**Counter(type(each.effect).__name__ for each in performed))

    def _scheduled(self, performed: list[Performed]) -> asyncio.Future[None]:
        """The effects being performed, each session's once what was decided before them of that session is.

        [LAW:no-ambient-temporal-coupling] one session's effects are performed in the order they were decided, and no
        session's wait on another's: the turn queued behind a turn told from the wire is marked before that turn's Stop
        hook lets Claude Code go on to run it, and a turn is told before its session is said to be gone.
        """
        grouped: dict[SessionId | None, list[int]] = {}
        for index, each in enumerate(performed):
            grouped.setdefault(_ordered_by(each.effect), []).append(index)
        # Each chain started now, as it is decided, never once the applying first runs.
        return asyncio.ensure_future(_all([self._chained(session, performed, group) for session, group in grouped.items()]))

    def _chained(self, session: SessionId | None, performed: list[Performed], indices: list[int]) -> asyncio.Task[None]:
        after = None if session is None else self._performing.get(session)
        performing = asyncio.create_task(self._perform_after(after, performed, indices))
        if session is not None:
            self._performing[session] = performing
            performing.add_done_callback(lambda done: self._performing.pop(session) if self._performing.get(session) is done else None)
        return performing

    async def _perform_after(self, after: asyncio.Task[None] | None, performed: list[Performed], indices: list[int]) -> None:
        if after is not None:
            # Only its order matters here: how it went is its own caller's to hear.
            await asyncio.wait({after})
        await self._perform_all(performed, indices)

    async def stop(self, event: Stopped) -> StopHeld:
        """Apply a Stop, and return once the reducer has decided whose it is, or once the hold has passed without that,
        saying which.

        Claude Code waits on the hook meanwhile, so what deciding it calls for, a comparison and the mark of a turn
        queued behind it, is done before Claude Code goes on [LAW:no-ambient-temporal-coupling]. The hold is the one
        bound on that wait, whatever the transcript's reading costs: past it, the hook is let go.
        """
        waiting = asyncio.get_running_loop().create_future()
        self._waiting[event.request] = waiting
        try:
            await self.apply(event)
            await asyncio.wait_for(asyncio.shield(waiting), self._stop_hold)
            # [LAW:no-ambient-temporal-coupling] which resolved this hold is read off the hold, not off whether
            # shutdown has begun by the time this resumes.
            return "let go" if event.request in self._let_go else "decided"
        except TimeoutError:
            # Claude Code goes on before the Stop is decided.
            await self.apply(Abandoned(event.session, event.request, self._clock()))
            return "let go"
        except asyncio.CancelledError:
            await self.apply(Abandoned(event.session, event.request, self._clock()))
            raise
        finally:
            del self._waiting[event.request]
            self._let_go.discard(event.request)

    async def ask(self, event: PermissionRequested) -> HookReply:
        """Apply a permission request and wait for its reply: an answer, a withdrawal, or the deny at its deadline."""
        if self._released:
            # A hook that reached the socket as shutdown began would otherwise wait with nothing left to answer it.
            return Withdraw()
        waiting = asyncio.get_running_loop().create_future()
        # [LAW:no-ambient-temporal-coupling] registered before the event is applied, so no reply can be decided
        # before there is somewhere for it to go.
        self._waiting[event.request] = waiting
        try:
            await self.apply(event)
            return await waiting
        except asyncio.CancelledError:
            # The hook's connection closed: the user answered No or Esc at the dialog, which kills the hook, or
            # Claude Code gave up on it, or exited. No reply can reach it now, so the session stops waiting,
            # and a voice answer after this is told the request is gone.
            # [LAW:no-silent-failure] a reply decided in the instant before the close was logged as sent; this says it was not.
            match waiting.result() if waiting.done() and not waiting.cancelled() else None:
                case None | Withdraw():
                    # Nothing was decided, or only a withdrawal, which prints nothing either way.
                    lost = ""
                case decision:
                    lost = f"; its reply {decision} was never delivered"
            logger.info(f"the hook for session {event.session} request {event.request} closed{lost}")
            await self.apply(Abandoned(event.session, event.request, self._clock()))
            raise
        finally:
            del self._waiting[event.request]
            self._let_go.discard(event.request)

    def release_waiting(self) -> None:
        """At shutdown, let every waiting hook go undecided, and every later one as it asks: its session's own dialog stands, and the daemon can exit."""
        self._released = True
        for request, waiting in self._waiting.items():
            if not waiting.done():
                logger.info(f"shutting down: request {request} is left to its session's dialog")
                waiting.set_result(Withdraw())
                self._let_go.add(request)

    async def answer(self, request: RequestId, decision: Decision) -> Outcome:
        answered = Answer(request, decision)
        before = self._registry
        self._registry, outcome, effects = answer(before, answered)
        if not _quiet(before, self._registry, effects):
            performed = _unperformed(effects)
            # Performed at once, in no session's order: the session is waiting on this very reply.
            await self._applied(answered, performed, self._perform_all(performed, list(range(len(performed)))), begun())
        return outcome

    async def draft(self, request: DraftRequest) -> DraftOutcome:
        """Apply a draft request. A send is typed into its session, and the outcome is whether that was done."""
        match request:
            case drafts.SendDraft(session=session):
                decidable: drafts.Decidable = drafts.Sending(session, await self._pane(session))
            case drafts.StageDraft() | drafts.AmendDraft() | drafts.DiscardDraft():
                decidable = request
        self._registry, decided = drafts.decide(self._registry, decidable)
        match decided:
            case Type() as effect:
                return await self._type(effect)
            case outcome:
                return outcome

    async def keyboard(self, request: KeyboardRequest) -> KeyboardOutcome:
        """Apply a command or an interrupt. It is typed into its session, and the outcome is whether that was done."""
        pane = await self._pane(request.session)
        match keyboard.decide(self._registry, request, pane):
            case Type() as effect:
                return await self._type(effect)
            case outcome:
                return outcome

    async def _pane(self, session: SessionId) -> Keyboard:
        """The tmux pane whose keys reach the session now, read before a request to type into it is decided."""
        match self._registry.sessions.get(session):
            case None:
                # A session the registry does not know is answered as unknown before its pane is looked at.
                return NotInTmux()
            case known:
                [pane] = await self._keyboards([known.membership.pid])
                return pane

    async def _type[I: Input](self, effect: Type[I]) -> Typed[I] | NotTyped[I]:
        # [LAW:single-enforcer] everything typed into a session passes here, so every one is in the log before it is typed,
        # joined to the event of the tool call that typed it by its span.
        self._record(Typing(effect, here()))
        try:
            await self._typist(effect)
        except Untyped as error:
            # [LAW:no-silent-failure] said, with what was to be typed, on the line that pairs with the Typing before it.
            self._record(TypingFailed(effect, str(error)))
            return NotTyped(effect.session, effect.input, str(error))
        return Typed(effect.session, effect.input)

    async def keep_time(self, period: float) -> None:
        """Tell the reducer the time once a period, until cancelled. The period is how late a deadline can be heard."""
        while True:
            await asyncio.sleep(period)
            await self.apply(Tick(self._clock()))

    async def story(self) -> Story:
        """The next turn a session finished or the next session gone, in the order they happened."""
        return await self._story.get()

    async def heard(self) -> Heard:
        """The next thing a session has to say to the user, in the order the reducer decided it."""
        return await self._heard.get()

    def live_count(self) -> int:
        """How many sessions have not ended, without reading their transcripts."""
        return len(self._registry.live())

    def live_members(self) -> list[Membership]:
        """The membership of every session that has not ended, without reading their transcripts."""
        return [session.membership for session in self._registry.live()]

    def status_read(self, session: SessionId) -> bool:
        """Whether Claude Code's status of the session has been read, so whether a turn of it runs is known."""
        live = self.live_session(session)
        return live is not None and status_stamp(live.state) is not None

    def live_sessions(self) -> dict[SessionId, Session]:
        """Every session that has not ended, as the registry holds it now, by id."""
        return {session.membership.id: session for session in self._registry.live()}

    def live_ids(self) -> list[SessionId]:
        """Every session that has not ended, by id."""
        return [session.membership.id for session in self._registry.live()]

    def live_session(self, session: SessionId) -> Session | None:
        """The session as the registry holds it now, or None once it has ended or if it never joined."""
        match self._registry.sessions.get(session):
            case Gone() | None:
                return None
            case Session() as live:
                return live

    def live(self) -> list[Listing[Session]]:
        return [self._listing(session) for session in self._registry.live()]

    def membership(self, session: SessionId) -> Membership | None:
        """Where any session the registry has heard of works, ended or not: a session's last turn is told after it ends."""
        known = self._registry.sessions.get(session)
        return None if known is None else known.membership

    def listing(self, session: SessionId) -> Listing[Known] | None:
        """Any session the registry has heard of, ended or not; None for one it never has."""
        known = self._registry.sessions.get(session)
        return None if known is None else self._listing(known)

    def _listing[S: Known](self, session: S) -> Listing[S]:
        try:
            held = session_name(session.membership.transcript)
        except (Rejected, OSError) as error:
            # [LAW:no-silent-failure] a transcript hands cannot read names no session; the session is still
            # listed, spoken, and answered under its project, and the log says why it has no name.
            logger.error(f"cannot read the name of session {session.membership.id} from {session.membership.transcript}: {error}")
            return Listing(session, None)
        return Listing(session, self._names.current(session.membership.id, held))

    async def _perform_all(self, performed: list[Performed], indices: list[int]) -> None:
        """Perform the effects at `indices` in order, each written back into `performed` as it ends; the first to fail
        raises, and those after it stay not performed."""
        # [LAW:no-shared-mutable-globals] `performed` is owned by the one applying that made it: each effect's slot is
        # written by the one chain that performs it, and read by the applying once every chain has ended.
        for index in indices:
            effect = performed[index].effect
            began = time.monotonic()
            try:
                await self._perform(effect)
            except asyncio.CancelledError:
                performed[index] = Performed(effect, "cancelled", since(began), None)
                raise
            except Exception as error:
                performed[index] = Performed(effect, "failed", since(began), f"{type(error).__name__}: {error}")
                raise
            performed[index] = Performed(effect, "ok", since(began), None)

    async def _perform(self, effect: Effect) -> None:
        match effect:
            case Audit(record=record):
                logger.log(*_audited(record))
            case Reply(session=session, request=request, reply=reply):
                self._reply(session, request, reply)
            case Speak() | Narrate() | Note() | Progress() | Tell():
                self._heard.put_nowait(effect)
            case Summarise() | SessionGone():
                self._story.put_nowait(effect)
            case Snapshot() | Compare():
                # [LAW:no-silent-failure] what a repository says about a turn is best effort, and may never
                # cost the turn the thing it was read for. Both reads are guarded here rather than each
                # guarded where it is called [LAW:single-enforcer], because it is one promise: a mark may not
                # fail the prompt hook that is waiting on it, and a reading may not cost the turn the
                # Summarise queued behind it, which is what has the turn spoken at all.
                try:
                    await self._repository(effect)
                except Exception as error:
                    logger.error(f"what the repository of session {effect.session} says could not be read: {type(error).__name__}: {error}")

    async def _repository(self, effect: Repository) -> None:
        match effect:
            case Snapshot(session=session, cwd=cwd):
                await self._changes.snapshot(session, cwd)
            case Compare(session=session, again=again):
                await self._changes.compare(session, again)

    def _reply(self, session: SessionId, request: RequestId, reply: HookReply) -> None:
        waiting = self._waiting.get(request)
        match waiting:
            case None:
                # Most often Claude Code's own timeout killed the hook, and its handler has since forgotten the request.
                why = "no hook is waiting on it"
            case _ if waiting.cancelled():
                # A closed connection cancels the future at once; the handler forgets the request only when it next runs.
                why = "its hook has closed"
            case _ if waiting.done():
                why = f"its hook was already given {waiting.result()}"
            case _:
                logger.info(f"replying to session {session} request {request}: {reply}")
                waiting.set_result(reply)
                return
        # [LAW:no-silent-failure] said with its cause, in the words the hook's own close uses for a reply it never got.
        logger.warning(f"reply {reply} for session {session} request {request} was never delivered: {why}")


def _unperformed(effects: list[Effect]) -> list[Performed]:
    return [Performed(effect, "not performed", None, None) for effect in effects]


def _quiet(before: Registry, after: Registry, effects: list[Effect]) -> bool:
    """Whether what the registry applied changed nothing and called for nothing, as a quiet tick or an answer to a request
    no session waits on: no unit of work, so no event, where the ticker alone would leave one every second. An answer that
    did nothing is said on the event of the tool call that gave it."""
    return not effects and after == before


async def _all(performing: list[asyncio.Task[None]]) -> None:
    """Every one of `performing` ended, however each ended; then the first to fail raises."""
    for ended in await asyncio.gather(*performing, return_exceptions=True):
        if isinstance(ended, BaseException):
            raise ended


def _ordered_by(effect: Effect) -> SessionId | None:
    """The session whose order the effect keeps; None for one that waits on nothing: a line, or what is heard."""
    match effect:
        case Reply(session=session) | Summarise(session=session) | SessionGone(session=session) | Snapshot(session=session) | Compare(session=session):
            return session
        case Audit() | Speak() | Narrate() | Note() | Progress() | Tell():
            return None


def _audited(record: AuditRecord) -> tuple[str, str]:
    match record:
        case Unregistered(event=ToolFinished() as event):
            # Every tool call of a session with no membership file lands here, as its prompts and stops do, which already warn.
            return "DEBUG", f"ToolFinished for session {event.session}, which never joined"
        case Unregistered(event=event):
            return "WARNING", f"{type(event).__name__} for session {event.session}, which never joined"
        case AfterEnd(event=event):
            return "WARNING", f"{type(event).__name__} for session {event.session}, which had already ended"
        case Unmatched(session=session, prompt=prompt):
            return "INFO", f"Stop of turn {prompt} in session {session} ended nothing: its turn was told, or no record read through it names its id"
        case Unclosed(session=session, prompt=prompt):
            return "INFO", f"reply closing turn {prompt} in session {session} on the wire ended nothing: its turn was told, or its session is in no turn by that id"
        case Holding(session=session, prompt=prompt):
            return "DEBUG", f"Stop of turn {prompt} in session {session} is held until a record names its id"
        case Unsettled(session=session, prompt=prompt):
            return "INFO", f"Stop of turn {prompt} in session {session} ended nothing: its session ended or started again before a record named its id"
