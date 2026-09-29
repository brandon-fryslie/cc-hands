"""The one owner of the session registry."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger

from hands.core import drafts, keyboard
from hands.core.drafts import DraftOutcome, DraftRequest
from hands.core.effects import AfterEnd, Audit, AuditRecord, Compare, Decision, Effect, Heard, HookReply, Input, Narrate, Note, NotTyped, Reply, Repository, SessionGone, Snapshot, Speak, Story, Summarise, Type, Typed, Holding, Unmatched, Unregistered, Unsettled, Withdraw
from hands.core.events import Abandoned, Event, PermissionRequested, Stopped, Tick, ToolFinished
from hands.core.keyboard import KeyboardOutcome, KeyboardRequest
from hands.core.permissions import Answer, Outcome, answer
from hands.core.reducer import reduce
from hands.core.session import Gone, Instant, Known, Membership, Registry, RequestId, Session, SessionId, status_stamp
from hands.core.status import Stamp
from hands.sessions.audit import Applied, EffectFailed, Performed, Record, Typing
from hands.sessions.clock import stamp_now
from hands.sessions.delta import Changes, NoChanges
from hands.sessions.payload import Rejected
from hands.sessions.transcript import ai_title
from hands.sessions.typing import Untyped, type_into


@dataclass(frozen=True)
class Listing[S: Known]:
    session: S
    title: str | None  # Claude Code's ai-title, absent until it has named the session


class Sessions:
    """Applies events, draft and keyboard requests, and answers to waiting sessions through the core, performs their effects, and answers who is running."""

    def __init__(
        self,
        permission_deadline: float,
        clock: Callable[[], Instant],
        record: Record,
        changes: Changes | None = None,
        typist: Callable[[Type[Input]], None] = type_into,
        stamp: Callable[[], Stamp] = stamp_now,
    ) -> None:
        # [LAW:no-shared-mutable-globals] the registry is replaced only here, one event or request at a time.
        self._registry = Registry(permission_deadline=permission_deadline, sessions={}, drafts={})
        # [LAW:effects-at-boundaries] the one clock: hooks, answers, and ticks are all stamped from it.
        self._clock = clock
        # [LAW:one-source-of-truth] the wall clock Claude Code stamps its statuses and records with: a Stop is heard on
        # it and a transcript read through on it, and the two are compared.
        self._stamp = stamp
        # [LAW:single-enforcer] every event and every effect passes through here, so here is where each becomes an audit line.
        self._record = record
        # What a turn did to the repository it ran in. A daemon given none tells every turn by its steps alone.
        self._changes = changes or NoChanges()
        # What types a Type into its session, or raises Untyped.
        self._typist = typist
        # A blocking hook's connection waits on its future; only a Reply effect resolves one, until shutdown lets them all go.
        self._waiting: dict[RequestId, asyncio.Future[HookReply]] = {}
        # Set once, at shutdown: from then on a permission hook is let go as soon as it asks.
        self._released = False
        self._heard: asyncio.Queue[Heard] = asyncio.Queue()
        # Apart from what is heard: a summary takes seconds of model time, which must not hold up a permission request.
        self._story: asyncio.Queue[Story] = asyncio.Queue()

    def now(self) -> Instant:
        return self._clock()

    def stamp(self) -> Stamp:
        return self._stamp()

    async def apply(self, event: Event) -> None:
        before = self._registry
        self._registry, effects = reduce(before, event)
        if effects or self._registry != before:
            self._record(Applied(event))
        await self._perform_all(effects)

    async def ask(self, event: PermissionRequested | Stopped) -> HookReply:
        """Apply a hook Claude Code waits on and wait for its reply: a permission request's answer, withdrawal, or deny at
        its deadline; a Stop's letting go once the reducer has decided whose it is."""
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

    def release_waiting(self) -> None:
        """At shutdown, let every waiting hook go undecided, and every later one as it asks: its session's own dialog stands, and the daemon can exit."""
        self._released = True
        for request, waiting in self._waiting.items():
            if not waiting.done():
                logger.info(f"shutting down: request {request} is left to its session's dialog")
                waiting.set_result(Withdraw())

    async def answer(self, request: RequestId, decision: Decision) -> Outcome:
        self._registry, outcome, effects = answer(self._registry, Answer(request, decision))
        await self._perform_all(effects)
        return outcome

    async def draft(self, request: DraftRequest) -> DraftOutcome:
        """Apply a draft request. A send is typed into its session, and the outcome is whether that was done."""
        self._registry, decided = drafts.decide(self._registry, request)
        match decided:
            case Type() as effect:
                return await self._type(effect)
            case outcome:
                return outcome

    async def keyboard(self, request: KeyboardRequest) -> KeyboardOutcome:
        """Apply a command or an interrupt. It is typed into its session, and the outcome is whether that was done."""
        match keyboard.decide(self._registry, request):
            case Type() as effect:
                return await self._type(effect)
            case outcome:
                return outcome

    async def _type[I: Input](self, effect: Type[I]) -> Typed[I] | NotTyped[I]:
        # [LAW:single-enforcer] everything typed into a session passes here, so every one is in the log before it is typed.
        self._record(Typing(effect))
        try:
            await asyncio.to_thread(self._typist, effect)
        except Untyped as error:
            # [LAW:no-silent-failure] said, with what was to be typed, which is nowhere else now.
            logger.error(f"{effect.input} was not typed into session {effect.session}: {error}")
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
        return [_listing(session) for session in self._registry.live()]

    def membership(self, session: SessionId) -> Membership | None:
        """Where any session the registry has heard of works, ended or not: a session's last turn is told after it ends."""
        known = self._registry.sessions.get(session)
        return None if known is None else known.membership

    def listing(self, session: SessionId) -> Listing[Known] | None:
        """Any session the registry has heard of, ended or not; None for one it never has."""
        known = self._registry.sessions.get(session)
        return None if known is None else _listing(known)

    async def _perform_all(self, effects: list[Effect]) -> None:
        for effect in effects:
            match effect:
                case Audit(record=record):
                    # The effect is the line itself; its record is written as it is, not wrapped as performed.
                    self._record(record)
                case _:
                    pass
            try:
                await self._perform(effect)
            except Exception as error:
                self._record(EffectFailed(effect, str(error)))
                raise
            match effect:
                case Audit():
                    pass
                case _:
                    self._record(Performed(effect))

    async def _perform(self, effect: Effect) -> None:
        match effect:
            case Audit(record=record):
                logger.log(*_audited(record))
            case Reply(session=session, request=request, reply=reply):
                self._reply(session, request, reply)
            case Speak() | Narrate() | Note():
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
        if waiting is None:
            # [LAW:no-silent-failure] the hook's connection is gone, most often because Claude Code's own timeout killed it.
            logger.warning(f"reply {reply} for session {session} request {request}, which no hook is waiting on")
            return
        logger.info(f"replying to session {session} request {request}: {reply}")
        waiting.set_result(reply)


def _listing[S: Known](session: S) -> Listing[S]:
    try:
        title = ai_title(session.membership.transcript)
    except (Rejected, OSError) as error:
        # [LAW:no-silent-failure] a transcript hands cannot read names no session; the session is still
        # listed, spoken, and answered under its directory, and the log says why it has no title.
        logger.error(f"cannot read the title of session {session.membership.id} from {session.membership.transcript}: {error}")
        title = None
    return Listing(session, title)


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
        case Holding(session=session, prompt=prompt):
            return "DEBUG", f"Stop of turn {prompt} in session {session} is held until a record names its id"
        case Unsettled(session=session, prompt=prompt):
            return "INFO", f"Stop of turn {prompt} in session {session} ended nothing: its session ended or started again before a record named its id"
