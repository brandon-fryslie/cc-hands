"""The one owner of the session registry."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger

from hands.core.drafts import DraftOutcome, DraftRequest, decide
from hands.core.effects import AfterEnd, Audit, AuditRecord, Decision, Effect, Heard, HookReply, Narrate, Reply, Sending, Speak, Type, Unregistered
from hands.core.events import Event, PermissionRequested, Tick
from hands.core.permissions import AnswerPermission, PermissionOutcome, answer
from hands.core.reducer import reduce
from hands.core.session import Instant, Registry, RequestId, Session, SessionId
from hands.sessions.tmux import type_into
from hands.sessions.transcript import ai_title


@dataclass(frozen=True)
class Listing:
    session: Session
    title: str | None  # Claude Code's ai-title, absent until it has named the session


class Sessions:
    """Applies events, draft requests, and permission answers through the core, performs their effects, and answers who is running."""

    def __init__(self, permission_deadline: float, clock: Callable[[], Instant]) -> None:
        # [LAW:no-shared-mutable-globals] the registry is replaced only here, one event or request at a time.
        self._registry = Registry(permission_deadline=permission_deadline, sessions={}, drafts={})
        # [LAW:effects-at-boundaries] the one clock: hooks, answers, and ticks are all stamped from it.
        self._clock = clock
        # A blocking hook's connection waits on its future; only a Reply effect resolves one.
        self._waiting: dict[RequestId, asyncio.Future[HookReply]] = {}
        self._heard: asyncio.Queue[Heard] = asyncio.Queue()

    def now(self) -> Instant:
        return self._clock()

    async def apply(self, event: Event) -> None:
        self._registry, effects = reduce(self._registry, event)
        await self._perform_all(effects)

    async def ask(self, event: PermissionRequested) -> HookReply:
        """Apply a permission request and wait for its reply: an answer, a withdrawal, or the deny at its deadline."""
        waiting = asyncio.get_running_loop().create_future()
        # [LAW:no-ambient-temporal-coupling] registered before the event is applied, so no reply can be decided
        # before there is somewhere for it to go.
        self._waiting[event.request] = waiting
        try:
            await self.apply(event)
            return await waiting
        finally:
            # A hook Claude Code gave up on cancels this wait; a reply decided later is logged as unheard.
            del self._waiting[event.request]

    async def answer(self, request: RequestId, decision: Decision) -> PermissionOutcome:
        self._registry, outcome, effects = answer(self._registry, AnswerPermission(request, decision, at=self._clock()))
        await self._perform_all(effects)
        return outcome

    async def draft(self, request: DraftRequest) -> DraftOutcome:
        # [LAW:no-ambient-temporal-coupling] committed before any effect is awaited, so a hook
        # that lands while tmux types is reduced against this registry and not overwritten by it.
        # A send that fails while typing has let go of its draft: some of it may be in the pane.
        self._registry, outcome, effects = decide(self._registry, request)
        # A decided send runs to the end even if its caller is cancelled: the draft has already
        # been let go, so stopping part way would lose it without typing it.
        await asyncio.shield(self._perform_all(effects))
        return outcome

    async def keep_time(self, period: float) -> None:
        """Tell the reducer the time once a period, until cancelled. The period is how late a deadline can be heard."""
        while True:
            await asyncio.sleep(period)
            await self.apply(Tick(self._clock()))

    async def heard(self) -> Heard:
        """The next thing a session has to say to the user, in the order the reducer decided it."""
        return await self._heard.get()

    def live(self) -> list[Listing]:
        return [_listing(session) for session in self._registry.live()]

    def listing(self, session: SessionId) -> Listing | None:
        """Any session the registry has heard of, ended or not; None for one it never has."""
        known = self._registry.sessions.get(session)
        return None if known is None else _listing(known)

    async def _perform_all(self, effects: list[Effect]) -> None:
        for effect in effects:
            await self._perform(effect)

    async def _perform(self, effect: Effect) -> None:
        match effect:
            case Audit(record=record):
                logger.log(*_audited(record))
            case Type(pane=pane, input=input):
                await type_into(pane, input)
            case Reply(session=session, request=request, reply=reply):
                self._reply(session, request, reply)
            case Speak() | Narrate():
                self._heard.put_nowait(effect)

    def _reply(self, session: SessionId, request: RequestId, reply: HookReply) -> None:
        waiting = self._waiting.get(request)
        if waiting is None:
            # [LAW:no-silent-failure] the hook's connection is gone, most often because Claude Code's own timeout killed it.
            logger.warning(f"reply {reply} for session {session} request {request}, which no hook is waiting on")
            return
        logger.info(f"replying to session {session} request {request}: {reply}")
        waiting.set_result(reply)


def _listing(session: Session) -> Listing:
    return Listing(session, ai_title(session.membership.transcript))


def _audited(record: AuditRecord) -> tuple[str, str]:
    match record:
        case Unregistered(event=event):
            return "WARNING", f"{type(event).__name__} for session {event.session}, which never joined"
        case AfterEnd(event=event):
            return "WARNING", f"{type(event).__name__} for session {event.session}, which had already ended"
        case Sending(session=session, pane=pane, text=text):
            return "INFO", f"sending to session {session} in pane {pane}: {text!r}"
