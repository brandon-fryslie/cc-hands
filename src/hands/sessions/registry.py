"""The one owner of the session registry."""

import asyncio
from dataclasses import dataclass

from loguru import logger

from hands.core.drafts import DraftOutcome, DraftRequest, decide
from hands.core.effects import AfterEnd, Audit, AuditRecord, Effect, Sending, Type, Unregistered
from hands.core.events import Event
from hands.core.reducer import reduce
from hands.core.session import Registry, Session, SessionId
from hands.sessions.tmux import type_into
from hands.sessions.transcript import ai_title


@dataclass(frozen=True)
class Listing:
    session: Session
    title: str | None  # Claude Code's ai-title, absent until it has named the session


class Sessions:
    """Applies events and draft requests through the core, performs their effects, and answers who is running."""

    def __init__(self, permission_timeout: float) -> None:
        # [LAW:no-shared-mutable-globals] the registry is replaced only here, one event or request at a time.
        self._registry = Registry(permission_timeout=permission_timeout, sessions={}, drafts={})

    async def apply(self, event: Event) -> None:
        self._registry, effects = reduce(self._registry, event)
        await _perform_all(effects)

    async def draft(self, request: DraftRequest) -> DraftOutcome:
        # [LAW:no-ambient-temporal-coupling] committed before any effect is awaited, so a hook
        # that lands while tmux types is reduced against this registry and not overwritten by it.
        # A send that fails while typing has let go of its draft: some of it may be in the pane.
        self._registry, outcome, effects = decide(self._registry, request)
        # A decided send runs to the end even if its caller is cancelled: the draft has already
        # been let go, so stopping part way would lose it without typing it.
        await asyncio.shield(_perform_all(effects))
        return outcome

    def live(self) -> list[Listing]:
        return [_listing(session) for session in self._registry.live()]

    def listing(self, session: SessionId) -> Listing | None:
        """Any session the registry has heard of, ended or not; None for one it never has."""
        known = self._registry.sessions.get(session)
        return None if known is None else _listing(known)


def _listing(session: Session) -> Listing:
    return Listing(session, ai_title(session.membership.transcript))


async def _perform_all(effects: list[Effect]) -> None:
    for effect in effects:
        await _perform(effect)


async def _perform(effect: Effect) -> None:
    match effect:
        case Audit(record=record):
            logger.log(*_audited(record))
        case Type(pane=pane, input=input):
            await type_into(pane, input)


def _audited(record: AuditRecord) -> tuple[str, str]:
    match record:
        case Unregistered(event=event):
            return "WARNING", f"{type(event).__name__} for session {event.session}, which never joined"
        case AfterEnd(event=event):
            return "WARNING", f"{type(event).__name__} for session {event.session}, which had already ended"
        case Sending(session=session, pane=pane, text=text):
            return "INFO", f"sending to session {session} in pane {pane}: {text!r}"
