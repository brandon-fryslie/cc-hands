"""The one owner of the session registry."""

from dataclasses import dataclass

from loguru import logger

from hands.core.effects import AfterEnd, Audit, AuditRecord, Effect, Unregistered
from hands.core.events import Event
from hands.core.reducer import reduce
from hands.core.session import Registry, Session
from hands.sessions.transcript import ai_title


@dataclass(frozen=True)
class Listing:
    session: Session
    title: str | None  # Claude Code's ai-title, absent until it has named the session


class Sessions:
    """Applies events through the reducer, performs its effects, and answers who is running."""

    def __init__(self, permission_timeout: float) -> None:
        # [LAW:no-shared-mutable-globals] the registry is replaced only here, one event at a time.
        self._registry = Registry(permission_timeout=permission_timeout, sessions={})

    def apply(self, event: Event) -> None:
        self._registry, effects = reduce(self._registry, event)
        for effect in effects:
            _perform(effect)

    def live(self) -> list[Listing]:
        return [Listing(session, ai_title(session.membership.transcript)) for session in self._registry.live()]


def _perform(effect: Effect) -> None:
    match effect:
        case Audit(record=record):
            logger.warning(_audited(record))


def _audited(record: AuditRecord) -> str:
    match record:
        case Unregistered(event=event):
            return f"{type(event).__name__} for session {event.session}, which never joined"
        case AfterEnd(event=event):
            return f"{type(event).__name__} for session {event.session}, which had already ended"
