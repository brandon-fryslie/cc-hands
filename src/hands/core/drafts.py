"""What the user asks of a draft, what came of it, and the one function that decides."""

from dataclasses import dataclass

from hands.core.session import Gone, Registry, Session, SessionId, Staged


@dataclass(frozen=True)
class StageDraft:
    session: SessionId
    draft: Staged


@dataclass(frozen=True)
class AmendDraft:
    session: SessionId
    draft: Staged


@dataclass(frozen=True)
class DiscardDraft:
    session: SessionId


DraftRequest = StageDraft | AmendDraft | DiscardDraft


@dataclass(frozen=True)
class DraftStaged:
    session: SessionId
    draft: Staged
    replaced: Staged | None


@dataclass(frozen=True)
class DraftAmended:
    session: SessionId
    before: Staged
    after: Staged


@dataclass(frozen=True)
class DraftDiscarded:
    session: SessionId
    draft: Staged


@dataclass(frozen=True)
class UnknownSession:
    session: SessionId


@dataclass(frozen=True)
class NothingStaged:
    session: SessionId


@dataclass(frozen=True)
class SessionEnded:
    session: SessionId


DraftOutcome = DraftStaged | DraftAmended | DraftDiscarded | UnknownSession | NothingStaged | SessionEnded


def decide(registry: Registry, request: DraftRequest) -> tuple[Registry, DraftOutcome]:
    """One draft request in; the next registry and what came of it out. No I/O, and nothing for an edge to do."""
    match registry.sessions.get(request.session):
        case None:
            return registry, UnknownSession(request.session)
        case Session() as session:
            return _decide(registry, request, session, registry.drafts.get(request.session))


def _decide(registry: Registry, request: DraftRequest, session: Session, staged: Staged | None) -> tuple[Registry, DraftOutcome]:
    id = request.session
    match (request, staged, session.state):
        case (AmendDraft() | DiscardDraft(), None, _):
            return registry, NothingStaged(id)
        case (DiscardDraft(), Staged() as draft, _):
            # A draft outlives its session's end so that it can still be thrown away.
            return registry.unstage(id), DraftDiscarded(id, draft)
        case (_, _, Gone()):
            return registry, SessionEnded(id)
        case (StageDraft(draft=draft), _, _):
            return registry.stage(id, draft), DraftStaged(id, draft, replaced=staged)
        case (AmendDraft(draft=after), Staged() as before, _):
            return registry.stage(id, after), DraftAmended(id, before, after)
