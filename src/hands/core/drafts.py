"""What the user asks of a draft, what came of it, and the one function that decides."""

from dataclasses import dataclass

from hands.core.effects import Audit, Effect, Sending, Text, Type
from hands.core.session import Blocked, Gone, Idle, Permission, Registry, Session, SessionId, Staged, Working


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


@dataclass(frozen=True)
class SendDraft:
    session: SessionId


DraftRequest = StageDraft | AmendDraft | DiscardDraft | SendDraft


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
class DraftSent:
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


@dataclass(frozen=True)
class OutsideTmux:
    session: SessionId


@dataclass(frozen=True)
class AwaitingPermission:
    """The session shows a permission dialog, which would swallow the text and take the Enter as its answer."""

    session: SessionId
    permission: Permission


DraftOutcome = (
    DraftStaged
    | DraftAmended
    | DraftDiscarded
    | DraftSent
    | UnknownSession
    | NothingStaged
    | SessionEnded
    | OutsideTmux
    | AwaitingPermission
)


def decide(registry: Registry, request: DraftRequest) -> tuple[Registry, DraftOutcome, list[Effect]]:
    """One draft request in; the next registry, what came of it, and the effects it calls for out. No I/O."""
    match registry.sessions.get(request.session):
        case None:
            return registry, UnknownSession(request.session), []
        case Session() as session:
            return _decide(registry, request, session, registry.drafts.get(request.session))


def _decide(
    registry: Registry, request: DraftRequest, session: Session, staged: Staged | None
) -> tuple[Registry, DraftOutcome, list[Effect]]:
    id = request.session
    match (request, staged, session.state, session.membership.pane):
        case (AmendDraft() | DiscardDraft() | SendDraft(), None, _, _):
            return registry, NothingStaged(id), []
        case (DiscardDraft(), Staged() as draft, _, _):
            # A draft outlives its session's end so that it can still be thrown away.
            return registry.unstage(id), DraftDiscarded(id, draft), []
        case (_, _, Gone(), _):
            return registry, SessionEnded(id), []
        case (_, _, _, None):
            return registry, OutsideTmux(id), []
        case (StageDraft(draft=draft), _, _, _):
            return registry.stage(id, draft), DraftStaged(id, draft, replaced=staged), []
        case (AmendDraft(draft=after), Staged() as before, _, _):
            return registry.stage(id, after), DraftAmended(id, before, after), []
        case (SendDraft(), Staged(), Blocked(on=permission), _):
            return registry, AwaitingPermission(id, permission), []
        case (SendDraft(), Staged() as draft, Idle() | Working(), str() as pane):
            # A working session is an ordinary send: Claude Code queues a prompt submitted
            # mid-turn and runs it when the turn ends, so the daemon holds no queue of its own.
            # [LAW:no-silent-failure] the audit comes first, so a send that fails while typing is still on record.
            return registry.unstage(id), DraftSent(id, draft), [Audit(Sending(id, pane, draft.text)), Type(pane, Text(draft.text))]
