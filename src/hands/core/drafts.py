"""What the user asks of a draft, what came of it, and the one function that decides."""

from dataclasses import dataclass
from pathlib import Path

from hands.core.effects import NotTyped, Text, Type, Typed
from hands.core.reach import AtItsDialog, SessionEnded, UnknownSession, Unreached, Unwrapped
from hands.core.session import Gone, Registry, Session, SessionId, Staged, at_a_dialog


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
class NothingStaged:
    session: SessionId


DraftOutcome =DraftStaged | DraftAmended | DraftDiscarded | NothingStaged | Unreached | Typed[Text] | NotTyped[Text]


def decide(registry: Registry, request: DraftRequest) -> tuple[Registry, DraftOutcome | Type[Text]]:
    """One draft request in; the next registry and what came of it out, or what to type. No I/O."""
    match registry.sessions.get(request.session):
        case None:
            # This is also what keeps a draft out of a session at a startup dialog - workspace trust, a project's new MCP
            # servers - since Claude Code runs no hook, SessionStart included, until they are answered (measured on 2.1.283).
            return registry, UnknownSession(request.session)
        case Session() as session:
            return _decide(registry, request, session, registry.drafts.get(request.session))


def _decide(registry: Registry, request: DraftRequest, session: Session, staged: Staged | None) -> tuple[Registry, DraftOutcome | Type[Text]]:
    id = request.session
    match (request, staged, session.state, session.membership.fritter):
        case (AmendDraft() | DiscardDraft() | SendDraft(), None, _, _):
            return registry, NothingStaged(id)
        case (DiscardDraft(), Staged() as draft, _, _):
            # A draft outlives its session's end so that it can still be thrown away.
            return registry.unstage(id), DraftDiscarded(id, draft)
        case (_, _, Gone(), _):
            return registry, SessionEnded(id)
        case (StageDraft(draft=draft), _, _, _):
            return registry.stage(id, draft), DraftStaged(id, draft, replaced=staged)
        case (AmendDraft(draft=after), Staged() as before, _, _):
            return registry.stage(id, after), DraftAmended(id, before, after)
        case (SendDraft(), Staged(), _, None):
            return registry, Unwrapped(id)
        case (SendDraft(), Staged(), _, Path()) if at_a_dialog(session):
            return registry, AtItsDialog(id)
        case (SendDraft(), Staged() as draft, _, Path() as socket):
            # Sent the moment it is decided: the draft leaves the registry here, so there is never a second send of it.
            # A working session queues what is typed into it until its turn ends (measured on 2.1.270).
            return registry.unstage(id), Type(id, socket, session.membership.pid, Text(draft.text))
