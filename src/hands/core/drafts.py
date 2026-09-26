"""What the user asks of a draft, what came of it, and the functions that decide."""

from dataclasses import dataclass
from pathlib import Path

from hands.core.effects import Landed, Landing, MaybeTyped, NotTyped, Text, Type
from hands.core.session import AtDialog, Blocked, Gone, Registry, Sending, Session, SessionId, Staged, Unsure


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
class NotSent:
    """Nothing of the draft was typed, so it is staged as it was and can be sent again."""

    session: SessionId
    reason: str


@dataclass(frozen=True)
class MaybeSent:
    """The send failed after some or all of the draft may have reached the session: it is Unsure now."""

    session: SessionId
    reason: str


@dataclass(frozen=True)
class SentBefore:
    """A send of an Unsure draft, refused: the last one may already have reached the session."""

    session: SessionId
    reason: str


@dataclass(frozen=True)
class StillSending:
    session: SessionId


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
class Unwrapped:
    """The session was not started under fritter, so there is nothing to type into."""

    session: SessionId


@dataclass(frozen=True)
class AtItsDialog:
    """The session is waiting at a dialog, which would take a typed draft as its answer."""

    session: SessionId


DraftOutcome = (
    DraftStaged
    | DraftAmended
    | DraftDiscarded
    | DraftSent
    | NotSent
    | MaybeSent
    | SentBefore
    | StillSending
    | UnknownSession
    | NothingStaged
    | SessionEnded
    | Unwrapped
    | AtItsDialog
)


def decide(registry: Registry, request: DraftRequest) -> tuple[Registry, DraftOutcome | Type]:
    """One draft request in; the next registry and what came of it out, or the typing whose landing will say. No I/O."""
    match registry.sessions.get(request.session):
        case None:
            return registry, UnknownSession(request.session)
        case Session() as session:
            return _decide(registry, request, session)


def _decide(registry: Registry, request: DraftRequest, session: Session) -> tuple[Registry, DraftOutcome | Type]:
    id = request.session
    match (request, registry.drafts.get(id), session.state, session.membership.fritter):
        case (_, Sending(), _, _):
            return registry, StillSending(id)
        case (AmendDraft() | DiscardDraft() | SendDraft(), None, _, _):
            return registry, NothingStaged(id)
        case (DiscardDraft(), (Staged() as draft) | Unsure(draft=draft), _, _):
            # A draft outlives its session's end so that it can still be thrown away.
            return registry.unstage(id), DraftDiscarded(id, draft)
        case (_, _, Gone(), _):
            return registry, SessionEnded(id)
        case (StageDraft(draft=draft), (Staged() as before) | Unsure(draft=before) | (None as before), _, _):
            return registry.hold(id, draft), DraftStaged(id, draft, replaced=before)
        case (AmendDraft(draft=after), (Staged() as before) | Unsure(draft=before), _, _):
            return registry.hold(id, after), DraftAmended(id, before, after)
        case (SendDraft(), Unsure(reason=reason), _, _):
            return registry, SentBefore(id, reason)
        case (SendDraft(), Staged(), _, None):
            return registry, Unwrapped(id)
        case (SendDraft(), Staged(), Blocked() | AtDialog(), Path()):
            return registry, AtItsDialog(id)
        case (SendDraft(), Staged() as draft, _, Path() as socket):
            # A working session queues what is typed into it until its turn ends (measured on 2.1.270), so a send
            # to one is an ordinary send.
            return registry.hold(id, Sending(draft)), Type(id, socket, session.membership.pid, Text(draft.text))


def land(registry: Registry, effect: Type, landing: Landing) -> tuple[Registry, DraftOutcome]:
    """What came of typing a draft in; the next registry and what became of the draft out."""
    id = effect.session
    match (registry.drafts.get(id), landing):
        case (Sending(draft=draft), Landed()):
            return registry.unstage(id), DraftSent(id, draft)
        case (Sending(draft=draft), NotTyped(reason=reason)):
            return registry.hold(id, draft), NotSent(id, reason)
        case (Sending(draft=draft), MaybeTyped(reason=reason)):
            return registry.hold(id, Unsure(draft, reason)), MaybeSent(id, reason)
        case (held, _):
            # [LAW:no-silent-failure] a Sending draft is left only here, and every request refuses one, so this is a
            # bug: said, rather than letting a landing decide the fate of some other draft.
            raise RuntimeError(f"a send to session {id} landed with its draft {held!r}, which is not being sent")
