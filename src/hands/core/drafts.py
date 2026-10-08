"""What the user asks of a draft, what came of it, and the one function that decides."""

from dataclasses import dataclass

from hands.core.effects import Fritter, NotTyped, Text, Type, Typed
from hands.core.reach import AtItsDialog, SessionEnded, UnknownSession, Unreached, Unwrapped, prompter
from hands.core.session import Gone, Known, Registry, Session, SessionId, Staged
from hands.core.tmux import Keyboard, Pane


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
class Sending:
    """A send, with where keys typed for its session go in tmux, read just before it is decided.

    [LAW:types-are-the-program] only a send types, so only a send carries a pane: staging, amending, and discarding are
    decided the moment they are asked, in the order they were asked, with nothing read first.
    """

    session: SessionId
    pane: Keyboard


Decidable = StageDraft | AmendDraft | DiscardDraft | Sending


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


def decide(registry: Registry, request: Decidable) -> tuple[Registry, DraftOutcome | Type[Text]]:
    """One draft request in; the next registry and what came of it out, or what to type. No I/O."""
    match registry.sessions.get(request.session):
        case None:
            # This is also what keeps a draft out of a session at a startup dialog - workspace trust, a project's new MCP
            # servers - since Claude Code runs no hook, SessionStart included, until they are answered (measured on 2.1.283).
            return registry, UnknownSession(request.session)
        case known:
            return _decide(registry, request, known, registry.drafts.get(request.session))


def _decide(registry: Registry, request: Decidable, session: Known, staged: Staged | None) -> tuple[Registry, DraftOutcome | Type[Text]]:
    id = request.session
    match (request, staged, session):
        case (AmendDraft() | DiscardDraft() | Sending(), None, _):
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
        case (Sending(pane=pane), Staged() as draft, Session() as session):
            match prompter(session, pane):
                case Unwrapped() | AtItsDialog() as unreached:
                    return registry, unreached
                case Fritter() | Pane() as by:
                    # Sent the moment it is decided: the draft leaves the registry here, so there is never a second send
                    # of it. A working session queues what is typed into it until its turn ends (measured on 2.1.270).
                    return registry.unstage(id), Type(id, by, Text(draft.text))
