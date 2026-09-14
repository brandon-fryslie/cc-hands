"""A session's membership and lifecycle state, and the registry that holds them."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NewType, Self

SessionId = NewType("SessionId", str)
RequestId = NewType("RequestId", str)
TmuxPane = NewType("TmuxPane", str)
Instant = float  # monotonic seconds

# Prompt text that holds no terminal control characters, so typing it into a pane
# cannot press a key the text does not name. Made only where the model's words are parsed.
PromptText = NewType("PromptText", str)


@dataclass(frozen=True)
class Membership:
    """Which process and pane a session is, as the shim recorded it at SessionStart."""

    id: SessionId
    pid: int
    pane: TmuxPane | None  # None when Claude Code runs outside tmux
    cwd: Path
    transcript: Path


@dataclass(frozen=True)
class Permission:
    tool: str
    input: Mapping[str, object]


# [LAW:types-are-the-program] a session is in exactly one of these, and each
# carries only what is true of that state: a working session has a start, a
# blocked one has the request it waits on and when that request expires.
@dataclass(frozen=True)
class Idle:
    pass


@dataclass(frozen=True)
class Working:
    since: Instant


@dataclass(frozen=True)
class Blocked:
    on: Permission
    request: RequestId
    deadline: Instant


@dataclass(frozen=True)
class Gone:
    pass


SessionState = Idle | Working | Blocked | Gone


@dataclass(frozen=True)
class Session:
    membership: Membership
    state: SessionState


@dataclass(frozen=True)
class Resolution:
    """A spoken phrase the model turned into something exact, such as a file name."""

    heard: str
    meant: str


@dataclass(frozen=True)
class Staged:
    """A draft waiting for the user's word: the text to send and how it was resolved."""

    text: PromptText
    resolutions: tuple[Resolution, ...]


@dataclass(frozen=True)
class Registry:
    permission_timeout: float
    sessions: Mapping[SessionId, Session]
    # [LAW:types-are-the-program] a session with no entry has nothing staged; there is no empty draft.
    drafts: Mapping[SessionId, Staged]

    def put(self, session: Session) -> Self:
        return replace(self, sessions={**self.sessions, session.membership.id: session})

    def stage(self, session: SessionId, draft: Staged) -> Self:
        return replace(self, drafts={**self.drafts, session: draft})

    def unstage(self, session: SessionId) -> Self:
        return replace(self, drafts={id: draft for id, draft in self.drafts.items() if id != session})

    def live(self) -> list[Session]:
        return [session for session in self.sessions.values() if not isinstance(session.state, Gone)]
