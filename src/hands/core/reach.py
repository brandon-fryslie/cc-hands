"""Why a request to type into a session did not reach it, whatever the request was."""

from dataclasses import dataclass

from hands.core.session import SessionId


@dataclass(frozen=True)
class UnknownSession:
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
    """The session is waiting at a dialog, which would take what is typed as its answer."""

    session: SessionId


Unreached = UnknownSession | SessionEnded | Unwrapped | AtItsDialog
