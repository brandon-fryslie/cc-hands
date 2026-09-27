"""What the user asks a session's keyboard to do besides send a draft, what came of it, and the one function that decides."""

from dataclasses import dataclass
from pathlib import Path

from hands.core.effects import Command, Key, NotTyped, Type, Typed
from hands.core.reach import AtItsDialog, SessionEnded, UnknownSession, Unreached, Unwrapped
from hands.core.session import Gone, Idle, Registry, Running, Session, SessionId
from hands.core.status import Waiting


@dataclass(frozen=True)
class SendCommand:
    session: SessionId
    command: Command


@dataclass(frozen=True)
class Interrupt:
    """Stop what the session is doing, as Escape at its keyboard does."""

    session: SessionId


KeyboardRequest = SendCommand | Interrupt


@dataclass(frozen=True)
class NothingRunning:
    """The session is at its prompt, so there is nothing to interrupt."""

    session: SessionId


KeyboardOutcome = Unreached | NothingRunning | Typed[Command | Key] | NotTyped[Command | Key]


def decide(registry: Registry, request: KeyboardRequest) -> KeyboardOutcome | Type[Command | Key]:
    """One request in; what came of it, or what to type. The registry is unchanged either way. No I/O."""
    id = request.session
    match registry.sessions.get(id):
        case None:
            # A session at a startup dialog is here too: Claude Code runs no hook until it is answered (2.1.283).
            return UnknownSession(id)
        case Session(state=state, membership=member):
            match (request, state, member.fritter):
                case (_, Gone(), _):
                    return SessionEnded(id)
                case (_, _, None):
                    return Unwrapped(id)
                case (SendCommand(), Running(status=Waiting()), Path()):
                    # A dialog takes the command's characters and its Return as the answer to what it asked.
                    return AtItsDialog(id)
                case (SendCommand(command=command), _, Path() as socket):
                    # A working session queues it, and runs it as a command once its turn ends (measured on 2.1.283).
                    return Type(id, socket, member.pid, command)
                case (Interrupt(), Idle(), Path()):
                    return NothingRunning(id)
                case (Interrupt(), _, Path() as socket):
                    # [LAW:types-are-the-program] Escape is the one key a request can press, and at a dialog it is the
                    # dialog's own "no": it closes and the turn stops, which is what stop means (a question, 2.1.283).
                    return Type(id, socket, member.pid, Key("escape"))
