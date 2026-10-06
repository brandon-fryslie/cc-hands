"""What the user asks a session's keyboard to do besides send a draft, what came of it, and the one function that decides."""

from dataclasses import dataclass

from hands.core.effects import Command, Fritter, Key, NotTyped, Type, Typed
from hands.core.reach import AtItsDialog, SessionEnded, UnknownSession, Unreached, Unwrapped, writer
from hands.core.session import Gone, Idle, Registry, Running, Session, SessionId
from hands.core.status import Waiting
from hands.core.tmux import InPane, Pane


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


def decide(registry: Registry, request: KeyboardRequest, pane: InPane) -> KeyboardOutcome | Type[Command | Key]:
    """One request in, with the tmux pane its session runs in; what came of it, or what to type. The registry is
    unchanged either way. No I/O."""
    id = request.session
    match registry.sessions.get(id):
        case None:
            # A session at a startup dialog is here too: Claude Code runs no hook until it is answered (2.1.283).
            return UnknownSession(id)
        case Gone():
            return SessionEnded(id)
        case Session(state=state, membership=member):
            match (request, state, writer(member, pane)):
                case (_, _, Unwrapped() as unwrapped):
                    return unwrapped
                case (SendCommand(), Running(status=Waiting()), _):
                    # A dialog takes the command's characters and its Return as the answer to what it asked.
                    return AtItsDialog(id)
                case (SendCommand(command=command), _, Fritter() | Pane() as by):
                    # A working session queues it, and runs it as a command once its turn ends (measured on 2.1.283).
                    return Type(id, by, command)
                case (Interrupt(), Idle(), _):
                    return NothingRunning(id)
                case (Interrupt(), _, Fritter() | Pane() as by):
                    # [LAW:types-are-the-program] Escape is the one key a request can press, and at a dialog it is the
                    # dialog's own "no": it closes and the turn stops, which is what stop means (a question, 2.1.283).
                    return Type(id, by, Key("escape"))
