"""A session hands drives on the user's standing order: what the user asks of a drive, what came of it, and the one
function that decides.

Driving is the user's word given once for many sends: "keep billing going until the tests pass". While a session is
driven, each turn it finishes is handed to the brain to act on, and the brain may type the session's next prompt with
no "send it" in between. [LAW:single-enforcer] this module is where that leave is checked: a prompt reaches a session
undictated only through `decide`, and only while the registry holds a drive for that session. A draft the user staged is
never touched by a drive; it waits for their word as it always does.
"""

from dataclasses import dataclass

from hands.core.effects import Fritter, NotTyped, Text, Type, Typed
from hands.core.reach import AtItsDialog, SessionEnded, UnknownSession, Unreached, Unwrapped, writer
from hands.core.session import Drive, Gone, PromptText, Registry, Running, Session, SessionId
from hands.core.status import Waiting
from hands.core.tmux import Keyboard, Pane

# [LAW:no-mode-explosion] the cap on what one standing order sends: a drive that has sent this many prompts ends, and the
# user gives the order again to go on. It bounds the loop a send makes, since each send ends in a turn handed back.
SENDS = 20


@dataclass(frozen=True)
class StartDrive:
    session: SessionId
    order: str


@dataclass(frozen=True)
class StopDrive:
    session: SessionId


@dataclass(frozen=True)
class DriveSend:
    session: SessionId
    text: PromptText


DriveRequest = StartDrive | StopDrive | DriveSend


@dataclass(frozen=True)
class DriveSending:
    """A send under a drive, with where keys typed for its session go in tmux, read just before it is decided."""

    session: SessionId
    text: PromptText
    pane: Keyboard


Decidable = StartDrive | StopDrive | DriveSending


@dataclass(frozen=True)
class Driving:
    session: SessionId
    drive: Drive
    replaced: Drive | None


@dataclass(frozen=True)
class DriveStopped:
    session: SessionId
    drive: Drive


@dataclass(frozen=True)
class NotDriven:
    """The user gave no standing order for this session, or it ended: what is sent to it waits for their word."""

    session: SessionId


@dataclass(frozen=True)
class DriveSpent:
    """The send that used the last of the drive's sends: typed, and the drive ended with it."""

    typed: Typed[Text] | NotTyped[Text]
    drive: Drive


DriveOutcome = Driving | DriveStopped | NotDriven | DriveSpent | Unreached | Typed[Text] | NotTyped[Text]


@dataclass(frozen=True)
class Send:
    """Type the prompt; the drive goes on."""

    type: Type[Text]


@dataclass(frozen=True)
class LastSend:
    """Type the prompt; the drive ended as it was decided, having sent all it may."""

    type: Type[Text]
    drive: Drive


def decide(registry: Registry, request: Decidable) -> tuple[Registry, DriveOutcome | Send | LastSend]:
    """One drive request in; the next registry and what came of it out, or what to type. No I/O."""
    id = request.session
    match (request, registry.sessions.get(id), registry.drives.get(id)):
        case (_, None, _):
            return registry, UnknownSession(id)
        case (StopDrive(), _, None) | (DriveSending(), _, None):
            return registry, NotDriven(id)
        case (StopDrive(), _, Drive() as drive):
            return registry.undrive(id), DriveStopped(id, drive)
        case (_, Gone(), _):
            return registry, SessionEnded(id)
        case (StartDrive(order=order), Session(), replaced):
            drive = Drive(order, 0)
            return registry.drive(id, drive), Driving(id, drive, replaced)
        case (DriveSending(text=text, pane=pane), Session(state=state, membership=member), Drive() as drive):
            match (state, writer(member, pane)):
                case (_, Unwrapped() as unwrapped):
                    return registry, unwrapped
                case (Running(status=Waiting()), _):
                    # A dialog would take the prompt as its answer, and what a session asks is the user's to answer.
                    return registry, AtItsDialog(id)
                case (_, Fritter() | Pane() as by):
                    sent = Drive(drive.order, drive.sends + 1)
                    typed = Type(id, by, Text(text))
                    # [LAW:dataflow-not-control-flow] the count decides which send this is, never a flag set beside it.
                    match sent.sends < SENDS:
                        case True:
                            return registry.drive(id, sent), Send(typed)
                        case False:
                            return registry.undrive(id), LastSend(typed, sent)
