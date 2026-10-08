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
from hands.core.reach import AtItsDialog, SessionEnded, UnknownSession, Unreached, Unwrapped, prompter
from hands.core.session import Drive, Gone, PromptText, Registry, Session, SessionId
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


@dataclass(frozen=True)
class HandedBack:
    """The brain's turn on a driven session's finished turn is over: `handed` is the drive as the turn was handed under."""

    session: SessionId
    handed: Drive


@dataclass(frozen=True)
class Unsent:
    """A send under a drive did not reach the session: `sent` is the drive the send stored, `was` the one before it."""

    session: SessionId
    sent: Drive
    was: Drive


DriveRequest = StartDrive | StopDrive | DriveSend | HandedBack


@dataclass(frozen=True)
class DriveSending:
    """A send under a drive, with where keys typed for its session go in tmux, read just before it is decided."""

    session: SessionId
    text: PromptText
    pane: Keyboard


Decidable = StartDrive | StopDrive | DriveSending | HandedBack | Unsent


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


@dataclass(frozen=True)
class DriveDropped:
    """The brain's turn on a driven turn ended with the drive as it was handed: no prompt sent, no stop, no new order.
    No turn will come back to act on, so the drive ended with it."""

    session: SessionId
    drive: Drive


@dataclass(frozen=True)
class DriveWentOn:
    """The brain's turn on a driven turn moved the drive on: it sent, stopped, or the order changed."""

    session: SessionId


DriveOutcome = Driving | DriveStopped | NotDriven | DriveSpent | DriveDropped | DriveWentOn | Unreached | Typed[Text] | NotTyped[Text]


@dataclass(frozen=True)
class Send:
    """Type the prompt; the drive goes on, as `drive`, with the send counted."""

    type: Type[Text]
    drive: Drive


@dataclass(frozen=True)
class LastSend:
    """Type the prompt; the drive ended as it was decided, having sent all it may."""

    type: Type[Text]
    drive: Drive


def decide(registry: Registry, request: Decidable) -> tuple[Registry, DriveOutcome | Send | LastSend]:
    """One drive request in; the next registry and what came of it out, or what to type. No I/O."""
    id = request.session
    match (request, registry.sessions.get(id), registry.drives.get(id)):
        case (HandedBack(handed=handed), _, now) if now == handed:
            # [LAW:single-enforcer] the one check that a drive is live: every turn handed to the brain ends in a send,
            # a stop, or this, so a drive is never held with nobody driving it, however the brain's turn ended.
            return registry.undrive(id), DriveDropped(id, handed)
        case (HandedBack(), _, _):
            return registry, DriveWentOn(id)
        case (Unsent(sent=sent, was=was), _, now) if now == sent:
            # Nothing reached the session, so the send is not counted: the drive stands as it was handed.
            return registry.drive(id, was), DriveWentOn(id)
        case (Unsent(), _, _):
            # Stopped, reordered, or ended while the send was typed: that stands.
            return registry, DriveWentOn(id)
        case (_, None, _):
            return registry, UnknownSession(id)
        case (StopDrive(), _, Drive() as drive):
            return registry.undrive(id), DriveStopped(id, drive)
        case (_, Gone(), _):
            return registry, SessionEnded(id)
        case (StopDrive(), _, None) | (DriveSending(), _, None):
            return registry, NotDriven(id)
        case (StartDrive(order=order), Session(), None):
            drive = Drive(order, 0)
            return registry.drive(id, drive), Driving(id, drive, None)
        case (StartDrive(order=order), Session(), Drive(sends=sends) as replaced):
            # A new order for a driven session goes on from the sends the one it replaces made, so the cap bounds the loop
            # however often the order is given again; the count starts over only once a drive has ended.
            drive = Drive(order, sends)
            return registry.drive(id, drive), Driving(id, drive, replaced)
        case (DriveSending(text=text, pane=pane), Session() as session, Drive() as drive):
            match prompter(session, pane):
                case Unwrapped() | AtItsDialog() as unreached:
                    return registry, unreached
                case Fritter() | Pane() as by:
                    sent = Drive(drive.order, drive.sends + 1)
                    typed = Type(id, by, Text(text))
                    # [LAW:dataflow-not-control-flow] the count decides which send this is, never a flag set beside it.
                    match sent.sends < SENDS:
                        case True:
                            return registry.drive(id, sent), Send(typed, sent)
                        case False:
                            return registry.undrive(id), LastSend(typed, sent)
