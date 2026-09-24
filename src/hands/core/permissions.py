"""What the user decides about a session waiting on them, what came of it, and the one function that decides."""

from dataclasses import dataclass

from hands.core.effects import Allow, AllowWith, Answers, Decision, Deny, Effect, HookReply, Reply
from hands.core.session import Blocked, Blocker, Instant, Permission, Question, Registry, RequestId, Session, SessionId, Working


@dataclass(frozen=True)
class Answer:
    request: RequestId
    decision: Decision
    at: Instant


@dataclass(frozen=True)
class Answered:
    session: SessionId
    on: Blocker
    decision: Decision


@dataclass(frozen=True)
class NotWaiting:
    """No session waits on this request: it was answered, withdrawn, or denied at its deadline."""

    request: RequestId


@dataclass(frozen=True)
class Unfit:
    """The decision does not answer what the session asked, so nothing was sent and the session still waits."""

    request: RequestId
    on: Blocker
    decision: Decision


Outcome = Answered | NotWaiting | Unfit


def answer(registry: Registry, request: Answer) -> tuple[Registry, Outcome, list[Effect]]:
    """One answer in; the next registry, what came of it, and the reply it calls for out. No I/O."""
    waiting = [session for session in registry.sessions.values() if _waits_on(session, request.request)]
    match waiting:
        case [Session(membership=membership, state=Blocked(on=on))]:
            match _reply(on, request.decision):
                case None:
                    return registry, Unfit(request.request, on, request.decision), []
                case reply:
                    # The turn carries on from here, with the tool run or refused.
                    answered = registry.put(Session(membership, Working(since=request.at)))
                    return answered, Answered(membership.id, on, request.decision), [Reply(membership.id, request.request, reply)]
        case _:
            return registry, NotWaiting(request.request), []


def _reply(on: Blocker, decision: Decision) -> HookReply | None:
    """What the hook is sent when this decision answers what the session asked; None when it does not answer it."""
    # [LAW:types-are-the-program] each blocker takes the decisions that answer it, and a refusal answers either.
    match (on, decision):
        case (_, Deny()):
            return decision
        case (Permission(), Allow()):
            return decision
        case (Question(asked=asked, input=input), Answers(chosen=chosen)) if len(chosen) == len(asked):
            # The shape Claude Code's own dialog answers with: each question's text keys the label chosen for it.
            return AllowWith({**input, "answers": {question.question: answer for question, answer in zip(asked, chosen)}})
        case _:
            return None


def _waits_on(session: Session, request: RequestId) -> bool:
    match session.state:
        case Blocked(request=held):
            return held == request
        case _:
            return False
