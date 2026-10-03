"""What the user decides about a session waiting on them, what came of it, and the one function that decides."""

import re
from dataclasses import dataclass, replace

from hands.core.effects import Allow, AllowWith, Answers, Approve, Decision, Deny, Effect, HookReply, KeepPlanning, Reply
from hands.core.session import Blocker, Held, Known, Permission, Plan, Question, Registry, RequestId, Session, SessionId


@dataclass(frozen=True)
class Answer:
    request: RequestId
    decision: Decision


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
        case [Session(dialog=Held(on=on)) as session]:
            match _reply(on, request.decision):
                case None:
                    return registry, Unfit(request.request, on, request.decision), []
                case reply:
                    # The turn carries on from here, with the tool run or refused.
                    answered = registry.put(replace(session, dialog=None))
                    return answered, Answered(session.membership.id, on, request.decision), [Reply(session.membership.id, request.request, reply)]
        case _:
            return registry, NotWaiting(request.request), []


def _reply(on: Blocker, decision: Decision) -> HookReply | None:
    """What the hook is sent when this decision answers what the session asked; None when it does not answer it."""
    # [LAW:types-are-the-program] each blocker takes the decisions that answer it, and a refusal answers either.
    match (on, decision):
        case (_, Deny()):
            return decision
        case (Permission(), Allow()) | (Plan(), Approve()):
            return decision
        case (Plan(), KeepPlanning(message=message)):
            # Refused, so ExitPlanMode never runs and the session stays in plan mode.
            return Deny(message)
        case (Question(asked=asked, input=input), Answers(chosen=chosen)) if len(chosen) == len(asked):
            # The shape Claude Code's own dialog answers with: each question's text keys the label chosen for it.
            return AllowWith({**input, "answers": {question.question: answer for question, answer in zip(asked, chosen)}})
        case _:
            return None


def _waits_on(session: Known, request: RequestId) -> bool:
    match session:
        case Session(dialog=Held(request=held)):
            return held == request
        case _:
            return False


# [LAW:types-are-the-program] a yes is said only in these words, and at least one of them says it: anything else the user
# says to a permission request, however it opens, is theirs to be read in full by what asked, and runs nothing.
_ASSENT = frozenset({"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "alright", "right", "fine", "good", "allow", "go", "ahead", "do"})
_ASSENTING = _ASSENT | {"it", "please", "for", "that's", "all", "sounds"}


def heard(words: str) -> Allow | Deny:
    """What a spoken answer to a permission request decides: an Allow for a plain yes, and for anything else a Deny that
    carries the user's words, so what asked hears what they said instead."""
    # Every word counts, digits and all, and an apostrophe is one however the transcript types it: a word outside the yes
    # words keeps the answer theirs to read.
    said = re.findall(r"[\w']+", words.lower().replace("’", "'"))
    if said and set(said) <= _ASSENTING and _ASSENT & set(said):
        return Allow()
    return Deny(f'The user was asked whether to allow this, and answered by voice, so it did not run. Do what they said, which was: "{words}"')
