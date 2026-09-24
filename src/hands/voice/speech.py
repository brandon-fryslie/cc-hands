"""What sessions say to the user unasked: announcements spoken as written, moments the intermediary explains."""

import json
from collections.abc import Awaitable, Callable

from pipecat.frames.frames import Frame, LLMMessagesAppendFrame, TTSSpeakFrame

from hands.core.effects import Allow, Announcement, Answers, Asking, DeadlineNear, Decision, Deny, Expired, Heard, Narrate, Speak, WaitingForYou
from hands.core.permissions import Answered, NotWaiting, Outcome, Unfit
from hands.core.session import AskedQuestion, Blocker, Permission, Question, SessionId
from hands.sessions.registry import Sessions
from hands.voice.readback import spoken_name

# A tool input is shown to the model whole up to this many characters; a longer one is cut and says so.
_INPUT_SHOWN = 800

Names = Callable[[SessionId], str]


async def relay(sessions: Sessions, queue_frame: Callable[[Frame], Awaitable[None]]) -> None:
    """Hand everything the sessions say to the pipeline, in the order it was decided, until cancelled."""
    while True:
        heard = await sessions.heard()
        await queue_frame(frame(heard, lambda id: spoken_name(sessions, id)))


def frame(heard: Heard, names: Names) -> Frame:
    # [LAW:one-type-per-behavior] the route is the effect's own variant: Speak needs no model, Narrate needs one to explain.
    match heard:
        case Speak(announcement=announcement):
            # Kept in the context, so the intermediary knows what the user has already been told.
            return TTSSpeakFrame(announcement_text(announcement, names))
        case Narrate(moment=moment):
            return LLMMessagesAppendFrame([{"role": "user", "content": narration(moment, names)}], run_llm=True)


def narration(moment: Asking, names: Names) -> str:
    return f"[hands] The Claude Code session {names(moment.session)} {_asks(moment.on)} Request id: {moment.request}. {_ask_user(moment.on)}"


def _asks(on: Blocker) -> str:
    match on:
        case Permission(tool=tool, input=input):
            shown = json.dumps(dict(input), ensure_ascii=False)
            shown = shown if len(shown) <= _INPUT_SHOWN else f"{shown[:_INPUT_SHOWN]}... (cut short)"
            return f"is waiting for permission to use {tool} with this input: {shown}."
        case Question(asked=asked):
            # Shown whole, however long: a question cut short cannot be answered.
            return "is asking the user:\n" + "\n".join(f"{number}. {_question(question)}" for number, question in enumerate(asked, 1))


def _question(question: AskedQuestion) -> str:
    several = " More than one may be chosen." if question.several else ""
    options = "; ".join(option.label if option.description is None else f"{option.label} ({option.description})" for option in question.options)
    return f"{question.question} Options: {options}.{several}" if options else f"{question.question} Answered in the user's own words."


def _ask_user(on: Blocker) -> str:
    match on:
        case Permission():
            return (
                "Tell the user in one short sentence what it wants to do, and ask whether to allow it. "
                "When they decide, call answer_permission with that request id."
            )
        case Question():
            return (
                "Put the questions to the user as a person would, with their options, one at a time. "
                "When they have answered all of them, call answer_question with that request id."
            )


def announcement_text(announcement: Announcement, names: Names) -> str:
    match announcement:
        case DeadlineNear(session=session, on=on, remaining=remaining):
            return f"{round(remaining)} seconds left to answer {names(session)} {_about(on)}."
        case Expired(session=session, on=on):
            # Said as what hands did: an answer typed at the dialog meanwhile would already have settled it.
            return f"Nobody answered {names(session)} {_about(on)} in time, so I told it no."
        case WaitingForYou(session=session):
            return f"{names(session)} is waiting for you."


def _about(on: Blocker) -> str:
    match on:
        case Permission(tool=tool):
            return f"about {tool}"
        case Question():
            return "about its question"


def answer_readback(outcome: Outcome, names: Names) -> str:
    match outcome:
        case Answered(session=session, on=on, decision=decision):
            return f"{_done(on, decision)} for {names(session)}."
        case NotWaiting():
            return "That request is no longer waiting: it was already answered, answered at the keyboard, or denied at its deadline."
        case Unfit(on=on, decision=decision):
            return _unfit(on, decision)


def _done(on: Blocker, decision: Decision) -> str:
    match (on, decision):
        case (_, Deny()):
            return f"Denied {_what(on)}"
        case (_, Allow()):
            return f"Allowed {_what(on)}"
        case (_, Answers(chosen=chosen)):
            return f"Answered {'; '.join(chosen)}"


def _what(on: Blocker) -> str:
    match on:
        case Permission(tool=tool):
            return tool
        case Question():
            return "its question"


def _unfit(on: Blocker, decision: Decision) -> str:
    # Handed to the model, which is what called the wrong tool or miscounted; the session still waits.
    match (on, decision):
        case (Question(asked=asked), Answers(chosen=chosen)):
            return f"Nothing was sent: it asked {len(asked)} questions and was given {len(chosen)} answers. Give one answer for each, in the order they were asked."
        case (Question(), _):
            return "Nothing was sent: that request is a question. Answer it with answer_question, or deny it with answer_permission."
        case (Permission(), _):
            return "Nothing was sent: that request is a permission, not a question. Answer it with answer_permission."
