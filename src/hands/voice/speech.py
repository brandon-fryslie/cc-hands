"""What sessions say to the user unasked: announcements spoken as written, moments the intermediary explains."""

import json
from itertools import pairwise
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

from pipecat.frames.frames import DataFrame, Frame, LLMMessagesAppendFrame, TTSSpeakFrame, UninterruptibleFrame

from hands.core.attention import Overlay, Route, progress_route
from hands.core.effects import Allow, Announcement, Answers, Approve, Asking, DeadlineNear, Decision, Deny, Expired, Heard, KeepPlanning, ModeAfterPlan, ModeChanged, Narrate, Note, Progress, SessionGone, Speak
from hands.core.pending import Briefing, Finished, News, Pending, Unread, Working, went_on
from hands.core.progress import Doing, said
from hands.core.permissions import Answered, NotWaiting, Outcome, Unfit
from hands.core.session import AskedQuestion, Blocker, Permission, Plan, PromptId, Question, SessionId
from hands.sessions.audit import Record, Relayed, Routed
from hands.sessions.registry import Sessions
from hands.voice.readback import spoken_mode

# How much of what a session replied is handed to the model in one telling, however many turns it folds. Every telling
# grows the model's history toward compaction, so what is handed is bounded; a session asked to end on a concise
# overview writes far less than this.
REPLY_SHOWN = 1500

# A tool input is shown to the model whole up to this many characters; a longer one is cut and says so.
_INPUT_SHOWN = 800

Names = Callable[[SessionId], str]

# How an approved plan goes on, in the words of the choice that approved it.
_AFTER_PLAN: Mapping[ModeAfterPlan, str] = {
    "resume": ", in the mode it had before planning",
    "acceptEdits": ", with its edits accepted automatically",
    "default": ", asking you about each edit",
}


@dataclass(frozen=True)
class Pushed:
    """The model is told how the sessions stand by notes in its context: one when hands starts, and one at each change."""


@dataclass(frozen=True)
class Tailed:
    """The model reads how the sessions stand at the tail of each request its turn makes, so no note goes into its context."""


Telling = Pushed | Tailed


@dataclass
class Narrated(DataFrame, UninterruptibleFrame):
    """A message from hands the brain takes as a turn of its own, once no turn of the user's is waiting.

    Never put in Pipecat's context, where it would be one message with whatever the user said beside it: the brain keeps
    its own history. Kept through a barge-in, which stops what is said, not what is still to be told. `unsaid` is what
    hands says as written if the brain cannot take the turn: that it could not be told, never the turn's own words.
    """

    text: str
    unsaid: str


@dataclass
class Unprompted(DataFrame, UninterruptibleFrame):
    """Something hands has to tell of the sessions, on its way to the floor, which makes the frames that tell it as it
    lets it go. Kept through a barge-in: none of it has started to play."""

    pending: Pending


@dataclass
class Aloud(DataFrame, UninterruptibleFrame):
    """A line hands says as written, held in hands' lane behind what it has already handed the brain, so the user
    hears a session's story in the order it happened."""

    spoken: TTSSpeakFrame


def handed(text: str, unsaid: str, telling: Telling) -> Frame:
    """A message from hands for the model to say in its own words, as the model's telling takes one."""
    match telling:
        case Pushed():
            return LLMMessagesAppendFrame([{"role": "user", "content": text}], run_llm=True)
        case Tailed():
            return Narrated(text, unsaid)


def as_written(spoken: TTSSpeakFrame, telling: Telling) -> Frame:
    """A line hands says as written, in order with what it hands the model: a model in the pipeline takes frames one at
    a time, and the brain's stage holds it in hands' lane."""
    match telling:
        case Pushed():
            return spoken
        case Tailed():
            return Aloud(spoken)


def bounded(text: str, limit: int) -> str:
    """The text whole up to `limit` characters, and cut there, saying so, past it."""
    return text if len(text) <= limit else f"{text[:limit]}... (cut short)"


# How a session is attended to as its progress is relayed: whether it is the focus, and its overlay, read as it is.
Attending = Callable[[SessionId], Awaitable[tuple[bool, Overlay]]]


async def relay(sessions: Sessions, queue_frame: Callable[[Frame], Awaitable[None]], record: Record, attending: Attending) -> None:
    """Hand what the sessions say to the floor, in the order it was decided, until cancelled."""
    while True:
        heard = await sessions.heard()
        record(Relayed(heard))
        for pending in await _pending(heard, record, attending):
            await queue_frame(Unprompted(pending))


async def _pending(heard: Heard, record: Record, attending: Attending) -> tuple[Pending, ...]:
    """What the floor is handed of what was heard: progress as the focus and the overlay route it, the rest as it is."""
    match heard:
        case Progress(session=session, turn=turn, doings=doings):
            focused, overlay = await attending(session)
            route = progress_route(focused, overlay)
            # [LAW:nothing-unseen] which way progress went, and what decided it.
            record(Routed(session, focused, overlay, route))
            return _routed(route, session, turn, doings)
        case Speak() | Narrate() | Note():
            return (heard,)


def _routed(route: Route, session: SessionId, turn: frozenset[PromptId], doings: tuple[Doing, ...]) -> tuple[Pending, ...]:
    match route:
        case "play":
            return (Working(session, turn, doings),)
        case "note":
            # [LAW:one-source-of-truth] the session listing says what a working session last set out to do, for either
            # model to read when asked, so nothing is added to a context that keeps every message it is given.
            return ()


def frames(pending: Pending, telling: Telling, names: Names) -> Sequence[Frame]:
    # [LAW:one-type-per-behavior] the route is the pending thing's own variant: Speak needs no model, Narrate needs one to explain.
    match pending, telling:
        case Speak(announcement=announcement), _:
            # Kept in the context, so the intermediary knows what the user has already been told. In hands' lane under the
            # brain, so a deadline is heard after the question it counts down, never ahead of it.
            return (as_written(TTSSpeakFrame(announcement_text(announcement, names)), telling),)
        case Narrate(moment=moment), _:
            return (handed(narration(moment, names), f"{names(moment.session)} is waiting on you about {_what(moment.on)}.", telling),)
        case Note(fact=fact), Pushed():
            return (LLMMessagesAppendFrame([{"role": "user", "content": noted(fact, names)}], run_llm=False),)
        case Note(), Tailed():
            # [LAW:one-source-of-truth] the tail of the brain's next request says how the session stands now.
            return ()
        case Finished(session=session, news=news), _:
            name = names(session)
            return (handed(told(session, name, news), f"{name} finished {_turns(news)}, and I could not tell it.", telling),)
        case Unread(session=session), _:
            return (as_written(TTSSpeakFrame(f"{names(session)} finished a turn, and I could not read it.", append_to_context=False), telling),)
        case SessionGone(session=session), _:
            return (as_written(TTSSpeakFrame(f"The session {names(session)} is gone."), telling),)
        case Briefing(note=note), _:
            return (LLMMessagesAppendFrame([{"role": "user", "content": note}], run_llm=False),)
        case Working(session=session, doings=doings), _:
            # Said as written: what it is doing is arithmetic over its calls, with nothing for a model to add. Kept out of
            # a pushed context, which keeps every message it is given and would take one every few seconds a session
            # works: the session listing says what it last set out to do [LAW:one-source-of-truth].
            return (as_written(TTSSpeakFrame(f"{names(session)}: {said(doings)}.", append_to_context=False), telling),)


def told(session: SessionId, name: str, news: Sequence[News]) -> str:
    """Finished turns as the model is handed them: the last thing the session said in each, what hands read of each
    that those words may not say, and what it is waiting on, which the model ends by asking.

    Only the last turn's question is put: a session that went on to another turn was answered, at the keyboard or
    by the turn that followed.
    """
    # [LAW:single-enforcer] one bound for the whole telling, shared among its turns, however many were folded into it.
    shown = REPLY_SHOWN // len(news)
    accounts = _account(news[0], shown) + "".join(f"{_then(before, each)}{_account(each, shown)}" for before, each in pairwise(news))
    asked = news[-1].asked
    ending = (
        f"It is waiting on the user's answer to this, so end by asking it, with what it refers to, so they can answer without looking at the screen: {asked}"
        if asked
        else "It asks the user nothing."
    )
    return (
        f"[hands] The Claude Code session {name} (id {session}) finished {_turns(news)}. {accounts}"
        f"Tell the user what it concretely did, in your own words, in one or two spoken sentences, naming the session. {ending}"
    )


def _account(news: News, shown: int) -> str:
    reply = f"The last thing it said was:\n\n{bounded(news.reply, shown)}\n\n" if news.reply is not None else "It said nothing. "
    facts = f"From its record, hands adds: {news.facts} " if news.facts else ""
    return f"{reply}{facts}"


def _then(before: News, after: News) -> str:
    return "Then it went on: " if went_on(before, after) else "Then, in the turn after that: "


def _turns(news: Sequence[News]) -> str:
    # A telling that tells more of the turn before it is not another turn.
    count = len(news) - sum(went_on(before, after) for before, after in pairwise(news))
    return "a turn" if count == 1 else f"{count} turns"


def noted(fact: ModeChanged, names: Names) -> str:
    return f"[hands] The Claude Code session {names(fact.session)} is now in {spoken_mode(fact.mode)}. Say nothing about it unless the user asks."


def narration(moment: Asking, names: Names) -> str:
    return f"[hands] The Claude Code session {names(moment.session)} {_asks(moment.on)} Request id: {moment.request}. {_ask_user(moment.on)}"


def _asks(on: Blocker) -> str:
    match on:
        case Permission(tool=tool, input=input):
            shown = bounded(json.dumps(dict(input), ensure_ascii=False), _INPUT_SHOWN)
            return f"is waiting for permission to use {tool} with this input: {shown}."
        case Question(asked=asked):
            # Shown whole, however long: a question cut short cannot be answered.
            return "is asking the user:\n" + "\n".join(f"{number}. {_question(question)}" for number, question in enumerate(asked, 1))
        case Plan(text=text):
            # Shown whole, however long: the model can only summarise what it was given.
            return f"has a plan for the user to approve:\n{text}\n"


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
        case Plan():
            return (
                "Tell the user in a few spoken sentences what the plan would do, not the plan itself, and ask whether to approve it. "
                "When they decide, call answer_plan with that request id."
            )


def announcement_text(announcement: Announcement, names: Names) -> str:
    match announcement:
        case DeadlineNear(session=session, on=on, remaining=remaining):
            return f"{round(remaining)} seconds left to answer {names(session)} about {_what(on)}."
        case Expired(session=session, on=on):
            # Said as what hands did: an answer typed at the dialog meanwhile would already have settled it.
            return f"Nobody answered {names(session)} about {_what(on)} in time, so {_left(on)}."


def answer_readback(outcome: Outcome, names: Names) -> str:
    match outcome:
        case Answered(session=session, on=on, decision=decision):
            return f"{_done(decision, _what(on), names(session))}."
        case NotWaiting():
            return "That request is no longer waiting for a voice answer: it was already answered, answered at the keyboard, or its deadline passed."
        case Unfit(on=on, decision=decision):
            return _unfit(on, decision)


def _done(decision: Decision, what: str, name: str) -> str:
    match decision:
        case Deny():
            return f"Denied {what} for {name}"
        case Allow():
            return f"Allowed {what} for {name}"
        case Answers(chosen=chosen):
            return f"Answered {'; '.join(answer or 'nothing' for answer in chosen)} for {name}"
        case Approve(mode=mode):
            return f"Approved {what} for {name}{_AFTER_PLAN[mode]}"
        case KeepPlanning():
            return f"Sent {what} back to keep planning for {name}"


def _left(on: Blocker) -> str:
    match on:
        case Permission():
            return "I told it no"
        case Question() | Plan():
            return "it is left waiting at its dialog"


def _what(on: Blocker) -> str:
    match on:
        case Permission(tool=tool):
            return tool
        case Question():
            return "its question"
        case Plan():
            return "its plan"


def _unfit(on: Blocker, decision: Decision) -> str:
    # Handed to the model, which is what called the wrong tool or miscounted; the session still waits.
    match (on, decision):
        case (Question(asked=asked), Answers(chosen=chosen)):
            return f"Nothing was sent: it asked {len(asked)} questions and was given {len(chosen)} answers. Give one answer for each, in the order they were asked."
        case (Question(), _):
            return "Nothing was sent: that request is a question. Answer it with answer_question, or deny it with answer_permission."
        case (Plan(), _):
            return "Nothing was sent: that request is a plan. Approve it or send it back to planning with answer_plan."
        case (Permission(), _):
            return "Nothing was sent: that request is a permission. Answer it with answer_permission."
