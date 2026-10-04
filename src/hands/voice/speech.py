"""What sessions say to the user unasked: announcements spoken as written, moments the intermediary explains."""

import json
from itertools import pairwise
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePath

from pipecat.frames.frames import DataFrame, Frame, LLMAssistantPushAggregationFrame, LLMMessagesAppendFrame, TTSSpeakFrame, UninterruptibleFrame

from hands.core.attention import Amount, Attention, Overlay, Route, progress_route
from hands.core.effects import Allow, Announcement, Answers, Approve, Asking, DeadlineNear, Decision, Deny, Expired, Heard, KeepPlanning, ModeAfterPlan, ModeChanged, Narrate, Note, Progress, SessionGone, Speak
from hands.core.pending import Briefing, Finished, News, Pending, Unread, Working, went_on
from hands.core.progress import lowered, said
from hands.core.permissions import Answered, NotWaiting, Outcome, Unfit
from hands.core.session import AskedQuestion, Blocker, Permission, Plan, PromptId, Question, SessionId
from hands.core.turn import AgentTask
from hands.sessions.registry import Sessions
from hands.voice.readback import spoken_mode
from hands.voice.utterance import Utterance, Utterances, uttering

# How much of what a session replied is handed to the model in one telling, however many turns it folds. Every telling
# grows the model's history toward compaction, so what is handed is bounded; a session asked to end on a concise
# overview writes far less than this.
REPLY_SHOWN = 1500

# A tool input is shown to the model whole up to this many characters; a longer one is cut and says so.
_INPUT_SHOWN = 800

Names = Callable[[SessionId], str]

# How much of a finished turn the model is asked to say, as the user set it.
_HOW_MUCH: Mapping[Amount, str] = {
    "brief": "Tell the user in a few words which session finished and the one thing it did, naming the session.",
    "full": "Tell the user what it concretely did, in your own words, in one or two spoken sentences, naming the session.",
}

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
    `session` is the one whose turn or question it tells, which is told to the user as the brain takes it, and
    `utterances` what it says of the sessions, which the brain's stage sends what it says of them with.
    """

    text: str
    unsaid: str
    session: SessionId
    utterances: tuple[Utterance, ...]


@dataclass
class Told(DataFrame):
    """An API model has said hands' telling of a session's turn or its question: what the user says next is taken first
    as said to that session. Passed on in order behind the telling, so words the user spoke before it reach the session
    they were meant for, and dropped with it by a barge-in that comes before the model has said it, so a telling the
    model never said moves nothing. Like the brain's stage, it moves the focus as the model says the telling, not as it
    is played: a barge-in on the playing stops what is heard, not the move."""

    session: SessionId


@dataclass
class Unprompted(DataFrame, UninterruptibleFrame):
    """Something hands has to tell of the sessions, on its way to the floor, which makes the frames that tell it as it
    lets it go. Kept through a barge-in: none of it has started to play. `utterances` is what hands heard that it tells,
    none for what hands has to tell of its own."""

    pending: Pending
    utterances: tuple[Utterance, ...]


@dataclass
class Aloud(DataFrame, UninterruptibleFrame):
    """A line hands says as written, held in hands' lane behind what it has already handed the brain, so the user
    hears a session's story in the order it happened, and sent with the frames that say what of `utterances` was heard."""

    spoken: TTSSpeakFrame
    utterances: tuple[Utterance, ...]


def as_a_turn(spoken: TTSSpeakFrame) -> tuple[Frame, Frame]:
    """A line hands says as written with no reply of the model's under way, as a turn of its own: over once it is said.

    [LAW:no-ambient-temporal-coupling] the end of the turn is sent behind the line by whoever knows no reply is under
    way. Pipecat's service ends the turn of a line kept in the context itself only while it takes no reply as under way,
    and one that pushes its own text frames, as pocket-tts does, takes a reply as under way from its start frame to the
    next barge-in (1.10.0): the line stayed an open turn, written at the next key press as cut off. A line kept out of
    the context opens no turn, and the end sent behind it ends none.
    """
    return (spoken, LLMAssistantPushAggregationFrame())


@dataclass(frozen=True)
class AsWritten:
    """A line hands says as written."""

    spoken: TTSSpeakFrame


@dataclass(frozen=True)
class InOwnWords:
    """A message from hands, telling of `session`, for the model to say in its own words; `unsaid` is what hands says as
    written where the brain cannot take it."""

    text: str
    unsaid: str
    session: SessionId


@dataclass(frozen=True)
class Known:
    """What the model is to know and never say: notes for its context, none where the tail of its requests tells it."""

    notes: tuple[LLMMessagesAppendFrame, ...]


# How something hands tells is said. [LAW:one-type-per-behavior] the frames each makes depend on the model's telling
# alone, and `sent` is the one place that decides them.
Saying = AsWritten | InOwnWords | Known


def sent(saying: Saying, telling: Telling, utterances: tuple[Utterance, ...]) -> Sequence[Frame]:
    """The frames that say `saying` under the model's telling, sent with what tells which of `utterances` was heard;
    what is never said settles them `silent` here."""
    match saying, telling:
        case AsWritten(spoken=spoken), Pushed():
            return uttering(utterances, (spoken,))
        case AsWritten(spoken=spoken), Tailed():
            # In hands' lane, in order with what it hands the brain: the brain's stage sends it with what tells it was heard.
            return (Aloud(spoken, utterances),)
        case InOwnWords(text=text, session=session), Pushed():
            # An API model's stage answers the context it is handed before it takes the next frame, so Told follows the
            # telling as said, and words the user speaks while it is said wait behind it.
            return uttering(utterances, (LLMMessagesAppendFrame([{"role": "user", "content": text}], run_llm=True), Told(session)))
        case InOwnWords(text=text, unsaid=unsaid, session=session), Tailed():
            # The brain's stage puts the user's words ahead of hands', so it moves the focus itself as it takes the telling.
            return (Narrated(text, unsaid, session, utterances),)
        case Known(notes=notes), _:
            # Never said, so never read off the speaker: a brain turn speaking beside it would lend it its audio.
            for utterance in utterances:
                utterance.settle("silent")
            return notes


def bounded(text: str, limit: int) -> str:
    """The text whole up to `limit` characters, and cut there, saying so, past it."""
    return text if len(text) <= limit else f"{text[:limit]}... (cut short)"


# How a session is attended to as its progress is relayed: what hands is set to say unprompted, whether the session is
# the focus, and its overlay, each read as it is.
Attending = Callable[[SessionId], Awaitable[tuple[Attention, bool, Overlay]]]


async def relay(
    sessions: Sessions,
    utterances: Utterances,
    queue_frame: Callable[[Frame], Awaitable[None]],
    attending: Attending,
    play: Callable[[Progress, Amount, Utterance], None],
    working: Callable[[], None],
) -> None:
    """Hand what the sessions say to the floor, in the order it was decided, until cancelled; progress to be played is
    handed to `play`, since text in it waits on a summary, and what a session asks never waits behind that. Progress that
    is heard is told to `working` as it comes, ahead of its summary: progress only noted makes no sound."""
    while True:
        heard = await sessions.heard()
        utterance = utterances.heard(_teller(heard), heard)
        match heard:
            case Progress(session=session):
                attention, focused, overlay = await attending(session)
                route = progress_route(attention, focused, overlay)
                # [LAW:nothing-unseen] which way progress went, and what decided it.
                utterance.annotate(attention=attention, focused=focused, overlay=overlay, route=route)
                _routed(route, heard, utterance, play, working)
            case Speak() | Narrate() | Note():
                await queue_frame(Unprompted(heard, (utterance,)))


def _teller(heard: Heard) -> SessionId:
    match heard:
        case Speak(announcement=DeadlineNear(session=session) | Expired(session=session)) | Narrate(moment=Asking(session=session)) | Note(fact=ModeChanged(session=session)) | Progress(session=session):
            return session


def _routed(route: Route, progress: Progress, utterance: Utterance, play: Callable[[Progress, Amount, Utterance], None], working: Callable[[], None]) -> None:
    match route:
        case "brief" | "full":
            working()
            play(progress, route, utterance)
        case "note":
            # [LAW:one-source-of-truth] the session listing says what a working session last set out to do, for either
            # model to read when asked, so nothing is added to a context that keeps every message it is given.
            utterance.settle("noted")


def frames(pending: Pending, telling: Telling, names: Names) -> Saying:
    # [LAW:one-type-per-behavior] the route is the pending thing's own variant: Speak needs no model, Narrate needs one to explain.
    match pending, telling:
        case Speak(announcement=announcement), _:
            # Kept in the context, so the intermediary knows what the user has already been told. In hands' lane under the
            # brain, so a deadline is heard after the question it counts down, never ahead of it.
            return AsWritten(TTSSpeakFrame(announcement_text(announcement, names)))
        case Narrate(moment=moment), _:
            return InOwnWords(narration(moment, names), f"{names(moment.session)} is waiting on you about {_what(moment.on)}.", moment.session)
        case Note(fact=fact), Pushed():
            return Known((LLMMessagesAppendFrame([{"role": "user", "content": noted(fact, names)}], run_llm=False),))
        case Note(), Tailed():
            # [LAW:one-source-of-truth] the tail of the brain's next request says how the session stands now.
            return Known(())
        case Finished(session=session, news=news, amount=amount), _:
            name = names(session)
            return InOwnWords(told(session, name, news, amount), f"{name} finished {_turns(news)}, and I could not tell it.", session)
        case Unread(session=session), _:
            return AsWritten(TTSSpeakFrame(f"{names(session)} finished a turn, and I could not read it.", append_to_context=False))
        case SessionGone(session=session), _:
            return AsWritten(TTSSpeakFrame(f"The session {names(session)} is gone."))
        case Briefing(note=note), _:
            return Known((LLMMessagesAppendFrame([{"role": "user", "content": note}], run_llm=False),))
        case Working(session=session, of=of, doings=doings), _:
            # Said as written: what it is doing is arithmetic over its calls, with nothing for a model to add. Kept out of
            # a pushed context, which keeps every message it is given and would take one every few seconds a session
            # works: the session listing says what it last set out to do [LAW:one-source-of-truth].
            return AsWritten(TTSSpeakFrame(f"{_doer(names(session), of)}: {said(doings)}.", append_to_context=False))


def _doer(name: str, of: frozenset[PromptId] | AgentTask) -> str:
    """Who did what progress tells: the session, or its subagent, named by the job the call that started it gave it."""
    match of:
        case AgentTask(description=description):
            return f"{name}, its subagent to {lowered(description)}"
        case frozenset():
            return name


def told(session: SessionId, name: str, news: Sequence[News], amount: Amount) -> str:
    """Finished turns as the model is handed them: the last thing the session said in each, what hands read of each
    that those words may not say, what it is waiting on, which the model ends by asking, and how much of it to say.

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
        f"{_HOW_MUCH[amount]} {ending}"
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


def brain_asks(on: Permission) -> str:
    """What hands says for a permission the brain's own setup asks about: the command it would run or the address it would
    fetch, or the tool and the file it would touch by name and never by path, and that a yes allows it."""
    match on.input:
        # Said whole, however long: a yes allows exactly this, so the user hears all of it.
        case {"command": str() as command}:
            what = f"{on.tool} to run {command}"
        case {"url": str() as url}:
            what = f"{on.tool} on {url}"
        case {"file_path": str() as path} | {"notebook_path": str() as path}:
            what = f"{on.tool} on {PurePath(path).name}"
        case _:
            what = on.tool
    return f"May I use {what}? Say yes to allow it."


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
