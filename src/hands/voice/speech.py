"""What sessions say to the user unasked: announcements spoken as written, moments the intermediary explains."""

import json
from itertools import pairwise
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePath

from pipecat.frames.frames import DataFrame, Frame, TTSSpeakFrame, UninterruptibleFrame

from hands.core.attention import Amount, Attention, Overlay, Route, Spoken, Steering, Withheld, progress_route
from hands.core.effects import Allow, Announcement, Answers, Approve, Asking, DeadlineNear, Decision, Deny, Expired, Heard, KeepPlanning, ModeAfterPlan, Narrate, Progress, SessionGone, Speak, Tell, WentOn
from hands.core.occurrences import route as occurrence_route, said as occurrence_said
from hands.core.pending import Finished, Mentioned, News, Pending, Unread, Working, steered, went_on
from hands.core.progress import lowered, said
from hands.core.permissions import Answered, NotWaiting, Outcome, Unfit
from hands.core.drive import SENDS
from hands.core.session import AskedQuestion, Drive, Blocker, Permission, Plan, PromptId, Question, SessionId
from hands.core.sentences import bounded
from hands.core.turn import AgentTask
from hands.sessions.registry import Sessions
from hands.voice.utterance import Utterance, Utterances

# How much of what a session replied is handed to the model in one telling, however many turns it folds. Every telling
# grows the model's history toward compaction, so what is handed is bounded; a session asked to end on a concise
# overview writes far less than this.
REPLY_SHOWN = 1500

# A tool input is shown to the model whole up to this many characters; a longer one is cut and says so.
_INPUT_SHOWN = 800
# A refused command is named by this many of its opening words.
_OPENING_WORDS = 6

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


@dataclass
class Narrated(DataFrame, UninterruptibleFrame):
    """A message from hands the brain takes as a turn of its own, once no turn of the user's is waiting.

    Never put in Pipecat's context, where it would be one message with whatever the user said beside it: the brain keeps
    its own history. Kept through a barge-in, which stops what is said, not what is still to be told. `unsaid` is what
    hands says as written if the brain cannot take the turn: that it could not be told, never the turn's own words.
    `session` is the one whose turn or question it tells, which is told to the user as the brain takes it, and
    `utterances` what it says of the sessions, which the brain's stage sends what it says of them with. `drive` is the
    standing order a turn is handed to the brain under, to act on rather than tell, so the user's focus stays where it is.
    """

    text: str
    unsaid: str
    session: SessionId
    utterances: tuple[Utterance, ...]
    drive: Drive | None


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
    drive: Drive | None


# How something hands tells is said. [LAW:one-type-per-behavior] `sent` is the one place that decides the frames each makes.
Saying = AsWritten | InOwnWords


def sent(saying: Saying, utterances: tuple[Utterance, ...]) -> Sequence[Frame]:
    """The frames that say `saying`, sent with what tells which of `utterances` was heard."""
    match saying:
        case AsWritten(spoken=spoken):
            # In hands' lane, in order with what it hands the brain: the brain's stage sends it with what tells it was heard.
            return (Aloud(spoken, utterances),)
        case InOwnWords(text=text, unsaid=unsaid, session=session, drive=drive):
            # The brain's stage puts the user's words ahead of hands', so it moves the focus itself as it takes the telling.
            return (Narrated(text, unsaid, session, utterances, drive),)


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
            case Tell(session=session, occurrence=occurrence):
                attention, _, overlay = await attending(session)
                route = occurrence_route(attention, overlay, occurrence)
                # [LAW:nothing-unseen] whether it was said, how much of it, and what was set that decided it.
                utterance.annotate(attention=attention, overlay=overlay, route=route)
                await _mentioned(route, Tell(session, occurrence), utterance, queue_frame)
            case Speak() | Narrate():
                await queue_frame(Unprompted(heard, (utterance,)))


def _teller(heard: Heard) -> SessionId:
    match heard:
        case Speak(announcement=DeadlineNear(session=session) | Expired(session=session) | WentOn(session=session)) | Narrate(moment=Asking(session=session)) | Progress(session=session) | Tell(session=session):
            return session


async def _mentioned(route: Route, told: Tell, utterance: Utterance, queue_frame: Callable[[Frame], Awaitable[None]]) -> None:
    match route:
        case "brief" | "full":
            await queue_frame(Unprompted(Mentioned(told.session, told.occurrence, route), (utterance,)))
        case "note":
            # [LAW:one-source-of-truth] the utterance's event in the audit log holds it, for catch_up and the log to tell.
            utterance.settle("noted")


def _routed(route: Route, progress: Progress, utterance: Utterance, play: Callable[[Progress, Amount, Utterance], None], working: Callable[[], None]) -> None:
    match route:
        case "brief" | "full":
            working()
            play(progress, route, utterance)
        case "note":
            # [LAW:one-source-of-truth] the session listing says what a working session last set out to do, for the brain
            # to read when asked.
            utterance.settle("noted")


def frames(pending: Pending, names: Names) -> Saying:
    # [LAW:one-type-per-behavior] the route is the pending thing's own variant: Speak needs no model, Narrate needs one to explain.
    match pending:
        case Speak(announcement=announcement):
            # In hands' lane, so a deadline is heard after the question it counts down, never ahead of it.
            return AsWritten(TTSSpeakFrame(announcement_text(announcement, names)))
        case Narrate(moment=moment):
            return InOwnWords(narration(moment, names), f"{names(moment.session)} is waiting on you about {_what(moment.on)}.", moment.session, None)
        case Finished(session=session, news=news, telling=telling):
            name = names(session)
            acting = _acting(telling, news[-1].asked)
            return InOwnWords(told(session, name, news, telling), _untold(name, news, acting), session, acting)
        case Unread(session=session, stopped=stopped):
            ended = "" if stopped is None else ", so I stopped driving it"
            return AsWritten(TTSSpeakFrame(f"{names(session)} finished a turn, and I could not read it{ended}.", append_to_context=False))
        case SessionGone(session=session):
            return AsWritten(TTSSpeakFrame(f"The session {names(session)} is gone."))
        case Mentioned(session=session, occurrence=occurrence, amount=amount):
            # Said as written: what the hook said, worded here, with nothing for a model to add.
            return AsWritten(TTSSpeakFrame(occurrence_said(occurrence, names(session), amount)))
        case Working(session=session, of=of, doings=doings):
            # Said as written: what it is doing is arithmetic over its calls, with nothing for a model to add; the session
            # listing says what it last set out to do [LAW:one-source-of-truth].
            return AsWritten(TTSSpeakFrame(f"{_doer(names(session), of)}: {said(doings)}.", append_to_context=False))


def _doer(name: str, of: frozenset[PromptId] | AgentTask) -> str:
    """Who did what progress tells: the session, or its subagent, named by the job the call that started it gave it."""
    match of:
        case AgentTask(description=description):
            return f"{name}, its subagent to {lowered(description)}"
        case frozenset():
            return name


def _acting(telling: Amount | Steering, asked: str) -> Drive | None:
    """The drive the brain acts on the turn under; None when it tells the user, as it does a driven session's question,
    which is theirs to answer and so moves their focus to the session that asked it."""
    match steered(telling), asked:
        case Drive() as drive, "":
            return drive
        case _:
            return None


def _untold(name: str, news: Sequence[News], acting: Drive | None) -> str:
    """What hands says as written when the brain cannot take the turn; under a drive, that the drive waits on the user."""
    match acting:
        case None:
            return f"{name} finished {_turns(news)}, and I could not tell it."
        case Drive():
            return f"{name} finished {_turns(news)} while I was driving it, and I could not act on it: tell me to go on, or to stop driving it."


def told(session: SessionId, name: str, news: Sequence[News], telling: Amount | Steering) -> str:
    """Finished turns as the model is handed them: the last thing the session said in each, what hands read of each
    that those words may not say, what it is waiting on, and what the model is to do with them.

    Only the last turn's question is put: a session that went on to another turn was answered, at the keyboard or
    by the turn that followed.
    """
    # [LAW:single-enforcer] one bound for the whole telling, shared among its turns, however many were folded into it.
    shown = REPLY_SHOWN // len(news)
    accounts = _account(news[0], shown) + "".join(f"{_then(before, each)}{_account(each, shown)}" for before, each in pairwise(news))
    return f"[hands] The Claude Code session {name} (id {session}) finished {_turns(news)}. {accounts}{_how(telling, news[-1].asked, name, session)}"


def _how(telling: Amount | Steering, asked: str, name: str, session: SessionId) -> str:
    """What the model is to do with the turns: say as much of them as is set, ending on what the session asks the user;
    or, under a drive, send the next prompt, unless the session asks the user something, which is theirs to answer."""
    ask = f"end by asking it, with what it refers to, so they can answer without looking at the screen: {asked}"
    match telling, asked:
        case ("brief" | "full") as amount, "":
            return f"{_HOW_MUCH[amount]} It asks the user nothing."
        case ("brief" | "full") as amount, _:
            return f"{_HOW_MUCH[amount]} It is waiting on the user's answer to this, so {ask}"
        case Steering(drive=Drive(order=order, sends=sends), ear=ear), "":
            return (
                f"You are driving {name} under the user's standing order: \"{order}\". You have sent it {sends} of the "
                f"{SENDS} prompts the order allows. Act on it now: call drive_send with session {session} and the next prompt "
                "that moves it toward the order, or call stop_driving when the order is met, or when it needs the user's "
                f"decision or is going wrong. {_steered(ear, name)}"
            )
        case Steering(drive=Drive(order=order)), _:
            # [LAW:single-enforcer] what a driven session asks the user stays theirs: the drive stops here, never answered by a send.
            return (
                f"You are driving {name} under the user's standing order: \"{order}\", and it is waiting on the user's "
                f"answer to a question, which is theirs to give. Call stop_driving with session {session}, then tell the "
                f"user you stopped driving it, and {ask}"
            )


def _steered(ear: Spoken | Withheld, name: str) -> str:
    """What the user hears of a driven turn: a sentence of what was done, unless they set it to be held."""
    match ear:
        case Spoken():
            return f"Then tell the user in one short sentence what you did, naming {name}."
        case Withheld():
            return "Then say nothing to the user: they set this session's finished turns to be held."


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
    """What hands says for a permission the brain's own setup asks about, and that a yes allows it."""
    # The command said whole, however long: a yes allows exactly this, so the user hears all of it.
    return f"May I use {_use(on, lambda command: command)}? Say yes to allow it."


def brain_refused(on: Permission) -> str:
    """What hands says for a permission the user was asked and refused, when the brain said nothing after it: without it,
    silence would pass for the work done."""
    # The command named by its opening words: the user heard it whole when asked, and this only says which it was.
    return f"I did not use {_use(on, _opening)}, so that is not done."


def _opening(command: str) -> str:
    words = command.split()
    return " ".join(words[:_OPENING_WORDS]) + (" and so on" if len(words) > _OPENING_WORDS else "")


def _use(on: Permission, saying: Callable[[str], str]) -> str:
    """The use a permission is for: the command it would run, as `saying` says it, or the address it would fetch, or the
    tool and the file it would touch by name and never by path."""
    match on.input:
        case {"command": str() as command}:
            return f"{on.tool} to run {saying(command)}"
        case {"url": str() as url}:
            return f"{on.tool} on {url}"
        case {"file_path": str() as path} | {"notebook_path": str() as path}:
            return f"{on.tool} on {PurePath(path).name}"
        case _:
            return on.tool


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
        case WentOn(session=session, on=on):
            return f"Nobody answered {names(session)} about {_what(on)} in time, so it went on without an answer."


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
