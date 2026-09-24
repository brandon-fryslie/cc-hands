"""A turn, heard: the headline that plays when it stops, the questions it left open, and one section per topic.

The tree is cut from the turn's own types. `Step` already separates an edit from a test run from a command that
committed, so the sections are a match over that union rather than a table of rules invented beside it, and a
segment's record ids are read off the steps it holds rather than claimed by the model that wrote its text
`[LAW:one-source-of-truth]`. That is what keeps "that part" a lookup: the map is redrawn from the territory on
every build, and a summariser that invents a version number cannot invent a uuid to go with it.

Only the headline's text comes from a model. Every count here is arithmetic over typed steps, so the part of the
narration that carries facts cannot be hallucinated, and the model is asked only for the thing nothing else can
do — saying what a change did.
"""

import re
from dataclasses import dataclass

from hands.core.delta import Delta
from hands.core.spoken import spoken_count, spoken_ref
from hands.core.turn import (
    Branched,
    Budget,
    Committed,
    Delegated,
    Edited,
    Happening,
    GitChange,
    Interruption,
    Looked,
    Other,
    Planned,
    PullRequested,
    Pushed,
    Question,
    Questioned,
    Ran,
    Ref,
    Said,
    Tested,
    Turn,
    body,
)


@dataclass(frozen=True)
class Topic:
    """What a segment is about: what to call it, and the word for one of the things in it.

    [LAW:one-type-per-behavior] the topics differ by two words each and in nothing else, so they are values of
    one type rather than a class apiece, and a new one is a new constant rather than new code.
    """

    name: str
    thing: str


THE_HEADLINE = Topic("the headline", "turn")
A_QUESTION = Topic("a question", "question")
WHAT_IT_SAID = Topic("what it said", "message")
THE_CHANGE = Topic("the change", "edit")
THE_TESTS = Topic("the tests", "test run")
THE_COMMIT = Topic("the commit", "command")
THE_COMMANDS = Topic("the commands", "command")
WHAT_IT_READ = Topic("what it read", "look")
THE_PLAN = Topic("the plan", "task")
THE_SUBAGENTS = Topic("the subagents", "subagent")
THE_OTHER_TOOLS = Topic("the other tools", "tool call")
THE_REPOSITORY = Topic("the repository", "file")
THE_INTERRUPTION = Topic("the interruption", "interruption")

# The steps a section is cut from. A question and an interruption are not among them: each plays at the top level at
# every length, so the type that says which steps fall into sections is also the type that says they never do
# [LAW:types-are-the-program]. Without it, `topic_of` would need an arm for a step it can never be handed.
Sectioned = Said | Edited | Ran | Tested | Looked | Planned | Delegated | Other


def topic_of(step: Sectioned) -> Topic:
    """The section a step belongs to, which is a fact about its kind."""
    match step:
        case Said():
            return WHAT_IT_SAID
        case Edited():
            return THE_CHANGE
        case Ran(git=git):
            # A command that moved the repository is the result the listener came for; one that did not is a command.
            return THE_COMMIT if git else THE_COMMANDS
        case Tested():
            return THE_TESTS
        case Looked():
            return WHAT_IT_READ
        case Planned():
            return THE_PLAN
        case Delegated():
            return THE_SUBAGENTS
        case Other():
            return THE_OTHER_TOOLS


@dataclass(frozen=True)
class Segment:
    """One thing the narration can say, and everything it was made from.

    `text` is what plays. `covers` and `delta` are what it was cut from, kept so a segment can be opened into a
    longer telling without going back to the transcript, and so that `refs` is derived rather than stored: the
    records a segment names are exactly the records it holds, and the two cannot come apart.
    """

    topic: Topic
    text: str
    covers: tuple[Happening, ...] = ()
    delta: Delta = Delta()

    @property
    def refs(self) -> tuple[Ref, ...]:
        """The transcript records this segment summarises, for the lookup behind "the details of that part"."""
        return tuple(happening.ref for happening in self.covers if happening.ref is not None)


def opened(segment: Segment, budget: Budget) -> str:
    """The segment as a summariser's message, for a telling longer than the line it plays.

    The whole turn's own rendering, over the part this segment holds, so what a segment opens into can never be
    described by rules the turn it was cut from was not.
    """
    return body(segment.covers, segment.delta, budget)


@dataclass(frozen=True)
class Narration:
    """A turn's narration tree: what plays at Stop, and what is there to be opened afterwards.

    `repository` holds nothing at all for a turn that left the repository where it found it, rather than being a
    segment that may be absent, so the caller speaks it unconditionally [LAW:dataflow-not-control-flow].
    """

    headline: Segment
    # Nothing for a turn that finished, rather than a segment that may be absent, as `repository` is.
    interrupted: tuple[Segment, ...]
    repository: tuple[Segment, ...]
    questions: tuple[Segment, ...]
    sections: tuple[Segment, ...]

    def said(self) -> str:
        """What plays when the turn stops: the headline, cut to its number of sentences, and what git says.

        The repository's part is the one clause of the top level no model wrote, and that is the point. Whether
        a turn committed is a fact its steps and its delta both record, and a summariser asked for it was
        measured on 2026-09-22 both dropping it and reporting it with the hash read out. Said from the types it
        can be neither, and the instruction is free to say nothing about commits at all [LAW:one-source-of-truth].

        The sections and the verbatim question segments are built and not spoken, for two different reasons. A menu of topics
        read after every turn would defeat the length number, whose whole point is a turn short enough to sit
        through; a listener who wants a topic asks for it. A question is held back because a question segment
        still holds the words Claude typed at a screen — a path, a hash, a "(Recommended)" — and reading those
        out is the one thing this epic forbids, while the headline has already been told to end on the turn's
        question in spoken form. Playing both would say it twice, once well and once verbatim
        [LAW:one-source-of-truth]. Making a question play at every length is `hands-narration-2mc.4mu`, which
        owns finding them in prose as well and can summarise them once it does.
        """
        reported, asked = _reported_and_asked(self.headline.text)
        # The question goes last whatever the summariser put where: it is the one sentence the listener answers,
        # and a fact read out after it leaves them holding the answer to something already gone by.
        # That the user stopped it goes first: it is what they are listening for, and it says why what follows is unfinished.
        return " ".join([*(part.text for part in self.interrupted), *reported, *(part.text for part in self.repository), *asked])


def narration(said: str, turn: Turn, delta: Delta, sentences: int) -> Narration:
    """The tree for one turn: `said` is what a summariser made of the whole of it, and the rest is arithmetic.

    The delta belongs to the turn rather than to any step of it — it is what all of them came to, and it names
    files no step reports — so it is in the headline's reach and in a section of its own, and in no other.
    """
    asked = [step for step in turn.steps if isinstance(step, Questioned)]
    sectioned = [step for step in turn.steps if not isinstance(step, Questioned | Interruption)]
    changes = tuple(change for step in turn.steps if isinstance(step, Ran) for change in step.git)
    return Narration(
        headline=Segment(THE_HEADLINE, _headline(said, sentences), (turn.opening, *turn.steps), delta),
        # Said by the types and not left to the summariser, for the reason the repository is: whether the turn was
        # stopped is a fact its record holds, and a model asked for it can drop it [LAW:one-source-of-truth]. Said of a
        # turn that ends on one: a queued message that cut a tool off mid-turn let the turn go on, and it is a step.
        interrupted=tuple(Segment(THE_INTERRUPTION, "You interrupted it.", (step,)) for step in turn.steps[-1:] if isinstance(step, Interruption)),
        repository=_repository(delta, changes),
        questions=tuple(_asked(question, step) for step in asked for question in step.questions),
        sections=_sections(sectioned),
    )


def _headline(said: str, sentences: int) -> str:
    """The summariser's reply cut to the length the top level was given, keeping a closing question whole.

    The length is held here and not by the instruction alone, for the reason the spoken-form filter is held in
    one place: a rule a model is asked to follow is obeyed or not and checked by nobody. Asked for one sentence,
    the local model wrote two in six of eight tellings on 2026-09-22, and each extra sentence is seconds a
    listener cannot skip. Every question is kept, whatever the number and however many there are, because a turn
    waiting on an answer that never asks is worse than a long one [LAW:single-enforcer]. All of them and not the
    last: a small model routinely splits one choice over two sentences, and keeping only the last leaves the
    listener a dangling alternative with the question that gave it meaning deleted.

    Nothing cut is lost: the sections hold every step the headline was made from, and opening one is what they
    are for.
    """
    reported, asked = _reported_and_asked(said)
    kept = " ".join([*reported[:sentences], *asked])
    # A reply that ended without a stop would run into what git says next as one long sentence, and speech has
    # no other way to hear the join.
    return kept if not kept or kept.endswith((".", "!", "?")) else f"{kept}."


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _reported_and_asked(text: str) -> tuple[list[str], list[str]]:
    """The text's sentences, split into what it told and what it asked.

    One definition, because two places recognise a question and neither may drift from the other: the headline
    is cut to a length every question survives, and what plays puts the questions after what git says. Question
    detection growing cleverer than a trailing mark is `hands-narration-2mc.4mu`, and this is the one place it
    has to land [LAW:one-source-of-truth].
    """
    written = sentences_of(text)
    return [sentence for sentence in written if not sentence.endswith("?")], [sentence for sentence in written if sentence.endswith("?")]


def sentences_of(text: str) -> list[str]:
    """The text as sentences. One definition, so what the length is enforced by and what it is measured by agree."""
    return [sentence for sentence in _SENTENCE.split(text.strip()) if sentence]


def _sections(steps: list[Sectioned]) -> tuple[Segment, ...]:
    """One segment per kind of step present, in the order the kinds first appear.

    First-appearance order rather than a ranking, because the turn's own order is a fact and a ranking would be
    one more table to keep true [LAW:one-source-of-truth].
    """
    grouped: dict[Topic, list[Sectioned]] = {}
    for step in steps:
        grouped.setdefault(topic_of(step), []).append(step)
    return tuple(
        Segment(topic, f"{_capitalised(topic.name)}: {_counted(len(held), topic.thing)}.", tuple(held))
        for topic, held in grouped.items()
    )


def _repository(delta: Delta, changes: tuple[GitChange, ...]) -> tuple[Segment, ...]:
    """What the turn did to the repository, from the two records of it, or nothing where it did not move.

    Both sources are read because each sees what the other misses: a `git commit` inside a compound command
    carries no operation for a step to record, and a delta read against the turn's start names the commit
    anyway. Whichever saw it, the same words come out, and a change both of them saw is said once.
    """
    said: list[str] = [
        *(_action(change) for change in changes),
        *(["committed"] if delta.commits else []),
        *([f"left {_counted(len(delta.files), 'file')} different"] if delta.files else []),
    ]
    # Ordered and deduplicated in one step: the steps and the delta both see a commit, and it is said once.
    did = list(dict.fromkeys(said))
    if did:
        return (Segment(THE_REPOSITORY, f"It {_listed(did)}.", (), delta),)
    # A delta holding only a patch is git having answered one question and not the next: something moved, and
    # saying that badly beats saying nothing [LAW:no-silent-failure].
    return (Segment(THE_REPOSITORY, "The repository moved.", (), delta),) if delta else ()


def _action(change: GitChange) -> str:
    """What one recorded git operation is called out loud. Never the hash or the number, which cannot be heard.

    A ref is said by `spoken_ref` and not copied: this clause is the one part of the top level no model wrote, so
    the instruction cannot cover it, and the filter in front of the speaker will not read a bare `feature/x` as a
    path for reasons of its own. Copied through, TTS reads the slash aloud.
    """
    match change:
        case Committed():
            return "committed"
        case Pushed(branch=branch):
            return f"pushed {spoken_ref(branch)}"
        case Branched(ref=ref, action=action):
            return f"{action} {spoken_ref(ref)}"
        case PullRequested(action=action):
            return f"{action} a pull request"


def _asked(question: Question, step: Questioned) -> Segment:
    """A question, said as a question.

    The one segment whose text is the thing itself rather than a count of it: a topic only has to be named for a
    listener to choose it, where a question has to be asked for them to answer it.
    """
    options = "" if not question.options else f" Either {_listed(list(question.options))}."
    unanswered = f"It is asking: {question.question}{options}"
    return Segment(A_QUESTION, unanswered if question.answer is None else f"It asked: {question.question} You chose {question.answer}.", (step,))


def _counted(many: int, thing: str) -> str:
    """A count and the thing counted, with the number as the word the instruction asks the model for.

    No filter downstream turns a digit back into a word, and these clauses are the ones no model wrote, so a
    digit written here is a digit the listener gets in the middle of a sentence of words.
    """
    return f"{spoken_count(many)} {thing}{'' if many == 1 else 's'}"


def _listed(parts: list[str]) -> str:
    """The parts as one spoken list, which needs no comma before the last of two."""
    return parts[-1] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"


def _capitalised(name: str) -> str:
    """A topic's name at the start of a sentence; `str.capitalize` would lower-case the rest of it."""
    return name[:1].upper() + name[1:]
