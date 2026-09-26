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
from hands.core.spoken import spoken, spoken_count, spoken_ref
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
    render,
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
THE_QUESTION = Topic("the question", "question")
WHAT_YOU_ANSWERED = Topic("what you answered", "answer")
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

# The steps a section is cut from. A question and an interruption are not among them: each has a segment of its own
# at the top level, and an open question and an interruption play at every length, so the type that says which steps
# fall into sections is also the type that says they never do [LAW:types-are-the-program]. Without it, `topic_of`
# would need an arm for a step it can never be handed.
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
    segment that may be absent, so the caller speaks it unconditionally [LAW:dataflow-not-control-flow]. So do
    `interrupted` for a turn that finished and `questions` for one that is waiting on nothing.
    """

    headline: Segment
    interrupted: tuple[Segment, ...]
    repository: tuple[Segment, ...]
    # What the turn is waiting on the listener to answer. Plays at every length.
    questions: tuple[Segment, ...]
    # What the user already answered through `AskUserQuestion`: there to be opened, and never played, since it is
    # news to nobody.
    answered: tuple[Segment, ...]
    sections: tuple[Segment, ...]

    def said(self) -> str:
        """What plays when the turn stops: that it was stopped, the headline, what git says, and what it asked.

        The repository's part is the one clause of the top level no model wrote, and that is the point. Whether
        a turn committed is a fact its steps and its delta both record, and a summariser asked for it was
        measured on 2026-09-22 both dropping it and reporting it with the hash read out. Said from the types it
        can be neither, and the instruction is free to say nothing about commits at all [LAW:one-source-of-truth].

        The question goes last, whatever the summariser put where: it is the one sentence the listener answers,
        and a fact read out after it leaves them holding the answer to something already gone by. It is said
        once, because the headline does not hold it: `narration` takes it out of the summariser's reply and into
        a segment of its own. The sections are built and not spoken: a menu of topics read after every turn would
        defeat the length number, whose whole point is a turn short enough to sit through, and a listener who
        wants a topic asks for it.
        """
        # That the user stopped it goes first: it is what they are listening for, and it says why what follows is unfinished.
        parts = (*self.interrupted, self.headline, *self.repository, *self.questions)
        # A turn stopped before it did anything has an empty headline, which adds nothing rather than a space.
        return " ".join(part.text for part in parts if part.text)


def narration(said: str, turn: Turn, delta: Delta, sentences: int) -> Narration:
    """The tree for one turn: `said` is what a summariser made of the whole of it, and the rest is arithmetic.

    The delta belongs to the turn rather than to any step of it — it is what all of them came to, and it names
    files no step reports — so it is in the headline's reach and in a section of its own, and in no other.

    Whether the turn asked anything is the daemon's to say, from the turn's own closing text and its
    `AskUserQuestion` calls, and never the summariser's: a model asked to end on the turn's question can drop it,
    and can end on one the turn never asked [LAW:one-source-of-truth]. The summariser is left the one thing only
    it can do, which is to put the question in words that can be heard.
    """
    reported, worded = reported_and_asked(said)
    answered = [step for step in turn.steps if isinstance(step, Questioned)]
    sectioned = [step for step in turn.steps if not isinstance(step, Questioned | Interruption)]
    changes = tuple(change for step in turn.steps if isinstance(step, Ran) for change in step.git)
    return Narration(
        headline=Segment(THE_HEADLINE, _headline(reported, sentences), (turn.opening, *turn.steps), delta),
        # Said by the types and not left to the summariser, for the reason the repository is: whether the turn was
        # stopped is a fact its record holds, and a model asked for it can drop it [LAW:one-source-of-truth]. Said of a
        # turn that ends on one: a queued message that cut a tool off mid-turn let the turn go on, and it is a step.
        interrupted=tuple(Segment(THE_INTERRUPTION, "You interrupted it.", (step,)) for step in turn.steps[-1:] if isinstance(step, Interruption)),
        repository=_repository(delta, changes),
        questions=_questions(worded, open_questions(turn)),
        answered=tuple(_answered(question, step) for step in answered for question in step.questions if question.answer is not None),
        sections=_sections(sectioned),
    )


def _headline(reported: list[str], sentences: int) -> str:
    """What the summariser reported, cut to the length the top level was given.

    The length is held here and not by the instruction alone, for the reason the spoken-form filter is held in
    one place: a rule a model is asked to follow is obeyed or not and checked by nobody. Asked for one sentence,
    the local model wrote two in six of eight tellings on 2026-09-22, and each extra sentence is seconds a
    listener cannot skip [LAW:single-enforcer]. A question is neither counted nor cut, because it is not here:
    it is the question segment's, which plays at every length.

    Nothing cut is lost: the sections hold every step the headline was made from, and opening one is what they
    are for.
    """
    kept = " ".join(reported[:sentences])
    # A reply that ended without a stop would run into what git says next as one long sentence, and speech has
    # no other way to hear the join.
    return kept if not kept or kept.endswith((".", "!", "?")) else f"{kept}."


@dataclass(frozen=True)
class InText:
    """A sentence of the turn's closing text that asks the listener something, or offers them something."""

    step: Said
    asked: str


@dataclass(frozen=True)
class InDialog:
    """A question put through `AskUserQuestion` that nobody answered."""

    step: Questioned
    question: Question

    @property
    def asked(self) -> str:
        return self.question.question


# A question a turn left waiting on the listener, and the step that put it.
Open = InText | InDialog


def open_questions(turn: Turn) -> tuple[Open, ...]:
    """What the turn is waiting on the listener to answer: whatever it asked through `AskUserQuestion` and never
    had answered, and whatever its closing text asks.

    Closing text only, meaning text that is the turn's last step: a turn that asked something and then went on
    working had its answer or did not need one, and text an interruption followed was cut off, not left waiting.
    """
    unanswered = [
        InDialog(step, question) for step in turn.steps if isinstance(step, Questioned) for question in step.questions if question.answer is None
    ]
    closing = [InText(step, sentence) for step in turn.steps[-1:] if isinstance(step, Said) for sentence in reported_and_asked(step.text)[1]]
    return (*unanswered, *closing)


def shown(turn: Turn, delta: Delta, budget: Budget) -> str:
    """The summariser's message: the turn as `render` writes it, and what the daemon found it waiting on.

    Told rather than left for the model to find, because the model is the one wording what the daemon decided:
    the local model was measured on 2026-09-25 leaving out a question asked above two more sections, and ending
    four of nine turns that asked nothing on a question of its own. What it asks of a turn that asked nothing is
    dropped either way, and a question it leaves out is said in Claude's words, which are worse to hear. Given in
    spoken form, because the model copies what it read last: handed "leaving sync.ts unreviewed" here, it said
    "sync.ts" in the report above the question as well.
    """
    waiting = open_questions(turn)
    told = f"The turn is waiting on the user's answer to this, and your report ends by asking it:\n  {_unworded(waiting)}" if waiting else "The turn asks the user nothing."
    return f"{render(turn, delta, budget)}\n\n{told}"


def _questions(worded: list[str], waiting: tuple[Open, ...]) -> tuple[Segment, ...]:
    """The one segment that asks what the turn is waiting on, or nothing for a turn waiting on nothing.

    In the summariser's words where it asked and the turn is waiting on one thing, because they are the words
    made to be heard, and whatever it asked can only be that thing. The closing text is one thing however many
    sentences it asks in, since Claude splits one choice over two; each unanswered dialog question is another.
    Waiting on two, nothing says which of them the summariser's words cover — asked one, it can drop the other,
    and then that one is said zero times — so each is said in Claude's own words instead, put in spoken form,
    as it is where the summariser asked nothing: a turn waiting on an answer it never asks for is the failure
    this segment exists to prevent, and saying a question less well beats not saying it [LAW:no-silent-failure].
    What the summariser asked of a turn that asked nothing is dropped, which the instruction forbids it to write.
    """
    things = len(dict.fromkeys(question.step if isinstance(question, InText) else question for question in waiting))
    match waiting:
        case ():
            return ()
        case _:
            holding = tuple(dict.fromkeys(question.step for question in waiting))
            return (Segment(THE_QUESTION, " ".join(worded) if worded and things == 1 else _unworded(waiting), holding),)


def _unworded(waiting: tuple[Open, ...]) -> str:
    """The questions as Claude put them, for a turn whose summariser did not ask them, in spoken form.

    Through `spoken` here as well as in front of the speaker, because these words were written for a screen and
    are said whole, and a line put in spoken form here is one the eval can hold to it; the filter changes nothing
    the second time. "(Recommended)" is Claude marking an option for a reader, which a listener would hear as
    part of the option's name, so it is taken out.
    """
    return spoken(" ".join(_put(question) for question in waiting)).text


def _put(question: Open) -> str:
    """One question as Claude put it, framed as Claude's so its "I" and "me" are heard as the session's."""
    match question:
        case InDialog(question=Question(options=offered)):
            options = [_RECOMMENDED.sub("", option) for option in offered]
            return f"It is asking: {question.asked}{f' Either {_listed(options, "or")}.' if options else ''}"
        case InText():
            # Said rather than asked: the sentence may be an offer, "Say the word and I'll do it", and not a question.
            return f"It said: {question.asked}"


_RECOMMENDED = re.compile(r"\s*\(recommended\)", re.IGNORECASE)

# What reading a text for its questions reads past. A question mark inside a code block, a code span, a quotation or
# an emphasised aside is written about rather than asked; one with a word straight after it is inside an address's
# query or a name; and one an arrow follows was answered on its own line. Each is muted rather than removed, so the
# sentence around it keeps its words.
_FENCED = re.compile(r"^[ \t]*(?P<run>(?P<mark>[`~])(?P=mark){2,}).*?(?:^[ \t]*(?P=run)(?P=mark)*[ \t]*$|\Z)", re.MULTILINE | re.DOTALL)
_SPANS = r"`[^`\n]*`|\"[^\"\n]*\"|“[^”\n]*”|(?<![\w*])\*[^*\n]+\*(?![\w*])|(?<!\w)_[^_\n]+_(?!\w)"
_QUOTED = re.compile(rf"{_SPANS}|\?(?=[\w=&/%])|\?(?=\s*(?:→|->|=>))")
# The same spans, taken out before a sentence is read for an offer: "should it" quoted is somebody else's question.
_SPAN = re.compile(_SPANS)
_MUTED = "\x00"
_PARAGRAPH = re.compile(r"\n[ \t]*\n")
_LISTED = re.compile(r"^[ \t]*(?:[-*+]|\d{1,2}[.)])[ \t]+")
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]")
# A sentence ends at its mark, or past the bold, bracket or quotation that closes with it.
_SENTENCE = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][*_)\"'”’])\s+|(?<=[.!?][*_)\"'”’]{2})\s+")
_CLOSERS = "*_)\"'”’ "
# A question put to the listener outright, which is asked wherever in the text it stands.
# The summariser's own voice is here too, since it asks with the session as "it": "Want it to carry on?"
_ADDRESSED = re.compile(r"\b(?:you|your|yours|want (?:me|it) to|should (?:I|it)|shall (?:I|it)|may I|can I|do I)\b", re.IGNORECASE)
_CHOICE = re.compile(r"\bor\b", re.IGNORECASE)
# A sentence that looks ahead to an answer still to come, rather than giving one: to the listener, a condition on
# what they say, or what happens once they have.
_AHEAD = re.compile(r"\b(?:you|your|you'd|if|once|whether|unless|either|until|I'll|I will|I'd|I would|tell me|let me know)\b", re.IGNORECASE)
# An offer or a choice put without a question mark, which is asked only where the text ends on it.
_OFFERED = re.compile(
    r"\b(?:let me know|tell me (?:which|whether|if|where|what|how|when)|say the word|your call|up to you"
    r"|if you(?:'d| would)? (?:like|prefer|rather|want)|want (?:me|it) to|would you like|should (?:I|it)|shall (?:I|it)"
    r"|do you want|pending your|awaiting your|ready to \w+ when you are)\b",
    re.IGNORECASE,
)


def reported_and_asked(text: str) -> tuple[list[str], list[str]]:
    """The text's sentences, split into what it told and what it asked of the listener.

    One definition, because the summariser's reply, Claude's own closing text and the idle nudge are all read for
    their questions, and were there two rules the headline could ask what the nudge says is no question, or the
    nudge promise one that is never said [LAW:one-source-of-truth].

    A question put to the listener outright is asked wherever it stands: "Want me to do it?" asked above three
    more sections still waits on an answer. Any other question is asked only where the text ends on it — its
    last paragraph, or the one that introduces the list it ends on — and only where its own paragraph does not
    go on to tell something after it: "Why did it fail? The cache was stale." is the text asking itself, and it
    answered. An offer is asked where the text ends on it: "Say the word and I'll do it." The shapes were read
    off 3,038 closing texts in this machine's transcripts on 2026-09-25, and the eval holds a real turn of each
    shape that decides a case, asked and not.
    """
    muted = _QUOTED.sub(lambda quoted: quoted.group(0).replace("?", _MUTED), _FENCED.sub("", text))
    paragraphs = [paragraph for paragraph in _PARAGRAPH.split(muted.strip()) if paragraph.strip()]
    closing = len(paragraphs) - 1
    # A choice laid out as a list is asked by the sentence above it, and the text ends on the list.
    while closing > 0 and all(_LISTED.match(line) for line in paragraphs[closing].splitlines() if line.strip()):
        closing -= 1
    read = [(asks, sentence.replace(_MUTED, "?")) for at, paragraph in enumerate(paragraphs) for asks, sentence in _read(paragraph, at >= closing)]
    return [sentence for asks, sentence in read if not asks], [sentence for asks, sentence in read if asks]


def _read(paragraph: str, closing: bool) -> list[tuple[bool, str]]:
    """Each sentence of a paragraph, and whether it asks the listener something.

    Read from the end, because whether a question was the text asking itself depends on what follows it: a
    question the same line goes on to tell after — "Why did it fail? The cache was stale." — answered itself.
    Unless something later in the paragraph looks ahead to an answer still to come — "What is pi? A pointer is
    enough. Once I have that I'll pin the interface." — which in 3,038 closing texts on this machine is what
    every real question followed by more of its line did. The line and not the paragraph, because the lines
    after a list item are the other items, which answer nothing.
    """
    read: list[tuple[bool, str]] = []
    ahead = False
    for line in reversed(_lines(paragraph)):
        told = False
        for sentence in reversed(line):
            own = _SPAN.sub("", sentence)
            questioned = sentence.rstrip(_CLOSERS).endswith("?")
            addressed = questioned and bool(_ADDRESSED.search(own))
            # A choice where the text ends is put to someone: "does it mean the suite, or also the smoke test?" was
            # followed by what each option would cost, which tells and answers nothing.
            chosen = questioned and bool(_CHOICE.search(own))
            asks = addressed or closing and (bool(_OFFERED.search(own)) or chosen or questioned and (ahead or not told))
            told = told or not asks
            ahead = ahead or not asks and bool(_AHEAD.search(own))
            read.append((asks, sentence))
    return read[::-1]


def _lines(paragraph: str) -> list[list[str]]:
    """A paragraph's sentences, a line at a time so a list's items are their own. No heading is among them: a
    title names what follows it and asks nothing."""
    lines = [_LISTED.sub("", line).strip() for line in paragraph.splitlines() if not _HEADING.match(line)]
    return [[sentence for sentence in _SENTENCE.split(line) if sentence] for line in lines]


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


def _answered(question: Question, step: Questioned) -> Segment:
    """A question the user already answered, and what they chose, for a listener who asks what it was."""
    return Segment(WHAT_YOU_ANSWERED, f"It asked: {question.question} You chose {question.answer}.", (step,))


def _counted(many: int, thing: str) -> str:
    """A count and the thing counted, with the number as the word the instruction asks the model for.

    No filter downstream turns a digit back into a word, and these clauses are the ones no model wrote, so a
    digit written here is a digit the listener gets in the middle of a sentence of words.
    """
    return f"{spoken_count(many)} {thing}{'' if many == 1 else 's'}"


def _listed(parts: list[str], last: str = "and") -> str:
    """The parts as one spoken list, with `last` before the last of them, which needs no comma before it."""
    return parts[-1] if len(parts) == 1 else f"{', '.join(parts[:-1])} {last} {parts[-1]}"


def _capitalised(name: str) -> str:
    """A topic's name at the start of a sentence; `str.capitalize` would lower-case the rest of it."""
    return name[:1].upper() + name[1:]
