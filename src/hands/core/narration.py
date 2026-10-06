"""A turn, heard: what hands knows of it from its types, the questions it left open, and one section per topic.

The tree is cut from the turn's own types. `Step` already separates an edit from a test run from a command that
committed, so the sections are a match over that union rather than a table of rules invented beside it, and a
segment's record ids are read off the steps it holds rather than claimed by a model `[LAW:one-source-of-truth]`.
That is what keeps "that part" a lookup: the map is redrawn from the territory on every build.

No text here comes from a model. Every count is arithmetic over typed steps, so the facts hands hands on with a
turn — that it was interrupted, what git says, what it is waiting on — cannot be hallucinated, and what the turn
did is left to the words of the session that did it.
"""

import re
from dataclasses import dataclass

from hands.core.delta import Branched, Committed, Delta, GitChange, PullRequested, Pushed
from hands.core.spoken import counted, spoken, spoken_ref
from hands.core.subagents import Subagent
from hands.core.turn import (
    Budget,
    Delegated,
    Edited,
    Happening,
    Interruption,
    Looked,
    Other,
    Planned,
    Question,
    Questioned,
    Ran,
    Ref,
    Reported,
    Said,
    Step,
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


THE_QUESTION = Topic("the question", "question")
WHAT_IT_ASKED = Topic("what it asked", "question")
WHAT_IT_SAID = Topic("what it said", "message")
THE_CHANGE = Topic("the change", "edit")
THE_TESTS = Topic("the tests", "test run")
THE_COMMIT = Topic("the commit", "command")
THE_COMMANDS = Topic("the commands", "command")
WHAT_IT_READ = Topic("what it read", "look")
THE_PLAN = Topic("the plan", "task")
THE_SUBAGENTS = Topic("the subagents", "subagent")
THE_NOTIFICATIONS = Topic("the notifications", "notification")
THE_OTHER_TOOLS = Topic("the other tools", "tool call")
THE_REPOSITORY = Topic("the repository", "file")
THE_INTERRUPTION = Topic("the interruption", "interruption")

# The steps a section is cut from. A question and an interruption are not among them: each has a segment of its own
# at the top level, and an open question and an interruption play at every length, so the type that says which steps
# fall into sections is also the type that says they never do [LAW:types-are-the-program]. Without it, `topic_of`
# would need an arm for a step it can never be handed.
Sectioned = Said | Edited | Ran | Tested | Looked | Planned | Delegated | Reported | Other


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
        case Reported():
            # Something that arrived, not something Claude did: counted with the subagents, a report would count the
            # subagent its launch already counts a second time, and with the tools it would be a call nobody made.
            return THE_NOTIFICATIONS
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
    """A turn's narration tree: what hands hands the model of it at Stop, and what is there to be opened afterwards.

    `repository` holds nothing at all for a turn that left the repository where it found it, rather than being a
    segment that may be absent, so the caller speaks it unconditionally [LAW:dataflow-not-control-flow]. So do
    `interrupted` for a turn that finished and `questions` for one that is waiting on nothing.
    """

    interrupted: tuple[Segment, ...]
    repository: tuple[Segment, ...]
    # What the turn is waiting on the listener to answer. Plays at every length.
    questions: tuple[Segment, ...]
    # What it asked through `AskUserQuestion` and is not waiting on any more, answered or gone past: there to be
    # opened, and never played, since it is news to nobody.
    settled: tuple[Segment, ...]
    sections: tuple[Segment, ...]
    # What each subagent that reported back in this turn did, read from its own transcript: there to be opened, so
    # "what did the reviewer find" is answered from the reviewer's steps and not from the line its parent made of them.
    subagents: tuple[Segment, ...]

    def facts(self) -> str:
        """What hands adds of the turn from its types: that it was stopped, and what git says.

        Whether a turn committed is a fact its steps and its delta both record, and a summariser asked for it was
        measured on 2026-09-22 both dropping it and reporting it with the hash read out: handed on from the types it
        can be neither [LAW:one-source-of-truth]. The sections are built and not handed on: a listener who wants a
        topic asks for it.
        """
        # That the user stopped it goes first: it is what they are listening for, and it says why what follows is unfinished.
        return " ".join(part.text for part in (*self.interrupted, *self.repository))

    def asked(self) -> str:
        """What the turn is waiting on the listener to answer."""
        return " ".join(question.text for question in self.questions)

    @property
    def parts(self) -> tuple[Segment, ...]:
        """Every segment there is to open, in the order the top level plays: "more on that" can reach anything the turn did."""
        return (*self.interrupted, *self.repository, *self.questions, *self.settled, *self.sections, *self.subagents)


def narration(turn: Turn, delta: Delta, subagents: tuple[Subagent, ...]) -> Narration:
    """The tree for one turn, all of it arithmetic over the turn's steps and what git says it came to.

    Whether the turn asked anything, and what, is the daemon's to find, from the turn's own closing text and its
    `AskUserQuestion` calls, and never a model's: a model asked to find it can miss one, and can find one the turn
    never asked [LAW:one-source-of-truth]. The model it is handed to puts it in its own words; whatever hands says
    as written, with no model, says it as found.
    """
    waiting = open_questions(turn)
    sectioned = [step for step in turn.steps if not isinstance(step, Questioned | Interruption)]
    changes = tuple(change for step in turn.steps if isinstance(step, Ran) for change in step.git)
    return Narration(
        # Said by the types, for the reason the repository is: whether the turn was stopped is a fact its record holds,
        # and a model asked for it can drop it [LAW:one-source-of-truth]. Said of a turn that ends on one: a queued
        # message that cut a tool off mid-turn let the turn go on, and it is a step.
        interrupted=tuple(Segment(THE_INTERRUPTION, "You interrupted it.", (step,)) for step in _acted(turn)[-1:] if isinstance(step, Interruption)),
        repository=_repository(delta, changes),
        questions=_questions(waiting),
        settled=tuple(
            _settled(question, step)
            for step in turn.steps
            if isinstance(step, Questioned)
            for question in step.questions
            if InDialog(step, question) not in waiting
        ),
        sections=_sections(sectioned),
        # A subagent stopped before it did anything has nothing to open.
        subagents=tuple(_own_work(subagent) for subagent in subagents if subagent.steps),
    )


def _own_work(subagent: Subagent) -> Segment:
    """One subagent's work, a part of its own named by the job it was given: two subagents in one turn are opened one at
    a time, each as deep as it was asked for. Named in lower case, as a part is asked for."""
    topic = Topic(f"the subagent's work on {subagent.description.lower()}", "step")
    return Segment(topic, f"A subagent's own work on {subagent.description}: {counted(len(subagent.steps), 'step')}.", subagent.steps)


@dataclass(frozen=True)
class InText:
    """What the turn's closing text asks the listener or offers them: one thing, however many sentences it takes,
    since Claude splits one choice over two."""

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

    Only where the turn ended on it: a turn that asked something and then went on working had its answer or did
    not need one. A dialog is left waiting where nothing but the user stopping it came after, which is how an
    Escape at the dialog ends a turn; one Claude went on past was declined with a message or refused by a hook, as
    5 of the 94 unanswered dialogs in this machine's transcripts were on 2026-09-25. Text an interruption
    followed was cut off, not left waiting.
    """
    acted = _acted(turn)
    unanswered = [
        InDialog(step, question)
        for at, step in enumerate(acted)
        if isinstance(step, Questioned) and all(isinstance(later, Interruption) for later in acted[at + 1 :])
        for question in step.questions
        if question.answer is None
    ]
    closing = [InText(step, " ".join(asked)) for step in acted[-1:] if isinstance(step, Said) for asked in [asked_in(step.text)] if asked]
    return (*unanswered, *closing)


def _acted(turn: Turn) -> list[Step]:
    """The turn's steps that Claude or the user took, which are what say where it ended: a notification arriving after a
    question or an interruption, as 10 of this machine's 5,243 mid-turn ones did on 2026-10-03, moves nothing."""
    return [step for step in turn.steps if not isinstance(step, Reported)]


def _questions(waiting: tuple[Open, ...]) -> tuple[Segment, ...]:
    """The one segment that asks what the turn is waiting on, or nothing for a turn waiting on nothing."""
    match waiting:
        case ():
            return ()
        case _:
            holding = tuple(dict.fromkeys(question.step for question in waiting))
            return (Segment(THE_QUESTION, _unworded(waiting), holding),)


def _unworded(waiting: tuple[Open, ...]) -> str:
    """The questions as Claude put them, in spoken form.

    Through `spoken` here as well as in front of the speaker, because these words were written for a screen and
    are said whole; the filter changes nothing the second time. "(Recommended)" is Claude marking an option for a reader, which a listener would hear as
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
_MARKER = r"[ \t]*(?:[-*+]|\d{1,2}[.)])[ \t]+"
_LISTED = re.compile(rf"^{_MARKER}")
# Where a paragraph's next list item or table row begins. A line break anywhere else is a wrap, inside one sentence or
# between two.
_ITEM = re.compile(rf"\n(?={_MARKER}|[ \t]*\|)")
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]")
# A sentence ends at its mark, or past the bold, bracket or quotation that closes with it, but not at an "e.g." or an
# "i.e." inside it.
_SENTENCE = re.compile(r"(?<=[.!?])(?<!\be\.g\.)(?<!\bi\.e\.)\s+|(?<=[.!?][*_)\"'”’])\s+|(?<=[.!?][*_)\"'”’]{2})\s+")
_CLOSERS = "*_)\"'”’ "
# A question put to the listener outright, which is asked wherever in the text it stands.
# Asked of the session as "it" too: "Want it to carry on?"
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


def asked_in(text: str) -> list[str]:
    """The sentences in which the text asks the listener something, as `reading` reads them."""
    return [sentence for asks, sentence in reading(text) if asks]


def reading(text: str) -> list[tuple[bool, str]]:
    """The text's sentences in order, each with whether it asks the listener something.

    A question put to the listener outright is asked wherever it stands: "Want me to do it?" asked above three
    more sections still waits on an answer. Any other question is asked only where the text ends on it — its
    last paragraph, or the one that introduces the list it ends on — and only where its own paragraph does not
    go on to tell something after it: "Why did it fail? The cache was stale." is the text asking itself, and it
    answered. An offer is asked where the text ends on it: "Say the word and I'll do it." The shapes were read
    off 3,038 closing texts in this machine's transcripts on 2026-09-25, and tests/fixtures/turns holds a real turn
    of each shape that decides a case, asked and not.
    """
    muted = _QUOTED.sub(lambda quoted: quoted.group(0).replace("?", _MUTED), _FENCED.sub("", text))
    paragraphs = [paragraph for paragraph in _PARAGRAPH.split(muted.strip()) if paragraph.strip()]
    closing = len(paragraphs) - 1
    # A choice laid out as a list is asked by the sentence above it, and the text ends on the list.
    while closing > 0 and all(_LISTED.match(line) for line in paragraphs[closing].splitlines() if line.strip()):
        closing -= 1
    return [(asks, sentence.replace(_MUTED, "?")) for at, paragraph in enumerate(paragraphs) for asks, sentence in _read(paragraph, at >= closing)]


def _read(paragraph: str, closing: bool) -> list[tuple[bool, str]]:
    """Each sentence of a paragraph, and whether it asks the listener something.

    Read from the end, because whether a question was the text asking itself depends on what follows it: a
    question its own item goes on to tell after — "Why did it fail? The cache was stale." — answered itself.
    Unless something later in the paragraph looks ahead to an answer still to come — "What is pi? A pointer is
    enough. Once I have that I'll pin the interface." — which in 3,038 closing texts on this machine is what
    every real question followed by more of its line did. The item and not the paragraph, because what follows
    a list item is the other items, which answer nothing.
    """
    read: list[tuple[bool, str]] = []
    ahead = False
    for item in reversed(_items(paragraph)):
        told = False
        for sentence in reversed(item):
            own = _SPAN.sub("", sentence)
            questioned = _questioned(sentence)
            addressed = questioned and bool(_ADDRESSED.search(own))
            # A choice where the text ends is put to someone: "does it mean the suite, or also the smoke test?" was
            # followed by what each option would cost, which tells and answers nothing.
            chosen = questioned and bool(_CHOICE.search(own))
            asks = addressed or closing and (bool(_OFFERED.search(own)) or chosen or questioned and (ahead or not told))
            told = told or not asks
            ahead = ahead or not asks and bool(_AHEAD.search(own))
            read.append((asks, sentence))
    return read[::-1]


def _questioned(sentence: str) -> bool:
    return sentence.rstrip(_CLOSERS).endswith("?")


def _items(paragraph: str) -> list[list[str]]:
    """A paragraph's sentences, a list item or table row at a time so each is its own, and the prose around them whole
    however it was wrapped: a sentence broken over two lines is one sentence. No heading is among them: a title
    names what follows it and asks nothing."""
    kept = "\n".join(line for line in paragraph.splitlines() if not _HEADING.match(line))
    items = [" ".join(_LISTED.sub("", item).split()) for item in _ITEM.split(kept)]
    return [[sentence for sentence in _SENTENCE.split(item) if sentence] for item in items]


def _sections(steps: list[Sectioned]) -> tuple[Segment, ...]:
    """One segment per kind of step present, in the order the kinds first appear.

    First-appearance order rather than a ranking, because the turn's own order is a fact and a ranking would be
    one more table to keep true [LAW:one-source-of-truth].
    """
    grouped: dict[Topic, list[Sectioned]] = {}
    for step in steps:
        grouped.setdefault(topic_of(step), []).append(step)
    return tuple(
        Segment(topic, f"{_capitalised(topic.name)}: {counted(len(held), topic.thing)}.", tuple(held))
        for topic, held in grouped.items()
    )


def _repository(delta: Delta, changes: tuple[GitChange, ...]) -> tuple[Segment, ...]:
    """What the turn did to the repository, from the two records of it, or nothing where it did not move.

    Both sources are read because each sees what the other misses: a `git commit`, a push, or a `checkout -b`
    inside a compound command carries no operation for a step to record, and a delta read against the turn's
    start names it anyway. Whichever saw it, the same words come out, and a change both of them saw is said once.
    """
    said: list[str] = [
        *(_action(change) for change in changes),
        *(["committed"] if delta.commits else []),
        *(_action(change) for change in delta.changes),
        *([f"left {counted(len(delta.files), 'file')} different"] if delta.files else []),
    ]
    # Ordered and deduplicated in one step: the steps and the delta both see a commit, a push, or a pull request,
    # and each is said once.
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


def _settled(question: Question, step: Questioned) -> Segment:
    """A question the turn is not waiting on any more, and how it ended, for a listener who asks what it was."""
    match question.answer:
        case str() as chosen:
            return Segment(WHAT_IT_ASKED, f"It asked: {question.question} You chose {chosen}.", (step,))
        case None:
            return Segment(WHAT_IT_ASKED, f"It asked: {question.question} It went on without an answer.", (step,))


def _listed(parts: list[str], last: str = "and") -> str:
    """The parts as one spoken list, with `last` before the last of them, which needs no comma before it."""
    return parts[-1] if len(parts) == 1 else f"{', '.join(parts[:-1])} {last} {parts[-1]}"


def _capitalised(name: str) -> str:
    """A topic's name at the start of a sentence; `str.capitalize` would lower-case the rest of it."""
    return name[:1].upper() + name[1:]
