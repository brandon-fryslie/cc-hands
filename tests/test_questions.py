"""What a turn is waiting on the listener to answer, read off its text by the daemon and by no model."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from hands.core.narration import asked_in, open_questions, reading
from hands.core.turn import Answering, Asked, Continuing, Interruption, Notified, Opening, Said, Step, Turn
from hands.sessions.transcript import turn_record
from hands.sessions.turning import Turning

# Real turns, each lifted whole out of a real transcript, beside a few words of each question it is waiting on.
TURNS = Path(__file__).parent / "fixtures" / "turns"


@dataclass(frozen=True)
class Case:
    name: str
    turn: Turn
    # A few words of each question the turn is waiting on, as Claude wrote them; empty for a turn that asked nothing.
    asks: tuple[str, ...]


def cases() -> list[Case]:
    """Every case under `fixtures/turns`, in name order. A case may name another's transcript rather than copy it, which
    is what lets the same turn appear once as a first telling and once as a second [LAW:one-source-of-truth]."""
    found: list[Case] = []
    for expectation in sorted(TURNS.glob("*/expect.json")):
        written = json.loads(expectation.read_text())
        named = written.get("transcript", "transcript.jsonl")
        opening, steps = _folded(expectation.parent / named if "/" not in named else TURNS / named)
        told = int(written.get("told", 0))
        turn = Turn(opening, tuple(steps[told:]), Answering() if told == 0 else Continuing(told))
        # Required, not defaulted: a case silent about its question would count as asking none.
        found.append(Case(expectation.parent.name, turn, tuple(written["asks"])))
    return found


def _folded(transcript: Path) -> tuple[Opening, list[Step]]:
    """The transcript, read into a turn by the recognisers the daemon reads with."""
    turning = Turning()
    for line in transcript.read_bytes().splitlines():
        record = turn_record(line)
        if record is None:
            continue
        match turning.consume(record):
            case Asked() | Notified() as opening:
                turning.clear()
                turning.begin(opening)
            case Interruption() | None:
                pass
    assert turning.opening is not None, f"{transcript} holds no turn"
    return turning.opening, turning.steps()


def asked(text: str) -> list[str]:
    return asked_in(text)


@pytest.mark.parametrize(
    ("text", "questions"),
    [
        ("Fixed the test. Want me to look at the others?", ["Want me to look at the others?"]),
        ("Fixed it. Want me to also update the docs?", ["Want me to also update the docs?"]),
        ("Should I push, or wait for review?", ["Should I push, or wait for review?"]),
        # Put to the listener, so a sentence after it in the same paragraph does not answer it.
        ("Where do you want the directory? I'll scaffold the daemon there.", ["Where do you want the directory?"]),
        ("Which do you prefer?\n\n- The transport.\n- The keyboard.", ["Which do you prefer?"]),
        ("Should it update the logout code, or roll the rename back?", ["Should it update the logout code, or roll the rename back?"]),
        # Offers with no question mark, which a trailing-mark rule never hears.
        ("I'd remove the dev build. Say the word and I'll do it.", ["Say the word and I'll do it."]),
        ("Ready to scaffold when you are — just tell me where.", ["Ready to scaffold when you are — just tell me where."]),
        ("Left both branches alone. Let me know if you want them deleted.", ["Let me know if you want them deleted."]),
        ("Your call — which do you want?", ["Your call — which do you want?"]),
        # A question put to the listener outright, which went on to three more paragraphs.
        (
            "Trivial fix — want me to do it?\n\n## Another thing\n\nThe cache is stale.\n\nThe fix is already live.",
            ["Trivial fix — want me to do it?"],
        ),
        # A choice laid out as a list after the question that offers it.
        ("Which should go first?\n\n1. The transport.\n2. The keyboard.", ["Which should go first?"]),
        ("**Want me to merge it?**", ["**Want me to merge it?**"]),
        ("Two gaps are left (should I file them?)", ["Two gaps are left (should I file them?)"]),
        # An "e.g." ends no sentence, so the question is asked whole.
        ("It could move to the root (e.g. `templates/`) — want me to move it?", ["It could move to the root (e.g. `templates/`) — want me to move it?"]),
    ],
)
def test_a_question_or_an_offer_the_text_ends_on_is_asked(text: str, questions: list[str]) -> None:
    assert asked(text) == questions


@pytest.mark.parametrize(
    "text",
    [
        "Done. All twelve tests pass.",
        "",
        # The text asked itself and answered, in an earlier paragraph or in the one it ends on.
        "Why did I say it? Genre pressure, honestly.\n\nThe corrected sheet is in.",
        "Why did it fail? The cache was stale. Fixed now.",
        "Is this right? I think so, the tests agree.",
        # The same, with the answer on the question's next line.
        "**Why did it fail?**\nThe cache was stale. Fixed now.",
        # Inside code, a code span, a quotation, and an aside in italics: written about, not asked.
        "The tool list:\n\n```\nread_session(session, since?)\nspeak(text)\n```\n\nIt is built.",
        "It now reads `listening?.method ?? chosen`.",
        'The first launch asks "How should it give you what you say?" with two buttons.',
        "Diagnostic, in the style of the others: *who is waiting on this hold?*",
        # A question with a word straight after it is inside an address or a name.
        "The review is only returned with ?limit=50 on https://git.example/pulls/193/reviews?limit=50.",
        # Answered on its own line.
        "1. Are snapshots machine-global? → **yes**\n2. Is dedup in scope? → **no**",
        # A question not addressed to the listener, well above where the text ends.
        "Does the cache survive a restart? It does not.\n\nSo the fix reads it fresh every time.",
        # A heading names what follows and asks nothing.
        "## Needs your call\n\nNothing is open any more.",
    ],
)
def test_a_text_that_asks_the_listener_nothing_is_not_read_as_asking(text: str) -> None:
    assert asked(text) == []


def test_a_sentence_wrapped_over_two_lines_is_one_sentence() -> None:
    assert reading("The rename is in, but three session tests\nstill fail.") == [(False, "The rename is in, but three session tests still fail.")]


def test_a_question_mark_muted_for_being_quoted_is_given_back_to_the_sentence_that_holds_it() -> None:
    assert reading('It asks "why?" at the end. Want me to change that?') == [(False, 'It asks "why?" at the end.'), (True, "Want me to change that?")]


@pytest.mark.parametrize("case", cases(), ids=lambda case: case.name)
def test_every_real_turn_is_read_for_its_questions_with_no_miss_and_no_false_alarm(case: Case) -> None:
    found = [question.asked for question in open_questions(case.turn)]
    missed = [asked for asked in case.asks if not any(asked in question for question in found)]
    alarmed = [question for question in found if not any(asked in question for asked in case.asks)]
    assert (missed, alarmed) == ([], [])


def test_the_real_turns_hold_turns_that_ask_and_turns_that_do_not_including_the_shapes_a_question_mark_misreads() -> None:
    """At least three of each, and among them the cases a trailing question mark gets wrong both ways."""
    named = {case.name: case for case in cases()}
    asking = [case for case in named.values() if case.asks]
    assert len(asking) >= 3 and len(named) - len(asking) >= 3
    # An offer with no question mark anywhere in its closing text.
    assert "?" not in _closing(named["offered-without-a-question-mark"])
    # A question mark that is not a question the listener is asked.
    assert "?" in _closing(named["answered-its-own-question"])
    assert "?" in _closing(named["quoted-a-question"])


def _closing(case: Case) -> str:
    last = case.turn.steps[-1]
    assert isinstance(last, Said)
    return last.text
