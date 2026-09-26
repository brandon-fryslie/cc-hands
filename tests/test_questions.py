"""What a turn is waiting on the listener to answer, read off its text by the daemon and by no model."""

import importlib.util
import sys
from pathlib import Path

import pytest

from hands.core.narration import reported_and_asked

# The eval's own reading of its fixtures, so the cases this holds to zero are the cases it reports on.
_SPEC = importlib.util.spec_from_file_location("narration_eval", Path(__file__).parents[1] / "evals" / "narration.py")
assert _SPEC is not None and _SPEC.loader is not None
EVAL = importlib.util.module_from_spec(_SPEC)
sys.modules["narration_eval"] = EVAL
_SPEC.loader.exec_module(EVAL)


def asked(text: str) -> list[str]:
    return reported_and_asked(text)[1]


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
    assert reported_and_asked("The rename is in, but three session tests\nstill fail.") == (["The rename is in, but three session tests still fail."], [])


def test_a_question_mark_muted_for_being_quoted_is_given_back_to_the_sentence_that_holds_it() -> None:
    reported, questions = reported_and_asked('It asks "why?" at the end. Want me to change that?')
    assert reported == ['It asks "why?" at the end.'] and questions == ["Want me to change that?"]


@pytest.mark.parametrize("case", EVAL.cases(), ids=lambda case: case.name)
def test_every_real_turn_in_the_eval_is_read_for_its_questions_with_no_miss_and_no_false_alarm(case: object) -> None:
    """The eval reports the same reading over the same fixtures; this holds it to zero with no model to reach."""
    found = EVAL.detection(case)
    assert (found.missed, found.alarmed) == ((), ())


def test_the_eval_holds_turns_that_ask_and_turns_that_do_not_including_the_shapes_a_question_mark_misreads() -> None:
    """At least three of each, and among them the cases a trailing question mark gets wrong both ways."""
    named = {case.name: case for case in EVAL.cases()}
    asking = [case for case in named.values() if case.asks]
    assert len(asking) >= 3 and len(named) - len(asking) >= 3
    # An offer with no question mark anywhere in its closing text.
    assert "?" not in named["offered-without-a-question-mark"].turn.steps[-1].text
    # A question mark that is not a question the listener is asked.
    assert "?" in named["answered-its-own-question"].turn.steps[-1].text
    assert "?" in named["quoted-a-question"].turn.steps[-1].text
