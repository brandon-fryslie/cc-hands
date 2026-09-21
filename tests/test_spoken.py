"""Spoken form: nothing reaches text-to-speech as written.

The table below is the specification. Each row is a shape that turns up in real Claude replies and in the
summaries hands makes of them, written the way it arrives and the way it has to be heard.
"""

import re
from pathlib import Path
from typing import cast

import pytest
from loguru import logger

from hands.core.spoken import Leak, spoken
from hands.core.turn import Said
from hands.sessions.backfill import read_since
from hands.voice.spoken import SpokenForm

FIXTURE = Path(__file__).parent / "fixtures" / "session.jsonl"

WRITTEN_AND_SPOKEN = [
    # What the summariser was actually heard doing on this machine: a bare file name, extension and all.
    ("Fixed the count in notes.txt.", "Fixed the count in notes."),
    ("Fixed the count in src/notes.txt.", "Fixed the count in notes."),
    # Which must not cost the rule that keeps ordinary sentences intact.
    ("It is slow, e.g. on a big repository.", "It is slow, e.g. on a big repository."),
    # Code names: every symbol in one is read aloud, so the name becomes the words it is made of.
    ("`test_refresh` still fails.", "test refresh still fails."),
    ("getUserName returns nothing.", "get user name returns nothing."),
    ("It calls sessions.story() in a loop.", "It calls sessions story in a loop."),
    # A path is heard as its file, which is what a developer calls it out loud.
    ("Edited src/hands/core/turn.py.", "Edited turn."),
    # Unless two of them in the same breath would be the same word, when the directory comes back.
    (
        "Edited src/hands/core/turn.py and src/hands/sessions/turn.py.",
        "Edited core turn and sessions turn.",
    ),
    # Hashes, ids and addresses: named by what they are, because no character of them can be heard.
    ("Committed a1b2c3d.", "Committed a commit."),
    ("Committed as commit a1b2c3d.", "Committed as commit."),
    ("See https://docs.pipecat.ai/guide for the rest.", "See a link for the rest."),
    ("Request req_011CewaSjsqPyMa9o6U1HSUg timed out.", "Request timed out."),
    # Flags are their names; the dashes and the equals sign are how one is typed.
    ("Run it with --max-count=5.", "Run it with max count 5."),
    # Markdown is for the eye. A heading is a cue, which in speech is a sentence of its own.
    ("## Review cycle\n\nAll four passes are clean.", "Review cycle.\nAll four passes are clean."),
    ("It is **done** and *tested*.", "It is done and tested."),
    ("[the design doc](https://example.com/doc) says otherwise.", "the design doc says otherwise."),
    # A list read straight through is heard as one long sentence, so its items are counted aloud.
    (
        "- ran the suite\n- fixed the flaky test\n- committed",
        "First, ran the suite. Second, fixed the flaky test. Third, committed.",
    ),
    # One bullet is not a sequence, and "First," in front of a lone item promises a second.
    ("- ran the suite", "ran the suite."),
    # Ordinary English is left alone, which is most of what is said.
    ("The well-known issue is that three tests fail. Should it fix them?", "The well-known issue is that three tests fail. Should it fix them?"),
    ("It defaced the output, then passed.", "It defaced the output, then passed."),
]


@pytest.mark.parametrize(("written", "said"), WRITTEN_AND_SPOKEN)
def test_what_is_written_and_what_is_heard(written: str, said: str) -> None:
    assert spoken(written).text == said


def test_a_block_of_code_is_said_as_what_it_was_and_reported() -> None:
    """Speech cannot carry code at all, and a block that got this far was never summarised.

    Said rather than dropped, because a developer who cannot see the screen still needs to know something
    was there; reported as well, because the fault is upstream of here.
    """
    said = spoken("Here you go:\n```python\nx = 1\ny = 2\n```\nThat is all.")
    assert said.text == "Here you go:\na block of code of 2 lines.\nThat is all."
    assert said.leaks == (Leak("code", 2),)


def test_a_fence_nothing_ever_closes_does_not_read_the_rest_of_the_reply_out() -> None:
    """What a reply cut off mid-block leaves, and the one thing this exists to prevent."""
    said = spoken("As follows:\n```\nfirst line\nsecond line")
    assert said.text == "As follows:\na block of code of 2 lines."


def test_a_table_is_counted_in_rows_and_the_rule_under_its_heading_is_not_one() -> None:
    said = spoken("| name | count |\n| --- | --- |\n| a | 1 |\n| b | 2 |")
    assert said.text == "a table of 3 rows." and said.leaks == (Leak("table", 3),)


def test_a_sentence_with_a_pipe_in_it_is_not_a_table() -> None:
    """One row is a line with a pipe in it; a table is two of them in a row [LAW:carrying-cost]."""
    assert "a table" not in spoken("| that is the whole of it |").text


def test_a_diff_only_counts_as_one_where_it_says_so() -> None:
    """Prose starts with a dash often enough that a leading dash is no tell, and a list of three points
    would be swallowed as a patch."""
    said = spoken("diff --git a/a.py b/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\nDone.")
    assert said.text == "a diff of 4 lines.\nDone." and said.leaks == (Leak("diff", 4),)
    assert not spoken("- first point\n- second point\n- third point").leaks


def real_replies() -> list[str]:
    """Assistant text as Claude Code actually wrote it, read out of the fixture by the daemon's own reader.

    The same recognisers the tail uses, rather than a second transcript parser living in a test file
    [LAW:one-source-of-truth].
    """
    return [said.text for said in read_since(FIXTURE, None).happenings if isinstance(said, Said) and said.text.strip()]


def test_the_fixture_holds_real_replies_to_check_against() -> None:
    """[LAW:verifiable-goals] the check below is worth nothing if it runs over an empty list."""
    assert len(real_replies()) >= 4


@pytest.mark.parametrize("written", real_replies())
def test_nothing_that_cannot_be_heard_survives_a_real_reply(written: str) -> None:
    """The property the whole ticket comes down to: no backtick, no pipe table, no fence reaches speech.

    Enforced by the conversion itself rather than by the rules above it, so a shape nobody thought of is
    still stripped of its marks instead of read out one symbol at a time.
    """
    said = spoken(written).text
    assert "`" not in said
    assert "|" not in said
    assert "```" not in said
    assert not re.search(r"^\s*#{1,6}\s", said, re.MULTILINE), "a heading survived as a hash"
    assert "http://" not in said and "https://" not in said


async def test_the_filter_says_what_it_had_to_leave_out() -> None:
    """[LAW:no-silent-failure] the user hears that something was there; the log says what, and how big.

    A leak is not this filter's fault and cannot be fixed here — it means something upstream handed the
    ear what it owed the summariser — so the one useful thing to do with it is say so.
    """
    said: list[str] = []
    sink = logger.add(lambda message: said.append(message), level="WARNING")
    try:
        assert await SpokenForm().filter("Here:\n```\nx = 1\n```") == "Here:\na block of code of 1 line."
    finally:
        logger.remove(sink)
    assert any("a block of code of 1 line" in line for line in said)


async def test_an_ordinary_sentence_passes_through_the_filter_unremarked() -> None:
    said: list[str] = []
    sink = logger.add(lambda message: said.append(message), level="WARNING")
    try:
        assert await SpokenForm().filter("All twelve tests pass.") == "All twelve tests pass."
    finally:
        logger.remove(sink)
    assert not said


def test_the_pipeline_puts_the_filter_where_every_utterance_crosses_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The conversion is worth nothing unless it is installed, and there is exactly one place it goes.

    The real service loads model weights, which is not a unit test's business; what is this repository's
    business is that the filter is handed to it at all [LAW:single-enforcer].
    """
    from hands.voice import pipeline as built

    given: dict[str, object] = {}

    from pipecat.processors.frame_processor import FrameProcessor

    class Recorded(FrameProcessor):
        """Enough of a processor for the pipeline to link, and nothing that needs model weights."""

        Settings = built.PocketTTSService.Settings

        def __init__(self, **kwargs: object) -> None:
            given.update(kwargs)
            super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)

    monkeypatch.setattr(built, "PocketTTSService", Recorded)
    built.build_voice(
        built.VoiceConfig(
            llm=built.OpenAICompatibleBackend(base_url="http://example/v1", model="m"),
            whisper_model="mlx-community/whisper-tiny",
            voice="cosette",
        ),
        tools=[],
    )
    filters = given["text_filters"]
    assert isinstance(filters, list) and [type(one) for one in cast(list[object], filters)] == [SpokenForm]
