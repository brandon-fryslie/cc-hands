"""Spoken form: nothing reaches text-to-speech as written.

The table below is the specification. Each row is a shape that turns up in real Claude replies and in the
summaries hands makes of them, written the way it arrives and the way it has to be heard.
"""

import re
from pathlib import Path
from typing import cast

import pytest
from loguru import logger

from hands.core.spoken import Leak, spoken, spoken_ref
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
    # An absolute path is the form Claude Code's own tools require, so it is the form most replies carry.
    ("Edited /Users/bmf/code/cc-hands/src/hands/core/turn.py today.", "Edited turn today."),
    ("Read /etc/hosts for that.", "Read hosts for that."),
    # A name is a name whatever word stands in front of it. The trigger word makes an id of what follows
    # only where the tell is there too, and the case-insensitive flag, set on the whole rule, reached into
    # the tell and cancelled it: the same name was dropped after "token" and spoken anywhere else.
    ("The token refresh_token_v2 expired.", "The token refresh token v2 expired."),
    ("It runs update_database_2 nightly.", "It runs update database 2 nightly."),
    # A constant is shouted, not opaque: one case present is no tell at all, and an id needs both.
    ("Set HTTP_TIMEOUT_30 in the config.", "Set HTTP TIMEOUT 30 in the config."),
    ("Use MAX_RETRIES_5 instead.", "Use MAX RETRIES 5 instead."),
    # "run" is a verb before it is a label, and as a trigger word it deleted the object of the sentence.
    ("I will run update_Database_2 tomorrow.", "I will run update Database 2 tomorrow."),
    # An address written inside angle brackets takes them with it rather than leaving them to be swept up.
    ("Check <https://x.com> now.", "Check a link now."),
    # A bracket inside the address is part of it; stopping at the first one read the rest of it aloud.
    ("[doc](https://x.com/a_(b)) rest", "doc rest"),
    ("It is ~~struck~~ out.", "It is struck out."),
    ("> quoted line", "quoted line"),
    # A full stop with no space after it is a sentence, not a file: `go`, `c`, `h` and `sh` are all words.
    ("Ran it on node.js and Deno.Go figure.", "Ran it on node and Deno.Go figure."),
    # A module named rather than called: the dots are read out one by one just the same.
    ("See hands.core.spoken for details.", "See hands core spoken for details."),
    # The period ends the sentence, and taking it with the address runs the next sentence into this one.
    ("See https://docs.pipecat.ai/guide. It explains the rest.", "See a link. It explains the rest."),
]


# Sentences with no tell in them anywhere, which must come back exactly as they went in. This list is what
# holds the module to the law it states about itself: every rule asks for a tell that ordinary English does
# not have. Each line below is a shape that once had a rule fire on it, and each lost a word when it did
# [LAW:carrying-cost] — a promise in a docstring is a map nobody redraws, so it is kept here instead.
PROSE = [
    "Read and/or write the file.",
    "The input/output split matters.",
    "It runs 24/7 on the box.",
    "Shipped on 12/25/2025 as planned.",
    "The ratio is 1/2 of the total.",
    "The file is 1048576 bytes long.",
    "There are 1234567 records.",
    "It calls base64 encode on the body.",
    "It defaced the output, then passed.",
    # Four marks that say something in an ordinary sentence. Dropped for being markdown, each left a
    # grammatical sentence stating a different fact than the one that was written.
    "Latency is now < 200 ms.",
    "Coverage is > 80 percent.",
    "It removed ~500 lines.",
    "The area is 3 * 4 metres.",
    # A class name is long, numbered and mixed case, which is every tell an id has except its length.
    "It uses Base64Decoder now.",
    "Switched to Float32Array for speed.",
    "Added an OAuth2TokenStore class.",
    "The HTTP2Handler is done.",
    # A year at the start of a line is a sentence, not the number of a list item.
    "2024. It was a good year.\n1999. It was better.",
    "The well-known issue is that three tests fail. Should it fix them?",
]


@pytest.mark.parametrize(("written", "said"), WRITTEN_AND_SPOKEN)
def test_what_is_written_and_what_is_heard(written: str, said: str) -> None:
    assert spoken(written).text == said


@pytest.mark.parametrize("written", PROSE)
def test_a_sentence_with_no_tell_in_it_comes_back_whole(written: str) -> None:
    """The cost side of every rule above, and the reason each one asks for a tell.

    A rule with no tell does not merely fail to help: it deletes a word out of the middle of a sentence and
    leaves it grammatical, so nothing downstream can notice. "and/or" became "or", and "1048576 bytes"
    became "a commit bytes".
    """
    assert spoken(written).text == written


@pytest.mark.parametrize("written", ["base64_encode", "sha256_checksum", "parse_utf8_header"])
def test_a_long_name_made_of_words_is_said_as_its_words_and_not_dropped_as_an_id(written: str) -> None:
    """Long, and carrying a digit, and still a name: the tell an id has that these do not is mixed case."""
    assert spoken(f"It calls {written} on the body.").text == f"It calls {written.replace('_', ' ')} on the body."


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


def test_an_inline_span_at_the_start_of_a_line_does_not_open_a_fence() -> None:
    """A backtick fence may carry no backtick in its info string, so this is a sentence, not an opening.

    Read as an opening it swallowed every line after it to the end of the reply, which is the whole of what
    the listener would have heard [LAW:parse-dont-validate].
    """
    said = spoken("```bash``` is what I ran.\nThen everything else in this reply.\nAnd more.")
    assert said.text == "bash is what I ran.\nThen everything else in this reply.\nAnd more."
    assert not said.leaks


def test_a_fence_is_closed_only_by_one_at_least_as_long_as_itself() -> None:
    """Quoting a three-backtick block inside a four-backtick one is how a model shows markup.

    Closed by length-blind matching, the outer block ended at the inner opening and the quoted code was
    read out loud — the one thing this module exists to prevent.
    """
    said = spoken("Here is a nested fence:\n````\nouter\n```\ninner\n```\nouter again\n````\nDone.")
    assert said.text == "Here is a nested fence:\na block of code of 5 lines.\nDone."
    assert "inner" not in said.text and said.leaks == (Leak("code", 5),)


def test_a_list_keeps_counting_across_a_wrapped_item_and_a_blank_line() -> None:
    """The two shapes a model actually writes a list in, both of which broke the count.

    A wrapped item ended the run, so the item after it was announced to the listener as "First". A blank
    line between items ended it too, leaving a string of single items that `_counted` declines to number
    at all — no counting, on the shape this function most exists for.
    """
    wrapped = spoken("- this is a long item\n  that continues here\n- second item\n- third item").text
    assert wrapped == "First, this is a long item that continues here. Second, second item. Third, third item."
    assert spoken("- a\n\n- b\n\n- c").text == "First, a. Second, b. Third, c."


@pytest.mark.parametrize("written", ["```python\n```", "|---|---|\n|:-:|:-:|"])
def test_a_block_that_held_nothing_is_not_reported_as_a_leak(written: str) -> None:
    """Nothing was there, so there is nothing to say and nothing to warn about.

    "a block of code of 0 lines" tells the listener something was there when nothing was, and a closing
    fence arriving alone is the known shape of the streamed path — so it would warn on every one.
    """
    said = spoken(written)
    assert said.text == "" and said.leaks == ()


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


def test_a_list_written_under_a_hunk_is_spoken_rather_than_swallowed_by_it() -> None:
    """A hunk says how many lines it covers, so the diff ends where it says and the bullets survive.

    Every line of a bulleted list opens with a dash, which is also how a removal opens. Read by shape the
    whole list went into the patch and the listener heard a line count in place of the reply.
    """
    said = spoken("@@ -1 +1 @@\n-x = 1\n+x = 2\n- I fixed the bug\n- I ran the suite\nDone.")
    assert said.leaks == (Leak("diff", 3),)
    assert said.text == "a diff of 3 lines.\nFirst, I fixed the bug. Second, I ran the suite.\nDone."


def test_a_hunk_that_counts_more_than_one_line_a_side_is_followed_to_its_end() -> None:
    said = spoken("@@ -1,2 +1,3 @@\n context\n-gone\n+added\n+also\nAfter.")
    assert said.leaks == (Leak("diff", 5),) and said.text == "a diff of 5 lines.\nAfter."


def test_a_bulleted_question_keeps_its_question_mark_and_gains_no_full_stop() -> None:
    assert spoken("- Should it fix them?\n- Yes, do it").text == "First, Should it fix them? Second, Yes, do it."


def test_a_blank_line_that_ended_a_diff_is_not_counted_as_part_of_it() -> None:
    """The count is the only thing the listener is given about a block they will never hear."""
    said = spoken("diff --git a/a.py b/a.py\n@@ -1 +1 @@\n-x = 1\n\nDone.")
    assert said.leaks == (Leak("diff", 3),) and said.text == "a diff of 3 lines.\nDone."


@pytest.mark.parametrize("written", ["Done.\n\n---\n\nNext up.", "Done.\n***\nNext up.", "Heading\n=======\nbody"])
def test_a_line_that_is_only_marks_is_drawn_rather_than_said(written: str) -> None:
    """A rule across the page and the dashes under a heading are for the eye; there is nothing to read."""
    said = spoken(written).text
    assert "-" not in said and "=" not in said and said.count("\n") == 1


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


def test_a_ref_is_said_by_the_caller_that_knows_it_is_one() -> None:
    """`_PATH` will not read a bare `feature/narration-tree` as a path, and must not: a rule loose enough to
    catch it also catches "and/or", "24/7" and "input/output", and costs every one of them a word. So a ref is
    said where its type already says what it is, and every word of it is kept — a branch is named so it can be
    told from the others, and "release/1.2" heard as "1.2" is the one thing it is not."""
    assert spoken_ref("feature/narration-tree") == "feature narration tree"
    assert spoken_ref("release/1.2") == "release 1.2"
    assert spoken_ref("fix_the_thing") == "fix the thing"
    assert spoken_ref("main") == "main"


def test_a_ref_said_that_way_is_left_alone_by_the_filter_in_front_of_the_speaker() -> None:
    said = spoken_ref("feature/narration-tree")
    heard = spoken(f"It pushed {said}.")
    assert heard.text == f"It pushed {said}." and heard.leaks == ()


def test_prose_the_ref_rule_would_have_eaten_is_never_shown_to_it() -> None:
    """The filter still sees these, and still leaves them whole, which is the reason `spoken_ref` is separate."""
    for untouched in ("It handled input/output.", "It ran the job 24/7.", "It checked and/or fixed it."):
        assert spoken(untouched).text == untouched
