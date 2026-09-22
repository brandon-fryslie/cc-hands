"""A turn's narration tree: what plays when it stops, what is left to open, and what each part was cut from."""

from hands.core.delta import Changed, Commit, Delta
from hands.core.narration import Narration, narration, opened
from hands.core.turn import (
    Asked,
    Budget,
    Committed,
    Delegated,
    Edited,
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
)

ROOMY = Budget(opening=10_000, said=10_000, input=10_000, result=10_000, steps=100, files=100, commits=100, changes=10_000)


def told(*steps: object, delta: Delta = Delta(), headline: str = "It did the thing.", sentences: int = 3) -> Narration:
    """Roomy by default, so a case about the tree is not also a case about the length."""
    return narration(headline, Turn(Asked(None, "fix the test"), tuple(steps)), delta, sentences)  # pyright: ignore[reportArgumentType]


def topics(narrated: Narration) -> list[str]:
    return [section.topic.name for section in narrated.sections]


def test_each_kind_of_step_is_its_own_section_in_the_order_the_kinds_first_appear() -> None:
    narrated = told(
        Said(None, "Looking."),
        Looked(None, "Read", "/a/b.py", "two lines"),
        Edited(None, "/a/b.py", False, "@@"),
        Looked(None, "Grep", "retry", "one hit"),
    )
    assert topics(narrated) == ["what it said", "what it read", "the change"]


def test_a_section_counts_what_it_holds_rather_than_reading_any_of_it() -> None:
    narrated = told(Edited(None, "/a/b.py", False, "@@"), Edited(None, "/a/c.py", True, "print(1)"))
    [change] = narrated.sections
    assert change.text == "The change: 2 edits."


def test_one_of_a_kind_is_counted_in_the_singular() -> None:
    [change] = told(Edited(None, "/a/b.py", False, "@@")).sections
    assert change.text == "The change: 1 edit."


def test_a_command_that_moved_the_repository_is_the_commit_and_one_that_did_not_is_a_command() -> None:
    narrated = told(
        Ran(None, "pytest", None, False, "ok", ()),
        Ran(None, "git push", None, False, "", (Committed("f0f9776", "committed"), Pushed("main"))),
    )
    assert topics(narrated) == ["the commands", "the commit"]


def test_a_segment_names_the_records_it_holds_and_only_those() -> None:
    edited = Edited(Ref("u2"), "/a/b.py", False, "@@")
    narrated = told(Said(Ref("u1"), "Looking."), edited, Said(None, "Done."))
    said, change = narrated.sections
    # The closing reply the Stop hook stands in with has no record yet, so it contributes no ref and is not invented.
    assert said.refs == (Ref("u1"),)
    assert change.refs == (Ref("u2"),)


def test_the_headline_covers_the_whole_turn_including_what_opened_it() -> None:
    narrated = told(Said(Ref("u2"), "Done."))
    assert narrated.headline.refs == (Ref("u2"),)
    assert narrated.headline.covers == (Asked(None, "fix the test"), Said(Ref("u2"), "Done."))


def test_a_question_is_a_segment_of_its_own_and_never_falls_into_a_section() -> None:
    asked = Questioned(Ref("u3"), (Question("Roll the rename back?", ("roll back", "press on"), None),))
    narrated = told(Said(None, "Two tests fail."), asked)
    assert topics(narrated) == ["what it said"]
    [question] = narrated.questions
    assert question.text == "It is asking: Roll the rename back? Either roll back and press on."
    assert question.refs == (Ref("u3"),)


def test_a_question_already_answered_says_what_was_chosen_rather_than_asking_it_again() -> None:
    asked = Questioned(None, (Question("Roll the rename back?", (), "press on"),))
    [question] = told(asked).questions
    assert question.text == "It asked: Roll the rename back? You chose press on."


def test_every_question_of_one_step_gets_its_own_segment() -> None:
    asked = Questioned(None, (Question("A?", (), None), Question("B?", (), None)))
    assert [question.text for question in told(asked).questions] == ["It is asking: A?", "It is asking: B?"]


def test_what_plays_is_the_headline_and_neither_a_section_nor_a_question_read_out_as_written() -> None:
    """A question segment holds the words Claude typed at a screen — here a path and a hash — and the headline
    has already been asked to end on the turn's question. Playing both says it twice, once verbatim."""
    asked = Questioned(None, (Question("Roll src/auth.py back to a1b2c3d?", (), None),))
    narrated = told(Edited(None, "/a/b.py", False, "@@"), asked, headline="The rename is in. Should it roll the auth code back?")
    assert narrated.said() == "The rename is in. Should it roll the auth code back?"
    assert [question.text for question in narrated.questions] == ["It is asking: Roll src/auth.py back to a1b2c3d?"]


def test_what_the_repository_did_is_said_from_the_types_and_never_from_the_model() -> None:
    """The summariser was measured both dropping the commit and reading its hash out; said from here it can do
    neither, which is why the instruction says nothing about commits at all."""
    delta = Delta(files=(Changed("/a/b.py", 3, 1), Changed("/a/c.py", 0, 9)), commits=(Commit("f0f9776", "tidy"),))
    narrated = told(Ran(None, "make fmt", None, False, "", ()), delta=delta, headline="It reformatted the tree.")
    [repository] = narrated.repository
    assert repository.text == "It committed and left 2 files different."
    assert narrated.said() == "It reformatted the tree. It committed and left 2 files different."


def test_a_commit_both_the_step_and_the_delta_saw_is_said_once() -> None:
    delta = Delta(commits=(Commit("f0f9776", "tidy"),))
    pushed = Ran(None, "git push", None, False, "", (Committed("f0f9776", "committed"), Pushed("main")))
    [repository] = told(pushed, delta=delta).repository
    assert repository.text == "It committed and pushed main."


def test_a_commit_no_step_recorded_is_still_said_because_the_delta_read_it() -> None:
    """A `git commit` inside a compound command carries no operation for a step to hold, and the delta read
    against the turn's start names it anyway."""
    heredoc = Ran(None, "python3 - <<'EOF'\n...\nEOF", None, False, "db9618c sessions: medium review", ())
    [repository] = told(heredoc, delta=Delta(commits=(Commit("db9618c", "sessions"),))).repository
    assert repository.text == "It committed."


def test_a_pull_request_is_named_without_its_number() -> None:
    opened_pr = Ran(None, "gh pr create", None, False, "", (PullRequested(104, "https://x/104", "created"),))
    [repository] = told(opened_pr).repository
    assert repository.text == "It created a pull request."


def test_a_repository_that_did_not_move_is_said_nothing_about_because_a_listener_is_told_what_happened() -> None:
    narrated = told(Said(None, "Thought about it."), headline="It thought about it.")
    assert narrated.repository == () and narrated.said() == "It thought about it."


def test_a_delta_that_holds_only_a_patch_still_says_the_repository_moved() -> None:
    narrated = told(Said(None, "Done."), delta=Delta(patch="@@ -1 +1 @@"))
    assert narrated.repository[0].text == "The repository moved."


def test_a_turn_with_no_steps_at_all_still_has_a_headline_to_play() -> None:
    narrated = told(headline="It had nothing to do.")
    assert narrated.said() == "It had nothing to do." and narrated.sections == ()


def test_opening_a_section_renders_exactly_its_own_steps_by_the_rules_the_whole_turn_got() -> None:
    narrated = told(Said(None, "Looking."), Ran(None, "pytest", None, True, "1 failed", ()))
    _said, commands = narrated.sections
    assert opened(commands, ROOMY) == "Claude ran pytest\nOutput (exit code not zero): 1 failed"


def test_opening_the_repository_shows_what_git_said_because_it_holds_no_step_to_show() -> None:
    delta = Delta(files=(Changed("/a/b.py", 3, 1),), patch="@@ -1 +1 @@")
    [repository] = told(Said(None, "Done."), delta=delta).repository
    assert "The repository is different" in opened(repository, ROOMY)
    assert "What changed:\n@@ -1 +1 @@" in opened(repository, ROOMY)


def test_every_kind_of_step_lands_in_some_section_so_nothing_a_turn_did_is_dropped() -> None:
    """A step with no section would be a result no listener could ever reach [LAW:no-silent-failure]."""
    steps = (
        Said(None, "a"),
        Edited(None, "/a/b.py", False, "@@"),
        Ran(None, "ls", None, False, "", ()),
        Tested(None, "pytest", 12, 0, ()),
        Looked(None, "Read", "/a/b.py", "x"),
        Planned(None, "write it", "completed"),
        Delegated(None, "Explore", "find the thing", "found it"),
        Other(None, "Skill", "{}", "loaded", False),
    )
    narrated = told(*steps)
    assert sum(len(section.covers) for section in narrated.sections) == len(steps)


def test_the_headline_is_cut_to_its_number_of_sentences_rather_than_asked_for_it() -> None:
    """A length held only by an instruction is obeyed or not and checked by nobody, which is how a model asked
    for one sentence was measured writing two."""
    said = "The rename is in. Three tests still fail. The docs are updated."
    assert told(Said(None, "x"), headline=said, sentences=1).headline.text == "The rename is in."
    assert told(Said(None, "x"), headline=said, sentences=2).headline.text == "The rename is in. Three tests still fail."


def test_a_closing_question_survives_the_cut_at_every_length() -> None:
    """A turn waiting on an answer that never asks is worse than a long one."""
    said = "The rename is in. Three tests still fail. Should it roll the rename back?"
    assert told(Said(None, "x"), headline=said, sentences=1).headline.text == "The rename is in. Should it roll the rename back?"


def test_a_report_that_is_only_a_question_is_still_asked() -> None:
    assert told(Said(None, "x"), headline="Want it to carry on?", sentences=1).headline.text == "Want it to carry on?"


def test_the_question_is_said_last_whatever_the_summariser_put_where() -> None:
    """A fact read out after the question leaves the listener holding the answer to something already gone by."""
    pushed = Ran(None, "git push", None, False, "", (Pushed("main"),))
    narrated = told(pushed, headline="The entry is gone. Want it to carry on?", sentences=1)
    assert narrated.said() == "The entry is gone. It pushed main. Want it to carry on?"
