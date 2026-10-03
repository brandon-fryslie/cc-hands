"""A turn's narration tree: what plays when it stops, what is left to open, and what each part was cut from."""

from hands.core.delta import Branched, Changed, Commit, Committed, Delta, PullRequested, Pushed
from hands.core.narration import Narration, narration, opened
from hands.core.spoken import spoken
from hands.core.turn import (
    Asked,
    Budget,
    Delegated,
    Edited,
    Interruption,
    Looked,
    Other,
    Planned,
    Question,
    Questioned,
    Ran,
    Ref,
    Said,
    Tested,
    Turn,
)

ROOMY = Budget(opening=10_000, said=10_000, input=10_000, result=10_000, steps=100, files=100, commits=100, changes=10_000)


def told(*steps: object, delta: Delta = Delta()) -> Narration:
    return narration(Turn(Asked(None, "fix the test"), tuple(steps)), delta)  # pyright: ignore[reportArgumentType]


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
    assert change.text == "The change: two edits."


def test_one_of_a_kind_is_counted_in_the_singular() -> None:
    [change] = told(Edited(None, "/a/b.py", False, "@@")).sections
    assert change.text == "The change: one edit."


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


def test_an_unanswered_question_is_a_segment_of_its_own_and_never_falls_into_a_section() -> None:
    asked = Questioned(Ref("u3"), (Question("Roll the rename back?", ("roll back", "press on"), None),))
    narrated = told(Said(None, "Two tests fail."), asked)
    assert topics(narrated) == ["what it said"]
    [question] = narrated.questions
    assert question.text == "It is asking: Roll the rename back? Either roll back or press on."
    assert question.refs == (Ref("u3"),)


def test_a_question_already_answered_is_there_to_open_and_is_not_asked_again() -> None:
    asked = Questioned(None, (Question("Roll the rename back?", (), "press on"),))
    narrated = told(asked)
    assert narrated.questions == ()
    assert [answer.text for answer in narrated.settled] == ["It asked: Roll the rename back? You chose press on."]


def test_every_question_the_turn_waits_on_is_in_its_one_question_segment() -> None:
    asked = Questioned(None, (Question("A?", (), None), Question("B?", (), None)))
    assert [question.text for question in told(asked).questions] == ["It is asking: A? It is asking: B?"]


def test_a_question_is_said_in_claudes_words_put_in_spoken_form() -> None:
    """A turn waiting on an answer it never asks for is the failure the question segment exists to prevent."""
    asked = Questioned(None, (Question("Roll `src/auth.py` back to a1b2c3d?", ("Roll back (Recommended)", "Press on"), None),))
    said = told(Edited(None, "/a/b.py", False, "@@"), asked).asked()
    heard = spoken(said)
    assert heard.text == said and heard.leaks == ()
    assert said.startswith("It is asking: Roll ") and said.endswith(" Either Roll back or Press on.")
    assert not any(written in said for written in ("src/", "a1b2c3d", "Recommended", "`"))


def test_a_question_asked_in_prose_plays_though_no_hook_blocked_on_it() -> None:
    closing = Said(Ref("u4"), "Fixed the refresh test.\n\nWant me to look at the other flaky ones too?")
    narrated = told(Ran(None, "pytest", None, False, "ok", ()), closing)
    assert narrated.asked() == "It said: Want me to look at the other flaky ones too?"
    assert [question.refs for question in narrated.questions] == [(Ref("u4"),)]


def test_an_offer_with_no_question_mark_is_said_in_claudes_words() -> None:
    closing = Said(None, "The dev build should not be on that Mac. Say the word and I'll remove it.")
    assert told(closing).asked() == "It said: Say the word and I'll remove it."


def test_a_turn_waiting_on_two_things_says_each() -> None:
    dialog = Questioned(None, (Question("Merge now?", ("Merge", "Wait"), None), Question("Re-run the review?", (), None)))
    said = told(Ran(None, "pytest", None, False, "ok", ()), dialog).asked()
    assert said == "It is asking: Merge now? Either Merge or Wait. It is asking: Re-run the review?"


def test_a_dialog_claude_went_on_past_is_there_to_open_and_is_not_asked() -> None:
    dialog = Questioned(None, (Question("Merge now?", ("Merge", "Wait"), None),))
    narrated = told(dialog, Said(None, "Merged, and the suites pass."))
    assert narrated.questions == ()
    assert [settled.text for settled in narrated.settled] == ["It asked: Merge now? It went on without an answer."]


def test_a_choice_split_over_two_sentences_is_said_as_the_one_thing_it_is() -> None:
    said = told(Said(None, "The rename is in. Should I update the logout code?\nOr roll it back?")).asked()
    assert said == "It said: Should I update the logout code? Or roll it back?"


def test_a_dialog_escaped_is_still_waiting_and_one_claude_went_on_past_is_not() -> None:
    dialog = Questioned(None, (Question("Merge now?", ("Merge", "Wait"), None),))
    assert [question.text for question in told(dialog, Interruption(None)).questions] == ["It is asking: Merge now? Either Merge or Wait."]
    assert told(dialog, Said(None, "Going with a merge, then."), Said(None, "Merged, and the suites pass.")).questions == ()


def test_a_turn_that_asked_nothing_has_nothing_to_say_of_a_question() -> None:
    narrated = told(Said(None, "Fixed it. All twelve tests pass."))
    assert narrated.asked() == "" and narrated.questions == ()


def test_a_question_the_turn_went_on_working_past_is_not_waiting() -> None:
    narrated = told(Said(None, "Want me to run the tests?"), Ran(None, "pytest", None, False, "ok", ()))
    assert narrated.questions == () and narrated.asked() == ""


def test_a_question_cut_off_by_an_interruption_is_not_waiting() -> None:
    narrated = told(Said(None, "Want me to run the tests?"), Interruption(None))
    assert narrated.questions == ()


def test_what_the_repository_did_is_said_from_the_types_and_never_from_the_model() -> None:
    """A summariser was measured both dropping the commit and reading its hash out; said from here it can do neither."""
    delta = Delta(files=(Changed("/a/b.py", 3, 1), Changed("/a/c.py", 0, 9)), commits=(Commit("f0f9776", "tidy"),))
    narrated = told(Ran(None, "make fmt", None, False, "", ()), delta=delta)
    [repository] = narrated.repository
    assert repository.text == "It committed and left two files different."
    assert narrated.facts() == "It committed and left two files different."


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


def test_a_push_no_step_recorded_is_said_as_a_push_and_not_only_as_a_commit() -> None:
    """`git commit -m x && git push` carries no operation; the delta read the push off the remote-tracking ref."""
    chained = Ran(None, "git commit -qm x && git push -q", None, False, "", ())
    delta = Delta(commits=(Commit("f0f9776", "x"),), changes=(Pushed("main"),))
    [repository] = told(chained, delta=delta).repository
    assert repository.text == "It committed and pushed main."


def test_a_branch_no_step_recorded_is_said() -> None:
    chained = Ran(None, "git checkout -b feature/narration && pytest", None, False, "", ())
    [repository] = told(chained, delta=Delta(changes=(Branched("feature/narration", "created branch"),))).repository
    assert repository.text == "It created branch feature narration."


def test_a_pull_request_no_step_recorded_is_said_without_its_number() -> None:
    script = Ran(None, "./ship.sh", None, False, "", ())
    delta = Delta(changes=(Pushed("fix"), PullRequested(7, "https://x/7", "created")))
    [repository] = told(script, delta=delta).repository
    assert repository.text == "It pushed fix and created a pull request."


def test_a_push_and_a_pull_request_both_sources_saw_are_said_once() -> None:
    recorded = Ran(None, "git push && gh pr create", None, False, "", (Pushed("fix"), PullRequested(7, "https://x/7", "created")))
    delta = Delta(changes=(Pushed("fix"), PullRequested(7, "https://x/7", "created")))
    [repository] = told(recorded, delta=delta).repository
    assert repository.text == "It pushed fix and created a pull request."


def test_a_pull_request_is_named_without_its_number() -> None:
    opened_pr = Ran(None, "gh pr create", None, False, "", (PullRequested(104, "https://x/104", "created"),))
    [repository] = told(opened_pr).repository
    assert repository.text == "It created a pull request."


def test_a_repository_that_did_not_move_is_said_nothing_about_because_a_listener_is_told_what_happened() -> None:
    narrated = told(Said(None, "Thought about it."))
    assert narrated.repository == () and narrated.facts() == ""


def test_a_delta_that_holds_only_a_patch_still_says_the_repository_moved() -> None:
    narrated = told(Said(None, "Done."), delta=Delta(patch="@@ -1 +1 @@"))
    assert narrated.repository[0].text == "The repository moved."


def test_a_turn_with_no_steps_at_all_has_nothing_to_say_and_no_section() -> None:
    narrated = told()
    assert narrated.facts() == narrated.asked() == "" and narrated.sections == ()


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
        Tested(None, "pytest", 12, 0, (), ""),
        Looked(None, "Read", "/a/b.py", "x"),
        Planned(None, "write it", "completed"),
        Delegated(None, "Explore", "find the thing", "found it"),
        Other(None, "Skill", "{}", "loaded", False),
    )
    narrated = told(*steps)
    assert sum(len(section.covers) for section in narrated.sections) == len(steps)


def test_a_branch_is_said_in_words_because_the_speaker_reads_a_slash_aloud() -> None:
    """This clause is the one part of the top level no model wrote, so the instruction cannot cover it, and the
    filter in front of the speaker will not read a bare `feature/x` as a path — by its own deliberate choice,
    because a rule loose enough to catch it also eats "and/or" and "24/7"."""
    pushed = Ran(None, "git push", None, False, "", (Pushed("feature/narration-tree"),))
    [repository] = told(pushed).repository
    assert repository.text == "It pushed feature narration tree."
    # Every word is kept: a branch is named so it can be told from the others.
    moved = Ran(None, "git switch", None, False, "", (Branched("release/1.2", "moved to"),))
    [where] = told(moved).repository
    assert where.text == "It moved to release 1.2."


def test_nothing_the_repository_clause_says_is_rewritten_by_the_filter_in_front_of_the_speaker() -> None:
    """[LAW:one-source-of-truth] `core.spoken` is what code-shaped means here, so it is what judges this."""
    pushed = Ran(None, "git push", None, False, "", (Pushed("feature/narration-tree"), Branched("release/1.2", "moved to")))
    [repository] = told(pushed, delta=Delta(commits=(Commit("f0f9776", "tidy"),))).repository
    heard = spoken(repository.text)
    assert heard.text == repository.text and heard.leaks == ()


def test_a_turn_the_user_interrupted_says_so_first_from_the_types_and_never_as_a_section() -> None:
    narrated = told(Said(None, "# Rivers"), Interruption(Ref("u9")), delta=Delta(files=(Changed("a.md", 3, 0),)))
    assert narrated.facts() == "You interrupted it. It left one file different."
    assert topics(narrated) == ["what it said"]
    assert [segment.refs for segment in narrated.interrupted] == [(Ref("u9"),)]


def test_a_turn_that_finished_says_nothing_of_an_interruption() -> None:
    assert told(Said(None, "Done.")).interrupted == ()


def test_a_turn_that_went_on_after_an_interruption_is_not_said_to_be_interrupted() -> None:
    """A message queued while a tool ran cuts the tool off, and the turn carries on with it."""
    assert told(Interruption(Ref("u9")), Said(None, "Done.")).interrupted == ()


def test_a_turn_with_two_interruptions_says_so_once() -> None:
    assert told(Interruption(Ref("u8")), Said(None, "On it."), Interruption(Ref("u9"))).facts() == "You interrupted it."


def test_a_turn_with_nothing_to_report_is_the_interruption_alone() -> None:
    assert told(Interruption(Ref("u9"))).facts() == "You interrupted it."
