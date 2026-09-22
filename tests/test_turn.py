"""A turn, rendered for the summariser: every kind of step says what it did, and each part is cut to its budget."""

from hands.core.delta import Changed, Commit, Delta
from hands.core.turn import (
    CUT,
    Asked,
    Branched,
    Budget,
    Committed,
    Continuing,
    Delegated,
    Edited,
    Looked,
    Notified,
    Other,
    Planned,
    PullRequested,
    Pushed,
    Question,
    Questioned,
    Ran,
    Said,
    Tested,
    Turn,
    render,
)

ROOMY = Budget(opening=10_000, said=10_000, input=10_000, result=10_000, steps=100, files=100, commits=100, changes=10_000)


def rendered(*steps: object, budget: Budget = ROOMY) -> str:
    """Everything after the opening, which every case below shares."""
    turn = Turn(Asked(None, "fix the test"), tuple(steps))  # pyright: ignore[reportArgumentType]
    return render(turn, Delta(), budget).removeprefix("The user asked:\nfix the test\n\n")


def test_a_notification_is_rendered_as_what_reported_rather_than_as_something_the_user_asked() -> None:
    assert render(Turn(Notified(None, "<task-notification>tests passed</task-notification>"), ()), Delta(), ROOMY).startswith("A background task reported:\n")


def test_text_a_command_and_a_tool_nobody_named_each_say_what_came_of_them() -> None:
    assert rendered(
        Said(None, "Looking."),
        Ran(None, "pytest", "Run the tests", True, "1 failed", ()),
        Other(None, "Skill", "{}", "loaded", False),
    ) == ("Claude said:\nLooking.\n\nClaude ran pytest (Run the tests)\nOutput (exit code not zero): 1 failed\n\nClaude used Skill: {}\nResult: loaded")


def test_a_command_says_everything_it_did_to_the_repository_because_one_command_can_do_several() -> None:
    git = (Committed("f0f9776", "committed"), Pushed("main"), Branched("origin/master", "rebased"), PullRequested(104, "https://x/104", "created"))
    assert rendered(Ran(None, "git push", None, False, "", git)) == (
        "Claude ran git push\nOutput: \n"
        "It committed f0f9776.\n"
        "It pushed main.\n"
        "It rebased origin/master.\n"
        "It created pull request 104, https://x/104."
    )


def test_an_edit_shows_its_hunks_and_a_new_file_shows_what_it_now_holds() -> None:
    assert rendered(Edited(None, "/a/b.py", False, "@@ -1,2 +1,2 @@\n-old\n+new")) == "Claude edited /a/b.py:\n@@ -1,2 +1,2 @@\n-old\n+new"
    assert rendered(Edited(None, "/a/new.py", True, "hello")) == "Claude wrote /a/new.py:\nhello"


def test_a_test_run_is_its_counts_and_the_names_that_failed_rather_than_its_scrollback() -> None:
    assert rendered(Tested(None, "pytest", 299, 2, ("test_a", "test_b"))) == "Claude ran the pytest tests: 2 failed, 299 passed\n  test_a\n  test_b"
    # A runner that counts nothing it did not fail says only what failed.
    assert rendered(Tested(None, "go", None, 1, ("TestX",))) == "Claude ran the go tests: 1 failed\n  TestX"


def test_a_look_a_task_and_a_subagent_each_read_as_what_they_are() -> None:
    assert rendered(Looked(None, "Grep", "def main", "a.py:3")) == "Claude used Grep on def main\nFound: a.py:3"
    assert rendered(Planned(None, "Rewrite the tail", "in_progress")) == "Claude's plan: Rewrite the tail is in_progress"
    assert rendered(Delegated(None, "Explore", "Find the parser", None)) == "Claude gave the Explore subagent this job: Find the parser\nIt is still working."
    assert rendered(Delegated(None, None, "Find the parser", "It is in tail.py.")) == (
        "Claude gave a subagent this job: Find the parser\nIt reported: It is in tail.py."
    )


def test_a_question_reads_as_asked_with_what_was_offered_and_what_was_chosen() -> None:
    answered = Question("Which one?", ("This", "That"), "That")
    unanswered = Question("And this?", (), None)
    assert rendered(Questioned(None, (answered, unanswered))) == (
        "Claude asked the user: Which one?\nOptions: This, That\nThe user chose: That\n"
        "Claude asked the user: And this?\nUnanswered."
    )


def test_a_suite_that_failed_whole_has_its_names_cut_to_budget_like_every_other_part() -> None:
    """Hundreds of failing names is exactly what a bad refactor prints, and exactly when one step could fill
    the whole prompt of a model asked for two sentences."""
    failing = tuple(f"test_{n}" for n in range(200))
    assert rendered(Tested(None, "pytest", 0, 200, failing), budget=Budget(opening=100, said=100, input=100, result=30, steps=10, files=10, commits=10, changes=100)) == (
        "Claude ran the pytest tests: 200 failed, 0 passed\n  test_0\n  test_1\n  test_2\n  test_3\n  te" + CUT
    )


def test_a_long_turn_keeps_how_it_started_and_how_it_ended_and_each_part_is_cut_to_its_budget() -> None:
    steps = tuple(Said(None, f"step {n}") for n in range(10))
    rendered_turn = render(Turn(Asked(None, "x" * 50), steps), Delta(), Budget(opening=10, said=100, input=100, result=100, steps=4, files=10, commits=10, changes=100))
    assert rendered_turn == "\n\n".join(
        [
            "The user asked:\n" + "x" * 10 + CUT,
            "Claude said:\nstep 0",
            "Claude said:\nstep 1",
            "(6 steps in the middle are left out)",
            "Claude said:\nstep 8",
            "Claude said:\nstep 9",
        ]
    )


def test_what_the_repository_says_is_told_after_the_steps_and_named_file_by_file() -> None:
    """A turn's result is not only what its steps report: a `sed` names no file, and a commit no step made."""
    delta = Delta(
        files=(Changed("src/a.py", 12, 9), Changed("logo.png", None, None)),
        commits=(Commit("abc1234", "tidy up"),),
        patch="@@ -1 +1 @@\n-x = 1\n+x = 2\n",
    )
    told = render(Turn(Asked(None, "tidy up"), (Said(None, "Done."),)), delta, ROOMY)
    assert told == "\n\n".join(
        [
            "The user asked:\ntidy up",
            "Claude said:\nDone.",
            "The repository is different, whether or not a step above says so:\n  src/a.py +12 -9\n  logo.png (binary)",
            "It made 1 commit:\n  abc1234 tidy up",
            "What changed:\n@@ -1 +1 @@\n-x = 1\n+x = 2",
        ]
    )


def test_a_formatter_that_touched_hundreds_of_files_is_counted_rather_than_listed() -> None:
    """Naming every file is what fills a prompt budgeted for two spoken sentences; the count is the story."""
    delta = Delta(files=tuple(Changed(f"src/m{n}.py", 1, 1) for n in range(40)))
    told = render(Turn(Asked(None, "format"), ()), delta, Budget(opening=100, said=100, input=100, result=100, steps=10, files=3, commits=10, changes=100))
    assert "  src/m0.py +1 -1\n  src/m1.py +1 -1\n  src/m2.py +1 -1\n  (and 37 more files)" in told
    assert "src/m3.py" not in told


def test_a_turn_that_pulled_a_history_is_counted_rather_than_listed() -> None:
    """The bound the files have, for the reason they have it: a pull brings hundreds and the count is the story.

    Left unbounded this was the one rendered section with no budget at all, so a `git pull --rebase` after a
    long absence put its whole subject list in a prompt sized for two spoken sentences.
    """
    delta = Delta(commits=tuple(Commit(f"abc{n:04d}", f"pulled {n}") for n in range(30)))
    told = render(Turn(Asked(None, "pull"), ()), delta, Budget(opening=100, said=100, input=100, result=100, steps=10, files=10, commits=2, changes=100))
    assert "It made 30 commits:\n  abc0000 pulled 0\n  abc0001 pulled 1\n  (and 28 more commits)" in told
    assert "pulled 2" not in told


def test_a_turn_that_changed_nothing_says_nothing_about_the_repository() -> None:
    assert "repository" not in render(Turn(Asked(None, "think about it"), (Said(None, "Thought."),)), Delta(), ROOMY)


def test_a_turn_told_once_already_carries_its_opening_as_context_rather_than_as_the_request() -> None:
    """Handed the opening as the request twice, a small model answers it twice: heard live on 2026-09-21 as a
    second summary restating the first half of a turn whose steps held none of it."""
    turn = Turn(Asked(None, "fix the test"), (Said(None, "Fixed."),), Continuing(3))
    assert render(turn, Delta(), ROOMY) == (
        "This turn has already been reported once, up to and including its first 3 steps, and none of that may be"
        " reported again. For context only, this is what opened it:\n"
        "The user asked:\nfix the test\n"
        "Report only what it did after that, below.\n\n"
        "Claude said:\nFixed."
    )


def test_one_step_already_told_is_said_in_the_singular() -> None:
    told = render(Turn(Asked(None, "go"), (), Continuing(1)), Delta(), ROOMY)
    assert "up to and including its first step," in told
