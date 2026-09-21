"""Every kind of step, recognised from real Claude Code records.

`tests/fixtures/steps.jsonl` is fourteen real call-and-result pairs cut out of this machine's transcripts, one
per shape the recognisers claim, in the order asserted below. `tests/fixtures/testruns/` is the output four real
test runners wrote, passing and failing, captured on 2026-09-21.
"""

from pathlib import Path

import pytest

from dataclasses import dataclass

from hands.core.session import Membership, SessionId
from hands.core.steps import Call, Result, recognise
from hands.core.testrun import report_of
from hands.core.turn import Asked, Committed, Delegated, Edited, Looked, Other, Planned, Questioned, Ran, Ref, Step, Tested
from hands.sessions.tail import Tails

FIXTURES = Path(__file__).parent / "fixtures"
SID = SessionId("bf411065-dc5c-4ec9-8302-61b84bdb5c53")
# The pairs carry no prompt between them, so one opens the turn they all belong to.
OPENING = '{"type":"user","uuid":"open-1","message":{"role":"user","content":"do everything"}}'


@dataclass
class Registry:
    """As much of the session registry as the tail asks about."""

    member: Membership

    def live_members(self) -> list[Membership]:
        return [self.member]

    def membership(self, session: SessionId) -> Membership | None:
        return self.member if session == self.member.id else None


@pytest.fixture
async def steps(tmp_path: Path) -> tuple[Step, ...]:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(f"{OPENING}\n{(FIXTURES / 'steps.jsonl').read_text()}")
    tails = Tails(Registry(Membership(SID, pid=4242, cwd=tmp_path, transcript=transcript)))
    telling = await tails.tell(SID, None)
    assert telling is not None and telling.turn.opening == Asked(Ref("open-1"), "do everything")
    return telling.turn.steps


def test_each_kind_of_record_is_recognised_as_the_kind_of_step_it_is(steps: tuple[Step, ...]) -> None:
    assert [type(step).__name__ for step in steps] == [
        "Edited",  # Edit
        "Edited",  # Write, of a file that did not exist
        "Ran",  # a Bash call that committed
        "Tested",  # a Bash call whose output pytest wrote
        "Ran",  # a Bash call that did neither
        "Looked",  # Read
        "Looked",  # Grep
        "Looked",  # WebSearch
        "Planned",  # TaskCreate
        "Planned",  # TaskUpdate
        "Delegated",  # Agent
        "Questioned",  # AskUserQuestion
        "Other",  # Skill, which no recogniser claims
        "Other",  # an Edit that failed, and so recorded no patch to be an Edited by
    ]


def test_an_edit_carries_its_hunks_and_a_write_of_a_new_file_carries_what_the_file_now_holds(steps: tuple[Step, ...]) -> None:
    edited, written = steps[0], steps[1]
    assert isinstance(edited, Edited) and not edited.created
    assert edited.path.endswith("src/hands/sessions/transcript.py") and edited.change.startswith("@@ -127,7 +127,10 @@")
    # Claude Code records no hunks for a file written from nothing: there was nothing to diff it against.
    assert isinstance(written, Edited) and written.created
    assert written.path.endswith("x.txt") and written.change == "hi"


def test_a_command_says_what_it_was_for_what_it_printed_and_what_it_did_to_the_repository(steps: tuple[Step, ...]) -> None:
    committed, ran = steps[2], steps[4]
    assert isinstance(committed, Ran) and committed.git == (Committed(sha="1fe6261", kind="committed"),)
    assert isinstance(ran, Ran) and ran.purpose == "Load lit workflow instructions" and not ran.failed
    assert ran.command.startswith("lit quickstart") and ran.output.startswith("Agent instructions for using links issue tracker")


def test_a_command_whose_output_a_test_runner_wrote_is_counted_rather_than_quoted(steps: tuple[Step, ...]) -> None:
    tested = steps[3]
    assert isinstance(tested, Tested)
    assert tested.runner == "pytest" and tested.passed == 299 and tested.failed == 0 and tested.failing == ()


def test_a_read_a_search_and_a_fetch_name_what_they_were_pointed_at(steps: tuple[Step, ...]) -> None:
    read, searched, fetched = steps[5], steps[6], steps[7]
    assert isinstance(read, Looked) and read.tool == "Read" and read.target.endswith("SKILL.md")
    assert isinstance(searched, Looked) and searched.tool == "Grep" and searched.target == "func parentEpicIDs"
    assert isinstance(fetched, Looked) and fetched.tool == "WebSearch" and fetched.found


def test_a_task_names_itself_when_it_is_made_and_its_number_when_only_its_status_changes(steps: tuple[Step, ...]) -> None:
    made, moved = steps[8], steps[9]
    assert made == Planned(made.ref, "Apply Atlantis plan on PR 145 (nomad-prod)", "created")
    # An update records the task's id and its new status, never its subject again.
    assert moved == Planned(moved.ref, "task 1", "in_progress")


def test_a_subagent_still_running_has_no_report_to_give(steps: tuple[Step, ...]) -> None:
    delegated = steps[10]
    assert isinstance(delegated, Delegated) and delegated.agent == "Explore"
    assert delegated.description == "Find claude-code JSONL parser" and delegated.report is None


def test_a_question_carries_the_options_offered_and_the_one_the_user_chose(steps: tuple[Step, ...]) -> None:
    questioned = steps[11]
    assert isinstance(questioned, Questioned)
    (question,) = questioned.questions
    assert question.question.startswith("The fix lands in ~/code/dotfiles/config/zshrc.home")
    assert question.options == ("Apply it, verify, commit", "Apply, don't commit", "Hands off — I'll do it")
    assert question.answer == "Apply it, verify, commit"


def test_a_tool_no_recogniser_claims_is_named_and_summarised_rather_than_dropped(steps: tuple[Step, ...]) -> None:
    other = steps[12]
    assert other == Other(other.ref, "Skill", '{"skill": "laws:code"}', "Launching skill: laws:code", failed=False)


def test_a_call_that_failed_recorded_no_result_to_be_recognised_by_and_is_told_as_the_failure_it_is(steps: tuple[Step, ...]) -> None:
    """Claude Code writes no structured result for an error, so a failed edit has no patch and is no `Edited`."""
    failed = steps[13]
    assert isinstance(failed, Other) and failed.tool == "Edit" and failed.failed
    assert failed.result.startswith("<tool_use_error>String to replace not found in file.")


@pytest.mark.parametrize(
    ("output", "runner", "passed", "failed", "failing"),
    [
        ("pytest.txt", "pytest", 2, 2, ("test_sample.py::test_divides", "test_sample.py::test_names")),
        ("gotest.txt", "go", None, 2, ("TestDivides", "TestNames")),
        ("cargotest.txt", "cargo", 2, 2, ("tests::names", "tests::divides")),
        ("vitest.txt", "vitest", 2, 2, ("src/sample.test.ts > divides", "src/sample.test.ts > names")),
    ],
)
def test_each_runner_is_read_from_the_output_it_really_writes(
    output: str, runner: str, passed: int | None, failed: int, failing: tuple[str, ...]
) -> None:
    report = report_of((FIXTURES / "testruns" / output).read_text())
    assert report is not None and report.runner == runner
    assert report.passed == passed and report.failed == failed
    # Each runner names a failure its own way, and the name is carried as the runner wrote it.
    assert report.failing == failing


def test_a_run_that_only_passed_is_still_a_test_run_and_output_that_is_not_one_is_not() -> None:
    passing = report_of("305 passed, 2 warnings in 15.00s\n")
    assert passing is not None and passing.passed == 305 and passing.failed == 0 and passing.failing == ()
    assert report_of("hello, nothing here\n") is None


def test_a_run_is_counted_by_its_own_summaries_and_not_by_a_number_printed_anywhere_above_them() -> None:
    """A cargo workspace writes one summary per test binary, and vitest counts its files on the line above its tests."""
    workspace = report_of(
        "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n"
        "---- tests::divides stdout ----\n"
        "test result: FAILED. 0 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n"
    )
    assert workspace is not None and workspace.passed == 2 and workspace.failed == 2
    files = report_of(" Test Files  1 failed | 1 passed (2)\n      Tests  3 failed | 5 passed (8)\n")
    assert files is not None and files.passed == 5 and files.failed == 3


def test_output_that_counted_nothing_because_nothing_ran_is_no_test_run() -> None:
    """`go test` writes the same FAIL line for a package that never built, with the reason where the time goes.
    Read as a run, it would be spoken as a suite where nothing failed, which is the opposite of what happened."""
    assert report_of("./main.go:5:2: undefined: foo\nFAIL\tsample [build failed]\n") is None
    # `ok` and a word is how another format entirely starts each line it writes.
    assert report_of("ok 1 - adds\nnot ok 2 - divides\n") is None
    passed = report_of("ok  \tsample\t0.005s\n")
    assert passed is not None and passed.runner == "go" and passed.failed == 0


def test_a_command_that_did_more_than_run_a_suite_is_told_as_the_command_it_was() -> None:
    """`Tested` is its counts and nothing else, so a call whose counts are not the whole story stays a `Ran`."""
    deployed = recognise(Call(None, "Bash", {"command": "pytest && ./deploy.sh"}, Result("2 passed in 0.1s\ndeploy.sh: not found", None, True)))
    assert isinstance(deployed, Ran) and deployed.failed and "not found" in deployed.output
    committing = Result("2 passed in 0.1s", {"gitOperation": {"commit": {"sha": "f0f9776"}}}, False)
    committed = recognise(Call(None, "Bash", {"command": "git commit -am x"}, committing))
    assert isinstance(committed, Ran) and committed.git == (Committed("f0f9776", "committed"),)
    # A suite that really did fail is still a run: what the command said and what the runner counted agree.
    assert recognise(Call(None, "Bash", {"command": "pytest"}, Result("1 failed, 2 passed in 0.1s", None, True))) == Tested(None, "pytest", 2, 1, ())


def test_a_summary_a_runner_drew_around_is_still_the_summary_it_wrote() -> None:
    """A runner hides the cursor while it redraws, which puts a control sequence where the line starts."""
    drawn = report_of("\x1b[?25l      Tests  2 failed | 2 passed (4)\n\x1b[?25h")
    assert drawn is not None and drawn.runner == "vitest" and drawn.failed == 2 and drawn.passed == 2


def test_a_run_that_both_failed_and_errored_is_counted_as_both() -> None:
    """pytest says the two in one breath, and hearing the first alone understates the run by exactly the rest."""
    both = report_of("=========== 3 failed, 2 errors, 1 passed in 1.0s ===========\n")
    assert both is not None and both.failed == 5 and both.passed == 1


def test_the_cases_of_one_failing_test_are_not_counted_as_more_failing_tests() -> None:
    """`go test` counts nothing, so the names it prints are the count, and a table-driven test prints one a case."""
    table = report_of(
        "--- FAIL: TestOuter (0.00s)\n"
        "    --- FAIL: TestOuter/case_a (0.00s)\n"
        "    --- FAIL: TestOuter/case_b (0.00s)\n"
        "FAIL\tsample\t0.005s\n"
    )
    assert table is not None and table.failed == 1 and table.failing == ("TestOuter",)


def test_a_repository_change_this_version_cannot_name_still_keeps_the_command_it_was() -> None:
    """That the record names an operation at all is what says the command did more than run a suite; which
    operation it was is the lesser half. An amend carries no sha, and a stash is a word nobody here reads."""
    amended = Result("2 passed in 0.1s", {"gitOperation": {"commit": {"kind": "amend"}}}, False)
    step = recognise(Call(None, "Bash", {"command": "git commit --amend"}, amended))
    assert isinstance(step, Ran) and step.command == "git commit --amend" and step.git == ()
    stashed = Result("2 passed in 0.1s", {"gitOperation": {"stash": {"ref": "stash@{0}"}}}, False)
    assert isinstance(recognise(Call(None, "Bash", {"command": "git stash"}, stashed)), Ran)


def test_a_call_no_result_has_come_back_for_is_told_as_having_none() -> None:
    step = recognise(Call(None, "Bash", {"command": "sleep 60"}, None))
    assert step == Ran(None, "sleep 60", None, failed=False, output="(no result)", git=())


def test_a_command_that_printed_nothing_says_so_rather_than_saying_nothing() -> None:
    result = Result(text="", structured={"stdout": "", "stderr": ""}, failed=False)
    step = recognise(Call(None, "Bash", {"command": "true"}, result))
    assert step == Ran(None, "true", None, failed=False, output="(no output)", git=())
