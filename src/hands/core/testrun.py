"""A test run as its own output names it: which runner, how many passed and failed, and what failed by name."""

import re
from dataclasses import dataclass

# Runners draw their summaries, and a command's output reaches the transcript with the drawing in it.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@dataclass(frozen=True)
class Report:
    """What a runner said about a run. `passed` is None for a runner whose output does not count what passed."""

    runner: str
    passed: int | None
    failed: int
    failing: tuple[str, ...]


@dataclass(frozen=True)
class Runner:
    """One runner as four patterns: what proves it ran, how it counts, and how it names a test that failed.

    [LAW:composability] the variability is values, not a function per runner, so a runner nobody here uses
    is added by writing patterns rather than code. `passed` and `failed` may be None where the runner's
    default output does not report that number; a run's failures are then counted by the names it printed.
    """

    name: str
    mark: re.Pattern[str]
    passed: re.Pattern[str] | None
    failed: re.Pattern[str] | None
    failing: re.Pattern[str]


# Patterns fitted to output captured from each runner on 2026-09-21, passing and failing.
# The first runner whose mark is found reads the output, so a mark says one runner and not another.
RUNNERS: tuple[Runner, ...] = (
    Runner(
        name="cargo",
        # `test result: FAILED. 2 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s`
        mark=re.compile(r"(?m)^test result:"),
        passed=re.compile(r"(\d+) passed"),
        failed=re.compile(r"(\d+) failed"),
        # Each failing test prints its own captured output under this heading, and again under `failures:`.
        failing=re.compile(r"(?m)^---- (\S+) stdout ----"),
    ),
    Runner(
        name="go",
        # `--- FAIL: TestDivides (0.00s)`, then `FAIL\tsample\t0.005s`. `go test` counts nothing without -v.
        # A package line counts only with the time it took: `FAIL\tsample [build failed]` counted nothing because
        # nothing ran, and a run that counted nothing must never be spoken as a run that nothing failed.
        mark=re.compile(r"(?m)^(?:--- FAIL: |(?:ok|FAIL)\s+\S+\s+(?:\d+(?:\.\d+)?s|\(cached\))\s*$)"),
        passed=None,
        failed=None,
        failing=re.compile(r"(?m)^\s*--- FAIL: (\S+)"),
    ),
    Runner(
        name="vitest",
        # `      Tests  2 failed | 2 passed (4)`, and one ` FAIL  src/sample.test.ts > names` per failure.
        mark=re.compile(r"(?m)^\s*Tests\s+\d"),
        passed=re.compile(r"(\d+) passed"),
        failed=re.compile(r"(\d+) failed"),
        failing=re.compile(r"(?m)^\s*FAIL\s+(\S.*?)\s*$"),
    ),
    Runner(
        name="pytest",
        # `2 failed, 2 passed in 0.02s`, bare under -q and inside a rule of `=` otherwise.
        mark=re.compile(r"(?m)^(?:=+ )?\d+ (?:passed|failed|error|skipped)[^\n]* in \d"),
        passed=re.compile(r"(\d+) passed"),
        failed=re.compile(r"(\d+) (?:failed|errors?)"),
        failing=re.compile(r"(?m)^(?:FAILED|ERROR) (\S+)"),
    ),
)


def report_of(output: str) -> Report | None:
    """What a test runner said in this output, or None where no runner's mark is in it.

    A command is a test run because its output is a test runner's, not because of how the command was
    spelled: `make test`, `just check` and a script all reach the same runners [LAW:one-source-of-truth].
    """
    text = _ANSI.sub("", output)
    for runner in RUNNERS:
        # [LAW:one-source-of-truth] a run is counted by the summaries its mark found and by nothing else in the
        # scrollback: a workspace writes one summary per binary, and the line above them counts files, not tests.
        summaries = tuple(line for line in text.splitlines() if runner.mark.search(line))
        if not summaries:
            continue
        failing = tuple(match.group(1) for match in runner.failing.finditer(text))
        counted = _count(runner.failed, summaries)
        return Report(runner.name, _count(runner.passed, summaries), len(failing) if counted is None else counted, failing)
    return None


def _count(pattern: re.Pattern[str] | None, summaries: tuple[str, ...]) -> int | None:
    """What every summary line counted, added up; None where this runner's output never says that number."""
    if pattern is None:
        return None
    found = [int(match.group(1)) for line in summaries if (match := pattern.search(line)) is not None]
    return sum(found) if found else None
