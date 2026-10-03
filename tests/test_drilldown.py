"""Drill-down: "more on that" opens a told turn into its parts, and a part a rung deeper each time it is asked."""

from typing import cast

from hands.core.delta import Delta
from hands.core.drilldown import LADDER, drill
from hands.core.narration import narration
from hands.core.session import PromptId, SessionId
from hands.core.turn import CUT, Asked, Edited, Said, Step, Tested, Turn
from hands.sessions.registry import Sessions
from hands.core.pending import News
from hands.voice.narrator import Recounts
from hands.voice.tools import Body, Result, expand_tool

SID = SessionId("s1")
WHY = "test_splits_on_commas\nAssertionError: assert ['a,b'] == ['a', 'b']"
STEPS: tuple[Step, ...] = (
    Edited(None, "/code/parser.py", False, "@@ -1 +1 @@\n-split(';')\n+split(',')"),
    Tested(None, "pytest", 11, 1, ("tests/test_parser.py::test_splits_on_commas",), WHY),
    Said(None, "The comma test still fails: the parser splits before it strips quotes. Want me to fix the order?"),
)


def held(steps: tuple[Step, ...] = STEPS, turn: str = "p1") -> Recounts:
    recounts = Recounts()
    recounts.put(SID, PromptId(turn), News(None, "news", "", "", narration(Turn(Asked(None, "fix the parser"), steps), Delta(), ()).parts, frozenset()))
    return recounts


def expand(recounts: Recounts) -> Body:
    """The tool's body, as the model calls it."""
    return expand_tool(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None), recounts).body


def parts(result: Result) -> list[tuple[str, str]]:
    return [(part["part"], part["told"]) for part in cast(list[dict[str, str]], result["parts"])]


async def test_a_failing_test_in_the_headline_opens_into_what_failed_and_why_and_further_each_time() -> None:
    recounts = held()
    call = expand(recounts)
    whole = await call(session=SID)
    assert ("the tests", "The tests: one test run.") in parts(whole) and whole["deeper"] and whole["depth"] == 0
    opened = await call(session=SID, part="the tests")
    [(part, text)] = parts(opened)
    assert part == "the tests" and opened["depth"] == 1 and "test_splits_on_commas" in text and "AssertionError: assert ['a,b']" in text
    further = await call(session=SID, part="the tests")
    assert further["depth"] == 2 and parts(further) == parts(opened) and not opened["deeper"]


async def test_asked_past_the_bottom_it_tells_the_longest_again_and_says_there_is_no_more() -> None:
    call = expand(held())
    for _ in LADDER:
        await call(session=SID, part="the change")
    bottom = await call(session=SID, part="the change")
    assert bottom["depth"] == len(LADDER) + 1 and not bottom["deeper"] and "parser.py" in parts(bottom)[0][1]


async def test_the_turn_as_a_whole_is_its_parts_lines_however_often_asked_and_moves_no_part() -> None:
    call = expand(held())
    first = await call(session=SID)
    tests = await call(session=SID, part="the tests")
    again = await call(session=SID)
    deeper = await call(session=SID, part="the tests")
    assert again == first and "The parser splits" not in str(again)
    assert (tests["depth"], deeper["depth"]) == (1, 2)


async def test_a_telling_that_found_nothing_new_keeps_how_far_a_part_was_opened() -> None:
    recounts = held()
    call = expand(recounts)
    await call(session=SID, part="the tests")
    recounts.put(SID, PromptId("p1"), None)
    assert (await call(session=SID, part="the tests"))["depth"] == 2


async def test_a_part_asked_for_by_another_name_is_said_to_be_missing_with_the_parts_there_are() -> None:
    assert await expand(held())(session=SID, part="the deployment") == {
        "error": "the last turn of s1 has no part 'the deployment'; its parts are the question, the change, the tests, what it said"
    }


async def test_with_no_turn_held_it_says_so_and_points_at_the_readings() -> None:
    assert await expand(Recounts())(session=SID) == {"error": "hands holds no finished turn of s1 to open; read_session and read_turn read what it did"}


async def test_a_new_telling_of_the_turn_starts_the_opening_over_and_holds_both_halves() -> None:
    recounts = held()
    call = expand(recounts)
    await call(session=SID, part="the tests")
    recounts.put(SID, PromptId("p1"), News(None, "then", "", "", narration(Turn(Asked(None, "fix the parser"), (Edited(None, "/code/quotes.py", True, "+strip"),)), Delta(), ()).parts, frozenset()))
    again = await call(session=SID, part="the change")
    assert again["depth"] == 1 and [part for part, _ in parts(again)] == ["the change", "the change"]


def test_the_deepest_rung_is_still_cut_rather_than_the_record_as_written() -> None:
    long = "word " * 5_000
    [section] = narration(Turn(Asked(None, "explain"), (Said(None, long),)), Delta(), ()).sections
    lengths = [len(drill((section,), depth).told[0][1]) for depth in range(len(LADDER))]
    assert lengths == sorted(lengths) and len(set(lengths)) == len(LADDER)
    assert drill((section,), len(LADDER)).told[0][1].endswith(CUT)


async def test_a_part_with_more_in_it_than_a_rung_shows_says_so_and_opens_further() -> None:
    call = expand(held((Said(None, "The parser splits before it strips quotes. " * 40),)))
    first, second = await call(session=SID, part="what it said"), await call(session=SID, part="what it said")
    assert first["deeper"] and len(parts(second)[0][1]) > len(parts(first)[0][1])
