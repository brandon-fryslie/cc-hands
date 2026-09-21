"""A finished turn, read from a transcript made of real Claude Code records, and rendered for the summariser."""

from pathlib import Path

import pytest

from hands.core.turn import CUT, Asked, Budget, Notified, Said, Turn, Used, render
from hands.sessions.transcript import UNTOLD, Reading, Told, read_turn

# Real records from this repository's own sessions: an earlier /clear prompt, the prompt that starts the turn,
# attachments, modes, a title, thinking, text, three Bash calls with their results (the last failed),
# an injected isMeta user record, a final text, and a record still being written.
FIXTURE = Path(__file__).parent / "fixtures" / "turn.jsonl"
ROOMY = Budget(opening=10_000, said=10_000, input=10_000, result=10_000, steps=100)


def test_the_turn_is_everything_said_and_used_after_the_last_prompt_in_order() -> None:
    turn = turn_of(FIXTURE)
    assert turn is not None
    assert isinstance(turn.opening, Asked) and turn.opening.text.startswith("I'd like you to go a bit further and architect a robust system")
    assert [type(step).__name__ for step in turn.steps] == ["Said", "Used", "Used", "Used", "Said"]
    said, loaded, inspected, commented, final = turn.steps
    assert isinstance(said, Said) and said.text.startswith("I'll start by loading the repo conventions")
    assert isinstance(loaded, Used)
    assert loaded == Used(
        tool="Bash", purpose="Load lit workflow and check git state", input=loaded.input, result=loaded.result, failed=False
    )
    assert loaded.input.startswith("lit quickstart 2>&1 | head -80") and loaded.result.startswith("Agent instructions for using links issue tracker")
    assert isinstance(inspected, Used) and inspected.purpose == "Inspect repo layout and remotes"
    # A call without a description has no purpose, and a result marked as an error is a failure.
    assert isinstance(commented, Used) and commented.purpose is None and commented.failed
    assert commented.result.startswith("Exit code 2")
    assert isinstance(final, Said) and final.text.startswith("API Error: Unable to connect to API")


def turn_of(transcript: Path, told: Told = UNTOLD, closing: str | None = None) -> Turn | None:
    reading = read_turn(transcript, told, closing)
    return None if reading is None else reading.turn


def lines(*records: str) -> str:
    return "".join(f"{record}\n" for record in records)


PROMPT = '{"type":"user","message":{"role":"user","content":"first"}}'
CALL = '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"sleep 60"}}]}}'
RESULT = '{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"done"}]}}'
DONE = '{"type":"assistant","message":{"content":[{"type":"text","text":"Done."}]}}'


def test_a_prompt_sent_as_blocks_with_an_image_opens_the_turn(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    image = '{"type":"user","origin":{"kind":"human"},"message":{"role":"user","content":[{"type":"text","text":"match this"},{"type":"image","source":{}}]}}'
    transcript.write_text(lines(PROMPT, DONE, image))
    assert turn_of(transcript) == Turn(Asked("match this\n[an image]"), ())


def test_a_notification_after_the_turn_ended_opens_a_turn_of_its_own_and_is_not_what_the_user_asked(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    notified = '{"type":"user","origin":{"kind":"task-notification"},"message":{"role":"user","content":"<task-notification>tests passed</task-notification>"}}'
    transcript.write_text(lines(PROMPT, DONE, notified, DONE))
    turn = turn_of(transcript)
    assert turn == Turn(Notified("<task-notification>tests passed</task-notification>"), (Said("Done."),))
    assert turn is not None and render(turn, ROOMY).startswith("A background task reported:\n")


@pytest.mark.parametrize("kind", ["human", "task-notification"])
def test_a_message_that_lands_while_a_tool_runs_belongs_to_the_turn_under_way(tmp_path: Path, kind: str) -> None:
    transcript = tmp_path / "t.jsonl"
    landed = f'{{"type":"user","origin":{{"kind":"{kind}"}},"message":{{"role":"user","content":"also this"}}}}'
    transcript.write_text(lines(PROMPT, CALL, RESULT, landed, DONE))
    turn = turn_of(transcript)
    assert turn is not None and turn.opening == Asked("first")
    assert [type(step).__name__ for step in turn.steps] == ["Used", "Said"]


def test_compactions_summary_does_not_open_a_turn(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    summary = '{"type":"user","isCompactSummary":true,"message":{"role":"user","content":"This session is being continued"}}'
    transcript.write_text(lines(PROMPT, DONE, summary, DONE))
    turn = turn_of(transcript)
    assert turn is not None and turn.opening == Asked("first") and turn.steps == (Said("Done."), Said("Done."))


def test_a_prompt_with_a_document_attached_opens_the_turn_and_names_the_document_rather_than_its_bytes(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    document = '{"type":"user","message":{"role":"user","content":[{"type":"document","source":{"data":"JVBERi0x"}},{"type":"text","text":"read this"}]}}'
    transcript.write_text(lines(PROMPT, DONE, document))
    assert turn_of(transcript) == Turn(Asked("[a document]\nread this"), ())


def test_a_reading_picks_up_after_the_steps_already_told_and_says_what_is_told_now(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    opened = '{"type":"user","uuid":"u1","message":{"role":"user","content":"first"}}'
    looked = '{"type":"assistant","uuid":"u2","message":{"content":[{"type":"text","text":"Looked."}]}}'
    call = '{"type":"assistant","uuid":"u3","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"pytest"}}]}}'
    result = '{"type":"user","uuid":"u4","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"1 passed"}]}}'
    transcript.write_text(lines(opened, looked, call, result))
    ran = Used("Bash", None, "pytest", "1 passed", False)
    assert read_turn(transcript, UNTOLD, None) == Reading(Turn(Asked("first"), (Said("Looked."), ran)), Told("u1", 2, None))
    assert read_turn(transcript, Told("u1", 1, None), None) == Reading(Turn(Asked("first"), (ran,)), Told("u1", 2, None))
    assert read_turn(transcript, Told("u1", 2, None), None) == Reading(Turn(Asked("first"), ()), Told("u1", 2, None))
    # What was told of another turn says nothing about this one, so all of it is still to tell.
    assert turn_of(transcript, told=Told("elsewhere", 2, None)) == Turn(Asked("first"), (Said("Looked."), ran))


def test_the_hooks_closing_reply_ends_a_turn_whose_transcript_does_not_hold_it_yet_and_is_not_doubled_when_it_does(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, CALL, RESULT))
    ran = Used("Bash", None, "sleep 60", "done", False)
    assert turn_of(transcript, closing="Done.") == Turn(Asked("first"), (ran, Said("Done.")))
    transcript.write_text(lines(PROMPT, CALL, RESULT, DONE))
    assert turn_of(transcript, closing="Done.") == Turn(Asked("first"), (ran, Said("Done.")))


def test_a_closing_reply_told_before_its_record_was_written_is_not_told_again_once_it_is(tmp_path: Path) -> None:
    """The stand-in is told once: the record Claude Code writes moments later gives way to it, whatever follows."""
    transcript = tmp_path / "t.jsonl"
    opened = '{"type":"user","uuid":"u1","message":{"role":"user","content":"first"}}'
    call = '{"type":"assistant","uuid":"u2","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"pytest"}}]}}'
    result = '{"type":"user","uuid":"u3","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"1 passed"}]}}'
    fixed = '{"type":"assistant","uuid":"u4","message":{"content":[{"type":"text","text":"Fixed."}]}}'
    listed = '{"type":"assistant","uuid":"u5","message":{"content":[{"type":"tool_use","id":"t2","name":"Bash","input":{"command":"ls"}}]}}'
    files = '{"type":"user","uuid":"u6","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t2","content":"a.py"}]}}'
    transcript.write_text(lines(opened, call, result))
    first = read_turn(transcript, UNTOLD, "Fixed.")
    assert first == Reading(Turn(Asked("first"), (Used("Bash", None, "pytest", "1 passed", False), Said("Fixed."))), Told("u1", 1, "Fixed."))
    # The Stop was blocked by another hook, so the same turn runs on, and by now Claude Code has written "Fixed." itself.
    transcript.write_text(lines(opened, call, result, fixed, listed, files))
    assert first is not None
    assert read_turn(transcript, first.told, "Done.") == Reading(
        Turn(Asked("first"), (Used("Bash", None, "ls", "a.py", False), Said("Done."))), Told("u1", 3, "Done.")
    )


def test_a_reply_a_later_turn_repeats_is_told_again_because_what_was_told_of_one_turn_says_nothing_of_the_next(tmp_path: Path) -> None:
    """Claude says the same short thing twice in a row: the second turn is its own, and is heard."""
    transcript = tmp_path / "t.jsonl"
    checked = '{"type":"user","uuid":"u1","message":{"role":"user","content":"check"}}'
    nothing = '{"type":"assistant","uuid":"u2","message":{"content":[{"type":"text","text":"Nothing to do."}]}}'
    again = '{"type":"user","uuid":"u3","message":{"role":"user","content":"check again"}}'
    transcript.write_text(lines(checked))
    first = read_turn(transcript, UNTOLD, "Nothing to do.")
    assert first == Reading(Turn(Asked("check"), (Said("Nothing to do."),)), Told("u1", 0, "Nothing to do."))
    transcript.write_text(lines(checked, nothing, again))
    assert first is not None
    assert read_turn(transcript, first.told, "Nothing to do.") == Reading(
        Turn(Asked("check again"), (Said("Nothing to do."),)), Told("u3", 0, "Nothing to do.")
    )


def test_a_closing_reply_is_matched_to_its_record_however_the_whitespace_around_it_differs(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    opened = '{"type":"user","uuid":"u1","message":{"role":"user","content":"first"}}'
    padded = '{"type":"assistant","uuid":"u2","message":{"content":[{"type":"text","text":"  Done.\\n"}]}}'
    transcript.write_text(lines(opened))
    first = read_turn(transcript, UNTOLD, "Done.")
    assert first == Reading(Turn(Asked("first"), (Said("Done."),)), Told("u1", 0, "Done."))
    transcript.write_text(lines(opened, padded))
    assert first is not None
    assert read_turn(transcript, first.told, "Done.") == Reading(Turn(Asked("first"), ()), Told("u1", 1, None))


def test_a_transcript_with_no_prompt_yet_has_no_turn(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('{"type":"ai-title","aiTitle":"x"}\n{"type":"user","isMeta":true,"message":{"role":"user","content":"injected"}}\n')
    assert read_turn(transcript, UNTOLD, None) is None


def test_a_prompt_with_nothing_after_it_is_a_turn_with_no_steps(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('{"type":"user","message":{"role":"user","content":"hello"}}\n')
    assert turn_of(transcript) == Turn(Asked("hello"), ())


def test_a_call_whose_result_never_came_is_shown_as_having_none(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        '{"type":"user","message":{"role":"user","content":"go"}}\n'
        '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"/a/b.py"}}]}}\n'
    )
    assert turn_of(transcript) == Turn(Asked("go"), (Used("Read", None, '{"file_path": "/a/b.py"}', "(no result)", False),))


def test_the_rendered_turn_names_each_step_and_its_outcome() -> None:
    turn = Turn(Asked("fix the test"), (Said("Looking."), Used("Bash", "Run the tests", "pytest", "1 failed", True), Used("Edit", None, "{}", "updated", False)))
    assert render(turn, ROOMY) == (
        "The user asked:\nfix the test\n\n"
        "Claude said:\nLooking.\n\n"
        "Claude used Bash (Run the tests): pytest\nResult (failed): 1 failed\n\n"
        "Claude used Edit: {}\nResult: updated"
    )


def test_a_long_turn_keeps_how_it_started_and_how_it_ended_and_each_part_is_cut_to_its_budget() -> None:
    steps = tuple(Said(f"step {n}") for n in range(10))
    rendered = render(Turn(Asked("x" * 50), steps), Budget(opening=10, said=100, input=100, result=100, steps=4))
    assert rendered == "\n\n".join(
        ["The user asked:\n" + "x" * 10 + CUT, "Claude said:\nstep 0", "Claude said:\nstep 1", "(6 steps in the middle are left out)", "Claude said:\nstep 8", "Claude said:\nstep 9"]
    )
