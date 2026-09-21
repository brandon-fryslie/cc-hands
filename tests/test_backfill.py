"""What a session did before anyone was listening, read from its transcript through the tail's own recognisers.

`tests/fixtures/session.jsonl` is eighty-four records cut whole out of a real session of 1,778 of them: two
turns, the tool calls and results they are mostly made of, and one `Agent` dispatch, whose own records this
Claude Code writes to the subagent's file rather than this one.
"""

from pathlib import Path

import pytest

from hands.core.turn import Asked, Delegated, Edited, Happening, Ran, Ref
from hands.sessions.backfill import Unseen, read_since

FIXTURE = Path(__file__).parent / "fixtures" / "session.jsonl"


def read(since: Ref | None = None) -> list[Happening]:
    return read_since(FIXTURE, since).happenings


def written(path: Path, *records: str) -> Path:
    path.write_text("".join(f"{record}\n" for record in records))
    return path


PROMPT = '{"uuid":"u1","type":"user","message":{"role":"user","content":"run the suite"}}'
CALL = '{"uuid":"u2","type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"pytest"}}]}}'
RESULT = '{"uuid":"u3","type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"3 tests did not pass"}]}}'
DONE = '{"uuid":"u4","type":"assistant","message":{"content":[{"type":"text","text":"Done."}]}}'


def test_a_session_nobody_has_heard_of_yet_is_read_whole_and_everything_names_its_record() -> None:
    """What was asked is in it, in its place: the steps alone say how a session spent an hour, never what for."""
    happenings = read()
    assert [type(happening).__name__ for happening in happenings] == [
        "Asked", "Said", "Ran", "Ran", "Said", "Asked", "Said", "Ran", "Ran", "Edited", "Ran", "Delegated", "Said"
    ]
    asked = [happening for happening in happenings if isinstance(happening, Asked)]
    assert asked[0].text.startswith("You know, im wondering if we should be using Pipecat")
    assert asked[1].text.startswith("Pytorch?")
    # Everything a backfill hands over came from a record, so every one of them can be read on from.
    assert all(happening.ref is not None for happening in happenings)


def test_what_came_after_a_record_is_what_the_reader_has_not_had() -> None:
    """The whole session read from a point in it: what came before is what the reader already has."""
    happenings = read()
    sixth = happenings[5].ref
    assert sixth is not None and read(sixth) == happenings[6:]
    # Cut at its own last record, a session has nothing left to say.
    last = happenings[-1].ref
    assert last is not None and read(last) == []


def test_a_subagent_is_one_delegated_step_and_not_the_work_it_did() -> None:
    """A subagent's own records are its own transcript's, which this Claude Code writes as a separate file.
    So in the session that sent it, a subagent is the one call that sent it — here launched to run in the
    background, whose report comes back later as a notification opening a turn of its own."""
    delegated = [step for step in read() if isinstance(step, Delegated)]
    assert len(delegated) == 1
    assert delegated[0].agent == "general-purpose"
    assert delegated[0].description == "Rewrite docs for Pipecat architecture"
    assert delegated[0].report is None


def test_a_call_answered_after_the_cut_is_still_one_step_that_knows_its_result() -> None:
    """Read from the mark on, that result would have had no call to belong to, and would have been dropped."""
    happenings = read()
    answered = [step for step in happenings if isinstance(step, Ran) and step.output != "(no result)"]
    assert answered and all(step.output for step in answered)
    edited = next(step for step in happenings if isinstance(step, Edited))
    assert edited.change


def test_a_call_shown_with_no_result_yet_is_never_the_mark_a_reading_goes_on_from(tmp_path: Path) -> None:
    """The reader is told the suite is being run, and must also be told that three tests did not pass.

    Marking a call that has not come back would go on from after it next time, so the result would arrive with
    nobody to tell: the one thing the reader was waiting to hear is the one thing it would never hear.
    """
    working = read_since(written(tmp_path / "working.jsonl", PROMPT, CALL), None)
    assert [type(happening).__name__ for happening in working.happenings] == ["Asked", "Ran"]
    running = working.happenings[1]
    assert isinstance(running, Ran) and running.output == "(no result)"
    # What was asked is finished business. The call the session is still inside is not.
    assert working.settled == 1

    done = read_since(written(tmp_path / "done.jsonl", PROMPT, CALL, RESULT, DONE), Ref("u1"))
    assert [type(happening).__name__ for happening in done.happenings] == ["Ran", "Said"]
    answered = done.happenings[0]
    assert isinstance(answered, Ran) and answered.output == "3 tests did not pass"
    assert done.settled == 2


def test_a_call_nothing_ever_answers_does_not_hold_the_reading_back(tmp_path: Path) -> None:
    """Only a call a reading ends on is unfinished. One the session carried on past is never coming back, and
    holding the mark behind it would read the rest of the session again, every time, for ever."""
    abandoned = read_since(written(tmp_path / "abandoned.jsonl", PROMPT, CALL, DONE), None)
    assert [type(happening).__name__ for happening in abandoned.happenings] == ["Asked", "Ran", "Said"]
    assert abandoned.settled == 3


def test_a_mark_this_transcript_never_held_is_said_rather_than_read_as_the_start(tmp_path: Path) -> None:
    """[LAW:no-silent-failure] otherwise a mark from another session re-tells this one from the top as if new."""
    transcript = written(tmp_path / "s.jsonl", PROMPT, CALL, RESULT, DONE)
    with pytest.raises(Unseen):
        read_since(transcript, Ref("u9"))


def test_a_transcript_that_is_being_written_is_read_only_as_far_as_its_last_whole_record(tmp_path: Path) -> None:
    """The file grows under the reading, and a record is whole only once its newline is written."""
    half = tmp_path / "half.jsonl"
    whole = FIXTURE.read_bytes()
    half.write_bytes(whole[: whole.rindex(b"\n") + 1] + b'{"type":"assistant","message":{"content":[{"typ')
    assert read_since(half, None).happenings == read()
