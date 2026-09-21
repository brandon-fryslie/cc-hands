"""What a session did before anyone was listening, read from its transcript through the tail's own recognisers.

`tests/fixtures/session.jsonl` is eighty-four records cut whole out of a real session of 1,778 of them: two
turns, the tool calls and results they are mostly made of, and one `Agent` dispatch, whose own records this
Claude Code writes to the subagent's file rather than this one.
"""

from pathlib import Path

from hands.core.turn import Delegated, Edited, Ran, Ref, Step
from hands.sessions.backfill import steps_since

FIXTURE = Path(__file__).parent / "fixtures" / "session.jsonl"


def read(since: Ref | None = None) -> list[Step]:
    return steps_since(FIXTURE, since)


def test_a_session_nobody_has_heard_of_yet_is_read_whole_and_every_step_names_its_record() -> None:
    steps = read()
    assert [type(step).__name__ for step in steps] == [
        "Said", "Ran", "Ran", "Said", "Said", "Ran", "Ran", "Edited", "Ran", "Delegated", "Said"
    ]
    # Every step of a backfill came from a record, so every one of them can be asked about again.
    assert all(step.ref is not None for step in steps)


def test_the_steps_after_a_record_are_the_ones_the_reader_has_not_had() -> None:
    """The whole session read from a point in it: what came before is what the reader already has."""
    steps = read()
    sixth = steps[5]
    assert sixth.ref is not None
    after = read(sixth.ref)
    assert after == steps[6:]
    # Cut at its own last record, a session has nothing left to say.
    last = steps[-1].ref
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
    steps = read()
    answered = [step for step in steps if isinstance(step, Ran) and step.output != "(no result)"]
    assert answered and all(step.output for step in answered)
    edited = next(step for step in steps if isinstance(step, Edited))
    assert edited.change


def test_a_transcript_that_is_being_written_is_read_only_as_far_as_its_last_whole_record(tmp_path: Path) -> None:
    """The file grows under the reading, and a record is whole only once its newline is written."""
    half = tmp_path / "half.jsonl"
    whole = FIXTURE.read_bytes()
    half.write_bytes(whole[: whole.rindex(b"\n") + 1] + b'{"type":"assistant","message":{"content":[{"typ')
    assert steps_since(half, None) == read()
