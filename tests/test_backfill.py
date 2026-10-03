"""What a session did before anyone was listening, read from its transcript through the tail's own recognisers.

`tests/fixtures/session.jsonl` is eighty-four records cut whole out of a real session of 1,778 of them: two
turns, the tool calls and results they are mostly made of, and one `Agent` dispatch, whose own records this
Claude Code writes to the subagent's file rather than this one.
"""

from pathlib import Path

from loguru import logger

from hands.core.turn import Asked, Commanded, Delegated, Edited, Happening, Interruption, Ran, Ref, Said, Shelled
from hands.sessions.backfill import read_transcript

FIXTURE = Path(__file__).parent / "fixtures" / "session.jsonl"


def read() -> list[Happening]:
    return read_transcript(FIXTURE).happenings


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
    working = read_transcript(written(tmp_path / "working.jsonl", PROMPT, CALL))
    assert [type(happening).__name__ for happening in working.happenings] == ["Asked", "Ran"]
    running = working.happenings[1]
    assert isinstance(running, Ran) and running.output == "(no result)"
    # What was asked is finished business. The call the session is still inside is not.
    assert working.settled == 1

    done = read_transcript(written(tmp_path / "done.jsonl", PROMPT, CALL, RESULT, DONE))
    assert [type(happening).__name__ for happening in done.happenings] == ["Asked", "Ran", "Said"]
    answered = done.happenings[1]
    assert isinstance(answered, Ran) and answered.output == "3 tests did not pass"
    assert done.settled == 3


def test_a_call_nothing_ever_answers_does_not_hold_the_reading_back(tmp_path: Path) -> None:
    """Only a call a reading ends on is unfinished. One the session carried on past is never coming back, and
    holding the mark behind it would read the rest of the session again, every time, for ever."""
    abandoned = read_transcript(written(tmp_path / "abandoned.jsonl", PROMPT, CALL, DONE))
    assert [type(happening).__name__ for happening in abandoned.happenings] == ["Asked", "Ran", "Said"]
    assert abandoned.settled == 3


def test_a_call_id_that_comes_round_again_does_not_take_the_first_call_with_it(tmp_path: Path) -> None:
    """A result is paired by id alone, so a repeated id means the earlier call can never be answered here.

    A reading folds a whole file without ever letting go, so the earlier slot would otherwise be told using
    the later call's command — a step describing work that was never done at that point in the session.
    """
    transcript = written(
        tmp_path / "twice.jsonl",
        PROMPT,
        '{"uuid":"u2","type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"pytest -x"}}]}}',
        '{"uuid":"u3","type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"pytest -q"}}]}}',
        '{"uuid":"u4","type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"all good"}]}}',
    )
    first, second = read_transcript(transcript).happenings[1:]
    assert isinstance(first, Ran) and first.command == "pytest -x" and first.output == "(no result)"
    assert isinstance(second, Ran) and second.command == "pytest -q" and second.output == "all good"


def test_a_call_still_out_while_another_answers_holds_the_mark_where_it_is(tmp_path: Path) -> None:
    """Calls run several at a time and their results land in any order: 138 of them did so on this machine.

    Marking past a call because a later one came back first puts its result behind the mark, where the fold
    reads it correctly and the cut then throws it away — the suite reported as run and never as failed.
    """
    calls = (
        PROMPT,
        CALL,
        '{"uuid":"u3","type":"assistant","message":{"content":[{"type":"tool_use","id":"t2","name":"Bash","input":{"command":"ls"}}]}}',
        '{"uuid":"u4","type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t2","content":"a.py"}]}}',
    )
    out = read_transcript(written(tmp_path / "out.jsonl", *calls))
    assert [type(happening).__name__ for happening in out.happenings] == ["Asked", "Ran", "Ran"]
    # The second call came back; the first is still out, so the mark stays behind it.
    assert out.settled == 1

    landed = read_transcript(written(tmp_path / "landed.jsonl", *calls, RESULT))
    assert [(step.command, step.output) for step in landed.happenings if isinstance(step, Ran)] == [
        ("pytest", "3 tests did not pass"),
        ("ls", "a.py"),
    ]


def test_a_transcript_that_is_being_written_is_read_only_as_far_as_its_last_whole_record(tmp_path: Path) -> None:
    """The file grows under the reading, and a record is whole only once its newline is written."""
    half = tmp_path / "half.jsonl"
    whole = FIXTURE.read_bytes()
    half.write_bytes(whole[: whole.rindex(b"\n") + 1] + b'{"type":"assistant","message":{"content":[{"typ')
    assert read_transcript(half).happenings == read()


def test_a_call_the_user_interrupted_is_read_as_over_and_the_interrupt_in_its_place(tmp_path: Path) -> None:
    """A second call left open when the user pressed Escape is never answered: the interrupt is the proof."""
    second = '{"uuid":"u5","type":"assistant","message":{"content":[{"type":"tool_use","id":"t2","name":"Bash","input":{"command":"sleep 60"}}]}}'
    cut = '{"uuid":"u6","type":"user","promptId":"p1","message":{"role":"user","content":[{"type":"text","text":"[Request interrupted by user]"}]}}'
    reading = read_transcript(written(tmp_path / "t.jsonl", PROMPT, CALL, RESULT, second, cut))
    assert reading.happenings[-1] == Interruption(Ref("u6"))
    assert reading.settled == len(reading.happenings)


def test_what_a_command_the_user_ran_printed_is_read_with_the_command_and_opens_nothing(tmp_path: Path) -> None:
    shell = '{"uuid":"u5","type":"user","message":{"role":"user","content":"<bash-input>git status</bash-input>"}}'
    output = '{"uuid":"u6","parentUuid":"u5","type":"user","message":{"role":"user","content":"<bash-stdout>clean</bash-stdout><bash-stderr></bash-stderr>"}}'
    happenings = read_transcript(written(tmp_path / "t.jsonl", PROMPT, DONE, shell, output, DONE)).happenings
    assert happenings[2:] == [Shelled(Ref("u5"), "git status", "clean"), Said(Ref("u4"), "Done.")]


def test_a_command_that_printed_nothing_is_read_as_having_no_output(tmp_path: Path) -> None:
    shell = '{"uuid":"u5","type":"user","message":{"role":"user","content":"<bash-input>true</bash-input>"}}'
    output = '{"uuid":"u6","parentUuid":"u5","type":"user","message":{"role":"user","content":"<bash-stdout></bash-stdout><bash-stderr></bash-stderr>"}}'
    happenings = read_transcript(written(tmp_path / "t.jsonl", PROMPT, DONE, shell, output)).happenings
    assert happenings[2:] == [Shelled(Ref("u5"), "true", None)]


COMPACT_TYPED = '{"uuid":"u5","type":"user","promptId":"p2","message":{"role":"user","content":"/compact"}}'
COMPACT_SUMMARY = '{"uuid":"u6","type":"user","isCompactSummary":true,"promptId":"p3","message":{"role":"user","content":"This session is being continued"}}'
COMPACT_RECORD = '{"uuid":"u7","type":"user","promptId":"p3","message":{"role":"user","content":"<command-name>/compact</command-name>\\n<command-message>compact</command-message>\\n<command-args></command-args>"}}'
COMPACTED = '{"uuid":"u8","parentUuid":"u7","type":"user","promptId":"p3","message":{"role":"user","content":"<local-command-stdout>\\u001b[2mCompacted\\u001b[22m</local-command-stdout>"}}'


def test_compact_typed_as_words_ahead_of_its_compaction_is_one_command_with_what_it_printed(tmp_path: Path) -> None:
    """As 2.1.286 writes it, live: the words `/compact`, the compaction, then its command record and what it printed, under
    a prompt id of their own and naming nothing of the words."""
    said: list[str] = []
    sink = logger.add(lambda message: said.append(message.record["message"]), level="DEBUG", filter="hands.sessions.turning")
    try:
        happenings = read_transcript(written(tmp_path / "t.jsonl", PROMPT, DONE, COMPACT_TYPED, COMPACT_SUMMARY, COMPACT_RECORD, COMPACTED)).happenings
    finally:
        logger.remove(sink)
    assert happenings[2:] == [Commanded(Ref("u7"), "/compact", "", "Compacted")]
    # [LAW:nothing-unseen] the join is said, with both records.
    assert said == ["/compact's record u7 is the command its words u5 opened a turn for, not a turn of its own"]


def test_a_compact_typed_and_never_run_is_a_command_of_its_own_beside_the_next_one(tmp_path: Path) -> None:
    """The first was stopped before it compacted anything: the record that follows is the second's, and joins only it."""
    again = '{"uuid":"u9","type":"user","promptId":"p4","message":{"role":"user","content":"/compact"}}'
    happenings = read_transcript(written(tmp_path / "t.jsonl", PROMPT, DONE, COMPACT_TYPED, again, COMPACT_SUMMARY, COMPACT_RECORD, COMPACTED)).happenings
    assert happenings[2:] == [Commanded(Ref("u5"), "/compact", ""), Commanded(Ref("u7"), "/compact", "", "Compacted")]


def test_a_prompt_claude_was_sent_is_asked_though_it_opens_with_a_slash_and_a_name(tmp_path: Path) -> None:
    """Both as Claude Code wrote them on 2.1.283: Claude answered the first, and the second names a directory, not a command."""
    help = '{"uuid":"u5","type":"user","promptId":"p2","promptSource":"typed","message":{"role":"user","content":" /help reply with only the word pong"}}'
    path = '{"uuid":"u6","type":"user","promptId":"p3","promptSource":"queued","message":{"role":"user","content":"/tmp is full, clean it up"}}'
    happenings = read_transcript(written(tmp_path / "t.jsonl", PROMPT, DONE, help, DONE, path)).happenings
    assert [happenings[2], happenings[4]] == [Asked(Ref("u5"), " /help reply with only the word pong"), Asked(Ref("u6"), "/tmp is full, clean it up")]
