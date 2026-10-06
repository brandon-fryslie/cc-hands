"""A session's transcript, followed as Claude Code writes it, and the turn it is read into."""

import asyncio
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from loguru import logger

from hands.core.events import CarriedOut, Continued, Interrupted, Launched, Progressed, Read, ReportedBack, Taken, Transcribed
from hands.core.session import Membership, PromptId, RequestId, SessionId
from hands.core.turn import AgentId, Asked, Commanded, Continuing, Interruption, Notified, Looked, Other, Ran, Ref, Said, Shelled, Turn, describe
from hands.voice.tools import TURN_SENTENCE_BUDGET
from hands.core.effects import Summarise
from hands.core.events import Attached, Joined, Prompted, StatusReported, Stopped
from hands.core import status
from hands.core.status import Report, Stamp
from hands.core.session import Idle, Opened, Told, Untold, delegating
from hands.sessions.delta import Deltas
from hands.sessions.registry import Sessions
from hands.sessions.tail import KEPT, StoodIn, Tails, Telling, keep_tailing
from hands.sessions.turning import Turning

# When hands heard a Stop, on the clock Claude Code stamps a status with.
STOP_HEARD = Stamp(1500)
STOP_REQUEST = RequestId("stop")

SID = SessionId("bf411065-dc5c-4ec9-8302-61b84bdb5c53")

# Real records from this repository's own sessions: an earlier /clear prompt, the prompt that starts the turn,
# attachments, modes, a title, thinking, text, three Bash calls with their results (the last failed),
# an injected isMeta user record, a final text, and a record still being written.
FIXTURE = Path(__file__).parent / "fixtures" / "turn.jsonl"


def member(transcript: Path) -> Membership:
    return Membership(SID, pid=4242, cwd=Path("/code/a"), transcript=transcript)


@dataclass
class Registry:
    """As much of the session registry as the tail asks about.

    A session it has heard of keeps its membership after it stops being live, which is what the real registry
    does, and what a `claude -p` session — gone the moment its turn stops — depends on to be told at all.
    """

    members: list[Membership]
    reported: bool = True

    def __post_init__(self) -> None:
        self.heard = list(self.members)

    def live_members(self) -> list[Membership]:
        return self.members

    def status_read(self, session: SessionId) -> bool:
        return self.reported

    def now(self) -> float:
        return 7.0

    def stamp(self) -> Stamp:
        return Stamp(5000)

    def membership(self, session: SessionId) -> Membership | None:
        return next((member for member in self.heard if member.id == session), None)


async def following(transcript: Path) -> Tails:
    tails = Tails(Registry([member(transcript)]))
    await tails.catch_up()
    return tails


async def followed_on(transcript: Path, *records: str) -> tuple[Tails, list[Transcribed]]:
    """A transcript hands has followed since a turn before these records, so none of them is history, and what reading them heard."""
    transcript.write_text(lines(PROMPT, DONE))
    tails = await following(transcript)
    with transcript.open("a") as more:
        more.write(lines(*records))
    return tails, heard(await tails.catch_up())


async def turn_of(transcript: Path, closing: str | None = None) -> Turn | None:
    telling = await (await following(transcript)).tell(SID, None, closing)
    return None if telling is None else telling.turn


def heard(transcribed: list[Transcribed]) -> list[Transcribed]:
    """What a catch-up says of the turns, without the readings that close each session's part of it."""
    return [event for event in transcribed if not isinstance(event, Read)]


def said_idle(at: float) -> StatusReported:
    """Claude Code setting the session idle, as its status file says it has when a turn is over however it ended."""
    return StatusReported(SID, Report(status.Idle(), Stamp(int(at * 1000))), at=at)


def said_busy(at: float) -> StatusReported:
    """Claude Code setting the session busy, as it does when a prompt is submitted: the tail reads a session once it has."""
    return StatusReported(SID, Report(status.Busy(), Stamp(int(at * 1000))), at=at)


def lines(*records: str) -> str:
    return "".join(f"{record}\n" for record in records)


PROMPT = '{"type":"user","message":{"role":"user","content":"first"}}'
CALL = '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"sleep 60"}}]}}'
RESULT = '{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"done"}]}}'
DONE = '{"type":"assistant","message":{"content":[{"type":"text","text":"Done."}]}}'
RAN = Ran(None, "sleep 60", None, failed=False, output="done", git=())


async def test_the_turn_is_everything_said_and_used_after_the_last_prompt_in_order() -> None:
    turn = await turn_of(FIXTURE)
    assert turn is not None
    assert isinstance(turn.opening, Asked) and turn.opening.text.startswith("I'd like you to go a bit further and architect a robust system")
    assert [type(step).__name__ for step in turn.steps] == ["Said", "Ran", "Ran", "Ran", "Said"]
    said, _loaded, _inspected, commented, final = turn.steps
    assert isinstance(said, Said) and said.text.startswith("I'll start by loading the repo conventions")
    assert isinstance(commented, Ran) and commented.purpose is None and commented.failed
    assert isinstance(final, Said) and final.text.startswith("API Error: Unable to connect to API")


async def test_a_prompt_sent_as_blocks_with_an_image_opens_the_turn(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    image = '{"type":"user","origin":{"kind":"human"},"message":{"role":"user","content":[{"type":"text","text":"match this"},{"type":"image","source":{}}]}}'
    transcript.write_text(lines(PROMPT, DONE, image))
    assert await turn_of(transcript) == Turn(Asked(None, "match this\n[an image]"), ())


async def test_a_notification_after_the_turn_ended_opens_a_turn_of_its_own_and_is_not_what_the_user_asked(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    notified = '{"type":"user","origin":{"kind":"task-notification"},"message":{"role":"user","content":"<task-notification>tests passed</task-notification>"}}'
    transcript.write_text(lines(PROMPT, DONE, notified, DONE))
    assert await turn_of(transcript) == Turn(Notified(None, "<task-notification>tests passed</task-notification>", None), (Said(None, "Done."),))


@pytest.mark.parametrize("kind", ["human", "task-notification"])
async def test_a_message_that_lands_while_a_tool_runs_belongs_to_the_turn_under_way(tmp_path: Path, kind: str) -> None:
    transcript = tmp_path / "t.jsonl"
    landed = f'{{"type":"user","origin":{{"kind":"{kind}"}},"message":{{"role":"user","content":"also this"}}}}'
    transcript.write_text(lines(PROMPT, CALL, RESULT, landed, DONE))
    assert await turn_of(transcript) == Turn(Asked(None, "first"), (RAN, Said(None, "Done.")))


async def test_compactions_summary_does_not_open_a_turn(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    summary = '{"type":"user","isCompactSummary":true,"message":{"role":"user","content":"This session is being continued"}}'
    transcript.write_text(lines(PROMPT, DONE, summary, DONE))
    assert await turn_of(transcript) == Turn(Asked(None, "first"), (Said(None, "Done."), Said(None, "Done.")))


async def test_a_prompt_with_a_document_attached_opens_the_turn_and_names_the_document_rather_than_its_bytes(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    document = '{"type":"user","message":{"role":"user","content":[{"type":"document","source":{"data":"JVBERi0x"}},{"type":"text","text":"read this"}]}}'
    transcript.write_text(lines(PROMPT, DONE, document))
    assert await turn_of(transcript) == Turn(Asked(None, "[a document]\nread this"), ())


async def test_a_transcript_with_no_prompt_yet_has_no_turn(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('{"type":"ai-title","aiTitle":"x"}\n{"type":"user","isMeta":true,"message":{"role":"user","content":"injected"}}\n')
    assert await (await following(transcript)).tell(SID, None, None) is None


async def test_a_transcript_that_cannot_be_read_when_a_turn_is_told_is_a_failure_the_narrator_says(tmp_path: Path) -> None:
    """A session that has written no transcript at all is normal while it is quiet, and wrong the moment it stops."""
    tails = await following(tmp_path / "unwritten.jsonl")
    with pytest.raises(FileNotFoundError):
        await tails.tell(SID, None, None)


async def test_a_stop_from_a_session_the_registry_does_not_list_tells_nothing(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE))
    assert await Tails(Registry([])).tell(SID, None, None) is None


async def test_a_session_that_stops_before_the_catch_up_has_reached_it_is_still_told(tmp_path: Path) -> None:
    """A turn takes seconds and the catch-up runs ten times a second, but a Stop never depends on having won that race."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE))
    telling = await Tails(Registry([member(transcript)])).tell(SID, None, None)
    assert telling is not None and telling.turn == Turn(Asked(None, "first"), (Said(None, "Done."),))


async def test_a_call_whose_result_never_came_is_shown_as_having_none(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        '{"type":"user","message":{"role":"user","content":"go"}}\n'
        '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"/a/b.py"}}]}}\n'
    )
    assert await turn_of(transcript) == Turn(Asked(None, "go"), (Looked(None, "Read", "/a/b.py", "(no result)"),))


async def test_only_what_was_appended_since_the_last_reading_is_read_again(tmp_path: Path) -> None:
    """The point of a tail: a turn is built as it is written, not re-derived from a file that grows all day."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE))
    tails = await following(transcript)
    first = await tails.tell(SID, None, None)
    assert first is not None and first.turn == Turn(Asked(None, "first"), (Said(None, "Done."),))
    await tails.spoken(first)
    with transcript.open("a") as more:
        more.write(lines(CALL, RESULT, DONE))
    second = await tails.tell(SID, None, None)
    assert second is not None and second.turn == Turn(Asked(None, "first"), (RAN, Said(None, "Done.")), Continuing(1))


async def test_a_record_only_half_written_is_not_read_until_the_newline_that_ends_it_is(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT) + DONE[:40])
    tails = await following(transcript)
    first = await tails.tell(SID, None, None)
    assert first is not None and first.turn == Turn(Asked(None, "first"), ())
    transcript.write_text(lines(PROMPT, DONE))
    second = await tails.tell(SID, None, None)
    assert second is not None and second.turn == Turn(Asked(None, "first"), (Said(None, "Done."),))


async def test_a_transcript_that_grew_shorter_is_read_again_from_its_start(tmp_path: Path) -> None:
    """Nothing Claude Code does cuts a transcript short, so this is the loud way to be wrong rather than a silent one."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, CALL, RESULT, DONE))
    tails = await following(transcript)
    assert await tails.tell(SID, None, None) is not None
    warnings: list[str] = []
    sink = logger.add(lambda message: warnings.append(message.record["message"]), level="WARNING", filter="hands")
    try:
        transcript.write_text(lines(PROMPT, DONE))
        again = await tails.tell(SID, None, None)
    finally:
        logger.remove(sink)
    assert again is not None and again.turn == Turn(Asked(None, "first"), (Said(None, "Done."),))
    assert warnings and "shorter than what was read of it" in warnings[0]


async def test_a_line_that_is_not_a_record_is_said_and_skipped_and_the_rest_of_the_turn_is_still_told(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, '{"type":"assistant","message":', DONE))
    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(message.record["message"]), level="ERROR", filter="hands")
    try:
        turn = await turn_of(transcript)
    finally:
        logger.remove(sink)
    assert turn == Turn(Asked(None, "first"), (Said(None, "Done."),))
    assert errors and "could not be read, so it is not told" in errors[0]


async def test_a_record_whose_message_is_not_an_object_is_skipped_and_the_rest_of_its_reading_is_still_told(tmp_path: Path) -> None:
    """Whole JSON can still be no record. It is refused where a line becomes a record, so that reading a turn out
    of it cannot raise part-way through a record already half consumed — which stops the daemon on a transcript
    the next run would read again and stop on again."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, '{"type":"assistant","message":"oops"}', DONE))
    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(message.record["message"]), level="ERROR", filter="hands")
    try:
        # The catch-up reads it too, and the loop it runs in is what a raise here would stop.
        tails = await following(transcript)
        telling = await tails.tell(SID, None, None)
    finally:
        logger.remove(sink)
    assert telling is not None and telling.turn == Turn(Asked(None, "first"), (Said(None, "Done."),))
    assert errors and "should be an object" in errors[0]


async def test_a_reading_picks_up_after_the_steps_already_told(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE))
    tails = await following(transcript)
    first = await tails.tell(SID, None, None)
    assert first is not None and first == Telling(SID, Turn(Asked(None, "first"), (Said(None, "Done."),)), number=1, through=1, ends_on="Done.", transcript=transcript)
    await tails.spoken(first)
    assert (await tails.tell(SID, None, None)) == Telling(SID, Turn(Asked(None, "first"), (), Continuing(1)), number=1, through=1, ends_on="Done.", transcript=transcript)


async def test_a_turn_told_but_never_spoken_is_told_again(tmp_path: Path) -> None:
    """The narrator marks a turn told only once the model has summarised it, so a failed summary loses nothing."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE))
    tails = await following(transcript)
    first = await tails.tell(SID, None, None)
    assert first is not None
    assert (await tails.tell(SID, None, None)) == first


async def test_the_hooks_closing_reply_ends_a_turn_whose_transcript_does_not_hold_it_yet_and_is_not_doubled_when_it_does(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, CALL, RESULT))
    tails = await following(transcript)
    first = await tails.tell(SID, None, "Done.")
    assert first is not None and first.turn == Turn(Asked(None, "first"), (RAN, Said(None, "Done.")))
    await tails.spoken(first)
    with transcript.open("a") as more:
        more.write(lines(DONE))
    second = await tails.tell(SID, None, "Done.")
    assert second is not None and second.turn == Turn(Asked(None, "first"), (), Continuing(2))


async def test_a_closing_reply_is_matched_to_its_record_however_the_whitespace_around_it_differs(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    padded = '{"type":"assistant","message":{"content":[{"type":"text","text":"  Done.\\n"}]}}'
    transcript.write_text(lines(PROMPT))
    tails = await following(transcript)
    first = await tails.tell(SID, None, "Done.")
    assert first is not None and first.ends_on == StoodIn("Done.")
    await tails.spoken(first)
    with transcript.open("a") as more:
        more.write(lines(padded))
    second = await tails.tell(SID, None, "Done.")
    assert second is not None and second.turn == Turn(Asked(None, "first"), (), Continuing(1))


async def test_a_closing_reply_the_turn_said_once_before_still_ends_it(tmp_path: Path) -> None:
    """The record of the reply is the one the turn ends on, so an earlier reply in the same words does not stand for it."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE, CALL, RESULT))
    assert await turn_of(transcript, closing="Done.") == Turn(Asked(None, "first"), (Said(None, "Done."), RAN, Said(None, "Done.")))


async def test_a_reply_a_later_turn_repeats_is_told_again_because_a_turn_is_told_only_what_it_did(tmp_path: Path) -> None:
    """Claude says the same short thing twice in a row: the second turn is its own, and is heard."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT))
    tails = await following(transcript)
    first = await tails.tell(SID, None, "Nothing to do.")
    assert first is not None and first.turn == Turn(Asked(None, "first"), (Said(None, "Nothing to do."),))
    await tails.spoken(first)
    with transcript.open("a") as more:
        more.write(lines(DONE, '{"type":"user","message":{"role":"user","content":"check again"}}'))
    second = await tails.tell(SID, None, "Nothing to do.")
    assert second is not None and second.turn == Turn(Asked(None, "check again"), (Said(None, "Nothing to do."),))


async def test_a_turn_told_while_the_next_one_opened_marks_nothing_of_the_next(tmp_path: Path) -> None:
    """Summarising takes a second of model time, which is long enough for the user to have asked something else."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE))
    tails = await following(transcript)
    first = await tails.tell(SID, None, None)
    assert first is not None
    with transcript.open("a") as more:
        more.write(lines('{"type":"user","message":{"role":"user","content":"second"}}', CALL, RESULT, DONE))
    # The summary of the first turn only now comes back, and its mark is not this turn's.
    assert (await tails.tell(SID, None, None)) is not None
    await tails.spoken(first)
    second = await tails.tell(SID, None, None)
    assert second is not None and second.turn == Turn(Asked(None, "second"), (RAN, Said(None, "Done.")))


async def test_a_session_the_registry_stops_listing_is_let_go_of_with_the_turn_it_was_holding(tmp_path: Path) -> None:
    """What the catch-up holds is bounded by the sessions that are live, so an ended session's turn — every byte
    of output it printed — is not held for as long as the daemon runs.

    A Stop that arrives afterwards is still told, from the transcript read again: that is what a `claude -p`
    session, gone the moment its turn stops, depends on. The turn it is told is that transcript's last, whether
    or not the session heard it before — a repeat, where letting the turn go the other way round would be a
    silence, and the same turn twice is the one of those two the user can do something about.
    """
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, CALL, RESULT, DONE))
    registry = Registry([member(transcript)])
    tails = Tails(registry)
    await tails.catch_up()
    told = await tails.tell(SID, None, None)
    assert told is not None and told.turn == Turn(Asked(None, "first"), (RAN, Said(None, "Done.")))
    await tails.spoken(told)
    registry.members.clear()
    await tails.catch_up()
    again = await tails.tell(SID, None, None)
    assert again is not None and again.turn == Turn(Asked(None, "first"), (RAN, Said(None, "Done.")))


async def test_a_session_that_exits_before_its_turn_is_told_is_still_told_all_of_it(tmp_path: Path) -> None:
    """`claude -p` is gone the moment its turn stops, and the tail has read that turn long before the Stop."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, CALL, RESULT, DONE))
    registry = Registry([member(transcript)])
    tails = Tails(registry)
    await tails.catch_up()
    registry.members.clear()
    await tails.catch_up()
    telling = await tails.tell(SID, None, None)
    assert telling is not None and telling.turn == Turn(Asked(None, "first"), (RAN, Said(None, "Done.")))


async def test_a_record_carrying_results_for_several_calls_hands_its_own_record_to_none_of_them(tmp_path: Path) -> None:
    """`toolUseResult` describes one call. Given to two, it would say one edit wrote the file the other did."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        lines(
            PROMPT,
            '{"type":"assistant","message":{"content":['
            '{"type":"tool_use","id":"t1","name":"Edit","input":{"file_path":"/a/one.py"}},'
            '{"type":"tool_use","id":"t2","name":"Edit","input":{"file_path":"/a/two.py"}}]}}',
            '{"type":"user","toolUseResult":{"filePath":"/a/one.py","structuredPatch":'
            '[{"oldStart":1,"oldLines":1,"newStart":1,"newLines":1,"lines":["-a","+b"]}]},'
            '"message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"},'
            '{"type":"tool_result","tool_use_id":"t2","content":"ok"}]}}',
        )
    )
    turn = await turn_of(transcript)
    # Neither is an `Edited`: an edit asserts the patch it made, and that record says whose patch it holds for neither.
    assert turn is not None and len(turn.steps) == 2 and all(isinstance(step, Other) for step in turn.steps)


async def test_what_a_turn_was_told_is_marked_only_while_nothing_else_is_reading(tmp_path: Path) -> None:
    """A summary takes seconds to come back, and a worker thread can be part-way through forgetting the turn it
    was about by then. The mark waits on the reading, so it never lands between the halves of a forgetting."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE))
    tails = await following(transcript)
    told = await tails.tell(SID, None, None)
    assert told is not None
    async with tails._reading:  # pyright: ignore[reportPrivateUsage]
        marking = asyncio.create_task(tails.spoken(told))
        await asyncio.sleep(0)
        assert not marking.done()
    await marking
    assert (await tails.tell(SID, None, None)) == Telling(SID, Turn(Asked(None, "first"), (), Continuing(1)), number=1, through=1, ends_on="Done.", transcript=transcript)


def call(id: str, output: str) -> tuple[str, str]:
    """A Bash call and its result, as two records."""
    used = f'{{"type":"assistant","message":{{"content":[{{"type":"tool_use","id":"{id}","name":"Bash","input":{{"command":"cat big"}}}}]}}}}'
    answered = f'{{"type":"user","message":{{"role":"user","content":[{{"type":"tool_result","tool_use_id":"{id}","content":"{output}"}}]}}}}'
    return used, answered


async def marked(tails: Tails, telling: Telling) -> list[str]:
    """What marking the telling spoken says of it."""
    said: list[str] = []
    sink = logger.add(lambda message: said.append(message.record["message"]), level="DEBUG", filter="hands.sessions.tail")
    try:
        await tails.spoken(telling)
    finally:
        logger.remove(sink)
    return said


async def test_a_long_turn_told_and_spoken_keeps_none_of_its_output_while_the_session_is_quiet(tmp_path: Path) -> None:
    """A session that goes quiet after a long turn holds none of it once it is heard, and the next telling of the same
    turn is only what came after, counted on from what was told."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, *(record for n in range(200) for record in call(f"c{n}", "x" * 5000))))
    tails = await following(transcript)
    told = await tails.tell(SID, None, None)
    assert told is not None and len(told.turn.steps) == 200
    assert await marked(tails, told) == [f"session {SID} turn 1 was told through step 200: let go of 200 steps, 0 of them calls with no result yet, 0 untold are held"]
    held = tails._following[SID].reading.turn  # pyright: ignore[reportPrivateUsage]
    assert (held.slots, held.calls, held.places) == ([], {}, {})
    with transcript.open("a") as more:
        more.write(lines(DONE))
    after = await tails.tell(SID, None, "Done.")
    assert after is not None and after.turn == Turn(Asked(None, "first"), (Said(None, "Done."),), Continuing(200)) and after.through == 201


async def test_a_call_told_before_its_result_came_is_not_told_again_when_the_result_lands(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, CALL))
    tails = await following(transcript)
    told = await tails.tell(SID, None, None)
    assert told is not None and told.turn.steps == (Ran(None, "sleep 60", None, failed=False, output="(no result)", git=()),)
    assert await marked(tails, told) == [f"session {SID} turn 1 was told through step 1: let go of 1 steps, 1 of them calls with no result yet, 0 untold are held"]
    with transcript.open("a") as more:
        more.write(lines(RESULT, DONE))
    after = await tails.tell(SID, None, None)
    assert after is not None and after.turn == Turn(Asked(None, "first"), (Said(None, "Done."),), Continuing(1))


async def test_a_second_stop_before_the_told_replys_record_lands_tells_nothing_again(tmp_path: Path) -> None:
    """The reply was told from the hook's copy, and the steps before it were let go of: what the turn ended on is still known."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, CALL, RESULT))
    tails = await following(transcript)
    first = await tails.tell(SID, None, "Done.")
    assert first is not None
    assert await marked(tails, first) == [f"session {SID} turn 1 was told through step 1 on the hook's copy of its reply: let go of 1 steps, 0 of them calls with no result yet, 0 untold are held"]
    again = await tails.tell(SID, None, "Done.")
    assert again is not None and again.turn == Turn(Asked(None, "first"), (), Continuing(1))
    await tails.spoken(again)
    with transcript.open("a") as more:
        more.write(lines(DONE))
    landed = await tails.tell(SID, None, "Done.")
    assert landed is not None and landed.turn == Turn(Asked(None, "first"), (), Continuing(2))


def test_a_mark_behind_what_was_let_go_of_is_refused_rather_than_letting_go_of_untold_steps() -> None:
    turning = Turning()
    turning.slots = [Said(None, "one"), Said(None, "two")]
    turning.forget(1)
    with pytest.raises(ValueError, match="1 were let go of"):
        turning.forget(0)
    assert turning.steps() == [Said(None, "two")]


async def test_a_session_registered_again_is_told_from_the_transcript_it_has_now(tmp_path: Path) -> None:
    """Where a session's transcript is, is the registry's to say. Registered again, it writes a new one, and
    reading the file it used to write would follow a session that has stopped writing and say nothing ever again."""
    first = tmp_path / "a.jsonl"
    first.write_text(lines(PROMPT, DONE))
    registry = Registry([member(first)])
    tails = Tails(registry)
    told = await tails.tell(SID, None, None)
    assert told is not None
    await tails.spoken(told)
    second = tmp_path / "b.jsonl"
    second.write_text(lines('{"type":"user","message":{"role":"user","content":"again"}}', DONE))
    # The registry now knows the session by its new transcript, and by nothing of the old one.
    registry.members[:] = [member(second)]
    registry.heard[:] = [member(second)]
    await tails.catch_up()
    telling = await tails.tell(SID, None, None)
    assert telling is not None and telling.turn == Turn(Asked(None, "again"), (Said(None, "Done."),))


async def test_the_catch_up_and_a_stop_never_read_the_same_bytes_twice(tmp_path: Path) -> None:
    """Both read off the loop in a thread, and a Stop reads the transcript the catch-up is already reading."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE, CALL, RESULT, DONE))
    tails = Tails(Registry([member(transcript)]))
    await asyncio.gather(*(tails.catch_up() for _ in range(8)), tails.tell(SID, None, None), tails.tell(SID, None, None))
    telling = await tails.tell(SID, None, None)
    # Read twice, every step would be here twice over.
    assert telling is not None
    assert telling.turn == Turn(Asked(None, "first"), (Said(None, "Done."), RAN, Said(None, "Done.")))


# An interrupt fires no hook (2.1.281). Claude Code writes these records instead, as captured from a live session.
ASKED = '{"type":"user","promptId":"p1","message":{"role":"user","content":"Write an essay about rivers."}}'
WRITING = '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"# Rivers"}]}}'
CUT_OFF = '{"type":"user","promptId":"p1","uuid":"u9","message":{"role":"user","content":[{"type":"text","text":"[Request interrupted by user]"}]}}'
LOOPING = '{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"toolu_9","name":"Bash","input":{"command":"for i in $(seq 60); do date; sleep 1; done"}}]}}'
REJECTED = (
    '{"type":"user","promptId":"p1","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_9","is_error":true,'
    '"content":"The user doesn\'t want to proceed with this tool use. The tool use was rejected."}]}}'
)
CUT_OFF_MID_TOOL = '{"type":"user","promptId":"p1","uuid":"u9","message":{"role":"user","content":[{"type":"text","text":"[Request interrupted by user for tool use]"}]}}'


async def test_a_turn_cut_off_while_claude_wrote_is_heard_as_interrupted_and_told_with_what_it_had_said(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    tails, read = await followed_on(transcript, ASKED, WRITING, CUT_OFF)
    assert read == [Taken(SID, PromptId("p1"), None, 7.0), Interrupted(SID, PromptId("p1"), at=7.0)]
    telling = await tails.tell(SID, None, None)
    # The record of the interrupt is written as a user's message, and is not read as the next thing asked.
    assert telling is not None and telling.turn == Turn(Asked(None, "Write an essay about rivers."), (Said(None, "# Rivers"), Interruption(Ref("u9"))))


async def test_a_turn_cut_off_while_a_tool_ran_is_heard_as_interrupted(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    tails, read = await followed_on(transcript, ASKED, LOOPING, REJECTED, CUT_OFF_MID_TOOL)
    assert [event for event in read if not isinstance(event, Progressed)] == [Taken(SID, PromptId("p1"), None, 7.0), Interrupted(SID, PromptId("p1"), at=7.0)]
    telling = await tails.tell(SID, None, None)
    assert telling is not None and telling.turn.steps[-1] == Interruption(Ref("u9"))


async def test_the_prompt_after_an_interrupt_opens_a_turn_of_its_own(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, WRITING, CUT_OFF, '{"type":"user","promptId":"p2","message":{"role":"user","content":"Shorter."}}', DONE))
    assert await turn_of(transcript) == Turn(Asked(None, "Shorter."), (Said(None, "Done."),))


async def test_a_prompt_that_only_mentions_an_interrupt_is_a_prompt(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    quoting = '{"type":"user","promptId":"p1","message":{"role":"user","content":"Why did I see [Request interrupted by user] there?"}}'
    transcript.write_text(lines(quoting))
    tails = Tails(Registry([member(transcript)]))
    assert heard(await tails.catch_up()) == [Taken(SID, PromptId("p1"), None, 7.0)]
    telling = await tails.tell(SID, None, None)
    assert telling is not None and telling.turn.opening == Asked(None, "Why did I see [Request interrupted by user] there?")


@pytest.mark.parametrize(("stamp", "written"), [('"2026-09-24T17:48:23.455Z"', Stamp(1790272103455)), (None, None)])
async def test_a_prompt_is_taken_with_when_claude_code_wrote_it(tmp_path: Path, stamp: str | None, written: Stamp | None) -> None:
    """On the clock Claude Code stamps a status with, so an idle it set after the record can be told from one before."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED if stamp is None else ASKED.replace('"promptId"', f'"timestamp":{stamp},"promptId"')))
    assert heard(await Tails(Registry([member(transcript)])).catch_up()) == [Taken(SID, PromptId("p1"), written, 7.0)]


@pytest.mark.parametrize("stamp", ['"yesterday"', '"2026-09-24T17:48:23.455"', "1790272103455"])
async def test_a_record_whose_time_cannot_be_read_is_said_read_as_having_none_and_still_told(tmp_path: Path, stamp: str) -> None:
    """A time with no zone would be read as this machine's, hours off the epoch Claude Code stamps a status in."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED.replace('"promptId"', f'"timestamp":{stamp},"promptId"'), WRITING))
    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(message.record["message"]), level="ERROR", filter="hands")
    try:
        tails = Tails(Registry([member(transcript)]))
        assert heard(await tails.catch_up()) == [Taken(SID, PromptId("p1"), None, 7.0)]
    finally:
        logger.remove(sink)
    assert len(errors) == 1 and "has a time that cannot be read" in errors[0]
    telling = await tails.tell(SID, None, None)
    assert telling is not None and telling.turn == Turn(Asked(None, "Write an essay about rivers."), (Said(None, "# Rivers"),))


async def test_an_interrupt_the_stops_own_reading_found_is_still_handed_out_at_the_next_catch_up(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, WRITING))
    tails = await following(transcript)
    with transcript.open("a") as more:
        more.write(lines(CUT_OFF))
    await tails.tell(SID, None, None)
    assert heard(await tails.catch_up()) == [Interrupted(SID, PromptId("p1"), at=7.0)]
    assert heard(await tails.catch_up()) == []


async def test_an_interrupt_whose_record_names_no_turn_is_said_and_ends_nothing(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(message.record["message"]), level="ERROR", filter="hands")
    try:
        assert (await followed_on(transcript, ASKED, WRITING, CUT_OFF.replace('"promptId":"p1",', "")))[1] == [Taken(SID, PromptId("p1"), None, 7.0)]
    finally:
        logger.remove(sink)
    assert errors == [f"session {SID} was interrupted, but the record of it names no prompt, so its turn is told without it"]


async def test_the_older_record_of_an_interrupt_written_as_a_plain_string_is_one_too(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    plain = '{"type":"user","promptId":"p1","message":{"role":"user","content":"[Request interrupted by user for tool use]"}}'
    assert (await followed_on(transcript, ASKED, WRITING, plain))[1] == [Taken(SID, PromptId("p1"), None, 7.0), Interrupted(SID, PromptId("p1"), at=7.0)]


# A message queued while the loop ran, flushed by Escape, as captured live on 2.1.281: the cancelled call's result and the
# interrupt already carry the queued message's own new id, and so does everything Claude answers after it.
FLUSHED = REJECTED.replace('"p1"', '"p2"')
FLUSHING = CUT_OFF_MID_TOOL.replace('"p1"', '"p2"')
QUEUED = '{"type":"user","promptId":"p2","message":{"role":"user","content":"also say banana"}}'
BANANA = '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"banana"}]}}'


async def test_a_turn_that_goes_on_under_a_queued_prompt_is_heard_to_after_the_interrupt_that_flushed_it(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, LOOPING, FLUSHED, FLUSHING, QUEUED, BANANA))
    tails = Tails(Registry([member(transcript)]))
    assert heard(await tails.catch_up()) == [Taken(SID, PromptId("p1"), None, 7.0), Taken(SID, PromptId("p2"), None, 7.0), Interrupted(SID, PromptId("p2"), at=7.0), Continued(SID, was=PromptId("p1"), now=PromptId("p2"))]


async def test_a_queued_command_taken_in_mid_turn_is_heard_from_the_results_claude_answers(tmp_path: Path) -> None:
    """As in a transcript on this machine: a queued /rate-limit-options writes no prompt, and the turn goes on under its id."""
    transcript = tmp_path / "t.jsonl"
    ran = '{"type":"user","promptId":"p2","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_9","content":"done"}]}}'
    transcript.write_text(lines(ASKED, LOOPING, ran, WRITING))
    assert heard(await Tails(Registry([member(transcript)])).catch_up()) == [Taken(SID, PromptId("p1"), None, 7.0), Taken(SID, PromptId("p2"), None, 7.0), Continued(SID, was=PromptId("p1"), now=PromptId("p2"))]


async def test_an_escape_in_the_turn_a_queued_prompt_went_on_as_leaves_the_session_idle(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, LOOPING))
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(member(transcript), "startup"))
    await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p1")))
    await sessions.apply(said_busy(at=1.0))
    tails = Tails(sessions)
    with transcript.open("a") as more:
        more.write(lines(FLUSHED, FLUSHING, QUEUED, LOOPING, REJECTED.replace('"p1"', '"p2"')))
    for transcribed in await tails.catch_up():
        await sessions.apply(transcribed)
    # Claude Code sets idle ~100 ms before it writes the record of the interrupt (2.1.282).
    await sessions.apply(said_idle(at=2.0))
    with transcript.open("a") as more:
        more.write(lines(CUT_OFF_MID_TOOL.replace('"p1"', '"p2"')))
    for transcribed in await tails.catch_up():
        await sessions.apply(transcribed)
    assert await asyncio.wait_for(sessions.story(), 5.0) == Summarise(SID, PromptId("p2"), None)
    live = sessions.live_session(SID)
    assert live is not None and isinstance(live.state, Idle)


async def test_the_stand_in_claude_code_writes_after_a_question_it_stopped_is_not_what_claude_said(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    stand_in = '{"type":"assistant","message":{"model":"<synthetic>","role":"assistant","content":[{"type":"text","text":"No response requested."}]}}'
    transcript.write_text(lines(ASKED, WRITING, CUT_OFF, stand_in))
    telling = await (await following(transcript)).tell(SID, None, None)
    assert telling is not None and telling.turn.steps == (Said(None, "# Rivers"), Interruption(Ref("u9")))


async def test_a_record_that_names_no_prompt_does_not_lose_the_turn_claude_is_answering(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    unnamed = '{"type":"user","isMeta":true,"message":{"role":"user","content":"injected"}}'
    ran = '{"type":"user","promptId":"p2","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_9","content":"done"}]}}'
    transcript.write_text(lines(ASKED, WRITING, unnamed, LOOPING, ran, WRITING))
    assert heard(await Tails(Registry([member(transcript)])).catch_up()) == [Taken(SID, PromptId("p1"), None, 7.0), Taken(SID, PromptId("p2"), None, 7.0), Continued(SID, was=PromptId("p1"), now=PromptId("p2"))]


async def test_a_prompt_cancelled_while_its_hooks_ran_ends_at_the_idle_and_the_one_resent_opens_its_own_turn(tmp_path: Path) -> None:
    """Escape during UserPromptSubmit writes nothing (2.1.281): the prompt's hook opens its turn, and Claude Code's idle
    ends it, so the prompt sent again is a turn of its own."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("")
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(member(transcript), "startup"))
    await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p0")))
    tails = Tails(sessions)
    for _ in range(3):
        for transcribed in await tails.catch_up():
            await sessions.apply(transcribed)
    # The Escape puts the prompt back in the box and sets the session idle (2.1.282).
    await sessions.apply(said_idle(at=2.0))
    live = sessions.live_session(SID)
    assert live is not None and isinstance(live.state, Idle) and isinstance(live.turn, Untold)
    await sessions.apply(Prompted(SID, at=4.0, mode=None, prompt=PromptId("p1")))
    transcript.write_text(lines(ASKED))
    for transcribed in await tails.catch_up():
        await sessions.apply(transcribed)
    live = sessions.live_session(SID)
    assert live is not None and live.turn == Opened(PromptId("p1"))


async def test_a_prompt_whose_first_record_is_its_interrupt_is_heard_taken_before_it_is_heard_stopped(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    assert (await followed_on(transcript, CUT_OFF))[1] == [Taken(SID, PromptId("p1"), None, 7.0), Interrupted(SID, PromptId("p1"), at=7.0)]


async def test_the_tail_hands_each_interrupt_it_reads_to_the_registry_which_tells_the_turn_claude_code_said_is_over(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, WRITING))
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(member(transcript), "startup"))
    await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p1")))
    tailing = asyncio.create_task(keep_tailing(Tails(sessions), 0.01, sessions.apply))
    try:
        await sessions.apply(said_idle(at=2.0))
        with transcript.open("a") as more:
            more.write(lines(CUT_OFF))
        assert await asyncio.wait_for(sessions.story(), 5.0) == Summarise(SID, PromptId("p1"), None)
    finally:
        tailing.cancel()
    live = sessions.live_session(SID)
    assert live is not None and isinstance(live.state, Idle)


# The next prompt, read before the turn before it is told: the narrator is seconds behind on another session's summary.
NEXT_ASKED = '{"type":"user","promptId":"p2","message":{"role":"user","content":"Shorter."}}'
RIVERS = Turn(Asked(None, "Write an essay about rivers."), (Said(None, "# Rivers"), Interruption(Ref("u9"))))


async def test_an_interrupted_turn_is_told_as_itself_after_the_next_prompt_was_read(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, WRITING, CUT_OFF, NEXT_ASKED, DONE))
    tails = await following(transcript)
    first = await tails.tell(SID, PromptId("p1"), None)
    assert first is not None and first.turn == RIVERS
    await tails.spoken(first)
    second = await tails.tell(SID, PromptId("p2"), "Done.")
    assert second is not None and second.turn == Turn(Asked(None, "Shorter."), (Said(None, "Done."),))


async def test_a_stopped_turn_is_told_as_itself_after_the_next_prompt_was_read(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, WRITING, NEXT_ASKED))
    tails = await following(transcript)
    telling = await tails.tell(SID, PromptId("p1"), "# Rivers")
    assert telling is not None and telling.turn == Turn(Asked(None, "Write an essay about rivers."), (Said(None, "# Rivers"),))


async def test_a_turn_told_after_the_next_opened_is_marked_told_on_itself_and_let_go(tmp_path: Path) -> None:
    """The mark lands on the turn the telling was made of, which has no more records coming, so it is let go of; the
    turn after it is still whole."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, WRITING, CUT_OFF))
    tails = await following(transcript)
    first = await tails.tell(SID, PromptId("p1"), None)
    assert first is not None
    with transcript.open("a") as more:
        more.write(lines(NEXT_ASKED, DONE))
    await tails.catch_up()
    await tails.spoken(first)
    assert await tails.tell(SID, PromptId("p1"), None) is None
    after = await tails.tell(SID, PromptId("p2"), None)
    assert after is not None and after.turn == Turn(Asked(None, "Shorter."), (Said(None, "Done."),))


async def test_an_ended_turn_whose_summary_failed_is_told_whole_again(tmp_path: Path) -> None:
    """Nothing marked it told, and telling it let go of nothing but the turns before it."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, WRITING, CUT_OFF, NEXT_ASKED, DONE))
    tails = await following(transcript)
    first = await tails.tell(SID, PromptId("p1"), None)
    assert first is not None and first == await tails.tell(SID, PromptId("p1"), None)


async def test_only_the_last_turns_that_ended_are_kept(tmp_path: Path) -> None:
    """Read from its start, a transcript ends every turn before the daemon attached untold."""
    transcript = tmp_path / "t.jsonl"
    asked = [f'{{"type":"user","promptId":"q{n}","message":{{"role":"user","content":"turn {n}"}}}}' for n in range(KEPT + 3)]
    transcript.write_text(lines(*asked))
    tails = await following(transcript)
    assert await tails.tell(SID, PromptId("q1"), None) is None
    kept = await tails.tell(SID, PromptId("q3"), None)
    assert kept is not None and kept.turn.opening == Asked(None, "turn 3")


async def test_a_turn_is_named_by_the_id_it_went_on_under_after_a_flush(tmp_path: Path) -> None:
    """In the order a flush was written live on 2.1.281: the queued message straight after the call it cut off."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, LOOPING, QUEUED, FLUSHED, FLUSHING, BANANA))
    tails = await following(transcript)
    by_opening, by_flush = await tails.tell(SID, PromptId("p1"), None), await tails.tell(SID, PromptId("p2"), None)
    assert by_opening is not None and by_opening == by_flush


async def test_a_turn_no_record_names_is_not_told_and_said(tmp_path: Path) -> None:
    """Another turn told in its place would be the very thing a named turn is for; nothing is told instead."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, WRITING, CUT_OFF, NEXT_ASKED, DONE))
    tails = await following(transcript)
    warnings: list[str] = []
    sink = logger.add(lambda message: warnings.append(message.record["message"]), level="WARNING", filter="hands")
    try:
        assert await tails.tell(SID, PromptId("p9"), None) is None
    finally:
        logger.remove(sink)
    assert warnings == [f"no turn kept from {transcript} carries prompt p9, so there is nothing to tell of it"]


async def test_a_turn_ended_unheard_is_told_as_itself_whatever_order_the_prompt_and_its_records_arrive_in(tmp_path: Path) -> None:
    """The adversarial order, end to end: p2's hook is applied after Claude Code said p1 is over and before the tail has
    read p1's interrupt, and the narrator gets to p1 only after the tail has read p2's opening. p1 is told as itself,
    interrupt and all; p2 only its own steps."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, WRITING))
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(member(transcript), "startup"))
    await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p1")))
    tails = Tails(sessions)
    for transcribed in await tails.catch_up():
        await sessions.apply(transcribed)
    await sessions.apply(said_idle(at=4.0))
    await sessions.apply(Prompted(SID, at=5.0, mode=None, prompt=PromptId("p2")))
    with transcript.open("a") as more:
        more.write(lines(CUT_OFF, NEXT_ASKED, DONE))
    for transcribed in await tails.catch_up():
        await sessions.apply(transcribed)
    await sessions.apply(Stopped(SID, "Done.", mode=None, prompt=PromptId("p2"), again=False, heard=STOP_HEARD, request=STOP_REQUEST))

    ended = await asyncio.wait_for(sessions.story(), 2.0)
    assert ended == Summarise(SID, PromptId("p1"), None)
    first = await tails.tell(SID, PromptId("p1"), None)
    assert first is not None and first.turn == RIVERS
    await tails.spoken(first)
    stopped = await asyncio.wait_for(sessions.story(), 2.0)
    assert stopped == Summarise(SID, PromptId("p2"), "Done.")
    second = await tails.tell(SID, PromptId("p2"), "Done.")
    assert second is not None and second.turn == Turn(Asked(None, "Shorter."), (Said(None, "Done."),))
    live = sessions.live_session(SID)
    assert live is not None and isinstance(live.turn, Told) and live.turn.turn == PromptId("p2")


async def test_a_turn_taken_and_ended_before_the_tail_read_any_of_it_is_told_as_itself_with_its_own_changes(tmp_path: Path) -> None:
    """hands-keyboard-gxr.90g, end to end: p1 is taken, writes a file, and is interrupted, Claude Code says so, and p2's
    hook is applied and marked, all before the tail reads a record of p1. p1 is told as itself, with the file it wrote
    and not the one p2 goes on to write, and p2 is told after it, with only its own."""
    root = tmp_path / "work"
    root.mkdir()
    for args in (("init", "-q"), ("config", "user.email", "t@example.com"), ("config", "user.name", "Test"), ("commit", "-q", "--allow-empty", "-m", "first")):
        subprocess.run(("git", "-C", str(root), *args), check=True)
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("")
    deltas = Deltas(record=lambda _entry: None, inherited=os.environ)
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None, changes=deltas)
    await sessions.apply(Joined(Membership(SID, pid=4242, cwd=root, transcript=transcript), "startup"))
    tails = Tails(sessions)
    await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p1")))
    (root / "rivers.md").write_text("# Rivers\n")
    await sessions.apply(said_idle(at=4.0))
    await sessions.apply(Prompted(SID, at=5.0, mode=None, prompt=PromptId("p2")))
    (root / "shorter.md").write_text("Rivers.\n")
    transcript.write_text(lines(ASKED, WRITING, CUT_OFF, NEXT_ASKED, DONE))
    for transcribed in await tails.catch_up():
        await sessions.apply(transcribed)
    await sessions.apply(Stopped(SID, "Done.", mode=None, prompt=PromptId("p2"), again=False, heard=STOP_HEARD, request=STOP_REQUEST))
    # Both readings are taken before anything is asserted, so a failure leaves no git command running.
    changed = [[file.path for file in (await deltas.taken(SID)).files] for _ in range(2)]

    assert await asyncio.wait_for(sessions.story(), 2.0) == Summarise(SID, PromptId("p1"), None)
    first = await tails.tell(SID, PromptId("p1"), None)
    assert first is not None and first.turn == RIVERS
    await tails.spoken(first)
    assert await asyncio.wait_for(sessions.story(), 2.0) == Summarise(SID, PromptId("p2"), "Done.")
    second = await tails.tell(SID, PromptId("p2"), "Done.")
    assert second is not None and second.turn == Turn(Asked(None, "Shorter."), (Said(None, "Done."),))
    assert changed == [["rivers.md"], ["shorter.md"]]
    live = sessions.live_session(SID)
    assert live is not None and isinstance(live.turn, Told) and live.turn.turn == PromptId("p2")


async def test_a_telling_that_names_no_turn_lets_go_of_none_that_ended(tmp_path: Path) -> None:
    """A Stop with no prompt_id says nothing of which turn ended, so the turns kept for their tellings stay kept."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, WRITING, CUT_OFF, NEXT_ASKED, DONE))
    tails = await following(transcript)
    assert await tails.tell(SID, None, None) is not None
    kept = await tails.tell(SID, PromptId("p1"), None)
    assert kept is not None and kept.turn == RIVERS


async def test_each_reading_says_how_far_it_read_after_what_it_found(tmp_path: Path) -> None:
    """On the clock taken before the file was opened: every record Claude Code had written by then is in the reading."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, WRITING, CUT_OFF))
    tails = Tails(Registry([member(transcript)]))
    assert (await tails.catch_up())[-1] == Read(SID, Stamp(5000))
    assert await tails.catch_up() == [Read(SID, Stamp(5000))]


async def test_a_reading_that_stops_inside_a_record_being_written_says_nothing_of_how_far_it_read(tmp_path: Path) -> None:
    """The record may have been begun before the reading, so the reading is not through everything written by then."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED) + CUT_OFF[:30])
    tails = Tails(Registry([member(transcript)]))
    assert await tails.catch_up() == [Taken(SID, PromptId("p1"), None, 7.0)]
    with transcript.open("a") as file:
        file.write(CUT_OFF[30:] + "\n")
    assert await tails.catch_up() == [Interrupted(SID, PromptId("p1"), at=7.0), Read(SID, Stamp(5000))]


async def test_a_transcript_not_written_yet_is_read_through_all_there_is(tmp_path: Path) -> None:
    """A turn whose prompt was cancelled before Claude Code wrote anything is still told once its window passes."""
    tails = Tails(Registry([member(tmp_path / "t.jsonl")]))
    assert await tails.catch_up() == [Read(SID, Stamp(5000))]


async def test_a_transcript_that_cannot_be_read_is_waited_on_no_longer(tmp_path: Path) -> None:
    """The turn is told, and its telling says why the transcript could not be read, rather than waiting forever unsaid."""
    unreadable = tmp_path / "t.jsonl"
    unreadable.mkdir()
    tails = Tails(Registry([member(unreadable)]))
    assert await tails.catch_up() == [Read(SID, Stamp(5000))]


# Three turns over before hands followed the session: one a queued message was flushed into, one cut off, one answered.
# Then the turn it is in, whose record Claude Code wrote before the busy it set again after a dialog.
HISTORY = (ASKED, LOOPING, FLUSHED, FLUSHING, QUEUED, BANANA, NEXT_ASKED.replace('"p2"', '"p9"'), CUT_OFF.replace('"p1"', '"p9"'), PROMPT.replace('"type":"user"', '"type":"user","promptId":"p3"'), DONE)
RUNNING = '{"type":"user","promptId":"p4","timestamp":"1970-01-01T00:00:02Z","message":{"role":"user","content":"Now the tests."}}'


async def test_a_transcript_read_from_its_start_hands_on_only_the_turn_it_is_in(tmp_path: Path) -> None:
    """hands-status-tlo.egc: every turn before the one the transcript ends in was over before hands followed the session."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(*HISTORY, RUNNING, LOOPING))
    tails = Tails(Registry([member(transcript)]))
    said: list[str] = []
    sink = logger.add(lambda message: said.append(message.record["message"]), level="INFO", filter="hands.sessions.tail")
    try:
        assert heard(await tails.catch_up()) == [Taken(SID, PromptId("p4"), Stamp(2000), 7.0), Continued(SID, was=PromptId("p3"), now=PromptId("p4"))]
    finally:
        logger.remove(sink)
    assert said == [f"read the transcript of session {SID} from its start: 8 of 10 events are of turns over before hands followed it; the last goes by ['p4'] and may be running; calls that turn made before hands followed it, not heard as progress: 1"]
    with transcript.open("a") as more:
        more.write(lines(CUT_OFF_MID_TOOL.replace('"p1"', '"p4"')))
    assert heard(await tails.catch_up()) == [Interrupted(SID, PromptId("p4"), at=7.0)]


async def test_a_turn_a_queued_message_was_flushed_into_is_one_turn_however_many_readings_it_spans(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(NEXT_ASKED.replace('"p2"', '"p0"'), DONE, ASKED, LOOPING, FLUSHED, FLUSHING, QUEUED, BANANA))
    assert heard(await Tails(Registry([member(transcript)])).catch_up()) == [
        Taken(SID, PromptId("p1"), None, 7.0),
        Continued(SID, was=PromptId("p0"), now=PromptId("p1")),
        Taken(SID, PromptId("p2"), None, 7.0),
        Interrupted(SID, PromptId("p2"), at=7.0),
        Continued(SID, was=PromptId("p1"), now=PromptId("p2")),
    ]


async def test_a_session_is_not_read_until_claude_code_has_said_whether_it_runs(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(RUNNING))
    registry = Registry([member(transcript)], reported=False)
    tails = Tails(registry)
    assert await tails.catch_up() == []
    registry.reported = True
    assert heard(await tails.catch_up()) == [Taken(SID, PromptId("p4"), Stamp(2000), 7.0)]


async def test_a_session_followed_from_mid_turn_works_under_its_own_prompt_and_its_stop_ends_it(tmp_path: Path) -> None:
    """hands-status-tlo.egc: a daemon restarted while the session runs attaches it from its membership file, reads it
    running, and reads its transcript from the start."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(*HISTORY, RUNNING, LOOPING))
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Attached(member(transcript)))
    await sessions.apply(StatusReported(SID, Report(status.Busy(), Stamp(3000)), at=3.0))
    tails = Tails(sessions)
    for transcribed in await tails.catch_up():
        await sessions.apply(transcribed)
    live = sessions.live_session(SID)
    assert live is not None and live.turn == Opened(PromptId("p4"))
    await sessions.apply(Stopped(SID, "Done.", mode=None, prompt=PromptId("p4"), again=False, heard=STOP_HEARD, request=STOP_REQUEST))
    assert await asyncio.wait_for(sessions.story(), 5.0) == Summarise(SID, PromptId("p4"), "Done.")
    live = sessions.live_session(SID)
    assert live is not None and live.turn == Told(PromptId("p4"))


async def test_a_first_record_still_being_written_explains_no_reading(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(RUNNING[:40])
    tails = Tails(Registry([member(transcript)]))
    said: list[str] = []
    sink = logger.add(lambda message: said.append(message.record["message"]), level="INFO", filter="hands.sessions.tail")
    try:
        assert [await tails.catch_up() for _ in range(3)] == [[], [], []]
    finally:
        logger.remove(sink)
    assert said == []


# Claude Code's own records for what the user ran rather than wrote, in the shapes 2.1.285 writes them: the caveat
# ahead of a command it carries out itself, the command, and what it printed, which names the command as its parent and
# is as often a system record as a user one; a skill command, whose skill body is a meta record; and a `!` command
# with its output.
CAVEAT = '{"type":"user","isMeta":true,"promptId":"p2","message":{"role":"user","content":"<local-command-caveat>The command below was run directly in Claude Code, not sent to you as a request.</local-command-caveat>"}}'
MODEL = '{"uuid":"c1","type":"user","promptId":"p2","message":{"role":"user","content":"<command-name>/model</command-name>\\n            <command-message>model</command-message>\\n            <command-args></command-args>"}}'
MODEL_SET = '{"uuid":"c1o","parentUuid":"c1","type":"system","subtype":"local_command","content":"<local-command-stdout>Set model to `Sonnet 5.5` for this session only</local-command-stdout>","level":"info"}'
COMPACT = '{"uuid":"c2","type":"user","promptId":"p2","message":{"role":"user","content":"<command-name>/compact</command-name>\\n            <command-message>compact</command-message>\\n            <command-args></command-args>"}}'
COMPACTED = '{"uuid":"c2o","parentUuid":"c2","type":"user","promptId":"p2","message":{"role":"user","content":"<local-command-stdout>\\u001b[2mCompacted (ctrl+o to see full summary)\\u001b[22m</local-command-stdout>"}}'
SKILL = '{"uuid":"c3","type":"user","origin":{"kind":"human"},"promptId":"p2","message":{"role":"user","content":"<command-message>delegate-some-shit</command-message>\\n<command-name>/delegate-some-shit</command-name>\\n<command-args>lh86 to a subagent now</command-args>"}}'
SKILL_BODY = '{"type":"user","isMeta":true,"promptId":"p2","message":{"role":"user","content":[{"type":"text","text":"Base directory for this skill: /skills/delegate"}]}}'
SHELL = '{"uuid":"c4","type":"user","promptId":"p2","message":{"role":"user","content":"<bash-input>lit next</bash-input>"}}'
SHELL_OUT = '{"uuid":"c4o","parentUuid":"c4","type":"user","promptId":"p2","message":{"role":"user","content":"<bash-stdout>hands-narration-8ip  open</bash-stdout><bash-stderr>sync: 1 local change</bash-stderr>"}}'


async def test_a_command_claude_code_carries_out_opens_one_turn_of_its_own_with_what_it_printed(tmp_path: Path) -> None:
    """The bug this closes: /model's command and its output were each read as a prompt, so the tail opened two turns."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE, CAVEAT, MODEL, MODEL_SET))
    tails = await following(transcript)
    assert await turn_of(transcript) == Turn(Commanded(Ref("c1"), "/model", "", "Set model to `Sonnet 5.5` for this session only"), ())
    # The prompt's turn and the command's: its output opened no third.
    assert tails._following[SID].reading.number == 2  # pyright: ignore[reportPrivateUsage]


async def test_what_compact_printed_reaches_the_turn_without_the_terminals_colours(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE, COMPACT, COMPACTED))
    assert await turn_of(transcript) == Turn(Commanded(Ref("c2"), "/compact", "", "Compacted (ctrl+o to see full summary)"), ())


async def test_a_skill_command_opens_the_turn_claude_answers_with_its_arguments(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE, SKILL, SKILL_BODY, DONE))
    assert await turn_of(transcript) == Turn(Commanded(Ref("c3"), "/delegate-some-shit", "lh86 to a subagent now"), (Said(None, "Done."),))


async def test_a_shell_command_opens_one_turn_with_its_output_and_no_markup(tmp_path: Path) -> None:
    """The bug this closes: a `!` answer opened on its bash records, and the narrator was handed XML as what the user asked."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE, SHELL, SHELL_OUT, DONE))
    shelled = Shelled(Ref("c4"), "lit next", "hands-narration-8ip  open\nstderr: sync: 1 local change")
    assert await turn_of(transcript) == Turn(shelled, (Said(None, "Done."),))
    assert "<" not in describe(shelled, TURN_SENTENCE_BUDGET)


async def test_what_another_command_printed_is_no_part_of_a_skill_commands_turn(tmp_path: Path) -> None:
    """Output joins the command it names as its parent, never whichever command opened the turn it lands in."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE, SKILL, SKILL_BODY, DONE, MODEL_SET))
    assert await turn_of(transcript) == Turn(Commanded(Ref("c3"), "/delegate-some-shit", "lh86 to a subagent now"), (Said(None, "Done."),))


async def test_a_command_claude_code_writes_as_a_system_record_opens_its_turn_like_any_other(tmp_path: Path) -> None:
    """The bug this closes: /mcp and /model, written as system local_command records, were skipped with their output."""
    mcp = '{"uuid":"c5","type":"system","subtype":"local_command","content":"<command-name>/mcp</command-name>\\n<command-message>mcp</command-message>\\n<command-args></command-args>","level":"info"}'
    dismissed = '{"uuid":"c5o","parentUuid":"c5","type":"system","subtype":"local_command","content":"<local-command-stdout>MCP dialog dismissed</local-command-stdout>","level":"info"}'
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE, mcp, dismissed))
    assert await turn_of(transcript) == Turn(Commanded(Ref("c5"), "/mcp", "", "MCP dialog dismissed"), ())


async def test_what_a_local_command_printed_is_no_answer_of_claudes_under_the_commands_prompt(tmp_path: Path) -> None:
    """The bug this closes: a local_command output record, read as Claude answering, carried p1's turn on under /model's p2."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, WRITING))
    tails = await following(transcript)
    with transcript.open("a") as more:
        more.write(lines(MODEL, MODEL_SET))
    assert heard(await tails.catch_up()) == [Taken(SID, PromptId("p2"), None, 7.0), CarriedOut(SID, PromptId("p2"), at=7.0)]


async def test_compact_typed_as_words_and_its_command_record_open_one_turn(tmp_path: Path) -> None:
    """The bug this closes: the words /compact and the record Claude Code writes after the compaction opened a turn each."""
    typed = '{"uuid":"c7","type":"user","promptId":"p1","message":{"role":"user","content":"/compact"}}'
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE, typed, COMPACT, COMPACTED))
    tails = await following(transcript)
    assert await turn_of(transcript) == Turn(Commanded(Ref("c2"), "/compact", "", "Compacted (ctrl+o to see full summary)"), ())
    assert tails._following[SID].reading.number == 2  # pyright: ignore[reportPrivateUsage]


async def test_a_skill_run_in_a_fork_written_as_typed_words_is_read_as_a_command_by_its_output(tmp_path: Path) -> None:
    """The bug this closes: `/code-review medium <pr>` was told as words the user asked, and its output dropped."""
    typed = '{"uuid":"c6","type":"user","promptId":"p2","message":{"role":"user","content":"/code-review medium 100"}}'
    launched = '{"uuid":"c6o","parentUuid":"c6","type":"system","subtype":"local_command","content":"<local-command-stdout>Running in the background as @code-review</local-command-stdout>","level":"info"}'
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE, typed, launched))
    assert await turn_of(transcript) == Turn(Commanded(Ref("c6"), "/code-review", "medium 100", "Running in the background as @code-review"), ())


async def test_what_a_command_printed_that_joins_no_turn_is_said(tmp_path: Path) -> None:
    said: list[str] = []
    sink = logger.add(lambda message: said.append(message.record["message"]), level="WARNING", filter="hands.sessions.turning")
    try:
        transcript = tmp_path / "t.jsonl"
        transcript.write_text(lines(PROMPT, DONE, MODEL_SET))
        assert await turn_of(transcript) == Turn(Asked(None, "first"), (Said(None, "Done."),))
    finally:
        logger.remove(sink)
    assert any("names record c1, which opened no turn" in line for line in said)


# A subagent started in the background and its report, as 2.1.289 writes them: the launch is the call's result, the
# report a notification, at the prompt as a user record and mid-turn as an attachment.
DELEGATE = '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t7","name":"Agent","input":{"description":"Draft tickets","prompt":"Draft them.","run_in_background":true}}]}}'
LAUNCHED = (
    '{"type":"user","promptId":"p1","timestamp":"1970-01-01T00:00:01.200Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t7","content":"Async agent launched successfully."}]},'
    '"toolUseResult":{"isAsync":true,"status":"async_launched","agentId":"a587b79fd8308e5bc","description":"Draft tickets"}}'
)
REPORT = '<task-notification>\\n<task-id>a587b79fd8308e5bc</task-id>\\n<status>killed</status>\\n<summary>Agent \\"Draft tickets\\" was stopped</summary>\\n</task-notification>'
REPORTED_AT_PROMPT = f'{{"type":"user","promptId":"p3","origin":{{"kind":"task-notification"}},"message":{{"role":"user","content":"{REPORT}"}}}}'
REPORTED_MID_TURN = f'{{"type":"attachment","attachment":{{"type":"queued_command","commandMode":"task-notification","prompt":"{REPORT}"}}}}'
AGENT = AgentId("a587b79fd8308e5bc")


async def test_a_subagent_started_in_the_background_in_a_turn_before_the_one_the_file_ends_in_is_heard_as_out(tmp_path: Path) -> None:
    """Read from its start, the turns before the one it ends in are history, but a subagent one of them started may still
    work, and only its report says it does not: its launch is heard whatever turn it was read into."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, DELEGATE, LAUNCHED, DONE, NEXT_ASKED, DONE))
    assert heard(await Tails(Registry([member(transcript)])).catch_up()) == [Launched(SID, AGENT, Stamp(1200)), Taken(SID, PromptId("p2"), None, 7.0), Continued(SID, was=PromptId("p1"), now=PromptId("p2"))]


@pytest.mark.parametrize("reported", [REPORTED_AT_PROMPT, REPORTED_MID_TURN])
async def test_a_background_tasks_notification_is_heard_as_its_report_back_at_the_prompt_or_mid_turn(tmp_path: Path, reported: str) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, DELEGATE, LAUNCHED, DONE))
    tails = await following(transcript)
    with transcript.open("a") as more:
        more.write(lines(reported))
    assert [event for event in heard(await tails.catch_up()) if not isinstance(event, Taken)] == [ReportedBack(SID, AGENT)]


# What Claude Code writes as a turn ends, as 2.1.289 writes it: what its Stop hooks did, then how long the turn took.
STOP_SUMMARY = '{"type":"system","subtype":"stop_hook_summary","hookCount":1,"hookInfos":[],"hookErrors":[],"preventedContinuation":false,"stopReason":"","hasOutput":false,"level":"suggestion","timestamp":"1970-01-01T00:00:01.300Z","uuid":"e1"}'
TURN_DURATION = '{"type":"system","subtype":"turn_duration","durationMs":1300,"messageCount":4,"timestamp":"1970-01-01T00:00:01.301Z","uuid":"e2","isMeta":false}'


@pytest.mark.parametrize("ending", [(STOP_SUMMARY, TURN_DURATION), (STOP_SUMMARY,), (TURN_DURATION,), (CUT_OFF,)])
async def test_a_transcript_read_from_its_start_whose_last_turn_ended_opens_no_turn(tmp_path: Path, ending: tuple[str, ...]) -> None:
    """hands-session-mgmt-a7t.byl: the turn that launched a subagent in the background is over though the session reads
    busy, and only the launch is heard of it."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, DELEGATE, LAUNCHED, DONE, *ending))
    said: list[str] = []
    sink = logger.add(lambda message: said.append(message.record["message"]), level="INFO", filter="hands.sessions.tail")
    try:
        assert heard(await Tails(Registry([member(transcript)])).catch_up()) == [Launched(SID, AGENT, Stamp(1200))]
    finally:
        logger.remove(sink)
    held = 2 if ending == (CUT_OFF,) else 1
    assert said == [f"read the transcript of session {SID} from its start: {held} of {held + 1} events are of turns over before hands followed it; the last goes by ['p1'] and ended, its {held} events held back until a record says it went on; calls that turn made before hands followed it, not heard as progress: 1"]


async def test_a_turn_a_stop_hook_sent_on_after_its_end_record_may_still_be_running(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, DONE, STOP_SUMMARY, DONE))
    assert heard(await Tails(Registry([member(transcript)])).catch_up()) == [Taken(SID, PromptId("p1"), None, 7.0)]


# What a Stop hook that sends Claude on writes before Claude answers, as 2.1.289 writes it.
SENT_ON = '{"type":"user","isMeta":true,"promptId":"p1","message":{"role":"user","content":"Stop hook feedback: keep going"}}'


@pytest.mark.parametrize("going_on", [SENT_ON, DONE])
async def test_a_turn_read_from_its_start_to_its_end_record_is_heard_once_a_stop_hook_sends_it_on(tmp_path: Path, going_on: str) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, DONE, STOP_SUMMARY))
    tails = Tails(Registry([member(transcript)]))
    assert heard(await tails.catch_up()) == []
    said: list[str] = []
    sink = logger.add(lambda message: said.append(message.record["message"]), level="INFO", filter="hands.sessions.tail")
    try:
        with transcript.open("a") as more:
            more.write(lines(going_on))
        assert heard(await tails.catch_up()) == [Taken(SID, PromptId("p1"), None, 7.0)]
    finally:
        logger.remove(sink)
    assert said == [f"the transcript of session {SID} went on after the end record it was read from its start to: 1 of 1 events held back are heard"]


async def test_a_turn_read_from_its_start_to_the_interrupt_that_flushed_a_queued_message_is_heard_as_it_goes_on(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, LOOPING, FLUSHED, FLUSHING))
    tails = Tails(Registry([member(transcript)]))
    assert heard(await tails.catch_up()) == []
    with transcript.open("a") as more:
        more.write(lines(QUEUED, BANANA))
    assert heard(await tails.catch_up()) == [Taken(SID, PromptId("p1"), None, 7.0), Taken(SID, PromptId("p2"), None, 7.0), Interrupted(SID, PromptId("p2"), at=7.0), Continued(SID, was=PromptId("p1"), now=PromptId("p2"))]


async def test_a_turn_read_from_its_start_to_its_end_stays_over_through_its_last_end_record_and_the_next_prompt(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, DONE, STOP_SUMMARY))
    tails = Tails(Registry([member(transcript)]))
    assert heard(await tails.catch_up()) == []
    with transcript.open("a") as more:
        more.write(lines(TURN_DURATION))
    assert heard(await tails.catch_up()) == []
    with transcript.open("a") as more:
        more.write(lines(ASKED.replace('"p1"', '"p2"')))
    assert heard(await tails.catch_up()) == [Taken(SID, PromptId("p2"), None, 7.0)]


async def test_a_session_attached_while_its_background_subagent_works_is_delegating_and_its_report_is_a_turn_of_its_own(tmp_path: Path) -> None:
    """hands-session-mgmt-a7t.byl: attached after the launching turn's Stop, as a restarted daemon is, the session is at its
    prompt with a subagent out; the report opens its own turn, and its Stop tells that turn alone."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, DELEGATE, LAUNCHED, DONE, STOP_SUMMARY, TURN_DURATION))
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Attached(member(transcript)))
    await sessions.apply(said_busy(3.0))
    tails = Tails(sessions)
    for transcribed in await tails.catch_up():
        await sessions.apply(transcribed)
    live = sessions.live_session(SID)
    assert live is not None and live.turn == Told() and live.background == frozenset({AGENT}) and delegating(live)
    with transcript.open("a") as more:
        more.write(lines(REPORTED_AT_PROMPT, DONE))
    for transcribed in await tails.catch_up():
        await sessions.apply(transcribed)
    live = sessions.live_session(SID)
    assert live is not None and live.turn == Opened(PromptId("p3")) and not delegating(live)
    await sessions.apply(Stopped(SID, "Done.", mode=None, prompt=PromptId("p3"), again=False, heard=STOP_HEARD, request=STOP_REQUEST))
    assert await asyncio.wait_for(sessions.story(), 5.0) == Summarise(SID, PromptId("p3"), "Done.")
    live = sessions.live_session(SID)
    assert live is not None and live.turn == Told(PromptId("p3"))


# What 2.1.289 writes for a command it carries out itself, its output a user record under the command's prompt id; and
# for /goal, which hands Claude a prompt of its own in the same append as what it printed.
MODEL_PRINTED = '{"uuid":"c1o","parentUuid":"c1","type":"user","promptId":"p2","message":{"role":"user","content":"<local-command-stdout>Set model to `Sonnet 5.5` for this session only</local-command-stdout>"}}'
GOAL = '{"uuid":"c8","type":"user","promptId":"p2","message":{"role":"user","content":"<command-name>/goal</command-name>\\n            <command-message>goal</command-message>\\n            <command-args>ship it</command-args>"}}'
GOAL_SET = '{"uuid":"c8o","parentUuid":"c8","type":"user","promptId":"p2","message":{"role":"user","content":"<local-command-stdout>Goal set: ship it</local-command-stdout>"}}'
GOAL_PROMPT = '{"type":"user","isMeta":true,"promptId":"p2","message":{"role":"user","content":"A session-scoped Stop hook is now active with condition: \\"ship it\\""}}'


async def test_a_command_claude_code_carries_out_is_over_once_it_printed_and_said_so_once(tmp_path: Path) -> None:
    """hands-session-mgmt-a7t.9tp: no Stop fires for /model, so what it printed is the only record its turn is over."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, DONE))
    tails = await following(transcript)
    with transcript.open("a") as more:
        more.write(lines(MODEL))
    assert heard(await tails.catch_up()) == [Taken(SID, PromptId("p2"), None, 7.0)]
    with transcript.open("a") as more:
        more.write(lines(MODEL_PRINTED))
    assert heard(await tails.catch_up()) == [CarriedOut(SID, PromptId("p2"), at=7.0)]
    assert heard(await tails.catch_up()) == []


@pytest.mark.parametrize("going_on", [(GOAL, GOAL_SET, GOAL_PROMPT), (SHELL, SHELL_OUT)], ids=["goal", "shell"])
async def test_a_command_claude_answers_is_not_over_once_it_printed(tmp_path: Path, going_on: tuple[str, ...]) -> None:
    """/goal hands Claude a prompt, and Claude answers a `!` command: their Stop ends the turn, not what they printed."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, DONE))
    tails = await following(transcript)
    with transcript.open("a") as more:
        more.write(lines(*going_on))
    assert heard(await tails.catch_up()) == [Taken(SID, PromptId("p2"), None, 7.0)]


@pytest.mark.parametrize("printed", [MODEL_PRINTED, MODEL_SET], ids=["user", "system"])
async def test_a_file_read_from_its_start_whose_last_turn_is_a_command_that_printed_opens_no_turn(tmp_path: Path, printed: str) -> None:
    """hands-session-mgmt-a7t.9tp, from PR #265's review: attached after a command ran while a subagent works, the command's
    turn is over, so nothing of it opens one."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, DELEGATE, LAUNCHED, DONE, STOP_SUMMARY, TURN_DURATION, MODEL, printed))
    assert heard(await Tails(Registry([member(transcript)])).catch_up()) == [Launched(SID, AGENT, Stamp(1200))]


async def test_a_command_held_back_as_carried_out_is_not_once_its_turn_goes_on(tmp_path: Path) -> None:
    """Read from its start to what /goal printed, its turn looks carried out; the prompt it hands Claude, read after, says it is not."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, DONE, GOAL, GOAL_SET))
    tails = await following(transcript)
    with transcript.open("a") as more:
        more.write(lines(GOAL_PROMPT))
    assert heard(await tails.catch_up()) == [Taken(SID, PromptId("p2"), None, 7.0)]


@pytest.mark.parametrize(
    ("records", "said"),
    [
        # The words /compact under one id and its command record under another: which turn they name, nothing read says.
        (('{"uuid":"c7","type":"user","promptId":"p1","message":{"role":"user","content":"/compact"}}', COMPACT, COMPACTED), ["which turn it ends is not known"]),
        # Written as system records, which carry no prompt id: no record of it was taken, so it opened no turn to end.
        (('{"uuid":"c5","type":"system","subtype":"local_command","content":"<command-name>/mcp</command-name>\\n<command-message>mcp</command-message>\\n<command-args></command-args>","level":"info"}', MODEL_SET), []),
    ],
    ids=["two ids", "no id"],
)
async def test_a_command_whose_turn_goes_by_no_one_id_ends_none(tmp_path: Path, records: tuple[str, ...], said: list[str]) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, DONE))
    tails = await following(transcript)
    with transcript.open("a") as more:
        more.write(lines(*records))
    warnings: list[str] = []
    sink = logger.add(lambda message: warnings.append(message.record["message"]), level="WARNING", filter="hands")
    try:
        transcribed = heard(await tails.catch_up())
    finally:
        logger.remove(sink)
    assert not any(isinstance(event, CarriedOut) for event in transcribed)
    assert [phrase for phrase in said if any(phrase in warning for warning in warnings)] == said


async def test_a_command_run_while_a_subagent_works_in_the_background_returns_the_session_to_its_prompt(tmp_path: Path) -> None:
    """hands-session-mgmt-a7t.9tp: Claude Code sets no idle while the subagent works, so before this the command's turn
    stayed open until the subagent reported back, and the session read as working."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, DELEGATE, LAUNCHED, DONE, STOP_SUMMARY, TURN_DURATION))
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Attached(member(transcript)))
    await sessions.apply(said_busy(3.0))
    tails = Tails(sessions)
    for transcribed in await tails.catch_up():
        await sessions.apply(transcribed)
    with transcript.open("a") as more:
        more.write(lines(MODEL, MODEL_PRINTED))
    for transcribed in await tails.catch_up():
        await sessions.apply(transcribed)
    assert await asyncio.wait_for(sessions.story(), 5.0) == Summarise(SID, PromptId("p2"), None)
    live = sessions.live_session(SID)
    assert live is not None and live.turn == Told(PromptId("p2")) and delegating(live)
