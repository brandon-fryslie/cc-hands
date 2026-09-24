"""A session's transcript, followed as Claude Code writes it, and the turn it is read into."""

import asyncio
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from loguru import logger

from hands.core.events import Continued, Interrupted, Taken
from hands.core.session import Membership, PromptId, SessionId
from hands.core.turn import Asked, Continuing, Interruption, Notified, Looked, Other, Ran, Ref, Said, Turn
from hands.core.effects import Summarise
from hands.core.events import Joined, Prompted, Stopped
from hands.core.session import Idle, Submitted, Working
from hands.sessions.delta import Deltas
from hands.sessions.registry import Sessions
from hands.sessions.tail import KEPT, Tails, Telling, keep_tailing

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

    def __post_init__(self) -> None:
        self.heard = list(self.members)

    def live_members(self) -> list[Membership]:
        return self.members

    def now(self) -> float:
        return 7.0

    def membership(self, session: SessionId) -> Membership | None:
        return next((member for member in self.heard if member.id == session), None)


async def following(transcript: Path) -> Tails:
    tails = Tails(Registry([member(transcript)]))
    await tails.catch_up()
    return tails


async def turn_of(transcript: Path, closing: str | None = None) -> Turn | None:
    telling = await (await following(transcript)).tell(SID, None, closing)
    return None if telling is None else telling.turn


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
    assert await turn_of(transcript) == Turn(Notified(None, "<task-notification>tests passed</task-notification>"), (Said(None, "Done."),))


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
    launchd would then start it again to read."""
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
    assert first is not None and first == Telling(SID, Turn(Asked(None, "first"), (Said(None, "Done."),)), number=1, through=1, stood_in=None)
    await tails.spoken(first)
    assert (await tails.tell(SID, None, None)) == Telling(SID, Turn(Asked(None, "first"), (), Continuing(1)), number=1, through=1, stood_in=None)


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
    assert first is not None and first.stood_in == "Done."
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
    assert (await tails.tell(SID, None, None)) == Telling(SID, Turn(Asked(None, "first"), (), Continuing(1)), number=1, through=1, stood_in=None)


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


async def test_how_far_behind_the_newest_record_was_when_it_was_read_is_measured(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    stamped = '{"type":"assistant","timestamp":"2026-09-01T12:00:00.000Z","message":{"content":[{"type":"text","text":"Done."}]}}'
    transcript.write_text(lines(PROMPT, stamped))
    tails = await following(transcript)
    assert tails.lag is not None and tails.lag > 0
    # A record with no timestamp to compare against is read without one, rather than with a made-up one.
    transcript.write_text(lines(PROMPT, stamped, DONE))
    await tails.catch_up()
    assert tails.lag is None


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
    transcript.write_text(lines(ASKED, WRITING, CUT_OFF))
    tails = Tails(Registry([member(transcript)]))
    assert await tails.catch_up() == [Taken(SID, PromptId("p1"), opens=True), Interrupted(SID, PromptId("p1"), at=7.0)]
    telling = await tails.tell(SID, None, None)
    # The record of the interrupt is written as a user's message, and is not read as the next thing asked.
    assert telling is not None and telling.turn == Turn(Asked(None, "Write an essay about rivers."), (Said(None, "# Rivers"), Interruption(Ref("u9"))))


async def test_a_turn_cut_off_while_a_tool_ran_is_heard_as_interrupted(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, LOOPING, REJECTED, CUT_OFF_MID_TOOL))
    tails = Tails(Registry([member(transcript)]))
    assert await tails.catch_up() == [Taken(SID, PromptId("p1"), opens=True), Interrupted(SID, PromptId("p1"), at=7.0)]
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
    assert await tails.catch_up() == [Taken(SID, PromptId("p1"), opens=True)]
    telling = await tails.tell(SID, None, None)
    assert telling is not None and telling.turn.opening == Asked(None, "Why did I see [Request interrupted by user] there?")


async def test_an_interrupt_the_stops_own_reading_found_is_still_handed_out_at_the_next_catch_up(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, WRITING))
    tails = await following(transcript)
    with transcript.open("a") as more:
        more.write(lines(CUT_OFF))
    await tails.tell(SID, None, None)
    assert await tails.catch_up() == [Interrupted(SID, PromptId("p1"), at=7.0)]
    assert await tails.catch_up() == []


async def test_an_interrupt_whose_record_names_no_turn_is_said_and_ends_nothing(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, WRITING, CUT_OFF.replace('"promptId":"p1",', "")))
    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(message.record["message"]), level="ERROR", filter="hands")
    try:
        assert await Tails(Registry([member(transcript)])).catch_up() == [Taken(SID, PromptId("p1"), opens=True)]
    finally:
        logger.remove(sink)
    assert errors == [f"session {SID} was interrupted, but the record of it names no prompt, so its turn cannot be ended"]


async def test_the_older_record_of_an_interrupt_written_as_a_plain_string_is_one_too(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, WRITING, '{"type":"user","promptId":"p1","message":{"role":"user","content":"[Request interrupted by user for tool use]"}}'))
    assert await Tails(Registry([member(transcript)])).catch_up() == [Taken(SID, PromptId("p1"), opens=True), Interrupted(SID, PromptId("p1"), at=7.0)]


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
    assert await tails.catch_up() == [Taken(SID, PromptId("p1"), opens=True), Taken(SID, PromptId("p2"), opens=False), Interrupted(SID, PromptId("p2"), at=7.0), Continued(SID, was=PromptId("p1"), now=PromptId("p2"))]


async def test_a_queued_command_taken_in_mid_turn_is_heard_from_the_results_claude_answers(tmp_path: Path) -> None:
    """As in a transcript on this machine: a queued /rate-limit-options writes no prompt, and the turn goes on under its id."""
    transcript = tmp_path / "t.jsonl"
    ran = '{"type":"user","promptId":"p2","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_9","content":"done"}]}}'
    transcript.write_text(lines(ASKED, LOOPING, ran, WRITING))
    assert await Tails(Registry([member(transcript)])).catch_up() == [Taken(SID, PromptId("p1"), opens=True), Taken(SID, PromptId("p2"), opens=False), Continued(SID, was=PromptId("p1"), now=PromptId("p2"))]


async def test_an_escape_in_the_turn_a_queued_prompt_went_on_as_leaves_the_session_idle(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, LOOPING))
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(member(transcript), "startup"))
    await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p1")))
    tailing = asyncio.create_task(keep_tailing(Tails(sessions), 0.01, sessions.apply))
    try:
        with transcript.open("a") as more:
            more.write(lines(FLUSHED, FLUSHING, QUEUED, LOOPING, REJECTED.replace('"p1"', '"p2"'), CUT_OFF_MID_TOOL.replace('"p1"', '"p2"')))
        assert await asyncio.wait_for(sessions.story(), 5.0) == Summarise(SID, PromptId("p2"), None)
    finally:
        tailing.cancel()
    listing = sessions.listing(SID)
    assert listing is not None and isinstance(listing.session.state, Idle)


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
    assert await Tails(Registry([member(transcript)])).catch_up() == [Taken(SID, PromptId("p1"), opens=True), Taken(SID, PromptId("p2"), opens=False), Continued(SID, was=PromptId("p1"), now=PromptId("p2"))]


async def test_a_prompt_cancelled_while_its_hooks_ran_never_makes_the_session_working_and_the_one_resent_does(tmp_path: Path) -> None:
    """Escape during UserPromptSubmit writes nothing (2.1.281): only the record of a turn says its prompt was taken."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("")
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(member(transcript), "startup"))
    await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p0")))
    tails = Tails(sessions)
    for _ in range(3):
        for transcribed in await tails.catch_up():
            await sessions.apply(transcribed)
    listing = sessions.listing(SID)
    assert listing is not None and listing.session.state == Submitted(since=1.0)
    await sessions.apply(Prompted(SID, at=4.0, mode=None, prompt=PromptId("p1")))
    transcript.write_text(lines(ASKED))
    for transcribed in await tails.catch_up():
        await sessions.apply(transcribed)
    listing = sessions.listing(SID)
    assert listing is not None and listing.session.state == Working(since=4.0)


async def test_a_prompt_whose_first_record_is_its_interrupt_is_heard_taken_before_it_is_heard_stopped(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(CUT_OFF))
    assert await Tails(Registry([member(transcript)])).catch_up() == [Taken(SID, PromptId("p1"), opens=False), Interrupted(SID, PromptId("p1"), at=7.0)]


async def test_the_tail_hands_each_interrupt_it_reads_to_the_registry_and_the_session_is_idle(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, WRITING))
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(member(transcript), "startup"))
    await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p1")))
    tailing = asyncio.create_task(keep_tailing(Tails(sessions), 0.01, sessions.apply))
    try:
        with transcript.open("a") as more:
            more.write(lines(CUT_OFF))
        assert await asyncio.wait_for(sessions.story(), 5.0) == Summarise(SID, PromptId("p1"), None)
    finally:
        tailing.cancel()
    listing = sessions.listing(SID)
    assert listing is not None and isinstance(listing.session.state, Idle)


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
    """The adversarial order, end to end: p2's hook is applied before the tail has read p1's interrupt, and the narrator
    gets to p1 only after the tail has read p2's opening. p1 is told as itself, interrupt and all; p2 only its own steps."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, WRITING))
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    await sessions.apply(Joined(member(transcript), "startup"))
    await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p1")))
    tails = Tails(sessions)
    for transcribed in await tails.catch_up():
        await sessions.apply(transcribed)
    await sessions.apply(Prompted(SID, at=5.0, mode=None, prompt=PromptId("p2")))
    with transcript.open("a") as more:
        more.write(lines(CUT_OFF, NEXT_ASKED, DONE))
    for transcribed in await tails.catch_up():
        await sessions.apply(transcribed)
    await sessions.apply(Stopped(SID, "Done.", mode=None, prompt=PromptId("p2")))

    ended = await asyncio.wait_for(sessions.story(), 2.0)
    assert ended == Summarise(SID, PromptId("p1"), None)
    first = await tails.tell(SID, PromptId("p1"), None)
    assert first is not None and first.turn == RIVERS
    await tails.spoken(first)
    stopped = await asyncio.wait_for(sessions.story(), 2.0)
    assert stopped == Summarise(SID, PromptId("p2"), "Done.")
    second = await tails.tell(SID, PromptId("p2"), "Done.")
    assert second is not None and second.turn == Turn(Asked(None, "Shorter."), (Said(None, "Done."),))
    listing = sessions.listing(SID)
    assert listing is not None and listing.session.state == Idle()


async def test_a_turn_taken_and_ended_before_the_tail_read_any_of_it_is_told_as_itself_with_its_own_changes(tmp_path: Path) -> None:
    """hands-keyboard-gxr.90g, end to end: p1 is taken, writes a file, and is interrupted, and p2's hook is applied and
    marked, all before the tail reads a record of p1. p1's records read after are the proof it ran: it is told as
    itself, with the file it wrote and not the one p2 goes on to write, and p2 is told after it, with only its own."""
    root = tmp_path / "work"
    root.mkdir()
    for args in (("init", "-q"), ("config", "user.email", "t@example.com"), ("config", "user.name", "Test"), ("commit", "-q", "--allow-empty", "-m", "first")):
        subprocess.run(("git", "-C", str(root), *args), check=True)
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("")
    deltas = Deltas()
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None, changes=deltas)
    await sessions.apply(Joined(Membership(SID, pid=4242, cwd=root, transcript=transcript), "startup"))
    tails = Tails(sessions)
    await sessions.apply(Prompted(SID, at=1.0, mode=None, prompt=PromptId("p1")))
    (root / "rivers.md").write_text("# Rivers\n")
    await sessions.apply(Prompted(SID, at=5.0, mode=None, prompt=PromptId("p2")))
    (root / "shorter.md").write_text("Rivers.\n")
    transcript.write_text(lines(ASKED, WRITING, CUT_OFF, NEXT_ASKED, DONE))
    for transcribed in await tails.catch_up():
        await sessions.apply(transcribed)
    await sessions.apply(Stopped(SID, "Done.", mode=None, prompt=PromptId("p2")))
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
    listing = sessions.listing(SID)
    assert listing is not None and listing.session.state == Idle()


async def test_a_telling_that_names_no_turn_lets_go_of_none_that_ended(tmp_path: Path) -> None:
    """A Stop with no prompt_id says nothing of which turn ended, so the turns kept for their tellings stay kept."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, WRITING, CUT_OFF, NEXT_ASKED, DONE))
    tails = await following(transcript)
    assert await tails.tell(SID, None, None) is not None
    kept = await tails.tell(SID, PromptId("p1"), None)
    assert kept is not None and kept.turn == RIVERS
