"""A session's transcript, followed as Claude Code writes it, and the turn it is read into."""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest
from loguru import logger

from hands.core.session import Membership, SessionId
from hands.core.turn import Asked, Notified, Looked, Ran, Said, Turn
from hands.sessions.tail import Tails, Telling

SID = SessionId("bf411065-dc5c-4ec9-8302-61b84bdb5c53")

# Real records from this repository's own sessions: an earlier /clear prompt, the prompt that starts the turn,
# attachments, modes, a title, thinking, text, three Bash calls with their results (the last failed),
# an injected isMeta user record, a final text, and a record still being written.
FIXTURE = Path(__file__).parent / "fixtures" / "turn.jsonl"


def member(transcript: Path) -> Membership:
    return Membership(SID, pid=4242, cwd=Path("/code/a"), transcript=transcript)


@dataclass
class Registry:
    """As much of the session registry as the tail asks about."""

    members: list[Membership]

    def live_members(self) -> list[Membership]:
        return self.members

    def membership(self, session: SessionId) -> Membership | None:
        return next((member for member in self.members if member.id == session), None)


async def following(transcript: Path) -> Tails:
    tails = Tails(Registry([member(transcript)]))
    await tails.catch_up()
    return tails


async def turn_of(transcript: Path, closing: str | None = None) -> Turn | None:
    telling = await (await following(transcript)).tell(SID, closing)
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
    assert await turn_of(transcript) == Turn(Asked("match this\n[an image]"), ())


async def test_a_notification_after_the_turn_ended_opens_a_turn_of_its_own_and_is_not_what_the_user_asked(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    notified = '{"type":"user","origin":{"kind":"task-notification"},"message":{"role":"user","content":"<task-notification>tests passed</task-notification>"}}'
    transcript.write_text(lines(PROMPT, DONE, notified, DONE))
    assert await turn_of(transcript) == Turn(Notified("<task-notification>tests passed</task-notification>"), (Said(None, "Done."),))


@pytest.mark.parametrize("kind", ["human", "task-notification"])
async def test_a_message_that_lands_while_a_tool_runs_belongs_to_the_turn_under_way(tmp_path: Path, kind: str) -> None:
    transcript = tmp_path / "t.jsonl"
    landed = f'{{"type":"user","origin":{{"kind":"{kind}"}},"message":{{"role":"user","content":"also this"}}}}'
    transcript.write_text(lines(PROMPT, CALL, RESULT, landed, DONE))
    assert await turn_of(transcript) == Turn(Asked("first"), (RAN, Said(None, "Done.")))


async def test_compactions_summary_does_not_open_a_turn(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    summary = '{"type":"user","isCompactSummary":true,"message":{"role":"user","content":"This session is being continued"}}'
    transcript.write_text(lines(PROMPT, DONE, summary, DONE))
    assert await turn_of(transcript) == Turn(Asked("first"), (Said(None, "Done."), Said(None, "Done.")))


async def test_a_prompt_with_a_document_attached_opens_the_turn_and_names_the_document_rather_than_its_bytes(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    document = '{"type":"user","message":{"role":"user","content":[{"type":"document","source":{"data":"JVBERi0x"}},{"type":"text","text":"read this"}]}}'
    transcript.write_text(lines(PROMPT, DONE, document))
    assert await turn_of(transcript) == Turn(Asked("[a document]\nread this"), ())


async def test_a_transcript_with_no_prompt_yet_has_no_turn(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('{"type":"ai-title","aiTitle":"x"}\n{"type":"user","isMeta":true,"message":{"role":"user","content":"injected"}}\n')
    assert await (await following(transcript)).tell(SID, None) is None


async def test_a_transcript_that_cannot_be_read_when_a_turn_is_told_is_a_failure_the_narrator_says(tmp_path: Path) -> None:
    """A session that has written no transcript at all is normal while it is quiet, and wrong the moment it stops."""
    tails = await following(tmp_path / "unwritten.jsonl")
    with pytest.raises(FileNotFoundError):
        await tails.tell(SID, None)


async def test_a_stop_from_a_session_the_registry_does_not_list_tells_nothing(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE))
    assert await Tails(Registry([])).tell(SID, None) is None


async def test_a_session_that_stops_before_the_catch_up_has_reached_it_is_still_told(tmp_path: Path) -> None:
    """A turn takes seconds and the catch-up runs ten times a second, but a Stop never depends on having won that race."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE))
    telling = await Tails(Registry([member(transcript)])).tell(SID, None)
    assert telling is not None and telling.turn == Turn(Asked("first"), (Said(None, "Done."),))


async def test_a_call_whose_result_never_came_is_shown_as_having_none(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        '{"type":"user","message":{"role":"user","content":"go"}}\n'
        '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"/a/b.py"}}]}}\n'
    )
    assert await turn_of(transcript) == Turn(Asked("go"), (Looked(None, "Read", "/a/b.py", "(no result)"),))


async def test_only_what_was_appended_since_the_last_reading_is_read_again(tmp_path: Path) -> None:
    """The point of a tail: a turn is built as it is written, not re-derived from a file that grows all day."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE))
    tails = await following(transcript)
    first = await tails.tell(SID, None)
    assert first is not None and first.turn == Turn(Asked("first"), (Said(None, "Done."),))
    tails.spoken(first)
    with transcript.open("a") as more:
        more.write(lines(CALL, RESULT, DONE))
    second = await tails.tell(SID, None)
    assert second is not None and second.turn == Turn(Asked("first"), (RAN, Said(None, "Done.")))


async def test_a_record_only_half_written_is_not_read_until_the_newline_that_ends_it_is(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT) + DONE[:40])
    tails = await following(transcript)
    first = await tails.tell(SID, None)
    assert first is not None and first.turn == Turn(Asked("first"), ())
    transcript.write_text(lines(PROMPT, DONE))
    second = await tails.tell(SID, None)
    assert second is not None and second.turn == Turn(Asked("first"), (Said(None, "Done."),))


async def test_a_transcript_that_grew_shorter_is_read_again_from_its_start(tmp_path: Path) -> None:
    """Nothing Claude Code does cuts a transcript short, so this is the loud way to be wrong rather than a silent one."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, CALL, RESULT, DONE))
    tails = await following(transcript)
    assert await tails.tell(SID, None) is not None
    warnings: list[str] = []
    sink = logger.add(lambda message: warnings.append(message.record["message"]), level="WARNING", filter="hands")
    try:
        transcript.write_text(lines(PROMPT, DONE))
        again = await tails.tell(SID, None)
    finally:
        logger.remove(sink)
    assert again is not None and again.turn == Turn(Asked("first"), (Said(None, "Done."),))
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
    assert turn == Turn(Asked("first"), (Said(None, "Done."),))
    assert errors and "could not be read, so it is not told" in errors[0]


async def test_a_reading_picks_up_after_the_steps_already_told(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE))
    tails = await following(transcript)
    first = await tails.tell(SID, None)
    assert first is not None and first == Telling(SID, Turn(Asked("first"), (Said(None, "Done."),)), number=1, through=1, stood_in=None)
    tails.spoken(first)
    assert (await tails.tell(SID, None)) == Telling(SID, Turn(Asked("first"), ()), number=1, through=1, stood_in=None)


async def test_a_turn_told_but_never_spoken_is_told_again(tmp_path: Path) -> None:
    """The narrator marks a turn told only once the model has summarised it, so a failed summary loses nothing."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE))
    tails = await following(transcript)
    first = await tails.tell(SID, None)
    assert first is not None
    assert (await tails.tell(SID, None)) == first


async def test_the_hooks_closing_reply_ends_a_turn_whose_transcript_does_not_hold_it_yet_and_is_not_doubled_when_it_does(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, CALL, RESULT))
    tails = await following(transcript)
    first = await tails.tell(SID, "Done.")
    assert first is not None and first.turn == Turn(Asked("first"), (RAN, Said(None, "Done.")))
    tails.spoken(first)
    with transcript.open("a") as more:
        more.write(lines(DONE))
    second = await tails.tell(SID, "Done.")
    assert second is not None and second.turn == Turn(Asked("first"), ())


async def test_a_closing_reply_is_matched_to_its_record_however_the_whitespace_around_it_differs(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    padded = '{"type":"assistant","message":{"content":[{"type":"text","text":"  Done.\\n"}]}}'
    transcript.write_text(lines(PROMPT))
    tails = await following(transcript)
    first = await tails.tell(SID, "Done.")
    assert first is not None and first.stood_in == "Done."
    tails.spoken(first)
    with transcript.open("a") as more:
        more.write(lines(padded))
    second = await tails.tell(SID, "Done.")
    assert second is not None and second.turn == Turn(Asked("first"), ())


async def test_a_closing_reply_the_turn_said_once_before_still_ends_it(tmp_path: Path) -> None:
    """The record of the reply is the one the turn ends on, so an earlier reply in the same words does not stand for it."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE, CALL, RESULT))
    assert await turn_of(transcript, closing="Done.") == Turn(Asked("first"), (Said(None, "Done."), RAN, Said(None, "Done.")))


async def test_a_reply_a_later_turn_repeats_is_told_again_because_a_turn_is_told_only_what_it_did(tmp_path: Path) -> None:
    """Claude says the same short thing twice in a row: the second turn is its own, and is heard."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT))
    tails = await following(transcript)
    first = await tails.tell(SID, "Nothing to do.")
    assert first is not None and first.turn == Turn(Asked("first"), (Said(None, "Nothing to do."),))
    tails.spoken(first)
    with transcript.open("a") as more:
        more.write(lines(DONE, '{"type":"user","message":{"role":"user","content":"check again"}}'))
    second = await tails.tell(SID, "Nothing to do.")
    assert second is not None and second.turn == Turn(Asked("check again"), (Said(None, "Nothing to do."),))


async def test_a_turn_told_while_the_next_one_opened_marks_nothing_of_the_next(tmp_path: Path) -> None:
    """Summarising takes a second of model time, which is long enough for the user to have asked something else."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE))
    tails = await following(transcript)
    first = await tails.tell(SID, None)
    assert first is not None
    with transcript.open("a") as more:
        more.write(lines('{"type":"user","message":{"role":"user","content":"second"}}', CALL, RESULT, DONE))
    # The summary of the first turn only now comes back, and its mark is not this turn's.
    assert (await tails.tell(SID, None)) is not None
    tails.spoken(first)
    second = await tails.tell(SID, None)
    assert second is not None and second.turn == Turn(Asked("second"), (RAN, Said(None, "Done.")))


async def test_a_session_the_registry_stops_listing_is_forgotten_with_what_it_was_told(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(PROMPT, DONE))
    registry = Registry([member(transcript)])
    tails = Tails(registry)
    await tails.catch_up()
    told = await tails.tell(SID, None)
    assert told is not None
    tails.spoken(told)
    registry.members.clear()
    await tails.catch_up()
    assert await tails.tell(SID, None) is None


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
    await asyncio.gather(*(tails.catch_up() for _ in range(8)), tails.tell(SID, None), tails.tell(SID, None))
    telling = await tails.tell(SID, None)
    # Read twice, every step would be here twice over.
    assert telling is not None
    assert telling.turn == Turn(Asked("first"), (Said(None, "Done."), RAN, Said(None, "Done.")))
