"""read_session hands the intermediary a sentence per turn a session finished; read_turn, one turn's steps in order from the point it last read to."""

import shutil
import tempfile
from pathlib import Path
from typing import Any


from hands.core.events import Ended, Joined, StatusReported
from hands.core.session import Membership, SessionId
from hands.core.status import Busy, Idle, Report, Stamp
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.sessions.registry import Sessions
from hands.sessions.sentences import Sentences
from hands.voice.sentences import SummaryStore, Turns
from hands.voice.summarising import summarise_turns
from hands.voice.tool import Tool
from hands.voice.tools import READBACK_COUNT, TURNS_PAGE, session_tools

FIXTURE = Path(__file__).parent / "fixtures" / "session.jsonl"
SID = SessionId("s1")


async def joined(transcript: Path) -> Sessions:
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _entry: None)
    await sessions.apply(Joined(Membership(SID, pid=4242, cwd=Path("/code/a"), transcript=transcript), "startup"))
    return sessions


async def at_prompt(sessions: Sessions) -> Sessions:
    """The session as Claude Code reports it sitting at its prompt: its last turn is over."""
    await sessions.apply(StatusReported(SID, Report(Idle(), Stamp(1)), at=0.0))
    return sessions


def tools(sessions: Sessions, store: SummaryStore | None = None) -> dict[str, Tool]:
    return {tool.name: tool for tool in session_tools(sessions, store or SummaryStore(Sentences(Path(tempfile.mkdtemp()) / "sentences.db")))}


async def sentences(sessions: Sessions, store: SummaryStore, before: int = 0) -> dict[str, Any]:
    """The session a sentence per turn, as read_session hands it over."""
    return dict(await tools(sessions, store)["read_session"].body(session="s1", before=before))


async def read(sessions: Sessions, session: str = "s1", since: str = "", turn: int = 1) -> dict[str, Any]:
    """One turn's steps, as read_turn hands them over."""
    return dict(await tools(sessions)["read_turn"].body(session=session, turn=turn, since=since))


async def test_a_session_is_read_back_in_order_and_everything_names_its_record(tmp_path: Path) -> None:
    transcript = tmp_path / "s1.jsonl"
    shutil.copy(FIXTURE, transcript)
    sessions = await joined(transcript)
    first, second = await read(sessions), await read(sessions, turn=2)
    said = [happening["what"] for happening in first["happened"] + second["happened"]]
    # What was asked is in its place among what was done, so an hour of work reads as work on something.
    assert [words.split()[1].rstrip(":") for words in said] == [
        "user", "said", "ran", "ran", "said", "user", "said", "ran", "ran", "wrote", "ran", "gave", "said"
    ]
    assert said[0].startswith("The user asked:\nYou know, im wondering if we should be using Pipecat")
    # A command is cut to its budget, as every part of a step is, and keeps the purpose Claude gave it.
    assert said[2].startswith("Claude ran ls ~/code | grep -i pipecat") and "... (cut) (Check local Pipecat presence" in said[2]
    # A subagent is one step: the job it was given, and that it has not reported back.
    assert said[11] == "Claude gave the general-purpose subagent this job: Rewrite docs for Pipecat architecture\nIt is still working."
    # The record id is what the next reading is asked from, so everything carries the one it came from.
    assert all(happening["record"] for happening in first["happened"] + second["happened"])
    assert first["more"] is False and second["more"] is False
    # Each turn is its own: the first ends where the second is asked.
    assert [words.split()[1].rstrip(":") for words in said[:5]] == ["user", "said", "ran", "ran", "said"]


async def test_reading_on_from_where_it_read_to_gives_what_came_after_and_nothing_twice(tmp_path: Path) -> None:
    transcript = tmp_path / "s1.jsonl"
    shutil.copy(FIXTURE, transcript)
    sessions = await joined(transcript)
    whole = await read(sessions)
    third = whole["happened"][2]["record"]
    after = await read(sessions, since=third)
    assert after["happened"] == whole["happened"][3:]
    # Read on from its own last record, a session has nothing left to say and says so without an error.
    assert (await read(sessions, since=whole["happened"][-1]["record"]))["happened"] == []


def said(uuid: str, text: str) -> str:
    return f'{{"uuid":"{uuid}","type":"assistant","message":{{"content":[{{"type":"text","text":"{text}"}}]}}}}'


async def test_a_session_with_more_than_one_reading_says_where_to_read_on_from(tmp_path: Path) -> None:
    """An hour of work is hundreds of steps, and all of them at once is a context spent on history."""
    transcript = tmp_path / "s1.jsonl"
    records = ['{"uuid":"u0","type":"user","message":{"role":"user","content":"go"}}']
    records += [said(f"s{n}", f"step {n}") for n in range(60)]
    transcript.write_text("".join(f"{record}\n" for record in records))
    answer = await read(await joined(transcript))
    assert len(answer["happened"]) == READBACK_COUNT
    assert answer["more"] is True and answer["working"] is False
    assert answer["more_since"] == answer["happened"][-1]["record"]


async def test_a_record_holding_both_settled_work_and_a_call_still_running_is_never_the_mark(tmp_path: Path) -> None:
    """The mark names a record; a record holds the text and the call it introduces, written together.

    Marking it because its text is finished would go on from after the whole record next time, which loses the
    result of the call inside it — the one thing the mark is held back for in the first place.
    """
    transcript = tmp_path / "s1.jsonl"
    prompt = '{"uuid":"u1","type":"user","message":{"role":"user","content":"run the suite"}}\n'
    both = (
        '{"uuid":"u2","type":"assistant","message":{"content":['
        '{"type":"text","text":"now the suite"},'
        '{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"pytest"}}]}}\n'
    )
    transcript.write_text(prompt + both)
    sessions = await joined(transcript)

    working = await read(sessions)
    assert [happening["record"] for happening in working["happened"]] == ["u1", "u2", "u2"]
    # Nothing was paged off the end, and yet there is more to come: two facts, answered separately.
    assert working["more"] is False and working["working"] is True
    assert working["more_since"] == "u1"

    transcript.write_text(
        prompt + both
        + '{"uuid":"u3","type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"3 tests did not pass"}]}}\n'
    )
    landed = await read(sessions, since=working["more_since"])
    assert landed["happened"][1]["what"] == "Claude ran pytest\nOutput: 3 tests did not pass"
    assert landed["working"] is False


async def test_a_record_carrying_a_whole_page_is_handed_over_long_rather_than_short(tmp_path: Path) -> None:
    """Cut to fit, the page would name that record as the mark and everything past the cut would be dropped."""
    transcript = tmp_path / "s1.jsonl"
    calls = ",".join(
        f'{{"type":"tool_use","id":"c{n}","name":"Bash","input":{{"command":"echo {n}"}}}}' for n in range(READBACK_COUNT + 5)
    )
    results = ",".join(f'{{"type":"tool_result","tool_use_id":"c{n}","content":"{n}"}}' for n in range(READBACK_COUNT + 5))
    transcript.write_text(
        '{"uuid":"u1","type":"user","message":{"role":"user","content":"go"}}\n'
        + '{"uuid":"u2","type":"assistant","message":{"content":[' + calls + ']}}\n'
        + '{"uuid":"u3","type":"user","message":{"role":"user","content":[' + results + ']}}\n'
    )
    sessions = await joined(transcript)
    # The record begins inside the page, so the page ends before it rather than splitting it.
    first = await read(sessions)
    assert [happening["record"] for happening in first["happened"]] == ["u1"]
    assert first["more"] is True and first["more_since"] == "u1"
    # On its own it is longer than a page, and is handed over whole: none of it can be marked and skipped.
    rest = await read(sessions, since="u1")
    assert len(rest["happened"]) == READBACK_COUNT + 5
    assert rest["more"] is False and rest["more_since"] == "u2"


async def test_a_reading_never_ends_inside_a_record_so_the_rest_of_one_is_never_skipped(tmp_path: Path) -> None:
    """The mark names a record and a reading goes on from after it, so a page that split one would lose its tail.

    Nearly every record carries one happening: 8 of the 628,822 on this machine carry more than one, a text
    and the call it introduces. Rare, and silent when it happens — the reading just never mentions what the
    skipped block did — which is why the page is cut where a record is, rather than wherever forty falls.
    """
    transcript = tmp_path / "s1.jsonl"
    records = ['{"uuid":"u0","type":"user","message":{"role":"user","content":"go"}}']
    records += [f'{{"uuid":"t{n}","type":"assistant","message":{{"content":[{{"type":"text","text":"step {n}"}}]}}}}' for n in range(38)]
    # One record carrying two happenings, which the page would otherwise end in the middle of.
    records.append(
        '{"uuid":"both","type":"assistant","message":{"content":['
        '{"type":"text","text":"now the suite"},{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"pytest"}}]}}'
    )
    records.append('{"uuid":"res","type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}}')
    transcript.write_text("".join(f"{record}\n" for record in records))
    sessions = await joined(transcript)

    answer = await read(sessions)
    assert len(answer["happened"]) == READBACK_COUNT - 1
    assert answer["more"] is True and answer["more_since"] == "t37"
    # The split record comes back whole in the next reading, both of the happenings it carries.
    after = await read(sessions, since=answer["more_since"])
    assert [happening["record"] for happening in after["happened"]] == ["both", "both"]
    assert after["happened"][1]["what"].startswith("Claude ran pytest")


async def test_what_a_session_is_still_running_is_told_and_its_result_is_told_when_it_lands(tmp_path: Path) -> None:
    """The mark never passes a call the session has not come back from.

    Otherwise the reading goes on from after it next time: the user is told the suite is being run, and the
    three tests that did not pass are told to nobody, because their result sits behind the mark.
    """
    transcript = tmp_path / "s1.jsonl"
    prompt = '{"uuid":"u1","type":"user","message":{"role":"user","content":"run the suite"}}\n'
    call = '{"uuid":"u2","type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"pytest"}}]}}\n'
    transcript.write_text(prompt + call)
    sessions = await joined(transcript)

    working = await read(sessions)
    assert [happening["record"] for happening in working["happened"]] == ["u1", "u2"]
    assert working["happened"][1]["what"] == "Claude ran pytest\nOutput: (no result)"
    # Shown, so the user hears what the session is doing — but not marked as read.
    assert working["more_since"] == "u1"

    transcript.write_text(
        prompt + call
        + '{"uuid":"u3","type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"3 tests did not pass"}]}}\n'
    )
    landed = await read(sessions, since=working["more_since"])
    assert [happening["record"] for happening in landed["happened"]] == ["u2"]
    assert landed["happened"][0]["what"] == "Claude ran pytest\nOutput: 3 tests did not pass"


async def test_a_record_that_names_no_id_is_never_the_mark_handed_back(tmp_path: Path) -> None:
    """A mark of nothing reads as "from the start" next time, which tells the whole session over again."""
    transcript = tmp_path / "s1.jsonl"
    transcript.write_text(
        '{"uuid":"u1","type":"user","message":{"role":"user","content":"go"}}\n'
        '{"type":"assistant","message":{"content":[{"type":"text","text":"Done."}]}}\n'
    )
    answer = await read(await joined(transcript))
    assert [happening["record"] for happening in answer["happened"]] == ["u1", None]
    assert answer["more_since"] == "u1"


async def test_a_session_that_has_ended_is_never_said_to_be_in_the_middle_of_something(tmp_path: Path) -> None:
    """A session killed inside a call will never write its result [LAW:one-source-of-truth].

    Whether one can still answer is the registry's to say, not the file's: held to the file alone, the mark
    would sit behind that call for ever and the intermediary would keep calling a dead session busy.
    """
    transcript = tmp_path / "s1.jsonl"
    transcript.write_text(
        '{"uuid":"u1","type":"user","message":{"role":"user","content":"run the suite"}}\n'
        '{"uuid":"u2","type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"pytest"}}]}}\n'
    )
    sessions = await joined(transcript)
    assert (await read(sessions))["working"] is True

    await sessions.apply(Ended(SID, "other"))
    ended = await read(sessions)
    assert ended["working"] is False
    assert ended["more_since"] == "u2"


async def test_a_mark_this_turn_never_held_is_said_rather_than_read_as_the_start(tmp_path: Path) -> None:
    """[LAW:no-silent-failure] a mark from another session would otherwise re-tell this one from the top."""
    transcript = tmp_path / "s1.jsonl"
    shutil.copy(FIXTURE, transcript)
    sessions = await joined(transcript)
    answer = await read(sessions, since="not-a-record")
    assert answer == {"error": "turn 1 of session s1 has no record not-a-record; call again with since empty to read from its start"}
    # A mark from another turn of the same session is not this turn's either.
    later = (await read(sessions, turn=2))["happened"][0]["record"]
    assert (await read(sessions, since=later))["error"].startswith("turn 1 of session s1 has no record")


async def test_a_turn_the_session_has_not_had_is_said_rather_than_read_as_another(tmp_path: Path) -> None:
    transcript = tmp_path / "s1.jsonl"
    shutil.copy(FIXTURE, transcript)
    assert await read(await joined(transcript), turn=3) == {"error": "session s1 has turns 1 to 2, and no turn 3"}


async def test_a_session_the_registry_never_heard_of_is_said_to_be_no_session(tmp_path: Path) -> None:
    answer = await read(await joined(tmp_path / "s1.jsonl"), session="nobody")
    assert answer == {"error": "there is no session nobody"}


async def test_a_transcript_that_cannot_be_read_is_said_rather_than_answered_with_nothing(tmp_path: Path) -> None:
    """[LAW:no-silent-failure] the model is told why it got nothing, rather than being handed nothing."""
    answer = await read(await joined(tmp_path / "gone.jsonl"))
    assert answer == {"error": "the transcript of session s1 could not be read"}


# read_session: a sentence per finished turn.


def asked(number: int) -> str:
    return f'{{"uuid":"q{number}","type":"user","message":{{"role":"user","content":"do task {number}"}}}}'


def hour(turns: int, steps: int) -> list[str]:
    """A long session: each turn a request and many steps."""
    return [record for turn in range(turns) for record in [asked(turn), *(said(f"s{turn}-{n}", f"step {n} of task {turn}") for n in range(steps))]]


async def summarised(store: SummaryStore, recorded: list[Entry]) -> None:
    """Run the pass the store was asked for, with a summariser that names each turn it is shown."""
    wanted = await store.wanted()
    assert isinstance(wanted, Turns)

    async def summarise(page: str) -> str:
        return "\n".join(f"{due.id}: What {due.id} did." for due in wanted.due if f'id="{due.id}"' in page)

    await summarise_turns(wanted, store, summarise, recorded.append)


async def test_an_hour_long_session_is_a_sentence_per_turn_said_off_the_voice_path(tmp_path: Path) -> None:
    """An hour of work read raw is thousands of tokens; a sentence per turn is a few hundred."""
    transcript = tmp_path / "s1.jsonl"
    transcript.write_text("".join(f"{record}\n" for record in hour(turns=30, steps=40)))
    sessions = await at_prompt(await joined(transcript))
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))

    # Nothing is said yet: each finished turn is its request, and none of it waited on a model.
    first = await sentences(sessions, store)
    assert [turn["turn"] for turn in first["turns"]] == list(range(1, 31))
    assert first["turns"][4] == {"turn": 5, "opened": "The user asked:\ndo task 4"}
    assert first["unsummarised"] == 30 and first["working"] is False and first["earlier"] == 0

    recorded: list[Entry] = []
    await summarised(store, recorded)
    [pass_] = recorded
    assert isinstance(pass_, WideEvent) and (pass_.event, pass_.outcome, pass_.facts["session"]) == ("summary.turns", "ok", SID)
    assert pass_.counts == {"known": 0, "asked": 30, "said": 30, "calls": 2, "failed_calls": 0, "stray": 0}

    then = await sentences(sessions, store)
    assert then["turns"][4] == {"turn": 5, "summary": "What turn-5 did."}
    assert then["unsummarised"] == 0
    # Said once, kept: a second read asks for nothing more.
    assert store._wanted.empty()  # pyright: ignore[reportPrivateUsage]
    # The raw steps of one turn are one call away, and are all of that turn and nothing of another.
    steps = await read(sessions, turn=5)
    assert len(steps["happened"]) == READBACK_COUNT and steps["more"] is True
    rest = await read(sessions, turn=5, since=steps["more_since"])
    assert [step["what"] for step in rest["happened"]] == ["Claude said:\nstep 39 of task 4"] and rest["more"] is False


async def test_the_turn_a_running_session_is_on_is_not_summarised_until_it_is_done(tmp_path: Path) -> None:
    """A turn still growing would be summarised under a key it will not have once it is done."""
    transcript = tmp_path / "s1.jsonl"
    transcript.write_text("".join(f"{record}\n" for record in hour(turns=2, steps=3)))
    sessions = await joined(transcript)
    await sessions.apply(StatusReported(SID, Report(Busy(), Stamp(1)), at=0.0))
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))

    answer = await sentences(sessions, store)
    assert answer["working"] is True and answer["unsummarised"] == 1
    assert answer["turns"][1] == {"turn": 2, "opened": "The user asked:\ndo task 1"}
    wanted = await store.wanted()
    assert isinstance(wanted, Turns) and [due.id for due in wanted.due] == ["turn-1"]


async def test_a_session_whose_status_is_not_read_yet_has_its_last_turn_left_unsummarised(tmp_path: Path) -> None:
    """Attached mid-turn, a session is heard of before its status is read; half a turn summarised would be kept for good."""
    transcript = tmp_path / "s1.jsonl"
    transcript.write_text("".join(f"{record}\n" for record in hour(turns=2, steps=3)))
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))

    answer = await sentences(await joined(transcript), store)
    assert answer["working"] is False and answer["unsummarised"] == 1
    wanted = await store.wanted()
    assert isinstance(wanted, Turns) and [due.id for due in wanted.due] == ["turn-1"]


async def test_a_long_day_is_read_newest_first_a_page_at_a_time(tmp_path: Path) -> None:
    transcript = tmp_path / "s1.jsonl"
    transcript.write_text("".join(f"{record}\n" for record in hour(turns=TURNS_PAGE + 10, steps=1)))
    sessions = await at_prompt(await joined(transcript))
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))

    newest = await sentences(sessions, store)
    assert [turn["turn"] for turn in newest["turns"]] == list(range(11, TURNS_PAGE + 11))
    assert newest["earlier"] == 10 and newest["unsummarised"] == TURNS_PAGE
    before = await sentences(sessions, store, before=11)
    assert [turn["turn"] for turn in before["turns"]] == list(range(1, 11)) and before["earlier"] == 0
    # Only what was handed over is asked for: the store is never sent a whole day by one read.
    first = await store.wanted()
    assert isinstance(first, Turns) and len(first.due) == TURNS_PAGE
    assert "error" in await sentences(sessions, store, before=TURNS_PAGE + 11)


async def test_a_transcript_that_starts_part_way_through_a_turn_says_so_rather_than_what_was_asked(tmp_path: Path) -> None:
    transcript = tmp_path / "s1.jsonl"
    transcript.write_text("".join(f"{record}\n" for record in [said("r0", "picking up where it was"), *hour(turns=1, steps=1)]))
    answer = await sentences(await joined(transcript), SummaryStore(Sentences(tmp_path / "sentences.db")))
    assert answer["turns"][0] == {"turn": 1, "began": "Claude said:\npicking up where it was"}
    assert answer["turns"][1] == {"turn": 2, "opened": "The user asked:\ndo task 0"}


async def test_a_turn_said_after_it_was_queued_is_not_asked_for_again(tmp_path: Path) -> None:
    """Read twice while the first pass runs, or told aloud in between: the second pass finds it said."""
    transcript = tmp_path / "s1.jsonl"
    transcript.write_text("".join(f"{record}\n" for record in hour(turns=3, steps=1)))
    sessions = await at_prompt(await joined(transcript))
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    await sentences(sessions, store)
    first = await store.wanted()
    # Taken, so a read in the meantime queues the same turns again.
    await sentences(sessions, store)
    assert isinstance(first, Turns)
    asked: list[str] = []

    async def summarise(page: str) -> str:
        asked.append(page)
        return "\n".join(f"{due.id}: What {due.id} did." for due in first.due)

    recorded: list[Entry] = []
    await summarise_turns(first, store, summarise, recorded.append)
    second = await store.wanted()
    assert isinstance(second, Turns)
    await summarise_turns(second, store, summarise, recorded.append)
    assert len(asked) == 1
    [_, again] = recorded
    # Everything said since it was queued: a pass of zeros but what it found known.
    assert isinstance(again, WideEvent) and again.counts == {"known": 3, "asked": 0, "said": 0, "calls": 0, "failed_calls": 0, "stray": 0}


async def test_a_turns_pass_says_what_its_replies_left_out_and_its_calls_raised(tmp_path: Path) -> None:
    transcript = tmp_path / "s1.jsonl"
    transcript.write_text("".join(f"{record}\n" for record in hour(turns=3, steps=1)))
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    await sentences(await at_prompt(await joined(transcript)), store)
    wanted = await store.wanted()
    assert isinstance(wanted, Turns)
    _, second, third = (due.id for due in wanted.due)
    calls = 0

    async def summarise(page: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("no answer in 180s")
        # Each reply names the second turn and one made up: asked for the third, it leaves it out.
        return f"{second}: What it did.\nz9: Not asked."

    recorded: list[Entry] = []
    await summarise_turns(wanted, store, summarise, recorded.append, batch=1)
    [pass_] = recorded
    assert isinstance(pass_, WideEvent) and pass_.outcome == "failed" and pass_.error == "the summariser failed on 1 things: TimeoutError: no answer in 180s"
    assert pass_.counts == {"known": 0, "asked": 3, "said": 1, "calls": 3, "failed_calls": 1, "stray": 3}
    assert (pass_.facts["left_out"], pass_.facts["errors"]) == ((third,), ("TimeoutError: no answer in 180s",))
