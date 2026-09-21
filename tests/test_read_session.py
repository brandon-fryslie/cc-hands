"""read_session hands the intermediary what a session did, in order, from the point it last read to."""

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from pipecat.services.llm_service import FunctionCallParams

from hands.core.events import Joined
from hands.core.session import Membership, SessionId
from hands.sessions.registry import Sessions
from hands.voice.tools import READBACK_COUNT, read_session_tool

FIXTURE = Path(__file__).parent / "fixtures" / "session.jsonl"
SID = SessionId("s1")


async def joined(transcript: Path) -> Sessions:
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _entry: None)
    await sessions.apply(Joined(Membership(SID, pid=4242, cwd=Path("/code/a"), transcript=transcript), "startup"))
    return sessions


async def read(sessions: Sessions, session: str = "s1", since: str = "") -> dict[str, Any]:
    results: list[object] = []

    async def capture(result: object, **_: object) -> None:
        results.append(result)

    tool = read_session_tool(sessions)
    await tool(cast(FunctionCallParams, SimpleNamespace(result_callback=capture)), session=session, since=since)
    [result] = results
    return cast(dict[str, Any], result)


async def test_a_session_is_read_back_in_order_and_everything_names_its_record(tmp_path: Path) -> None:
    transcript = tmp_path / "s1.jsonl"
    shutil.copy(FIXTURE, transcript)
    answer = await read(await joined(transcript))
    said = [happening["what"] for happening in answer["happened"]]
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
    assert all(happening["record"] for happening in answer["happened"])
    assert answer["more"] is False


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


async def test_a_session_with_more_than_one_reading_says_where_to_read_on_from(tmp_path: Path) -> None:
    """An hour of work is hundreds of steps, and all of them at once is a context spent on history."""
    transcript = tmp_path / "s1.jsonl"
    body = FIXTURE.read_bytes()
    # The same real session over again, which is a session that ran for twice as long.
    transcript.write_bytes(body * 6)
    answer = await read(await joined(transcript))
    assert len(answer["happened"]) == READBACK_COUNT
    assert answer["more"] is True and answer["more_since"] == answer["happened"][-1]["record"]


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


async def test_a_mark_this_session_never_held_is_said_rather_than_read_as_the_start(tmp_path: Path) -> None:
    """[LAW:no-silent-failure] a mark from another session would otherwise re-tell this one from the top."""
    transcript = tmp_path / "s1.jsonl"
    shutil.copy(FIXTURE, transcript)
    answer = await read(await joined(transcript), since="not-a-record")
    assert answer == {"error": "session s1 has no record not-a-record; call again with since empty to read from the start"}


async def test_a_session_the_registry_never_heard_of_is_said_to_be_no_session(tmp_path: Path) -> None:
    answer = await read(await joined(tmp_path / "s1.jsonl"), session="nobody")
    assert answer == {"error": "there is no session nobody"}


async def test_a_transcript_that_cannot_be_read_is_said_rather_than_answered_with_nothing(tmp_path: Path) -> None:
    """[LAW:no-silent-failure] the model is told why it got nothing, rather than being handed nothing."""
    answer = await read(await joined(tmp_path / "gone.jsonl"))
    assert answer == {"error": "the transcript of session s1 could not be read"}
