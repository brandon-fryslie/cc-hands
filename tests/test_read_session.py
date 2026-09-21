"""read_session hands the intermediary what a session did, in order, from the point it last read to."""

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from pipecat.services.llm_service import FunctionCallParams

from hands.core.events import Joined
from hands.core.session import Membership, SessionId
from hands.sessions.registry import Sessions
from hands.voice.tools import STEP_READBACK, read_session_tool

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


async def test_a_session_is_read_back_in_order_and_every_step_names_its_record(tmp_path: Path) -> None:
    transcript = tmp_path / "s1.jsonl"
    shutil.copy(FIXTURE, transcript)
    answer = await read(await joined(transcript))
    said = [step["step"] for step in answer["steps"]]
    assert [words.split()[1].rstrip(":") for words in said] == ["said", "ran", "ran", "said", "said", "ran", "ran", "wrote", "ran", "gave", "said"]
    # A command is cut to its budget, as every part of a step is, and keeps the purpose Claude gave it.
    assert said[1].startswith("Claude ran ls ~/code | grep -i pipecat") and "... (cut) (Check local Pipecat presence" in said[1]
    # A subagent is one step: the job it was given, and that it has not reported back.
    assert said[9] == "Claude gave the general-purpose subagent this job: Rewrite docs for Pipecat architecture\nIt is still working."
    # The record id is what the next reading is asked from, so every step carries the one it came from.
    assert all(step["record"] for step in answer["steps"])
    assert answer["more"] is False


async def test_reading_on_from_where_it_read_to_gives_what_came_after_and_nothing_twice(tmp_path: Path) -> None:
    transcript = tmp_path / "s1.jsonl"
    shutil.copy(FIXTURE, transcript)
    sessions = await joined(transcript)
    whole = await read(sessions)
    third = whole["steps"][2]["record"]
    after = await read(sessions, since=third)
    assert after["steps"] == whole["steps"][3:]
    # Read on from its own last record, a session has nothing left to say and says so without an error.
    assert (await read(sessions, since=whole["steps"][-1]["record"]))["steps"] == []


async def test_a_session_with_more_than_one_reading_of_steps_says_where_to_read_on_from(tmp_path: Path) -> None:
    """An hour of work is hundreds of steps, and all of them at once is a context spent on history."""
    transcript = tmp_path / "s1.jsonl"
    body = FIXTURE.read_bytes()
    # The same real session over again, which is a session that ran for twice as long.
    transcript.write_bytes(body * 6)
    answer = await read(await joined(transcript))
    assert len(answer["steps"]) == STEP_READBACK
    assert answer["more"] is True and answer["more_since"] == answer["steps"][-1]["record"]


async def test_a_session_the_registry_never_heard_of_is_said_to_be_no_session(tmp_path: Path) -> None:
    answer = await read(await joined(tmp_path / "s1.jsonl"), session="nobody")
    assert answer == {"error": "there is no session nobody"}


async def test_a_transcript_that_cannot_be_read_is_said_rather_than_answered_with_nothing(tmp_path: Path) -> None:
    """[LAW:no-silent-failure] the model is told why it got nothing, rather than being handed nothing."""
    answer = await read(await joined(tmp_path / "gone.jsonl"))
    assert answer == {"error": "the transcript of session s1 could not be read"}
