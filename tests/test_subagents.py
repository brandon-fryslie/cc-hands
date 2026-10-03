"""A subagent's own work is read from its own transcript and told with the turn it reported back to, so what the
reviewer found is answered from the reviewer's steps rather than from the line its parent made of them."""

import json
from pathlib import Path
from typing import cast

from loguru import logger

from hands.core.delta import Delta
from hands.core.session import Membership, PromptId, SessionId
from hands.core.steps import Call, Result, recognise
from hands.core.turn import AgentId, Delegated, Other, Ran, Ref, Said
from hands.sessions.audit import Entry, Recounted
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions
from hands.sessions.subagents import read_subagent
from hands.sessions.tail import Tails
from hands.voice.narrator import Recounts, recount
from hands.voice.tools import expand_tool

import pytest

from test_narrator import Registry

SID = SessionId("s1")
REVIEWER = AgentId("a1b2c3d4e5f6a7b8c")
FOUND = "parse() drops the last field when a line ends in a comma"
WORK = "the subagents' own work"


def _record(fields: dict[str, object]) -> str:
    return json.dumps(fields, separators=(",", ":"))


def notified(task: str) -> str:
    """A task notification as Claude Code writes it into the parent's transcript, opening a turn of its own."""
    text = f"<task-notification>\n<task-id>{task}</task-id>\n<status>completed</status>\n<summary>Agent \"/code-review medium 66\" finished</summary>\n<result>One finding.</result>\n</task-notification>"
    return _record({"type": "user", "uuid": "n1", "promptId": "p2", "origin": {"kind": "task-notification", "producer": "session-task"}, "message": {"role": "user", "content": text}})


def parent(tmp_path: Path, task: str) -> Path:
    """A session whose turn opens on a notification and ends on the parent's one-line account of it."""
    transcript = tmp_path / "s1.jsonl"
    reply = _record({"type": "assistant", "uuid": "r1", "message": {"content": [{"type": "text", "text": "The review found one issue."}]}})
    transcript.write_text(f"{notified(task)}\n{reply}\n")
    return transcript


def reviewer(transcript: Path, description: str | None = "/code-review medium 66") -> None:
    """The reviewer's own transcript and meta file, where Claude Code writes them: every record a sidechain one."""
    folder = transcript.with_suffix("") / "subagents"
    folder.mkdir(parents=True)
    own: dict[str, object] = {"isSidechain": True, "agentId": REVIEWER}
    records: list[dict[str, object]] = [
        {**own, "type": "user", "uuid": "a1", "message": {"role": "user", "content": "Review PR 66 for bugs."}},
        {**own, "type": "assistant", "uuid": "a2", "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "git diff master...HEAD"}}]}},
        {**own, "type": "user", "uuid": "a3", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "+    return line.split(',')[:-1]"}]}},
        {**own, "type": "assistant", "uuid": "a4", "message": {"content": [{"type": "text", "text": f"Bug: {FOUND}."}]}},
    ]
    (folder / f"agent-{REVIEWER}.jsonl").write_text("".join(f"{_record(record)}\n" for record in records))
    meta = {"agentType": "general-purpose"} | ({} if description is None else {"description": description})
    (folder / f"agent-{REVIEWER}.meta.json").write_text(json.dumps(meta))


async def told(transcript: Path, recorded: list[Entry]) -> Recounts:
    recounts = Recounts()
    tails = Tails(Registry(Membership(SID, pid=4242, cwd=Path("/code/cc-hands"), transcript=transcript)))
    await recount(tails, SID, PromptId("p2"), None, recorded.append, Delta(), "summaries", recounts)
    return recounts


async def opened(recounts: Recounts, part: str) -> list[str]:
    call = expand_tool(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None), recounts).body
    result = await call(session=SID, part=part)
    return [entry["told"] for entry in cast(list[dict[str, str]], result["parts"])]


async def test_what_the_reviewer_found_is_answered_from_the_reviewers_own_steps(tmp_path: Path) -> None:
    transcript = parent(tmp_path, REVIEWER)
    reviewer(transcript)
    recorded: list[Entry] = []
    recounts = await told(transcript, recorded)
    whole = await expand_tool(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None), recounts).body(session=SID)
    lines = [entry["told"] for entry in cast(list[dict[str, str]], whole["parts"]) if entry["part"] == WORK]
    # Named by the job it was given, and counted: its prompt is the job, and not a step of it.
    assert lines == ["A subagent's own work on /code-review medium 66: two steps."]
    [work] = await opened(recounts, WORK)
    assert FOUND in work and "git diff master...HEAD" in work
    [recounted] = [entry for entry in recorded if isinstance(entry, Recounted)]
    assert recounted.subagents == (REVIEWER,) and WORK in recounted.topics


async def test_a_notification_from_a_task_with_no_transcript_of_its_own_is_told_as_it_was(tmp_path: Path) -> None:
    """A background command or a monitor notifies as a subagent does, and has no subagent's work to tell."""
    recorded: list[Entry] = []
    recounts = await told(parent(tmp_path, "b7x9k2"), recorded)
    held = recounts.of(SID)
    assert held is not None and WORK not in {segment.topic.name for segment in held.parts}
    [recounted] = [entry for entry in recorded if isinstance(entry, Recounted)]
    assert recounted.subagents == ()


async def test_a_subagent_whose_work_cannot_be_read_is_said_and_its_turn_is_still_told(tmp_path: Path) -> None:
    transcript = parent(tmp_path, REVIEWER)
    reviewer(transcript, description=None)
    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(str(message)), level="ERROR")
    try:
        recorded: list[Entry] = []
        recounts = await told(transcript, recorded)
    finally:
        logger.remove(sink)
    assert recounts.of(SID) is not None
    assert errors and REVIEWER in errors[0] and "names no description" in errors[0]


def test_a_subagent_is_read_with_the_sessions_own_fold(tmp_path: Path) -> None:
    transcript = tmp_path / "s1.jsonl"
    reviewer(transcript)
    subagent = read_subagent(transcript, REVIEWER)
    assert subagent is not None and subagent.description == "/code-review medium 66"
    ran, said = subagent.steps
    assert isinstance(ran, Ran) and ran.command == "git diff master...HEAD"
    assert said == Said(Ref("a4"), f"Bug: {FOUND}.")
    assert read_subagent(transcript, "b7x9k2") is None


def test_a_meta_file_that_is_not_json_is_rejected(tmp_path: Path) -> None:
    transcript = tmp_path / "s1.jsonl"
    reviewer(transcript)
    (transcript.with_suffix("") / "subagents" / f"agent-{REVIEWER}.meta.json").write_text("{")
    with pytest.raises(Rejected):
        read_subagent(transcript, REVIEWER)


def test_a_skill_run_in_a_fork_of_its_own_is_a_delegation_to_the_subagent_it_names() -> None:
    """/code-review runs this way: the reviewer is a subagent, launched in the background, which reports back later."""
    forked = {"success": True, "commandName": "code-review", "status": "forked", "background": True, "agentId": REVIEWER, "result": "Running in the background as @code-review"}
    result = Result('Skill "code-review" launched (forked execution, running in the background).', forked, False)
    step = recognise(Call(Ref("u1"), "Skill", {"skill": "code-review", "args": "medium 66"}, result))
    assert step == Delegated(Ref("u1"), "code-review", "/code-review medium 66", None, REVIEWER)


def test_a_skill_loaded_into_the_session_is_no_delegation() -> None:
    loaded = Result("Launching skill: laws:code", {"success": True, "commandName": "laws:code"}, False)
    assert isinstance(recognise(Call(Ref("u1"), "Skill", {"skill": "laws:code"}, loaded)), Other)
