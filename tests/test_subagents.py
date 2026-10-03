"""A subagent's own work is read from its own transcript and told with the turn it reported back to, so what the
reviewer found is answered from the reviewer's steps rather than from the line its parent made of them."""

import json
from pathlib import Path
from typing import cast

from loguru import logger

from hands.core.attention import Spoken
from hands.core.delta import Delta
from hands.core.session import Membership, PromptId, SessionId
from hands.core.steps import Call, Result, recognise
from hands.core.narration import narration
from hands.core.subagents import Subagent, reporting
from hands.core.turn import AgentId, AgentTask, Asked, Continuing, Delegated, Notified, Other, Ran, Ref, Reported, Said, Turn
from hands.sessions.audit import Entry, Recounted
from hands.sessions.payload import Payload
from hands.sessions.registry import Sessions
from hands.sessions.subagents import read_subagent
from hands.sessions.tail import Tails
from hands.sessions.transcript import edge_of, turn_record
from hands.sessions.turning import Turning
from hands.voice.narrator import Recounts, recount
from hands.voice.tools import expand_tool

from test_narrator import Registry

SID = SessionId("s1")
REVIEWER = AgentId("a1b2c3d4e5f6a7b8c")
FOUND = "parse() drops the last field when a line ends in a comma"
WORK = "the subagent's work on /code-review medium 66"


def _record(fields: dict[str, object]) -> str:
    return json.dumps(fields, separators=(",", ":"))


def notification(task: str, summary: str) -> str:
    """A task notification's own markup, the same wherever Claude Code writes it."""
    return f"<task-notification>\n<task-id>{task}</task-id>\n<status>completed</status>\n<summary>{summary}</summary>\n<result>One finding.</result>\n</task-notification>"


def notified(task: str, summary: str) -> str:
    """A task notification as Claude Code writes it into the parent's transcript, opening a turn of its own."""
    text = notification(task, summary)
    return _record({"type": "user", "uuid": "n1", "promptId": "p2", "origin": {"kind": "task-notification", "producer": "session-task"}, "message": {"role": "user", "content": text}})


def reported(task: str, summary: str) -> str:
    """A task notification as Claude Code writes it when the session is still working: an attachment of the turn under way."""
    attachment = {"type": "queued_command", "prompt": notification(task, summary), "commandMode": "task-notification", "origin": {"kind": "task-notification", "producer": "session-task"}}
    return _record({"type": "attachment", "uuid": "q1", "parentUuid": "u2", "attachment": attachment})


def working(tmp_path: Path, task: str) -> Path:
    """A session that launches the reviewer in the background and is still working when it reports back."""
    transcript = tmp_path / "s1.jsonl"
    records = [
        _record({"type": "user", "uuid": "u1", "promptId": "p2", "message": {"role": "user", "content": "Review PR 66 while you fix the parser."}}),
        _record({"type": "assistant", "uuid": "c1", "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Agent", "input": {"description": "/code-review medium 66", "prompt": "Review PR 66.", "run_in_background": True}}]}}),
        _record({"type": "user", "uuid": "u2", "promptId": "p2", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Async agent launched successfully."}]}}),
        reported(task, 'Agent "/code-review medium 66" finished'),
        _record({"type": "assistant", "uuid": "r1", "message": {"content": [{"type": "text", "text": "The review found one issue."}]}}),
    ]
    transcript.write_text("".join(f"{record}\n" for record in records))
    return transcript


def parent(tmp_path: Path, task: str, summary: str = 'Agent "/code-review medium 66" finished') -> Path:
    """A session whose turn opens on a notification and ends on the parent's one-line account of it."""
    transcript = tmp_path / "s1.jsonl"
    reply = _record({"type": "assistant", "uuid": "r1", "message": {"content": [{"type": "text", "text": "The review found one issue."}]}})
    transcript.write_text(f"{notified(task, summary)}\n{reply}\n")
    return transcript


def reviewer(transcript: Path, *, forked: bool = False) -> None:
    """The reviewer's own transcript, where Claude Code writes it: every record a sidechain one. A fork's starts with
    the parent's call that launched it, copied in, where any other subagent's starts with its prompt."""
    folder = transcript.with_suffix("") / "subagents"
    folder.mkdir(parents=True)
    own: dict[str, object] = {"isSidechain": True, "agentId": REVIEWER}
    launched: list[dict[str, object]] = [
        {"type": "fork-context-ref", "agentId": REVIEWER, "parentSessionId": "s1"},
        {**own, "parentUuid": None, "type": "assistant", "uuid": "a0", "message": {"content": [{"type": "tool_use", "id": "t0", "name": "Agent", "input": {"description": "/code-review medium 66", "subagent_type": "fork", "prompt": "Review PR 66."}}]}},
        {**own, "parentUuid": "a0", "type": "user", "uuid": "a1", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "Fork started — processing in background"}, {"type": "text", "text": "<fork-boilerplate>Review PR 66.</fork-boilerplate>"}]}},
    ]
    prompted: list[dict[str, object]] = [{**own, "parentUuid": None, "type": "user", "uuid": "a1", "message": {"role": "user", "content": "Review PR 66 for bugs."}}]
    records: list[dict[str, object]] = [
        *(launched if forked else prompted),
        {**own, "parentUuid": "a1", "type": "assistant", "uuid": "a2", "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "git diff master...HEAD"}}]}},
        {**own, "parentUuid": "a2", "type": "user", "uuid": "a3", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "+    return line.split(',')[:-1]"}]}},
        {**own, "parentUuid": "a3", "type": "assistant", "uuid": "a4", "message": {"content": [{"type": "text", "text": f"Bug: {FOUND}."}]}},
    ]
    (folder / f"agent-{REVIEWER}.jsonl").write_text("".join(f"{_record(record)}\n" for record in records))


async def told(transcript: Path, recorded: list[Entry]) -> Recounts:
    recounts = Recounts()
    tails = Tails(Registry(Membership(SID, pid=4242, cwd=Path("/code/cc-hands"), transcript=transcript)))
    await recount(tails, SID, PromptId("p2"), None, recorded.append, Delta(), Spoken("full", "finished"), recounts)
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
    # The telling says whose work it carries, so progress heard of that subagent gives way to it.
    held = recounts.of(SID)
    assert held is not None and held.tellings[-1].reported == frozenset({REVIEWER})


async def test_a_reviewer_that_reports_while_its_parent_works_is_told_with_that_turn(tmp_path: Path) -> None:
    transcript = working(tmp_path, REVIEWER)
    reviewer(transcript)
    recorded: list[Entry] = []
    recounts = await told(transcript, recorded)
    [work] = await opened(recounts, WORK)
    assert FOUND in work
    [recounted] = [entry for entry in recorded if isinstance(entry, Recounted)]
    assert recounted.subagents == (REVIEWER,) and WORK in recounted.topics
    held = recounts.of(SID)
    assert held is not None and held.tellings[-1].reported == frozenset({REVIEWER})
    # One subagent, launched and then heard from: counted once, and its report as what arrived, not a second subagent.
    told_parts = {segment.topic.name: segment.text for segment in held.parts}
    assert told_parts["the subagents"] == "The subagents: one subagent."
    assert told_parts["the notifications"] == "The notifications: one notification."


def test_a_notification_handed_to_a_working_session_is_a_step_of_its_turn() -> None:
    """Not an opening, and not a tool: a message queued after it is folded into the same turn, as it was before it."""
    turning = Turning()
    queued = _record({"type": "user", "uuid": "u3", "promptId": "p2", "message": {"role": "user", "content": "and the docs"}})
    records = [
        _record({"type": "user", "uuid": "u1", "promptId": "p2", "message": {"role": "user", "content": "Review PR 66."}}),
        _record({"type": "assistant", "uuid": "c1", "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "pytest"}}]}}),
        _record({"type": "user", "uuid": "u2", "promptId": "p2", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "1 passed"}]}}),
        reported(REVIEWER, 'Agent "/code-review medium 66" finished'),
        queued,
    ]
    edges = [turning.consume(record) for line in records if (record := turn_record(line.encode())) is not None]
    assert [type(edge) for edge in edges] == [Asked, type(None), type(None), type(None), type(None)]
    [report] = [step for step in turning.steps() if isinstance(step, Reported)]
    assert report.ref == Ref("q1") and report.agent == AgentTask(REVIEWER, "/code-review medium 66")


def test_an_attachment_that_only_mentions_a_notification_is_no_record_of_a_turn() -> None:
    hooked = _record({"type": "attachment", "uuid": "h1", "attachment": {"type": "hook_additional_context", "commandMode": "task-notification"}})
    assert b'"commandMode":"task-notification"' in hooked.encode() and turn_record(hooked.encode()) is None


async def test_a_notification_from_a_background_command_is_told_as_it_was(tmp_path: Path) -> None:
    """A background command or a monitor notifies as a subagent does, and has no subagent's work to tell."""
    recorded: list[Entry] = []
    recounts = await told(parent(tmp_path, "b7x9k2", 'Background command "pytest" completed (exit code 0)'), recorded)
    held = recounts.of(SID)
    assert held is not None and WORK not in {segment.topic.name for segment in held.parts}
    [recounted] = [entry for entry in recorded if isinstance(entry, Recounted)]
    assert recounted.subagents == () and recounted.unread == ()


async def test_a_subagent_whose_work_cannot_be_read_is_said_and_its_turn_is_still_told(tmp_path: Path) -> None:
    transcript = parent(tmp_path, REVIEWER)
    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(str(message)), level="ERROR")
    try:
        recorded: list[Entry] = []
        recounts = await told(transcript, recorded)
    finally:
        logger.remove(sink)
    assert recounts.of(SID) is not None
    assert errors and REVIEWER in errors[0] and "FileNotFoundError" in errors[0]
    [recounted] = [entry for entry in recorded if isinstance(entry, Recounted)]
    assert recounted.subagents == () and recounted.unread == (REVIEWER,)


def test_a_subagent_is_read_with_the_sessions_own_fold(tmp_path: Path) -> None:
    transcript = tmp_path / "s1.jsonl"
    reviewer(transcript)
    subagent = read_subagent(transcript, AgentTask(REVIEWER, "/code-review medium 66"))
    ran, said = subagent.steps
    assert isinstance(ran, Ran) and ran.command == "git diff master...HEAD"
    assert said == Said(Ref("a4"), f"Bug: {FOUND}.")


def test_a_forks_work_starts_after_the_call_that_launched_it(tmp_path: Path) -> None:
    """A fork's transcript opens on its parent's call, copied in: the job, not a step of the work."""
    transcript = tmp_path / "s1.jsonl"
    reviewer(transcript, forked=True)
    subagent = read_subagent(transcript, AgentTask(REVIEWER, "/code-review medium 66"))
    assert [type(step) for step in subagent.steps] == [Ran, Said]


def test_a_later_telling_of_a_notified_turn_reads_its_subagent_no_more() -> None:
    """The notification opens the turn, and only the telling that answers it carries the work; a telling after a
    blocked stop tells only the steps since."""
    task = AgentTask(REVIEWER, "/code-review medium 66")
    opening = Notified(None, "<task-notification/>", task)
    assert reporting(Turn(opening, ())) == (task,)
    assert reporting(Turn(opening, (), Continuing(3))) == ()


def test_a_subagent_that_did_nothing_has_no_work_to_open() -> None:
    tree = narration(Turn(Asked(None, "review it"), ()), Delta(), (Subagent(REVIEWER, "/code-review medium 66", ()),))
    assert tree.subagents == ()


def test_a_notification_names_the_subagent_and_the_job_it_was_given(tmp_path: Path) -> None:
    transcript = parent(tmp_path, REVIEWER, 'Agent "say "hi" twice" was stopped by Claude')
    [line, _] = transcript.read_bytes().split(b"\n", 1)
    opening = edge_of(Payload.parse(line), mid_tool=False)
    assert isinstance(opening, Notified) and opening.agent == AgentTask(REVIEWER, 'say "hi" twice')


def test_a_skill_run_in_a_fork_of_its_own_is_a_delegation_to_the_subagent_it_names() -> None:
    """/code-review runs this way: the reviewer is a subagent, launched in the background, which reports back later."""
    forked = {"success": True, "commandName": "code-review", "status": "forked", "background": True, "agentId": REVIEWER, "result": "Running in the background as @code-review"}
    result = Result('Skill "code-review" launched (forked execution, running in the background).', forked, False)
    step = recognise(Call(Ref("u1"), "Skill", {"skill": "code-review", "args": "medium 66"}, result))
    assert step == Delegated(Ref("u1"), "code-review", "/code-review medium 66", None, REVIEWER)


def test_a_skill_run_in_a_fork_in_the_foreground_reports_its_result_without_claude_codes_heading() -> None:
    forked = {"success": True, "commandName": "code-review", "status": "forked", "agentId": REVIEWER, "result": "One finding."}
    result = Result('Skill "code-review" completed (forked execution).\n\nResult:\nOne finding.', forked, False)
    step = recognise(Call(Ref("u1"), "Skill", {"skill": "code-review", "args": "medium 66"}, result))
    assert step == Delegated(Ref("u1"), "code-review", "/code-review medium 66", "One finding.", REVIEWER)


def test_a_skill_loaded_into_the_session_is_no_delegation() -> None:
    loaded = Result("Launching skill: laws:code", {"success": True, "commandName": "laws:code"}, False)
    assert isinstance(recognise(Call(Ref("u1"), "Skill", {"skill": "laws:code"}, loaded)), Other)


def test_two_subagents_in_one_turn_are_opened_one_at_a_time() -> None:
    reviewer, tester = Subagent(REVIEWER, "/code-review medium 66", (Said(None, "One bug."),)), Subagent(AgentId("b2"), "Run the tests", (Said(None, "All pass."),))
    tree = narration(Turn(Asked(None, "review and test it"), ()), Delta(), (reviewer, tester))
    assert [segment.topic.name for segment in tree.subagents] == [WORK, "the subagent's work on run the tests"]


def test_a_transcript_that_starts_part_way_through_its_parents_keeps_its_first_step(tmp_path: Path) -> None:
    """A fork written before Claude Code copied the launching call in starts on its own work, under a parent in the
    parent's transcript."""
    transcript = tmp_path / "s1.jsonl"
    folder = transcript.with_suffix("") / "subagents"
    folder.mkdir(parents=True)
    own: dict[str, object] = {"isSidechain": True, "agentId": REVIEWER}
    records: list[dict[str, object]] = [
        {**own, "parentUuid": "p9", "type": "assistant", "uuid": "a2", "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "git diff master...HEAD"}}]}},
        {**own, "parentUuid": "a2", "type": "user", "uuid": "a3", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "+    return line"}]}},
    ]
    (folder / f"agent-{REVIEWER}.jsonl").write_text("".join(f"{_record(record)}\n" for record in records))
    [ran] = read_subagent(transcript, AgentTask(REVIEWER, "/code-review medium 66")).steps
    assert isinstance(ran, Ran)
