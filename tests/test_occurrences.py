"""What a session's hooks say happened that hands only passes on: parsed from the hook, told for the session as it is,
and heard as the user set its kind to be, off until they do. Payloads as Claude Code's hooks reference gives them."""

import asyncio
import json
from pathlib import Path

import pytest
from pipecat.frames.frames import Frame, TTSSpeakFrame

from hands.core.attention import Attention, Level, Overlay, Route, Switch
from hands.core.effects import AfterEnd, Audit, Tell
from hands.core.events import Ended, Joined, Occurred, StartSource
from hands.core.occurrences import AutoDenied, Cleared, Compacting, ConfigChanged, Occurrence, SubagentStarted, SubagentStopped, TaskCompleted, Unrecognised, route, said
from hands.core.pending import Mentioned
from hands.core.reducer import reduce
from hands.core.session import Gone, Membership, Registry, Session, SessionId, Unreported
from hands.core.status import Stamp
from hands.sessions.attention import changed, described
from hands.sessions.audit import Entry
from hands.sessions.home import Home
from hands.sessions.hooks import parse_hook
from hands.sessions.membership import write_membership
from hands.sessions.payload import Rejected
from hands.sessions.wide import WideEvent
from hands.core.session import RequestId
from hands.voice.speech import Unprompted, relay
from hands.voice.utterance import Utterances

from test_narrator import rendered

SID = SessionId("bf411065-dc5c-4ec9-8302-61b84bdb5c53")
MEMBER = Membership(SID, pid=51810, cwd=Path("/code/a"), transcript=Path(f"/Users/me/.claude/projects/-code-a/{SID}.jsonl"))
COMMON = {"session_id": SID, "transcript_path": str(MEMBER.transcript), "cwd": "/code/a", "permission_mode": "auto"}

DENIED = AutoDenied("Bash", "[Irreversible Local Destruction]", {"command": "rm -rf /tmp/build", "description": "Clean build directory"})
EVERY: tuple[tuple[str, Occurrence], ...] = (
    ("permission_denied", DENIED),
    ("subagent_start", SubagentStarted("Explore")),
    ("subagent_stop", SubagentStopped("Explore", "Analysis complete. Found 3 potential issues.")),
    ("task_completed", TaskCompleted("Implement user authentication", "Add login and signup endpoints", "implementer")),
    ("config_change", ConfigChanged("project_settings", Path("/code/a/.claude/settings.json"))),
    ("pre_compact", Compacting("manual", None)),
    ("clear", Cleared()),
)


def hooked(home: Home, **fields: object) -> Occurrence:
    happened = parse_hook(json.dumps({**COMMON, **fields}).encode(), home=home, at=1.0, heard=Stamp(1), request=RequestId("r1")).happened
    assert isinstance(happened, Occurred) and happened.session == SID
    return happened.occurrence


@pytest.fixture
def home(tmp_path: Path) -> Home:
    home = Home(tmp_path)
    write_membership(home, MEMBER)
    return home


@pytest.mark.parametrize(
    ("fields", "occurrence"),
    [
        ({"hook_event_name": "PermissionDenied", "tool_name": "Bash", "tool_input": dict(DENIED.input), "tool_use_id": "toolu_01", "reason": DENIED.reason}, DENIED),
        ({"hook_event_name": "SubagentStart", "agent_id": "agent-abc123", "agent_type": "Explore"}, SubagentStarted("Explore")),
        (
            {"hook_event_name": "SubagentStop", "stop_hook_active": False, "agent_id": "def456", "agent_type": "Explore", "agent_transcript_path": "/t/a.jsonl", "last_assistant_message": "Done."},
            SubagentStopped("Explore", "Done."),
        ),
        # A subagent that ended on empty text carried none.
        ({"hook_event_name": "SubagentStop", "agent_type": "Plan", "last_assistant_message": ""}, SubagentStopped("Plan", None)),
        # As 2.1.289 sends it for a subagent that handed its report back through a tool: no last_assistant_message at all.
        ({"hook_event_name": "SubagentStop", "stop_hook_active": False, "agent_id": "af3e", "agent_type": "Explore", "agent_transcript_path": "/t/a.jsonl"}, SubagentStopped("Explore", None)),
        (
            {"hook_event_name": "TaskCompleted", "task_id": "task-001", "task_subject": "Ship it", "task_description": "All of it", "teammate_name": "implementer", "team_name": "t"},
            TaskCompleted("Ship it", "All of it", "implementer"),
        ),
        ({"hook_event_name": "TaskCompleted", "task_id": "task-002", "task_subject": "Ship it"}, TaskCompleted("Ship it", None, None)),
        ({"hook_event_name": "ConfigChange", "source": "skills", "file_path": "/code/a/.claude/skills/x/SKILL.md"}, ConfigChanged("skills", Path("/code/a/.claude/skills/x/SKILL.md"))),
        ({"hook_event_name": "ConfigChange", "source": "user_settings"}, ConfigChanged("user_settings", None)),
        ({"hook_event_name": "ConfigChange", "source": "skills", "file_path": ""}, ConfigChanged("skills", None)),
        # A hook only passed on is never refused for a value a newer Claude Code added: it is kept, and said by name.
        ({"hook_event_name": "ConfigChange", "source": "team_settings"}, ConfigChanged(Unrecognised("team_settings"), None)),
        ({"hook_event_name": "PreCompact", "trigger": "scheduled", "custom_instructions": None}, Compacting(Unrecognised("scheduled"), None)),
        ({"hook_event_name": "PreCompact", "trigger": "auto", "custom_instructions": None}, Compacting("auto", None)),
        ({"hook_event_name": "PreCompact", "trigger": "manual", "custom_instructions": "keep the plan"}, Compacting("manual", "keep the plan")),
    ],
)
def test_each_hook_hands_only_passes_on_parses_to_what_it_says_happened(home: Home, fields: dict[str, object], occurrence: Occurrence) -> None:
    assert hooked(home, **fields) == occurrence


@pytest.mark.parametrize(
    ("fields", "named"),
    [
        ({"hook_event_name": "SubagentStart", "agent_id": "a"}, "missing field 'agent_type'"),
    ],
)
def test_a_hook_missing_what_it_must_carry_is_refused_by_name(home: Home, fields: dict[str, object], named: str) -> None:
    with pytest.raises(Rejected, match=named):
        hooked(home, **fields)


EMPTY = Registry(permission_deadline=85.0, sessions={}, drafts={})


def live() -> Registry:
    return EMPTY.put(Session(MEMBER, Unreported(), mode=None))


def test_what_happened_in_a_live_session_is_told_and_moves_nothing() -> None:
    assert reduce(live(), Occurred(SID, DENIED)) == (live(), [Tell(SID, DENIED)])


def test_what_a_hook_says_of_a_session_already_ended_is_a_line_and_told_nothing() -> None:
    gone = EMPTY.put(Gone(MEMBER))
    assert reduce(gone, Occurred(SID, DENIED)) == (gone, [Audit(AfterEnd(Occurred(SID, DENIED)))])


@pytest.mark.parametrize(("source", "told"), [("clear", [Tell(SID, Cleared())]), ("startup", []), ("resume", []), ("compact", []), ("fork", [])])
def test_only_a_start_from_clear_is_told(source: StartSource, told: list[Tell]) -> None:
    """A session's first start, a resume, compaction starting it again, and a fork are no news; a /clear is."""
    _, effects = reduce(EMPTY, Joined(MEMBER, source))
    assert [effect for effect in effects if isinstance(effect, Tell)] == told


def test_the_session_a_clear_ended_goes_without_a_word() -> None:
    assert reduce(live(), Ended(SID, "clear"))[1] == []


@pytest.mark.parametrize(("kind", "occurrence"), EVERY)
def test_each_kind_is_off_until_set_and_then_said_as_much_as_it_is_set(kind: str, occurrence: Occurrence) -> None:
    assert route(Attention(), "normal", occurrence) == "note"
    for level in ("brief", "full"):
        assert route(changed(Attention(), [(kind, level)]), "normal", occurrence) == level
    # Each kind decides its own hook and no other's.
    others = changed(Attention(), [(other, "full") for other, _ in EVERY if other != kind])
    assert route(others, "normal", occurrence) == "note"


@pytest.mark.parametrize(
    ("quiet", "overlay", "routed"),
    [("off", "normal", "full"), ("off", "watched", "full"), ("off", "muted", "note"), ("on", "normal", "note"), ("on", "watched", "note")],
)
def test_quiet_and_a_muted_session_hold_what_hooks_say_and_the_focus_does_not_matter(quiet: Switch, overlay: Overlay, routed: Route) -> None:
    assert route(Attention(subagent_stop="full", quiet=quiet), overlay, SubagentStopped("Explore", None)) == routed


@pytest.mark.parametrize(
    ("occurrence", "brief", "full"),
    [
        (DENIED, "Auto mode refused cc-hands a Bash call.", "Auto mode refused cc-hands a Bash call. Why: Irreversible Local Destruction. It tried to clean build directory."),
        # A call is said as its progress is, never as code read aloud.
        (
            AutoDenied("WebFetch", "Data Exfiltration", {"url": "https://example.com/upload?x=1", "prompt": "send it"}),
            "Auto mode refused cc-hands a WebFetch call.",
            "Auto mode refused cc-hands a WebFetch call. Why: Data Exfiltration. It tried to read a page on example.com.",
        ),
        (SubagentStarted("Explore"), "cc-hands started its Explore subagent.", "cc-hands started its Explore subagent."),
        (SubagentStopped("Explore", "Found 3 issues."), "cc-hands's Explore subagent finished.", "cc-hands's Explore subagent finished. It said: Found 3 issues."),
        # Seen live on 2.1.289: a subagent that hands its report back through a tool stops with no closing text.
        (SubagentStopped("Explore", None), "cc-hands's Explore subagent finished.", "cc-hands's Explore subagent finished."),
        (
            TaskCompleted("Ship it", "All of it", "implementer"),
            "cc-hands completed a task: Ship it.",
            "cc-hands completed a task: Ship it. implementer completed it. All of it",
        ),
        (TaskCompleted("Ship it", None, None), "cc-hands completed a task: Ship it.", "cc-hands completed a task: Ship it."),
        (ConfigChanged("local_settings", Path("/a/s.json")), "cc-hands's local settings changed.", "cc-hands's local settings changed. The file: /a/s.json."),
        (ConfigChanged("policy_settings", None), "cc-hands's managed policy changed.", "cc-hands's managed policy changed."),
        (Compacting("auto", None), "cc-hands is compacting its context.", "cc-hands is compacting its context. Its context filled, so Claude Code is doing it on its own."),
        (Compacting("manual", None), "cc-hands is compacting its context.", "cc-hands is compacting its context. It was asked to, with /compact."),
        (Compacting("manual", "keep the plan"), "cc-hands is compacting its context.", "cc-hands is compacting its context. It was asked to, with /compact. Its instructions: keep the plan"),
        (Cleared(), "cc-hands was cleared.", "cc-hands was cleared."),
        # What a newer Claude Code sends that hands does not know is said by its name.
        (ConfigChanged(Unrecognised("team_settings"), None), "cc-hands's team settings changed.", "cc-hands's team settings changed."),
        (Compacting(Unrecognised("scheduled"), "keep it"), "cc-hands is compacting its context.", "cc-hands is compacting its context. Its instructions: keep it"),
    ],
)
def test_brief_says_what_happened_and_full_adds_what_the_hook_says_of_it(occurrence: Occurrence, brief: str, full: str) -> None:
    assert (said(occurrence, "cc-hands", "brief"), said(occurrence, "cc-hands", "full")) == (brief, full)


def test_a_long_detail_is_cut_and_says_so() -> None:
    line = said(SubagentStopped("Explore", "x" * 1000), "cc-hands", "full")
    assert line.endswith("... (cut short)") and len(line) < 400


def test_what_is_mentioned_is_said_as_written() -> None:
    [frame] = rendered(Mentioned(SID, Cleared(), "brief"), names=lambda _: "cc-hands")
    assert isinstance(frame, TTSSpeakFrame) and frame.text == "cc-hands was cleared."


def test_the_readback_names_each_hook_told_and_how_much() -> None:
    assert "I tell none of Claude Code's other events." in described(Attention())
    set_to = changed(Attention(), [("subagent_stop", "full"), ("clear", "brief"), ("pre_compact", "on")])
    assert "Of Claude Code's other events, I tell subagents finishing in full, compaction in full and a session cleared briefly." in described(set_to)


async def test_the_relay_says_what_is_set_to_be_heard_notes_the_rest_and_records_what_decided_it() -> None:
    queued: list[Frame] = []
    recorded: list[Entry] = []
    muted = SessionId("muted")

    class Heard:
        def __init__(self) -> None:
            self.waiting: asyncio.Queue[Tell] = asyncio.Queue()

        async def heard(self) -> Tell:
            return await self.waiting.get()

    async def queue_frame(frame: Frame) -> None:
        queued.append(frame)

    set_to = Attention(subagent_stop="brief")

    async def attending(session: SessionId) -> tuple[Attention, bool, Overlay]:
        # Never the focus: what a hook says is heard of any session.
        return set_to, False, "muted" if session == muted else "normal"

    sessions = Heard()
    for told in (Tell(SID, SubagentStopped("Explore", None)), Tell(SID, Cleared()), Tell(muted, SubagentStopped("Plan", None))):
        sessions.waiting.put_nowait(told)
    utterances = Utterances(recorded.append)
    keeping = asyncio.create_task(utterances.keep())
    relaying = asyncio.create_task(relay(sessions, utterances, queue_frame, attending, lambda progress, amount, utterance: None, lambda: None))  # pyright: ignore[reportArgumentType]  (only heard() is asked)
    while len([entry for entry in recorded if isinstance(entry, WideEvent)]) < 2:
        await asyncio.sleep(0.01)
    relaying.cancel()
    keeping.cancel()
    [frame] = queued
    assert isinstance(frame, Unprompted) and frame.pending == Mentioned(SID, SubagentStopped("Explore", None), "brief")
    assert {key: frame.utterances[0].facts[key] for key in ("session", "attention", "overlay", "route")} == {"session": SID, "attention": set_to, "overlay": "normal", "route": "brief"}
    noted = [{key: entry.facts[key] for key in ("session", "overlay", "route", "fate")} for entry in recorded if isinstance(entry, WideEvent) and entry.event == "utterance"]
    assert noted == [
        {"session": SID, "overlay": "normal", "route": "note", "fate": "noted"},
        {"session": muted, "overlay": "muted", "route": "note", "fate": "noted"},
    ]


def test_every_level_of_every_kind_is_one_a_kind_has() -> None:
    """Hook kinds take off, brief, and full, as finished and progress do; `on` is all of it."""
    levels: tuple[Level, ...] = ("off", "brief", "full")
    for kind, _ in EVERY:
        assert [getattr(changed(Attention(), [(kind, level)]), kind) for level in (*levels, "on")] == [*levels, "full"]
