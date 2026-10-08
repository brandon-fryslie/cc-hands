"""The intermediary is told which sessions run and never their history, can decline to answer, and its eval judges the tools the daemon gives it."""

import json
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hands.voice.speakers import Room
from hands.core.session import SessionId
from hands.core.wire import Exchanged, MainTurn, Unreached
from hands.spotify import Catalogue, Missing
from hands.sessions.audit import SEGMENT_GLOB, AuditLog, Transcribed, segment
from hands.sessions.wide import annotate, fail, root, unit
from hands.sessions.home import Home
from hands.sessions.registry import Sessions
from hands.voice.briefing import tail
from hands.voice.intermediary_instruction import brain_instruction
from hands.sessions.sentences import Sentences
from hands.voice.sentences import SummaryStore
from hands.voice.narrator import Recounts
from hands.voice.player import Player
from hands.voice.refocus import Refocus
from hands.daemon.config import Config, OwnModel, Settings
from hands.voice.ptt import PushToTalk
from hands.voice.trigger import Triggers
from hands.voice.wakeword import Pretrained
from hands.voice.tools import intermediary_tools

AUTH = {"id": "5b0e2f4e-3c1a-4d8e-9f21-7a6c0d9e1b34", "name": "cc-hands, auth refactor", "state": "idle", "mode": "manual mode"}
FRESH = {"id": "c7d1a9e2-8f40-4b6a-a2d3-1e5f9c0b7a68", "name": "cc-hands", "state": "working", "mode": "not reported yet"}


def names(sessions: Sessions) -> list[str]:
    with tempfile.TemporaryDirectory() as home:
        return [tool.name for tool in intermediary_tools(sessions, SummaryStore(Sentences(Path(home) / "sentences.db")), Home(Path(home)), Recounts(), Player(lambda _entry: None), Refocus(sessions, Home(Path(home)), lambda _entry: None), PushToTalk(lambda _entry: None).switch, Triggers(), Pretrained(), OwnModel(Home(Path(home)), Settings(None, Config()), lambda _config: None), Catalogue(Missing("no credentials in tests")), Room(Path(home), lambda _entry: None), {"TMUX_TMPDIR": home}, lambda: None)]


def test_the_tail_names_each_session_by_name_state_and_mode_with_the_id_for_the_tools_and_says_it_is_current() -> None:
    told = tail([AUTH], None)
    assert told.startswith("[hands] ")
    assert f'"cc-hands, auth refactor" (id {AUTH["id"]}), idle, permission mode: manual mode' in told
    assert "as this message is sent" in told and "Say nothing about this unless the user asks." in told


def test_the_tail_with_nothing_running_says_so() -> None:
    assert tail([], None) == "[hands] No Claude Code sessions are running now. No session is focused. Say nothing about this unless the user asks."


def test_the_prompt_names_no_tool_the_daemon_does_not_give() -> None:
    # The rule in intermediary_instruction: a prompt that asks for a tool before it exists gets that tool paraphrased.
    # Every tool is snake_case, so every snake_case name in the prompt is a tool, bar the code names it quotes as ones never to say.
    quoted_code_names = {"parse_date", "test_invoice_total"}
    given = set(names(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)))
    prompt = brain_instruction(Path("/home/hands/audit"), Path("/home/hands/brain"), "hands recall", None)
    named = set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", prompt)) - quoted_code_names
    assert named, "the prompt names no tool at all"
    assert named <= given, f"the prompt names {sorted(named - given)}, which the daemon does not give"


def test_the_brain_is_told_of_the_log_of_lit_and_of_its_own_setup_and_it_keeps_the_closing_words_last() -> None:
    told = brain_instruction(Path("/my home/audit"), Path("/my home/brain"), "'/my python' -P -m hands.daemon recall", None)
    # Working a tracker is done from a shell, and the brain works one with lit.
    assert "lit quickstart" in told
    # Asked to install a skill and told nothing of its setup, the brain made it in ~/.claude/skills, the user's own
    # (hands-brain-d8g.33b, 2.1.288): its skills are installed, listed, changed, and removed in its own config directory.
    assert "/my home/brain/skills/haiku/SKILL.md" in told
    assert told.rindex("\n\n# ") == told.index("\n\n# Above all")
    # A home with a space in it is one argument to every command the brain is shown.
    assert f"'/my home/audit'/{SEGMENT_GLOB}" in told
    # The brain recalls with the command line it is handed, as it is handed it.
    assert "The recall command is: '/my python' -P -m hands.daemon recall\n" in told


def test_a_personality_the_user_chose_is_told_last_before_the_closing_words() -> None:
    own, chosen = brain_instruction(Path("/a"), Path("/b"), "hands recall", None), brain_instruction(Path("/a"), Path("/b"), "hands recall", "Dry and wry.")
    # In hands' own personality there is no section for one; a chosen one is its own section, and the rest is as it was.
    assert "# How you come across" not in own
    before, _, after = own.rpartition("\n\n# Above all")
    assert chosen.startswith(before + "\n\n# How you come across\n") and chosen.endswith("\n\n# Above all" + after)
    assert "\n\nDry and wry.\n\n" in chosen


@pytest.mark.skipif(shutil.which("jq") is None, reason="the brain's commands read the log with jq")
def test_the_commands_the_brain_is_shown_find_in_a_log_hands_wrote_what_they_say_they_find(tmp_path: Path) -> None:
    path = tmp_path / "a home" / "audit"
    log = AuditLog(path, clock=lambda: datetime(2026, 10, 3, tzinfo=UTC))
    log.record(Transcribed("send it"))
    # A line cut short, as a write that failed part way leaves it: the lines after it are still read.
    with segment(path, 0).open("a", encoding="utf-8") as torn:
        torn.write('{"at": "2026-10-03T00:00:00.000+00:00", "level": "err\n')
    log.record(Exchanged("x", SessionId("s1"), MainTurn(None), "POST", "/v1/messages", 2, (), 0.0, 0.0, Unreached("no route", 0.0), True, root()))
    with unit("tool.run", log.record):
        annotate(tool="list_sessions")
    with unit("summary.backlog", log.record):
        fail("lit exited 3")
    # A file that is no segment is no part of the log.
    (path / "audit.jsonl").write_text(json.dumps({"level": "error", "type": "Stray"}) + "\n")
    shown = [line[2:].partition(": ") for line in brain_instruction(path, tmp_path / "brain", "hands recall", None).splitlines() if line.startswith("- ")]
    commands = {label: command for label, _, command in shown if " | jq " in command}
    assert len(commands) == 4

    def found(label: str) -> list[str]:
        ran = subprocess.run(commands[label], shell=True, capture_output=True, text=True, check=True)
        return [line["event"] if line["type"] == "WideEvent" else line["type"] for line in map(json.loads, ran.stdout.splitlines())]

    assert found("the latest errors") == ["Exchanged", "summary.backlog"]
    assert found("what happened lately") == ["Transcribed", "tool.run", "summary.backlog"]
    assert found("one kind of line") == ["Transcribed"]
    assert found("the tools called lately") == ["tool.run"]
    # Once the log has rolled, what the closed segment holds is found too, before what came after.
    with unit("tool.run", AuditLog(path, clock=lambda: datetime(2026, 10, 3, tzinfo=UTC), segment_bytes=1).record):
        annotate(tool="list_sessions")
    assert found("the tools called lately") == ["tool.run", "tool.run"]
    assert found("the latest errors") == ["Exchanged", "summary.backlog"]
