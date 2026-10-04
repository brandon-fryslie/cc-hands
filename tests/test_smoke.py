"""`hands smoke`: each stage read off the record the pipeline writes as it reaches it, the session started as a terminal
outside any session starts one, and a run that stops short naming the stage on its command's event.

The whole run, against a running hands and a working session, is what the command is for: it is run by
hand on an installed Mac, not here."""

import asyncio
import json
import os
from aiortc import RTCPeerConnection
from aiortc.exceptions import InvalidStateError
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hands.core.effects import Summarise, Text, Type
from hands.core.session import PromptId, PromptText, SessionId
from hands.core.trace import Span
from hands.daemon.cli import main
from hands.daemon.smoke import FOLDER, QUIET_SECS, SESSION_GIVEN, WORDS, Caller, Ear, Line, NotReached, as_from_a_terminal, joined, parsed, proof
from hands.sessions import heartbeat
from hands.sessions.audit import HoldHeard, Levels, Typing, TypingFailed, Unsaid, encoded, segment
from hands.sessions.home import Home
from hands.voice import transcription
from hands.sessions.wide import Fact, Outcome, WideEvent

SESSION = SessionId("5086f176-e3cf-4222-bdb9-d7993c507d33")
LEVELS = Levels(captured_dbfs=-12.5, heard_dbfs=-40.0)
OTHER = SessionId("33a1f45e-fc95-41ca-98d0-a9310ab422fe")
SPAN = "e5fe41d042769d03"


def as_logged(record: object) -> Line:
    """`record` as the audit log holds it: the daemon's own encoding, so a field renamed there is renamed here."""
    return encoded(record)


def typing(session: SessionId, span: str = SPAN) -> Line:
    effect = Type(session, Path("/tmp/f/session.sock"), 7, Text(PromptText("Reply with the file.")))
    return as_logged(Typing(effect, Span("t", span, None)))


def event(name: str, span: str = "s", outcome: Outcome = "ok", error: str | None = None, **facts: Fact) -> Line:
    return as_logged(WideEvent(name, "t", span, None, datetime.now(UTC), 1.0, outcome, error, (), {}, facts))


def ended(span: str = SPAN) -> Line:
    return event("tool.run", span)


def hook(name: str, session: SessionId) -> Line:
    return event("hook", hook=name, session=session)


def stop(session: SessionId) -> Line:
    return hook("Stop", session)


def test_a_hold_is_heard_once_whisper_took_words_from_it() -> None:
    assert proof("heard", [], SESSION) is None
    assert proof("heard", [as_logged(HoldHeard(2, "tell the smoke session", (), LEVELS))], SESSION) == "heard 'tell the smoke session'"


def test_a_hold_whisper_took_no_words_from_is_never_heard_and_says_so() -> None:
    nothing = as_logged(HoldHeard(1, None, (Unsaid("Thank you.", 2.9, -1.2),), LEVELS))
    with pytest.raises(NotReached) as raised:
        proof("heard", [nothing], SESSION)
    assert (raised.value.stage, raised.value.why) == ("heard", "Whisper took no words from the hold; it dropped 1 segment(s)")


def utterance(session: SessionId, fate: str, outcome: Outcome = "ok", error: str | None = None) -> Line:
    heard = Summarise(session, PromptId("8dc206d5-ceb8-4cf1-887e-e00c5c57a55c"), "falcon.txt")
    return event("utterance", outcome=outcome, error=error, session=session, heard=heard, fate=fate)


def test_the_session_s_turn_is_settled_once_its_utterance_is_held_or_played_to_its_end() -> None:
    assert proof("told", [stop(SESSION), utterance(OTHER, "played")], SESSION) is None
    assert proof("told", [utterance(SESSION, "noted")], SESSION) == f"hands' telling of session {SESSION}'s turn: noted"
    assert proof("told", [utterance(SESSION, "played")], SESSION) == f"hands' telling of session {SESSION}'s turn: played"


def test_a_telling_cut_off_is_never_settled_and_says_so() -> None:
    with pytest.raises(NotReached) as raised:
        proof("told", [utterance(SESSION, "cut")], SESSION)
    assert (raised.value.stage, raised.value.why) == ("told", f"hands' telling of session {SESSION}'s turn was cut off before the question was asked")


def test_a_turn_hands_could_not_read_is_never_told_and_says_why() -> None:
    failed = utterance(SESSION, "played", "failed", "the turn could not be read: OSError: gone")
    with pytest.raises(NotReached) as raised:
        proof("told", [failed], SESSION)
    assert (raised.value.stage, raised.value.why) == ("told", f"hands could not tell session {SESSION}'s turn: the turn could not be read: OSError: gone")


def test_a_session_that_asks_permission_never_finishes_and_says_so() -> None:
    assert proof("finished", [hook("PermissionRequest", OTHER)], SESSION) is None
    with pytest.raises(NotReached) as raised:
        proof("finished", [hook("PermissionRequest", SESSION)], SESSION)
    assert raised.value.stage == "finished" and raised.value.why.startswith(f"session {SESSION} asked permission to use a tool")


def test_a_send_is_typed_only_once_the_unit_that_typed_it_has_ended() -> None:
    # The Typing line is written before the typing: alone, it says the send is under way, not that it landed.
    assert proof("typed", [typing(SESSION)], SESSION) is None
    assert proof("typed", [typing(SESSION), ended("another")], SESSION) is None
    assert proof("typed", [typing(SESSION), ended()], SESSION) == f"typed 'Reply with the file.' into session {SESSION}"


def test_a_send_that_failed_is_never_typed_and_says_why() -> None:
    effect = Type(SESSION, Path("/tmp/f/session.sock"), 7, Text(PromptText("Reply with the file.")))
    failed = as_logged(TypingFailed(effect, "this socket types into process 45443"))
    with pytest.raises(NotReached) as raised:
        proof("typed", [typing(SESSION), failed, ended()], SESSION)
    assert (raised.value.stage, raised.value.why) == ("typed", f"hands could not type into session {SESSION}: this socket types into process 45443")


def test_only_the_smoke_session_counts_for_its_send_and_its_finish() -> None:
    lines = [typing(OTHER), ended(), stop(OTHER)]
    assert proof("typed", lines, SESSION) is None
    assert proof("finished", lines, SESSION) is None
    assert proof("finished", [*lines, stop(SESSION)], SESSION) == f"session {SESSION} finished its turn"


def test_a_torn_line_is_no_evidence_and_breaks_none_of_the_rest() -> None:
    assert parsed(['{"type": "HoldHe', json.dumps({"type": "HoldHeard", "said": "hi"}), "[1]"]) == [{"type": "HoldHeard", "said": "hi"}]


def test_the_session_is_started_as_from_a_terminal_outside_any_session(tmp_path: Path) -> None:
    inside = {
        **{name: "given" for name in SESSION_GIVEN},
        # fritter's tap of the session the test was run in, and the proxy it replaced.
        "FRITTER_TAP": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9", "FRITTER_OUTER_HTTPS_PROXY": "http://proxy.lan:3128",
        # The user's own: kept.
        "CLAUDE_CODE_USE_BEDROCK": "1", "PATH": "/Users/me/.hands/bin:/usr/bin", "HANDS_HOME": "/elsewhere",
    }
    started = as_from_a_terminal(inside, Home(tmp_path))
    assert started == {"HTTPS_PROXY": "http://proxy.lan:3128", "CLAUDE_CODE_USE_BEDROCK": "1", "PATH": "/Users/me/.hands/bin:/usr/bin", "HANDS_HOME": str(tmp_path)}


def frame(level: int) -> bytes:
    return np.full(320, level, dtype=np.int16).tobytes()


def test_hands_has_finished_speaking_once_it_spoke_after_the_release_and_went_quiet() -> None:
    ear = Ear()
    ear.heard(frame(3000), 1.0)
    # Spoke, but before the release: nothing said back yet.
    assert ear.finished_speaking(2.0, 10.0) is None
    ear.heard(frame(0), 2.5)
    ear.heard(frame(3000), 3.0)
    assert ear.finished_speaking(2.0, 3.0 + QUIET_SECS / 2) is None
    assert ear.finished_speaking(2.0, 3.0 + QUIET_SECS) == 3.0
    assert ear.since(2.0) == frame(0) + frame(3000)


class Dropped:
    """A call's channel once the call has dropped, which refuses what is sent on it."""

    def send(self, data: bytes | str) -> None:
        raise InvalidStateError("RTCDataChannel is not open")


async def test_a_call_that_drops_stops_the_stage_being_said_and_says_why() -> None:
    peer = RTCPeerConnection()
    caller = Caller(peer, Dropped())  # pyright: ignore[reportArgumentType]  (the channel's send is all the caller uses)
    caller.tasks.append(asyncio.create_task(caller.keep_sending(), name="the smoke test's voice"))
    try:
        with pytest.raises(NotReached) as raised:
            await asyncio.wait_for(caller.say("typed", frame(3000)), 5)
        assert (raised.value.stage, raised.value.why) == ("typed", "the smoke test's voice stopped: InvalidStateError('RTCDataChannel is not open')")
    finally:
        await peer.close()


def test_the_smoke_session_is_the_one_that_joined_from_its_folder_since_the_test_began(tmp_path: Path) -> None:
    home = Home(tmp_path / "home")
    folder = tmp_path / "smoke"
    home.memberships.mkdir(parents=True)

    def member(session: SessionId, cwd: Path) -> None:
        record = {"pid": 7, "cwd": str(cwd), "transcript_path": f"/t/{session}.jsonl", "fritter_socket": "/tmp/f/session.sock"}
        home.membership(session).write_text(json.dumps(record))

    member(OTHER, folder)
    before = frozenset({OTHER})
    assert joined(home, folder, before) is None
    member(SessionId("0844f3f7-d8fe-5134-a45d-fa50e00dc5ec"), tmp_path / "elsewhere")
    home.membership(SessionId("0998fa11-3723-4c9a-bb7a-b454ae05dfca")).write_text('{"pid": ')
    assert joined(home, folder, before) is None
    member(SESSION, folder)
    found = joined(home, folder, before)
    assert found is not None and (found.id, found.fritter) == (SESSION, Path("/tmp/f/session.sock"))


def events(home: Home) -> list[dict[str, Any]]:
    return [line for line in map(json.loads, segment(home.audit, 0).read_text().splitlines()) if line["type"] == "WideEvent"]


def test_with_hands_not_running_the_run_stops_at_up_and_its_event_says_so(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path / "home")
    assert main(["--home", str(home.root), "smoke"]) == 1
    assert capsys.readouterr().out == f"FAILED up: hands has not run: there is no heartbeat at {home.status}; start it with `hands run`\n"
    [command] = events(home)
    assert (command["event"], command["outcome"], command["facts"]["command"], command["facts"]["failed_at"]) == ("hands.command", "failed", "smoke", "up")
    assert command["error"] == "exited 1"
    assert "reached" not in command["facts"] and "model" not in command["facts"]
    assert command["facts"]["why"] == f"hands has not run: there is no heartbeat at {home.status}; start it with `hands run`"


def test_a_run_that_stops_after_up_says_on_its_event_what_it_reached_and_why_it_stopped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    home = Home(tmp_path / "home")
    now = datetime.now(UTC)
    home.root.mkdir()
    heartbeat.write(home.status, heartbeat.Status(os.getpid(), now, now, heartbeat.HEARTBEAT, "running", None, 0, False, False))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert main(["--home", str(home.root), "smoke"]) == 1
    assert capsys.readouterr().out.splitlines()[-1] == "FAILED joined: there is no `claude` on PATH"
    [command] = events(home)
    facts = command["facts"]
    assert facts["word"] in WORDS and facts["folder"] == str((home.root / FOLDER).resolve())
    assert facts["model"] == transcription.MODEL
    assert (facts["reached"], facts["failed_at"], facts["why"], facts["daemon_errors"]) == ("up", "joined", "there is no `claude` on PATH", [])
    assert isinstance(facts["up_ms"], int) and facts["up"].startswith(f"hands is up: pid {os.getpid()}") and "joined_ms" not in facts
