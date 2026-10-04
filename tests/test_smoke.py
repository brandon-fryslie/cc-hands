"""`hands smoke`: each stage read off the record the pipeline writes as it reaches it, the session started as a terminal
outside any session starts one, and a run that stops short naming the stage on its command's event.

The whole run, against a running hands, a LowTalker, and a working session, is what the command is for: it is run by
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

from hands.core.session import SessionId
from hands.daemon.cli import main
from hands.daemon.smoke import FOLDER, QUIET_SECS, SESSION_GIVEN, WORDS, Caller, Ear, Line, NotReached, as_from_a_terminal, joined, parsed, proof
from hands.sessions import heartbeat
from hands.sessions.audit import segment
from hands.sessions.home import Home

SESSION = SessionId("5086f176-e3cf-4222-bdb9-d7993c507d33")
OTHER = SessionId("33a1f45e-fc95-41ca-98d0-a9310ab422fe")
SPAN = "e5fe41d042769d03"


def typing(session: SessionId, span: str = SPAN) -> Line:
    return {
        "type": "Typing",
        "effect": {"type": "Type", "session": session, "socket": "/tmp/f/session.sock", "pid": 7, "input": {"type": "Text", "prompt": "Reply with the file."}},
        "span": {"type": "Span", "trace_id": "t", "span_id": span, "parent_id": None},
    }


def ended(span: str = SPAN) -> Line:
    return {"type": "WideEvent", "event": "tool.run", "span_id": span, "outcome": "ok"}


def stop(session: SessionId) -> Line:
    return {"type": "WideEvent", "event": "hook", "span_id": "s", "facts": {"hook": "Stop", "session": session}}


def test_a_hold_is_heard_once_whisper_took_words_from_it() -> None:
    words: Line = {"type": "HoldHeard", "hold": 2, "said": "tell the smoke session", "dropped": []}
    assert proof("heard", [], SESSION) is None
    assert proof("heard", [words], SESSION) == "heard 'tell the smoke session'"


def test_a_hold_whisper_took_no_words_from_is_never_heard_and_says_so() -> None:
    nothing: Line = {"type": "HoldHeard", "hold": 1, "said": None, "dropped": [{"type": "Unsaid", "text": "Thank you."}]}
    with pytest.raises(NotReached) as raised:
        proof("heard", [nothing], SESSION)
    assert (raised.value.stage, raised.value.why) == ("heard", "Whisper took no words from the hold; it dropped 1 segment(s)")


def utterance(session: SessionId, delivered: Line, outcome: str = "ok", error: str | None = None) -> Line:
    heard = {"type": "Summarise", "session": session, "turn": "8dc206d5-ceb8-4cf1-887e-e00c5c57a55c", "closing": "falcon.txt"}
    return {"type": "WideEvent", "event": "utterance", "outcome": outcome, "error": error, "facts": {"session": session, "heard": heard, "delivered": delivered}}


def test_the_session_s_turn_told_unasked_is_told_once_the_reply_after_its_refocus_was_said() -> None:
    def refocused(session: SessionId) -> Line:
        return {"type": "Refocused", "session": session, "outcome": "moved", "failed": None}

    replied: Line = {"type": "Replied", "text": "It said falcon.", "interrupted": False}
    spoken = utterance(SESSION, {"type": "Spoken", "amount": "full", "why": "finished"})
    # Said before the telling was taken, or the telling of another session: not this telling.
    assert proof("told", [replied, stop(SESSION), spoken, refocused(OTHER), replied], SESSION) is None
    # Refocused is written as the telling is taken, before it is said.
    assert proof("told", [spoken, refocused(SESSION)], SESSION) is None
    assert proof("told", [spoken, refocused(SESSION), replied], SESSION) == f"hands told the user of session {SESSION}'s turn: 'It said falcon.'"


def test_the_session_s_turn_held_until_asked_needs_no_telling_waited_for() -> None:
    held = {"type": "Withheld", "why": "off"}
    assert proof("told", [utterance(OTHER, held)], SESSION) is None
    assert proof("told", [utterance(SESSION, held)], SESSION) == f"hands holds session {SESSION}'s turn until asked (off)"


def test_a_turn_hands_could_not_read_is_never_told_and_says_why() -> None:
    failed = utterance(SESSION, {"type": "Spoken", "amount": "full", "why": "finished"}, "failed", "the turn could not be read: OSError: gone")
    with pytest.raises(NotReached) as raised:
        proof("told", [failed], SESSION)
    assert (raised.value.stage, raised.value.why) == ("told", f"hands could not tell session {SESSION}'s turn: the turn could not be read: OSError: gone")


def test_a_send_is_typed_only_once_the_unit_that_typed_it_has_ended() -> None:
    # The Typing line is written before the typing: alone, it says the send is under way, not that it landed.
    assert proof("typed", [typing(SESSION)], SESSION) is None
    assert proof("typed", [typing(SESSION), ended("another")], SESSION) is None
    assert proof("typed", [typing(SESSION), ended()], SESSION) == f"typed 'Reply with the file.' into session {SESSION}"


def test_a_send_that_failed_is_never_typed_and_says_why() -> None:
    failed: Line = {"type": "TypingFailed", "effect": typing(SESSION)["effect"], "reason": "this socket types into process 45443"}
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
    assert "reached" not in command["facts"]
    assert command["facts"]["transcription"] == "http://127.0.0.1:8610/v1"
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
    assert (facts["reached"], facts["failed_at"], facts["why"], facts["daemon_errors"]) == ("up", "joined", "there is no `claude` on PATH", [])
    assert isinstance(facts["up_ms"], int) and "joined_ms" not in facts
