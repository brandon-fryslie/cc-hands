"""catch_up reads what the user missed out of the audit log: every session that finished while they were away."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from hands.core.sentences import cut
from hands.core.effects import Effect, SessionGone, Summarise, Tell
from hands.core.occurrences import AutoDenied, Cleared, SubagentStopped
from hands.core.events import Joined, Tick
from hands.core.session import Membership, PromptId, SessionId
from hands.sessions.audit import Announced, AuditLog, Entry, Replied, Transcribed
from hands.sessions.home import Home
from hands.sessions.registry import Performed, Sessions
from hands.sessions.wide import WideEvent
from hands.voice.tools import Called, CATCH_UP_CLOSINGS, CATCH_UP_LEAST, audited, catch_up_tool

LEFT = datetime(2026, 10, 3, 14, 0, tzinfo=UTC)
BACK = LEFT + timedelta(minutes=10)


def member(name: str) -> Membership:
    return Membership(SessionId(name), pid=len(name), cwd=Path("/code") / name, transcript=Path(f"/nowhere/{name}.jsonl"))


def applied(effect: Effect, failed: str | None = None) -> WideEvent:
    """The event of the registry applying what called for `effect`, which was performed, or failed with `failed`."""
    done = Performed(effect, "ok", 0.5, None) if failed is None else Performed(effect, "failed", 0.5, failed)
    return WideEvent("applied", "e" * 32, "f" * 16, None, LEFT, 1.0, "ok" if failed is None else "failed", failed, (), {}, {"applied": Tick(0.0), "effects": (done,)})


def told(session: str, prompt: str, closing: str | None) -> WideEvent:
    return applied(Summarise(SessionId(session), PromptId(prompt), closing))


def gone(session: str) -> WideEvent:
    return applied(SessionGone(SessionId(session)))


def written(home: Home, lines: list[tuple[datetime, Entry]]) -> None:
    clock = iter(at for at, _ in lines)
    log = AuditLog(home.audit, clock=lambda: next(clock))
    for _, entry in lines:
        log.record(entry)


async def sessions_of(*names: str) -> Sessions:
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _: None)
    for name in names:
        await sessions.apply(Joined(member(name), "startup"))
    return sessions


async def catch_up(sessions: Sessions, home: Home, minutes: int = 0) -> object:
    return await catch_up_tool(sessions, home, lambda: BACK).body(minutes=minutes)


NOTHING: dict[str, object] = {"since_minutes_ago": None, "unreadable_lines": 0, "finished": [], "ended": [], "occurred": [], "announced": []}


async def test_what_did_i_miss_after_ten_minutes_lists_every_session_that_finished(tmp_path: Path) -> None:
    """The window opens at the user's words before the ones being answered: what came before it they heard."""
    home = Home(tmp_path)
    written(
        home,
        [
            (LEFT - timedelta(minutes=5), told("docs", "d0", "Heard before they left.")),
            (LEFT, Transcribed("I'm getting coffee")),
            (LEFT, Replied("Enjoy it.", interrupted=False)),
            (LEFT + timedelta(minutes=1), told("docs", "d1", "Fixed the links.")),
            (LEFT + timedelta(minutes=2), told("parser", "p1", "Parser fixed, tests pass.")),
            (LEFT + timedelta(minutes=3), Announced("hands could not reach the model.", "speech")),
            # A turn whose telling could not be begun still finished.
            (LEFT + timedelta(minutes=4), applied(Summarise(SessionId("docs"), PromptId("d2"), None), "the model is down")),
            (LEFT + timedelta(minutes=5), Announced("hands could not reach the model.", "speech")),
            (LEFT + timedelta(minutes=6), told("deploy", "x1", "Deployed.")),
            (LEFT + timedelta(minutes=8), gone("scratch")),
            (BACK, Transcribed("what did I miss")),
        ],
    )
    called: list[object] = []
    tool = audited(catch_up_tool(await sessions_of("docs", "parser", "deploy", "scratch"), home, lambda: BACK), called.append)

    result = await tool.body(minutes=0)

    assert result == {
        "since_minutes_ago": 10,
        "unreadable_lines": 0,
        "finished": [
            {"session": "docs", "turns": 2, "closing": "Fixed the links."},
            {"session": "parser", "turns": 1, "closing": "Parser fixed, tests pass."},
            {"session": "deploy", "turns": 1, "closing": "Deployed."},
        ],
        "ended": ["scratch"],
        "occurred": [],
        "announced": [{"text": "hands could not reach the model.", "times": 2}],
    }
    # [LAW:nothing-unseen] the call's event carries where the window opened.
    [event] = called
    assert isinstance(event, WideEvent) and event.facts == {"tool": "catch_up", "called": Called({"minutes": 0}, result)}


async def test_what_a_hook_said_happened_is_caught_up_on_whether_or_not_it_was_said(tmp_path: Path) -> None:
    """A kind set off is told only when asked for: here, its long text and a call's input bounded as a closing's least share."""
    home = Home(tmp_path)
    long = "x" * (CATCH_UP_LEAST * 2)
    written(
        home,
        [
            (LEFT, Transcribed("back in a bit")),
            (LEFT + timedelta(minutes=1), applied(Tell(SessionId("docs"), AutoDenied("Bash", "[Data Exfiltration]", {"command": long})))),
            (LEFT + timedelta(minutes=2), applied(Tell(SessionId("docs"), SubagentStopped("Plan", "first")))),
            (LEFT + timedelta(minutes=2), applied(Tell(SessionId("docs"), SubagentStopped("Explore", long)))),
            (LEFT + timedelta(minutes=3), applied(Tell(SessionId("docs"), Cleared()))),
            (BACK, Transcribed("what did I miss")),
        ],
    )
    result = cast(dict[str, object], await catch_up(await sessions_of("docs"), home))
    assert result["occurred"] == [
        {"session": "docs", "times": 1, "type": "AutoDenied", "tool": "Bash", "reason": "[Data Exfiltration]", "input": cut(json.dumps({"command": long}), CATCH_UP_LEAST)},
        # Counted by kind, with the newest said.
        {"session": "docs", "times": 2, "type": "SubagentStopped", "agent_type": "Explore", "closing": cut(long, CATCH_UP_LEAST)},
        {"session": "docs", "times": 1, "type": "Cleared"},
    ]


async def test_minutes_reaches_back_past_what_the_user_said(tmp_path: Path) -> None:
    home = Home(tmp_path)
    written(
        home,
        [
            (BACK - timedelta(minutes=90), told("docs", "d0", "Too long ago.")),
            (BACK - timedelta(minutes=50), told("docs", "d1", "Within the hour.")),
            (BACK - timedelta(minutes=5), Transcribed("anything new?")),
            (BACK, Transcribed("what happened in the last hour")),
        ],
    )

    assert await catch_up(await sessions_of("docs"), home, minutes=60) == {
        **NOTHING,
        "since_minutes_ago": 60,
        "finished": [{"session": "docs", "turns": 1, "closing": "Within the hour."}],
    }


async def test_minutes_back_from_now_is_never_negative(tmp_path: Path) -> None:
    assert await catch_up(await sessions_of(), Home(tmp_path), minutes=-60) == {"error": "minutes is how far back to look, so it cannot be -60"}


async def test_a_line_a_failed_write_left_torn_is_counted_and_the_rest_read(tmp_path: Path) -> None:
    home = Home(tmp_path)
    written(home, [(LEFT, Transcribed("back soon"))])
    with (home.audit / "00000000000000000000.jsonl").open("ab") as log:
        # Cut inside the em dash's three bytes.
        log.write('{"at": "2026-10-03T14:00:00+00:00", "type": "Announced", "text": "Fixed \u2014'.encode()[:-1])
    written(home, [(LEFT, told("docs", "d1", "Done.")), (BACK, Transcribed("what did I miss"))])

    assert await catch_up(await sessions_of("docs"), home) == {
        **NOTHING,
        "since_minutes_ago": 10,
        "unreadable_lines": 1,
        "finished": [{"session": "docs", "turns": 1, "closing": "Done."}],
    }


async def test_with_no_earlier_words_it_reads_the_whole_log_and_names_sessions_hands_never_listed_by_id(tmp_path: Path) -> None:
    home = Home(tmp_path)
    long = "word " * 4000
    written(home, [(LEFT, told("gone", "g1", long)), (BACK, Transcribed("what did I miss"))])

    assert await catch_up(await sessions_of(), home) == {**NOTHING, "finished": [{"session": "gone", "turns": 1, "closing": cut(long, CATCH_UP_CLOSINGS)}]}


async def test_a_log_not_yet_written_is_nothing_missed(tmp_path: Path) -> None:
    assert await catch_up(await sessions_of(), Home(tmp_path)) == NOTHING


async def test_a_hundred_sessions_finishing_are_each_named_and_share_the_words(tmp_path: Path) -> None:
    home = Home(tmp_path)
    long = "word " * 400
    names = [f"s{n}" for n in range(100)]
    written(home, [(LEFT, Transcribed("back soon")), *((LEFT, told(name, "p", long)) for name in names), (BACK, Transcribed("what did I miss"))])

    result = await catch_up(await sessions_of(*names), home)

    assert [(done["session"], done["closing"]) for done in result["finished"]] == [(name, cut(long, max(CATCH_UP_LEAST, CATCH_UP_CLOSINGS // 100))) for name in names]  # type: ignore[index]
