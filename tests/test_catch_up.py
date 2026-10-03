"""catch_up reads what the user missed out of the audit log: every session that finished while they were away."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from hands.core.sentences import cut
from hands.core.events import Closed, Died, Ended, Joined, Stopped
from hands.core.session import Membership, PromptId, RequestId, SessionId
from hands.core.status import Stamp
from hands.sessions.audit import Announced, Applied, AuditLog, Entry, Replied, Transcribed
from hands.sessions.home import Home
from hands.sessions.registry import Sessions
from hands.voice.tools import CATCH_UP_CLOSINGS, CATCH_UP_LEAST, catch_up_tool

LEFT = datetime(2026, 10, 3, 14, 0, tzinfo=UTC)
BACK = LEFT + timedelta(minutes=10)


def member(name: str) -> Membership:
    return Membership(SessionId(name), pid=len(name), cwd=Path("/code") / name, transcript=Path(f"/nowhere/{name}.jsonl"))


def stopped(session: str, prompt: str, closing: str | None, again: bool = False) -> Applied:
    return Applied(Stopped(SessionId(session), closing, "default", PromptId(prompt), again, Stamp(1), RequestId(f"{session}-{prompt}")))


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


async def test_what_did_i_miss_after_ten_minutes_lists_every_session_that_finished(tmp_path: Path) -> None:
    """The window opens at the user's words before the ones being answered: what came before it they heard."""
    home = Home(tmp_path)
    written(
        home,
        [
            (LEFT - timedelta(minutes=5), stopped("docs", "d0", "Heard before they left.")),
            (LEFT, Transcribed("I'm getting coffee")),
            (LEFT, Replied("Enjoy it.", interrupted=False)),
            (LEFT + timedelta(minutes=1), stopped("docs", "d1", "Fixed the links.")),
            # The reply heard on the wire, then the Stop of the same turn, then a Stop a hook blocked: one turn.
            (LEFT + timedelta(minutes=2), Applied(Closed(SessionId("parser"), PromptId("p1"), "Parser draft."))),
            (LEFT + timedelta(minutes=2), stopped("parser", "p1", "Parser draft.")),
            (LEFT + timedelta(minutes=3), stopped("parser", "p1", "Parser fixed, tests pass.", again=True)),
            (LEFT + timedelta(minutes=4), stopped("docs", "d2", None)),
            (LEFT + timedelta(minutes=5), Announced("hands could not reach the model.", "speech")),
            (LEFT + timedelta(minutes=6), stopped("deploy", "x1", "Deployed.")),
            (LEFT + timedelta(minutes=7), Applied(Ended(SessionId("deploy"), "prompt_input_exit"))),
            (LEFT + timedelta(minutes=8), Applied(Died(member("scratch")))),
            (BACK, Transcribed("what did I miss")),
        ],
    )

    assert await catch_up(await sessions_of("docs", "parser", "deploy", "scratch"), home) == {
        "since_minutes_ago": 10,
        "finished": [
            {"session": "docs", "turns": 2, "closing": "Fixed the links."},
            {"session": "parser", "turns": 1, "closing": "Parser fixed, tests pass."},
            {"session": "deploy", "turns": 1, "closing": "Deployed."},
        ],
        "ended": ["deploy", "scratch"],
        "announced": ["hands could not reach the model."],
    }


async def test_minutes_reaches_back_past_what_the_user_said(tmp_path: Path) -> None:
    home = Home(tmp_path)
    written(
        home,
        [
            (BACK - timedelta(minutes=90), stopped("docs", "d0", "Too long ago.")),
            (BACK - timedelta(minutes=50), stopped("docs", "d1", "Within the hour.")),
            (BACK - timedelta(minutes=5), Transcribed("anything new?")),
            (BACK, Transcribed("what happened in the last hour")),
        ],
    )

    assert await catch_up(await sessions_of("docs"), home, minutes=60) == {
        "since_minutes_ago": 60,
        "finished": [{"session": "docs", "turns": 1, "closing": "Within the hour."}],
        "ended": [],
        "announced": [],
    }


async def test_with_no_earlier_words_it_reads_the_whole_log_and_names_sessions_hands_never_listed_by_id(tmp_path: Path) -> None:
    home = Home(tmp_path)
    long = "word " * 4000
    written(home, [(LEFT, stopped("gone", "g1", long)), (BACK, Transcribed("what did I miss"))])

    assert await catch_up(await sessions_of(), home) == {
        "since_minutes_ago": None,
        "finished": [{"session": "gone", "turns": 1, "closing": cut(long, CATCH_UP_CLOSINGS)}],
        "ended": [],
        "announced": [],
    }


async def test_a_log_not_yet_written_is_nothing_missed(tmp_path: Path) -> None:
    assert await catch_up(await sessions_of(), Home(tmp_path)) == {"since_minutes_ago": None, "finished": [], "ended": [], "announced": []}



async def test_a_hundred_sessions_finishing_are_each_named_and_share_the_words(tmp_path: Path) -> None:
    home = Home(tmp_path)
    long = "word " * 400
    names = [f"s{n}" for n in range(100)]
    written(home, [(LEFT, Transcribed("back soon")), *((LEFT, stopped(name, "p", long)) for name in names), (BACK, Transcribed("what did I miss"))])

    result = await catch_up(await sessions_of(*names), home)

    assert [(done["session"], done["closing"]) for done in result["finished"]] == [(name, cut(long, max(CATCH_UP_LEAST, CATCH_UP_CLOSINGS // 100))) for name in names]  # type: ignore[index]
