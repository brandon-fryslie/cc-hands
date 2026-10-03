"""Progress while a session works: its calls read as they are made, gathered until they settle, and heard by the focus."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path

import pytest
from pipecat.frames.frames import Frame, TTSSpeakFrame

from hands.core.attention import Overlay, Route, progress_route
from hands.core.effects import Progress
from hands.core.events import Displayed, Progressed, StatusReported, Tick
from hands.core.pending import Finished, News, Pending, Unread, Working, coalesce
from hands.core.progress import EDITING, WRITING, LONGEST, RUNNING, SETTLE, Doing, Gathering, doing, explained, said
from hands.core.reducer import reduce
from hands.core.session import Gone, Membership, Opened, PromptId, Registry, Running, Session, SessionId, Told, Turn, Untold
from hands.core import status
from hands.core.status import Busy, Report, Stamp
from hands.sessions.audit import Applied, Entry, ProgressTold, Relayed, Routed, encoded
from hands.sessions.tail import Tails
from hands.voice.speech import Aloud, Pushed, Tailed, Unprompted, frames, relay
from hands.voice.summary import SummaryFailed
from hands.voice.working import Playing, keep_playing
from hands.voice.tools import describe_listing
from hands.sessions.registry import Listing

SID = SessionId("bf411065-dc5c-4ec9-8302-61b84bdb5c53")
OTHER = SessionId("other")
TURN = PromptId("p1")
IN_TURN = frozenset({TURN})
IN_NEXT = frozenset({PromptId("p2")})
MEMBER = Membership(SID, pid=4242, cwd=Path("/code/a"), transcript=Path("/code/a/t.jsonl"))
TESTS = Doing(RUNNING, "run the test suite")


def edit(name: str) -> Doing:
    return Doing(EDITING, f"edit {name}")


@pytest.mark.parametrize(
    ("tool", "input", "done"),
    [
        pytest.param("Bash", {"command": "uv run pytest", "description": "Run the test suite"}, "run the test suite", id="a command by what it is for"),
        pytest.param("Bash", {"command": "ls"}, "run a command", id="a command with no description, never its code"),
        pytest.param("Grep", {"pattern": r"def\s+_\w+\("}, "search the code", id="a regular expression is code, never read out"),
        pytest.param("Glob", {"pattern": "**/*.py"}, "search the code", id="a glob is code, never read out"),
        pytest.param("Bash", {"command": "gh pr view", "description": "PR status"}, "PR status", id="an initialism kept as written"),
        pytest.param("Edit", {"file_path": "/code/a/src/tail.py"}, "edit tail.py", id="an edit by the file's name"),
        pytest.param("Write", {"file_path": "/code/a/new.md", "content": "x"}, "edit new.md", id="a write is an edit"),
        pytest.param("Read", {"file_path": "/code/a/README.md"}, "read README.md", id="a read"),
        pytest.param("Grep", {"pattern": "Progressed"}, "search for Progressed", id="a search"),
        pytest.param("WebFetch", {"url": "https://docs.python.org/3/library/re.html"}, "read a page on docs.python.org", id="a fetch by its site"),
        pytest.param("Agent", {"description": "Find hook callers", "prompt": "..."}, "start a subagent to find hook callers", id="a subagent"),
        pytest.param("mcp__claude_ai_Firecrawl__firecrawl_search", {}, "use firecrawl search", id="an MCP tool by its own name"),
    ],
)
def test_a_call_is_said_by_what_it_sets_out_to_do(tool: str, input: Mapping[str, object], done: str) -> None:
    made = doing(tool, input)
    assert made is not None and said((made,)) == done


@pytest.mark.parametrize("tool", ["AskUserQuestion", "ExitPlanMode"])
def test_a_call_that_asks_the_user_is_no_progress(tool: str) -> None:
    # Its PermissionRequest hook speaks it as it is asked.
    assert doing(tool, {}) is None


@pytest.mark.parametrize(
    ("doings", "sentence"),
    [
        pytest.param((TESTS,), "run the test suite", id="one call"),
        pytest.param(tuple(edit(f"f{n}.py") for n in range(10)), "edit ten files", id="ten edits are one clause"),
        pytest.param((edit("tail.py"),) * 10, "edit tail.py", id="ten edits of one file are that file"),
        pytest.param((Doing(RUNNING, None),) * 8, "run eight commands", id="calls that say nothing of themselves are each their own"),
        pytest.param((edit("a.py"), TESTS, edit("b.py"), TESTS), "edit two files, then run the test suite", id="each kind once, in the order first done"),
    ],
)
def test_a_burst_is_one_clause(doings: tuple[Doing, ...], sentence: str) -> None:
    assert said(doings) == sentence


def test_a_burst_is_due_once_it_settles_and_no_later_than_its_longest_wait() -> None:
    assert Gathering((TESTS,), "", 10.0, 11.0).due() == 11.0 + SETTLE
    assert Gathering((TESTS,), "", 10.0, 10.0 + LONGEST).due() == 10.0 + LONGEST


@pytest.mark.parametrize(
    ("focused", "overlay", "route"),
    [
        (True, "normal", "play"),
        (True, "watched", "play"),
        (True, "muted", "note"),
        (False, "normal", "note"),
        (False, "watched", "note"),
        (False, "muted", "note"),
    ],
)
def test_only_the_focus_is_heard_working_and_never_a_muted_one(focused: bool, overlay: Overlay, route: Route) -> None:
    assert progress_route(focused, overlay) == route


def finished(session: SessionId) -> Finished:
    return Finished(session, (News(TURN, "Done.", "", "", ()),))


@pytest.mark.parametrize(
    ("pending", "told"),
    [
        pytest.param(
            (Working(SID, IN_TURN, (edit("a.py"),)), Working(OTHER, IN_TURN, (TESTS,)), Working(SID, IN_TURN, (edit("b.py"),))),
            (Working(SID, IN_TURN, (edit("a.py"), edit("b.py"))), Working(OTHER, IN_TURN, (TESTS,))),
            id="a session's progress folds into one telling where its first stood",
        ),
        pytest.param(
            (Working(SID, IN_TURN, (TESTS,)), finished(SID), Working(OTHER, IN_TURN, (TESTS,))),
            (finished(SID), Working(OTHER, IN_TURN, (TESTS,))),
            id="progress the turn's result follows is said by the result",
        ),
        pytest.param(
            (finished(SID), Working(SID, IN_NEXT, (TESTS,))),
            (finished(SID), Working(SID, IN_NEXT, (TESTS,))),
            id="progress of the turn after a result is still heard",
        ),
        pytest.param(
            (Working(SID, IN_NEXT, (TESTS,)), finished(SID)),
            (Working(SID, IN_NEXT, (TESTS,)), finished(SID)),
            id="progress of the next turn is heard though the turn before's result is told after it",
        ),
        pytest.param(
            (Working(SID, IN_NEXT, (TESTS,)), Unread(SID)),
            (Unread(SID),),
            id="a turn that could not be read goes by no id, so the progress it follows is said by it",
        ),
    ],
)
def test_progress_folds_and_gives_way_to_the_result(pending: tuple[Pending, ...], told: tuple[Pending, ...]) -> None:
    assert coalesce(pending, {}) == told


def test_progress_heard_is_said_as_written_in_the_lane_its_telling_keeps() -> None:
    [pushed] = frames(Working(SID, IN_TURN, (TESTS,)), Pushed(), names=lambda _: "cc-hands")
    # Never in a pushed context, which keeps every message: the listing says what a session last set out to do.
    assert isinstance(pushed, TTSSpeakFrame) and pushed.text == "cc-hands: run the test suite." and not pushed.append_to_context
    [tailed] = frames(Working(SID, IN_TURN, (TESTS,)), Tailed(), names=lambda _: "cc-hands")
    assert isinstance(tailed, Aloud) and tailed.spoken.text == "cc-hands: run the test suite."


def running(turn: Turn) -> Registry:
    session = Session(MEMBER, Running(Busy(), Stamp(1000), None), mode=None, turn=turn)
    return Registry(permission_deadline=60.0, sessions={SID: session}, drafts={})


def turn_of(registry: Registry) -> Turn:
    session = registry.sessions[SID]
    assert isinstance(session, Session)
    return session.turn


def test_calls_are_gathered_on_the_turn_and_told_once_they_settle() -> None:
    gathered, effects = reduce(running(Opened(TURN)), Progressed(SID, (TURN,), (edit("a.py"),), at=10.0))
    gathered, more = reduce(gathered, Progressed(SID, (TURN,), (TESTS,), at=11.0))
    assert effects == more == []
    assert turn_of(gathered) == Opened(TURN, gathering=Gathering((edit("a.py"), TESTS), "", 10.0, 11.0), latest=TESTS)
    waiting, early = reduce(gathered, Tick(11.0 + SETTLE - 0.5))
    assert early == [] and waiting == gathered
    told, effects = reduce(gathered, Tick(11.0 + SETTLE))
    assert effects == [Progress(SID, frozenset({TURN}), (edit("a.py"), TESTS), "")]
    # Told once: what the session last set out to do is kept for anyone who asks.
    assert turn_of(told) == Opened(TURN, latest=TESTS)
    assert reduce(told, Tick(30.0)) == (told, [])


def test_a_session_that_never_pauses_is_heard_at_its_longest_wait() -> None:
    registry = running(Opened(TURN))
    for at in range(0, int(LONGEST) + 1):
        registry, _ = reduce(registry, Progressed(SID, (TURN,), (edit(f"f{at}.py"),), at=float(at)))
    _, effects = reduce(registry, Tick(LONGEST))
    assert [type(effect) for effect in effects] == [Progress]


@pytest.mark.parametrize(
    "turn",
    [
        pytest.param(Opened(PromptId("p2")), id="a turn by another id"),
        pytest.param(Untold(TURN, frozenset(), Stamp(2000)), id="a turn since over, whose result is told instead"),
        pytest.param(Told(TURN), id="a turn told"),
    ],
)
def test_calls_read_and_text_displayed_late_move_nothing(turn: Turn) -> None:
    before = running(turn)
    assert reduce(before, Progressed(SID, (TURN,), (TESTS,), at=10.0)) == (before, [])
    # Claude Code goes on displaying a reply for seconds after its turn's Stop (2.1.280).
    assert reduce(before, Displayed(SID, (TURN,), "1. A line of it.\n", at=10.0)) == (before, [])


def test_what_a_turn_gathered_and_did_not_tell_goes_with_it_as_it_ends() -> None:
    gathered, _ = reduce(running(Opened(TURN)), Progressed(SID, (TURN,), (TESTS,), at=10.0))
    ended, _ = reduce(gathered, StatusReported(SID, Report(status.Idle(), Stamp(3000)), at=11.0))
    assert isinstance(turn_of(ended), Untold)
    assert reduce(ended, Tick(30.0))[1] == []


def lines(*records: str) -> str:
    return "".join(f"{record}\n" for record in records)


ASKED = '{"type":"user","promptId":"p1","message":{"role":"user","content":"Run the tests."}}'
CALL = '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"uv run pytest","description":"Run the test suite"}}]}}'
RESULT = '{"type":"user","promptId":"p1","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"3 passed"}]}}'
EDITS = '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t2","name":"Edit","input":{"file_path":"/a/x.py"}},{"type":"tool_use","id":"t3","name":"AskUserQuestion","input":{}}]}}'


class Known:
    """As much of the registry as the tail asks about."""

    def __init__(self, transcript: Path) -> None:
        self.member = Membership(SID, pid=4242, cwd=transcript.parent, transcript=transcript)

    def live_members(self) -> list[Membership]:
        return [self.member]

    def status_read(self, session: SessionId) -> bool:
        return True

    def now(self) -> float:
        return 7.0

    def stamp(self) -> Stamp:
        return Stamp(5000)

    def membership(self, session: SessionId) -> Membership | None:
        return self.member


async def test_each_call_a_running_turn_makes_is_read_once_as_progress(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED))
    tails = Tails(Known(transcript))
    await tails.catch_up()
    with transcript.open("a") as file:
        file.write(lines(CALL))
    assert [event for event in await tails.catch_up() if isinstance(event, Progressed)] == [Progressed(SID, (TURN,), (TESTS,), 7.0)]
    with transcript.open("a") as file:
        file.write(lines(RESULT, EDITS))
    # The result is the turn's to tell; a question is spoken as it is asked.
    assert [event for event in await tails.catch_up() if isinstance(event, Progressed)] == [Progressed(SID, (TURN,), (edit("x.py"),), 7.0)]
    assert [event for event in await tails.catch_up() if isinstance(event, Progressed)] == []


async def test_the_calls_a_turn_made_before_hands_followed_it_are_history(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, CALL))
    assert [event for event in await Tails(Known(transcript)).catch_up() if isinstance(event, Progressed)] == []


async def test_the_relay_hands_the_focus_on_to_be_played_leaves_any_other_to_the_listing_and_says_why() -> None:
    queued: list[Frame] = []
    played: list[Progress] = []
    recorded: list[Entry] = []

    class Heard:
        def __init__(self) -> None:
            self.waiting: asyncio.Queue[Progress] = asyncio.Queue()

        async def heard(self) -> Progress:
            return await self.waiting.get()

    async def queue_frame(frame: Frame) -> None:
        queued.append(frame)

    async def attending(session: SessionId) -> tuple[bool, Overlay]:
        return session == SID, "normal"

    sessions = Heard()
    for session in (SID, OTHER):
        sessions.waiting.put_nowait(Progress(session, frozenset({TURN}), (TESTS,), ""))
    relaying = asyncio.create_task(relay(sessions, queue_frame, recorded.append, attending, played.append))  # pyright: ignore[reportArgumentType]  (only heard() is asked)
    while len([entry for entry in recorded if isinstance(entry, Routed)]) < 2:
        await asyncio.sleep(0.01)
    relaying.cancel()
    assert queued == []
    assert played == [Progress(SID, frozenset({TURN}), (TESTS,), "")]
    assert [entry for entry in recorded if not isinstance(entry, Relayed)] == [Routed(SID, True, "normal", "play"), Routed(OTHER, False, "normal", "note")]


def test_what_a_session_last_set_out_to_do_is_in_its_listing() -> None:
    session = Session(MEMBER, Running(Busy(), Stamp(1000), None), mode=None, turn=Opened(TURN, latest=TESTS))
    assert describe_listing(Listing(session, "cc-hands"))["state"] == "working; the last thing it set out to do: run the test suite"
    assert describe_listing(Listing(Session(MEMBER, Running(Busy(), Stamp(1000), None), mode=None, turn=Opened(TURN)), "cc-hands"))["state"] == "working"


def test_calls_read_are_a_line_the_audit_log_can_write() -> None:
    # Found live: a set of prompt ids is no line, and the event that carried one went unrecorded.
    assert encoded(Applied(Progressed(SID, (TURN,), (TESTS,), at=7.0)))["event"] == {
        "type": "Progressed",
        "session": SID,
        "turn": ["p1"],
        "doings": [{"type": "Doing", "work": {"type": "Work", "several": "run {count} commands", "one": "run a command"}, "alone": "run the test suite"}],
        "at": 7.0,
    }


def test_progress_relayed_and_told_is_a_line_the_audit_log_can_write() -> None:
    # Found live: the log could not write the set progress carries its turn as, and every burst's Relayed and Performed
    # lines went unrecorded.
    heard = encoded(Relayed(Progress(SID, frozenset({PromptId("p2"), TURN}), (TESTS,), "a line\n")))["heard"]
    assert isinstance(heard, dict) and heard["turn"] == ["p1", "p2"] and heard["written"] == "a line\n"
    assert encoded(ProgressTold(SID, 7, "explain how DNS works", None, current=True))["explained"] == "explain how DNS works"


def test_calls_read_after_their_session_ended_are_behind_not_wrong() -> None:
    # The tail reads a transcript a moment past its session's end: no audit line says a hook came late.
    gone = Registry(permission_deadline=60.0, sessions={SID: Gone(MEMBER)}, drafts={})
    assert reduce(gone, Progressed(SID, (TURN,), (TESTS,), at=10.0)) == (gone, [])
    assert reduce(gone, Displayed(SID, (TURN,), "a line\n", at=10.0)) == (gone, [])


def test_text_is_gathered_with_the_calls_and_told_with_them_once_both_settle() -> None:
    registry, _ = reduce(running(Opened(TURN)), Displayed(SID, (TURN,), "First, how DNS works.\n", at=10.0))
    registry, _ = reduce(registry, Displayed(SID, (TURN,), "1. The OS asks its resolver.\n2. The resolver asks the root.\n", at=11.0))
    registry, _ = reduce(registry, Progressed(SID, (TURN,), (TESTS,), at=12.0))
    # A line displayed holds the burst open as a call does: a long explanation is told once it has settled or waited longest.
    assert reduce(registry, Tick(11.0 + SETTLE))[1] == []
    told, effects = reduce(registry, Tick(12.0 + SETTLE))
    assert effects == [Progress(SID, frozenset({TURN}), (TESTS,), "First, how DNS works.\n1. The OS asks its resolver.\n2. The resolver asks the root.\n")]
    # Text is not a call: what the session last set out to do is still the call it made.
    assert turn_of(told) == Opened(TURN, latest=TESTS)


def test_an_explanation_still_being_written_is_told_at_its_longest_wait() -> None:
    registry = running(Opened(TURN))
    for at in range(0, int(LONGEST) * 4):
        registry, _ = reduce(registry, Displayed(SID, (TURN,), f"{at}. A line.\n", at=at / 4))
    _, effects = reduce(registry, Tick(LONGEST))
    assert [(type(effect), effect.doings) for effect in effects if isinstance(effect, Progress)] == [(Progress, ())]


def test_text_is_said_by_its_summary_ahead_of_the_calls() -> None:
    assert said((explained("Explain how DNS resolution works."), TESTS)) == "explain how DNS resolution works, then run the test suite"
    assert said((Doing(WRITING, None), TESTS)) == "write something, then run the test suite"
    # Two bursts whose text could not be summarised, folded before they are said: what either says is not known.
    assert said((Doing(WRITING, None), Doing(WRITING, None))) == "write two things"


async def played(progress: Progress, turn: Callable[[], Turn], explain: Callable[[str], Awaitable[str]]) -> tuple[list[Pending], list[ProgressTold]]:
    """What the player hands the floor of one progress, and its line in the audit log, with the session's turn as `turn`
    says when it is asked."""
    queued: list[Frame] = []
    recorded: list[Entry] = []

    async def queue_frame(frame: Frame) -> None:
        queued.append(frame)

    def live_session(_session: SessionId) -> Session:
        return Session(MEMBER, Running(Busy(), Stamp(1000), None), mode=None, turn=turn())

    playing = Playing()
    playing.put_nowait(progress)
    player = asyncio.create_task(keep_playing(playing, live_session, queue_frame, recorded.append, explain))
    while not any(isinstance(entry, ProgressTold) for entry in recorded):
        await asyncio.sleep(0.01)
    player.cancel()
    return [frame.pending for frame in queued if isinstance(frame, Unprompted)], [entry for entry in recorded if isinstance(entry, ProgressTold)]


async def unasked(text: str) -> str:
    raise AssertionError(f"a burst of calls alone is said as written, not summarised: {text!r}")


async def test_a_burst_of_calls_alone_is_played_as_written() -> None:
    queued, told = await played(Progress(SID, frozenset({TURN}), (TESTS,), ""), lambda: Opened(TURN), unasked)
    assert queued == [Working(SID, IN_TURN, (TESTS,))]
    assert told == [ProgressTold(SID, 0, None, None, current=True)]


WRITTEN = "First, how DNS works.\n1. The OS asks its resolver.\n"


async def test_the_focus_is_heard_explaining_by_a_summary_of_what_it_wrote() -> None:
    asked: list[str] = []

    async def explain(text: str) -> str:
        asked.append(text)
        return "Explain how DNS resolution works."

    queued, told = await played(Progress(SID, frozenset({TURN}), (TESTS,), WRITTEN), lambda: Opened(TURN), explain)
    assert asked == [WRITTEN.strip()]
    assert queued == [Working(SID, IN_TURN, (explained("explain how DNS resolution works"), TESTS))]
    assert told == [ProgressTold(SID, len(WRITTEN), "explain how DNS resolution works", None, current=True)]
    [spoken] = frames(queued[0], Pushed(), names=lambda _: "cc-hands")
    assert isinstance(spoken, TTSSpeakFrame) and spoken.text == "cc-hands: explain how DNS resolution works, then run the test suite."


async def test_text_that_cannot_be_summarised_is_said_to_have_been_written_and_never_read_out() -> None:
    async def explain(_text: str) -> str:
        raise SummaryFailed("nothing came back")

    queued, told = await played(Progress(SID, frozenset({TURN}), (), WRITTEN), lambda: Opened(TURN), explain)
    assert queued == [Working(SID, IN_TURN, (Doing(WRITING, None),))]
    assert told == [ProgressTold(SID, len(WRITTEN), None, "SummaryFailed: nothing came back", current=True)]


async def test_progress_whose_turn_ended_while_it_was_summarised_is_not_played() -> None:
    turn: list[Turn] = [Opened(TURN)]

    async def explain(_text: str) -> str:
        # The turn ends in the seconds the summary takes; its result is told instead.
        turn[0] = Untold(TURN, frozenset(), Stamp(2000))
        return "explain how DNS resolution works"

    queued, told = await played(Progress(SID, frozenset({TURN}), (TESTS,), WRITTEN), lambda: turn[0], explain)
    assert queued == []
    assert told == [ProgressTold(SID, len(WRITTEN), "explain how DNS resolution works", None, current=False)]


async def test_a_burst_is_not_kept_waiting_on_the_summary_of_the_one_ahead_of_it() -> None:
    queued: list[Frame] = []
    asked: list[str] = []
    second_asked = asyncio.Event()

    async def explain(text: str) -> str:
        asked.append(text)
        if text == "first":
            # The first summary is still out when the second burst is routed; it comes back only once that one began.
            await second_asked.wait()
        else:
            second_asked.set()
        return f"explain the {text} thing"

    async def queue_frame(frame: Frame) -> None:
        queued.append(frame)

    playing = Playing()
    player = asyncio.create_task(keep_playing(playing, lambda _: Session(MEMBER, Running(Busy(), Stamp(1000), None), mode=None, turn=Opened(TURN)), queue_frame, lambda _: None, explain))
    playing.put_nowait(Progress(SID, frozenset({TURN}), (), "first"))
    playing.put_nowait(Progress(SID, frozenset({TURN}), (), "second"))
    await asyncio.wait_for(second_asked.wait(), timeout=1.0)
    while len(queued) < 2:
        await asyncio.sleep(0.01)
    player.cancel()
    assert asked == ["first", "second"]
    # Played in the order they settled, whichever summary came back first.
    assert [frame.pending for frame in queued if isinstance(frame, Unprompted)] == [
        Working(SID, IN_TURN, (explained("explain the first thing"),)),
        Working(SID, IN_TURN, (explained("explain the second thing"),)),
    ]
