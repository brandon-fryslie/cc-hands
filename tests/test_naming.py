"""The namer: after a turn, a session keeps a name that still fits its work, is given a new one when the work moves on,
and a reply that is not a name of three words at most is refused; every judging is one wide event."""

import json
from pathlib import Path

import pytest

from hands.core.session import Membership, SessionId
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.sessions.names import Due, Finished, Names
from hands.voice.naming import Judged, NotAName, asked, judge, parsed
from hands.voice.summary import SummaryFailed

SID = SessionId("s1")
PROJECT = Path("/code/cc-hands")


def member(path: Path, id: str = "s1", cwd: Path = PROJECT) -> Membership:
    return Membership(SessionId(id), pid=4242, cwd=cwd, transcript=path)


def transcript(tmp_path: Path, *names: str, file: str = "s1.jsonl") -> Path:
    path = tmp_path / file
    records = [{"type": "ai-title", "aiTitle": "A long title Claude Code wrote that nobody sees"}]
    records += [{"type": "custom-title", "customTitle": name, "sessionId": "s1"} for name in names]
    path.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records))
    return path


def answering(reply: str, heard: list[str] | None = None):
    async def name(text: str) -> str:
        if heard is not None:
            heard.append(text)
        return reply

    return name


async def judged(path: Path, reply: str, names: Names, heard: list[str] | None = None, beside: tuple[Membership, ...] = ()) -> WideEvent:
    recorded: list[Entry] = []
    turn = Finished(member(path), "I fixed how sessions are named.")
    await judge(turn, names, (turn.membership, *beside), answering(reply, heard), recorded.append)
    [line] = recorded
    assert isinstance(line, WideEvent) and line.event == "name.judged" and line.facts["session"] == SID
    return line


async def test_a_session_whose_work_moved_on_is_given_a_new_name_at_its_next_prompt(tmp_path: Path) -> None:
    names = Names()
    heard: list[str] = []
    line = await judged(transcript(tmp_path, "auth refactor"), '"Naming fix."', names, heard)
    assert (line.outcome, line.facts["judged"], line.facts["before"], line.facts["name"]) == ("ok", Judged.RENAMED, "auth refactor", "Naming fix")
    assert names.due(SID) == Due("Naming fix", "auth refactor")
    # The model is shown the name the session has now, and the last thing it said.
    assert heard == [asked("auth refactor", [], "I fixed how sessions are named.")]
    assert "Its name now: auth refactor" in heard[0] and "I fixed how sessions are named." in heard[0]


async def test_a_session_with_no_name_yet_is_given_its_first(tmp_path: Path) -> None:
    names = Names()
    line = await judged(transcript(tmp_path), "naming fix", names)
    assert (line.facts["judged"], line.facts["before"], line.facts["name"]) == (Judged.RENAMED, None, "naming fix")
    assert names.due(SID) == Due("naming fix", None)


async def test_a_name_that_still_fits_is_kept_and_nothing_is_given(tmp_path: Path) -> None:
    names = Names()
    # Whoever set it: a name given at the keyboard is judged as one hands gave.
    line = await judged(transcript(tmp_path, "old name", "naming fix"), "naming fix", names)
    assert (line.outcome, line.facts["judged"], line.facts["name"]) == ("ok", Judged.KEPT, "naming fix")
    assert names.due(SID) is None


async def test_a_reply_longer_than_three_words_is_refused_and_the_session_keeps_its_name(tmp_path: Path) -> None:
    names = Names()
    line = await judged(transcript(tmp_path, "auth refactor"), "fixing the session naming bug", names)
    assert (line.outcome, line.facts["judged"], line.facts["reply"]) == ("failed", Judged.REFUSED, "fixing the session naming bug")
    assert "name" not in line.facts
    assert names.due(SID) is None


async def test_a_model_that_fails_leaves_the_name_as_it_is_and_says_why(tmp_path: Path) -> None:
    names = Names()
    recorded: list[Entry] = []

    async def failing(_text: str) -> str:
        raise SummaryFailed("the model returned no summary")

    await judge(Finished(member(transcript(tmp_path, "auth refactor")), "done"), names, (), failing, recorded.append)
    [line] = recorded
    assert isinstance(line, WideEvent) and (line.outcome, line.facts["judged"]) == ("failed", Judged.FAILED) and "no summary" in (line.error or "")
    assert names.due(SID) is None


async def test_a_name_that_cannot_be_read_is_not_judged(tmp_path: Path) -> None:
    path = tmp_path / "s1.jsonl"
    path.write_text('{"type":"custom-title"}\n')
    heard: list[str] = []
    line = await judged(path, "naming fix", Names(), heard)
    assert (line.outcome, line.facts["judged"]) == ("failed", Judged.UNREAD) and heard == []


async def test_a_later_name_replaces_one_not_yet_given(tmp_path: Path) -> None:
    names = Names()
    await judged(transcript(tmp_path), "first idea", names)
    await judged(transcript(tmp_path), "second idea", names)
    assert (names.due(SID), names.due(SID)) == (Due("second idea", None), None)


async def test_a_name_decided_and_not_yet_given_is_the_one_judged_and_kept(tmp_path: Path) -> None:
    names = Names()
    path = transcript(tmp_path, "auth refactor")
    await judged(path, "naming fix", names)
    heard: list[str] = []
    # The session has not prompted since, so Claude Code still holds the old name; the decided one is what it will have.
    line = await judged(path, "naming fix", names, heard)
    assert (line.facts["judged"], line.facts["before"]) == (Judged.KEPT, "naming fix")
    assert "Its name now: naming fix" in heard[0]
    # Still decided against the name Claude Code holds, which is what the prompt checks before giving it.
    assert names.due(SID) == Due("naming fix", "auth refactor")


async def test_a_rename_since_a_decision_is_the_name_judged(tmp_path: Path) -> None:
    names = Names()
    await judged(transcript(tmp_path, "auth refactor"), "naming fix", names)
    heard: list[str] = []
    line = await judged(transcript(tmp_path, "auth refactor", "my own name"), "my own name", names, heard)
    assert (line.facts["judged"], line.facts["before"]) == (Judged.KEPT, "my own name")
    assert "Its name now: my own name" in heard[0]


async def test_a_name_longer_than_three_words_the_session_already_has_is_kept(tmp_path: Path) -> None:
    names = Names()
    line = await judged(transcript(tmp_path, "auth token refresh rework"), "auth token refresh rework", names)
    assert (line.facts["judged"], line.facts["name"]) == (Judged.KEPT, "auth token refresh rework")
    assert names.due(SID) is None


async def test_the_model_is_shown_the_names_of_the_other_sessions_in_the_project_only(tmp_path: Path) -> None:
    names = Names()
    names.rename(SessionId("s3"), "hook tests", None)
    beside = (
        member(transcript(tmp_path, "naming fix", file="s2.jsonl"), "s2"),
        member(transcript(tmp_path, file="s3.jsonl"), "s3"),
        member(transcript(tmp_path, "elsewhere", file="s4.jsonl"), "s4", cwd=Path("/code/other")),
    )
    heard: list[str] = []
    await judged(transcript(tmp_path), "naming review", names, heard, beside)
    assert "The other sessions in its project: naming fix; hook tests\n" in heard[0]


def test_the_model_is_shown_the_end_of_a_long_closing_where_the_overview_is() -> None:
    shown = asked(None, [], "detail " * 400 + "Where it stands: the naming fix is merged.")
    assert shown.endswith("Where it stands: the naming fix is merged.") and "(the start cut)" in shown


@pytest.mark.parametrize("reply", ["", "   ", '""', "one two three four"])
def test_what_is_not_a_name_of_one_to_three_words_is_refused(reply: str) -> None:
    with pytest.raises(NotAName):
        parsed(reply, "auth refactor")


def test_a_name_is_its_words_without_the_quotes_or_stop_around_them() -> None:
    assert parsed("  'naming   fix'.\n", None) == "naming fix"
