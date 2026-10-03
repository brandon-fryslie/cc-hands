"""The namer: after a turn, a session keeps a name that still fits its work, is given a new one when the work moves on,
and a reply that is not a name of three words at most is refused; every judging is one audit line."""

import json
from pathlib import Path

import pytest

from hands.core.session import SessionId
from hands.sessions.audit import Entry, Named
from hands.sessions.names import Finished, Names
from hands.voice.naming import NotAName, asked, judge, parsed
from hands.voice.summary import SummaryFailed

SID = SessionId("s1")


def transcript(tmp_path: Path, *names: str) -> Path:
    path = tmp_path / "s1.jsonl"
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


async def judged(path: Path, reply: str, names: Names, heard: list[str] | None = None) -> Named:
    recorded: list[Entry] = []
    await judge(Finished(SID, path, "I fixed how sessions are named."), names, answering(reply, heard), recorded.append)
    [line] = recorded
    assert isinstance(line, Named)
    return line


async def test_a_session_whose_work_moved_on_is_given_a_new_name_at_its_next_prompt(tmp_path: Path) -> None:
    names = Names()
    heard: list[str] = []
    line = await judged(transcript(tmp_path, "auth refactor"), '"Naming fix."', names, heard)
    assert (line.outcome, line.before, line.name) == ("renamed", "auth refactor", "Naming fix")
    assert names.due(SID) == "Naming fix"
    # The model is shown the name the session has now, and the last thing it said.
    assert heard == [asked("auth refactor", "I fixed how sessions are named.")]
    assert "Its name now: auth refactor" in heard[0] and "I fixed how sessions are named." in heard[0]


async def test_a_session_with_no_name_yet_is_given_its_first(tmp_path: Path) -> None:
    names = Names()
    line = await judged(transcript(tmp_path), "naming fix", names)
    assert (line.outcome, line.before, line.name) == ("renamed", None, "naming fix")
    assert names.due(SID) == "naming fix"


async def test_a_name_that_still_fits_is_kept_and_nothing_is_given(tmp_path: Path) -> None:
    names = Names()
    # Whoever set it: a name given at the keyboard is judged as one hands gave.
    line = await judged(transcript(tmp_path, "old name", "naming fix"), "naming fix", names)
    assert (line.outcome, line.name) == ("kept", "naming fix")
    assert names.due(SID) is None


async def test_a_reply_longer_than_three_words_is_refused_and_the_session_keeps_its_name(tmp_path: Path) -> None:
    names = Names()
    line = await judged(transcript(tmp_path, "auth refactor"), "fixing the session naming bug", names)
    assert (line.outcome, line.name, line.reply) == ("refused", None, "fixing the session naming bug")
    assert names.due(SID) is None


async def test_a_model_that_fails_leaves_the_name_as_it_is_and_says_why(tmp_path: Path) -> None:
    names = Names()
    recorded: list[Entry] = []

    async def failing(_text: str) -> str:
        raise SummaryFailed("the model returned no summary")

    await judge(Finished(SID, transcript(tmp_path, "auth refactor"), "done"), names, failing, recorded.append)
    [line] = recorded
    assert isinstance(line, Named) and line.outcome == "failed" and "no summary" in (line.error or "")
    assert names.due(SID) is None


async def test_a_name_that_cannot_be_read_is_not_judged(tmp_path: Path) -> None:
    path = tmp_path / "s1.jsonl"
    path.write_text('{"type":"custom-title"}\n')
    heard: list[str] = []
    line = await judged(path, "naming fix", Names(), heard)
    assert line.outcome == "unread" and heard == []


async def test_a_later_name_replaces_one_not_yet_given(tmp_path: Path) -> None:
    names = Names()
    await judged(transcript(tmp_path), "first idea", names)
    await judged(transcript(tmp_path), "second idea", names)
    assert (names.due(SID), names.due(SID)) == ("second idea", None)


@pytest.mark.parametrize("reply", ["", "   ", '""', "one two three four"])
def test_what_is_not_a_name_of_one_to_three_words_is_refused(reply: str) -> None:
    with pytest.raises(NotAName):
        parsed(reply)


def test_a_name_is_its_words_without_the_quotes_or_stop_around_them() -> None:
    assert parsed("  'naming   fix'.\n") == "naming fix"
