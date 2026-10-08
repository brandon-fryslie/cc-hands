import json
import os
import sys
from pathlib import Path

import pytest

from hands.core.session import Membership, SessionId
from hands.sessions.questiontimeout import QuestionTimeout, Unread, question_timeout, timeout_of
from hands.sessions.terminals import Launch


def config(tmp_path: Path, settings: object | None = None) -> Path:
    directory = tmp_path / "claude"
    directory.mkdir()
    if settings is not None:
        (directory / "settings.json").write_text(json.dumps(settings))
    return directory


def started(*arguments: str, directory: Path = Path("/")) -> Launch:
    return Launch(directory, arguments)


@pytest.mark.parametrize(("value", "seconds"), [("60s", 60.0), ("5m", 300.0), ("10m", 600.0), ("never", None)])
def test_the_users_settings_set_it_to_each_value_claude_code_has(tmp_path: Path, value: str, seconds: float | None) -> None:
    assert timeout_of(started(), config(tmp_path, {"askUserQuestionTimeout": value})) == QuestionTimeout(seconds, "userSettings", ())


def test_with_nothing_set_a_question_is_waited_on_as_long_as_it_takes(tmp_path: Path) -> None:
    assert timeout_of(started(), config(tmp_path)) == QuestionTimeout(None, None, ())


def test_settings_given_at_launch_outrank_the_users(tmp_path: Path) -> None:
    flag = json.dumps({"askUserQuestionTimeout": "10m"})
    assert timeout_of(started("--settings", flag), config(tmp_path, {"askUserQuestionTimeout": "60s"})) == QuestionTimeout(600.0, "flagSettings", ())


def test_a_settings_file_given_at_launch_is_read_from_where_claude_was_started(tmp_path: Path) -> None:
    (tmp_path / "mine.json").write_text(json.dumps({"askUserQuestionTimeout": "60s"}))
    assert timeout_of(started("--model", "opus", "--settings=mine.json", directory=tmp_path), config(tmp_path)) == QuestionTimeout(60.0, "flagSettings", ())


def test_the_last_settings_given_at_launch_are_the_ones_claude_code_takes(tmp_path: Path) -> None:
    first, last = json.dumps({"askUserQuestionTimeout": "60s"}), json.dumps({"askUserQuestionTimeout": "5m"})
    assert timeout_of(started("--settings", first, "--settings", last), config(tmp_path)).seconds == 300.0


def test_launch_settings_that_leave_it_unset_fall_through_to_the_users(tmp_path: Path) -> None:
    flag = json.dumps({"model": "opus", "askUserQuestionTimeout": "2m"})
    assert timeout_of(started("--settings", flag), config(tmp_path, {"askUserQuestionTimeout": "60s"})) == QuestionTimeout(60.0, "userSettings", ())


def test_settings_that_cannot_be_read_are_said_and_the_rest_still_count(tmp_path: Path) -> None:
    directory = config(tmp_path)
    (directory / "settings.json").write_text("{not json")
    read = timeout_of(Unread("the session's process has exited"), directory)
    assert (read.seconds, read.set_by) == (None, None)
    assert read.unread[0] == "flagSettings: the session's process has exited"
    assert read.unread[1].startswith(f"userSettings: {directory / 'settings.json'}: not JSON")
    assert timeout_of(started("--settings", json.dumps({"askUserQuestionTimeout": "60s"})), Unread("no config")) == QuestionTimeout(60.0, "flagSettings", ("userSettings: no config",))


def test_a_running_sessions_own_process_and_config_are_read(tmp_path: Path) -> None:
    # This test's own process stands in for the session's claude: it was started with no --settings.
    transcript = config(tmp_path, {"askUserQuestionTimeout": "60s"}) / "projects" / "-code-a" / "s.jsonl"
    membership = Membership(SessionId("s"), pid=os.getpid(), cwd=Path.cwd(), transcript=transcript)
    assert "--settings" not in " ".join(sys.argv)
    assert question_timeout(membership) == QuestionTimeout(60.0, "userSettings", ())
