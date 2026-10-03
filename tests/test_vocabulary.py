"""Whisper's vocabulary: read from the focused session's repository and the running sessions, newest last."""

import os
import subprocess
from pathlib import Path

import mlx_whisper
import pytest
from pipecat.services.whisper.stt import WhisperSTTServiceMLX

from hands.core.session import Membership, Running, Session, SessionId
from hands.core.status import Busy, Stamp
from hands.sessions.audit import Entry, Primed, level
from hands.sessions.focus import Unreadable, set_focus
from hands.sessions.home import Home
from hands.sessions.registry import Listing, Sessions
from hands.voice.vocabulary import WORDS, Lexicon, vocabulary
from hands.voice.whisper import Whisper

ENVIRONMENT = {"PATH": os.environ["PATH"], "HOME": "/nonexistent"}


def git(repository: Path, *args: str) -> None:
    identity = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True, env={**ENVIRONMENT, **identity})


def commit(repository: Path, *names: str) -> None:
    for name in names:
        (repository / name).parent.mkdir(parents=True, exist_ok=True)
        (repository / name).write_text(name)
    git(repository, "add", "-A")
    git(repository, "commit", "-qm", "work")


def session(cwd: Path, name: str = "s1") -> Session:
    return Session(Membership(SessionId(name), pid=4242, cwd=cwd, transcript=cwd / "t.jsonl"), Running(Busy(), Stamp(1000), None), mode=None)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "shop"
    root.mkdir()
    git(root, "init", "-q", "-b", "auth-rework")
    return root


async def test_the_focused_sessions_newest_files_and_branch_come_before_the_sessions_names(repository: Path, tmp_path: Path) -> None:
    commit(repository, "src/old_billing.ts")
    commit(repository, "src/authMiddleware.ts", "README.md")
    # Uncommitted work is newer than any commit, and a file renamed is named where it went.
    (repository / "src/sessionStore.ts").write_text("new")
    git(repository, "mv", "README.md", "GUIDE.md")
    focus = session(repository / "src")
    other = session(tmp_path / "cc-hands", "s2")

    primed = await vocabulary([Listing(focus, None), Listing(other, "dictation bias")], focus, ENVIRONMENT)

    assert primed.words.index("old_billing") < primed.words.index("authMiddleware") < primed.words.index("sessionStore")
    assert "GUIDE" in primed.words and primed.words.count("README") == 1
    assert primed.words[-4:] == ("auth-rework", "src", "cc-hands", "dictation bias")
    assert (primed.focus, primed.failed) == (SessionId("s1"), None)
    assert level(primed) == "info"


async def test_only_the_newest_words_are_kept(repository: Path) -> None:
    commit(repository, *(f"old_{n}.py" for n in range(WORDS + 10)))
    commit(repository, "authMiddleware.ts")
    focus = session(repository)

    primed = await vocabulary([], focus, ENVIRONMENT)

    assert len(primed.words) == WORDS
    assert primed.words[-2:] == ("authMiddleware", "auth-rework")


async def test_a_session_outside_any_repository_is_primed_with_the_sessions_alone(tmp_path: Path) -> None:
    focus = session(tmp_path / "notes")
    (tmp_path / "notes").mkdir()

    primed = await vocabulary([Listing(focus, "planning")], focus, {**ENVIRONMENT, "GIT_CEILING_DIRECTORIES": str(tmp_path)})

    assert (primed.words, primed.failed) == (("notes", "planning"), None)


async def test_a_repository_git_cannot_read_primes_nothing_of_its_own_and_says_why(repository: Path) -> None:
    focus = session(repository)

    primed = await vocabulary([Listing(focus, None)], focus, {"PATH": "/nonexistent"})

    assert primed.words == ("shop",)
    assert primed.failed is not None and "cannot run git" in primed.failed
    assert level(primed) == "error"


async def test_a_focus_that_cannot_be_read_says_so(tmp_path: Path) -> None:
    primed = await vocabulary([], Unreadable("garbled"), ENVIRONMENT)

    assert (primed.focus, primed.words, primed.failed) == (None, (), "the focus: garbled")
    assert level(primed) == "error"


async def test_with_nothing_focused_and_nothing_running_whisper_is_unprimed_and_that_is_recorded(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    home = Home(tmp_path)
    set_focus(home, None)
    lexicon = Lexicon(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=recorded.append), home, ENVIRONMENT, recorded.append)

    assert await lexicon() is None
    assert [(entry.focus, entry.words, entry.failed) for entry in recorded if isinstance(entry, Primed)] == [(None, (), None)]


async def test_whisper_transcribes_each_hold_primed_with_the_vocabulary_as_it_is_then(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[object] = []

    def transcribe(_audio: object, **options: object) -> dict[str, object]:
        asked.append(options["initial_prompt"])
        return {"segments": []}

    monkeypatch.setattr(mlx_whisper, "transcribe", transcribe)
    prompts = iter(["authMiddleware", "sessionStore"])

    async def prompt() -> str | None:
        return next(prompts)

    whisper = Whisper(settings=WhisperSTTServiceMLX.Settings(model="unused"), prompt=prompt)
    for hold in (1, 2):
        whisper._transcribing.append(hold)  # pyright: ignore[reportPrivateUsage]  (the hold a release queues)
        [frame async for frame in whisper.run_stt(b"\x00\x00" * 160)]
    # The load is unprimed; each hold is primed with what the vocabulary was when it was transcribed.
    assert asked == [None, "authMiddleware", "sessionStore"]
