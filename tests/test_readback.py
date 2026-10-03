"""What the user hears about a draft, from the outcome alone."""

import pytest

from hands.core.drafts import DraftAmended, DraftStaged
from hands.core.session import PromptText, Resolution, SessionId, Staged
from hands.core.spoken import spelled, spoken
from hands.voice.readback import readback

HELPER = Resolution("token helper", "tokenHelper.ts")


def amended(before: str, after: str, *resolutions: Resolution) -> str:
    was = Staged(PromptText(before), ())
    now = Staged(PromptText(after), resolutions)
    return readback(DraftAmended(SessionId("s1"), before=was, after=now), "cc-hands")


@pytest.mark.parametrize(
    ("before", "after", "heard"),
    [
        ("fix the old bug", "fix the new bug", "'old' is now 'new'"),
        ("fix the bug", "fix the bug and test it", "added 'and test it'"),
        ("fix the bug quickly", "fix the bug", "removed 'quickly'"),
        ("fix the bug then test", "fix the bug\nthen test", "added 'a line break'"),
        ("fix the  bug", "fix the bug", "only the spacing changed"),
        ("fix the bug", "fix the bug", "nothing changed"),
    ],
)
def test_an_amend_is_heard_as_what_changed(before: str, after: str, heard: str) -> None:
    assert amended(before, after) == f"In the draft for cc-hands: {heard}"


def test_an_amend_names_only_the_resolutions_it_added() -> None:
    assert amended("use the helper", "use the token helper", HELPER) == (
        "In the draft for cc-hands, reading 'token helper' as token Helper dot ts: added 'token'"
    )


def heard(text: str, *resolutions: Resolution) -> str:
    """A staged draft's readback as the speaker receives it, through the filter every utterance crosses."""
    return spoken(readback(DraftStaged(SessionId("s1"), Staged(PromptText(text), resolutions), replaced=None), "cc-hands")).text


def test_a_path_in_a_draft_is_heard_whole_so_same_named_files_are_told_apart() -> None:
    assert heard("fix src/auth.py") == "Draft for cc-hands: fix src slash auth dot py"
    assert heard("fix lib/auth.py") == "Draft for cc-hands: fix lib slash auth dot py"
    assert heard("fix auth.py") == "Draft for cc-hands: fix auth dot py"


@pytest.mark.parametrize(
    ("meant", "said"),
    [
        ("tokenHelper.ts", "token Helper dot ts"),
        ("token_helper.ts", "token underscore helper dot ts"),
        ("token-helper.ts", "token dash helper dot ts"),
        ("/Users/bmf/.hands/wire.sock", "slash Users slash bmf slash dot hands slash wire dot sock"),
    ],
)
def test_a_resolution_is_heard_as_the_exact_token_it_resolved_to(meant: str, said: str) -> None:
    assert heard("use it", Resolution("the helper", meant)) == f"Draft for cc-hands, reading 'the helper' as {said}: use it"


def test_an_amended_path_is_heard_whole() -> None:
    assert spoken(amended("fix src/auth.py", "fix lib/auth.py")).text == (
        "In the draft for cc-hands: 'src slash auth dot py' is now 'lib slash auth dot py'"
    )


@pytest.mark.parametrize("token", ["src/auth.py", "user_id", "HTTP2Handler", "~/code/cc-hands", "a+b@c:d", "tokenHelper"])
def test_the_speaker_filter_leaves_a_spelled_token_as_written(token: str) -> None:
    assert spoken(spelled(token)).text == spelled(token)
