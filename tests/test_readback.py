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


@pytest.mark.parametrize(
    ("draft", "said"),
    [
        ("fix src/auth.py", "fix src slash auth dot py"),
        ("fix lib/auth.py.", "fix lib slash auth dot py."),
        ("fix src/auth.test.ts", "fix src slash auth dot test dot ts"),
        ("edit ~/code/x.py and ~/.zshrc", "edit tilde slash code slash x dot py and tilde slash dot zshrc"),
        ("edit src/auth.py:42", "edit src slash auth dot py colon 42"),
        ("fix scripts/run", "fix scripts slash run"),
        ("see https://x.com/a.py", "see https colon slash slash x dot com slash a dot py"),
        ("rename user_id to userId", "rename user underscore id to user Id"),
        ("run with --force", "run with dash dash force"),
        ("revert a1b2c3d", "revert a 1 b 2 c 3 d"),
        ("(see notes.md)", "(see notes dot md)"),
        ("fix the login flow", "fix the login flow"),
    ],
)
def test_a_draft_is_heard_as_what_will_be_typed(draft: str, said: str) -> None:
    assert heard(draft) == f"Draft for cc-hands: {said}"


@pytest.mark.parametrize(
    ("meant", "said"),
    [
        ("tokenHelper.ts", "token Helper dot ts"),
        ("token_helper.ts", "token underscore helper dot ts"),
        ("token-helper.ts", "token dash helper dot ts"),
        ("/Users/bmf/.hands/wire.sock", "slash Users slash bmf slash dot hands slash wire dot sock"),
        ("a1b2c3d4", "a 1 b 2 c 3 d 4"),
        ("sessionBase64encoder.ts", "session Base 64 encoder dot ts"),
    ],
)
def test_a_resolution_is_heard_as_the_exact_token_it_resolved_to(meant: str, said: str) -> None:
    assert heard("use it", Resolution("the helper", meant)) == f"Draft for cc-hands, reading 'the helper' as {said}: use it"


def test_an_amended_path_is_heard_whole() -> None:
    assert spoken(amended("fix src/auth.py", "fix lib/auth.py")).text == (
        "In the draft for cc-hands: 'src slash auth dot py' is now 'lib slash auth dot py'"
    )


# Each one is something `spoken` rewrites when it is not spelled: a path, a name, a flag, a sha, an id, a
# uuid, an id after the word that names it, a URL.
@pytest.mark.parametrize(
    "token",
    [
        "src/auth.py", "user_id", "HTTP2Handler", "~/code/cc-hands", "a+b@c:d", "tokenHelper", "a1b2c3d4",
        "Xyzabcdefghijklmnop12", "build/a1b2c3d.js", "session sessionBase64encoder.ts", "--max-count=5",
        "0f8fad5b-d9cb-469f-a165-70867728950e", "request abcDEF123456xyz", "https://x.com/a.py", "foo.d.ts",
    ],
)
def test_the_speaker_filter_leaves_a_spelled_token_as_written(token: str) -> None:
    assert spoken(spelled(token)).text == spelled(token)
