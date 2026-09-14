"""What the user hears about a draft, from the outcome alone."""

import pytest

from hands.core.drafts import DraftAmended
from hands.core.session import PromptText, Resolution, SessionId, Staged
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
        "In the draft for cc-hands, reading 'token helper' as tokenHelper.ts: added 'token'"
    )
