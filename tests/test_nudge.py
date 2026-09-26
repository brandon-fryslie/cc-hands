"""What the nudge says about a session left at its prompt."""

import pytest

from hands.core.effects import WaitingForYou
from hands.core.session import SessionId
from hands.voice.speech import announcement_text


@pytest.mark.parametrize(
    ("asking", "said"),
    [(True, "cc-hands has a question for you."), (False, "cc-hands is waiting for you.")],
)
def test_a_session_whose_turn_ended_on_a_question_is_said_to_have_one(asking: bool, said: str) -> None:
    assert announcement_text(WaitingForYou(SessionId("s1"), asking), lambda _: "cc-hands") == said
