"""The push-to-talk decisions, with no pipeline and no audio device."""

import pytest

from hands.voice.hold import Move
from hands.voice.ptt import Gate

LOUD = b"\x7f\x7f" * 160
QUIET = b"\x00\x00" * 160


def test_starts_up_and_silent() -> None:
    gate = Gate()
    assert gate.key == "up"
    assert gate.audible(LOUD) == QUIET


def test_a_started_turn_is_heard() -> None:
    gate = Gate().after("start")
    assert gate.audible(LOUD) == LOUD


@pytest.mark.parametrize("ended", ["stop", "drop"])
def test_an_ended_turn_is_silence_of_the_same_length(ended: Move) -> None:
    gate = Gate().after("start").after(ended)
    assert gate.audible(LOUD) == QUIET


def test_only_a_dropped_turn_leaves_the_key_dropped_and_the_next_turn_clears_it() -> None:
    assert Gate().after("start").after("stop").key == "up"
    dropped = Gate().after("start").after("drop")
    assert dropped.key == "dropped"
    assert dropped.after("start").key == "down"

