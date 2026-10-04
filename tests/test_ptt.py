"""The push-to-talk decisions, with no pipeline and no audio device."""

import pytest

from hands.voice.hold import Move
from hands.sessions.audit import Entry, Moved
from hands.voice.ptt import Gate, Key, PushToTalk
from hands.voice.tools import modality_tool

LOUD = b"\x7f\x7f" * 160
QUIET = b"\x00\x00" * 160


def test_starts_up_and_silent() -> None:
    gate = Gate()
    assert gate.key == "up"
    assert gate.audible(LOUD) == QUIET


def test_a_started_turn_is_heard() -> None:
    gate = Gate().after("start", "desk")
    assert gate.audible(LOUD) == LOUD


@pytest.mark.parametrize("ended", ["stop", "drop", "expire"])
def test_an_ended_turn_is_silence_of_the_same_length(ended: Move) -> None:
    gate = Gate().after("start", "desk").after(ended, "desk")
    assert gate.audible(LOUD) == QUIET


def test_a_press_is_heard_before_it_means_talk_and_silence_once_it_is_shift() -> None:
    assert Gate().after("arm", "desk").audible(LOUD) == LOUD
    assert Gate().after("arm", "desk").after("disarm", "desk").audible(LOUD) == QUIET


def test_only_a_dropped_turn_leaves_the_key_dropped_and_the_next_turn_clears_it() -> None:
    assert Gate().after("start", "desk").after("stop", "desk").key == "up"
    dropped = Gate().after("start", "desk").after("drop", "desk")
    assert dropped.key == "dropped"
    assert Gate().after("start", "desk").after("expire", "desk").key == "dropped"  # thrown away, never sent
    assert dropped.after("arm", "desk").after("start", "desk").key == "down"


def test_a_turn_opened_at_the_other_place_moves_the_gate_there() -> None:
    gate = Gate().after("start", "phone")
    assert (gate.key, gate.place) == ("down", "phone")
    assert gate.hears("phone") and not gate.hears("desk")


@pytest.mark.parametrize("move", ["arm", "disarm", "stop", "drop", "expire"])
def test_the_other_place_cannot_touch_a_turn_but_by_opening_one(move: Move) -> None:
    # A Shift typed at the desk while the user talks on the phone.
    on_the_phone = Gate().after("start", "phone")
    assert on_the_phone.after(move, "desk") == on_the_phone


def test_leaving_a_place_mid_hold_throws_the_hold_away() -> None:
    assert Gate().after("start", "phone").moved("desk") == Gate("dropped", "desk")
    assert Gate().after("arm", "desk").moved("phone") == Gate("dropped", "phone")
    assert Gate().moved("phone") == Gate("up", "phone")
    assert Gate("down", "phone").moved("phone") == Gate("down", "phone")


def test_every_move_of_the_place_is_recorded_once_by_what_made_it() -> None:
    recorded: list[Entry] = []
    key = PushToTalk(recorded.append)
    key.move("arm", "desk")
    key.go("phone")  # a call arrives while Shift is held at the desk
    key.move("start", "phone")
    key.move("stop", "phone")
    key.move("start", "desk")
    key.move("stop", "desk")
    assert recorded == [Moved(to="phone", by="call", dropped=True), Moved(to="desk", by="turn", dropped=False)]


@pytest.mark.parametrize("held", ["arming", "down"])
def test_a_turn_opened_at_one_place_while_a_hold_is_open_at_the_other_drops_both(held: Key) -> None:
    # Right Shift held at the desk, and the phone's button pressed: two hands on two keys.
    gate = Gate(held, "desk")
    assert gate.took("start", "phone") == "drop"
    assert gate.after("start", "phone") == Gate("dropped", "phone")
    recorded: list[Entry] = []
    key = PushToTalk(recorded.append)
    key.move("start", "phone")
    assert key.move("start", "desk") == "drop"
    assert recorded[-1] == Moved(to="desk", by="turn", dropped=True)


@pytest.mark.parametrize("ending", ["stop", "drop", "expire"])
def test_the_end_of_a_hold_the_gate_already_threw_away_ends_nothing_more(ending: Move) -> None:
    # The phone's turn was dropped by a press at the desk, whose hold now ends: nothing is sent, nothing cued twice.
    dropped = Gate("dropped", "desk")
    assert dropped.took(ending, "desk") is None
    assert dropped.after(ending, "desk") == dropped
    assert dropped.took("arm", "desk") == "arm"


def test_the_other_place_moves_nothing_and_the_gate_says_so() -> None:
    on_the_phone = Gate("down", "phone")
    assert on_the_phone.took("stop", "desk") is None
    assert on_the_phone.took("start", "phone") == "start"


def test_the_way_the_user_talks_sets_whether_they_can_see_a_screen() -> None:
    key = PushToTalk(lambda _: None)
    assert key.modality == "screen"
    key.go("phone")  # a call arrives
    assert key.modality == "audio-only"
    key.move("start", "desk")  # a turn opened at the desk while the call is up
    assert key.modality == "screen"


def test_a_switch_by_voice_holds_until_the_user_talks_from_the_other_place() -> None:
    key = PushToTalk(lambda _: None)
    key.switch("audio-only")
    # Turns at the place it was switched at keep it.
    key.move("start", "desk")
    key.move("stop", "desk")
    assert key.modality == "audio-only"
    key.go("phone")
    key.switch("screen")
    key.move("start", "phone")
    assert key.modality == "screen"
    key.go("desk")
    assert key.modality == "screen"
    key.go("phone")
    assert key.modality == "audio-only"


async def test_set_modality_switches_it_says_so_and_refuses_what_is_neither() -> None:
    key = PushToTalk(lambda _: None)
    switch = modality_tool(key)
    assert await switch.body(modality="audio-only") == {"modality": "audio-only", "readback": "Okay, audio only."}
    assert key.modality == "audio-only"
    assert await switch.body(modality="screen") == {"modality": "screen", "readback": "Okay, you can see a screen."}
    assert key.modality == "screen"
    assert "error" in await switch.body(modality="video")
    assert key.modality == "screen"
