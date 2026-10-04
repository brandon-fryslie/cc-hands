"""The push-to-talk decisions, with no pipeline and no audio device."""

import pytest

from hands.voice.hold import Move
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
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


def test_every_turn_is_counted_sent_or_thrown_away_and_the_key_rests_after_either() -> None:
    assert Gate().after("start", "desk").after("stop", "desk") == Gate("up", "desk", sent=1)
    assert Gate().after("start", "desk").after("drop", "desk") == Gate("up", "desk", dropped=1)
    assert Gate().after("start", "desk").after("expire", "desk") == Gate("up", "desk", dropped=1)  # never sent
    # A turn ended and the next opened before a frame is captured: the count says the first was sent.
    assert Gate().after("start", "desk").after("stop", "desk").after("arm", "desk").after("start", "desk") == Gate("down", "desk", sent=1)


def test_an_engaged_desk_is_heard_between_turns_and_each_turn_ends_back_to_listening() -> None:
    engaged = Gate().after("listen", "desk")
    assert (engaged.key, engaged.listens, engaged.turn_open) == ("listening", True, False)
    assert engaged.audible(LOUD) == LOUD
    turn = engaged.after("arm", "desk").after("start", "desk")
    assert turn.turn_open
    assert turn.after("stop", "desk").key == "listening"
    assert engaged.after("arm", "desk").after("disarm", "desk").key == "listening"
    assert turn.after("expire", "desk") == Gate("listening", "desk", listens=True, dropped=1)  # and listens on


@pytest.mark.parametrize(("before", "after"), [("listening", "up"), ("arming", "arming"), ("down", "down")])
def test_disengaging_stops_the_desk_listening_and_leaves_a_press_or_a_turn_to_end_as_it_ends(before: Key, after: Key) -> None:
    gate = Gate(before, "desk", listens=True).after("deafen", "desk")
    assert (gate.key, gate.listens) == (after, False)


def test_the_phone_never_listens_between_turns_and_the_desk_does_again_once_hands_is_back() -> None:
    at_the_phone = Gate().after("listen", "desk").moved("phone")
    assert (at_the_phone.key, at_the_phone.audible(LOUD)) == ("up", QUIET)
    assert at_the_phone.after("start", "phone").after("stop", "phone").key == "up"
    assert at_the_phone.moved("desk").key == "listening"
    # Engaged while hands is at the phone: the desk listens once hands is back, and the call hears nothing meanwhile.
    engaged_away = Gate().moved("phone").after("listen", "desk")
    assert (engaged_away.key, engaged_away.place, engaged_away.listens) == ("up", "phone", True)
    assert engaged_away.moved("desk").key == "listening"


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
    assert Gate().after("start", "phone").moved("desk") == Gate("up", "desk", dropped=1)
    assert Gate().after("arm", "desk").moved("phone") == Gate("up", "phone")  # no turn open, so none thrown away
    assert Gate().moved("phone") == Gate("up", "phone")
    assert Gate("down", "phone").moved("phone") == Gate("down", "phone")


def moves(recorded: list[Entry]) -> list[WideEvent]:
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "place.moved"]


def test_every_move_of_the_place_is_recorded_once_by_what_made_it() -> None:
    recorded: list[Entry] = []
    key = PushToTalk(recorded.append)
    key.move("arm", "held key")
    key.move("start", "held key")
    key.go("phone")  # a call arrives while the user talks at the desk
    key.move("start", "phone button")
    key.move("stop", "phone button")
    key.move("start", "held key")
    key.move("stop", "held key")
    assert [move.facts for move in moves(recorded)] == [
        {"before": "desk", "after": "phone", "by": "call", "dropped": True},
        {"before": "phone", "after": "desk", "by": "turn", "dropped": False},
    ]
    assert all(move.outcome == "ok" for move in moves(recorded)) and len(recorded) == 2


@pytest.mark.parametrize("held", ["arming", "down"])
def test_a_turn_opened_at_one_place_while_a_hold_is_open_at_the_other_drops_both(held: Key) -> None:
    # Right Shift held at the desk, and the phone's button pressed: two hands on two keys.
    gate = Gate(held, "desk")
    assert gate.took("start", "phone") == "drop"
    assert gate.after("start", "phone") == Gate("up", "phone", dropped=1 if held == "down" else 0)
    recorded: list[Entry] = []
    key = PushToTalk(recorded.append)
    key.move("start", "phone button")
    assert key.move("start", "held key") == "drop"
    assert moves(recorded)[-1].facts == {"before": "phone", "after": "desk", "by": "turn", "dropped": True}


@pytest.mark.parametrize("ending", ["stop", "drop", "expire"])
def test_the_end_of_a_hold_the_gate_already_threw_away_ends_nothing_more(ending: Move) -> None:
    # The phone's turn was dropped by a press at the desk, whose hold now ends: nothing is sent, nothing cued twice.
    dropped = Gate("up", "desk", dropped=1)
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
    key.move("start", "held key")  # a turn opened at the desk while the call is up
    assert key.modality == "screen"


def test_a_turn_is_known_by_the_edge_that_opened_it_until_another_opens() -> None:
    key = PushToTalk(lambda _: None)
    key.go("phone")  # a call arrives: it opens no turn
    assert key.opened == "held key"
    key.move("start", "phone button")
    key.move("stop", "phone button")
    assert key.opened == "phone button"
    key.go("desk")
    key.move("arm", "held key")  # a press still arming opens nothing
    key.move("disarm", "held key")
    assert key.opened == "phone button"
    key.move("start", "held key")
    assert key.opened == "held key"


def test_a_switch_by_voice_holds_until_hands_moves_between_the_desk_and_the_phone() -> None:
    key = PushToTalk(lambda _: None)
    key.switch("audio-only")
    # Turns at the place it was switched at keep it.
    key.move("start", "held key")
    key.move("stop", "held key")
    assert key.modality == "audio-only"
    key.go("phone")
    key.switch("screen")
    key.move("start", "phone button")
    assert key.modality == "screen"
    key.go("desk")
    assert key.modality == "screen"
    key.go("phone")
    assert key.modality == "audio-only"


async def test_set_modality_switches_it_says_so_and_refuses_what_is_neither() -> None:
    key = PushToTalk(lambda _: None)
    switch = modality_tool(key.switch)
    assert await switch.body(modality="audio-only") == {"modality": "audio-only", "readback": "Okay, audio only."}
    assert key.modality == "audio-only"
    assert await switch.body(modality="screen") == {"modality": "screen", "readback": "Okay, you can see a screen."}
    assert key.modality == "screen"
    assert "error" in await switch.body(modality="video")
    assert key.modality == "screen"
