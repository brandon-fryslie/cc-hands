"""The push-to-talk decisions, with no pipeline and no audio device."""

import pytest

from hands.voice.hold import Move
from hands.voice.ptt import KEY_VAD_PARAMS, Gate, KeyVAD, PushToTalk

LOUD = b"\x7f\x7f" * 160
QUIET = b"\x00\x00" * 160


def test_starts_up_and_silent() -> None:
    gate = Gate()
    assert gate.key == "up"
    assert gate.confidence == 0.0


def test_a_started_turn_is_full_confidence_and_heard() -> None:
    gate = Gate().after("start")
    assert gate.confidence == 1.0
    assert gate.audible(LOUD) == LOUD


@pytest.mark.parametrize("ended", ["stop", "drop"])
def test_an_ended_turn_is_silence_of_the_same_length(ended: Move) -> None:
    gate = Gate().after("start").after(ended)
    assert gate.confidence == 0.0
    assert gate.audible(LOUD) == QUIET


def test_only_a_dropped_turn_leaves_the_key_dropped_and_the_next_turn_clears_it() -> None:
    assert Gate().after("start").after("stop").key == "up"
    dropped = Gate().after("start").after("drop")
    assert dropped.key == "dropped"
    assert dropped.after("start").key == "down"


def test_key_vad_reports_the_key_not_the_audio() -> None:
    key = PushToTalk()
    vad = KeyVAD(key, sample_rate=16000)
    assert vad.voice_confidence(LOUD) == 0.0
    key.move("start")
    assert vad.voice_confidence(QUIET) == 1.0
    key.move("stop")
    assert vad.voice_confidence(LOUD) == 0.0


def test_key_vad_thresholds_let_the_key_decide_alone() -> None:
    assert KEY_VAD_PARAMS.min_volume == 0.0
    assert KEY_VAD_PARAMS.confidence <= 1.0
    assert KEY_VAD_PARAMS.start_secs == KEY_VAD_PARAMS.stop_secs


def test_one_analysis_frame_is_twenty_milliseconds() -> None:
    vad = KeyVAD(PushToTalk())
    vad.set_sample_rate(16000)
    assert vad.num_frames_required() == 320
