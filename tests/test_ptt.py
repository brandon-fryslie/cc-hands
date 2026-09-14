"""The push-to-talk decisions, with no pipeline and no audio device."""

from hands.voice.ptt import KEY_VAD_PARAMS, Gate, KeyVAD, PushToTalk

LOUD = b"\x7f\x7f" * 160
QUIET = b"\x00\x00" * 160


def test_starts_up_and_silent() -> None:
    gate = Gate()
    assert gate.key == "up"
    assert gate.confidence == 0.0


def test_press_starts_a_turn_and_is_full_confidence() -> None:
    gate, turn = Gate().moved("down")
    assert turn == "start"
    assert gate.confidence == 1.0


def test_release_stops_the_turn_and_is_silence() -> None:
    held, _ = Gate().moved("down")
    gate, turn = held.moved("up")
    assert turn == "stop"
    assert gate.confidence == 0.0


def test_repeated_position_is_not_a_transition() -> None:
    held, _ = Gate().moved("down")
    same, turn = held.moved("down")
    assert turn == "none"
    assert same == held
    up, turn = Gate().moved("up")
    assert turn == "none"
    assert up == Gate()


def test_key_up_hears_silence_of_the_same_length() -> None:
    assert Gate().audible(LOUD) == QUIET
    held, _ = Gate().moved("down")
    assert held.audible(LOUD) == LOUD


def test_key_vad_reports_the_key_not_the_audio() -> None:
    key = PushToTalk()
    vad = KeyVAD(key, sample_rate=16000)
    assert vad.voice_confidence(LOUD) == 0.0
    assert key.move_key("down") == "start"
    assert vad.voice_confidence(QUIET) == 1.0
    assert key.move_key("up") == "stop"
    assert vad.voice_confidence(LOUD) == 0.0


def test_key_vad_thresholds_let_the_key_decide_alone() -> None:
    assert KEY_VAD_PARAMS.min_volume == 0.0
    assert KEY_VAD_PARAMS.confidence <= 1.0
    assert KEY_VAD_PARAMS.start_secs == KEY_VAD_PARAMS.stop_secs


def test_one_analysis_frame_is_twenty_milliseconds() -> None:
    vad = KeyVAD(PushToTalk())
    vad.set_sample_rate(16000)
    assert vad.num_frames_required() == 320
