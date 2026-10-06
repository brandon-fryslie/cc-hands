"""The echo canceller, run for real on WebRTC's AEC3 against a room made of arithmetic."""

import numpy as np
import pytest

from hands.voice.echo import EchoCanceller

HEARD_RATE, PLAYED_RATE = 16000, 24000


def _power_db(audio: bytes) -> float:
    samples = np.frombuffer(audio, np.int16).astype(np.float64)
    return float(10 * np.log10(np.mean(samples**2) + 1e-9))


def _noise(seconds: float, rate: int, seed: int) -> np.ndarray:
    """Noise held below about 4 kHz, as speech through a speaker is: none of it folds over on the way to 16 kHz."""
    white = np.random.default_rng(seed).normal(0, 6000, int(seconds * rate))
    return np.convolve(white, np.ones(6) / 6, mode="same").clip(-32768, 32767).astype(np.int16)


def test_the_echo_of_what_the_speaker_played_is_taken_out_of_what_the_microphone_hears() -> None:
    canceller = EchoCanceller()
    played = _noise(6.0, PLAYED_RATE, seed=1)
    # The room: the speaker's sound 120 ms later at half strength, at the microphone's rate.
    at_microphone = np.interp(np.arange(0, len(played), PLAYED_RATE / HEARD_RATE), np.arange(len(played)), played) * 0.5
    echo = np.concatenate([np.zeros(int(0.12 * HEARD_RATE)), at_microphone])[: len(at_microphone)].astype(np.int16)
    heard_raw, heard_clean = b"", b""
    # 20 ms buffers on both sides, as the transport writes and captures; the speaker's chunks straddle its frames,
    # 3 samples over and then 3 under, as a cue of any length leaves them.
    played_bytes, chunk = played.tobytes(), 2 * int(0.02 * PLAYED_RATE)
    for index in range(len(echo) // 320):
        start, end = index * chunk + (6 if index % 2 else 0), (index + 1) * chunk + (0 if index % 2 else 6)
        canceller.played(played_bytes[start:end], PLAYED_RATE, 1)
        buffer = echo[index * 320 : (index + 1) * 320].tobytes()
        cleaned = canceller.heard(buffer, HEARD_RATE, 1)
        assert len(cleaned) == len(buffer)
        if index >= 100:  # after two seconds to learn the room
            heard_raw, heard_clean = heard_raw + buffer, heard_clean + cleaned
    assert _power_db(heard_raw) - _power_db(heard_clean) > 15  # about 22 dB on AEC3 today; nothing taken out is 0


def test_a_voice_with_nothing_playing_is_heard_as_it_was_said() -> None:
    canceller = EchoCanceller()
    voice = _noise(2.0, HEARD_RATE, seed=2).tobytes()
    cleaned = b"".join(canceller.heard(voice[start : start + 640], HEARD_RATE, 1) for start in range(0, len(voice), 640))
    assert abs(_power_db(cleaned[16000:]) - _power_db(voice[16000:])) < 3


def test_a_voice_after_the_speaker_stops_being_written_to_is_heard_as_it_was_said() -> None:
    """The pipeline writes nothing once a reply is interrupted; the canceller's reference goes on, as silence."""
    canceller = EchoCanceller()
    played = _noise(3.0, HEARD_RATE, seed=1)
    for index in range(len(played) // 320):  # the reply, and its echo straight back
        buffer = played[index * 320 : (index + 1) * 320].tobytes()
        canceller.played(buffer, HEARD_RATE, 1)
        canceller.heard(buffer, HEARD_RATE, 1)
    voice = _noise(2.0, HEARD_RATE, seed=2).tobytes()  # then the user, with nothing more written
    cleaned = b"".join(canceller.heard(voice[start : start + 640], HEARD_RATE, 1) for start in range(0, len(voice), 640))
    assert abs(_power_db(cleaned[16000:]) - _power_db(voice[16000:])) < 3


def test_a_microphone_buffer_of_part_of_a_frame_is_refused() -> None:
    with pytest.raises(ValueError, match="not whole 10 ms frames"):
        EchoCanceller().heard(bytes(330), HEARD_RATE, 1)


def test_a_stereo_microphone_is_heard_in_frames_of_both_channels() -> None:
    canceller = EchoCanceller()
    voice = np.repeat(_noise(1.0, HEARD_RATE, seed=3), 2).tobytes()  # each sample on both channels
    cleaned = b"".join(canceller.heard(voice[start : start + 1280], HEARD_RATE, 2) for start in range(0, len(voice), 1280))
    assert len(cleaned) == len(voice)
    assert canceller.counts()["heard"] == 100  # one second of 10 ms frames, not two
    assert abs(_power_db(cleaned[32000:]) - _power_db(voice[32000:])) < 3


def test_the_canceller_counts_frames_heard_with_nothing_playing_and_sound_it_had_to_drop() -> None:
    canceller = EchoCanceller()
    canceller.heard(bytes(640), HEARD_RATE, 1)  # two frames, nothing played
    canceller.played(bytes(320 * 102), HEARD_RATE, 1)  # 102 frames played to a microphone that takes none
    canceller.heard(bytes(320), HEARD_RATE, 1)
    assert canceller.counts() == {"heard": 3, "unplayed": 2, "dropped": 2, "slipped": 0}


def test_reference_held_through_a_whole_second_that_never_needed_it_is_slipped_but_for_one_buffer() -> None:
    """As a speaker started before the microphone leaves it: 30 frames ahead, then one played for every one heard."""
    canceller = EchoCanceller()
    canceller.played(bytes(320 * 30), HEARD_RATE, 1)
    for _ in range(50):  # one second of 20 ms buffers
        canceller.played(bytes(640), HEARD_RATE, 1)
        canceller.heard(bytes(640), HEARD_RATE, 1)
    assert canceller.counts() == {"heard": 100, "unplayed": 0, "dropped": 0, "slipped": 28}
    for _ in range(50):  # the next second holds only the spare buffer, and slips nothing more
        canceller.played(bytes(640), HEARD_RATE, 1)
        canceller.heard(bytes(640), HEARD_RATE, 1)
    assert canceller.counts() == {"heard": 200, "unplayed": 0, "dropped": 0, "slipped": 28}
