"""The cues a turn's edges show and play: a terminal line and a short tone for each, and nothing for Shift."""

import numpy as np

from hands.voice.cues import CUE_LEVEL, CUE_SECONDS, DROPPED, OPENED, SENT, Cue, cues, sound

RATE = 24000


def test_a_turn_shows_its_edges_and_nothing_of_shift() -> None:
    assert [cue.line for move in ("arm", "start", "stop") for cue in cues(move)] == ["turn: started", "turn: ended"]
    assert [cue.line for move in ("arm", "start", "drop") for cue in cues(move)] == ["turn: started", "turn: dropped"]
    assert [cue.line for move in ("arm", "start", "expire") for cue in cues(move)] == ["turn: started", "turn: dropped, open 120s"]
    assert cues("arm") + cues("disarm") == ()


def samples(cue: Cue, channels: int = 1) -> np.ndarray:
    return np.frombuffer(sound(cue, RATE, channels), np.int16).astype(np.float64) / 32767


def pitch(tone: np.ndarray) -> float:
    """Hz, from the zero crossings of the tone's loud middle."""
    middle = tone[len(tone) // 4 : 3 * len(tone) // 4]
    crossings = np.count_nonzero(np.diff(np.signbit(middle)))
    return crossings / 2 / (len(middle) / RATE)


def test_each_cue_is_short_soft_and_starts_and_ends_in_silence() -> None:
    for cue in (OPENED, SENT, DROPPED):
        audio = samples(cue)
        assert len(audio) == round(RATE * CUE_SECONDS) * 2 * len(cue.glides)
        assert 0.9 * CUE_LEVEL < np.abs(audio).max() <= CUE_LEVEL
        assert np.abs(audio[:5]).max() < 0.01 and np.abs(audio[-5:]).max() < 0.01  # no click at either end


def test_opening_rises_sending_falls_and_dropping_is_low_and_twice() -> None:
    n = round(RATE * CUE_SECONDS)
    opened, sent, dropped = samples(OPENED)[:n], samples(SENT)[:n], samples(DROPPED)
    assert pitch(opened[: n // 2]) < pitch(opened[n // 2 :])
    assert pitch(sent[: n // 2]) > pitch(sent[n // 2 :])
    assert pitch(dropped[:n]) < 400 and np.abs(dropped[2 * n : 3 * n]).max() > 0.9 * CUE_LEVEL


def test_every_channel_carries_the_tone() -> None:
    stereo = samples(OPENED, channels=2)
    assert np.array_equal(stereo[0::2], stereo[1::2]) and np.array_equal(stereo[0::2], samples(OPENED))
