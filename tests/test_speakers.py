from pathlib import Path

import numpy as np
import pytest

from hands.sessions.audit import ByHand, Matched, Other, Speaker, Untellable, Untold
from hands.voice import speakers
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.voice import fetch
from hands.voice.speakers import Speakers, spoken_as, teller
from hands.voice.transcription import RATE


class Voices(Speakers):
    """Speakers whose model hears each hold as the voice its first sample names: 1 the user, 2 someone else."""

    def __init__(self, directory: Path) -> None:
        self._voiceprint = directory / speakers.VOICEPRINT
        self._taught = np.load(self._voiceprint) if self._voiceprint.exists() else None

    def _embedding(self, samples: bytes) -> np.ndarray:
        return {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}[int(np.frombuffer(samples, dtype=np.int16)[0])]


def voice(who: int, seconds: float = 2.0) -> bytes:
    return np.full(int(seconds * RATE), who, dtype=np.int16).tobytes()


def test_a_voice_is_untold_until_the_users_hand_has_taught_the_voiceprint(tmp_path: Path) -> None:
    heard = Voices(tmp_path)
    assert heard.told(voice(2), "engaged conversation") == Untold("no voiceprint")
    assert heard.told(voice(1), "held key") == ByHand(taught=True)
    assert heard.told(voice(1), "wake word") == Matched(1.0)
    assert heard.told(voice(2), "engaged conversation") == Other(0.0)


def test_a_hold_too_short_to_teach_is_still_told(tmp_path: Path) -> None:
    heard = Voices(tmp_path)
    assert heard.told(voice(2, 0.5), "held key") == ByHand(taught=False)
    assert heard.told(voice(1), "held key") == ByHand(taught=True)
    # Someone else's short "yes" is theirs, not the user's.
    assert heard.told(voice(2, 0.3), "engaged conversation") == Other(0.0)


def test_only_the_desks_microphone_teaches_the_voiceprint(tmp_path: Path) -> None:
    heard = Voices(tmp_path)
    assert heard.told(voice(1), "phone button") == ByHand(taught=False)
    assert heard.told(voice(1), "wake word") == Untold("no voiceprint")


def test_a_speaker_model_that_does_not_load_fails_each_hold_rather_than_the_voice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def offline(*_: object) -> tuple[str, ...]:
        raise OSError("offline")

    monkeypatch.setattr(fetch, "fetched", offline)
    recorded: list[Entry] = []
    told = teller(tmp_path, recorded.append)
    assert [(entry.event, entry.outcome) for entry in recorded if isinstance(entry, WideEvent)] == [("speakers.loaded", "failed")]
    with pytest.raises(RuntimeError, match="the speaker model did not load: OSError: offline"):
        told(voice(1), "held key")


def test_the_voiceprint_is_kept_across_runs(tmp_path: Path) -> None:
    Voices(tmp_path).told(voice(1), "held key")
    assert Voices(tmp_path).told(voice(2), "engaged conversation") == Other(0.0)


@pytest.mark.parametrize(("speaker", "given"), [(Other(0.1), "[someone else in the room] hi"), (Matched(0.9), "hi"), (ByHand(True), "hi"), (Untold("no voiceprint"), "hi"), (Untellable("OSError"), "hi")])
def test_only_someone_elses_words_are_marked(speaker: Speaker, given: str) -> None:
    assert spoken_as("hi", speaker) == given
