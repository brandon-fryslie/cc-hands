from pathlib import Path

import numpy as np
import pytest

from hands.sessions.audit import ByHand, Entry, Matched, Other, Speaker, Untellable, Untold
from hands.sessions.wide import WideEvent
from hands.voice import fetch, speakers
from hands.voice.speakers import Speakers, _Conversation, spoken_as, teller  # pyright: ignore[reportPrivateUsage]
from hands.voice.transcription import RATE
from hands.voice.trigger import Opener
from hands.voice.turnstop import Hold

# The voices the stand-in model hears: 1 the owner, 2 someone else, 3 the owner's voice heard a little otherwise.
EMBEDDINGS = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0]), 3: np.array([0.8, 0.6])}


class Voices(Speakers):
    """Speakers whose model hears each hold as the voice its first sample names."""

    def __init__(self, directory: Path, recorded: list[Entry] | None = None) -> None:
        self._voiceprint = directory / speakers.VOICEPRINT
        self._taught = np.load(self._voiceprint) if self._voiceprint.exists() else None
        self._record = (recorded if recorded is not None else []).append
        self._conversation = _Conversation(0)

    def _embedding(self, samples: bytes) -> np.ndarray:
        return EMBEDDINGS[int(np.frombuffer(samples, dtype=np.int16)[0])]


def voice(who: int, seconds: float = 2.0) -> bytes:
    return np.full(int(seconds * RATE), who, dtype=np.int16).tobytes()


def hold(opener: Opener, conversation: int = 0) -> Hold:
    return Hold(1, opener, conversation)


def learnt(recorded: list[Entry]) -> list[dict[str, object]]:
    return [dict(entry.facts) for entry in recorded if isinstance(entry, WideEvent) and entry.event == "speakers.learnt"]


def test_a_hold_of_the_desks_key_teaches_and_measures_the_owners_own_similarity(tmp_path: Path) -> None:
    heard = Voices(tmp_path)
    assert heard.told(voice(2), hold("engaged conversation")) == Untold("no voiceprint")
    assert heard.told(voice(1), hold("held key")) == ByHand(taught=True, similarity=None)
    assert heard.told(voice(3), hold("held key")) == ByHand(taught=True, similarity=0.8)
    assert heard.told(voice(1), hold("wake word")) == Matched(0.949)
    assert heard.told(voice(2), hold("engaged conversation")) == Other(0.316)


def test_a_hold_too_short_to_teach_is_still_told(tmp_path: Path) -> None:
    heard = Voices(tmp_path)
    assert heard.told(voice(2, 0.5), hold("held key")) == ByHand(taught=False, similarity=None)
    assert heard.told(voice(1), hold("held key")) == ByHand(taught=True, similarity=None)
    # Someone else's short "yes" is theirs, not the owner's.
    assert heard.told(voice(2, 0.3), hold("engaged conversation")) == Other(0.0)


def test_the_phone_never_teaches(tmp_path: Path) -> None:
    heard = Voices(tmp_path)
    assert heard.told(voice(1), hold("phone button")) == ByHand(taught=False, similarity=None)
    assert heard.told(voice(1), hold("wake word")) == Untold("no voiceprint")


def test_a_conversation_with_one_voice_in_it_teaches_the_owners_print_once_it_is_over(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    heard = Voices(tmp_path, recorded)
    assert heard.told(voice(1), hold("engaged conversation", 1)) == Untold("no voiceprint")
    # Its last hold, short, is told after the owner disengaged: still the first conversation's.
    assert heard.told(voice(1, 0.4), hold("engaged conversation", 1)) == Untold("no voiceprint")
    assert heard.told(voice(1), hold("engaged conversation", 2)) == Matched(1.0)
    assert learnt(recorded) == [{"conversation": 1, "holds": 2, "alike": 1.0, "owners": None, "alone": True, "taught": 1}]
    # Used in the next conversation, with someone else in it.
    assert heard.told(voice(2), hold("engaged conversation", 2)) == Other(0.0)


def test_a_conversation_with_two_voices_in_it_teaches_nothing(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    heard = Voices(tmp_path, recorded)
    heard.told(voice(1), hold("engaged conversation", 1))
    heard.told(voice(2), hold("engaged conversation", 1))
    assert heard.told(voice(2), hold("engaged conversation", 2)) == Untold("no voiceprint")
    assert [(facts["alone"], facts["taught"]) for facts in learnt(recorded)] == [(False, 0)]


def test_someone_elses_solo_conversation_is_not_the_owners(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    heard = Voices(tmp_path, recorded)
    heard.told(voice(1), hold("held key"))
    heard.told(voice(2), hold("engaged conversation", 1))
    heard.told(voice(1), hold("engaged conversation", 2))
    assert learnt(recorded)[0]["owners"] == 0.0 and learnt(recorded)[0]["taught"] == 0


def test_a_speaker_model_that_does_not_load_fails_each_voice_opened_hold_rather_than_the_voice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def offline(*_: object) -> tuple[str, ...]:
        raise OSError("offline")

    monkeypatch.setattr(fetch, "fetched", offline)
    recorded: list[Entry] = []
    told = teller(tmp_path, recorded.append)
    assert [(entry.event, entry.outcome) for entry in recorded if isinstance(entry, WideEvent)] == [("speakers.loaded", "failed")]
    # A hold the owner's hand opened is theirs without the model.
    assert told(voice(1), hold("held key")) == ByHand(taught=False, similarity=None)
    with pytest.raises(RuntimeError, match="the speaker model did not load: OSError: offline"):
        told(voice(1), hold("engaged conversation"))


def test_a_voiceprint_that_is_no_print_fails_the_load_naming_it(tmp_path: Path) -> None:
    np.save(tmp_path / speakers.VOICEPRINT, np.array([np.nan, 1.0]))
    with pytest.raises(ValueError, match="is not a voiceprint"):
        speakers._loaded(tmp_path / speakers.VOICEPRINT)  # pyright: ignore[reportPrivateUsage]


def test_the_voiceprint_is_kept_across_runs(tmp_path: Path) -> None:
    Voices(tmp_path).told(voice(1), hold("held key"))
    assert Voices(tmp_path).told(voice(2), hold("engaged conversation")) == Other(0.0)


@pytest.mark.parametrize(
    ("speaker", "given"),
    [(Other(0.1), "[someone else in the room: hi]"), (Matched(0.9), "hi"), (ByHand(True, None), "hi"), (Untold("no voiceprint"), "hi"), (Untellable("OSError"), "hi")],
)
def test_only_someone_elses_words_are_marked(speaker: Speaker, given: str) -> None:
    assert spoken_as("hi", speaker) == given
