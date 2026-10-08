from pathlib import Path

import numpy as np
import pytest

from hands.sessions.audit import ByHand, Entry, Guest, Matched, Other, Speaker, Untellable, Untold
from hands.sessions.wide import WideEvent
from hands.voice import fetch, speakers
from hands.voice.speakers import Room, Speakers, _Conversation, spoken_as, teller  # pyright: ignore[reportPrivateUsage]
from hands.voice.transcription import RATE
from hands.voice.trigger import Opener
from hands.voice.tools import name_voice_tool
from hands.voice.turnstop import Hold

# The voices the stand-in model hears: 1 the owner, 2 someone else, 3 the owner's voice heard a little otherwise, 4 a
# third person, 5 the second heard a little otherwise.
EMBEDDINGS = {1: np.array([1.0, 0.0, 0.0]), 2: np.array([0.0, 1.0, 0.0]), 3: np.array([0.8, 0.6, 0.0]), 4: np.array([0.0, 0.0, 1.0]), 5: np.array([0.0, 0.8, 0.6])}


class Voices(Speakers):
    """Speakers whose model hears each hold as the voice its first sample names."""

    def __init__(self, directory: Path, recorded: list[Entry] | None = None) -> None:
        self._voiceprint = directory / speakers.VOICEPRINT
        self._taught = np.load(self._voiceprint) if self._voiceprint.exists() else None
        self._record = (recorded if recorded is not None else []).append
        self._room = Room(directory, self._record)
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
    assert heard.told(voice(2), hold("engaged conversation")) == Other(0.316, Guest(1, None, None, True))


def test_a_hold_too_short_to_teach_is_still_told(tmp_path: Path) -> None:
    heard = Voices(tmp_path)
    assert heard.told(voice(2, 0.5), hold("held key")) == ByHand(taught=False, similarity=None)
    assert heard.told(voice(1), hold("held key")) == ByHand(taught=True, similarity=None)
    # Someone else's short "yes" is theirs, not the owner's.
    assert heard.told(voice(2, 0.3), hold("engaged conversation")) == Other(0.0, None)


def test_the_phone_never_teaches(tmp_path: Path) -> None:
    heard = Voices(tmp_path)
    assert heard.told(voice(1), hold("phone button")) == ByHand(taught=False, similarity=None)
    assert heard.told(voice(1), hold("wake word")) == Untold("no voiceprint")


def test_a_conversation_with_one_voice_in_it_teaches_the_owners_print_once_it_is_over(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    heard = Voices(tmp_path, recorded)
    for _ in range(speakers.FIRST_PRINT_HOLDS):
        assert heard.told(voice(1), hold("engaged conversation", 1)) == Untold("no voiceprint")
    # Its last hold, short, is told after the owner disengaged: still the first conversation's.
    assert heard.told(voice(1, 0.4), hold("engaged conversation", 1)) == Untold("no voiceprint")
    assert heard.told(voice(1), hold("engaged conversation", 2)) == Matched(1.0)
    assert learnt(recorded) == [{"conversation": 1, "holds": 4, "alike": 1.0, "owners": None, "alone": True, "taught": 3}]
    # Used in the next conversation, with someone else in it.
    assert heard.told(voice(2), hold("engaged conversation", 2)) == Other(0.0, Guest(1, None, None, True))


def test_one_remark_alone_is_not_the_first_print(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    heard = Voices(tmp_path, recorded)
    heard.told(voice(2), hold("engaged conversation", 1))
    assert heard.told(voice(1), hold("engaged conversation", 2)) == Untold("no voiceprint")
    assert [(facts["alone"], facts["taught"]) for facts in learnt(recorded)] == [(False, 0)]


def test_the_wake_words_holds_are_told_and_never_learnt_from(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    heard = Voices(tmp_path, recorded)
    for conversation in (1, 2, 3, 4):
        heard.told(voice(1), hold("wake word", conversation))
    assert heard.told(voice(1), hold("engaged conversation", 5)) == Untold("no voiceprint")
    assert learnt(recorded) == []


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
    told = teller(tmp_path, Room(tmp_path, recorded.append), recorded.append)
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
    assert Voices(tmp_path).told(voice(2), hold("engaged conversation")) == Other(0.0, Guest(1, None, None, True))


@pytest.mark.parametrize(
    ("speaker", "given"),
    [
        (Other(0.1, None), "[someone else in the room: hi]"),
        (Other(0.1, Guest(2, None, None, True)), "[someone else in the room, voice 2, name not yet known: hi]"),
        (Other(0.1, Guest(2, "Sam", 0.9, True)), "[Sam, someone else in the room: hi]"), (Matched(0.9), "hi"), (ByHand(True, None), "hi"), (Untold("no voiceprint"), "hi"), (Untellable("OSError"), "hi")],
)
def test_only_someone_elses_words_are_marked(speaker: Speaker, given: str) -> None:
    assert spoken_as("hi", speaker) == given


def test_everyone_else_has_a_print_of_their_own(tmp_path: Path) -> None:
    heard = Voices(tmp_path)
    heard.told(voice(1), hold("held key"))
    assert heard.told(voice(2), hold("wake word")) == Other(0.0, Guest(1, None, None, True))
    assert heard.told(voice(4), hold("wake word")) == Other(0.0, Guest(2, None, None, True))
    # Heard again, a little otherwise, and too briefly to teach: still the first of them.
    assert heard.told(voice(5, 0.3), hold("wake word")) == Other(0.0, Guest(1, None, 0.8, False))


def test_a_name_is_kept_with_its_print_across_runs(tmp_path: Path) -> None:
    heard = Voices(tmp_path)
    heard.told(voice(1), hold("held key"))
    heard.told(voice(2), hold("wake word"))
    heard._room.named(1, "Sam")  # pyright: ignore[reportPrivateUsage]
    assert Voices(tmp_path).told(voice(2), hold("wake word")) == Other(0.0, Guest(1, "Sam", 1.0, True))


def test_naming_a_voice_nobody_has_names_the_ones_there_are(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match=r"no one in the room has voice 3: the voices are \[\]"):
        Room(tmp_path, lambda _entry: None).named(3, "Sam")


def test_a_room_that_is_no_room_fails_naming_it(tmp_path: Path) -> None:
    (tmp_path / speakers.ROOM).write_text("[{}]")
    with pytest.raises(ValueError, match="is not a room of voices"):
        Room(tmp_path, lambda _entry: None).placed(np.array([1.0, 0.0, 0.0]), True)


async def test_name_voice_names_a_voice_the_room_has_and_refuses_one_it_has_not(tmp_path: Path) -> None:
    room = Room(tmp_path, lambda _entry: None)
    room.placed(np.array([1.0, 0.0, 0.0]), True)
    name_voice = name_voice_tool(room)
    assert await name_voice.body(voice=1, name="Sam") == {"named": "Sam"}
    assert await name_voice.body(voice=2, name="Ada") == {"error": "no one in the room has voice 2: the voices are [1]"}
    assert room.placed(np.array([1.0, 0.0, 0.0]), False) == Guest(1, "Sam", 1.0, False)


def test_the_room_is_read_once_a_run_saying_how_many_voices_and_names_it_keeps(tmp_path: Path) -> None:
    Room(tmp_path, lambda _entry: None).placed(np.array([1.0, 0.0, 0.0]), True)
    recorded: list[Entry] = []
    room = Room(tmp_path, recorded.append)
    room.named(1, "Sam")
    room.placed(np.array([1.0, 0.0, 0.0]), True)
    assert [dict(entry.facts) for entry in recorded if isinstance(entry, WideEvent) and entry.event == "speakers.room"] == [{"voices": 1, "named": 0}]
