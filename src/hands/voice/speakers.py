"""Whose voice each hold is in: the owner's, or someone else's in the room, so that two people can talk with hands among
them and hands knows which of them said what. Anyone who is not the owner is someone else, and each of them is known
by a voiceprint of their own and, once they have said it, their name: the Room keeps both across runs.

The owner's voiceprint is learnt without asking anyone to enrol, only from what can only be theirs: a hold of the desk's
key, and an engaged conversation at the desk with one voice in it. A hold the voice opened, in an engaged conversation
or after the wake word, is told by how like that print it sounds; an engaged conversation is learnt from once it is over
and known to have been the owner's alone. The wake word listens for as long as it is the trigger, with no conversation
ending in it, so its holds are told and never learnt from. Each voice is heard as an embedding by 3D-Speaker's CAM++, trained on VoxCeleb, run on the CPU
with ONNX Runtime through sherpa-onnx: about 55 ms a hold on an M-series Mac.
"""

import asyncio
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from hands.sessions.audit import ByHand, Guest, Matched, Other, Record, Speaker, Unplaced, Untellable, Untold
from hands.sessions.files import replace_whole
from hands.sessions.wide import annotate, unit
from hands.voice import fetch
from hands.voice.transcription import RATE
from hands.voice.trigger import Opener
from hands.voice.turnstop import Hold

# The model every voice is heard with, fetched from sherpa-onnx's release into the home on the first run: 29 MB.
MODEL = "wespeaker_en_voxceleb_CAM++.onnx"
RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models"
# Long enough for the model on a slow connection; a stalled fetch fails the voice load rather than hang it, as
# Whisper's own model not yet fetched does.
FETCH_SECONDS = 120
# The owner's voiceprint: the sum of every embedding it was taught, whose direction is the print.
VOICEPRINT = "voiceprint.npy"
# Everyone else's voiceprints and names (Room).
ROOM = "room.json"

# The cosine at and above which two voices are one: a hold's to the owner's print, and each hold's to every other in a
# conversation learnt from. CAM++'s own verification threshold sits about here; not yet measured on two people in one
# room, so each hold's similarity is in its Voiced line, the owner's own held-key holds among them, to tune it by.
SAME_VOICE = 0.5
# The least voice a hold must hold to teach the voiceprint: an embedding of less is mostly the room. A hold the voice
# opened is told however short, since the short ones are the yeses that move sessions.
LEAST_SECONDS = 1.0
# The holds long enough to teach that a conversation must have to teach the first print, with none to check it against:
# one remark of someone else's, the owner silent, is not a print.
FIRST_PRINT_HOLDS = 3

How = Literal["by hand", "by voice"]


def how_opened(opener: Opener) -> How:
    """Whether `opener` is the owner's hand, so the hold is theirs, or a voice, which could be anyone's in the room."""
    match opener:
        case "held key" | "phone button" | "typed":
            return "by hand"
        case "engaged conversation" | "wake word":
            return "by voice"


@dataclass
class _Conversation:
    """The voice-opened holds heard so far in one of the desk's conversations: each one's embedding, and whether it was
    long enough to teach."""

    number: int
    heard: list[tuple[np.ndarray, bool]] = field(default_factory=list[tuple[np.ndarray, bool]])


class Speakers:
    """The one owner of the owner's voiceprint [LAW:single-enforcer]: each hold with words in it is told here, and what
    can only be the owner's teaches the print, kept in `directory` across runs. It blocks while the model runs, so it is
    called off the event loop, on the one thread Whisper runs on."""

    def __init__(self, directory: Path, room: "Room", record: Record) -> None:
        # [LAW:nothing-unseen] the load is a unit of work of its own: whether it fetched the model, and whether a
        # voiceprint was already taught.
        with unit("speakers.loaded", record):
            annotate(model=MODEL)
            # Here, not at the top: a native library that does not load fails the load, which `teller` survives.
            import sherpa_onnx

            # Off the event loop, on the voice load's own thread, so it runs a loop of its own for the fetch.
            annotate(fetched=bool(asyncio.run(fetch.fetched(directory, RELEASE, (MODEL,), FETCH_SECONDS))))
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(directory / MODEL), num_threads=2)
            )
            self._voiceprint = directory / VOICEPRINT
            self._taught = _loaded(self._voiceprint)
            annotate(voiceprint=self._taught is not None)
        self._room = room
        self._record = record
        self._conversation = _Conversation(0)

    def told(self, samples: bytes, hold: Hold) -> Speaker:
        """Whose voice mono 16-bit `samples` at RATE, with no padding after them, are in, the voice of `hold`."""
        long_enough = seconds(samples) >= LEAST_SECONDS
        match how_opened(hold.opener):
            case "by hand":
                # Only the desk's microphone teaches: a hold the voice opens is heard there, and a phone's voice would
                # sound like someone else's beside it.
                if hold.opener != "held key" or not long_enough:
                    return ByHand(taught=False, similarity=None)
                embedding = self._embedding(samples)
                # The owner's own, so its similarity is what SAME_VOICE is tuned by.
                similarity = None if self._taught is None else _cosine(embedding, self._taught)
                self._teach((embedding,))
                return ByHand(taught=True, similarity=similarity)
            case "by voice":
                embedding = self._embedding(samples)
                if hold.opener == "engaged conversation":
                    # [LAW:no-ambient-temporal-coupling] a conversation is over when a hold of the next one is told,
                    # so its last hold, told after the owner disengaged, is still counted in it.
                    if hold.conversation != self._conversation.number:
                        # Moved on before it is learnt from, so a print that failed to save is not taught twice.
                        over, self._conversation = self._conversation, _Conversation(hold.conversation)
                        self._learn(over)
                    self._conversation.heard.append((embedding, long_enough))
                if self._taught is None:
                    return Untold("no voiceprint")
                similarity = _cosine(embedding, self._taught)
                if similarity >= SAME_VOICE:
                    return Matched(similarity)
                try:
                    guest = self._room.placed(embedding, long_enough)
                except Exception as error:
                    # [LAW:no-silent-failure] the Room failing is an error line, and the hold is still someone else's:
                    # an Untellable hold's words would reach the brain as the owner's.
                    guest = Unplaced(f"{type(error).__name__}: {error}")
                return Other(similarity, guest)

    def _learn(self, conversation: _Conversation) -> None:
        """Teach the print from `conversation` if it was the owner's alone: every hold in it one voice, and that voice
        the print's, where there is a print yet."""
        if not conversation.heard:
            return
        # [LAW:nothing-unseen] each conversation over is judged once: how many holds, whether one voice, how many taught.
        with unit("speakers.learnt", self._record):
            embeddings = np.stack([embedding for embedding, _ in conversation.heard])
            # The least alike of any two holds: two voices make at least one pair unalike, however many holds are each's.
            alike = round(float((embeddings @ embeddings.T).min()), 3)
            owners = None if self._taught is None else _cosine(embeddings.sum(axis=0), self._taught)
            long_enough = tuple(embedding for embedding, long_enough in conversation.heard if long_enough)
            alone = alike >= SAME_VOICE and (len(long_enough) >= FIRST_PRINT_HOLDS if owners is None else owners >= SAME_VOICE)
            teaching = long_enough if alone else ()
            annotate(conversation=conversation.number, holds=len(conversation.heard), alike=alike, owners=owners, alone=alone, taught=len(teaching))
            self._teach(teaching)

    def _embedding(self, samples: bytes) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(RATE, np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768.0)
        stream.input_finished()
        embedding = np.array(self._extractor.compute(stream), dtype=np.float32)
        # [LAW:no-silent-failure] a print summed with one NaN is NaN for good, and like nobody, the owner included.
        if not np.isfinite(embedding).all() or not np.linalg.norm(embedding):
            raise ValueError("the speaker model heard no voice it could embed in the hold")
        return _unit(embedding)

    def _teach(self, embeddings: tuple[np.ndarray, ...]) -> None:
        if not embeddings:
            return
        taught = sum(embeddings, start=np.zeros_like(embeddings[0]))
        print_ = taught if self._taught is None else self._taught + taught
        partial = self._voiceprint.with_name(f"{VOICEPRINT}.partial.npy")
        np.save(partial, print_)
        partial.replace(self._voiceprint)
        # Only once kept: a print that failed to save is not the one told by.
        self._taught = print_


@dataclass
class _Known:
    """One of the others in the room: their number, the name they gave, and the sum of every embedding that taught
    their print, whose direction is the print."""

    voice: int
    name: str | None
    print_: np.ndarray


class Room:
    """The one owner of everyone else's voiceprints and names [LAW:single-enforcer], kept in `directory` across runs,
    so a voice heard in a new run is known by its print and its name without asking again. A voice that is like none of
    them, held long enough to make a print of, is someone new. Told on the speaker model's thread and named from the
    brain's tools, so each is under one lock; read the first time either needs it, so a room that does not load fails
    those, not the daemon."""

    def __init__(self, directory: Path, record: Record) -> None:
        self._path = directory / ROOM
        self._record = record
        self._lock = threading.Lock()
        self._known: list[_Known] | None = None

    def placed(self, embedding: np.ndarray, long_enough: bool) -> Guest | None:
        """Which of the others unit `embedding` is: the one whose print it is most like, taught by it if it is
        `long_enough`, or someone new where it is like none and long enough to be a print. None where it is too short
        to tell whose it is."""
        with self._lock:
            known = self._loaded()
            likeness, nearest = max(((_cosine(embedding, guest.print_), guest) for guest in known), key=lambda pair: pair[0], default=(0.0, None))
            if nearest is not None and likeness >= SAME_VOICE:
                if long_enough:
                    self._save([_Known(guest.voice, guest.name, guest.print_ + embedding) if guest is nearest else guest for guest in known])
                return Guest(nearest.voice, nearest.name, likeness, long_enough)
            if not long_enough:
                return None
            new = _Known(max((guest.voice for guest in known), default=0) + 1, None, embedding)
            self._save([*known, new])
            return Guest(new.voice, None, None, True)

    def named(self, voice: int, name: str) -> None:
        """Give the voice numbered `voice` the name `name`, kept with its print."""
        # [LAW:parse-dont-validate] a name is what goes inside spoken_as's brackets: one that closes them would put a
        # guest's words outside, where they read as the owner's.
        name = name.strip()
        if not name or any(mark in name for mark in "[]"):
            raise ValueError(f"{name!r} is not a name: a name has words in it and no brackets")
        with self._lock:
            known = self._loaded()
            if voice not in (guest.voice for guest in known):
                raise ValueError(f"no one in the room has voice {voice}: the voices are {[guest.voice for guest in known]}")
            self._save([_Known(guest.voice, name, guest.print_) if guest.voice == voice else guest for guest in known])

    def _loaded(self) -> list[_Known]:
        if self._known is None:
            # [LAW:nothing-unseen] the room is read once a run: how many voices it keeps, and how many have names.
            with unit("speakers.room", self._record):
                self._known = _room(self._path)
                annotate(voices=len(self._known), named=sum(guest.name is not None for guest in self._known))
        return self._known

    def _save(self, known: list[_Known]) -> None:
        # Private: voiceprints and names are the people's own.
        replace_whole(self._path, json.dumps([{"voice": guest.voice, "name": guest.name, "print": guest.print_.tolist()} for guest in known]), 0o600)
        # Only once kept: a room that failed to save is not the one told by.
        self._known = known


def _room(path: Path) -> list[_Known]:
    """The voices kept at `path`, none where none was ever heard; a file that is no room fails, naming it."""
    if not path.exists():
        return []
    try:
        known = [_Known(int(guest["voice"]), guest["name"], np.array(guest["print"], dtype=np.float32)) for guest in json.loads(path.read_text())]
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"{path} is not a room of voices ({error}): delete it, and the others in the room are heard anew") from error
    if any(guest.print_.ndim != 1 or not np.isfinite(guest.print_).all() or not np.linalg.norm(guest.print_) for guest in known):
        raise ValueError(f"{path} holds a voice that is no voiceprint: delete it, and the others in the room are heard anew")
    return known


def teller(directory: Path, room: Room, record: Record) -> Callable[[bytes, Hold], Speaker]:
    """What tells each hold's voice: Speakers in `directory`, or, where they could not be loaded, a teller that takes a
    hold the owner's hand opened as theirs, as it would be anyway, and fails each the voice opened with why, so its words
    still reach the brain as an Untellable hold's do and the voice is not lost with the model."""
    try:
        return Speakers(directory, room, record).told
    except Exception as error:
        # [LAW:no-silent-failure] the load failed on speakers.loaded, and each hold that needed it says so again.
        unloaded = f"the speaker model did not load: {type(error).__name__}: {error}"

        def unloaded_teller(_samples: bytes, hold: Hold) -> Speaker:
            match how_opened(hold.opener):
                case "by hand":
                    return ByHand(taught=False, similarity=None)
                case "by voice":
                    raise RuntimeError(unloaded)

        return unloaded_teller


def seconds(samples: bytes) -> float:
    """How long mono 16-bit `samples` at RATE last."""
    return len(samples) / 2 / RATE


def _loaded(voiceprint: Path) -> np.ndarray | None:
    """The print kept at `voiceprint`, none if it was never taught; one that is no print fails the load, naming it."""
    if not voiceprint.exists():
        return None
    taught = np.load(voiceprint)
    if taught.ndim != 1 or not np.isfinite(taught).all() or not np.linalg.norm(taught):
        raise ValueError(f"{voiceprint} is not a voiceprint: delete it, and the next holds of the desk's key teach anew")
    return taught


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _cosine(vector: np.ndarray, print_: np.ndarray) -> float:
    return round(float(_unit(vector) @ _unit(print_)), 3)


def spoken_as(said: str, speaker: Speaker) -> str:
    """What the brain is given of words `said` in `speaker`'s voice: someone else's are marked as theirs from start to
    end, by their name or, until they have given one, by their voice's number, so words of theirs and the owner's in one
    turn are told apart, and the owner's are given as they always were."""
    match speaker:
        case Other(guest=Guest(name=str() as name)):
            return f"[{name}, someone else in the room: {said}]"
        case Other(guest=Guest(voice=voice)):
            return f"[someone else in the room, voice {voice}, name not yet known: {said}]"
        case Other():
            return f"[someone else in the room: {said}]"
        case ByHand() | Matched() | Untold() | Untellable():
            return said
