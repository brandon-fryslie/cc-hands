"""Whose voice each hold is in: the user's, or someone else's in the room, so that two people can talk with hands among
them and hands knows which of them said what.

A hold the user's hand opened is theirs by how it was opened, so its voice teaches the user's voiceprint; nobody is
asked to enrol. A hold the voice opened, in an engaged conversation or after the wake word, is told by how like that
voiceprint it sounds. Each voice is heard as an embedding by 3D-Speaker's CAM++, trained on VoxCeleb, run on the CPU with
ONNX Runtime through sherpa-onnx: about 55 ms a hold on an M-series Mac. Nothing here loads Pipecat.
"""

import asyncio
from pathlib import Path
from typing import Literal

import numpy as np
import sherpa_onnx

from hands.sessions.audit import ByHand, Matched, Other, Record, Speaker, Untellable, Untold
from hands.sessions.wide import annotate, unit
from hands.voice import fetch
from hands.voice.transcription import RATE
from hands.voice.trigger import Opener

# The model every voice is heard with, fetched from sherpa-onnx's release into the home on the first run: 29 MB.
MODEL = "wespeaker_en_voxceleb_CAM++.onnx"
RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models"
# Long enough for the model on a slow connection; a stalled fetch fails the voice load rather than hang it, as
# Whisper's own model not yet fetched does.
FETCH_SECONDS = 120
# The user's voiceprint: the sum of every embedding it was taught, whose direction is the print.
VOICEPRINT = "voiceprint.npy"

# The cosine to the user's voiceprint at and above which a voice is theirs. CAM++'s own verification threshold sits
# about here; not yet measured on two people in one room, so each hold's similarity is in its Voiced line to tune it by.
SAME_VOICE = 0.5
# The least voice a hold must hold to teach the voiceprint: an embedding of less is mostly the room. A hold the voice
# opened is told however short, since the short ones are the yeses that move sessions.
LEAST_SECONDS = 1.0

How = Literal["by hand", "by voice"]


def how_opened(opener: Opener) -> How:
    """Whether `opener` is the user's hand, so the hold is theirs, or a voice, which could be anyone's in the room."""
    match opener:
        case "held key" | "phone button" | "typed":
            return "by hand"
        case "engaged conversation" | "wake word":
            return "by voice"


class Speakers:
    """The one owner of the user's voiceprint [LAW:single-enforcer]: each hold with words in it is told here, and a hold
    the user's hand opened teaches the print, kept in `directory` across runs. It blocks while the model runs, so it is
    called off the event loop, on the one thread Whisper runs on."""

    def __init__(self, directory: Path, record: Record) -> None:
        # [LAW:nothing-unseen] the load is a unit of work of its own: whether it fetched the model, and whether a
        # voiceprint was already taught.
        with unit("speakers.loaded", record):
            annotate(model=MODEL)
            # Off the event loop, on the voice load's own thread, so it runs a loop of its own for the fetch.
            annotate(fetched=bool(asyncio.run(fetch.fetched(directory, RELEASE, (MODEL,), FETCH_SECONDS))))
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(directory / MODEL), num_threads=2)
            )
            self._voiceprint = directory / VOICEPRINT
            self._taught: np.ndarray | None = np.load(self._voiceprint) if self._voiceprint.exists() else None
            annotate(voiceprint=self._taught is not None)

    def told(self, samples: bytes, opener: Opener) -> Speaker:
        """Whose voice mono 16-bit `samples` at RATE, with no padding after them, are in, a hold `opener` opened."""
        match how_opened(opener):
            case "by hand":
                long_enough = seconds(samples) >= LEAST_SECONDS
                if long_enough:
                    self._teach(self._embedding(samples))
                return ByHand(taught=long_enough)
            case "by voice":
                if self._taught is None:
                    return Untold("no voiceprint")
                similarity = round(float(self._embedding(samples) @ _unit(self._taught)), 3)
                return Matched(similarity) if similarity >= SAME_VOICE else Other(similarity)

    def _embedding(self, samples: bytes) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(RATE, np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768.0)
        stream.input_finished()
        return _unit(np.array(self._extractor.compute(stream), dtype=np.float32))

    def _teach(self, embedding: np.ndarray) -> None:
        self._taught = embedding if self._taught is None else self._taught + embedding
        partial = self._voiceprint.with_name(f"{VOICEPRINT}.partial.npy")
        np.save(partial, self._taught)
        partial.replace(self._voiceprint)


def seconds(samples: bytes) -> float:
    """How long mono 16-bit `samples` at RATE last."""
    return len(samples) / 2 / RATE


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def spoken_as(said: str, speaker: Speaker) -> str:
    """What the brain is given of words `said` in `speaker`'s voice: someone else's are marked as theirs, and the user's
    are given as they always were."""
    match speaker:
        case Other():
            return f"[someone else in the room] {said}"
        case ByHand() | Matched() | Untold() | Untellable():
            return said
