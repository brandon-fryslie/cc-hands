"""Which wake word the desk listens for under the wake word trigger (`hands.voice.wake`): one of openWakeWord's own, or
one the user trained with openWakeWord. config.toml names it (`hands.daemon.config`), which reads it without loading the
models, so nothing here does either.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# openWakeWord's own wake words, by how each is said, and the file of each in its release.
type Phrase = Literal["Hey Jarvis", "Hey Mycroft", "Hey Rhasspy", "Alexa"]
PRETRAINED: dict[Phrase, str] = {"Hey Jarvis": "hey_jarvis_v0.1.onnx", "Hey Mycroft": "hey_mycroft_v0.1.onnx", "Hey Rhasspy": "hey_rhasspy_v0.1.onnx", "Alexa": "alexa_v0.1.onnx"}


@dataclass(frozen=True)
class Pretrained:
    """One of openWakeWord's own wake words, its model fetched from openWakeWord's release."""

    phrase: Phrase = "Hey Jarvis"


@dataclass(frozen=True)
class Trained:
    """A wake word of the user's own: the ONNX model they trained with openWakeWord, and the phrase it was trained on,
    which is what hands tells them to say."""

    phrase: str
    model: Path


# [LAW:one-type-per-behavior] a word whose model hands fetches, or one whose model the user keeps.
type Word = Pretrained | Trained


def model_of(word: Word, models: Path) -> Path:
    """The file `word` is heard with: openWakeWord's own, in `models`, or the user's, where they keep it."""
    match word:
        case Pretrained(phrase=phrase):
            return models / PRETRAINED[phrase]
        case Trained(model=model):
            return model
