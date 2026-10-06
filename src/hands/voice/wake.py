"""The wake word: saying it opens a turn with no hand on the key, and end-of-turn detection closes it, as in an
engaged conversation (`hands.voice.engaged`), whose driver and models this edge shares.

The word is config.toml's (`hands.daemon.config`): one of openWakeWord's own, "Hey Jarvis" unless another is named, or
one the user trained with openWakeWord. Its model listens to the desk's microphone through the echo canceller. It is
half-duplex: while hands speaks the detector is given silence in place of the room, because a microphone open in a room
with speakers hears the pipeline's own voice. The desk listens for as long as the trigger is in use, so Whisper keeps the
second before each turn opens and the wake word, and anything said while the detector made sure of it, are in the turn.
"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import cast

import aiohttp
import numpy as np
from openwakeword.model import Model

from hands.sessions.wide import WideEvent, annotate, count, unit
from hands.voice.engaged import Act, Begun, Conversation, Disengaged, Engagement, Event, Listening, SpeechStarted, SpeechStopped, Talking, TurnEnded, TurnTooLong, Woke, Woken, in_turn, released
from hands.voice.wakeword import PRETRAINED, Pretrained, Trained, Word, model_of

# openWakeWord's release, and the files hands runs the wake word from, in ONNX: the two models that turn audio into the
# features every wake word model hears, and a model of each of openWakeWord's own wake words (`PRETRAINED`).
RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
MELSPECTROGRAM = "melspectrogram.onnx"
EMBEDDING = "embedding_model.onnx"
# Long enough for the three files, about 3 MB, on a slow connection; a stalled one fails the switch rather than hang it.
FETCH_SECONDS = 60
# openWakeWord's own default: its pretrained models are tuned to score a wake word above it and little else.
THRESHOLD = 0.5
# The only rate openWakeWord's models hear.
SAMPLE_RATE = 16_000


def step(engagement: Engagement, event: Event) -> tuple[Engagement, tuple[Act, ...]]:
    """The engagement after `event`, and what the edge does: the desk listens from the start, the wake word opens a turn,
    and the talk key does nothing."""
    match engagement.phase, event:
        case Disengaged(), Begun():
            return replace(engagement, phase=Listening()), ("listen",)
        case Listening(), Woken(at=at):
            return replace(engagement, phase=Woke(at)), ("arm", "start")
        # [LAW:types-are-the-program] the pause after the wake word is no end of the turn: Smart Turn judges the wake
        # word complete, so only once what is asked has started does a stop go to it. The driver hears afresh from
        # the wake, so what is asked starts as any speech does, asked after a pause or in the same breath.
        case Woke(since=since), SpeechStarted():
            return replace(engagement, phase=Talking(since)), ()
        case Woke() as turn, TurnTooLong():
            phase, acts = in_turn(turn, event)
            return replace(engagement, phase=phase), acts
        case Talking() as turn, SpeechStopped() | TurnEnded() | TurnTooLong():
            phase, acts = in_turn(turn, event)
            return replace(engagement, phase=phase), acts
        case _:
            return engagement, ()


# [LAW:one-source-of-truth] switched away from, the desk stops listening and a turn open is thrown away, as engaged.
WAKE = Conversation("trigger.awake", step, released, counts=("heard", "muted"))


async def fetched(models: Path, word: Word, release: str = RELEASE) -> tuple[str, ...]:
    """`word`'s models in `models`, each fetched from `release` unless already there, and whole or not there at all: the
    names of those fetched. A word the user trained is never fetched; raises FileNotFoundError where its model is not."""
    match word:
        case Pretrained(phrase=phrase):
            own: tuple[str, ...] = (PRETRAINED[phrase],)
        case Trained(model=model):
            if not model.is_file():
                raise FileNotFoundError(f"no wake word model at {model}")
            own = ()
    missing = tuple(name for name in (MELSPECTROGRAM, EMBEDDING, *own) if not (models / name).exists())
    models.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=FETCH_SECONDS), raise_for_status=True) as http:
        for name in missing:
            async with http.get(f"{release}/{name}") as response:
                partial = models / f"{name}.partial"
                partial.write_bytes(await response.read())
            # [LAW:one-source-of-truth] a file under its own name is the whole of it, so one there is never fetched again.
            partial.replace(models / name)
    return missing


class WakeWord:
    """A wake word's model, run locally on the CPU with ONNX Runtime, on 16 kHz mono audio, from the files `fetched` put
    in `models` and, for a word the user trained, from theirs."""

    def __init__(self, models: Path, word: Word) -> None:
        heard = model_of(word, models)
        # The model's score is given under its file's name.
        self._name = heard.stem
        self._model = Model(
            wakeword_models=[str(heard)], melspec_model_path=str(models / MELSPECTROGRAM), embedding_model_path=str(models / EMBEDDING), inference_framework="onnx"
        )

    def score(self, audio: bytes) -> float:
        """How sure the model is that the wake word has just been said, from 0 to 1, with `audio` heard last."""
        # Without `timing`, each model's score by the model's name.
        scores = cast(dict[str, float], self._model.predict(np.frombuffer(audio, dtype=np.int16)))  # pyright: ignore[reportUnknownMemberType]  (untyped in openWakeWord)
        return float(scores[self._name])

    def reset(self) -> None:
        """Forget the audio heard so far, so one saying of the wake word wakes hands once."""
        self._model.reset()


@asynccontextmanager
async def loaded(sample_rate: int, models: Path, word: Word, emit: Callable[[WideEvent], None]) -> AsyncGenerator[WakeWord]:
    """`word`'s model, loaded off the loop as its own unit of work."""
    with unit("trigger.wake_word_loaded", emit):
        # [LAW:nothing-unseen] which word the desk listens for, and the file it is heard with.
        annotate(phrase=word.phrase, model=str(model_of(word, models)))
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"the wake word is heard at {SAMPLE_RATE} Hz, and the desk's microphone runs at {sample_rate} Hz")
        heard = await asyncio.to_thread(WakeWord, models, word)
    yield heard


def listening(word: WakeWord, speaking: Callable[[], bool], emit: Callable[[WideEvent], None]) -> Callable[[bytes], Awaitable[bool]]:
    """Whether each buffer of the desk's microphone ends with the wake word, deaf while hands speaks."""

    async def woken(audio: bytes) -> bool:
        # [LAW:dataflow-not-control-flow] the detector hears every buffer; while hands speaks, what it hears is silence.
        muted = speaking()
        count(muted=int(muted))
        score = await asyncio.to_thread(word.score, bytes(len(audio)) if muted else audio)
        if score < THRESHOLD:
            return False
        # Off the loop as scoring is: forgetting refills the model with seconds of features.
        await asyncio.to_thread(word.reset)
        count(heard=1)
        # [LAW:nothing-unseen] each hearing is its own event, with how sure the model was; whether it opened a turn is
        # the engagement's to say, as its count of starts.
        with unit("trigger.wake_word_heard", emit):
            annotate(score=score, threshold=THRESHOLD)
        return True

    return woken
