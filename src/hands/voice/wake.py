"""The wake word: saying "Hey Jarvis" opens a turn with no hand on the key, and end-of-turn detection closes it, as in an
engaged conversation (`hands.voice.engaged`), whose driver and models this edge shares.

openWakeWord's pretrained "hey jarvis" model listens to the desk's microphone through the echo canceller. It is
half-duplex: while hands speaks the detector is given silence in place of the room, because a microphone open in a room
with speakers hears the pipeline's own voice. The desk listens for as long as the trigger is in use, so Whisper keeps the
second before each turn opens and the wake word, and anything said while the detector made sure of it, are in the turn.
"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import cast

import numpy as np
from openwakeword.model import Model
from openwakeword.utils import download_models

from hands.sessions.wide import WideEvent, annotate, count, unit
from hands.voice.engaged import Act, Begun, Conversation, Disengaged, Engagement, Event, Listening, SpeechStarted, SpeechStopped, Talking, TurnEnded, TurnTooLong, Woke, Woken, in_turn, released

# The pretrained model's name in openWakeWord: the name it is fetched and loaded by, and its score is given under.
MODEL = "hey_jarvis"
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
        # [LAW:types-are-the-program] the pause after "Hey Jarvis," is no end of the turn: Smart Turn judges "Hey
        # Jarvis" complete, so only once what is asked has started does a stop go to it. With no pause, the silence after
        # the whole of it ends the turn.
        case Woke(since=since), SpeechStarted():
            return replace(engagement, phase=Talking(since)), ()
        case Woke() as turn, TurnEnded() | TurnTooLong():
            phase, acts = in_turn(turn, event)
            return replace(engagement, phase=phase), acts
        case Talking() as turn, SpeechStopped() | TurnEnded() | TurnTooLong():
            phase, acts = in_turn(turn, event)
            return replace(engagement, phase=phase), acts
        case _:
            return engagement, ()


# [LAW:one-source-of-truth] switched away from, the desk stops listening and a turn open is thrown away, as engaged.
WAKE = Conversation("trigger.awake", step, released, counts=("woken", "muted"))


class WakeWord:
    """openWakeWord's "hey jarvis" model, run locally on the CPU with ONNX Runtime, on 16 kHz mono audio."""

    def __init__(self) -> None:
        # Fetched into openWakeWord's own models directory the first time, where `Model` looks for it by name; a model
        # already there is not fetched again.
        download_models(model_names=[MODEL])
        self._model = Model(wakeword_models=[MODEL], inference_framework="onnx")

    def score(self, audio: bytes) -> float:
        """How sure the model is that the wake word has just been said, from 0 to 1, with `audio` heard last."""
        # Without `timing`, each model's score by the model's name.
        scores = cast(dict[str, float], self._model.predict(np.frombuffer(audio, dtype=np.int16)))  # pyright: ignore[reportUnknownMemberType]  (untyped in openWakeWord)
        return float(scores[MODEL])

    def reset(self) -> None:
        """Forget the audio heard so far, so one saying of the wake word wakes hands once."""
        self._model.reset()


@asynccontextmanager
async def loaded(sample_rate: int, emit: Callable[[WideEvent], None]) -> AsyncGenerator[WakeWord]:
    """The wake word's model, fetched and loaded off the loop as its own unit of work."""
    with unit("trigger.wake_word_loaded", emit):
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"the wake word is heard at {SAMPLE_RATE} Hz, and the desk's microphone runs at {sample_rate} Hz")
        word = await asyncio.to_thread(WakeWord)
    yield word


def listening(word: WakeWord, speaking: Callable[[], bool], emit: Callable[[WideEvent], None]) -> Callable[[bytes], Awaitable[bool]]:
    """Whether each buffer of the desk's microphone ends with the wake word, deaf while hands speaks."""

    async def woken(audio: bytes) -> bool:
        # [LAW:dataflow-not-control-flow] the detector hears every buffer; while hands speaks, what it hears is silence.
        muted = speaking()
        count(muted=int(muted))
        score = await asyncio.to_thread(word.score, bytes(len(audio)) if muted else audio)
        if score < THRESHOLD:
            return False
        word.reset()
        count(woken=1)
        # [LAW:nothing-unseen] each waking is its own event, with how sure the model was.
        with unit("trigger.woken", emit):
            annotate(score=score, threshold=THRESHOLD)
        return True

    return woken
