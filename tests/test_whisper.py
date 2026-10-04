"""Whisper on MLX: each hold transcribed in hands' own process, with mlx_whisper's transcription standing in for the
model, and what hands takes as said from the segments it answers with."""

import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from io import BytesIO

import mlx_whisper
import numpy as np
import pytest
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame

from conftest import events
from hands.sessions.audit import Entry, HoldHeard, Levels, Unsaid
from hands.voice import transcription
from hands.voice.turnstop import TurnResolved
from hands.voice.whisper import Whisper


@dataclass
class Model:
    """mlx_whisper's transcription as these tests answer it: what it was asked, and the segments it gives, oldest first."""

    answers: list[list[tuple[str, float, float]] | Exception] = field(default_factory=list[list[tuple[str, float, float]] | Exception])
    asked: list[tuple[np.ndarray, dict[str, object]]] = field(default_factory=list[tuple[np.ndarray, dict[str, object]]])

    def transcribe(self, audio: np.ndarray, **options: object) -> dict[str, object]:
        self.asked.append((audio, options))
        match self.answers.pop(0) if self.answers else []:
            case Exception() as error:
                raise error
            case segments:
                return {"segments": [{"text": text, "compression_ratio": ratio, "avg_logprob": logprob} for text, ratio, logprob in segments]}


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> Model:
    model = Model()
    monkeypatch.setattr(mlx_whisper, "transcribe", model.transcribe)
    return model


def primed(*prompts: str | None) -> Callable[[], Awaitable[str | None]]:
    remaining = iter(prompts)

    async def prompt() -> str | None:
        return next(remaining)

    return prompt


def wav(samples: bytes, rate: int = 16_000) -> bytes:
    """A hold as Pipecat hands it to run_stt: mono 16-bit WAV at the pipeline's input rate, 16 kHz unless said."""
    out = BytesIO()
    with wave.open(out, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(samples)
    return out.getvalue()


SILENCE = wav(b"\x00\x00" * 16_000)

# How loud each hold queued here was, as the key's release measured it.
LEVELS = Levels(captured_dbfs=-12.5, heard_dbfs=-40.0)


async def transcribe(whisper: Whisper, hold: int, audio: bytes) -> list[Frame]:
    whisper._transcribing.append((hold, LEVELS))  # pyright: ignore[reportPrivateUsage]  (the hold a release queues)
    return [frame async for frame in whisper.run_stt(audio)]


async def test_whisper_has_loaded_the_model_its_holds_transcribe_with_once_built(model: Model) -> None:
    """MLX Whisper keeps the model it loaded for the process, keyed on what it was asked for, so the first hold pays for
    no load only if the one done at construction asked for exactly what a hold's transcription asks for."""
    recorded: list[Entry] = []
    whisper = Whisper(prompt=primed(None), record=recorded.append)
    await transcribe(whisper, 1, SILENCE)

    [(_, loaded), (_, held)] = model.asked
    assert loaded == held == {"path_or_hf_repo": transcription.MODEL, "temperature": 0.0, "language": "en", "initial_prompt": None}
    # The load is a unit of work of its own, saying which model the start waited on.
    [event] = events(recorded, "whisper.loaded")
    assert event.outcome == "ok" and event.facts["model"] == transcription.MODEL


async def test_each_hold_is_transcribed_from_its_samples_primed_with_the_vocabulary_as_it_is_then(model: Model) -> None:
    whisper = Whisper(prompt=primed("authMiddleware", None, "sessionStore"), record=lambda _: None)
    samples = np.array([0, 16384, -16384, 32767], dtype=np.int16)

    for hold in (1, 2, 3):
        await transcribe(whisper, hold, wav(samples.tobytes()))

    held = model.asked[1:]
    assert all(np.array_equal(audio, samples.astype(np.float32) / 32768.0) for audio, _ in held)
    assert [options["initial_prompt"] for _, options in held] == ["authMiddleware", None, "sessionStore"]


async def test_what_a_primed_whisper_makes_of_noise_is_not_said(model: Model) -> None:
    # What Whisper answered, primed: room noise, Gaussian noise read as a guess and as a loop, and "okay" said quietly
    # at -25 dBFS (hands-dictation-2bs.1xq, hands-dictation-d7i).
    model.answers += [
        [],
        [],
        [(".", 0.11, -0.5)],
        [("and slow-talking.", 0.68, -2.84)],
        [("and turnstop, " * 12, 17.1, -0.24)],
        [(" Okay.", 0.38, -1.11)],
    ]
    recorded: list[Entry] = []
    whisper = Whisper(prompt=primed(*["authMiddleware"] * 5), record=recorded.append)

    said = [frame.text for hold in range(1, 6) for frame in await transcribe(whisper, hold, SILENCE) if isinstance(frame, TranscriptionFrame)]

    assert said == ["Okay."]
    # Each hold is recorded with what was dropped from it and why, so a hold that sent nothing can be looked into.
    assert [entry for entry in recorded if isinstance(entry, HoldHeard)] == [
        HoldHeard(1, None, (), LEVELS),
        HoldHeard(2, None, (Unsaid(".", 0.11, -0.5),), LEVELS),
        HoldHeard(3, None, (Unsaid("and slow-talking.", 0.68, -2.84),), LEVELS),
        HoldHeard(4, None, (Unsaid(("and turnstop, " * 12).strip(), 17.1, -0.24),), LEVELS),
        HoldHeard(5, "Okay.", (), LEVELS),
    ]


async def test_a_hold_the_model_fails_is_said_as_an_error_and_whisper_transcribes_the_next(model: Model) -> None:
    model.answers += [[], RuntimeError("Metal ran out of memory"), [("Okay.", 0.38, -0.5)]]
    recorded: list[Entry] = []
    whisper = Whisper(prompt=primed(None, None), record=recorded.append)

    failed = await transcribe(whisper, 7, SILENCE)
    [error] = [frame for frame in failed if isinstance(frame, ErrorFrame)]
    await whisper.push_error_frame(error)
    after = await transcribe(whisper, 8, SILENCE)

    assert [type(frame) for frame in failed] == [ErrorFrame, TurnResolved]
    assert "hold 7" in error.error and "Metal ran out of memory" in error.error
    # A permanent error would make Whisper unusable, and Pipecat then gives it no hold to transcribe or fail aloud.
    assert error.category is not None and not error.category.is_permanent and whisper.is_usable
    assert [frame.text for frame in after if isinstance(frame, TranscriptionFrame)] == ["Okay."]
    assert [entry.hold for entry in recorded if isinstance(entry, HoldHeard)] == [8]


async def test_a_hold_at_a_rate_whisper_does_not_hear_is_said_as_an_error_and_never_transcribed(model: Model) -> None:
    recorded: list[Entry] = []
    whisper = Whisper(prompt=primed(None), record=recorded.append)
    loaded = len(model.asked)

    failed = await transcribe(whisper, 3, wav(b"\x00\x00" * 48_000, rate=48_000))

    [error] = [frame for frame in failed if isinstance(frame, ErrorFrame)]
    assert "48000 Hz" in error.error and len(model.asked) == loaded
