"""The echo canceller between a speaker and a microphone that keep time on clocks of their own, simulated to the sample.

Each device calls back every 20 ms at a phase of its own. The speaker's sound reaches its converter a fixed latency
after the callback that took it, crosses a room, and reaches the microphone's buffers a fixed latency before the callback
that hands them over. Readings are written at whatever moment the pipeline gets to them, with silences between. All of
it runs on one virtual clock, through hands' own playout and WebRTC's AEC3, so a run is the same on every machine.
"""

import heapq
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from hands.voice.echo import Echo, EchoCanceller
from hands.voice.microphone import Playout, Pull

PLAYED_RATE, HEARD_RATE = 24000, 16000
TICKS = 48000  # the clock's ticks a second: a whole number of them to every sample at either rate
PERIOD = TICKS // 50  # 20 ms, both devices' callback period
CHUNK = TICKS // 25  # 40 ms, the pipeline's chunk
# Callback to converter and converter to callback. 76 ms is PortAudio's low-latency output on a MacBook's speakers; with
# 40 ms in, they put a reading's echo about 120 ms behind the sound the canceller is given, short of the 177 ms measured
# live on that Mac, where a blocking stream held 50 ms more.
SPEAKER_LATENCY = 76 * TICKS // 1000
MICROPHONE_LATENCY = 40 * TICKS // 1000
ROOM_DELAY = 48  # microphone samples, 3 ms
NOISE = 33.0  # the room's floor, about -60 dBFS


class Speaker(Protocol):
    """Hands' side of a speaker device: written to by the pipeline, taken from by the device's callback."""

    def write(self, audio: bytes) -> None: ...
    def take(self, frames: int) -> bytes: ...


class Playing:
    """Hands' playout as the device drives it."""

    def __init__(self, echo: Echo) -> None:
        self._pull: Pull | None = None
        self.playout = Playout(self._opened, echo, PLAYED_RATE, 1, _unfailing)

    def _opened(self, pull: Pull) -> "Playing":
        self._pull = pull
        return self

    def write(self, audio: bytes) -> None:
        self.playout.write(audio)

    def take(self, frames: int) -> bytes:
        assert self._pull is not None
        return self._pull(None, frames, {}, 0)[0]

    def start_stream(self) -> None: ...
    def stop_stream(self) -> None: ...
    def close(self) -> None: ...


def _unfailing(error: Exception) -> None:
    raise AssertionError(f"the playout failed: {error}")


def _reading(seconds: float, seed: int) -> bytes:
    """Speech-like sound: noise held below about 4 kHz, in syllables four times a second, about -20 dBFS."""
    count = int(seconds * PLAYED_RATE)
    white = np.random.default_rng(seed).normal(0, 6000, count)
    voiced = np.convolve(white, np.ones(6) / 6, mode="same")
    syllables = 0.55 + 0.45 * np.sin(np.arange(count) * 2 * np.pi * 4 / PLAYED_RATE)
    return (voiced * syllables).clip(-32768, 32767).astype(np.int16).tobytes()


@dataclass(frozen=True)
class Heard:
    """What the microphone captured over the run, before the canceller and after, and where each reading's echo began."""

    raw: np.ndarray
    cleaned: np.ndarray
    onsets: list[int]  # microphone sample at which each reading's first sound arrives


def run_room(speaker: Callable[[Echo], Speaker], seed: int, readings: int = 8, microphone_late: int = 0) -> Heard:
    """Readings with silences between, written at random phases against both devices' callbacks; the microphone starts
    `microphone_late` ticks after the speaker, and the first reading two seconds after the microphone."""
    rng = np.random.default_rng(seed)
    canceller = EchoCanceller()
    playing = speaker(canceller)
    # The pipeline's writes: each reading begins at an arbitrary tick, after 1 to 2 s of silence.
    writes: list[tuple[int, bytes]] = []
    at = 2 * TICKS + microphone_late
    size = CHUNK // 2 * 2  # bytes in a 40 ms chunk: 960 samples at 24 kHz
    for index in range(readings):
        audio = _reading(1.6, seed=seed * 100 + index)
        writes += [(at + CHUNK * n, audio[start : start + size]) for n, start in enumerate(range(0, len(audio), size))]
        at += CHUNK * (len(audio) // size + 1) + int(rng.integers(TICKS, 2 * TICKS))
    end = at + TICKS
    speaker_phase, microphone_phase = int(rng.integers(0, PERIOD // 2)) * 2, int(rng.integers(0, PERIOD // 3)) * 3
    # (tick, order at one tick, what, the chunk a write writes)
    events: list[tuple[int, int, str, bytes]] = [(tick, 0, "write", chunk) for tick, chunk in writes]
    events += [(tick, 1, "speaker", b"") for tick in range(speaker_phase, end, PERIOD)]
    events += [(tick, 2, "microphone", b"") for tick in range(microphone_late + microphone_phase + PERIOD + MICROPHONE_LATENCY, end, PERIOD)]
    heapq.heapify(events)
    played = np.zeros(end // 2 + PLAYED_RATE, np.float64)  # what the speaker's converter plays, at 24 kHz
    heard_count = end // 3
    room = _Room(played, rng)
    raw, cleaned = np.zeros(heard_count, np.int16), np.zeros(heard_count, np.int16)
    while events:
        tick, _, what, chunk = heapq.heappop(events)
        match what:
            case "write":
                playing.write(chunk)
            case "speaker":
                buffer = np.frombuffer(playing.take(PERIOD // 2), np.int16)
                start = (tick + SPEAKER_LATENCY) // 2
                played[start : start + len(buffer)] = buffer
            case "microphone":
                last = (tick - MICROPHONE_LATENCY) // 3  # 16 kHz sample just past the buffer's end
                captured = room.heard(last - PERIOD // 3, last)
                index = last - PERIOD // 3
                raw[index : index + len(captured)] = captured
                cleaned[index : index + len(captured)] = np.frombuffer(canceller.heard(captured.tobytes(), HEARD_RATE, 1), np.int16)
            case _:
                raise AssertionError(what)
    canceller.close()
    # Where each reading's first sample reached the converter, then the microphone a flight later, at 16 kHz.
    starts = [tick // 2 for n, (tick, _) in enumerate(writes) if n == 0 or tick - writes[n - 1][0] > CHUNK]
    onsets = [(start + int(np.flatnonzero(played[start:])[0])) * 2 // 3 + ROOM_DELAY for start in starts]
    return Heard(raw, cleaned, onsets)


class _Room:
    """The speaker's sound at the microphone: a 3 ms flight, then a reverberant tail, at half strength, over a floor."""

    def __init__(self, played: np.ndarray, rng: np.random.Generator) -> None:
        self._played = played
        self._rng = rng
        tail = np.exp(-np.arange(480) / 80.0) * rng.normal(0, 0.3, 480)
        tail[0] = 1.0
        self._response = np.concatenate([np.zeros(ROOM_DELAY), tail]) * 0.5

    def heard(self, start: int, end: int) -> np.ndarray:
        reach = len(self._response)
        at = np.arange(start - reach + 1, end) * PLAYED_RATE / HEARD_RATE
        converted = np.interp(at, np.arange(len(self._played)), self._played, left=0.0)
        echo = np.convolve(converted, self._response, mode="valid")
        return (echo + self._rng.normal(0, NOISE, end - start)).clip(-32768, 32767).astype(np.int16)


def opening_reductions(heard: Heard, seconds: float = 0.4) -> list[float]:
    """dB taken out of each reading's opening: its first sound at the microphone, and the `seconds` after it."""
    span = int(seconds * HEARD_RATE)

    def power(samples: np.ndarray) -> float:
        return float(10 * np.log10(np.mean(samples.astype(np.float64) ** 2) + 1e-9))

    return [power(heard.raw[at : at + span]) - power(heard.cleaned[at : at + span]) for at in heard.onsets]


def test_every_readings_opening_is_cancelled_wherever_its_first_write_falls() -> None:
    """The canceller's reference keeps the speaker device's time through each silence, so a reading's opening meets its
    echo where the last reading's did. Given to the canceller as it was written instead, the reference moved against its
    echo by where the first write fell between the two devices' callbacks, and openings came through."""
    for seed in range(4):
        reductions = opening_reductions(run_room(Playing, seed))
        # About 36 dB on AEC3 today. Told at the write, three of these four rooms each let an opening through at 6 to 18.
        assert min(reductions) > 25, (seed, [round(each, 1) for each in reductions])


def test_a_microphone_started_after_the_speaker_still_meets_the_echo_of_what_it_plays() -> None:
    """PyAudio starts a stream as it opens it, and a reopen opens the speaker first, so the speaker's sound is held for
    the microphone for as long as the microphone takes to open. Held for good, that reference reached AEC3 after its
    echo, and nothing was cancelled at all."""
    for seed in range(2):
        reductions = opening_reductions(run_room(Playing, seed, microphone_late=600 * TICKS // 1000))
        assert min(reductions) > 25, (seed, [round(each, 1) for each in reductions])
