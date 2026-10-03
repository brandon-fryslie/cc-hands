"""Acoustic echo cancellation: the speaker's sound taken back out of what the Mac's microphone hears.

With speakers on, the room carries every reply to the microphone. WebRTC's echo canceller (AEC3, through LiveKit's
binding of its audio processing module) is told what the speaker plays, the far-end or reference signal, and subtracts
the echo of it from every microphone buffer, the near-end signal, estimating the delay and the room's response as it
goes. Measured 2026-10-03 on MacBook Pro speakers and microphone, through hands' own transport: about 27 dB of the
echo removed; a press mid-reply with nobody speaking left no word of the reply in 8 holds of 8, where the raw
microphone made one of it ("Run.", "in the") in every hold; a voice starting 50 ms after the press kept its first word
in 8 of 8, where muting the microphone until the reply's sound had died away lost it in 8 of 8.
"""

import threading
from collections import deque
from typing import Protocol

from livekit import rtc

# The canceller works on 10 ms frames, in both directions.
_FRAMES_PER_SECOND = 100

_SAMPLE_BYTES = 2  # 16-bit, as the transport opens both streams

# The most reference held for the microphone to take, in 10 ms frames. The speaker writes about one output buffer
# (138 ms on a MacBook's speakers) ahead of what it plays; anything older than this can never be matched to an echo,
# and is only there because the microphone has stopped taking frames: a Mac with no input device.
_HELD_FRAMES = 100


class Echo(Protocol):
    """What the transport asks of an echo canceller: told what the speaker plays, it cleans what the microphone hears."""

    def played(self, audio: bytes, sample_rate: int, channels: int) -> None: ...
    def heard(self, audio: bytes, sample_rate: int) -> bytes: ...
    def close(self) -> None: ...


class EchoCanceller:
    """WebRTC's AEC3, with its reference paced by the microphone's clock.

    AEC3 expects one frame of reference for every frame of microphone, as a device that plays and records at once
    gives them: when nothing has been written, the speaker plays silence. The pipeline writes only while it speaks,
    and up to an output buffer ahead, so what it writes is held, and each microphone frame takes the next frame of it,
    or silence when there is none. Held audio left at an interruption is taken as the device plays it out.
    """

    def __init__(self) -> None:
        self._apm = rtc.AudioProcessingModule(echo_cancellation=True)
        # The speaker's audio short of a whole frame, carried to its next chunk: a cue is any length. The speaker's
        # writer thread alone touches it.
        self._unplayed = b""
        # [LAW:no-shared-mutable-globals] written by the speaker's writer thread, taken by the microphone's capture
        # thread, under the one lock; a ring buffer, so the oldest goes first when nothing takes it.
        self._held: deque[rtc.AudioFrame] = deque(maxlen=_HELD_FRAMES)
        self._holding = threading.Lock()
        # The reference's format, from the speaker's first write; silence is made in it until then.
        self._format = (16000, 1)

    def played(self, audio: bytes, sample_rate: int, channels: int) -> None:
        """Hold sound as it goes to the speaker, silence included, for the microphone frames that hear its echo."""
        size = sample_rate // _FRAMES_PER_SECOND * channels * _SAMPLE_BYTES
        pending = self._unplayed + audio
        whole = len(pending) - len(pending) % size
        frames = [rtc.AudioFrame(pending[start : start + size], sample_rate, channels, sample_rate // _FRAMES_PER_SECOND) for start in range(0, whole, size)]
        self._unplayed = pending[whole:]
        with self._holding:
            self._held.extend(frames)
            self._format = (sample_rate, channels)

    def heard(self, audio: bytes, sample_rate: int) -> bytes:
        """A mono microphone buffer with the speaker's echo taken out; the same length, of whole 10 ms frames."""
        size = sample_rate // _FRAMES_PER_SECOND * _SAMPLE_BYTES
        # [LAW:parse-dont-validate] the microphone is opened on 20 ms buffers; any other length is a stream opened wrong.
        if len(audio) % size:
            raise ValueError(f"a microphone buffer of {len(audio)} bytes is not whole 10 ms frames at {sample_rate} Hz")
        cleaned = bytearray()
        for start in range(0, len(audio), size):
            # [LAW:dataflow-not-control-flow] every microphone frame is matched with one frame of reference.
            self._apm.process_reverse_stream(self._next_played())
            frame = rtc.AudioFrame(audio[start : start + size], sample_rate, 1, sample_rate // _FRAMES_PER_SECOND)
            self._apm.process_stream(frame)  # in place
            cleaned += frame.data.cast("B")
        return bytes(cleaned)

    def close(self) -> None:
        """Let go of the native canceller, once the microphone has stopped capturing.

        Left to the garbage collector, it is let go of after LiveKit's own exit handler has shut its runtime down, and
        LiveKit fails an assertion on every exit saying so.
        """
        self._apm._ffi_handle.dispose()  # pyright: ignore[reportPrivateUsage]  (LiveKit gives the module no close of its own)

    def _next_played(self) -> rtc.AudioFrame:
        with self._holding:
            if self._held:
                return self._held.popleft()
            rate, channels = self._format
        return rtc.AudioFrame.create(rate, channels, rate // _FRAMES_PER_SECOND)
