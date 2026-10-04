"""The transcript: the conversation in words, line by line, as it happens at the speaker and the microphone.

What the user said is the words Whisper heard of it. What hands said is each sentence as its first audio is written and
as it plays to its end or is cut off: the output transport lets a sentence's text go once the audio ahead of it is
written, which for the first sentence of a reply is before its own audio is made, then its audio, then its end (see
hands.voice.player). pocket-tts reports no word timings, so a sentence is the finest step told; whoever shows it sweeps
its words across the time it takes, at the speaking rate measured here off the sentences played to their end.
"""

from collections.abc import Callable
from dataclasses import dataclass

from pipecat.frames.frames import AggregatedTextFrame, InterruptionFrame, OutputAudioRawFrame, TranscriptionFrame, TTSTextFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameProcessor


@dataclass(frozen=True)
class Heard:
    """The user's words, as Whisper heard one hold of them."""

    text: str


@dataclass(frozen=True)
class Saying:
    """A sentence of hands' starting to play, and hands' speaking rate, in characters a second."""

    text: str
    chars_per_sec: float


@dataclass(frozen=True)
class Spoken:
    """The sentence playing has played to its end."""


@dataclass(frozen=True)
class Cut:
    """The user cut in: what was playing stopped where it was."""


Line = Heard | Saying | Spoken | Cut


class TranscriptObserver(BaseObserver):
    """Tells `told` each line of the transcript, read off what `stt`, Whisper, and `output`, the output transport, push.

    Every frame is observed once per processor boundary it crosses, so each line is read off the one processor that
    makes it: the user's words where Whisper pushes them, hands' where the output transport lets them go.
    """

    def __init__(self, stt: FrameProcessor, output: FrameProcessor, told: Callable[[Line], None]) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._stt = stt
        self._output = output
        self._told = told
        # Measured off each sentence played to its end, half from the latest; a typical voice's until the first has been.
        self._chars_per_sec = 15.0
        # The sentence let go whose audio has not begun, and the one playing with the seconds of its audio written.
        self._next: str | None = None
        self._playing: tuple[str, float] | None = None

    async def on_push_frame(self, data: FramePushed) -> None:
        match data.frame:
            case TranscriptionFrame(text=text) if data.source is self._stt:
                self._told(Heard(text))
            case TTSTextFrame() as frame if data.source is self._output and frame.will_be_spoken:
                # A sentence's audio made is a kind of AggregatedTextFrame too, so it is taken first.
                match self._playing:
                    case (text, lasted):
                        self._chars_per_sec = (self._chars_per_sec + len(text.strip()) / lasted) / 2
                    case None:
                        pass
                self._next = self._playing = None
                self._told(Spoken())
            case AggregatedTextFrame(text=text) as frame if data.source is self._output and frame.will_be_spoken:
                self._next = text
            case OutputAudioRawFrame() as frame if data.source is self._output:
                # A sentence starts with its first audio written, and lasts as long as all of it.
                if self._next is not None:
                    self._told(Saying(self._next, self._chars_per_sec))
                    self._playing, self._next = (self._next, 0.0), None
                match self._playing:
                    case (text, lasted):
                        self._playing = (text, lasted + frame.num_frames / frame.sample_rate)
                    case None:
                        pass
            case InterruptionFrame() if data.source is self._output:
                # A sentence let go in the moment of the barge-in never plays: forgotten with the one cut off.
                self._next = self._playing = None
                self._told(Cut())
            case _:
                pass
