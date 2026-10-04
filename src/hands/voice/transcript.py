"""The transcript: the conversation in words, line by line, as it happens at the speaker and the microphone.

What the user said is the words Whisper heard of it. What hands said is each sentence as it starts to play and as it
plays to its end or is cut off: the output transport lets a sentence's text go only once the audio ahead of it is
written, and its end only once its own audio is (see hands.voice.player). pocket-tts reports no word timings, so a
sentence is the finest step told; whoever shows it sweeps its words across the time it takes.
"""

from collections.abc import Callable
from dataclasses import dataclass

from pipecat.frames.frames import AggregatedTextFrame, InterruptionFrame, TranscriptionFrame, TTSTextFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameProcessor


@dataclass(frozen=True)
class Heard:
    """The user's words, as Whisper heard one hold of them."""

    text: str


@dataclass(frozen=True)
class Saying:
    """A sentence of hands' starting to play."""

    text: str


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

    async def on_push_frame(self, data: FramePushed) -> None:
        match data.frame:
            case TranscriptionFrame(text=text) if data.source is self._stt:
                self._told(Heard(text))
            case TTSTextFrame() as frame if data.source is self._output and frame.will_be_spoken:
                # A sentence's audio made is a kind of AggregatedTextFrame too, so it is taken first.
                self._told(Spoken())
            case AggregatedTextFrame(text=text) as frame if data.source is self._output and frame.will_be_spoken:
                self._told(Saying(text))
            case InterruptionFrame() if data.source is self._output:
                self._told(Cut())
            case _:
                pass
