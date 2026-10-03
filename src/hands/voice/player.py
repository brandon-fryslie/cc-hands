"""The player: what is on the speaker, read off the frames the TTS service and the output transport push, and what is
said again on request.

Pipecat's TTS hands each sentence on as an `AggregatedTextFrame` marked to be spoken, ahead of its audio, and a
`TTSTextFrame` once the sentence's audio is made (pocket-tts reports no word timings, so these are per sentence). The
output transport queues both with the audio and lets each go downstream only once the audio ahead of it is written to
the device (Pipecat 1.10.0, `BaseOutputTransport.handle_sync_frame`). So what the TTS service pushes is every sentence
the speaker was given, as soon as it is made; what the output transport pushes is a `TTSTextFrame` for a sentence that
has played to its end, and a barge-in after every sentence that finished before it. An observer sees each push as it
is made, in the order made, where a processor standing in the pipeline would take a barge-in ahead of a sentence's end
still waiting in its queue.

[LAW:single-enforcer] the one holder of `Playback`: its observer reports to it, and the playback tools act on it.
"""

from collections.abc import Callable

from pipecat.frames.frames import AggregatedTextFrame, Frame, InterruptionFrame, TTSSpeakFrame, TTSTextFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameProcessor

from hands.core import playback
from hands.core.playback import LastOne, NothingCut, NothingSaid, Played, Playback, Replay
from hands.sessions.audit import CutOff, Record

Act = Callable[[Playback], tuple[Playback, Played]]


class Player:
    """Where playback is, and the lines to say again, which `lines` hands the TTS service the moment they are asked for.

    The lines go straight to the speaker rather than through the model's stage: the call that asked for them is the
    whole reply, and a barge-in drops them from the pipeline like any sentence not yet played, while their reading holds
    them to go back to.
    """

    def __init__(self, record: Record) -> None:
        self._record = record
        self._playback = Playback()
        # Stood ahead of the TTS service, the one place the lines enter the pipeline; it passes everything else on.
        self.lines = IdentityFilter()

    def watching(self, speaker: FrameProcessor, output: FrameProcessor) -> BaseObserver:
        """The observer reading the speaker off what `speaker`, the TTS service, and `output`, the output transport, push."""
        return _Watch(self, speaker, output)

    def handed(self, frame: Frame) -> None:
        match frame:
            case TTSTextFrame():
                # A sentence's audio made, which is a kind of AggregatedTextFrame too: it has played only once the
                # output transport pushes it.
                pass
            case AggregatedTextFrame(text=text) if frame.will_be_spoken:
                self._playback = playback.handed(self._playback, text)
            case _:
                pass

    def played(self, frame: Frame) -> None:
        match frame:
            case TTSTextFrame() if frame.will_be_spoken:
                self._playback = playback.finished(self._playback)
            case InterruptionFrame():
                was = self._playback
                self._playback = playback.cut(was)
                # [LAW:nothing-unseen] every barge-in is a line, one that cut nothing off included.
                self._record(CutOff(None if was.over else (*was.reading, *was.coming)[was.played], len(self._playback.bookmarks)))
            case _:
                pass

    async def act(self, act: Act) -> tuple[str, ...]:
        """Moves playback by `act`, and says what it says, which is always something: a reading said again, or why not.
        What it says is held whole from the start, so a barge-in on it loses none of it."""
        moved, played = act(self._playback)
        lines = said(played)
        self._playback = playback.queued(moved, lines)
        for sentence in lines:
            # One sentence a frame, so each is a position.
            await self.lines.push_frame(TTSSpeakFrame(sentence))
        return lines


def said(played: Played) -> tuple[str, ...]:
    """What an act has hands say: the sentences again, or in one sentence why there are none."""
    match played:
        case Replay(sentences=sentences):
            return sentences
        case NothingCut():
            return ("Nothing was cut off to go back to.",)
        case NothingSaid():
            return ("I haven't said anything yet.",)
        case LastOne():
            return ("That was the last of it.",)


class _Watch(BaseObserver):
    """Shows the player what the speaker is handed and what it has played; every other push is one of those frames
    passing another processor, and tells it nothing."""

    def __init__(self, player: Player, speaker: FrameProcessor, output: FrameProcessor) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._player = player
        self._speaker = speaker
        self._output = output

    async def on_push_frame(self, data: FramePushed) -> None:
        if data.source is self._speaker:
            self._player.handed(data.frame)
        elif data.source is self._output:
            self._player.played(data.frame)
