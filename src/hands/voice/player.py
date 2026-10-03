"""The player: what is on the speaker, read off the pipeline on either side of it, and what is said again on request.

Pipecat's TTS hands each sentence on as an `AggregatedTextFrame` marked to be spoken, ahead of its audio, and a
`TTSTextFrame` once the sentence's audio is made (pocket-tts reports no word timings, so these are per sentence). The
output transport queues both with the audio and lets each go downstream only once the audio ahead of it is written to
the device (Pipecat 1.10.0, `BaseOutputTransport.handle_sync_frame`). So ahead of the transport, every sentence the
speaker was given is seen as soon as it is made; past it, a `TTSTextFrame` is a sentence that has played to its end,
and a barge-in is seen there after every sentence that finished before it.

[LAW:single-enforcer] the one holder of `Playback`: the two taps report to it, and the playback tools act on it.
"""

import asyncio
from collections.abc import Awaitable, Callable

from pipecat.frames.frames import AggregatedTextFrame, Frame, InterruptionFrame, TTSSpeakFrame, TTSTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from hands.core import playback
from hands.core.playback import LastOne, NothingCut, NothingSaid, Played, Playback, Replay
from hands.sessions.audit import CutOff, Record
from hands.voice.speech import Telling, as_written

Act = Callable[[Playback], tuple[Playback, Played]]


class Player:
    """Where playback is, and the lines to say again, which are said in order as hands says a line as written."""

    def __init__(self, record: Record) -> None:
        self._record = record
        self._playback = Playback()
        self._plays: asyncio.Queue[tuple[str, ...]] = asyncio.Queue()
        self.handing = _Tap(self._handed)
        self.playing = _Tap(self._played)

    def _handed(self, frame: Frame) -> None:
        match frame:
            case TTSTextFrame():
                # A sentence's end, which is a kind of AggregatedTextFrame too: the playing tap reads it, past the speaker.
                pass
            case AggregatedTextFrame(text=text) if frame.will_be_spoken:
                self._playback = playback.handed(self._playback, text)
            case _:
                pass

    def _played(self, frame: Frame) -> None:
        match frame:
            case TTSTextFrame() if frame.will_be_spoken:
                self._playback = playback.finished(self._playback)
            case InterruptionFrame():
                was = self._playback
                self._playback = playback.cut(was)
                # [LAW:nothing-unseen] every barge-in is a line, one that cut nothing off included.
                self._record(CutOff(None if was.over else was.reading[was.played], len(self._playback.interrupted)))
            case _:
                pass

    def act(self, act: Act) -> tuple[str, ...]:
        """Moves playback by `act`, and queues what it says, which is always something: a reading said again, or why not."""
        self._playback, played = act(self._playback)
        lines = said(played)
        self._plays.put_nowait(lines)
        return lines

    async def keep_playing(self, telling: Telling, queue_frame: Callable[[Frame], Awaitable[None]]) -> None:
        """Says each act's lines as hands says a line as written, one sentence a frame so each is a position, until cancelled."""
        while True:
            for sentence in await self._plays.get():
                await queue_frame(as_written(TTSSpeakFrame(sentence), telling))


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


class _Tap(FrameProcessor):
    """Shows the player every frame passing one point of the pipeline, and passes it on untouched."""

    def __init__(self, seen: Callable[[Frame], None]) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._seen = seen

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        self._seen(frame)
        await self.push_frame(frame, direction)
