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

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace

from pipecat.frames.frames import (
    AggregatedTextFrame,
    DataFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from hands.core import playback
from hands.core.playback import Bookmark, LastOne, NothingCut, NothingSaid, Played, Playback, Replay
from hands.sessions.audit import CutOff, Record, WentOn
from hands.voice.trigger import Edge, voice_activated
from hands.voice.turnstop import HoldDiscarded

Act = Callable[[Playback], tuple[Playback, Played]]


@dataclass(frozen=True)
class _Turn:
    """A turn of the user's, open: the edge that opened it, the reading it cut off, None while it has cut none off, and
    whether Whisper has found nothing said in it so far, neither words nor a hold it failed on or threw away."""

    by: Edge
    cut: Bookmark | None = None
    silent: bool = True


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
        # [LAW:no-shared-mutable-globals] each `heard` waiting on its line, by the mark behind it: settled by the output
        # passing the mark on, or by the next barge-in, whichever the output pushes first.
        self._hearing: dict[asyncio.Future[bool], Mark] = {}
        # The user's turn open, read off the output's pushes as the speaker is; None between turns.
        self._turn: _Turn | None = None

    @property
    def waiting(self) -> int:
        """How many readings cut off wait to be gone back to."""
        return len(self._playback.bookmarks)

    def watching(self, hearer: FrameProcessor, speaker: FrameProcessor, output: FrameProcessor, opened: Callable[[], Edge]) -> BaseObserver:
        """The observer reading the speaker off what `speaker`, the TTS service, and `output`, the output transport, push,
        and the user's turns off what `hearer`, Whisper, pushes, each with the edge `opened` says opened it."""
        return _Watch(self, hearer, speaker, output, opened)

    def opened(self, by: Edge) -> None:
        """The user's turn opened, by `by`."""
        self._turn = _Turn(by)

    def transcribed(self, frame: Frame) -> None:
        match self._turn, frame:
            case _Turn() as turn, TranscriptionFrame() | ErrorFrame() | HoldDiscarded():
                self._turn = replace(turn, silent=False)
            case _:
                pass

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

    async def played(self, frame: Frame) -> None:
        match frame:
            case TTSTextFrame() if frame.will_be_spoken:
                self._playback = playback.finished(self._playback)
            case Mark():
                # [LAW:no-ambient-temporal-coupling] read off the output's push, in order with its barge-ins: a mark
                # waiting in Marks' queue is one a barge-in empties out, though its line played to the end.
                for hearing, mark in self._hearing.items():
                    if mark is frame:
                        _settle(hearing, True)
            case InterruptionFrame():
                for hearing in self._hearing:
                    _settle(hearing, False)
                was = self._playback
                self._playback = playback.cut(was)
                # [LAW:nothing-unseen] every barge-in is a line, one that cut nothing off included.
                self._record(CutOff(None if was.over else (*was.reading, *was.coming)[was.played], len(self._playback.bookmarks)))
                match self._turn:
                    case _Turn() as turn if not was.over:
                        self._turn = replace(turn, cut=self._playback.stopped)
                    case _:
                        pass
            case UserStoppedSpeakingFrame():
                turn, self._turn = self._turn, None
                match turn:
                    # The voice opened the turn, it cut a reading off, and Whisper found nothing said in it: it was the
                    # room, or hands' own reply coming back through the microphone, and nobody cut in. Only while that
                    # reading is still the one stopped: one that has begun since is what the user hears now.
                    case _Turn(by=by, cut=Bookmark() as cut, silent=True) if voice_activated(by) and cut is self._playback.stopped:
                        lines = await self.act(playback.went_on)
                        # [LAW:nothing-unseen] every reading that went on is a line, as every barge-in is.
                        self._record(WentOn(lines[0], len(self._playback.bookmarks)))
                    case _:
                        pass
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

    async def heard(self, sentence: str) -> bool:
        """Says `sentence`, and returns once the speaker has played it to its end, True, or a barge-in has cut it off, False.

        [LAW:no-ambient-temporal-coupling] what follows a line only once the user has heard it waits on this, never on how
        long the line takes to say.
        """
        hearing = asyncio.get_running_loop().create_future()
        # Settled as the output passes it on, which the observer sees (`played`); Marks only takes it out of the pipeline.
        mark = Mark(lambda: None)
        self._hearing[hearing] = mark
        try:
            await self.lines.push_frame(TTSSpeakFrame(sentence, append_to_context=False))
            await self.lines.push_frame(mark)
            return await hearing
        finally:
            del self._hearing[hearing]


def _settle(hearing: "asyncio.Future[bool]", played: bool) -> None:
    # The first of a line's mark and a barge-in says how it went; the output passes a mark on only if no barge-in cut it off.
    if not hearing.done():
        hearing.set_result(played)


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


@dataclass
class Mark(DataFrame):
    """A point in what is said: `played` is called once the speaker has played everything handed it ahead of the mark, and
    never if a barge-in cut any of that off. The TTS service keeps a data frame behind the audio of what was handed it
    first, and the output transport passes it on only once that audio is written, or drops it with that audio on a barge-in
    (Pipecat 1.10.0)."""

    played: Callable[[], None]


class Marks(FrameProcessor):
    """Stood right behind the output transport: calls each mark's `played` as it arrives, and passes the rest on."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case Mark(played=played):
                played()
            case _:
                await self.push_frame(frame, direction)


class _Watch(BaseObserver):
    """Shows the player what the speaker is handed and what it has played, and what Whisper made of the user's turn; every
    other push is one of those frames passing another processor, and tells it nothing."""

    def __init__(self, player: Player, hearer: FrameProcessor, speaker: FrameProcessor, output: FrameProcessor, opened: Callable[[], Edge]) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._player = player
        self._hearer = hearer
        self._speaker = speaker
        self._output = output
        self._opened = opened

    async def on_push_frame(self, data: FramePushed) -> None:
        if data.source is self._hearer:
            self._player.transcribed(data.frame)
        elif data.source is self._speaker:
            self._player.handed(data.frame)
        elif data.source is self._output:
            match data.frame:
                # The user aggregator pushes a turn's start ahead of the interruption it broadcasts, so the turn is open
                # before anything it cuts off; the edge that opened it has moved the gate before either.
                case UserStartedSpeakingFrame():
                    self._player.opened(self._opened())
                case frame:
                    await self._player.played(frame)
