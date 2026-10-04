"""Utterances: each thing a session gives hands to say unasked, followed from the moment hands hears it to its fate, as
one wide event [LAW:nothing-unseen].

What hands hears passes through the relay or the narrator, which route it by what the user set; progress waits on its
summary; the floor holds it through the user's turn, folds it into another or finds it no longer so; and what is let go
is handed to the speaker, or to the model to say in its own words. The utterance is open across all of it, so its one
event says which way it went, what decided it, how long the floor held it, and how long until the user heard it.

Whether it was heard is read off the output transport, by two frames sent with what says it: `Uttering` ahead of it and
`Uttered` behind. `Uttering` is dropped by a barge-in like the words it leads, and `Uttered` is kept through one, so each
utterance's `Uttered` reaches the output transport, in order with the audio ahead of it. A barge-in the brain's turn goes
on through cuts off none of what that turn is still to say: the brain's stage leads the rest with `Resumed`. What is
never said, a note to the model's context, is never sent between them: its fate is `silent` as it is sent. Pushed by the output transport,
`Uttering` means what follows it is this utterance's, the first audio after it is its first audio, and a barge-in
before its `Uttered` cut it off; an `Uttered` with no `Uttering` before it was cut off before any of it played.
"""

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pipecat.frames.frames import DataFrame, Frame, InterruptionFrame, OutputAudioRawFrame, UninterruptibleFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameProcessor

from hands.core.effects import Heard, Story
from hands.core.session import SessionId
from hands.sessions.audit import Record
from hands.sessions.wide import Begun, Fact, annotate, begun, fail, unit

# How an utterance ended. `noted`: what the user set, or a turn or a burst with nothing new in it, kept it from being
# said. `dropped`: no longer so by the time it was to be told: out of date as the floor let it go, or progress of a turn
# that ended before its summary was ready. `silent`: handed on, and nothing of it was heard, as a note to the model's
# context is not. `played`: heard to its end. `cut`: a barge-in stopped it, before or after its first audio.
Fate = Literal["noted", "dropped", "silent", "played", "cut"]


@dataclass(eq=False)
class Utterance:
    """One thing a session gave hands to say unasked, from when it was heard to its fate: the facts each stage it passed
    through found, and why it failed, where a stage failed it and it went on. Its one event is emitted as it is settled."""

    begun: Begun
    facts: dict[str, Fact] = field(default_factory=dict[str, Fact])
    failure: str | None = None
    # When its first audio was written to the speaker, by time.monotonic().
    first_audio: float | None = None
    settled: asyncio.Future[Fate] = field(default_factory=lambda: asyncio.get_running_loop().create_future())

    def annotate(self, **facts: Fact) -> None:
        self.facts.update(facts)

    def fail(self, error: str) -> None:
        self.failure = error

    def settle(self, fate: Fate) -> None:
        """Its fate, settled by the one stage that knows it: settled twice is a bug, refused out loud."""
        self.settled.set_result(fate)


class Utterances:
    """[LAW:single-enforcer] where every utterance is opened and held open until its fate: `heard` begins one as hands
    hears it, and `keep` runs each as a unit of work, emitting its event as it is settled, or as hands stops."""

    def __init__(self, record: Record) -> None:
        self._record = record
        self._heard: asyncio.Queue[Utterance] = asyncio.Queue()

    def heard(self, session: SessionId, heard: Heard | Story) -> Utterance:
        """Something `session` gave hands to say unasked, `heard`, beginning now."""
        utterance = Utterance(begun(), {"session": session, "heard": heard})
        self._heard.put_nowait(utterance)
        return utterance

    async def keep(self) -> None:
        """Hold each utterance heard open as its own unit of work until it is settled, until cancelled: cancelled, each
        still open ends cancelled, so one hands stopped before it was said is still in the record."""
        async with asyncio.TaskGroup() as group:
            while True:
                group.create_task(self._held(await self._heard.get()))

    async def _held(self, utterance: Utterance) -> None:
        with unit("utterance", self._record, began=utterance.begun):
            try:
                fate = await utterance.settled
            finally:
                annotate(**utterance.facts)
                if utterance.first_audio is not None:
                    annotate(first_audio_ms=round((utterance.first_audio - utterance.begun.began) * 1000, 3))
                if utterance.failure is not None:
                    fail(utterance.failure)
            annotate(fate=fate)


@dataclass
class Uttering(DataFrame):
    """Leads what says `utterances`: dropped by a barge-in with it, so one that reaches the output transport says that what
    follows it is theirs."""

    utterances: tuple[Utterance, ...]


@dataclass
class Resumed(Uttering, UninterruptibleFrame):
    """Leads what is still to say `utterances` after a barge-in that left what says them going: kept through that
    barge-in, since none of what it leads was in flight to be dropped with it."""


@dataclass
class Uttered(DataFrame, UninterruptibleFrame):
    """Closes what says `utterances`: kept through a barge-in, so it always reaches the output transport, behind any of
    their audio that played."""

    utterances: tuple[Utterance, ...]


def uttering(utterances: tuple[Utterance, ...], saying: Sequence[Frame]) -> tuple[Frame, ...]:
    """`saying` as it is sent to say `utterances`, with the frames that tell what of it was heard."""
    return (Uttering(utterances), *saying, Uttered(utterances))


class Audible(BaseObserver):
    """Reads each utterance's fate off what the output transport pushes: after its `Uttering`, its first audio, and a
    barge-in before its `Uttered`, which settles it."""

    def __init__(self, output: FrameProcessor, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._output = output
        self._now = clock
        # What is on air, in the order it went: each utterance led on and not yet closed, and whether a barge-in cut it.
        self._on: dict[Utterance, bool] = {}

    async def on_push_frame(self, data: FramePushed) -> None:
        if data.source is not self._output:
            return
        match data.frame:
            case Uttering(utterances=utterances):
                self._on.update(dict.fromkeys(utterances, False))
            case OutputAudioRawFrame():
                for utterance in self._on:
                    if utterance.first_audio is None:
                        utterance.first_audio = self._now()
            case InterruptionFrame():
                self._on = dict.fromkeys(self._on, True)
            case Uttered(utterances=utterances):
                for utterance in utterances:
                    utterance.settle(_fate(self._on.pop(utterance, None), utterance.first_audio is not None))
            case _:
                pass



def _fate(cut: bool | None, heard: bool) -> Fate:
    """`cut` is None for an utterance whose `Uttering` a barge-in dropped before it reached the speaker."""
    match cut, heard:
        case (None, _) | (True, _):
            return "cut"
        case False, True:
            return "played"
        case False, False:
            return "silent"
