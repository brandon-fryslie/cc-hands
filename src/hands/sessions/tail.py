"""Each live session's transcript, followed from where it was last read, so a turn is known as it happens.

[LAW:one-source-of-truth] the tail is the only reader of a session's records, and the only place that knows how
much of a turn a session has been told. Nothing re-derives a turn from the whole file, so nothing can disagree
about where the turn started or what of it was heard.
"""

import asyncio
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

from loguru import logger

from hands.core.events import Continued, Interrupted, Taken, Transcribed
from hands.core.session import Instant, Membership, PromptId, SessionId
from hands.core.turn import Answering, Asked, Continuing, Interruption, Notified, Said, Step, Turn
from hands.sessions.payload import Payload, Rejected
from hands.sessions.transcript import prompt_of, turn_record
from hands.sessions.turning import Turning


@dataclass(frozen=True)
class Telling:
    """What a session has not been told of the turn it just finished, and the mark that says so once it is spoken.

    `number` is which turn of that session this is, so a mark made after a slow summary lands on the turn it was
    made of, never on one that opened while the model was answering.
    """

    session: SessionId
    turn: Turn
    number: int
    through: int
    stood_in: str | None


# How many turns that ended are kept for their tellings: a narrator that many turns behind on one session is not behind,
# it is stuck.
KEPT = 8


@dataclass
class Reading:
    """One turn as far as it has been read, how much of it was told, and every prompt id its records carry."""

    # Which turn of the session this is, counted across the whole following, so a mark made before a slow summary can
    # only ever land on the turn it was made for.
    number: int
    turn: Turning = field(default_factory=Turning)
    told: int = 0
    stood_in: str | None = None
    # A turn goes by the id of the prompt that opened it, and by any it went on under after a flush (2.1.281): a Stop
    # or an interrupt may name either.
    ids: set[PromptId] = field(default_factory=set[PromptId])


@dataclass
class Following:
    """One transcript as far as it has been read: the turn open in it, and the ones before it that may not be told yet."""

    path: Path
    offset: int = 0
    reading: Reading = field(default_factory=lambda: Reading(0))
    # [LAW:no-ambient-temporal-coupling] turns that ended, kept past the next prompt's record: the narrator can be
    # seconds behind, and a turn is told as itself however much has been read since it ended. Oldest first; let go of
    # once told, once a telling of a later turn shows their own is behind them, and past the last KEPT.
    ended: list[Reading] = field(default_factory=list[Reading])
    # The prompt_id on the user's side of the last record read, and the one Claude last answered under: the turn it is
    # in goes by the second, which a queued message changes mid-turn with no hook to say so.
    asked: PromptId | None = None
    answering: PromptId | None = None

    def consume(self, record: Payload) -> Interruption | None:
        """Read one record into the turn, setting the turn before it aside where this record opens a new one, and
        saying where it cut the turn off."""
        edge = self.reading.turn.consume(record)
        if isinstance(edge, Asked | Notified):
            self.open(edge)
        prompt = prompt_of(record)
        if prompt is not None:
            # After the opening is read, so the record that opens a turn names that turn and not the one before it.
            self.reading.ids.add(prompt)
        return edge if isinstance(edge, Interruption) else None

    def open(self, opening: Asked | Notified) -> None:
        """A turn opened: the one before it is set aside until it is told, and nothing read of it counts for this one."""
        if self.reading.turn.opening is not None:
            # Bounded: a transcript is read from its start, and every turn before the daemon attached ends here untold.
            self.ended = [*self.ended, self.reading][-KEPT:]
        self.reading = Reading(self.reading.number + 1, Turning(mid_tool=self.reading.turn.mid_tool))
        self.reading.turn.begin(opening)

    def find(self, turn: PromptId | None) -> Reading | None:
        """The turn a prompt id names, newest first; the one open, for no id; None where no record read carries it."""
        if turn is None:
            return self.reading
        return next((reading for reading in [self.reading, *reversed(self.ended)] if turn in reading.ids), None)

    def numbered(self, number: int) -> Reading | None:
        return next((reading for reading in [self.reading, *self.ended] if reading.number == number), None)

    def prompted(self, session: SessionId, record: Payload) -> Taken | Continued | None:
        """What this record says of the prompt the session is on: the first record under a prompt's id is Claude Code
        taking it, and Claude answering under an id it was not answering under before is its turn going on under that one."""
        match record.fields.get("type"):
            case "user":
                # A record that names no prompt says nothing of which one Claude is answering.
                was, self.asked = self.asked, prompt_of(record) or self.asked
                return None if self.asked is None or self.asked == was else Taken(session, self.asked)
            case _:
                # An assistant record: Claude answering whatever the user's side last carried.
                was, self.answering = self.answering, self.asked
                return None if was is None or self.answering is None or was == self.answering else Continued(session, was, self.answering)

    def restart(self) -> None:
        """Read this file again from its start: nothing read of the file it was says anything about the file it is."""
        # Numbered on from the turn that was, so a telling made before any of this marks nothing after it.
        self.reading = Reading(self.reading.number + 1)
        self.ended = []
        self.offset = 0
        self.asked = self.answering = None


class Known(Protocol):
    """What the tail needs of the registry: who is live, and where any session it has heard of keeps its transcript."""

    def live_members(self) -> list[Membership]: ...

    def now(self) -> Instant: ...

    def membership(self, session: SessionId) -> Membership | None: ...


class Tails:
    """Every live session's transcript, followed. Only the narrator asks it anything."""

    def __init__(self, known: Known) -> None:
        # [LAW:one-source-of-truth] where a session's transcript is, is the registry's to say, not this one's to keep.
        self._known = known
        self._following: dict[SessionId, Following] = {}
        # [LAW:no-ambient-temporal-coupling] reading happens off the loop in a thread, and a Stop reads the same
        # transcript the catch-up is reading. [LAW:single-enforcer] everything that touches a Following waits on
        # this, the marking of what was told included, so no two threads are ever inside one Following.
        self._reading = asyncio.Lock()
        # How far behind the newest record read was when it was read, in seconds, of whichever transcript the
        # last reading touched: it measures this loop keeping up, which is the loop's property and not a
        # session's. None where that record carried no timestamp, because then nothing measured it.
        self.lag: float | None = None
        # Everything read of a turn that no hook says and not yet handed out, in the order it was read, by whichever
        # reading found it: a Stop's reading can be the one that reads it. Touched only under the lock.
        self._transcribed: list[Transcribed] = []

    async def catch_up(self) -> list[Transcribed]:
        """Read what has been appended to every live session's transcript, and forget the sessions that are gone.

        Returns what was read of each turn since the last catch-up that no hook says — where its prompt was taken, where
        it was interrupted, and where it went on under a queued message's id — in the order it was read.
        """
        async with self._reading:
            members = self._known.live_members()
            live = {member.id for member in members}
            self._following = {session: following for session, following in self._following.items() if session in live}
            for member in members:
                following = self._following.setdefault(member.id, Following(member.transcript))
                if following.path != member.transcript:
                    # [LAW:one-source-of-truth] where a session's transcript is, is the registry's to say. A
                    # session registered again writes a new one, and nothing read of the old file says anything
                    # about the new: it is read from its start, and the turn it opens is the turn that is told.
                    following.path = member.transcript
                    following.restart()
                try:
                    await asyncio.to_thread(self._read, member.id, following)
                except FileNotFoundError:
                    # Claude Code creates the transcript with its first record, which a session may not have written yet.
                    logger.debug(f"session {member.id} has not written {following.path} yet")
                except OSError as error:
                    # [LAW:no-silent-failure] the offset does not move, so the same bytes are read again at the next catch-up.
                    logger.error(f"cannot read the transcript of session {member.id} from {following.path}: {error}")
            transcribed, self._transcribed = self._transcribed, []
            return transcribed

    async def tell(self, session: SessionId, turn: PromptId | None, closing: str | None) -> Telling | None:
        """What the session has not been told of the turn `turn` names, or None before anything has opened one.

        `turn` is any prompt id the turn's records carry, and names it whatever has been read since it ended; None
        tells the turn open now. `closing` is the reply the Stop hook carries. Measured over twelve live turns, Claude Code writes that
        reply's own record 46 to 77 ms after the hook fires, so the hook's copy stands in until the record lands.
        """
        async with self._reading:
            following = self._follow(session)
            if following is None:
                # A Stop from a session the registry does not list: there is no transcript to tell it from.
                return None
            # The turn stopped a moment ago, so the last records of it may not have been read by the loop yet.
            # A transcript that cannot be read raises here, where the narrator says so rather than saying nothing.
            await asyncio.to_thread(self._read, session, following)
            reading = following.find(turn)
            if reading is None:
                # [LAW:no-silent-failure] the whole transcript was just read, so its turn is not in it: read again from
                # the start of another file since, or let go of after its telling. Nothing is told rather than another turn.
                logger.warning(f"no turn kept from {following.path} carries prompt {turn}, so there is nothing to tell of it")
                return None
            if reading.turn.opening is None:
                return None
            if turn is not None:
                # Tellings are made in the order their turns ended, so the turns before the one named have had theirs.
                # A telling that names no turn says nothing of which one ended, so it lets go of none.
                following.ended = [held for held in following.ended if held.number >= reading.number]
            steps = reading.turn.steps()
            # Claude Code only ever appends, so the record of a stand-in that has since been written is the first step
            # after what was heard; counting it heard too is how the stand-in gives way without the reply being told twice.
            heard = reading.told + (1 if reading.stood_in is not None and _said_at(steps, reading.told) == reading.stood_in else 0)
            # [LAW:one-source-of-truth] the transcript is the record of what Claude said; the hook's copy stands in only
            # while the turn does not yet end on it. Both are read the same way, so a record padded with whitespace
            # neither misses its stand-in nor hides behind one.
            reply = _spoken(closing)
            stand_in = None if reply == _said_at(steps, len(steps) - 1) else reply
            shown = steps[heard:] if stand_in is None else [*steps[heard:], Said(None, stand_in)]
            # [LAW:types-are-the-program] a turn whose earlier steps went out already is a different thing to
            # report than a fresh one, and saying which it is here is what keeps the opening from being asked twice.
            standing = Answering() if heard == 0 else Continuing(heard)
            return Telling(session, Turn(reading.turn.opening, tuple(shown), standing), reading.number, len(steps), stand_in)

    async def spoken(self, telling: Telling) -> None:
        """Mark what a telling held as told, on the turn it was made of, whatever has opened since.

        Under the same lock as the reading: a summary takes seconds, and the turn it was asked about can open its
        successor in a worker thread while it comes back. Finding the turn by its number and writing what it was
        told are one step here, so a mark can never land on a turn it was not made of.
        """
        async with self._reading:
            following = self._following.get(telling.session)
            reading = None if following is None else following.numbered(telling.number)
            if following is None or reading is None:
                # Its transcript was read again from the start since, or the session is gone: there is nothing left to mark.
                return
            reading.told = telling.through
            reading.stood_in = telling.stood_in
            # A turn that ended has no more records coming, so told is all of it.
            following.ended = [held for held in following.ended if held is not reading]

    def _follow(self, session: SessionId) -> Following | None:
        """The session's transcript, followed from now if the catch-up has not reached it yet.

        A session's last turn is told after the session has ended — `claude -p` exits the moment its turn stops —
        so this asks the registry for any session it has heard of, not only for the ones still live.
        """
        following = self._following.get(session)
        if following is not None:
            return following
        membership = self._known.membership(session)
        return None if membership is None else self._following.setdefault(session, Following(membership.transcript))

    def _read(self, session: SessionId, following: Following) -> None:
        """Raises OSError, which the caller decides what to make of: a file not written yet, or one that cannot be read."""
        with following.path.open("rb") as file:
            size = file.seek(0, os.SEEK_END)
            if size < following.offset:
                # Reset in place: the narrator may be holding this very following while a summary comes back.
                logger.warning(f"the transcript of session {session} is shorter than what was read of it, so it is read again from its start")
                following.restart()
            file.seek(following.offset)
            raw = file.read()
        # [LAW:no-ambient-temporal-coupling] a record is whole only once its newline is written, so the bytes after
        # the last newline stay unread and unconsumed until the write that ends them.
        *complete, unfinished = raw.split(b"\n")
        following.offset += len(raw) - len(unfinished)
        for line in complete:
            try:
                record = turn_record(line)
            except Rejected as error:
                # [LAW:no-silent-failure] one unreadable line is skipped and said; the rest of the turn is still told.
                logger.error(f"a record in the transcript of session {session} could not be read, so it is not told: {error}")
                continue
            if record is not None:
                # The prompt first: a prompt's first record can be the one that interrupts it, and it was taken to be.
                prompted = following.prompted(session, record)
                if prompted is not None:
                    self._transcribed.append(prompted)
                if following.consume(record) is not None:
                    self._interrupt(session, record)
                self.lag = _lag(record)

    def _interrupt(self, session: SessionId, record: Payload) -> None:
        prompt = prompt_of(record)
        if prompt is None:
            # [LAW:no-silent-failure] a record that names no turn cannot end one, so the session stays working until its next prompt.
            logger.error(f"session {session} was interrupted, but the record of it names no prompt, so its turn cannot be ended")
            return
        # [LAW:effects-at-boundaries] stamped from the registry's one clock, as a hook is when it arrives.
        self._transcribed.append(Interrupted(session, prompt, self._known.now()))


async def keep_tailing(tails: Tails, period: float, apply: Callable[[Transcribed], Awaitable[None]]) -> None:
    """Read what every live transcript has gained, once a period, and apply what it held that no hook says, until cancelled.

    The period is how late a record can be turned into a step, which is what a turn narrated while it runs waits on,
    and how late a turn the user stopped is heard to have stopped.
    """
    while True:
        # Applied outside the reading: an interruption is told as a stopped turn is, and the telling reads the tail.
        for transcribed in await tails.catch_up():
            await apply(transcribed)
        await asyncio.sleep(period)


def _said_at(steps: list[Step], index: int) -> str | None:
    """What Claude said in the step at this place; None where the turn has no such step, or used a tool there."""
    step = steps[index] if 0 <= index < len(steps) else None
    return _spoken(step.text) if isinstance(step, Said) else None


def _spoken(text: str | None) -> str | None:
    """A reply as it is compared and told: what Claude wrote without the whitespace around it, and nothing for an empty one."""
    return None if text is None or not text.strip() else text.strip()


def _lag(record: Payload) -> float | None:
    """How long ago Claude Code wrote this record, by its own timestamp; None for a record that carries none."""
    stamp = record.fields.get("timestamp")
    if not isinstance(stamp, str):
        return None
    try:
        written = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return time.time() - written.timestamp()
