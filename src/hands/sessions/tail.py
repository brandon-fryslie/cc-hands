"""Each live session's transcript, followed from where it was last read, so a turn is known as it happens.

[LAW:one-source-of-truth] the tail is the only reader of a session's records, and the only place that knows how
much of a turn a session has been told. Nothing re-derives a turn from the whole file, so nothing can disagree
about where the turn started or what of it was heard.
"""

import asyncio
import os
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from loguru import logger

from hands.core.events import Continued, Interrupted, Progressed, Read, Taken, Transcribed
from hands.core.progress import Doing, doing
from hands.core.session import Instant, Membership, PromptId, SessionId
from hands.core.status import Stamp
from hands.core.turn import AgentId, AgentTask, Answering, Delegated, Continuing, Interruption, Opening, Said, Step, Turn
from hands.core.steps import Call
from hands.sessions.payload import Payload, Rejected
from hands.sessions.subagents import started_from, subagents_of, transcript_of
from hands.sessions.transcript import prompt_of, turn_record, written_of
from hands.sessions.turning import Turning


@dataclass(frozen=True)
class StoodIn:
    """A reply told from the Stop hook's copy before its own record was read: that record, when it lands, was heard."""

    text: str


# What the told part of a turn ends on, said: words its transcript holds, the hook's copy of words it does not hold yet,
# or nothing said.
type Ending = str | StoodIn | None


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
    # What the turn's told part ends on once this is: compared with the reply a later Stop of the same turn carries.
    ends_on: Ending
    # The file the turn was read from, beside which the subagents it ran keep transcripts of their own.
    transcript: Path


# How many turns that ended are kept for their tellings: a narrator that many turns behind on one session is not behind,
# it is stuck.
KEPT = 8


@dataclass
class Reading:
    """One turn as far as it has been read and not yet told, and every prompt id its records carry."""

    # Which turn of the session this is, counted across the whole following, so a mark made before a slow summary can
    # only ever land on the turn it was made for.
    number: int
    # [LAW:one-source-of-truth] how much of it was told is how much of it the turning let go of.
    turn: Turning = field(default_factory=Turning)
    # The turn's last words while nothing untold follows them.
    ended_on: Ending = None
    # A turn goes by the id of the prompt that opened it, and by any it went on under after a flush (2.1.281): a Stop
    # or an interrupt may name either.
    ids: set[PromptId] = field(default_factory=set[PromptId])
    # The calls of it already handed on as progress, by tool use id: only the turn's own calls, so it ends with the turn.
    called: set[str] = field(default_factory=set[str])

    def made(self) -> list[Doing]:
        """What each call read into the turn since this was last asked sets out to do, in the order they were made."""
        calls = self.turn.calls
        made = _doings(call for id, call in calls.items() if id not in self.called)
        # [LAW:carrying-cost] the calls the turn still holds, so a call let go of with its told steps is let go of here too.
        self.called = set(calls)
        return made


@dataclass
class Delegate:
    """A subagent's own transcript, followed from where it was last read, for the calls it makes while it works."""

    id: AgentId
    path: Path
    offset: int
    # The file Claude Code writes beside the transcript as it starts the subagent, naming the job it gave it.
    meta: Path
    # Whether the record its transcript starts from is still to be read: the job it was given, which is no call of its
    # own, though a fork's is the parent's call that launched it, copied in ahead of the fork's own work.
    job: bool
    turn: Turning = field(default_factory=Turning)
    # Who its calls are said to be the work of, once its transcript grows: read then, so a subagent that never works
    # again after hands begins following its session is never asked who started it. Unstarted where nothing says, which is
    # said once, and its calls are then told as nobody's.
    agent: "AgentTask | Unstarted | None" = None

    def made(self) -> list[Doing]:
        """What each call read since this was last asked sets out to do, in the order they were made."""
        made = _doings(self.turn.calls.values())
        # [LAW:carrying-cost] only what its calls set out to do is heard of a subagent while it works: what it did is read
        # from its transcript whole when it reports back, so nothing read here is kept.
        self.turn.forget(self.turn.forgotten + len(self.turn.slots))
        return made


def _doings(calls: Iterable[Call]) -> list[Doing]:
    """What each call sets out to do, in the order they were made: a call that asks the user sets out to do nothing that
    progress says, since it is spoken as it is asked."""
    return [each for each in (doing(call.tool, call.input) for call in calls) if each is not None]


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
    # Every subagent of the session known, by id, each followed in its own transcript.
    delegates: dict[AgentId, Delegate] = field(default_factory=dict[AgentId, Delegate])

    def consume(self, record: Payload) -> Interruption | None:
        """Read one record into the turn, setting the turn before it aside where this record opens a new one, and
        saying where it cut the turn off."""
        edge = self.reading.turn.consume(record)
        if isinstance(edge, Opening):
            self.open(edge)
        prompt = prompt_of(record)
        if prompt is not None:
            # After the opening is read, so the record that opens a turn names that turn and not the one before it.
            self.reading.ids.add(prompt)
        return edge if isinstance(edge, Interruption) else None

    def open(self, opening: Opening) -> None:
        """A turn opened: the one before it is set aside until it is told, and nothing read of it counts for this one."""
        if self.reading.turn.opening is not None:
            # Bounded: a transcript is read from its start, and every turn before the daemon attached ends here untold.
            self.ended = [*self.ended, self.reading][-KEPT:]
        self.reading = Reading(self.reading.number + 1, Turning(mid_tool=self.reading.turn.mid_tool, typed=self.reading.turn.typed))
        self.reading.turn.begin(opening)

    def find(self, turn: PromptId | None) -> Reading | None:
        """The turn a prompt id names, newest first; the one open, for no id; None where no record read carries it."""
        if turn is None:
            return self.reading
        return next((reading for reading in [self.reading, *reversed(self.ended)] if turn in reading.ids), None)

    def current(self) -> int:
        """The number of the first reading of the turn the transcript is in: the one open, and every one before it that
        went by a prompt id it goes by, as a turn a queued message was flushed into does."""
        ids, first = set(self.reading.ids), self.reading.number
        for reading in reversed(self.ended):
            if not reading.ids & ids:
                break
            ids |= reading.ids
            first = reading.number
        return first

    def numbered(self, number: int) -> Reading | None:
        return next((reading for reading in [self.reading, *self.ended] if reading.number == number), None)

    def prompted(self, session: SessionId, record: Payload, at: Instant) -> Taken | Continued | None:
        """What this record says of the prompt the session is on: the first record under a prompt's id is Claude Code
        taking it, and Claude answering under an id it was not answering under before is its turn going on under that one."""
        match record.fields.get("type"):
            case "user":
                # A record that names no prompt says nothing of which one Claude is answering.
                was, self.asked = self.asked, prompt_of(record) or self.asked
                return None if self.asked is None or self.asked == was else Taken(session, self.asked, _written(session, record), at)
            case "assistant":
                # Claude answering whatever the user's side last carried.
                was, self.answering = self.answering, self.asked
                return None if was is None or self.answering is None or was == self.answering else Continued(session, was, self.answering)
            case _:
                # A local_command record: Claude Code's own, written under no prompt id, and no answer of Claude's.
                return None

    def restart(self) -> None:
        """Read this file again from its start: nothing read of the file it was says anything about the file it is."""
        # Numbered on from the turn that was, so a telling made before any of this marks nothing after it.
        self.reading = Reading(self.reading.number + 1)
        self.ended = []
        self.offset = 0
        self.asked = self.answering = None
        self.delegates = {}


class Known(Protocol):
    """What the tail needs of the registry: who is live, and where any session it has heard of keeps its transcript."""

    def live_members(self) -> list[Membership]: ...

    def status_read(self, session: SessionId) -> bool: ...

    def now(self) -> Instant: ...

    def stamp(self) -> Stamp: ...

    def membership(self, session: SessionId) -> Membership | None: ...


class Tails:
    """Every live session's transcript, followed. Only the narrator asks it anything."""

    def __init__(self, known: Known) -> None:
        # [LAW:one-source-of-truth] where a session's transcript is, is the registry's to say, not this one's to keep,
        # and so is the clock how far a transcript was read is said on: the one a Stop is heard on.
        self._known = known
        self._following: dict[SessionId, Following] = {}
        # [LAW:no-ambient-temporal-coupling] reading happens off the loop in a thread, and a Stop reads the same
        # transcript the catch-up is reading. [LAW:single-enforcer] everything that touches a Following waits on
        # this, the marking of what was told included, so no two threads are ever inside one Following.
        self._reading = asyncio.Lock()
        # Everything read of a turn that no hook says and not yet handed out, in the order it was read, by whichever
        # reading found it: a Stop's reading can be the one that reads it. Touched only under the lock.
        self._transcribed: list[Transcribed] = []

    async def catch_up(self) -> list[Transcribed]:
        """Read what has been appended to every live session's transcript, and forget the sessions that are gone.

        Returns what was read of each turn since the last catch-up that no hook says — where its prompt was taken, where
        it was interrupted, and where it went on under a queued message's id — in the order it was read, each reading
        followed by how far it read.
        """
        async with self._reading:
            members = self._known.live_members()
            live = {member.id for member in members}
            self._following = {session: following for session, following in self._following.items() if session in live}
            # [LAW:no-ambient-temporal-coupling] a session is read once Claude Code has said whether it runs, so the
            # turn its transcript is in when hands begins following it lands on a status that decides whether it runs.
            for member in [member for member in members if self._known.status_read(member.id)]:
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
            untold = reading.turn.steps()
            # Claude Code only ever appends, so the record of a stand-in that has since been written is the first step
            # after what was heard; counting it heard too is how the stand-in gives way without the reply being told twice.
            skipped = 1 if isinstance(reading.ended_on, StoodIn) and _said_at(untold, 0) == reading.ended_on.text else 0
            heard = reading.turn.forgotten + skipped
            # [LAW:one-source-of-truth] the transcript is the record of what Claude said; the hook's copy stands in only
            # while the turn does not yet end on it. Both are read the same way, so a record padded with whitespace
            # neither misses its stand-in nor hides behind one.
            reply = _spoken(closing)
            ending = _said_at(untold, len(untold) - 1) if untold else reading.ended_on
            stand_in = None if reply == (ending.text if isinstance(ending, StoodIn) else ending) else reply
            shown = untold[skipped:] if stand_in is None else [*untold[skipped:], Said(None, stand_in)]
            # [LAW:types-are-the-program] a turn whose earlier steps went out already is a different thing to
            # report than a fresh one, and saying which it is here is what keeps the opening from being asked twice.
            standing = Answering() if heard == 0 else Continuing(heard)
            through = reading.turn.forgotten + len(untold)
            return Telling(session, Turn(reading.turn.opening, tuple(shown), standing), reading.number, through, ending if stand_in is None else StoodIn(stand_in), following.path)

    async def spoken(self, telling: Telling) -> None:
        """Mark what a telling held as told, on the turn it was made of, whatever has opened since, and let go of it.

        [LAW:carrying-cost] a turn's steps hold its whole output, and a session that goes quiet after a long turn would
        keep all of it until its next prompt; what is told is never shown again, so only what is untold is kept.

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
            let_go = telling.through - reading.turn.forgotten
            waiting = reading.turn.forget(telling.through)
            # [LAW:nothing-unseen] what a told turn still holds, so a session that keeps output resident is seen to; a
            # result that lands for a call let go of is never told, and a reply stood in for gives way to its record.
            stood_in = " on the hook's copy of its reply" if isinstance(telling.ends_on, StoodIn) else ""
            logger.debug(f"session {telling.session} turn {telling.number} was told through step {telling.through}{stood_in}: let go of {let_go} steps, {waiting} of them calls with no result yet, {len(reading.turn.slots)} untold are held")
            reading.ended_on = telling.ends_on
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
        # Before the file is opened, so every record Claude Code had written by then is in what is read.
        through = self._known.stamp()
        try:
            read = _appended(following.path, following.offset)
        except OSError:
            # Nothing written yet is nothing left unread, and a transcript that cannot be read gives nothing more to wait
            # for: the turn is told, and its telling says why it could not read it [LAW:no-silent-failure].
            self._transcribed.append(Read(session, through))
            raise
        if read.restarted:
            # Reset in place: the narrator may be holding this very following while a summary comes back.
            logger.warning(f"the transcript of session {session} is shorter than what was read of it, so it is read again from its start")
            following.restart()
        # Read from its start: every turn before the one the file ends in is over by now.
        history = following.offset == 0
        following.offset = read.offset
        # What each record says that no hook does, by the turn it was read into.
        heard: list[tuple[int, Transcribed]] = []
        for record in _records(read.lines, f"session {session}"):
            # The prompt first: a prompt's first record can be the one that interrupts it, and it was taken to be.
            # [LAW:effects-at-boundaries] stamped from the registry's one clock, as a hook is when it arrives.
            prompted = following.prompted(session, record, self._known.now())
            interrupted = None if following.consume(record) is None else self._interrupted(session, record)
            heard += [(following.reading.number, event) for event in (prompted, interrupted) if event is not None]
        # [LAW:single-enforcer] the one place a record is decided to be history: of a file read from its start, only the
        # turn it ends in may still be running, which Claude Code's status says; every turn before it was over before
        # hands followed the session, and says nothing to anyone.
        # A turn's progress is heard while it runs, and a turn read from a file's start began before hands followed it:
        # its calls so far are history, as backfill reads them, and only the calls it makes from here on are heard.
        made = following.reading.made()
        if made and not history:
            heard.append((following.reading.number, Progressed(session, tuple(sorted(following.reading.ids)), tuple(made), self._known.now())))
        heard += [(following.reading.number, event) for event in self._delegated(session, following, history)]
        current = following.current()
        live = [event for number, event in heard if not history or number >= current]
        if history and (heard or made):
            # [LAW:nothing-unseen] the decision explained: what was held back, and the turn the reading starts from.
            logger.info(
                f"read the transcript of session {session} from its start: {len(heard) - len(live)} of {len(heard)} events are of turns before the one it is in, which goes by {sorted(following.reading.ids)}; calls that turn made before hands followed it, not heard as progress: {len(made)}"
            )
        self._transcribed += live
        # [LAW:no-ambient-temporal-coupling] after the records it covers, so a telling decided by how far the transcript
        # was read has what that reading found; and not while a record is half written, which may have been begun before.
        if not read.unfinished:
            self._transcribed.append(Read(session, through))

    def _delegated(self, session: SessionId, following: Following, history: bool) -> list[Progressed]:
        """What each subagent of the session set out to do since its transcript was last read, said as its own.

        A subagent is known by the file Claude Code writes beside its transcript as it starts it. One started before
        hands began following its session is followed from where its transcript ends then: its work so far is history.
        """
        for meta in subagents_of(following.path).glob("agent-*.meta.json"):
            id = AgentId(meta.name.removeprefix("agent-").removesuffix(".meta.json"))
            if id not in following.delegates:
                path = transcript_of(following.path, id)
                offset = _whole(path) if history else 0
                if offset:
                    # [LAW:nothing-unseen] where its work starts being heard from, and how much of it is history.
                    logger.info(f"subagent {id} of session {session} was working before hands followed the session, so it is heard from where its transcript ends: {offset} bytes of it are history")
                following.delegates[id] = Delegate(id, path, offset, meta, job=offset == 0)
        progressed = list[Progressed]()
        for delegate in following.delegates.values():
            if _size(delegate.path) == delegate.offset:
                # A subagent that has done nothing new, as one that reported back long since has: its file is not opened.
                continue
            try:
                # Named before its calls are read, so a meta that cannot be read yet leaves them to be read with it.
                agent = _attributed(session, delegate, following)
                made = _read_delegate(session, delegate)
                if made and agent is not None:
                    progressed.append(Progressed(session, agent, tuple(made), self._known.now()))
            except (OSError, Rejected) as error:
                # [LAW:no-silent-failure] said, and the subagent's transcript is read on from where it was: what it does
                # next is heard, and its parent's own work is heard whatever became of this.
                logger.error(f"what subagent {delegate.id} of session {session} set out to do cannot be told: {type(error).__name__}: {error}")
        return progressed

    def _interrupted(self, session: SessionId, record: Payload) -> Interrupted | None:
        prompt = prompt_of(record)
        if prompt is None:
            # [LAW:no-silent-failure] a record that names no turn is the record of none, so the turn it stopped is told once
            # the transcript is read past the window Claude Code's idle set, without it.
            logger.error(f"session {session} was interrupted, but the record of it names no prompt, so its turn is told without it")
            return None
        # [LAW:effects-at-boundaries] stamped from the registry's one clock, as a hook is when it arrives.
        return Interrupted(session, prompt, self._known.now())


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


class Unstarted(Exception):
    """Nothing says which call started a subagent: it names no job, and its parent is not running one skill."""


def _whole(path: Path) -> int:
    """Where the transcript's last whole record ends: a record Claude Code is part way through writing is not read from
    its middle. None of it before Claude Code writes its first record."""
    try:
        with path.open("rb") as file:
            at = file.seek(0, os.SEEK_END)
            while at > 0:
                back = min(4096, at)
                at -= back
                file.seek(at)
                newline = file.read(back).rfind(b"\n")
                if newline >= 0:
                    return at + newline + 1
            return 0
    except FileNotFoundError:
        return 0


def _size(path: Path) -> int:
    """How much of a subagent's transcript is written: none before Claude Code writes its first record."""
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


@dataclass(frozen=True)
class Appended:
    """The whole records a transcript gained since it was last read, and where it is read from next."""

    lines: list[bytes]
    offset: int
    # Whether the file is shorter than what was read of it, and so was read again from its start.
    restarted: bool
    # Whether a record past the last whole one is half written.
    unfinished: bool


def _appended(path: Path, offset: int) -> Appended:
    """What the transcript gained past `offset`. Raises OSError for one that cannot be read, one not written yet among them."""
    with path.open("rb") as file:
        size = file.seek(0, os.SEEK_END)
        start = 0 if size < offset else offset
        file.seek(start)
        raw = file.read()
    # [LAW:no-ambient-temporal-coupling] a record is whole only once its newline is written, so the bytes after the last
    # newline stay unread until the write that ends them.
    *complete, unfinished = raw.split(b"\n")
    return Appended(complete, start + len(raw) - len(unfinished), size < offset, bool(unfinished))


def _records(lines: Iterable[bytes], whose: str) -> Iterator[Payload]:
    """The records a turn is made of among these lines."""
    for line in lines:
        try:
            record = turn_record(line)
        except Rejected as error:
            # [LAW:no-silent-failure] one unreadable line is skipped and said; the rest of the work is still told.
            logger.error(f"a record in the transcript of {whose} could not be read, so it is not told: {error}")
            continue
        if record is not None:
            yield record


def _read_delegate(session: SessionId, delegate: Delegate) -> list[Doing]:
    """What each call the subagent's transcript gained sets out to do. Raises OSError for a transcript that cannot be read."""
    read = _appended(delegate.path, delegate.offset)
    if read.restarted:
        logger.warning(f"the transcript of subagent {delegate.id} of session {session} is shorter than what was read of it, so it is read again from its start")
        delegate.job, delegate.turn = True, Turning()
    delegate.offset = read.offset
    for record in _records(read.lines, f"subagent {delegate.id} of session {session}"):
        if delegate.job:
            job = started_from(record)
            delegate.job = job is None
            if job:
                # The job it was given, which no call of its own made.
                continue
        delegate.turn.consume(record)
    return delegate.made()


def _attributed(session: SessionId, delegate: Delegate, following: Following) -> AgentTask | None:
    """Who the subagent's calls are the work of; None for one nothing says the start of, which is said once, and whose
    calls are never told as another's. Raises OSError and Rejected, as `_started` does, to be asked again."""
    if delegate.agent is None:
        try:
            delegate.agent = _started(session, delegate, following)
        except Unstarted as error:
            delegate.agent = error
            logger.error(f"what subagent {delegate.id} of session {session} sets out to do is never told: Unstarted: {error}")
    return delegate.agent if isinstance(delegate.agent, AgentTask) else None


def _started(session: SessionId, delegate: Delegate, following: Following) -> AgentTask:
    """The call that started the subagent, by the job it gave it: as Claude Code names it beside the transcript, or, for
    a skill run in a subagent of its own, which it names no job for, as the parent's call that invoked it: the one whose
    result names this subagent, as a fork run in the background has at once, and else the one skill the parent is
    still running, as a fork in the foreground is until it reports back.

    A subagent a subagent started is the work of the call in the session that started the first: what the session
    is doing is what is heard, and its subagents' own subagents are how that one does it.

    Raises OSError for a file that cannot be read, Rejected for one that is not JSON, and Unstarted where nothing says.
    """
    fields = Payload.parse(delegate.meta.read_bytes()).fields
    match fields.get("parentAgentId"), fields.get("description"):
        case str() as parent, _:
            ancestor = following.delegates.get(AgentId(parent))
            task = None if ancestor is None else _attributed(session, ancestor, following)
            if task is None:
                raise Unstarted(f"{delegate.meta} names subagent {parent} as its parent, whose work is told as nobody's")
            # [LAW:nothing-unseen] which said who started it.
            logger.info(f"subagent {delegate.id} was started by subagent {parent}, so it is heard as the work of that one's job: {task.description!r}")
            return task
        case _, str() as description if description.strip():
            logger.info(f"subagent {delegate.id} is heard as the work of the job {delegate.meta.name} names: {description.strip()!r}")
            return AgentTask(delegate.id, description.strip())
        case _:
            parent = following.reading.turn
            invoked = [step for reading in (following.reading, *following.ended) for step in reading.turn.slots if isinstance(step, Delegated) and step.id == delegate.id]
            running = [call.input for id, call in parent.calls.items() if call.tool == "Skill" and isinstance(parent.slots[parent.places[id]], str)]
            match invoked, running:
                case [Delegated(description=description), *_], _:
                    logger.info(f"subagent {delegate.id} names no job, so it is heard as the work of the call whose result names it: {description!r}")
                    return AgentTask(delegate.id, description)
                case _, [{"skill": str() as skill, **rest}]:
                    arguments = rest.get("args")
                    invoked = f"/{skill} {arguments if isinstance(arguments, str) else ''}".strip()
                    logger.info(f"subagent {delegate.id} names no job, so it is heard as the work of the one skill its parent is running: {invoked!r}")
                    return AgentTask(delegate.id, invoked)
                case _:
                    raise Unstarted(f"{delegate.meta} names no job, and its parent is running {len(running)} skills")


def _written(session: SessionId, record: Payload) -> Stamp | None:
    """When Claude Code wrote the record; None, and said, for a time that cannot be read, which costs the record nothing
    but that: its turn is still told, and only a turn no hook opened goes unopened for want of it."""
    try:
        return written_of(record)
    except Rejected as error:
        # [LAW:no-silent-failure] said, and read as no time, which opens nothing.
        logger.error(f"a record in the transcript of session {session} has a time that cannot be read, so it is read as having none: {error}")
        return None


def _said_at(steps: list[Step], index: int) -> str | None:
    """What Claude said in the step at this place; None where the turn has no such step, or used a tool there."""
    step = steps[index] if 0 <= index < len(steps) else None
    return _spoken(step.text) if isinstance(step, Said) else None


def _spoken(text: str | None) -> str | None:
    """A reply as it is compared and told: what Claude wrote without the whitespace around it, and nothing for an empty one."""
    return None if text is None or not text.strip() else text.strip()
