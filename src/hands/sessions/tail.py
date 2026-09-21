"""Each live session's transcript, followed from where it was last read, so a turn is known as it happens.

[LAW:one-source-of-truth] the tail is the only reader of a session's records, and the only place that knows how
much of a turn a session has been told. Nothing re-derives a turn from the whole file, so nothing can disagree
about where the turn started or what of it was heard.
"""

import asyncio
import os
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast

from loguru import logger

from hands.core.session import Membership, SessionId
from hands.core.steps import Call, Result, recognise
from hands.core.turn import Opening, Said, Step, Turn
from hands.sessions.payload import Payload, Rejected
from hands.sessions.transcript import blocks, holds_a_tool, opening_of, ref_of, result_text, structured_result, turn_record


@dataclass(frozen=True)
class Telling:
    """What a session has not been told of the turn it just finished, and the mark that says so once it is spoken.

    `number` is which turn of that session this is, so a mark made before a slow summary can never land on the
    turn that opened while the model was answering.
    """

    session: SessionId
    turn: Turn
    number: int
    through: int
    stood_in: str | None


@dataclass
class Following:
    """One transcript as far as it has been read, and the turn that is open in it."""

    path: Path
    offset: int = 0
    number: int = 0
    opening: Opening | None = None
    # A call's result arrives in a later record; its id holds the call's place in the order until it does.
    slots: list[Step | str] = field(default_factory=list[Step | str])
    calls: dict[str, Call] = field(default_factory=dict[str, Call])
    places: dict[str, int] = field(default_factory=dict[str, int])
    told: int = 0
    stood_in: str | None = None
    mid_tool: bool = False

    def open(self, opening: Opening) -> None:
        """A turn opened: nothing told of the last one counts for this one."""
        self.forget()
        self.opening = opening

    def restart(self) -> None:
        """Read this file again from its start: nothing read of the file it was says anything about the file it is."""
        self.forget()
        self.offset = 0
        self.mid_tool = False

    def forget(self) -> None:
        # The number says which turn this is, so a telling made before any of this marks nothing after it.
        self.number += 1
        self.opening = None
        self.slots = []
        self.calls = {}
        self.places = {}
        self.told = 0
        self.stood_in = None

    def settled(self) -> list[Step]:
        """Every step of the turn so far. A call whose result has not been written is told as having none."""
        return [slot if not isinstance(slot, str) else recognise(self.calls[slot]) for slot in self.slots]


class Known(Protocol):
    """What the tail needs of the registry: who is live, and where any session it has heard of keeps its transcript."""

    def live_members(self) -> list[Membership]: ...

    def membership(self, session: SessionId) -> Membership | None: ...


class Tails:
    """Every live session's transcript, followed. Only the narrator asks it anything."""

    def __init__(self, known: Known) -> None:
        # [LAW:one-source-of-truth] where a session's transcript is, is the registry's to say, not this one's to keep.
        self._known = known
        self._following: dict[SessionId, Following] = {}
        # How far behind the newest record read was when it was read, in seconds.
        self.lag: float | None = None

    async def catch_up(self) -> None:
        """Read what has been appended to every live session's transcript, and forget the sessions that are gone."""
        members = self._known.live_members()
        live = {member.id for member in members}
        self._following = {session: following for session, following in self._following.items() if session in live}
        for member in members:
            following = self._following.setdefault(member.id, Following(member.transcript))
            try:
                await asyncio.to_thread(self._read, member.id, following)
            except FileNotFoundError:
                # Claude Code creates the transcript with its first record, which a session may not have written yet.
                logger.debug(f"session {member.id} has not written {following.path} yet")
            except OSError as error:
                # [LAW:no-silent-failure] the offset does not move, so the same bytes are read again at the next catch-up.
                logger.error(f"cannot read the transcript of session {member.id} from {following.path}: {error}")

    async def tell(self, session: SessionId, closing: str | None) -> Telling | None:
        """What the session has not been told of its turn, or None before anything has opened one.

        `closing` is the reply the Stop hook carries. Measured over twelve live turns, Claude Code writes that
        reply's own record 46 to 77 ms after the hook fires, so the hook's copy stands in until the record lands.
        """
        following = self._follow(session)
        if following is None:
            # A Stop from a session the registry does not list: there is no transcript to tell it from.
            return None
        # The turn stopped a moment ago, so the last records of it may not have been read by the loop yet.
        # A transcript that cannot be read raises here, where the narrator says so rather than saying nothing.
        await asyncio.to_thread(self._read, session, following)
        if following.opening is None:
            return None
        steps = following.settled()
        # Claude Code only ever appends, so the record of a stand-in that has since been written is the first step
        # after what was heard; counting it heard too is how the stand-in gives way without the reply being told twice.
        heard = following.told + (1 if following.stood_in is not None and _said_at(steps, following.told) == following.stood_in else 0)
        # [LAW:one-source-of-truth] the transcript is the record of what Claude said; the hook's copy stands in only
        # while the turn does not yet end on it. Both are read the same way, so a record padded with whitespace
        # neither misses its stand-in nor hides behind one.
        reply = _spoken(closing)
        stand_in = None if reply == _said_at(steps, len(steps) - 1) else reply
        shown = steps[heard:] if stand_in is None else [*steps[heard:], Said(None, stand_in)]
        return Telling(session, Turn(following.opening, tuple(shown)), following.number, len(steps), stand_in)

    def spoken(self, telling: Telling) -> None:
        """Mark what a telling held as told. A telling of a turn that has since been replaced marks nothing."""
        following = self._following.get(telling.session)
        if following is None or following.number != telling.number:
            # The session opened another turn while this one was being summarised: its steps are its own to tell.
            return
        following.told = telling.through
        following.stood_in = telling.stood_in

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
                self._consume(following, record)
                self.lag = _lag(record)

    def _consume(self, following: Following, record: Payload) -> None:
        opening = opening_of(record, following.mid_tool)
        parts = blocks(record)
        mid_tool = holds_a_tool(record)
        if opening is not None:
            following.open(opening)
            following.mid_tool = mid_tool
            # The record that opens a turn is what was asked, not a step of the answer.
            return
        following.mid_tool = mid_tool
        ref = ref_of(record)
        structured = structured_result(record)
        for block in parts:
            match block:
                case {"type": "text", "text": str() as text} if record.fields.get("type") == "assistant" and text.strip():
                    following.slots.append(Said(ref, text))
                case {"type": "tool_use", "id": str() as id, "name": str() as name, "input": dict()}:
                    following.calls[id] = Call(ref, name, cast(dict[str, object], block["input"]), None)
                    following.places[id] = len(following.slots)
                    following.slots.append(id)
                case {"type": "tool_result", "tool_use_id": str() as id}:
                    place = following.places.get(id)
                    if place is not None:
                        result = Result(result_text(block.get("content")), structured, block.get("is_error") is True)
                        following.slots[place] = recognise(replace(following.calls[id], result=result))
                case _:
                    # Thinking is how Claude reached a result, not a result; the summariser is shown what a turn did.
                    pass


async def keep_tailing(tails: Tails, period: float) -> None:
    """Read what every live transcript has gained, once a period, until cancelled.

    The period is how late a record can be turned into a step, which is what a turn narrated while it runs waits on.
    """
    while True:
        await tails.catch_up()
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
