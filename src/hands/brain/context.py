"""The brain's context, kept from the proxy: old tool results sent as a line, and compaction steered to what a voice
session needs.

Each line is asked of a fork of the brain once the turn its result came in has ended, while the result is still whole
in the prefix the fork shares, so the fork is a read of the brain's cache. A line is used from the first main turn whose
batch takes its result, and on every request after that shares the brain's history: the forks and the compaction too,
so each of them shares the main turn's cached prefix.
"""

import asyncio
from collections.abc import Mapping
from typing import Protocol

from loguru import logger

from hands.brain.process import BrainGone, ForkFailed
from hands.core.context import Result, aged, key, line, question, results, sentence
from hands.core.sentences import Digest
from hands.core.session import SessionId
from hands.core.wire import (
    COMPACTION_OPENING,
    Change,
    Compaction,
    Exchanged,
    Fork,
    Held,
    Hold,
    MainTurn,
    Observed,
    Reached,
    Route,
    Send,
    Sent,
    Steer,
    Streamed,
    Stub,
    asked,
    tool_answers,
)
from hands.sessions.audit import Record, ResultsStubbed
from hands.sessions.proxy import Listener

# Turns a long result goes whole before it goes as a line, and how many turns' results go at once.
EVERY = 5

# What the brain's compaction is asked for instead of Claude Code's summary of a coding session. The first paragraph
# keeps the form Claude Code reads the summary back in (services/compact/prompt.ts): text only, an <analysis> block,
# then a <summary> block. Claude Code's own closing reminder, and any instructions it was given, follow it.
VOICE_COMPACTION = f"""{COMPACTION_OPENING}

- Do NOT use Read, Bash, Grep, Glob, Skill, or ANY other tool.
- You already have all the context you need in the conversation above.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.

You are the voice between a person and the Claude Code sessions they run. This summary replaces the conversation so
far, and you will go on talking with them from it alone. In <analysis>, go through the conversation in order and note
what matters below. Then write <summary> with these sections, in plain sentences:

1. Sessions: each session by its title, what it is working on, and how it last stood.
2. In flight: anything being drafted with the person — a message to a session, a ticket, a prompt — word for word as it
   now stands, and what it is waiting on.
3. Decisions: what the person decided or asked for, and whether each is done.
4. Promised: what you told the person you would do and have not done yet.

Leave out what tools returned that is already said above, and anything the person would not ask about again."""


class Forking(Protocol):
    """What the keeper needs of the brain: the session its requests carry, and a side question asked of a fork."""

    @property
    def session(self) -> SessionId: ...

    async def fork(self, question: str) -> str: ...


class Store(Protocol):
    def known(self, digest: Digest) -> str | None: ...

    def keep(self, said: Mapping[Digest, str]) -> None: ...


class Keeper:
    """The changes hands makes to the brain's history on the way out, and the forks that make them possible."""

    def __init__(self, brain: Forking, store: Store, every: int, record: Record) -> None:
        self._brain = brain
        self._store = store
        self._every = every
        self._record = record
        # [LAW:one-source-of-truth] what each old result goes as, decided once, when its batch is reached: its line, or
        # None when no sentence had been said of it by then. Never revisited, so the history stays byte-identical
        # between batches whatever the store learns later.
        self._decided: dict[str, str | None] = {}
        # Results heard on a main turn that the store has no sentence for, asked about once their turn ends.
        self._unsaid: dict[str, Result] = {}
        # Every call whose result has been heard, so each is looked up in the store once.
        self._heard: set[str] = set()
        self._asking: asyncio.Queue[Result] = asyncio.Queue()
        # The question a fork is being asked, and the result each of its requests on the wire held whole, by exchange:
        # a fork asked after a compaction, or one that forked from a history changed since, answers of what it cannot see.
        self._asked: dict[str, Result] = {}
        self._held: dict[str, Result] = {}

    def changes(self, sent: Sent) -> tuple[Change, ...]:
        """What a request of the brain's goes with: its old results as lines, and a compaction its steered prompt."""
        if sent.session != self._brain.session:
            return ()
        match sent.kind:
            case MainTurn():
                # [LAW:single-enforcer] only the brain's own loop moves the boundary: a fork carries the side question
                # as one more prompt, and counting it would stub a batch early and break the prefix it shares.
                self._decide(aged(sent.body, self._every))
                return self._stubs(sent.body)
            case Fork():
                return self._stubs(sent.body)
            case Compaction():
                return (*self._stubs(sent.body), Steer(VOICE_COMPACTION))
            case _:
                return ()

    def hear(self, observed: Observed) -> None:
        match observed:
            case Sent(session=session, kind=MainTurn(), body=body) if session == self._brain.session:
                fresh = [result for result in results(body) if result.call not in self._heard]
                self._heard.update(result.call for result in fresh)
                self._unsaid.update((result.call, result) for result in fresh if self._store.known(key(result)) is None)
            case Sent(exchange=exchange, session=session, kind=Fork(), body=body) if session == self._brain.session:
                # [LAW:no-ambient-temporal-coupling] the fork's request is heard before it goes upstream, so while it is
                # still being asked; its sentence is kept when its own exchange ends, whenever the fork's answer comes.
                whole = {(answer.call, answer.text) for answer in tool_answers(body)}
                for result in (result for said, result in self._asked.items() if any(said in text for text in asked(body))):
                    if (result.call, result.text) in whole:
                        self._held[exchange] = result
                    else:
                        logger.error(f"no sentence for {result.tool} call {result.call}: the fork's request did not hold the result whole")
            case Exchanged(exchange=exchange, kind=Fork(), reply=reply) if exchange in self._held:
                result = self._held.pop(exchange)
                try:
                    match reply:
                        case Reached(body=Streamed(message=message)):
                            self._store.keep({key(result): sentence(message)})
                        case _:
                            # Claude Code may ask again, as another exchange.
                            raise ValueError(f"the fork's request was answered {reply}")
                except ValueError as error:
                    logger.error(f"no sentence for {result.tool} call {result.call}: {error}")
            case Exchanged(session=session, kind=MainTurn(), reply=Held()) if session == self._brain.session:
                self._turn_ended()
            case Exchanged(session=session, kind=MainTurn(), reply=Reached(body=Streamed(message=message))) if (
                session == self._brain.session and message.stop_reason != "tool_use"
            ):
                self._turn_ended()
            case _:
                pass

    async def keep_asking(self) -> None:
        """Asks a fork for each unsaid result as its turn ends, one at a time, for as long as the brain runs."""
        while True:
            result = await self._asking.get()
            asking = question(result)
            self._asked[asking] = result
            try:
                # What the fork said is kept from the wire, as its exchange ends: this only waits for it to be done.
                await self._brain.fork(asking)
            except (ForkFailed, BrainGone) as error:
                # [LAW:no-silent-failure] the result goes whole once its batch is reached, and this says why. A brain
                # that is gone is the brain's watch to report, with its stderr.
                logger.error(f"no sentence for {result.tool} call {result.call}: {error}")
            finally:
                del self._asked[asking]

    def _turn_ended(self) -> None:
        for result in self._unsaid.values():
            self._asking.put_nowait(result)
        self._unsaid.clear()

    def _decide(self, old: tuple[Result, ...]) -> None:
        fresh = [result for result in old if result.call not in self._decided]
        for result in fresh:
            said = self._store.known(key(result))
            self._decided[result.call] = None if said is None else line(result, said)
        if fresh:
            unsaid = tuple(result.call for result in fresh if self._decided[result.call] is None)
            self._record(ResultsStubbed(tuple(result.call for result in fresh if result.call not in unsaid), unsaid))

    def _stubs(self, body: object) -> tuple[Change, ...]:
        return tuple(Stub(answer.call, said) for answer in tool_answers(body) if (said := self._decided.get(answer.call)) is not None)


class Hearing(Protocol):
    def hear(self, observed: Observed) -> None: ...


class Kept:
    """The brain's listener on the wire: the stage's route, with the keeper's changes made before the stage's own, and
    everything heard by the brain too, which reads its side questions' answers there."""

    def __init__(self, stage: Listener, keeper: Keeper, brain: Hearing) -> None:
        self._stage = stage
        self._keeper = keeper
        self._brain = brain

    def route(self, sent: Sent) -> Route:
        # The keeper's first: a history it cannot read fails the whole route before the stage has counted the request.
        kept = self._keeper.changes(sent)
        match self._stage.route(sent):
            case Send(changes=changes):
                return Send((*kept, *changes))
            case Hold() as held:
                return held

    def hear(self, observed: Observed) -> None:
        self._stage.hear(observed)
        self._keeper.hear(observed)
        self._brain.hear(observed)
