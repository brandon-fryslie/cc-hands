"""The brain's context, kept from the proxy: old tool results sent as a line, and compaction steered to what a voice
session needs.

Each line is asked as a side question once the turn its result came in has ended, of a Claude Code of its own that is
shown the result (`hands.brain.asides`): never of the brain, whose input is the user's. A line is used from the first
main turn whose batch takes its result, and on every request after that shares the brain's history: a fork of the
brain's and its compaction too, so each of them shares the main turn's cached prefix.
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from typing import Protocol

from loguru import logger

from hands.brain.asides import AsideFailed, TimeLimit
from hands.core.context import Result, aged, key, line, question, results
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
    tool_answers,
)
from hands.sessions.audit import Record
from hands.sessions.proxy import Listener
from hands.sessions.wide import annotate, count, unit

# Turns a long result goes whole before it goes as a line, and how many turns' results go at once.
EVERY = 5
# How long the sentence an old result goes as is waited on, from its turn, however long it waited for it: one reply,
# with thinking. Nothing waits on a line but the batch it is for.
LINE_TIME = TimeLimit(120.0)

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

1. Sessions: each session by its name, what it is working on, and how it last stood.
2. In flight: anything being drafted with the person — a message to a session, a ticket, a prompt — word for word as it
   now stands, and what it is waiting on.
3. Decisions: what the person decided or asked for, and whether each is done.
4. Promised: what you told the person you would do and have not done yet.

Leave out what tools returned that is already said above, and anything the person would not ask about again."""


class Store(Protocol):
    def known(self, digest: Digest) -> str | None: ...

    def keep(self, said: Mapping[Digest, str]) -> None: ...


class Keeper:
    """The changes hands makes to the brain's history on the way out, and the side questions that make them possible."""

    def __init__(self, session: SessionId, ask: Callable[[str], Awaitable[str]], store: Store, every: int, record: Record) -> None:
        # The session the brain's requests carry, and what answers a side question.
        self._session = session
        self._ask = ask
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

    def changes(self, sent: Sent) -> tuple[Change, ...]:
        """What a request of the brain's goes with: its old results as lines, and a compaction its steered prompt."""
        if sent.session != self._session:
            return ()
        match sent.kind:
            case MainTurn():
                # [LAW:single-enforcer] only the brain's own loop moves the boundary: a fork carries its question as one
                # more prompt, and counting it would stub a batch early and break the prefix it shares.
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
            case Sent(session=session, kind=MainTurn(), body=body) if session == self._session:
                fresh = [result for result in results(body) if result.call not in self._heard]
                self._heard.update(result.call for result in fresh)
                self._unsaid.update((result.call, result) for result in fresh if self._store.known(key(result)) is None)
            case Exchanged(session=session, kind=MainTurn(), reply=Held()) if session == self._session:
                self._turn_ended()
            case Exchanged(session=session, kind=MainTurn(), reply=Reached(body=Streamed(message=message))) if (
                session == self._session and message.stop_reason != "tool_use"
            ):
                self._turn_ended()
            case _:
                pass

    async def keep_asking(self) -> None:
        """Asks for a sentence for each unsaid result as its turn ends, one at a time, for as long as the brain runs."""
        while True:
            result = await self._asking.get()
            try:
                said = await self._ask(question(result))
            except AsideFailed as error:
                # [LAW:no-silent-failure] the result goes whole once its batch is reached, and this says why.
                logger.error(f"no sentence for {result.tool} call {result.call}: {error}")
                continue
            # One line, however the model broke it.
            self._store.keep({key(result): " ".join(said.split())})

    def _turn_ended(self) -> None:
        for result in self._unsaid.values():
            self._asking.put_nowait(result)
        self._unsaid.clear()

    def _decide(self, old: tuple[Result, ...]) -> None:
        fresh = [result for result in old if result.call not in self._decided]
        if fresh:
            # [LAW:nothing-unseen] a batch reached is one unit of work, its lookups in the store included: the calls whose
            # results go as a line from now on, and those that go whole because no sentence had been said of them by then.
            with unit("context.stubbing", self._record, counts=("stubbed", "whole")):
                for result in fresh:
                    said = self._store.known(key(result))
                    self._decided[result.call] = None if said is None else line(result, said)
                stubbed = tuple(result.call for result in fresh if self._decided[result.call] is not None)
                whole = tuple(result.call for result in fresh if self._decided[result.call] is None)
                annotate(stubbed=stubbed, whole=whole)
                count(stubbed=len(stubbed), whole=len(whole))

    def _stubs(self, body: object) -> tuple[Change, ...]:
        return tuple(Stub(answer.call, said) for answer in tool_answers(body) if (said := self._decided.get(answer.call)) is not None)


class Hearing(Protocol):
    def hear(self, observed: Observed) -> None: ...


class Asking(Hearing, Protocol):
    """What answers hands' side questions, as the wire sees it: a question's own request is its part."""

    def adopted(self, sent: Sent, routed: Route) -> Route: ...


class Kept:
    """The brain's listener on the wire: the stage's route, with the keeper's changes made before the stage's own and a
    side question's own request inside the question, and everything heard too by whatever else reads the wire: the
    brain, and what answers hands' side questions."""

    def __init__(self, stage: Listener, keeper: Keeper, asides: Asking, *others: Hearing) -> None:
        self._stage = stage
        self._keeper = keeper
        self._asides = asides
        self._others = (*others, asides)

    def route(self, sent: Sent) -> Route:
        # The keeper's first: a history it cannot read fails the whole route before the stage has counted the request.
        kept = self._keeper.changes(sent)
        match self._asides.adopted(sent, self._stage.route(sent)):
            # [LAW:one-source-of-truth] the stage's Send as it routed it, every field carried, with the keeper's changes first.
            case Send() as send:
                return replace(send, changes=(*kept, *send.changes))
            case Hold() as held:
                return held

    def hear(self, observed: Observed) -> None:
        self._stage.hear(observed)
        self._keeper.hear(observed)
        for other in self._others:
            other.hear(observed)
