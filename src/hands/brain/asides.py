"""Side questions: what hands asks in the background, each of a slim Claude Code started for it alone.

A summary of a backlog or of a session's turns, and the line an old tool result goes as, are asked here and never of
the brain. The brain's input is the user's: a question that is slow, or stuck, or never answered holds nothing a spoken
turn waits on, and nothing asked here is in the brain's conversation.

Each question has a Claude Code of its own, on the brain's login and with no tools, because a side question's answer
stays in the history of the ones asked after it (2.1.286, measured 2026-09-30: the third `/btw` typed into one process
carried the first two and their answers; `x` over an answer cleared all but that answer, and `/clear` cleared none).
Started for one question, it carries that question and nothing else, and it is ended with its answer. It leaves nothing
of the question in the brain's config directory (2.1.286, measured 2026-09-30 over 12 questions: no transcript, no line
of history, no session directory).

It is Claude Code as anyone runs it, interactive on a terminal, and the question is the prompt it is started with:
`claude "/btw ..."`. Claude Code asks its opening prompt itself once its input is up, so nothing is typed into it and
nothing depends on when its input comes up (a `/btw` typed as the input came up was left in it unsent, 12 of 12; as the
opening prompt it went out 0.4s after the start, 6 of 6). Its answer is read from the wire.
"""

import asyncio
import math
from dataclasses import dataclass
from uuid import uuid4

from loguru import logger

from hands.brain.process import ClaudeCode, Station, Unstartable, brain_claude, slim, spawn
from hands.core.effects import Command
from hands.core.session import CommandName, SessionId, pasted
from hands.core.wire import Exchanged, Fork, Observed, Reached, Streamed
from hands.core.wire import Text as Said
from hands.sessions.audit import AsideAnswered, Record

# The side question Claude Code asks of a fork of its session, which here holds nothing but the question.
ASIDE = CommandName("btw")
# No MCP server: with no built-in tools either, the question is all its request carries.
NO_SERVERS = '{"mcpServers": {}}'
# How long a side question's answer is waited on, from when its Claude Code is started: one reply, with thinking.
ASIDE_SECONDS = 120.0


class AsideFailed(Exception):
    """A side question has no answer: its Claude Code could not be started, ended first, answered with no words, or did
    not answer in time."""


def _unanswered(error: BaseException) -> str:
    """Why a question has no answer, as its line says it."""
    match error:
        case asyncio.CancelledError():
            return "its asker stopped waiting"
        case AsideFailed():
            return str(error)
        case _:
            # A fault of hands' own, said as itself: the line never gives a cause that did not happen.
            return repr(error)


@dataclass
class _Asked:
    """The question being asked now: the session its Claude Code was started under, and its answer, or why it has none."""

    # [LAW:one-source-of-truth] chosen by hands, so the request on the wire that asks the question is known as its own.
    session: SessionId
    answer: "asyncio.Future[str | AsideFailed]"


class Asides:
    """What answers hands' side questions: `ask` starts a Claude Code, asks it, and ends it."""

    def __init__(self, station: Station, record: Record) -> None:
        self._station = station
        self._record = record
        # One at a time: each is a process of its own, and nothing asked here is waited on by the user.
        self._one = asyncio.Lock()
        self._asked: _Asked | None = None

    async def ask(self, question: str) -> str:
        """The answer to `question`, from a Claude Code that is asked nothing else; raises AsideFailed when it has none."""
        loop = asyncio.get_running_loop()
        queued = loop.time()
        asked = _Asked(SessionId(str(uuid4())), loop.create_future())
        # When its Claude Code was started: never, for a question whose asker left while it waited its turn.
        began = math.inf

        def said(reply: str, failed: bool) -> None:
            ended = loop.time()
            started = min(began, ended)
            # [LAW:nothing-unseen] one line for each question, however it ended, in its turn or waiting for it.
            self._record(AsideAnswered(question, reply, failed, asked.session, started - queued, ended - started))

        try:
            async with self._one:
                began = loop.time()
                self._asked = asked
                try:
                    reply = await self._answer(asked, question)
                finally:
                    self._asked = None
        except BaseException as error:
            said(_unanswered(error), True)
            raise
        said(reply, False)
        return reply

    async def _answer(self, asked: _Asked, question: str) -> str:
        # The question as the characters it shows, behind the command, as the prompt its Claude Code opens with.
        opening = Command(ASIDE, pasted(question)).typed
        try:
            claude = await spawn(self._station, [*slim(brain_claude(), self._station.model, asked.session, (), NO_SERVERS), opening])
        except (Unstartable, OSError) as error:
            # No claude to run, or a question longer than a command line can be; a claude that cannot be run exits, and says why.
            raise AsideFailed(f"no Claude Code to ask: {error}") from error
        try:
            return await self._answered(claude, asked)
        finally:
            # [LAW:no-silent-failure] answered, failed, or given up on, its Claude Code is ended: none is left running.
            await claude.stop()

    async def _answered(self, claude: ClaudeCode, asked: _Asked) -> str:
        await asyncio.wait({asked.answer, claude.exit}, timeout=ASIDE_SECONDS, return_when=asyncio.FIRST_COMPLETED)
        # What its Claude Code showed says why there is no answer: a question it is still answering, or a screen over its input.
        if not asked.answer.done() and claude.exit.done():
            raise AsideFailed(f"its Claude Code exited ({claude.exit.result()}) before it answered; it showed:\n{claude.shown()}")
        if not asked.answer.done():
            raise AsideFailed(f"no answer in {ASIDE_SECONDS:.0f}s; its Claude Code showed:\n{claude.shown()}")
        match asked.answer.result():
            case AsideFailed() as failure:
                raise failure
            case said:
                return said

    def hear(self, observed: Observed) -> None:
        """A side question's answer, read from the wire: the reply to its own Claude Code's request."""
        asked = self._asked
        match observed:
            case Exchanged(session=session, kind=Fork(), reply=reply) if asked is not None and session == asked.session and not asked.answer.done():
                match reply:
                    case Reached(status=200, body=Streamed(message=message)):
                        said = " ".join(block.text for block in message.content if isinstance(block, Said)).strip()
                        # Read from the wire, not from what Claude Code shows: for a reply that did not end in words it
                        # shows words of its own.
                        ended = message.stop_reason == "end_turn" and bool(said)
                        asked.answer.set_result(said if ended else AsideFailed(f"the model's reply ended {message.stop_reason!r} with {said!r}"))
                    case _:
                        # Claude Code asks again after a request that failed; the question waits for that, or for its time.
                        logger.warning(f"a side question's request was answered {reply}")
            case _:
                pass
