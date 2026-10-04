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
`claude ... -- "/btw ..."`. Claude Code asks its opening prompt itself once its input is up, so nothing is typed into it and
nothing depends on when its input comes up (a `/btw` typed as the input came up was left in it unsent, 12 of 12; as the
opening prompt it went out 0.4s after the start, 6 of 6). Its answer is read from the wire.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from loguru import logger

from hands.brain.process import ClaudeCode, Station, Unstartable, brain_claude, slim, spawn
from hands.core.effects import Command
from hands.core.session import CommandName, SessionId, pasted
from hands.core.wire import Exchanged, Fork, Observed, Reached, Streamed
from hands.core.wire import Text as Said
from hands.sessions.audit import Record
from hands.sessions.wide import annotate, since, unit

# The side question Claude Code asks of a fork of its session, which here holds nothing but the question.
ASIDE = CommandName("btw")
# No built-in tool and no MCP server, whatever the brain's own setup gives it: the question is all its request carries.
CLOSED = ("--tools", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers": {}}')
# How long a side question's answer is waited on, from when its Claude Code is started: one reply, with thinking.
ASIDE_SECONDS = 120.0


def aside_command(claude: Path, model: str, session: SessionId, question: str) -> list[str]:
    """A side question's command line: the slim Claude Code, closed, opening with the question."""
    # The question as the characters it shows, behind the command, as the prompt its Claude Code opens with. `--` ends
    # the options: without it, --mcp-config takes every word up to the next option, the prompt as a second config file.
    return [*slim(claude, model, session), *CLOSED, "--", Command(ASIDE, pasted(question)).typed]


class AsideKind(StrEnum):
    """What a side question is asked for."""

    SUMMARY = "summary"  # a backlog's tickets or a session's turns, each as a sentence to be said
    EXPLANATION = "explanation"  # what a session is doing, from its progress
    NAME = "name"  # a session's name, judged from a turn it finished
    LINE = "line"  # the line an old tool result goes as in the brain's context


class Unanswered(StrEnum):
    """Why a side question has no answer."""

    UNSTARTED = "unstarted"  # no Claude Code could be started to ask it
    EXITED = "exited"  # its Claude Code ended before it answered
    TIMED_OUT = "timed_out"  # no answer in ASIDE_SECONDS
    WORDLESS = "wordless"  # the model's reply did not end in words


class AsideFailed(Exception):
    """A side question has no answer, `why` says which way, and its message what was seen of it."""

    def __init__(self, why: Unanswered, seen: str) -> None:
        super().__init__(seen)
        self.why = why


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

    async def ask(self, kind: AsideKind, question: str) -> str:
        """The answer to `question`, from a Claude Code that is asked nothing else; raises AsideFailed when it has none."""
        asked = _Asked(SessionId(str(uuid4())), asyncio.get_running_loop().create_future())
        # [LAW:nothing-unseen] one event for each question, however it ends: answered, failed, or left by its asker, in its
        # turn or still waiting for it. Asked inside another unit of work, such as a background pass, it is that one's part.
        with unit("brain.aside", self._record):
            annotate(kind=kind, question=question, session=asked.session)
            queued = time.monotonic()
            try:
                await self._one.acquire()
            finally:
                annotate(queued_ms=since(queued))
            # answering_ms only once it has its turn: a question left while it waited was never answered at all.
            answering = time.monotonic()
            self._asked = asked
            try:
                reply = await self._answer(asked, question)
            except AsideFailed as failure:
                annotate(unanswered=failure.why)
                raise
            finally:
                self._asked = None
                self._one.release()
                annotate(answering_ms=since(answering))
            annotate(reply=reply)
            return reply

    async def _answer(self, asked: _Asked, question: str) -> str:
        try:
            claude = await spawn(self._station, aside_command(brain_claude(self._station.inherited), self._station.model, asked.session, question))
        except (Unstartable, OSError) as error:
            # No claude to run, or a question longer than a command line can be; a claude that cannot be run exits, and says why.
            raise AsideFailed(Unanswered.UNSTARTED, f"no Claude Code to ask: {error}") from error
        try:
            return await self._answered(claude, asked)
        finally:
            # [LAW:no-silent-failure] answered, failed, or given up on, its Claude Code is ended: none is left running.
            await claude.stop()

    async def _answered(self, claude: ClaudeCode, asked: _Asked) -> str:
        await asyncio.wait({asked.answer, claude.exit}, timeout=ASIDE_SECONDS, return_when=asyncio.FIRST_COMPLETED)
        # What its Claude Code showed says why there is no answer: a question it is still answering, or a screen over its input.
        if not asked.answer.done() and claude.exit.done():
            raise AsideFailed(Unanswered.EXITED, f"its Claude Code exited ({claude.exit.result()}) before it answered; it showed:\n{claude.shown()}")
        if not asked.answer.done():
            raise AsideFailed(Unanswered.TIMED_OUT, f"no answer in {ASIDE_SECONDS:.0f}s; its Claude Code showed:\n{claude.shown()}")
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
                        asked.answer.set_result(said if ended else AsideFailed(Unanswered.WORDLESS, f"the model's reply ended {message.stop_reason!r} with {said!r}"))
                    case _:
                        # Claude Code asks again after a request that failed; the question waits for that, or for its time.
                        logger.warning(f"a side question's request was answered {reply}")
            case _:
                pass
