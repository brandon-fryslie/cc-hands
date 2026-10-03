"""Refocus: once hands has told the user of a session's turn or its question, that session is the focus, so what the
user says next, naming no session, is taken first as said to it: an answer to its question, or its next prompt.

It moves the one focus the user moves by voice [LAW:one-source-of-truth], so it holds as theirs does, across restarts and
until something moves it again, and the brain reads it where it reads theirs. A default target, never a lock: a session
the user names still gets what they say to it.
"""

import asyncio

from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from hands.core.session import SessionId
from hands.sessions.audit import Record, Refocused
from hands.sessions.focus import set_focus
from hands.sessions.home import Home
from hands.voice.speech import Told


class Refocus(FrameProcessor):
    """Behind the model's stage, where a telling's Told arrives in order with the user's words: the focus moves to the
    session told of, and the Told goes no further."""

    def __init__(self, home: Home, record: Record) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._home = home
        self._record = record

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case Told(session=session):
                self._record(Refocused(session, await refocus(self._home, session)))
            case _:
                await self.push_frame(frame, direction)


async def refocus(home: Home, session: SessionId) -> str | None:
    """Move the focus to `session`, just told of: why it could not be moved, or None once it is."""
    try:
        await asyncio.to_thread(set_focus, home, session)
    except OSError as error:
        # [LAW:no-silent-failure] logged, and the Refocused line is an error: the telling stands, and the focus is as it was.
        logger.error(f"cannot move the focus to session {session}, which was just told: {error}")
        return str(error)
    return None
