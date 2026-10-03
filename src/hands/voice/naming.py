"""The one task that names sessions, off the voice path: after each turn a session finishes, the model is asked whether
its name still fits the work, and a new one is held for the session's next prompt.

A session is spoken as its project and then its name, so the name says what the work is, in three words at most, and
nothing of the project.
"""

import asyncio
import time

from loguru import logger

from hands.sessions.audit import Named, NamingOutcome, Record
from hands.sessions.names import Finished, Names
from hands.sessions.payload import Rejected
from hands.sessions.transcript import session_name
from hands.voice.speech import bounded
from hands.voice.summary import SUMMARY_FAILURES, Summariser

NAME_INSTRUCTION = """\
You name a coding session so a person can tell it apart from the others they run, and name it back by voice. You are \
given the name it has now, if any, and the last thing the session said, which says what it did and where its work stands.

Reply with the name and nothing else: a descriptive phrase of at most three words, in lowercase plain words, saying \
what the work is. The person hears the project's name before it, so leave the project out.

Keep the name it has when it still fits the work, and reply with it exactly as given: a name that changes every turn \
cannot be learned. Give a new one only when the work has moved on to something else.

Do not reply like these:
- "Name: naming fix"  (a label before the name)
- "fixing the session naming bug in hands"  (more than three words, and names the project)
- "work"  (says nothing about the work)

A good reply:
naming fix
"""

# A name is a few words; this is room for them and nothing else.
NAME_MAX_TOKENS = 30
NAME_TIMEOUT_SECONDS = 60.0

# How much of the session's last reply the model is shown: a session told to end on a concise overview writes less.
CLOSING_SHOWN = 1500

# The most words a name may have: it is said before every sentence about its session.
NAME_WORDS = 3


class NotAName(Exception):
    """The model's reply is not a name hands can give a session."""


async def keep_naming(names: Names, name: Summariser, record: Record) -> None:
    """Judge each finished turn's session name, one at a time, until cancelled."""
    while True:
        await judge(await names.next_finished(), names, name, record)


async def judge(turn: Finished, names: Names, name: Summariser, record: Record) -> None:
    """Ask whether the session's name still fits the turn it finished; hold a new one for its next prompt; audit the judging."""
    began = time.monotonic()

    def judged(outcome: NamingOutcome, before: str | None, named: str | None, reply: str | None, error: str | None) -> None:
        record(Named(turn.session, outcome, before, named, reply, error, time.monotonic() - began))

    try:
        before = await asyncio.to_thread(session_name, turn.transcript)
    except (Rejected, OSError) as error:
        # [LAW:no-silent-failure] a name hands cannot read cannot be judged; the session keeps it, and the log says why.
        logger.error(f"cannot read the name of session {turn.session} from {turn.transcript}, so it is not judged: {error}")
        judged("unread", None, None, None, str(error))
        return
    try:
        reply = await name(asked(before, turn.closing))
    except SUMMARY_FAILURES as error:
        logger.error(f"the model gave no name for session {turn.session}: {type(error).__name__}: {error}")
        judged("failed", before, None, None, f"{type(error).__name__}: {error}")
        return
    try:
        decided = parsed(reply)
    except NotAName as error:
        logger.error(f"the model's name for session {turn.session} is not one: {error}")
        judged("refused", before, None, reply, str(error))
        return
    if decided == before:
        judged("kept", before, decided, reply, None)
        return
    names.rename(turn.session, decided)
    judged("renamed", before, decided, reply, None)


def asked(before: str | None, closing: str) -> str:
    """What the model is shown: the name the session has, and the last thing it said."""
    has = f"Its name now: {before}" if before is not None else "It has no name yet."
    return f"{has}\n\nThe last thing it said:\n\n{bounded(closing, CLOSING_SHOWN)}"


def parsed(reply: str) -> str:
    """The name in the model's reply: its words, without quotes or a closing stop around them.

    Raises NotAName for a reply of more than three words, or of none.
    """
    # [LAW:parse-dont-validate] the one place a reply becomes a name.
    words = reply.strip().strip("\"'`.").split()
    if not 0 < len(words) <= NAME_WORDS:
        raise NotAName(f"{reply!r} is {len(words)} words, and a name is 1 to {NAME_WORDS}")
    return " ".join(words)
