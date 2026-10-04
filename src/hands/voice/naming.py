"""The one task that names sessions, off the voice path: after each turn a session finishes, the model is asked whether
its name still fits the work, and a new one is held for the session's next prompt.

A session is spoken as its project and then its name, so the name says what the work is, in three words at most, and
nothing of the project.
"""

import asyncio
from collections.abc import Callable, Sequence
from enum import StrEnum

from hands.core.session import Membership
from hands.sessions.audit import Record
from hands.sessions.names import Finished, Names
from hands.sessions.payload import Rejected
from hands.sessions.transcript import session_name
from hands.sessions.wide import annotate, fail, unit
from hands.voice.summary import SUMMARY_FAILURES, Summariser

NAME_INSTRUCTION = """\
You name a coding session so a person can tell it apart from the others they run, and name it back by voice. You are \
given the name it has now, if any, the names of the other sessions in its project, and the last thing the session \
said, which says what it did and where its work stands.

Reply with the name and nothing else: a descriptive phrase of at most three words, in lowercase plain words, saying \
what the work is. The person hears the project's name before it, so leave the project out.

Keep the name it has when it still fits the work, and reply with it exactly as given: a name that changes every turn \
cannot be learned. Give a new one only when the work has moved on to something else. A new name is never one the other sessions have: \
the person tells sessions apart by it.

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

# How much of the end of the session's last reply the model is shown, where a session told to close on a concise
# overview of where its work stands writes one.
CLOSING_SHOWN = 1500

# The most words a name may have: it is said before every sentence about its session.
NAME_WORDS = 3


class Judged(StrEnum):
    """What came of judging a session's name."""

    RENAMED = "renamed"
    KEPT = "kept"
    UNREAD = "unread"
    FAILED = "failed"
    REFUSED = "refused"


class NotAName(Exception):
    """The model's reply is not a name hands can give a session."""


async def keep_naming(names: Names, live: Callable[[], Sequence[Membership]], name: Summariser, record: Record) -> None:
    """Judge each finished turn's session name, one at a time, until cancelled; `live` is every session not ended."""
    while True:
        await judge(await names.next_finished(), names, live(), name, record)


async def judge(turn: Finished, names: Names, live: Sequence[Membership], name: Summariser, record: Record) -> None:
    """Ask whether the session's name still fits the turn it finished, and hold a new one for its next prompt, as one
    unit of work."""
    session = turn.membership.id
    others = [other for other in live if other.cwd == turn.membership.cwd and other.id != session]
    # [LAW:nothing-unseen] `judged` says what came of it: `renamed` decided a new name, given at the session's next
    # prompt; `kept` found the name it has still fits; `unread` could not read the name it has; `failed` had no answer
    # from the model; `refused` had an answer that is not a name, which `reply` holds. The last three fail the unit.
    with unit("name.judged", record):
        annotate(session=session)
        try:
            held = await asyncio.to_thread(lambda: [session_name(each.transcript) for each in (turn.membership, *others)])
        except (Rejected, OSError) as error:
            # [LAW:no-silent-failure] names hands cannot read cannot be judged against; the session keeps its own.
            annotate(judged=Judged.UNREAD)
            fail(f"cannot read the names of the session and those beside it in {turn.membership.cwd}: {error}")
            return
        # [LAW:one-source-of-truth] a name decided and not yet given is the session's name from its next prompt on, so
        # it is the one judged, and the one the others' are.
        before, *beside = [names.current(each.id, given) for each, given in zip((turn.membership, *others), held, strict=True)]
        annotate(before=before)
        try:
            reply = await name(asked(before, [each for each in beside if each is not None], turn.closing))
        except SUMMARY_FAILURES as error:
            annotate(judged=Judged.FAILED)
            fail(f"the model gave no name: {type(error).__name__}: {error}")
            return
        annotate(reply=reply)
        try:
            decided = parsed(reply, before)
        except NotAName as error:
            annotate(judged=Judged.REFUSED)
            fail(f"the model's name is not one: {error}")
            return
        annotate(name=decided)
        if decided == before:
            annotate(judged=Judged.KEPT)
            return
        names.rename(session, decided, held[0])
        annotate(judged=Judged.RENAMED)


def asked(before: str | None, beside: Sequence[str], closing: str) -> str:
    """What the model is shown: the name the session has, the names beside it in its project, and the end of the last
    thing it said."""
    has = f"Its name now: {before}" if before is not None else "It has no name yet."
    others = f"The other sessions in its project: {'; '.join(beside)}" if beside else "No other session is in its project."
    shown = closing if len(closing) <= CLOSING_SHOWN else f"(the start cut) ...{closing[-CLOSING_SHOWN:]}"
    return f"{has}\n\n{others}\n\nThe last thing it said:\n\n{shown}"


def parsed(reply: str, before: str | None) -> str:
    """The name in the model's reply: its words, without quotes or a closing stop around them.

    Raises NotAName for a reply of no words, or of more than three that is not the name the session already has, which
    the user may have given it with /rename at any length.
    """
    # [LAW:parse-dont-validate] the one place a reply becomes a name.
    words = reply.strip().strip("\"'`.").split()
    named = " ".join(words)
    if named != before and not 0 < len(words) <= NAME_WORDS:
        raise NotAName(f"{reply!r} is {len(words)} words, and a new name is 1 to {NAME_WORDS}")
    return named
