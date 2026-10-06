"""The conversation page: the conversation with hands as the audit log holds it, what the user said, what hands said, and
each tool hands called, with a box to type to hands in.

It is served beside the phone's page, on its port and its addresses (`hands.voice.phonepage`), so a phone or the desk's
browser opens it from the address `hands phone` prints with `conversation` on its path. The page is open to anyone who
can reach the port; what it reads and sends is not: each request carries the phone's key, which the page takes from its
address's fragment as the phone's page does.

[LAW:one-source-of-truth] the conversation is the log's, folded into moments as `hands recall` folds it
(`hands.sessions.recall`); nothing here keeps a record of its own. Words typed are a hold of their own in the voice's
pipeline (`hands.voice.turnstop.Typed`), so they reach hands as words said do, and come back here as the log's.
"""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from importlib import resources
from pathlib import Path

from aiohttp import web

from hands.sessions.audit import Record, past
from hands.sessions.payload import Rejected
from hands.sessions.recall import Moment, Moments, Reading
from hands.sessions.wide import annotate, count, fail, unit
from hands.voice.phoneaddress import carries_key

# How many of the newest moments a page is given: an hour's talk and more, never the whole log.
NEWEST = 200

# How long a page's read waits for the conversation to change before answering with it as it is, well inside the minute
# a browser or a proxy between gives a request; and how often the log is looked at meanwhile, as `hands log` looks.
WAIT_SECONDS = 25.0
POLL_SECONDS = 0.25


class Conversation:
    """The log's moments, folded as far as the log has been read, for every page that reads them."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._moments = Moments(NEWEST)
        self._reading = Reading()
        # The log offset the next read begins at: 0 is the oldest segment kept, wherever retention has left it.
        self._offset = 0
        # [LAW:no-shared-mutable-globals] one read of the log at a time folds into the one set of moments.
        self._folding = asyncio.Lock()

    async def caught_up(self) -> int:
        """Fold in every line the log has gained since it was last read, off the loop; how many changes the moments have
        had, which moves whenever they do."""
        async with self._folding:
            await asyncio.to_thread(self._catch_up)
        return self._moments.changes

    def newest(self) -> list[Moment]:
        return self._moments.moments()

    def _catch_up(self) -> None:
        while True:
            lines, offset = past(self._directory, self._offset)
            for entry in self._reading.entries(lines):
                self._moments.take(entry)
            # A segment read to its end that a later one follows goes on at the later one; one that does not is the end.
            moved, self._offset = offset != self._offset, offset
            if not moved:
                return


def parse_seen(query: str | None) -> int | None:
    """The changes a page has seen, as its read says them; None for a page that has seen none, whose read waits for nothing."""
    match query:
        case None:
            return None
        case digits if digits.isdecimal():
            return int(digits)
        case _:
            raise Rejected("seen is the changes a page was last given, a whole number")


def parse_typed(body: bytes) -> str:
    """The words a page sent, as its request's body; raises Rejected naming what is wrong with them."""
    try:
        sent: object = json.loads(body)
    except ValueError as error:
        # Not JSON, or not text at all: JSONDecodeError and UnicodeDecodeError are both ValueErrors.
        raise Rejected(f"words typed are JSON: {error}") from error
    match sent:
        case {"text": str(text)} if text.strip():
            return text.strip()
        case _:
            raise Rejected("words typed are {text}, with something in it")


def conversation_routes(conversation: Conversation, typed: Callable[[str], Awaitable[None]], key: str, record: Record) -> list[web.RouteDef]:
    """The page, the read of the conversation, and the words typed into it.

    [LAW:nothing-unseen] each read is one `conversation.read` event, how long it waited and how many moments it gave;
    each send one `conversation.typed`, how much was typed; each refused says why.
    """
    page = resources.files("hands.voice").joinpath("conversation.html").read_text()

    async def shown(_request: web.Request) -> web.Response:
        return web.Response(text=page, content_type="text/html", headers={"Cache-Control": "no-store"})

    async def read(request: web.Request) -> web.Response:
        with unit("conversation.read", record, counts=("moments",)):
            if not carries_key(request.headers.get("Authorization", ""), key):
                fail("read without the phone's key")
                return web.Response(status=401, text="this page's address is missing the phone's key; open it from `hands phone`")
            try:
                seen = parse_seen(request.query.get("seen"))
            except Rejected as error:
                fail(str(error))
                return web.Response(status=400, text=str(error))
            began = time.monotonic()
            changes = await conversation.caught_up()
            # [LAW:no-ambient-temporal-coupling] the page says what it has seen, so a change made between its reads is
            # never missed: it is answered at once with whatever has changed since.
            while changes == seen and time.monotonic() - began < WAIT_SECONDS:
                await asyncio.sleep(POLL_SECONDS)
                changes = await conversation.caught_up()
            moments = conversation.newest()
            annotate(seen=seen, changes=changes, waited_ms=round((time.monotonic() - began) * 1000, 1))
            count(moments=len(moments))
            return web.json_response({"changes": changes, "moments": [_shown(moment) for moment in moments]})

    async def sent(request: web.Request) -> web.Response:
        with unit("conversation.typed", record):
            if not carries_key(request.headers.get("Authorization", ""), key):
                fail("typed without the phone's key")
                return web.Response(status=401, text="this page's address is missing the phone's key; open it from `hands phone`")
            try:
                text = parse_typed(await request.read())
            except Rejected as error:
                fail(str(error))
                return web.Response(status=400, text=str(error))
            annotate(chars=len(text))
            await typed(text)
            return web.Response(status=202)

    return [web.get("/conversation", shown), web.get("/conversation/moments", read), web.post("/conversation/typed", sent)]


def _shown(moment: Moment) -> dict[str, str]:
    return {"at": moment.at.isoformat(), "kind": moment.kind, "heading": moment.heading, "text": moment.text}
