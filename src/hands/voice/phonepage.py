"""The phone's page: served over HTTPS to the LAN and the tailnet, where a phone's browser opens it and calls hands; and
the conversation page beside it (`hands.voice.conversationpage`).

Where it is reached, and the certificate and key it is reached with, are `hands.voice.phoneaddress`'s.

The page is open to anyone who can reach the port; a call is not. The page sends the phone's key with its offer, and
an offer without it is refused.
"""

import asyncio
import datetime
import json
import ssl
from collections.abc import Awaitable, Callable, Coroutine
from functools import partial
from importlib import resources
from pathlib import Path
from typing import Never

from aiohttp import web
from loguru import logger

from hands.sessions.audit import Record
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.sessions.wide import annotate, begun, continuing, unit
from hands.voice.conversationpage import Conversation, conversation_routes
from hands.voice.phone import Asked, CallDeclined, CallRefused, Offer, Phone, call_ended
from hands.voice.phoneaddress import PHONE_HOST, PHONE_PORT, Tailnet, Untailed, carries_key, own_certificate, phone_key, tailnet

# How often a page served under the tailnet's name asks Tailscale for its certificate again: Tailscale renews one a
# month before its 90 days are out, so a day is far inside the time a renewed one has before the old one ends.
RENEW_SECONDS = 24 * 60 * 60.0


def parse_offer(body: object) -> Offer:
    """A page's offer as hands takes it; raises Rejected naming what is wrong with it."""
    match body:
        case {"sdp": str(sdp), "type": "offer", "rate": int(rate), "page": str(page), "claim": str(claim)} if rate > 0 and page and claim in ("take", "resume"):
            return Offer(sdp, "offer", rate, Asked(page, claim))
        case _:
            raise Rejected("an offer is {sdp, type: offer, rate, page, claim: take or resume}")


def phone_app(phone: Phone, key: str, record: Record) -> web.Application:
    """The page, and the one route that takes its calls."""
    page = resources.files("hands.voice").joinpath("phone.html").read_text()

    async def shown(_request: web.Request) -> web.Response:
        return web.Response(text=page, content_type="text/html", headers={"Cache-Control": "no-store"})

    async def offered(request: web.Request) -> web.Response:
        remote = request.remote or "unknown"
        began = begun()
        if not carries_key(request.headers.get("Authorization", ""), key):
            call_ended(record, began, remote, None, CallRefused("offered without the phone's key"))
            return web.Response(status=401, text="this page's address is missing the phone's key; open it from `hands phone`")
        try:
            offer = parse_offer(await request.json())
        except (Rejected, json.JSONDecodeError) as error:
            call_ended(record, began, remote, None, CallRefused(str(error)))
            return web.Response(status=400, text=str(error))
        except BaseException as error:
            # The body could not be read at all, or the request went as it was.
            call_ended(record, began, remote, None, error)
            raise
        match await phone.answer(offer, remote, began):
            case CallDeclined():
                return web.Response(status=409, text="another page has hands' phone; press Connect to take it")
            case answer:
                return web.json_response({"sdp": answer.sdp, "type": answer.type})

    app = web.Application()
    app.router.add_get("/", shown)
    app.router.add_post("/offer", offered)
    return app


async def serve_phone(phone: Phone, typed: Callable[[str], Awaitable[None]], home: Home, record: Record) -> Never:
    """Serve the page and take its calls on every address this machine has, until cancelled; under the tailnet's name,
    with the certificate Tailscale renews, asked for again every RENEW_SECONDS. The conversation page is served beside
    it, under the same key, handing `typed` what is typed into it.

    [LAW:nothing-unseen] serving it is one unit of work, `phone.served`: the port, and the tailnet name it is served
    under, or, served on the LAN alone, why Tailscale gave it none.
    """
    key = phone_key(home)
    app = phone_app(phone, key, record)
    app.add_routes(conversation_routes(Conversation(home.audit), typed, key, record))
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    try:
        with unit("phone.served", record):
            own = _context(*own_certificate(home, datetime.datetime.now(datetime.UTC)))
            kept: Callable[[], Coroutine[object, object, Never]]
            match await tailnet(home):
                case Tailnet(name=name, cert=cert, key=key):
                    tailed = _context(cert, key)

                    def chosen(connection: ssl.SSLObject, server_name: str | None, _context: ssl.SSLContext) -> None:
                        # [LAW:single-enforcer] the one place a name picks its certificate: the name the phone asked for.
                        if server_name == name:
                            connection.context = tailed

                    own.sni_callback = chosen  # pyright: ignore[reportAttributeAccessIssue]  (typeshed types the callback for SSLSocket alone)
                    annotate(tailnet=name)
                    kept = partial(_keep_renewed, tailed, home)
                case Untailed(reason=reason):
                    # [LAW:no-silent-failure] the page is still served on the LAN; why the tailnet name is not, is said.
                    logger.warning(f"the phone's page is served on the LAN alone: {reason}")
                    annotate(untailed=reason)
                    kept = _keep_served
            # Every request the site takes is work for no unit open here: the server's tasks copy the context it is
            # started in, and `phone.served` ends as the page comes up.
            with continuing(None):
                await _site(runner, own)
            # The port the page was bound on, which a phone's address names.
            [(_host, port, *_)] = runner.addresses
            annotate(port=port)
        await kept()
    finally:
        await runner.cleanup()


async def _keep_renewed(tailed: ssl.SSLContext, home: Home) -> Never:
    while True:
        await asyncio.sleep(RENEW_SECONDS)
        match await tailnet(home):
            case Tailnet(cert=cert, key=key):
                # A context's chain, loaded again, is the one every handshake after it shows.
                tailed.load_cert_chain(cert, key)
            case Untailed(reason=reason):
                logger.warning(f"the tailnet's certificate for the phone's page was not renewed: {reason}")


async def _keep_served() -> Never:
    while True:
        # Served until cancelled: there is no certificate of Tailscale's to renew.
        await asyncio.sleep(RENEW_SECONDS)


def _context(cert: Path, key: Path) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    return context


async def _site(runner: web.AppRunner, context: ssl.SSLContext) -> None:
    try:
        await web.TCPSite(runner, PHONE_HOST, PHONE_PORT, ssl_context=context).start()
    except OSError as error:
        raise RuntimeError(f"cannot serve the phone's page on port {PHONE_PORT}: {error}") from error
