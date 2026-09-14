"""The unix socket the shims post to."""

import socket
from pathlib import Path
from uuid import uuid4

from aiohttp import web
from loguru import logger

from hands.core.events import PermissionRequested
from hands.core.session import RequestId
from hands.sessions.home import Home
from hands.sessions.hooks import hook_output, parse_hook
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions


async def serve_hooks(home: Home, sessions: Sessions) -> web.AppRunner:
    """Listen on the home's socket until the returned runner is cleaned up."""

    async def hook(request: web.Request) -> web.Response:
        body = await request.read()
        try:
            event = parse_hook(body, home=home, at=sessions.now(), request=RequestId(uuid4().hex))
        except Rejected as error:
            # The shim prints this reply, so the session that sent the hook shows why.
            logger.error(f"rejected hook: {error}")
            return web.Response(status=400, text=str(error))
        match event:
            case PermissionRequested():
                # The one hook that waits: the response body is the answer Claude Code reads.
                output = hook_output(await sessions.ask(event))
                return web.Response(status=204) if output is None else web.json_response(output)
            case _:
                await sessions.apply(event)
                return web.Response(status=204)

    app = web.Application()
    app.router.add_post("/hook", hook)
    # A hook Claude Code killed closes its connection; cancelling the handler lets go of its wait,
    # so a reply decided afterwards is logged as unheard instead of as delivered.
    runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
    await runner.setup()
    claim_socket(home.socket)
    await web.UnixSite(runner, str(home.socket)).start()
    return runner


def claim_socket(path: Path) -> None:
    """Clear a socket left by a dead daemon; refuse one a live daemon is listening on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.connect(str(path))
    except FileNotFoundError:
        return
    except ConnectionRefusedError:
        path.unlink()
        return
    finally:
        probe.close()
    raise RuntimeError(f"another hands daemon is already listening on {path}")
