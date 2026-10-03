"""The unix socket the shims post to."""

import asyncio
import socket
from pathlib import Path
from uuid import uuid4

from aiohttp import web
from loguru import logger

from hands.core.events import Attached, PermissionRequested, Prompted, Stopped
from hands.core.session import Membership, RequestId
from hands.sessions.home import Home
from hands.sessions.audit import NameGiven, NameWithheld, Record
from hands.sessions.hooks import hook_output, name_output, parse_hook
from hands.sessions.names import Due, Finished, Names
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions
from hands.sessions.transcript import session_name


async def serve_hooks(home: Home, sessions: Sessions, names: Names, record: Record) -> web.AppRunner:
    """Listen on the home's socket until the returned runner is cleaned up.

    A finished turn has its session's name judged again, and a prompt is answered with the name decided since the last.
    """

    async def hook(request: web.Request) -> web.Response:
        body = await request.read()
        try:
            said = parse_hook(body, home=home, at=sessions.now(), heard=sessions.stamp(), request=RequestId(uuid4().hex))
        except Rejected as error:
            # The shim prints this reply, so the session that sent the hook shows why.
            logger.error(f"rejected hook: {error}")
            return web.Response(status=400, text=str(error))
        match said.joining:
            case Attached() as joining:
                await sessions.apply(joining)
            case None:
                pass
        match said.happened:
            case PermissionRequested() as asked:
                # The one hook answered: the response body is the answer Claude Code reads.
                output = hook_output(await sessions.ask(asked))
                return web.Response(status=204) if output is None else web.json_response(output)
            case Stopped() as stopped:
                await sessions.stop(stopped)
                match (stopped.closing, sessions.membership(stopped.session)):
                    case (str() as closing, Membership() as membership):
                        names.finished(Finished(membership, closing))
                    case _:
                        # A turn that closed on no reply has nothing to name it by, and a session that never joined
                        # has no transcript to read its name from; either keeps the name it has.
                        pass
                return web.Response(status=204)
            case Prompted() as prompted:
                await sessions.apply(prompted)
                # [LAW:no-ambient-temporal-coupling] a hook is the only way to hand Claude Code a title, and a prompt
                # the soonest one after a name is decided: the name waits here for it.
                match (names.due(prompted.session), sessions.membership(prompted.session)):
                    case (None, _):
                        return web.Response(status=204)
                    case (Due() as due, Membership(transcript=transcript)):
                        try:
                            held = await asyncio.to_thread(session_name, transcript)
                        except (Rejected, OSError) as error:
                            # [LAW:no-silent-failure] a title hands cannot read may be one the user set: the name is
                            # not given over it, and the log says why.
                            logger.error(f"cannot read the name of session {prompted.session} from {transcript}, so {due.name!r} is not given: {error}")
                            record(NameWithheld(prompted.session, due.name, due.against, None, str(error)))
                            return web.Response(status=204)
                        if held != due.against:
                            # The latest name set wins: one set since this was decided is newer than the decision.
                            record(NameWithheld(prompted.session, due.name, due.against, held, None))
                            return web.Response(status=204)
                        record(NameGiven(prompted.session, due.name))
                        return web.json_response(name_output(due.name))
                    case (Due(), None):
                        # Unreachable while a name is only decided for a session whose Stop found it joined.
                        raise AssertionError(f"a name is due for session {prompted.session}, which never joined")
            case happened:
                await sessions.apply(happened)
                return web.Response(status=204)

    app = web.Application()
    app.router.add_post("/hook", hook)

    async def let_go(_app: web.Application) -> None:
        sessions.release_waiting()

    # Cleanup waits for every handler to return, and a permission hook would wait out its deadline; runs before that wait.
    app.on_shutdown.append(let_go)
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
