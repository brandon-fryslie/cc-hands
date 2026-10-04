"""The unix socket the shims post to, and the loopback route Claude Code posts the text it displays to."""

import asyncio
from pathlib import Path
from uuid import uuid4

from aiohttp import web
from hands.core.events import Attached, PermissionRequested, Prompted, Stopped
from hands.core.session import Membership, RequestId
from hands.sessions.home import Home
from hands.sessions.audit import Record
from hands.sessions.hooks import hook_output, name_output, parse_display, parse_hook
from hands.sessions.names import Due, Finished, NameGiven, NameUnread, NameWithheld, Names
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions
from hands.sessions.transcript import session_name
from hands.sessions.wide import annotate, fail, unit


async def serve_hooks(home: Home, sessions: Sessions, names: Names, record: Record) -> web.AppRunner:
    """Listen on the home's socket until the returned runner is cleaned up.

    A finished turn has its session's name judged again, and a prompt is answered with the name decided since the last.
    """

    async def hook(request: web.Request) -> web.Response:
        # [LAW:nothing-unseen] one event for each hook posted, held open as long as the hook holds its session: the
        # event's duration is how long Claude Code waited on hands, and its facts say which branch it was answered by.
        with unit("hook", record):
            return await answer(await request.read())

    async def answer(body: bytes) -> web.Response:
        try:
            said = parse_hook(body, home=home, at=sessions.now(), heard=sessions.stamp(), request=RequestId(uuid4().hex))
        except Rejected as error:
            # The shim prints this reply, so the session that sent the hook shows why.
            fail(f"rejected hook: {error}")
            return web.Response(status=400, text=str(error))
        annotate(hook=said.name, session=said.session)
        match said.joining:
            case Attached() as joining:
                await sessions.apply(joining)
            case None:
                pass
        match said.happened:
            case PermissionRequested() as asked:
                # The one hook answered: the response body is the answer Claude Code reads.
                reply = await sessions.ask(asked)
                annotate(reply=reply)
                output = hook_output(reply)
                return web.Response(status=204) if output is None else web.json_response(output)
            case Stopped() as stopped:
                annotate(stop=await sessions.stop(stopped))
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
                        annotate(name=None)
                        return web.Response(status=204)
                    case (Due() as due, Membership(transcript=transcript)):
                        try:
                            held = await asyncio.to_thread(session_name, transcript)
                        except (Rejected, OSError) as error:
                            # [LAW:no-silent-failure] a title hands cannot read may be one the user set: the name is
                            # not given over it, and the event says why.
                            annotate(name=NameUnread(due.name, due.against))
                            fail(f"cannot read the name of session {prompted.session} from {transcript}, so {due.name!r} is not given: {error}")
                            return web.Response(status=204)
                        if held != due.against:
                            # The latest name set wins: one set since this was decided is newer than the decision.
                            annotate(name=NameWithheld(due.name, due.against, held))
                            return web.Response(status=204)
                        annotate(name=NameGiven(due.name))
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


async def serve_display(sessions: Sessions, host: str, port: int, path: str, record: Record) -> web.AppRunner:
    """Listen on `host`:`port` for MessageDisplay alone, until the returned runner is cleaned up.

    Raises RuntimeError when the port is taken: a daemon that cannot hear what Claude writes does not start as though it
    could, and the plugin's hooks name this port alone.
    """

    async def displayed(request: web.Request) -> web.Response:
        # One event for each post, as each hook on the socket is: this route takes MessageDisplay alone.
        with unit("hook", record):
            try:
                said = parse_display(await request.read(), at=sessions.now())
            except Rejected as error:
                # Claude Code logs a hook's failure in its debug log; the event says what was refused, and why.
                fail(f"rejected display: {error}")
                return web.Response(status=400, text=str(error))
            annotate(hook="MessageDisplay", session=said.session)
            await sessions.apply(said)
            return web.Response(status=204)

    app = web.Application()
    app.router.add_post(path, displayed)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    try:
        await web.TCPSite(runner, host, port).start()
    except OSError as error:
        await runner.cleanup()
        raise RuntimeError(f"cannot listen for the text Claude Code displays on {host}:{port}: {error}") from error
    return runner


def claim_socket(path: Path) -> None:
    """Clear a socket an earlier daemon left in the home, so this one can listen there.

    [LAW:single-enforcer] the daemon holds the home's lock before it listens (hands.daemon.cli.hold), so any socket here
    is an earlier run's, never a live daemon's.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
