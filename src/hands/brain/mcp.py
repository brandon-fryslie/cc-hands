"""hands' MCP server: the intermediary's tools, served over MCP's streamable HTTP transport to the brain.

    claude --mcp-config '{"mcpServers": {"hands": {"type": "http", "url": "<the server's url>"}}}'

The second adapter over the tool bodies in hands.voice.tools, beside Pipecat's. It speaks the part of MCP a tools-only
server needs: initialize, ping, tools/list, and tools/call, each POSTed as JSON-RPC and answered as JSON. It opens no
event stream of its own, so a GET is refused, as the transport allows.

Its tools type into sessions and answer permissions, and a port on 127.0.0.1 is reachable by every local process and
every web page, so each request carries a token only the brain's --mcp-config holds.
"""

import hmac
import inspect
import json
import secrets
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from aiohttp import web
from loguru import logger

from hands.sessions.audit import McpConnected, Record
from hands.sessions.payload import Payload, Rejected
from hands.voice.tools import Tool

# What the brain's --mcp-config names this server; its tools reach the model as mcp__hands__<tool>.
SERVER_NAME = "hands"
PATH = "/mcp"

# JSON-RPC's codes for a request that names no method this server has, and for one whose params do not parse.
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
PARSE_ERROR = -32700


@dataclass(frozen=True)
class McpServer:
    url: str
    token: str
    runner: web.AppRunner

    async def close(self) -> None:
        await self.runner.cleanup()

    def config(self) -> str:
        """The --mcp-config that points a Claude Code at this server and at nothing else, with the token that lets it in."""
        return json.dumps({"mcpServers": {SERVER_NAME: {"type": "http", "url": self.url, "headers": {"Authorization": f"Bearer {self.token}"}}}})


async def serve_mcp(tools: Sequence[Tool], record: Record) -> McpServer:
    """Listen on a free local port until the returned server is closed."""
    by_name = {tool.name: tool for tool in tools}
    token = secrets.token_urlsafe(32)

    @web.middleware
    async def admitted(request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]) -> web.StreamResponse:
        # [LAW:single-enforcer] the one door: nothing reaches a tool without the token.
        if not hmac.compare_digest(request.headers.get("Authorization", ""), f"Bearer {token}"):
            logger.warning(f"the MCP server refused a {request.method} without its token, from {request.remote} (Origin {request.headers.get('Origin')})")
            return web.Response(status=401)
        return await handler(request)

    async def post(request: web.Request) -> web.Response:
        try:
            message = json.loads(await request.read())
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            logger.warning(f"the MCP server was sent a body that is not JSON: {error}")
            return web.json_response(_error(None, PARSE_ERROR, f"not JSON: {error}"))
        match message:
            case {"method": str(method), "id": str() | int() as id}:
                return web.json_response(await answer(id, method, _params(message)))
            case _:
                # A notification, or the client's answer to a request this server never makes: nothing to say back.
                return web.Response(status=202)

    async def answer(id: str | int, method: str, params: Mapping[str, object]) -> Mapping[str, object]:
        match method:
            case "initialize":
                # A tools-only server says the same in every protocol version, so the client's own is the one agreed.
                version = params.get("protocolVersion")
                record(McpConnected(client=_object(params.get("clientInfo")), protocol=str(version)))
                return _result(id, {"protocolVersion": version, "capabilities": {"tools": {}}, "serverInfo": {"name": SERVER_NAME, "version": "0"}})
            case "ping":
                return _result(id, {})
            case "tools/list":
                return _result(id, {"tools": [{"name": tool.name, "description": tool.description, "inputSchema": tool.input_schema} for tool in tools]})
            case "tools/call":
                return await call(id, params)
            case "server/discover":
                # Claude Code 2.1.284 asks for it first and, refused, opens with initialize: a known question with a known no.
                return _error(id, METHOD_NOT_FOUND, f"no method {method}")
            case _:
                # [LAW:no-silent-failure] a method this server has not met is said where it will be seen, and to the client.
                logger.warning(f"the MCP server was asked for {method!r}, which it does not answer")
                return _error(id, METHOD_NOT_FOUND, f"no method {method}")

    async def call(id: str | int, params: Mapping[str, object]) -> Mapping[str, object]:
        try:
            fields = Payload(dict(params))
            tool = by_name.get(fields.text("name"))
            arguments = cast(dict[str, object], params.get("arguments") or {})
        except Rejected as error:
            return _error(id, INVALID_PARAMS, str(error))
        if tool is None:
            return _error(id, INVALID_PARAMS, f"no tool {params.get('name')}")
        try:
            inspect.signature(tool.body).bind(**arguments)
        except TypeError as error:
            # Arguments that do not fit the body's signature: the model is told, as a tool result, and can call again.
            return _result(id, {"content": [{"type": "text", "text": f"{tool.name} was called with the wrong arguments: {error}"}], "isError": True})
        try:
            result = await tool.body(**arguments)
        except Exception as error:
            # The audited wrapper on the body has logged it with its trace; the model hears that the tool failed.
            return _result(id, {"content": [{"type": "text", "text": f"{tool.name} failed: {type(error).__name__}: {error}"}], "isError": True})
        return _result(id, {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}], "isError": False})

    async def refused(_request: web.Request) -> web.Response:
        return web.Response(status=405, headers={"Allow": "POST"})

    app = web.Application(middlewares=[admitted])
    app.router.add_post(PATH, post)
    app.router.add_route("*", PATH, refused)
    # [LAW:no-ambient-temporal-coupling] a client that hangs up mid-call does not cancel it: a tool whose effect is half
    # done is worse than one whose result nobody reads, and the draft tools say so of themselves.
    runner = web.AppRunner(app, access_log=None, handler_cancellation=False)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    return McpServer(url=f"http://{host}:{port}{PATH}", token=token, runner=runner)


def _params(message: Mapping[str, object]) -> Mapping[str, object]:
    return _object(message.get("params"))


def _object(value: object) -> Mapping[str, object]:
    match value:
        case dict():
            return cast(dict[str, object], value)
        case _:
            return {}


def _result(id: str | int, result: object) -> Mapping[str, object]:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _error(id: str | int | None, code: int, message: str) -> Mapping[str, object]:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}
