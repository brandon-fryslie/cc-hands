"""The tools the intermediary can call."""

from collections.abc import Awaitable, Callable

from pipecat.services.llm_service import FunctionCallParams

from hands.core.session import Blocked, Gone, Idle, SessionState, Working
from hands.sessions.registry import Listing, Sessions

Tool = Callable[[FunctionCallParams], Awaitable[None]]


def list_sessions_tool(sessions: Sessions) -> Tool:
    async def list_sessions(params: FunctionCallParams) -> None:
        """List the running Claude Code sessions with their titles and what each is doing.

        Call this when the user asks what is running, what sessions exist, or
        what Claude is working on.
        """
        await params.result_callback({"sessions": [describe(listing) for listing in sessions.live()]})

    return list_sessions


def describe(listing: Listing) -> dict[str, str]:
    membership = listing.session.membership
    return {
        "id": membership.id,
        "title": listing.title or f"untitled, in {membership.cwd.name}",
        "state": _spoken_state(listing.session.state),
    }


def _spoken_state(state: SessionState) -> str:
    match state:
        case Idle():
            return "idle"
        case Working():
            return "working"
        case Blocked(on=permission):
            return f"waiting for permission to use {permission.tool}"
        case Gone():
            return "ended"
