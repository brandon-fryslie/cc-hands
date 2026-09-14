"""A hook's POST body, parsed once into a core event, and the reply a blocking hook prints."""

from collections.abc import Mapping

from hands.core.effects import Allow, Deny, HookReply, Withdraw
from hands.core.events import Ended, Event, Joined, PermissionRequested, Prompted, StartSource, Stopped
from hands.core.session import Instant, Permission, RequestId
from hands.sessions.home import Home
from hands.sessions.membership import read_membership
from hands.sessions.payload import Payload, Rejected


def parse_hook(raw: bytes, *, home: Home, at: Instant, request: RequestId) -> Event:
    """Raises Rejected, naming the problem, for anything that is not a hook hands handles."""
    # [LAW:parse-dont-validate] past this function nothing looks at hook JSON again.
    payload = Payload.parse(raw)
    session = payload.session_id()
    match payload.text("hook_event_name"):
        case "SessionStart":
            # The shim writes the membership file before it posts, so the start reads it.
            source = _start_source(payload.text("source"))
            return Joined(read_membership(home, session), source)
        case "UserPromptSubmit":
            return Prompted(session, at)
        case "Stop":
            return Stopped(session)
        case "PermissionRequest":
            permission = Permission(tool=payload.text("tool_name"), input=payload.mapping("tool_input"))
            return PermissionRequested(session, at, request, permission)
        case "SessionEnd":
            return Ended(session)
        case other:
            raise Rejected(f"hook event {other!r} is not one hands handles")


def _start_source(source: str) -> StartSource:
    match source:
        case "startup" | "resume" | "clear" | "compact":
            return source
        case other:
            raise Rejected(f"SessionStart source {other!r} is not one hands knows")


def hook_output(reply: HookReply) -> Mapping[str, object] | None:
    """What a waiting PermissionRequest hook prints for Claude Code; None leaves the question to its dialog."""
    # The reply shape Claude Code 2.1.270 parses from a PermissionRequest hook's stdout.
    match reply:
        case Allow():
            decision: dict[str, object] = {"behavior": "allow"}
        case Deny(message=message):
            decision = {"behavior": "deny", "message": message}
        case Withdraw():
            return None
    return {"hookSpecificOutput": {"hookEventName": "PermissionRequest", "decision": decision}}
