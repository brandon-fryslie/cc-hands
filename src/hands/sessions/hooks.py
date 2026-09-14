"""A hook's POST body, parsed once into a core event."""

from hands.core.events import Ended, Event, Joined, PermissionRequested, Prompted, Stopped
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
            return Joined(read_membership(home, session))
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
