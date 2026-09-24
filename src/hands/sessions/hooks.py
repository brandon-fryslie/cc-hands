"""A hook's POST body, parsed once into a core event, and the reply a blocking hook prints."""

from collections.abc import Mapping

from loguru import logger

from hands.core.effects import Allow, AllowWith, Deny, HookReply, Withdraw
from hands.core.events import Ended, EndReason, Event, Joined, PermissionRequested, Prompted, StartSource, Stopped, ToolFinished, Waited
from hands.core.session import AskedQuestion, Blocker, Instant, Option, Permission, Question, RequestId, SessionId
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
            return Stopped(session, _closing(payload))
        case "Notification":
            return _notified(session, payload.text("notification_type"))
        case "PermissionRequest":
            return PermissionRequested(session, at, request, _call(payload))
        case "PostToolUse" | "PostToolUseFailure":
            return ToolFinished(session, at, _call(payload))
        case "SessionEnd":
            return Ended(session, _end_reason(payload.text("reason")))
        case other:
            raise Rejected(f"hook event {other!r} is not one hands handles")


def _call(payload: Payload) -> Blocker:
    """A tool call as its hooks name it: AskUserQuestion is a question put to the user, every other tool a permission to run it."""
    tool, input = payload.text("tool_name"), payload.mapping("tool_input")
    match tool:
        case "AskUserQuestion":
            return Question(tuple(_asked(block) for block in Payload(input).items("questions")), input)
        case _:
            return Permission(tool=tool, input=input)


def _asked(block: object) -> AskedQuestion:
    fields = Payload.of(block, "each question")
    options = tuple(_option(option) for option in fields.optional_items("options"))
    return AskedQuestion(fields.text("question"), options, fields.optional_flag("multiSelect"))


def _option(option: object) -> Option:
    fields = Payload.of(option, "each option")
    return Option(fields.text("label"), fields.optional_text("description"))


def _notified(session: SessionId, kind: str) -> Waited:
    match kind:
        case "idle_prompt":
            return Waited(session)
        case other:
            # The hook's matcher lets only idle_prompt through, so another type is a settings file hands did not write.
            raise Rejected(f"notification {other!r} is not one hands handles")


def _closing(payload: Payload) -> str | None:
    """The reply the turn closed with, as the Stop hook carries it, and None when it carries none."""
    match payload.fields.get("last_assistant_message"):
        case str() as text:
            return text
        case None:
            return None
        case other:
            # Not refused, as an unknown start is: a turn is what the Stop hook says, and the reply it carries is how
            # that turn is narrated. Refusing the hook over the narration would leave the session working with nothing
            # left to stop it, so the turn is taken and the loud line says why it will be told without its last reply.
            logger.error(f"Stop carried last_assistant_message as {type(other).__name__}, not a string, so the turn is told without its closing reply")
            return None


def _start_source(source: str) -> StartSource:
    match source:
        case "startup" | "resume" | "clear" | "compact":
            return source
        case other:
            raise Rejected(f"SessionStart source {other!r} is not one hands knows")


def _end_reason(reason: str) -> EndReason:
    match reason:
        case "clear" | "resume" | "logout" | "prompt_input_exit" | "bypass_permissions_disabled" | "other":
            return reason
        case _:
            # Not refused, as an unknown start is: the shim has already removed the file, so a refused end would
            # leave the session listed with nothing left to end it. A reason this version does not know is spoken
            # as an end nobody chose, which is the loud way to be wrong.
            return "other"


def hook_output(reply: HookReply) -> Mapping[str, object] | None:
    """What a waiting PermissionRequest hook prints for Claude Code; None leaves the question to its dialog."""
    # The reply shape Claude Code 2.1.270 parses from a PermissionRequest hook's stdout.
    match reply:
        case Allow():
            decision: dict[str, object] = {"behavior": "allow"}
        case AllowWith(input=input):
            decision = {"behavior": "allow", "updatedInput": dict(input)}
        case Deny(message=message):
            decision = {"behavior": "deny", "message": message}
        case Withdraw():
            return None
    return {"hookSpecificOutput": {"hookEventName": "PermissionRequest", "decision": decision}}
