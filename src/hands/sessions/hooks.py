"""A hook's POST body, parsed once into a core event, and the reply a blocking hook prints."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import get_args, overload

from loguru import logger

from hands.core.effects import Allow, AllowWith, Approve, Deny, HookReply, ModeAfterPlan, Withdraw
from hands.core.events import Attached, Displayed, Ended, EndReason, Event, Joined, Occurred, PermissionRequested, Prompted, SessionEvent, StartSource, Stopped, ToolFinished
from hands.core.occurrences import AutoDenied, CompactTrigger, Compacting, ConfigChanged, ConfigSource, SubagentStarted, SubagentStopped, TaskCompleted, Unrecognised
from hands.core.session import AskedQuestion, Blocker, Instant, Mode, Option, PermissionMode, Permission, Plan, PlanApproved, PromptId, Question, FinishedCall, RequestId, SessionId, UnknownMode
from hands.core.status import Stamp
from hands.sessions.home import Home
from hands.sessions.membership import read_membership, recorded_membership
from hands.sessions.payload import Payload, Rejected


@dataclass(frozen=True)
class Hook:
    """What one hook says, applied in this order, and which hook said it: its name as Claude Code gives it, and its session."""

    name: str
    session: SessionId
    # The session as its file names it, for a daemon that may not have heard of it: a session running since before
    # the plugin never fired its start hook, and joins on whatever hook it fires first. One the registry holds is left
    # as it is. None for a start, which joins by itself; for an end, whose file the shim has removed; and for a hook
    # whose session has no file, which ended or whose process has moved on to another session.
    joining: Attached | None
    happened: Event


def parse_hook(raw: bytes, *, home: Home, at: Instant, heard: Stamp, request: RequestId) -> Hook:
    """Raises Rejected, naming the problem, for anything that is not a hook hands handles."""
    # [LAW:parse-dont-validate] past this function nothing looks at hook JSON again.
    payload = Payload.parse(raw)
    session = payload.session_id()
    name = payload.text("hook_event_name")
    match name:
        case "SessionStart":
            # The shim writes the membership file before it posts, so the start reads it.
            source = _start_source(payload.text("source"))
            return Hook(name, session, None, Joined(read_membership(home, session), source))
        case "SessionEnd":
            return Hook(name, session, None, Ended(session, _end_reason(payload.text("reason"))))
        case _:
            happened = _happened(payload, session, at, heard, request)
            recorded = recorded_membership(home, session)
            return Hook(name, session, None if recorded is None else Attached(recorded), happened)


def parse_display(raw: bytes, *, at: Instant) -> Displayed:
    """A MessageDisplay hook's POST body: the lines of Claude's text it displayed, in the turn its prompt_id names.

    Raises Rejected for anything else, so the one route that takes this hook takes nothing more. It joins no session:
    a session hands has not heard of is joined by its next shim hook, and text it displayed before then is behind.
    """
    # [LAW:parse-dont-validate] past this function nothing looks at hook JSON again.
    payload = Payload.parse(raw)
    match payload.text("hook_event_name"):
        case "MessageDisplay":
            return Displayed(payload.session_id(), (_prompt(payload),), payload.text("delta"), at)
        case other:
            raise Rejected(f"hook event {other!r} is not taken here: only MessageDisplay is posted to this route")


def _happened(payload: Payload, session: SessionId, at: Instant, heard: Stamp, request: RequestId) -> SessionEvent:
    """What a hook of a running session says happened in it."""
    match payload.text("hook_event_name"):
        case "UserPromptSubmit":
            return Prompted(session, at, _mode(payload), _prompt(payload))
        case "Stop":
            return Stopped(session, _closing(payload), _mode(payload), _prompt(payload), payload.flag("stop_hook_active"), heard, request)
        case "PermissionRequest":
            return PermissionRequested(session, at, request, called(payload), _mode(payload))
        case "PostToolUse" | "PostToolUseFailure":
            return ToolFinished(session, at, _ran(payload), _mode(payload))
        # What a hook hands only passes on says happened, in the fields its reference names (code.claude.com/docs/en/hooks).
        case "PermissionDenied":
            return Occurred(session, AutoDenied(payload.text("tool_name"), payload.text("reason"), payload.mapping("tool_input")))
        case "SubagentStart":
            return Occurred(session, SubagentStarted(payload.text("agent_type")))
        case "SubagentStop":
            return Occurred(session, SubagentStopped(payload.text("agent_type"), payload.optional_text("last_assistant_message") or None))
        case "TaskCompleted":
            completed = TaskCompleted(payload.text("task_subject"), payload.optional_text("task_description") or None, payload.optional_text("teammate_name") or None)
            return Occurred(session, completed)
        case "ConfigChange":
            return Occurred(session, ConfigChanged(_config_source(payload.text("source")), _path(payload.optional_text("file_path") or None)))
        case "PreCompact":
            return Occurred(session, Compacting(_trigger(payload.text("trigger")), payload.optional_text("custom_instructions") or None))
        case other:
            raise Rejected(f"hook event {other!r} is not one hands handles; a session that loaded hands' hooks before hands stopped hooking it takes the current ones with /reload-plugins")


# A value hands does not know is kept and said, as an unknown permission mode is: a hook hands only passes on is never
# refused in the session for what a newer Claude Code added.
def _config_source(source: str) -> ConfigSource | Unrecognised:
    match source:
        case "user_settings" | "project_settings" | "local_settings" | "policy_settings" | "skills":
            return source
        case other:
            return Unrecognised(other)


def _path(path: str | None) -> Path | None:
    return None if path is None else Path(path)


def _trigger(trigger: str) -> CompactTrigger | Unrecognised:
    match trigger:
        case "manual" | "auto":
            return trigger
        case other:
            return Unrecognised(other)


def called(payload: Payload) -> Blocker:
    """A tool call as its hooks name it: AskUserQuestion is a question put to the user, ExitPlanMode a plan put up for
    approval, every other tool a permission to run it."""
    match payload.text("tool_name"):
        case "ExitPlanMode":
            # Claude Code reads the plan file into the input before any hook sees it (2.1.281).
            return Plan(Payload(payload.mapping("tool_input")).text("plan"))
        case _:
            return _tool_call(payload)


def _ran(payload: Payload) -> FinishedCall:
    """A call that ran, named as its request was, so the two can be matched."""
    match payload.text("tool_name"):
        case "ExitPlanMode":
            # Its input is what the approval sent, which is empty unless the plan was edited at the dialog (2.1.281).
            return PlanApproved()
        case _:
            return _tool_call(payload)


def _tool_call(payload: Payload) -> Permission | Question:
    """Named the same asked and ran: AskUserQuestion is a question put to the user, every other tool a permission to run it."""
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
    # Claude Code sends an empty description as often as none, and both mean the option has nothing to add.
    return Option(fields.text("label"), fields.optional_text("description") or None)


def _closing(payload: Payload) -> str | None:
    """The reply the turn closed with, as the Stop hook carries it, and None when it carries none."""
    match payload.fields.get("last_assistant_message"):
        case str() as text:
            return text
        case None:
            return None
        case other:
            # Not refused, as a Stop that names no turn is: the prompt_id says which turn stopped, and the reply only
            # how that turn is narrated. Refusing the hook over the narration would lose the turn's telling, so the turn
            # is taken and the loud line says why it will be told without its last reply.
            logger.error(f"Stop carried last_assistant_message as {type(other).__name__}, not a string, so the turn is told without its closing reply")
            return None


def _mode(payload: Payload) -> Mode | None:
    """The session's permission mode as the hook reports it, which every hook hands reads a mode from carries (2.1.281)."""
    # Not refused when it is missing or strange, as a hook that names no turn is: the hook's event and its turn are what
    # move the session, and refusing a Stop over its mode would lose that turn's telling.
    # [LAW:no-silent-failure] a hook with no mode, or a strange one, leaves the session in the mode it last reported, and the log says why.
    event = payload.fields.get("hook_event_name")
    match (payload.fields.get("agent_id"), payload.fields.get("permission_mode")):
        case (str(), _):
            # Fired inside a subagent, which runs in its own mode (2.1.281): that mode is not the session's.
            return None
        case (_, str() as name):
            # An unknown mode is not logged: it is said by its name wherever the mode is, which is louder.
            return _KNOWN_MODES.get(name) or UnknownMode(name)
        case (_, None):
            logger.error(f"{event} carried no permission_mode, so the session keeps the mode it last reported")
            return None
        case (_, other):
            logger.error(f"{event} carried permission_mode as {type(other).__name__}, not a string, so the session keeps the mode it last reported")
            return None


def _prompt(payload: Payload) -> PromptId:
    """The prompt_id that names the turn a prompt opens or a Stop ends, which every one of them carries (2.1.281).

    Raises Rejected for a hook that names no turn: nothing inland could match it to its records or its Stop.
    """
    return PromptId(payload.text("prompt_id"))


# [LAW:one-source-of-truth] the modes hands knows are the type's, read off it rather than listed again.
_KNOWN_MODES: Mapping[str, PermissionMode] = {mode: mode for mode in get_args(PermissionMode)}


def _start_source(source: str) -> StartSource:
    match source:
        case "startup" | "resume" | "clear" | "compact" | "fork":
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


# With no mode set, ExitPlanMode goes back to the mode the session had before it planned.
_MODE_AFTER_PLAN: Mapping[ModeAfterPlan, list[object]] = {
    "resume": [],
    "acceptEdits": [{"type": "setMode", "mode": "acceptEdits", "destination": "session"}],
    "default": [{"type": "setMode", "mode": "default", "destination": "session"}],
}


def name_output(name: str) -> Mapping[str, object]:
    """What a UserPromptSubmit hook prints to set its session's name, the title Claude Code shows on its terminal (2.1.286)."""
    return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "sessionTitle": name}}


@overload
def hook_output(reply: Allow | AllowWith | Approve | Deny) -> Mapping[str, object]: ...
@overload
def hook_output(reply: HookReply) -> Mapping[str, object] | None: ...
def hook_output(reply: HookReply) -> Mapping[str, object] | None:
    """What a waiting PermissionRequest hook prints for Claude Code; None leaves the question to its dialog."""
    # The reply shape Claude Code parses from a PermissionRequest hook's stdout (2.1.270; the plan's, 2.1.281).
    match reply:
        case Allow():
            decision: dict[str, object] = {"behavior": "allow"}
        case AllowWith(input=input):
            decision = {"behavior": "allow", "updatedInput": dict(input)}
        case Approve(mode=mode):
            # What the plan dialog's own yeses send (2.1.281): an empty input, so the plan is read from its file as the
            # user left it, and the mode to leave plan mode for. Claude Code ignores an allow with no updatedInput for a
            # tool that asks the user something, and shows its dialog instead.
            decision = {"behavior": "allow", "updatedInput": {}, "updatedPermissions": _MODE_AFTER_PLAN[mode]}
        case Deny(message=message):
            decision = {"behavior": "deny", "message": message}
        case Withdraw():
            return None
    return {"hookSpecificOutput": {"hookEventName": "PermissionRequest", "decision": decision}}
