"""What sessions say to the user unasked: announcements spoken as written, moments the intermediary explains."""

import json
from collections.abc import Awaitable, Callable

from pipecat.frames.frames import Frame, LLMMessagesAppendFrame, TTSSpeakFrame

from hands.core.effects import Allow, Decision, Deny, Heard, Narrate, PermissionAsked, PermissionDeadlineNear, PermissionExpired, Speak
from hands.core.permissions import NotWaiting, PermissionAnswered, PermissionOutcome
from hands.core.session import SessionId
from hands.sessions.registry import Sessions
from hands.voice.readback import spoken_name

# A tool input is shown to the model whole up to this many characters; a longer one is cut and says so.
_INPUT_SHOWN = 800

Names = Callable[[SessionId], str]


async def relay(sessions: Sessions, queue_frame: Callable[[Frame], Awaitable[None]]) -> None:
    """Hand everything the sessions say to the pipeline, in the order it was decided, until cancelled."""
    while True:
        heard = await sessions.heard()
        await queue_frame(frame(heard, lambda id: spoken_name(sessions, id)))


def frame(heard: Heard, names: Names) -> Frame:
    # [LAW:one-type-per-behavior] the route is the effect's own variant: Speak needs no model, Narrate needs one to explain.
    match heard:
        case Speak(announcement=announcement):
            # Kept in the context, so the intermediary knows what the user has already been told.
            return TTSSpeakFrame(announcement_text(announcement, names))
        case Narrate(moment=moment):
            return LLMMessagesAppendFrame([{"role": "user", "content": narration(moment, names)}], run_llm=True)


def narration(moment: PermissionAsked, names: Names) -> str:
    shown = json.dumps(dict(moment.permission.input), ensure_ascii=False)
    shown = shown if len(shown) <= _INPUT_SHOWN else f"{shown[:_INPUT_SHOWN]}... (cut short)"
    return (
        f"[hands] The Claude Code session {names(moment.session)} is waiting for permission to use "
        f"{moment.permission.tool} with this input: {shown}. Request id: {moment.request}. "
        "Tell the user in one short sentence what it wants to do, and ask whether to allow it. "
        "When they decide, call answer_permission with that request id."
    )


def announcement_text(announcement: PermissionDeadlineNear | PermissionExpired, names: Names) -> str:
    match announcement:
        case PermissionDeadlineNear(session=session, permission=permission, remaining=remaining):
            return f"{round(remaining)} seconds left to answer {names(session)} about {permission.tool}."
        case PermissionExpired(session=session, permission=permission):
            # Said as what hands did: an answer typed at the dialog meanwhile would already have settled it.
            return f"Nobody answered {names(session)} about {permission.tool} in time, so I told it no."


def permission_readback(outcome: PermissionOutcome, names: Names) -> str:
    match outcome:
        case PermissionAnswered(session=session, permission=permission, decision=decision):
            return f"{_verb(decision)} {permission.tool} for {names(session)}."
        case NotWaiting():
            return "That request is no longer waiting: it was already answered, answered at the keyboard, or denied at its deadline."


def _verb(decision: Decision) -> str:
    match decision:
        case Allow():
            return "Allowed"
        case Deny():
            return "Denied"
