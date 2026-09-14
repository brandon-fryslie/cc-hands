"""The tools the intermediary can call."""

import re
from collections.abc import Callable
from typing import TypedDict, cast

from loguru import logger
from pipecat.adapters.schemas.direct_function import DirectFunction
from pipecat.services.llm_service import FunctionCallParams

from hands.core.drafts import AmendDraft, DiscardDraft, DraftRequest, SendDraft, StageDraft
from hands.core.session import Blocked, Gone, Idle, PromptText, Resolution, SessionId, SessionState, Staged, Working
from hands.sessions.payload import Payload, Rejected
from hands.sessions.registry import Listing, Sessions
from hands.sessions.tmux import TmuxFailed
from hands.voice.readback import readback

# A Pipecat direct function: its signature and docstring are the schema the model sees.
Tool = DirectFunction

# Every C0 and C1 control character but newline and tab: each would press a key in the pane.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def list_sessions_tool(sessions: Sessions) -> Tool:
    async def list_sessions(params: FunctionCallParams) -> None:
        """List the running Claude Code sessions with their titles and what each is doing.

        Call this when the user asks what is running, what sessions exist, or
        what Claude is working on.
        """
        await params.result_callback({"sessions": [describe(listing) for listing in sessions.live()]})

    return list_sessions


def describe(listing: Listing) -> dict[str, str]:
    return {
        "id": listing.session.membership.id,
        "title": spoken_title(listing),
        "state": _spoken_state(listing.session.state),
    }


def spoken_title(listing: Listing) -> str:
    return listing.title or f"untitled, in {listing.session.membership.cwd.name}"


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


class Resolved(TypedDict):
    heard: str
    meant: str


def draft_tools(sessions: Sessions) -> list[Tool]:
    """stage_draft, amend_draft, discard_draft, send_draft: nothing reaches a session until the user says send."""

    async def stage_draft(params: FunctionCallParams, session: str, text: str, resolutions: list[Resolved]) -> None:
        """Stage a prompt the user dictated for a session. Nothing is sent until send_draft.

        Say the returned readback to the user word for word.

        Args:
            session: The session's id, from list_sessions.
            text: The prompt, cleaned up from what the user said.
            resolutions: Each spoken phrase you turned into something exact, such as a file name, with what you made of it. Empty when you resolved nothing.
        """
        await _answer(params, sessions, session, lambda id: StageDraft(id, parse_draft(text, resolutions)))

    async def amend_draft(params: FunctionCallParams, session: str, text: str, resolutions: list[Resolved]) -> None:
        """Replace a session's staged draft with a corrected one when the user changes it.

        Say the returned readback to the user word for word.

        Args:
            session: The session's id, from list_sessions.
            text: The whole corrected prompt, not only the changed words.
            resolutions: Every resolution the corrected prompt relies on.
        """
        await _answer(params, sessions, session, lambda id: AmendDraft(id, parse_draft(text, resolutions)))

    async def discard_draft(params: FunctionCallParams, session: str) -> None:
        """Throw away a session's staged draft without sending it.

        Args:
            session: The session's id, from list_sessions.
        """
        await _answer(params, sessions, session, DiscardDraft)

    async def send_draft(params: FunctionCallParams, session: str) -> None:
        """Send a session's staged draft, exactly as it was read back. Call only when the user says to send.

        Args:
            session: The session's id, from list_sessions.
        """
        await _answer(params, sessions, session, SendDraft)

    return [stage_draft, amend_draft, discard_draft, send_draft]


async def _answer(
    params: FunctionCallParams, sessions: Sessions, session: object, request: Callable[[SessionId], DraftRequest]
) -> None:
    try:
        id = _session_id(session)
        outcome = sessions.draft(request(id))
    except (Rejected, TmuxFailed) as error:
        # [LAW:no-silent-failure] the model hears the failure and says it; the log keeps it.
        logger.error(f"draft tool failed: {error}")
        await params.result_callback({"error": str(error)})
        return
    listing = sessions.listing(id)
    name = id if listing is None else spoken_title(listing)
    await params.result_callback({"readback": readback(outcome, name)})


def _session_id(session: object) -> SessionId:
    match session:
        case str():
            return SessionId(session)
        case other:
            raise Rejected(f"session should be a session id string, got {type(other).__name__}")


def parse_draft(text: object, resolutions: object) -> Staged:
    """The model's arguments, parsed once into a draft whose text is safe to type."""
    # [LAW:parse-dont-validate] PromptText is made here and nowhere else.
    return Staged(_prompt_text(text), tuple(_resolution(item) for item in _items(resolutions)))


def _prompt_text(text: object) -> PromptText:
    match text:
        case str() if not text.strip():
            raise Rejected("the draft text is empty")
        case str() if _CONTROL.search(text):
            raise Rejected("the draft text holds a control character, which would press a key in the pane")
        case str():
            return PromptText(text)
        case other:
            raise Rejected(f"the draft text should be a string, got {type(other).__name__}")


def _items(resolutions: object) -> list[object]:
    match resolutions:
        case list():
            return cast(list[object], resolutions)
        case other:
            raise Rejected(f"resolutions should be a list, got {type(other).__name__}")


def _resolution(item: object) -> Resolution:
    match item:
        case dict():
            fields = Payload(cast(dict[str, object], item))
            return Resolution(heard=fields.text("heard"), meant=fields.text("meant"))
        case other:
            raise Rejected(f"each resolution should be an object with heard and meant, got {type(other).__name__}")
