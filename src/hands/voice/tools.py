"""The tools the intermediary can call."""

import asyncio
import functools
import re
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import TypedDict, cast

from loguru import logger
from pipecat.adapters.schemas import direct_function
from pipecat.adapters.schemas.direct_function import DirectFunction
from pipecat.frames.frames import FunctionCallResultProperties
from pipecat.services.llm_service import FunctionCallParams

from hands.core.drafts import AmendDraft, DiscardDraft, DraftRequest, StageDraft
from hands.core.effects import Allow, Answers, Approve, Decision, Deny, KeepPlanning, ModeAfterPlan
from hands.core.session import AtDialog, Blocked, Blocker, Gone, Idle, Permission, Plan, PromptText, Question, RequestId, Resolution, SessionId, SessionState, Staged, Submitted, Working
from hands.core.turn import Budget, Happening, Ref, describe
from hands.sessions.backfill import Unseen, read_since
from hands.sessions.audit import Called, Record
from hands.sessions.payload import Payload, Rejected
from hands.sessions.registry import Listing, Sessions
from hands.voice.readback import readback, spoken_mode, spoken_name, spoken_title
from hands.voice.speech import answer_readback

# A Pipecat direct function: its signature and docstring are the schema the model sees.
Tool = DirectFunction

# Pipecat's decorator is untyped; this names what it does to a tool.
_uncancelled_by_interruption = cast(Callable[[Tool], Tool], direct_function.tool_options(cancel_on_interruption=False))  # pyright: ignore[reportUnknownMemberType]

# How much of a session one reading hands over. A session that has run for an hour has hundreds of steps, and
# all of them at once is a context spent on history; the rest are read on from `more_since`, in order.
READBACK_COUNT = 40

# What the agent reads when the user says no and gives no reason.
DENIED_BY_VOICE = "The user denied this by voice."

# What the agent reads when the user sends a plan back without saying what to change.
SENT_BACK_BY_VOICE = "The user sent the plan back by voice without saying what to change. Ask them what they want different."

# answer_plan's approvals, by the mode each leaves plan mode for.
_APPROVALS: Mapping[str, ModeAfterPlan] = {"approve": "resume", "auto-accept edits": "acceptEdits", "manually approve edits": "default"}

# Every C0 and C1 control character but newline and tab: each would press a key when the draft is typed.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def audited(tool: Tool, record: Record) -> Tool:
    """The tool, with every call written to the audit log beside the result the model is handed."""

    # [LAW:single-enforcer] one wrapper for every tool, so no tool can be called without leaving its line.
    # functools.wraps keeps the signature and docstring, which are the schema the model sees.
    @functools.wraps(tool)
    async def call(params: FunctionCallParams, **arguments: object) -> None:
        answer = params.result_callback

        async def result_callback(result: object, *, properties: FunctionCallResultProperties | None = None) -> None:
            record(Called(params.function_name, arguments, result))
            await answer(result, properties=properties)

        try:
            await tool(replace(params, result_callback=result_callback), **arguments)
        except Exception:
            # [LAW:no-silent-failure] a tool that raises hands the model no result, so it has no Called line; Pipecat
            # logs the error under its own name, which the failure sink does not hear. This is its line.
            logger.exception(f"the tool {params.function_name} raised, called with {arguments!r}")
            raise

    return cast(Tool, call)


def list_sessions_tool(sessions: Sessions) -> Tool:
    async def list_sessions(params: FunctionCallParams) -> None:
        """List the running Claude Code sessions with their titles, what each is doing, and the permission mode each is in.

        Call this when the user asks what is running, what sessions exist, what
        Claude is working on, or what mode a session is in. A session's mode is the
        one it reported when it last did something: a mode changed at its keyboard
        while it sits at its prompt is seen when it is next prompted, and one changed
        in the middle of a turn at its next tool call.
        """
        await params.result_callback({"sessions": [describe_listing(listing) for listing in sessions.live()]})

    return list_sessions


# How much of one step the intermediary is shown when it reads a session back: enough to say what happened,
# and short enough that a screenful of them still leaves room for the conversation they are read into.
# A reading is of the transcript alone, so it shows no repository delta and budgets none: what a turn changed
# in git is told when the turn stops, to the summariser, and is not a session's to be read back out of.
READBACK_BUDGET = Budget(opening=200, said=400, input=120, result=200, steps=READBACK_COUNT, files=0, commits=0, changes=0)


def read_session_tool(sessions: Sessions) -> Tool:
    async def read_session(params: FunctionCallParams, session: str, since: str = "") -> None:
        """What a session has done, in the order it did it, from the point you last read to.

        Call this when the user asks what a session has been doing, or to catch up on one that was already
        running before you attached. When `more` comes back true there is history after what you were given:
        ask the user whether to hear it, and call again with `since` set to `more_since` if they want it. When
        `working` comes back true the session is in the middle of a call that has not come back; say what it is
        in the middle of, and read on later rather than now, when what it did will be there.

        Args:
            session: The session's id, from list_sessions.
            since: The record id you last read to, from an earlier call's `more_since`. Empty reads from the start.
        """
        listing = sessions.listing(SessionId(session))
        if listing is None:
            await params.result_callback({"error": f"there is no session {session}"})
            return
        transcript = listing.session.membership.transcript
        try:
            reading = await asyncio.to_thread(read_since, transcript, Ref(since) if since else None)
        except Unseen:
            # [LAW:no-silent-failure] a mark from another session, or from a transcript since rewritten, is said
            # rather than read as "from the start", which would narrate the whole session over again unasked.
            await params.result_callback({"error": f"session {session} has no record {since}; call again with since empty to read from the start"})
            return
        except OSError as error:
            # [LAW:no-silent-failure] the model is told why it got nothing, rather than being handed nothing.
            logger.error(f"cannot read what session {session} did from {transcript}: {error}")
            await params.result_callback({"error": f"the transcript of session {session} could not be read"})
            return
        # The earliest of what it has not had, not the newest: read on from `more_since` and a session is
        # caught up on in order, which is the only order any of it makes sense in.
        shown = _page(reading.happenings)
        # A call the session is still waiting on is shown but never marked as read, so its result is told once
        # it lands rather than falling into the gap between one reading and the next.
        # [LAW:one-source-of-truth] whether a session can still answer a call is the registry's to say, not the
        # file's. One that has ended will never write the result of the call it was killed inside, and a mark
        # held behind that call would leave the intermediary saying a dead session is still running something.
        ended = isinstance(listing.session.state, Gone)
        settled = shown if ended else shown[: min(reading.settled, len(shown))]
        # [LAW:one-source-of-truth] the mark names a record, and one record can carry both a settled happening
        # and the call the session is still inside — the text and the call it introduces are written together.
        # Marking that record would go on from after the whole of it, losing the very result the mark is held
        # back for, so the mark is the last record every happening of which is settled.
        waiting = {happening.ref for happening in shown[len(settled) :]}
        await params.result_callback(
            {
                "happened": [{"record": happening.ref, "what": describe(happening, READBACK_BUDGET)} for happening in shown],
                # Two different facts, so two answers: history this reading did not reach, and a call that has
                # not come back. Told as one, the model cannot tell "read on" from "wait and ask again".
                "more": len(reading.happenings) > len(shown),
                "working": len(settled) < len(shown),
                # A record carries no uuid only rarely, and naming nothing reads as "from the start" next time.
                "more_since": next(
                    (happening.ref for happening in reversed(settled) if happening.ref is not None and happening.ref not in waiting),
                    since,
                ),
            }
        )

    return read_session


def _page(happenings: list[Happening]) -> list[Happening]:
    """As much of a reading as one call hands over, ending where a record does.

    The mark the reader comes back with names a record, and a reading goes on from after that record, so a page
    that ended inside one would lose the rest of it [LAW:one-source-of-truth]. Nearly every record carries one
    happening and cannot be split: 8 of the 628,822 on this machine carry more than one — a text and the call
    it introduces, and one record with two calls in it. Rare enough to be left to chance is exactly what this
    is not, because the loss is silent: the reading simply never mentions what the skipped block did.
    """
    end = min(READBACK_COUNT, len(happenings))
    ref = happenings[end - 1].ref if end else None
    if end == len(happenings) or ref is None or happenings[end].ref != ref:
        return happenings[:end]
    start = end
    while start > 0 and happenings[start - 1].ref == ref:
        start -= 1
    if start > 0:
        return happenings[:start]
    # One record carrying a whole page of them: handed over long rather than short, because a page cut to fit
    # would name that record as the mark and drop everything after the cut [LAW:no-silent-failure].
    while end < len(happenings) and happenings[end].ref == ref:
        end += 1
    return happenings[:end]


def describe_listing(listing: Listing) -> dict[str, str]:
    return {
        "id": listing.session.membership.id,
        "title": spoken_title(listing),
        "state": _spoken_state(listing.session.state),
        "mode": "not reported yet" if listing.session.mode is None else spoken_mode(listing.session.mode),
    }


def _spoken_state(state: SessionState) -> str:
    match state:
        case Idle():
            return "idle"
        case Submitted():
            return "idle, with a prompt sent that Claude has not started on"
        case Working():
            return "working"
        case Blocked(on=on):
            return _waiting_on(on)
        case AtDialog(on=on):
            return f"{_waiting_on(on)} at the keyboard, too late to answer by voice"
        case Gone():
            return "ended"


def _waiting_on(on: Blocker) -> str:
    match on:
        case Permission(tool=tool):
            return f"waiting for permission to use {tool}"
        case Question():
            return "waiting for the user to answer its question"
        case Plan():
            return "waiting for the user to approve its plan"


class Resolved(TypedDict):
    heard: str
    meant: str


def draft_tools(sessions: Sessions) -> list[Tool]:
    """stage_draft, amend_draft, discard_draft: a prompt dictated for a session, read back until it is right. Nothing reaches a session from here."""

    async def stage_draft(params: FunctionCallParams, session: str, text: str, resolutions: list[Resolved]) -> None:
        """Stage a prompt the user dictated for a session. It is not sent: hands cannot type into a session yet.

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

    # A barge-in must not cancel a draft call part way: the draft would change without its readback being heard.
    return [_uncancelled_by_interruption(tool) for tool in (stage_draft, amend_draft, discard_draft)]


async def _answer(
    params: FunctionCallParams, sessions: Sessions, session: object, request: Callable[[SessionId], DraftRequest]
) -> None:
    # [LAW:no-silent-failure] the model hears each failure and says it; the log keeps it.
    try:
        id = _session_id(session)
        outcome = sessions.draft(request(id))
    except Rejected as error:
        logger.error(f"draft tool refused its arguments: {error}")
        await params.result_callback({"error": str(error)})
        return
    await params.result_callback({"readback": readback(outcome, spoken_name(sessions, id))})


def permission_tools(sessions: Sessions) -> list[Tool]:
    """answer_permission, answer_question, answer_plan: the only ways a voice answer reaches a session waiting on its dialog."""

    async def answer_permission(params: FunctionCallParams, request: str, decision: str, message: str = "") -> None:
        """Answer a session's permission request with what the user decided. Call it only after the user has said to allow or deny.

        Say the returned readback to the user.

        Args:
            request: The request id given with the permission request.
            decision: "allow" to let the tool run, or "deny" to refuse it.
            message: Only when denying: what the user wants the session to know or do instead, in their words.
        """
        await _decide(params, sessions, request, lambda: parse_decision(decision, message))

    async def answer_question(params: FunctionCallParams, request: str, answers: list[str]) -> None:
        """Answer the questions a session asked with what the user chose. Call it only once the user has answered every one.

        Say the returned readback to the user.

        Args:
            request: The request id given with the questions.
            answers: One answer per question, in the order they were asked: the label of the option the user chose, or their own words when no option fits. Where more than one may be chosen, join the labels with ", ". An empty answer when the user chose none.
        """
        await _decide(params, sessions, request, lambda: parse_answers(answers))

    async def answer_plan(params: FunctionCallParams, request: str, decision: str, message: str = "") -> None:
        """Answer a session's plan with what the user decided. Call it only after the user has approved the plan or asked for changes.

        Say the returned readback to the user.

        Args:
            request: The request id given with the plan.
            decision: "approve" to approve the plan and go on in the mode the session had before it planned; "auto-accept edits" or "manually approve edits" only when the user says how edits should go; or "keep planning" to send it back.
            message: Only when it keeps planning: what the user wants changed, in their words.
        """
        await _decide(params, sessions, request, lambda: parse_plan_decision(decision, message))

    # A barge-in must not cancel an answer part way: the user would never hear whether it went through.
    return [_uncancelled_by_interruption(tool) for tool in (answer_permission, answer_question, answer_plan)]


async def _decide(params: FunctionCallParams, sessions: Sessions, request: object, decision: Callable[[], Decision]) -> None:
    # [LAW:no-silent-failure] the model hears a refused answer and says it; the log keeps it.
    try:
        outcome = await sessions.answer(_request_id(request), decision())
    except Rejected as error:
        logger.error(f"{params.function_name} refused its arguments: {error}")
        await params.result_callback({"error": str(error)})
        return
    await params.result_callback({"readback": answer_readback(outcome, lambda id: spoken_name(sessions, id))})


def parse_decision(decision: object, message: object) -> Decision:
    """The model's answer, parsed once into the only two things a person can decide."""
    # [LAW:parse-dont-validate] a Decision is made here and nowhere else, so nothing but "allow" runs a tool.
    match (decision, message):
        case ("allow", ""):
            return Allow()
        case ("allow", str()):
            raise Rejected("a message goes only with deny; an allow carries none, so nothing was answered")
        case ("deny", ""):
            return Deny(DENIED_BY_VOICE)
        case ("deny", str()):
            return Deny(message)
        case ("allow" | "deny", other):
            raise Rejected(f"message should be a string, got {type(other).__name__}")
        case (other, _):
            raise Rejected(f"decision should be 'allow' or 'deny', got {other!r}")


def parse_plan_decision(decision: object, message: object) -> Approve | KeepPlanning:
    """The model's answer to a plan, parsed once into an approval for a mode or a plan sent back."""
    match (decision, message):
        case ("keep planning", ""):
            return KeepPlanning(SENT_BACK_BY_VOICE)
        case ("keep planning", str()):
            return KeepPlanning(message)
        case (str() as choice, "") if choice in _APPROVALS:
            return Approve(_APPROVALS[choice])
        case (str() as choice, str()) if choice in _APPROVALS:
            raise Rejected("a message goes only with keep planning; an approval carries none, so nothing was answered")
        case (str() as choice, other) if choice in _APPROVALS or choice == "keep planning":
            raise Rejected(f"message should be a string, got {type(other).__name__}")
        case (other, _):
            raise Rejected(f"decision should be 'approve', 'auto-accept edits', 'manually approve edits', or 'keep planning', got {other!r}")


def parse_answers(answers: object) -> Answers:
    """The model's answers, parsed once. An empty one leaves its question unanswered, as the dialog's own does."""
    return Answers(tuple(_answer_text(answer) for answer in _items(answers, "answers")))


def _answer_text(answer: object) -> str:
    match answer:
        case str():
            return answer
        case other:
            raise Rejected(f"each answer should be a string, got {type(other).__name__}")


def _request_id(request: object) -> RequestId:
    match request:
        case str() if request:
            return RequestId(request)
        case other:
            raise Rejected(f"request should be the request id string, got {other!r}")


def _session_id(session: object) -> SessionId:
    match session:
        case str():
            return SessionId(session)
        case other:
            raise Rejected(f"session should be a session id string, got {type(other).__name__}")


def parse_draft(text: object, resolutions: object) -> Staged:
    """The model's arguments, parsed once into a draft whose text is safe to type."""
    # [LAW:parse-dont-validate] PromptText is made here and nowhere else.
    return Staged(_prompt_text(text), tuple(_resolution(item) for item in _items(resolutions, "resolutions")))


def _prompt_text(text: object) -> PromptText:
    match text:
        case str() if not text.strip():
            raise Rejected("the draft text is empty")
        case str() if _CONTROL.search(text):
            raise Rejected("the draft text holds a control character, which would press a key when the draft is typed")
        case str():
            return PromptText(text)
        case other:
            raise Rejected(f"the draft text should be a string, got {type(other).__name__}")


def _items(value: object, what: str) -> list[object]:
    match value:
        case list():
            return cast(list[object], value)
        case other:
            raise Rejected(f"{what} should be a list, got {type(other).__name__}")


def _resolution(item: object) -> Resolution:
    match item:
        case dict():
            fields = Payload(cast(dict[str, object], item))
            return Resolution(heard=fields.text("heard"), meant=fields.text("meant"))
        case other:
            raise Rejected(f"each resolution should be an object with heard and meant, got {type(other).__name__}")
