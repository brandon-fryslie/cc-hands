"""The tools the intermediary can call, and the adapter that hands them to Pipecat.

A tool is a plain async body, called with the model's arguments and returning what the model is handed back, and the
schema the model sees, read once from the body's signature and docstring. [LAW:decomposition] what each tool does
knows nothing of who calls it: Pipecat's LLM stage is one adapter over the bodies, hands' MCP server another.
"""

import asyncio
import functools
import inspect
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Literal, TypedDict, cast, get_args, get_origin, get_type_hints, is_typeddict

import docstring_parser
from loguru import logger
from pipecat.adapters.schemas import direct_function
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import FunctionCallResultProperties
from pipecat.services.llm_service import FunctionCallParams

from hands.core.drafts import AmendDraft, DiscardDraft, SendDraft, StageDraft
from hands.core.effects import Allow, Answers, Approve, Command, Decision, Deny, KeepPlanning, ModeAfterPlan
from hands.core.keyboard import Interrupt, SendCommand
from hands.core.session import Blocker, Membership, CommandName, Dialog, Held, Idle, LetGo, KEYSTROKES, Permission, Plan, PromptText, Question, RequestId, Resolution, Running, Session, SessionId, SessionState, Staged, Unreported
from hands.core.status import Busy, Going, Shell, Unknown, UnknownReason, Waiting
from hands.core.delta import Delta
from hands.core.attention import Delivery, Overlay
from hands.core.drilldown import drill
from hands.core.sentences import Due, turn_digest
from hands.core.turn import Budget, Happening, Opening, body, describe, turns
from hands.sessions.backfill import Reading, read_transcript
from hands.sessions.backlog import BACKLOG, Backlog, Unread, read_backlog
from hands.sessions.audit import Called, Record
from hands.sessions.focus import Unreadable, focused, set_focus
from hands.sessions.payload import Payload, Rejected
from hands.sessions.home import Home
from hands.sessions.overlays import Overlays
from hands.sessions.registry import Listing, Sessions
from hands.sessions.summaries import described, set_summaries, summaries
from hands.core import playback
from hands.voice.narrator import Recount, Recounts, delivery, switch
from hands.voice.player import Player
from hands.voice.sentences import SummaryStore
from hands.voice.readback import identifier, keyboard_readback, readback, spoken_mode, spoken_name
from hands.voice.speech import answer_readback, told
from hands.voice.voices import VOICES, Voices, fetched, parse_voice, spoken

# What the model is handed back from a call: an object, as every tool API carries a result.
Result = Mapping[str, object]
Body = Callable[..., Awaitable[Result]]
JsonSchema = Mapping[str, object]
Handler = Callable[[FunctionCallParams], Awaitable[None]]


@dataclass(frozen=True)
class Tool:
    """One tool: the schema the model sees, the body that answers a call, and what a call means to the conversation."""

    name: str
    description: str
    properties: Mapping[str, JsonSchema]
    required: tuple[str, ...]
    body: Body
    # "reply": the model is asked to go on once it has the result. "silence": the call is the whole reply.
    then: Literal["reply", "silence"]
    # True when a barge-in must not stop a call part way: its effect would land without its readback heard.
    completes: bool

    @property
    def input_schema(self) -> JsonSchema:
        return {"type": "object", "properties": dict(self.properties), "required": list(self.required)}


def tool(body: Body, *, then: Literal["reply", "silence"] = "reply", completes: bool = False) -> Tool:
    """The body as a tool, its schema read once from its signature and docstring: the name, the text, and each argument's type and line."""
    # [LAW:parse-dont-validate] a signature with a type no schema says is refused here, as the daemon builds its tools.
    docstring = docstring_parser.parse(inspect.getdoc(body) or "")
    lines = {param.arg_name: param.description or "" for param in docstring.params}
    hints = get_type_hints(body)
    parameters = inspect.signature(body).parameters.values()
    properties = {parameter.name: {**_schema(hints[parameter.name]), "description": lines.get(parameter.name, "")} for parameter in parameters}
    required = tuple(parameter.name for parameter in parameters if parameter.default is inspect.Parameter.empty)
    return Tool(body.__name__, (docstring.description or "").strip(), properties, required, _closed(body, hints), then, completes)


def _closed(body: Body, hints: Mapping[str, object]) -> Body:
    """The body, refusing a call that names a value outside an argument's closed set, so the body's Literal holds."""
    # [LAW:single-enforcer] every adapter calls the tool's body, so the set its schema advertises is held here, once.
    closed = {name: get_args(hint) for name, hint in hints.items() if get_origin(hint) is Literal}

    @functools.wraps(body)
    async def call(**arguments: object) -> Result:
        refused = [f"{arguments[name]!r} is no {name}; it is one of {', '.join(allowed)}" for name, allowed in closed.items() if name in arguments and arguments[name] not in allowed]
        return {"error": "; ".join(refused)} if refused else await body(**arguments)

    return call


def _schema(hint: object) -> JsonSchema:
    match hint:
        case type() if hint is str:
            return {"type": "string"}
        case type() if hint is bool:
            return {"type": "boolean"}
        case type() if hint is int:
            return {"type": "integer"}
        case type() if is_typeddict(hint):
            fields = get_type_hints(hint)
            return {"type": "object", "properties": {name: _schema(field) for name, field in fields.items()}, "required": list(fields)}
        case _ if get_origin(hint) is Literal:
            return {"type": "string", "enum": list(get_args(hint))}
        case _ if get_origin(hint) is list:
            [item] = get_args(hint)
            return {"type": "array", "items": _schema(item)}
        case _:
            raise TypeError(f"a tool argument typed {hint!r} has no schema here")


def pipecat_function(tool: Tool) -> FunctionSchema:
    """The tool as Pipecat's LLM stage calls it: the schema it advertises, and a handler that hands back the body's reply."""
    # [LAW:single-enforcer] the one place a tool meets Pipecat, so what "silence" and "completes" mean there is said once.
    properties = None if tool.then == "reply" else FunctionCallResultProperties(run_llm=False)

    async def handler(params: FunctionCallParams) -> None:
        await params.result_callback(await tool.body(**params.arguments), properties=properties)

    # Pipecat's decorator is untyped; it only marks the handler with its call options.
    options = cast(Callable[[Handler], Handler], direct_function.tool_options(cancel_on_interruption=not tool.completes))  # pyright: ignore[reportUnknownMemberType]

    return FunctionSchema(tool.name, tool.description, {name: dict(schema) for name, schema in tool.properties.items()}, list(tool.required), handler=options(handler))


# How much of a session one reading hands over. A session that has run for an hour has hundreds of steps, and
# all of them at once is a context spent on history; the rest are read on from `more_since`, in order.
READBACK_COUNT = 40

# What the agent reads when the user says no and gives no reason.
DENIED_BY_VOICE = "The user denied this by voice."

# What the agent reads when the user sends a plan back without saying what to change.
SENT_BACK_BY_VOICE = "The user sent the plan back by voice without saying what to change. Ask them what they want different."

# answer_plan's approvals, by the mode each leaves plan mode for.
_APPROVALS: Mapping[str, ModeAfterPlan] = {"approve": "resume", "auto-accept edits": "acceptEdits", "manually approve edits": "default"}


# A slash command's name, with the slash the model may have kept from what the user said.
_COMMAND_NAME = re.compile(r"/?([A-Za-z0-9][A-Za-z0-9_:-]*)")


def audited(tool: Tool, record: Record) -> Tool:
    """The tool, with every call written to the audit log beside the result the model is handed."""

    # [LAW:single-enforcer] one wrapper on every body, whichever adapter calls it, so no call leaves no line.
    @functools.wraps(tool.body)
    async def call(**arguments: object) -> Result:
        try:
            result = await tool.body(**arguments)
        except Exception:
            # [LAW:no-silent-failure] a tool that raises hands the model no result, so it has no Called line; this is its line.
            logger.exception(f"the tool {tool.name} raised, called with {arguments!r}")
            raise
        record(Called(tool.name, arguments, result))
        return result

    return replace(tool, body=call)


def intermediary_tools(sessions: Sessions, store: SummaryStore, home: Home, recounts: Recounts, player: Player) -> list[Tool]:
    """Every tool the intermediary is given, in the order its schema lists them.

    [LAW:one-source-of-truth] the daemon hands the model these, and the eval judges the prompt against these, so a
    tool added here is one the eval's model is offered too.
    """
    overlays = Overlays(home)
    # Every tool that acts on one session, each taking the focus for the session the user did not name.
    on_a_session = [
        *session_tools(sessions, store),
        tell_turn_tool(sessions, recounts),
        expand_tool(sessions, recounts),
        *backlog_tools(sessions, store),
        *draft_tools(sessions),
        *keyboard_tools(sessions),
        set_overlay_tool(sessions, overlays),
    ]
    return [
        list_sessions_tool(sessions, overlays, home),
        focus_session_tool(sessions, home),
        *(defaulting_to_focus(tool, home) for tool in on_a_session),
        *permission_tools(sessions),
        turn_summaries_tool(home),
        *voice_tools(Voices(home, player.lines, fetched)),
        *playback_tools(player),
        stay_silent_tool(),
    ]


def playback_tools(player: Player) -> list[Tool]:
    """Going back over what was said: hands says it again from where the speaker was, never the model from memory.

    [LAW:nothing-unseen] each call's result is what hands said for it and how many cut-off readings still wait, so its
    Called line holds what was heard and where it left playback.
    """

    async def resume() -> Result:
        """Go back to what you were saying when the user cut in, and say it from where it stopped.

        Hands says it, exactly as it was said, from the start of the sentence that was cut off. Calling it is the whole
        reply: add no words of your own, and never retell what you remember saying.
        """
        return {"said": await player.act(playback.resume), "waiting": player.waiting}

    async def skip() -> Result:
        """Skip the sentence the user cut in on, and go on with what you were saying from the one after it.

        Hands says the rest. Calling it is the whole reply: add no words of your own.
        """
        return {"said": await player.act(playback.skip), "waiting": player.waiting}

    async def repeat() -> Result:
        """Say again the last thing you said, whole, exactly as it was said.

        Hands says it. Calling it is the whole reply: add no words of your own, and never say it again yourself.
        """
        return {"said": await player.act(playback.repeat), "waiting": player.waiting}

    return [tool(resume, then="silence"), tool(skip, then="silence"), tool(repeat, then="silence")]


def stay_silent_tool() -> Tool:
    async def stay_silent() -> Result:
        """Say nothing in reply to what was just heard, because it was not said to you.

        Calling it is the whole reply: add no words of your own.
        """
        # [LAW:no-silent-failure] the choice not to answer is still a Called line in the audit log, and with the model
        # not run on the result, nothing follows it to the speaker.
        return {"silent": True}

    return tool(stay_silent, then="silence")


def defaulting_to_focus(tool: Tool, home: Home) -> Tool:
    """The tool, its session argument made optional: a call that names no session acts on the focused one."""
    # [LAW:single-enforcer] the one place a session left unnamed becomes the focus, for every tool that acts on one, so
    # what "the focus" means is said once and the model never has to remember which session that is.
    if "session" not in tool.required:
        raise TypeError(f"the tool {tool.name} takes no session the focus could stand in for")
    line = tool.properties["session"]

    async def call(session: object = "", **arguments: object) -> Result:
        if not _unnamed(session):
            return await tool.body(session=session, **arguments)
        match await asyncio.to_thread(focused, home):
            case None:
                return {"error": "no session was named and none is focused: ask the user which session they mean"}
            case Unreadable(reason):
                return {"error": f"no session was named, and which one is focused cannot be read: {reason}"}
            case focus:
                # [LAW:nothing-unseen] the session the focus stood in for rides on the result, so its Called line says where the call went.
                return {**await tool.body(session=focus, **arguments), "focused_session": focus}

    # The body's own signature with session optional, so a call whose other arguments do not fit is refused as the body would refuse it.
    signature = inspect.signature(tool.body)
    call.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=[
            parameter.replace(kind=inspect.Parameter.KEYWORD_ONLY, default="" if parameter.name == "session" else parameter.default)
            for parameter in signature.parameters.values()
        ]
    )
    return replace(
        tool,
        properties={**tool.properties, "session": {**line, "description": f"{line['description']} Empty for the focused session."}},
        required=tuple(name for name in tool.required if name != "session"),
        body=call,
    )


def focus_session_tool(sessions: Sessions, home: Home) -> Tool:
    async def focus_session(session: str) -> Result:
        """Make a session the one the user is talking to: what they say for a session without naming one goes to it.

        Call this when the user says to focus a session, switch or move to one, or work in one, such as "switch to
        cc-hands" or "focus the laws session". Do not ask them to confirm it. It holds from the next thing they say, and
        across restarts, until they move it; a session they name still gets what they say to it, focused or not. Say
        the returned readback.

        Args:
            session: The session's id, from list_sessions. Empty to focus none, when the user says to stop working in one.
        """
        try:
            to = None if _unnamed(session) else _session_id(session)
            # [LAW:parse-dont-validate] only a session the registry holds can be focused.
            if to is not None and sessions.live_session(to) is None:
                raise Rejected(f"no running session has the id {to!r}; take one from list_sessions")
            await asyncio.to_thread(set_focus, home, to)
        except (Rejected, OSError) as error:
            logger.error(f"focus_session could not focus {session!r}: {error}")
            return {"error": str(error)}
        return {"readback": "No session is focused now." if to is None else f"Now on {spoken_name(sessions, to)}."}

    return tool(focus_session, completes=True)


def list_sessions_tool(sessions: Sessions, overlays: Overlays, home: Home) -> Tool:
    async def list_sessions() -> Result:
        """List the running Claude Code sessions by name, what each is doing, and the permission mode each is in.

        A session's name is its project, then a short name for its work: "cc-hands, naming fix". That is how the user
        speaks of a session, and how you speak of one to them.

        Call this when the user asks what is running, what sessions exist, what
        Claude is working on, or what mode a session is in. A session's mode is the
        one it reported when it last did something: a mode changed at its keyboard
        while it sits at its prompt is seen when it is next prompted, and one changed
        in the middle of a turn at its next tool call. `overlay` says how the user hears the turns it finishes:
        watched, normal, or muted (set_overlay). `focus` is the id of the session the user is talking to when they name
        none (focus_session), null when none is focused, or says why it cannot be read.
        """
        return {"sessions": [{**entry, "overlay": await _overlay(overlays, SessionId(entry["id"]))} for entry in standing(sessions)], "focus": await _focus(home)}

    return tool(list_sessions)


async def _focus(home: Home) -> SessionId | None | Mapping[str, str]:
    match await asyncio.to_thread(focused, home):
        case Unreadable(reason):
            return {"cannot_read": reason}
        case focus:
            return focus


async def _overlay(overlays: Overlays, session: SessionId) -> str:
    try:
        return await asyncio.to_thread(overlays.of, session)
    except (Rejected, OSError) as error:
        logger.error(f"cannot read the overlay of session {session} to list it: {error}")
        return f"unknown, its setting cannot be read: {error}"


# How much of one step the intermediary is shown when it reads a session back: enough to say what happened,
# and short enough that a screenful of them still leaves room for the conversation they are read into.
# A reading is of the transcript alone, so it shows no repository delta and budgets none: what a turn changed
# in git is told when the turn stops, to the summariser, and is not a session's to be read back out of.
READBACK_BUDGET = Budget(opening=200, said=400, input=120, result=200, steps=READBACK_COUNT, files=0, commits=0, changes=0)


# How much of a finished turn the summariser is shown to say it in one sentence: its request, and how it started and
# ended. What is longer than the summariser is shown of one thing loses its middle, never its end.
TURN_SENTENCE_BUDGET = Budget(opening=400, said=250, input=100, result=120, steps=16, files=0, commits=0, changes=0)

# How many turns one reading of a session hands over, newest last: a day of work is hundreds of turns, and what it did
# lately is what is asked about.
TURNS_PAGE = 40


def session_tools(sessions: Sessions, store: SummaryStore) -> list[Tool]:
    """read_session, read_turn: what a session has done, a sentence per finished turn first and one turn's steps on request."""

    async def read_session(session: str, before: int = 0) -> Result:
        """What a session has done, one turn at a time, in order: a sentence for each turn it has finished, and the request of the one it is on.

        Call this when the user asks what a session has been doing, or to catch up on one that was already
        running before you attached. Answer from the sentences; call read_turn to hear more of one turn, or of the
        one it is on. A finished turn with no summary yet has only what opened it (a request, or a command the user ran), and its sentence is being written.
        When `working` comes back true the session is still on its last turn. You are given its newest turns; when
        `earlier` is above zero there are that many before them, and calling again with `before` set to the first
        turn you were given reads those.

        Args:
            session: The session's id, from list_sessions.
            before: A turn number from an earlier call, to read the turns before it. 0 reads the newest.
        """
        found = await _session_reading(sessions, session)
        if isinstance(found, str):
            return {"error": found}
        member, reading = found
        spans = turns(reading.happenings)
        if not (before == 0 or 1 <= before <= len(spans)):
            return {"error": f"session {session} has turns 1 to {len(spans)}, and no turn {before}"}
        end = len(spans) if before == 0 else before - 1
        start = max(0, end - TURNS_PAGE)
        live = sessions.live_session(member.id)
        # [LAW:one-source-of-truth] whether a session's last turn is over is the registry's to say: the file cannot tell
        # a turn that ended from one waiting on a long call. Only a session at its prompt, or gone, proves it; one whose
        # status has not been read yet may be mid-turn, and a sentence of half a turn would be kept for good.
        over = live is None or isinstance(live.state, Idle)
        entries: list[dict[str, object]] = []
        due: list[Due] = []
        for number in range(start + 1, end + 1):
            happened = reading.happenings[spans[number - 1].start : spans[number - 1].stop]
            key = turn_digest(happened) if number < len(spans) or over else None
            sentence = None if key is None else store.known(key)
            if key is not None and sentence is None:
                due.append(Due(f"turn-{number}", key, body(happened, Delta(), TURN_SENTENCE_BUDGET), ()))
            entries.append({"turn": number, **({"summary": sentence} if sentence is not None else _opened(happened[0]))})
        # Every read is a sighting: the turns without a sentence are said in the background, never while this call waits.
        store.want_turns(member.id, due)
        return {
            "turns": entries,
            "earlier": start,
            "unsummarised": len(due),
            "working": live is not None and isinstance(live.state, Running),
        }

    async def read_turn(session: str, turn: int, since: str = "") -> Result:
        """The steps of one turn of a session, in the order it took them, from the point you last read to.

        Call this when the user wants more of a turn than its sentence, or wants to know what a session is doing
        in the turn it is on. When `more` comes back true there is more of the turn after what you were given:
        ask the user whether to hear it, and call again with `since` set to `more_since` if they want it. When
        `working` comes back true the session is in the middle of a call that has not come back; say what it is
        in the middle of, and read on later rather than now, when what it did will be there.

        Args:
            session: The session's id, from list_sessions.
            turn: The turn's number, from read_session.
            since: The record id you last read to, from an earlier call's `more_since`. Empty reads from the turn's start.
        """
        found = await _session_reading(sessions, session)
        if isinstance(found, str):
            return {"error": found}
        member, reading = found
        spans = turns(reading.happenings)
        if not 1 <= turn <= len(spans):
            return {"error": f"session {session} has turns 1 to {len(spans)}, and no turn {turn}"}
        span = spans[turn - 1]
        # A mark names a record and a reading goes on from after the whole of it.
        marked = [index for index in span if since and reading.happenings[index].ref == since]
        if since and not marked:
            # [LAW:no-silent-failure] a mark from another turn, or from a transcript since rewritten, is said rather
            # than read as "from the start", which would tell the whole turn over again unasked.
            return {"error": f"turn {turn} of session {session} has no record {since}; call again with since empty to read from its start"}
        start = marked[-1] + 1 if marked else span.start
        rest = reading.happenings[start : span.stop]
        # The earliest of what it has not had, not the newest: read on from `more_since` and a turn is
        # caught up on in order, which is the only order any of it makes sense in.
        shown = _page(rest)
        # A call the session is still waiting on is shown but never marked as read, so its result is told once
        # it lands rather than falling into the gap between one reading and the next.
        # [LAW:one-source-of-truth] whether a session can still answer a call is the registry's to say, not the
        # file's. One that has ended will never write the result of the call it was killed inside, and a mark
        # held behind that call would leave the intermediary saying a dead session is still running something.
        ended = sessions.live_session(member.id) is None
        settled = shown if ended else shown[: max(0, min(reading.settled - start, len(shown)))]
        # [LAW:one-source-of-truth] the mark names a record, and one record can carry both a settled happening
        # and the call the session is still inside — the text and the call it introduces are written together.
        # Marking that record would go on from after the whole of it, losing the very result the mark is held
        # back for, so the mark is the last record every happening of which is settled.
        waiting = {happening.ref for happening in shown[len(settled) :]}
        return {
            "happened": [{"record": happening.ref, "what": describe(happening, READBACK_BUDGET)} for happening in shown],
            # Two different facts, so two answers: more of the turn this reading did not reach, and a call that has
            # not come back. Told as one, the model cannot tell "read on" from "wait and ask again".
            "more": len(rest) > len(shown),
            "working": len(settled) < len(shown),
            # A record carries no uuid only rarely, and naming nothing reads as "from the start" next time.
            "more_since": next(
                (happening.ref for happening in reversed(settled) if happening.ref is not None and happening.ref not in waiting),
                since,
            ),
        }

    return [tool(read_session), tool(read_turn)]


def _opened(first: Happening) -> dict[str, object]:
    """How a turn with no sentence is told: by what opened it, or, in a transcript that starts part way through one, by the
    step it was first read at, which opened nothing."""
    return {"opened" if isinstance(first, Opening) else "began": describe(first, READBACK_BUDGET)}


async def _session_reading(sessions: Sessions, session: str) -> tuple[Membership, Reading] | str:
    """All of what a session has done, read fresh from its transcript; or why it could not be read."""
    member = sessions.membership(SessionId(session))
    if member is None:
        return f"there is no session {session}"
    try:
        return member, await asyncio.to_thread(read_transcript, member.transcript)
    except OSError as error:
        # [LAW:no-silent-failure] the model is told why it got nothing, rather than being handed nothing.
        logger.error(f"cannot read what session {session} did from {member.transcript}: {error}")
        return f"the transcript of session {session} could not be read"


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


def backlog_tools(sessions: Sessions, store: SummaryStore) -> list[Tool]:
    """read_backlog, read_ticket: a session's project backlog, a sentence per ticket first and the ticket's own words on request."""

    async def read_backlog(session: str) -> Result:
        """The backlog of the project a session works in: one sentence for the whole of it, and each epic and loose ticket in the order they are to be worked, each with its own sentence.

        Call this when the user asks what is in the backlog, what is left to do, or what comes next. Answer from the
        sentences; call read_ticket to hear more of one. A ticket with no summary yet has only its title, and its
        sentence is being written.

        Args:
            session: The id, from list_sessions, of a session working in the project.
        """
        match await _backlog(sessions, store, session):
            case str() as error:
                return {"error": error}
            case (backlog, said):
                roots = backlog.roots()
                return {
                    **({"summary": said[BACKLOG]} if BACKLOG in said else {}),
                    "items": [_ticket_line(backlog, said, id) for id in roots],
                    "unsummarised": sum(id not in said for id in roots),
                }

    async def read_ticket(session: str, ticket: str, full: bool = False) -> Result:
        """One ticket or epic from a session's project backlog: its sentence, where it sits, and what is still open under it.

        Call this when the user asks about a ticket or an epic, such as how an epic is going. Set `full` only when
        the user wants the detail the sentence leaves out: it hands over the ticket's whole description and every
        comment on it, which is long.

        Args:
            session: The id, from list_sessions, of a session working in the project.
            ticket: The ticket's id, from read_backlog or an earlier read_ticket.
            full: True to be handed the ticket's own words: its description and its comments.
        """
        match await _backlog(sessions, store, session):
            case str() as error:
                return {"error": error}
            case (backlog, said) if ticket not in backlog.tickets:
                return {"error": f"the backlog has no ticket {ticket}"}
            case (backlog, said):
                parent = backlog.parent.get(ticket)
                found = backlog.tickets[ticket]
                return {
                    **_ticket_line(backlog, said, ticket),
                    **({} if parent is None else {"parent": _ticket_line(backlog, said, parent)}),
                    "open_children": [_ticket_line(backlog, said, child) for child in backlog.open_children(ticket)],
                    **({"description": found.description, "comments": [{"by": comment.by, "at": comment.at, "body": comment.body} for comment in backlog.comments.get(ticket, ())]} if full else {}),
                }

    return [tool(read_backlog), tool(read_ticket)]


async def _backlog(sessions: Sessions, store: SummaryStore, session: str) -> tuple[Backlog, Mapping[str, str]] | str:
    """The session's project backlog read fresh, and every sentence already said of it; or why it could not be read."""
    member = sessions.membership(SessionId(session))
    if member is None:
        return f"there is no session {session}"
    try:
        backlog = await read_backlog(member.cwd)
    except (Unread, Rejected) as error:
        # [LAW:no-silent-failure] the model is told why it got nothing, and the log keeps it.
        logger.error(f"cannot read the backlog of session {session} in {member.cwd}: {error}")
        return f"the backlog in {member.cwd} could not be read: {error}"
    # Every read is a sighting: what changed since the last pass is said in the background, never while this call waits.
    store.want(member.cwd)
    return backlog, store.reckon(backlog.thing()).said


def _ticket_line(backlog: Backlog, said: Mapping[str, str], id: str) -> dict[str, object]:
    """A ticket as the backlog tools hand it over: its sentence where one is said, its state, and for a parent, how far its children are."""
    ticket = backlog.tickets[id]
    children = backlog.children.get(id, ())
    open_children = backlog.open_children(id)
    return {
        "id": id,
        "title": ticket.title,
        **({"summary": said[id]} if id in said else {}),
        # An epic has no status of its own in lit; how far it is lies in its children's counts.
        **({} if ticket.status is None else {"status": ticket.status}),
        **({"children_open": len(open_children), "children_done": len(children) - len(open_children)} if children else {}),
    }


def standing(sessions: Sessions) -> list[dict[str, str]]:
    """Every live session as list_sessions describes it."""
    return [describe_listing(listing) for listing in sessions.live()]


def describe_listing(listing: Listing[Session]) -> dict[str, str]:
    return {
        "id": listing.session.membership.id,
        "name": identifier(listing),
        "state": _spoken_state(listing.session.state, listing.session.dialog),
        "mode": "not reported yet" if listing.session.mode is None else spoken_mode(listing.session.mode),
    }


def _spoken_state(state: SessionState, dialog: Dialog | None) -> str:
    match (dialog, state):
        case (Held(on=on), _):
            # Said by what it asks, though the status saying it waits may not have been read yet.
            return _waiting_on(on)
        case (LetGo(on=on), _):
            return f"{_waiting_on(on)} at the keyboard, too late to answer by voice"
        case (None, _):
            return _stated(state)


def _stated(state: SessionState) -> str:
    match state:
        case Unreported():
            return "not reported yet"
        case Idle():
            return "idle"
        case Running(status=going):
            return _running(going)


def _running(going: Going) -> str:
    match going:
        case Waiting(reason=UnknownReason(name=name)):
            return f"waiting at a dialog: {name}"
        case Waiting(reason=reason):
            return f"waiting at a dialog: {reason}"
        case Busy() | Shell():
            return "working"
        case Unknown(name=name):
            return f"in a state hands does not know: {name}"


def _waiting_on(on: Blocker) -> str:
    match on:
        case Permission(tool=tool):
            return f"waiting for permission to use {tool}"
        case Question():
            return "waiting for the user to answer its question"
        case Plan():
            return "waiting for the user to approve its plan"


def tell_turn_tool(sessions: Sessions, recounts: Recounts) -> Tool:
    async def tell_turn(session: str) -> Result:
        """What a session's last finished turn did, as hands tells a turn when it finishes, and how it stands now.

        Call this when the user asks what a session just did or how its last turn went. Tell them as the returned turn
        says to. `now` is how the session stands at this moment, as list_sessions says it: a question the turn ended
        on may have been answered at the keyboard since, and a session working again is no longer waiting on it.

        Args:
            session: The session's id, from list_sessions.
        """
        try:
            id = _session_id(session)
        except Rejected as error:
            return {"error": str(error)}
        live = sessions.live_session(id)
        if live is None:
            return {"error": f"no running session has the id {id!r}; take one from list_sessions"}
        name = spoken_name(sessions, id)
        now = _spoken_state(live.state, live.dialog)
        match recounts.of(id):
            case None:
                # [LAW:no-silent-failure] said as what it is, never as a turn that did nothing.
                return {"error": f"no turn of {name} has finished since hands started; read_session reads what it did before"}
            case Recount(tellings=(), unread=False):
                return {"turn": f"[hands] The Claude Code session {name} finished a turn with nothing in it hands could tell. Tell the user so.", "now": now}
            case Recount(tellings=tellings, unread=unread):
                failed = (f"[hands] hands could not read {'the rest of ' if tellings else ''}the turn the Claude Code session {name} finished. Tell the user so.",)
                return {"turn": "\n\n".join((*(told(id, name, (telling,)) for telling in tellings), *(failed if unread else ()))), "now": now}

    return tool(tell_turn)


def expand_tool(sessions: Sessions, recounts: Recounts) -> Tool:
    async def expand(session: str, part: str = "") -> Result:
        """More of the last turn a session finished: with no part, the parts it opens into, such as the change, the
        tests, or the commit, each in a line; with a part, that part's records, at more length each time it is asked for.

        Tell it in your own words, in spoken form. There is no word-for-word reading: say what code does and what a
        command found, and never read out code, output, paths, or hashes as written, however much the user asks for.
        When `deeper` comes back false, asking for that part again tells no more.

        Args:
            session: The id of the session whose turn you just told, from the [hands] message that told it or from list_sessions.
            part: The part the user wants more of, by the name a call with it empty gave, such as "the tests". Empty for the list of parts.
        """
        try:
            id = _session_id(session)
        except Rejected as error:
            return {"error": str(error)}
        name = spoken_name(sessions, id)
        held = recounts.of(id)
        if held is None or not held.parts:
            # [LAW:no-silent-failure] said as what it is, never as a turn with nothing in it.
            return {"error": f"hands holds no finished turn of {name} to open; read_session and read_turn read what it did"}
        topic = part.strip().lower()
        chosen = held.parts if not topic else tuple(segment for segment in held.parts if segment.topic.name == topic)
        if not chosen:
            named = ", ".join(dict.fromkeys(segment.topic.name for segment in held.parts))
            return {"error": f"the last turn of {name} has no part {part!r}; its parts are {named}"}
        # The turn as a whole is told only as its parts' lines, the first rung, however often it is asked for: told
        # deeper, every part at once would be more than one answer can carry. A part asked for by name opens at the
        # rung below its line, and a rung further each time it is asked for: [LAW:one-source-of-truth] its depth is
        # counted by the part, so the list asked for between two askings neither moves nor repeats it.
        depth = recounts.open(id, topic) + 1 if topic else 0
        drilled = drill(chosen, depth)
        # [LAW:nothing-unseen] the depth rides on the result, so the Called line says how far down this asking went.
        return {"parts": [{"part": topic, "told": told} for topic, told in drilled.told], "deeper": drilled.deeper, "depth": depth}

    return tool(expand)


def turn_summaries_tool(home: Home) -> Tool:
    async def turn_summaries(on: bool) -> Result:
        """Turn spoken turn summaries on or off.

        On, every turn any session finishes is told to the user as it finishes, except a muted session's. Off, only a watched
        session's turns are (set_overlay), and any session's last turn is told when the user asks for it (tell_turn). What a session
        asks them, a permission, a question, or a plan, is said either way. Call this when the user asks to hear every
        session's turns, or to stop hearing them. It lasts until they change it, across restarts. Say the returned
        readback to the user.

        Args:
            on: true to tell every finished turn, false to tell only watched sessions' turns.
        """
        to = "on" if on else "off"
        try:
            await asyncio.to_thread(set_summaries, home, to)
        except OSError as error:
            logger.error(f"turn_summaries could not set the switch {to}: {error}")
            return {"error": str(error)}
        return {"readback": described(to)}

    return tool(turn_summaries, completes=True)


def voice_tools(voices: Voices) -> list[Tool]:
    """Choosing the voice hands speaks in, by ear: the user hears the voices said by hands, and keeps one.

    [LAW:nothing-unseen] each result names the voice hands speaks in after the call, and a hearing the voices it said,
    so the Called line holds what was heard and what was kept.
    """

    async def voices_on_offer() -> Result:
        """The voices hands can speak in, and the one it speaks in now.

        Call this when the user asks what voice you speak in, or which voices there are. Say a voice's name as a person
        would: bill_boerst is Bill Boerst.
        """
        try:
            return {"speaking_in": await voices.speaking_in(), "voices": VOICES}
        except (Rejected, OSError) as error:
            return {"error": str(error)}

    async def hear_voices(names: list[str]) -> Result:
        """Hands says a line in each voice named, one after another, each saying its own name, then speaks on in the
        voice it was using.

        Call this when the user wants to hear what voices sound like: a few at a time, at most five unless they ask for
        more, since each takes a few seconds. Say nothing before calling it. What you say after it is heard after the
        last voice, so keep it to a few words asking which they would like. Hearing a voice does not choose it;
        use_voice does. To hear voices again, after a barge-in too, call this again: going back or repeating says the
        words again in the voice you speak in, not in theirs.

        Args:
            names: the voices to hear, as voices_on_offer lists them.
        """
        try:
            heard = tuple(parse_voice(name) for name in names)
            return {"heard": heard, "speaking_in": await voices.hear(heard)}
        except (Rejected, OSError) as error:
            return {"error": str(error)}

    async def use_voice(name: str) -> Result:
        """Speak in this voice from now on. It lasts until the user chooses another, across restarts.

        Call this when the user picks a voice. What you say next is in it, so say the returned readback.

        Args:
            name: the voice, as voices_on_offer lists it.
        """
        try:
            voice = parse_voice(name)
            await voices.use(voice)
        except (Rejected, OSError) as error:
            return {"error": str(error)}
        return {"speaking_in": voice, "readback": f"This is {spoken(voice)}, and I'll speak in this voice from now on."}

    return [tool(voices_on_offer), tool(hear_voices), tool(use_voice, completes=True)]


def set_overlay_tool(sessions: Sessions, overlays: Overlays) -> Tool:
    async def set_overlay(session: str, overlay: Overlay) -> Result:
        """Set how the user hears a session's finished turns: watched, normal, or muted.

        `watched` tells them each turn it finishes, as it finishes. `normal` tells its turns only with spoken summaries
        on (turn_summaries), and otherwise when they ask (tell_turn); every session is normal until they change it.
        `muted` tells its turns only when they ask, even with spoken summaries on. Whatever its overlay, a session's
        permission requests, questions, and plans are said: they need an answer. Call this with watched when the user
        asks to be told when a session finishes, with muted when they ask to stop hearing about it or to mute it, and
        with normal when they unmute or unwatch it. It lasts until they change it, across restarts. Say the returned
        readback to the user: it says how the session's turns now reach them, spoken summaries considered.

        Args:
            session: The session's id, from list_sessions.
            overlay: watched, normal, or muted.
        """
        try:
            id = _session_id(session)
            # [LAW:single-enforcer] the registry is the one judge of which sessions are running.
            live = sessions.live_session(id)
            if live is None:
                raise Rejected(f"no running session has the id {id!r}; take one from list_sessions")
            await asyncio.to_thread(overlays.set, id, overlay)
        except (Rejected, OSError) as error:
            logger.error(f"set_overlay could not set session {session!r} to {overlay!r}: {error}")
            return {"error": str(error)}
        # [LAW:one-source-of-truth] the delivery the narrator computes, from the switch as it reads it, so the readback
        # says what will happen to the session's next turn.
        delivered = delivery(await switch(lambda: summaries(overlays.home)), overlay)
        return {"readback": _overlay_readback(spoken_name(sessions, id), delivered)}

    return tool(set_overlay, completes=True)


def _overlay_readback(name: str, delivered: Delivery) -> str:
    match delivered:
        case "watched":
            return f"I'll tell you each turn {name} finishes."
        case "summaries":
            return f"I'll tell you each turn {name} finishes, as I tell every session's with spoken summaries on."
        case "on request":
            return f"I'll hold {name}'s turns until you ask for one."
        case "muted":
            return f"{name} is muted: I'll hold its turns until you ask, even with spoken summaries on. It still speaks when it needs your answer."


class Resolved(TypedDict):
    heard: str
    meant: str


def draft_tools(sessions: Sessions) -> list[Tool]:
    """stage_draft, amend_draft, discard_draft, send_draft: a prompt dictated for a session, read back until it is right, then sent."""

    async def stage_draft(session: str, text: str, resolutions: list[Resolved]) -> Result:
        """Stage a prompt the user dictated for a session. It is not sent until the user says to send it.

        Say the returned readback to the user word for word.

        Args:
            session: The session's id, from list_sessions.
            text: The prompt.
            resolutions: Each spoken phrase you turned into something exact, such as a file name, with what you made of it. Empty when you resolved nothing.
        """
        return await _answer("stage_draft", sessions, session, lambda id: StageDraft(id, parse_draft(text, resolutions)), sessions.draft, readback)

    async def amend_draft(session: str, text: str, resolutions: list[Resolved]) -> Result:
        """Replace a session's staged draft with a corrected one when the user changes it.

        Say the returned readback to the user word for word.

        Args:
            session: The session's id, from list_sessions.
            text: The whole corrected prompt, not only the changed words.
            resolutions: Every resolution the corrected prompt relies on.
        """
        return await _answer("amend_draft", sessions, session, lambda id: AmendDraft(id, parse_draft(text, resolutions)), sessions.draft, readback)

    async def discard_draft(session: str) -> Result:
        """Throw away a session's staged draft without sending it.

        Args:
            session: The session's id, from list_sessions.
        """
        return await _answer("discard_draft", sessions, session, DiscardDraft, sessions.draft, readback)

    async def send_draft(session: str) -> Result:
        """Type a session's staged draft into it and press Return. Call it only once the user has said to send it.

        Say the returned readback to the user.

        Args:
            session: The session's id, from list_sessions.
        """
        return await _answer("send_draft", sessions, session, SendDraft, sessions.draft, readback)

    # A barge-in must not cancel a draft call part way: the draft would change, or be sent, without its readback heard.
    return [tool(body, completes=True) for body in (stage_draft, amend_draft, discard_draft, send_draft)]


async def _answer[R, O](
    name: str,
    sessions: Sessions,
    session: object,
    request: Callable[[SessionId], R],
    apply: Callable[[R], Awaitable[O]],
    say: Callable[[O, str], str],
) -> Result:
    # [LAW:no-silent-failure] the model hears each failure and says it; the log keeps it.
    try:
        id = _session_id(session)
        outcome = await apply(request(id))
    except Rejected as error:
        logger.error(f"{name} refused its arguments: {error}")
        return {"error": str(error)}
    return {"readback": say(outcome, spoken_name(sessions, id))}


def keyboard_tools(sessions: Sessions) -> list[Tool]:
    """send_command, interrupt_session: what the user would otherwise do at a session's keyboard besides send it a prompt."""

    async def send_command(session: str, command: str, args: str = "") -> Result:
        """Run a slash command in a session, such as compact, clear, or model. Call it only when the user asks for a command by name.

        What the user dictates as a prompt is a draft, even when it begins with a slash; this is only for commands.
        Say the returned readback to the user.

        Args:
            session: The session's id, from list_sessions.
            command: The command's name, such as "compact" or "model".
            args: What follows the name, such as "opus" for model. Empty when the user gave nothing.
        """
        return await _answer("send_command", sessions, session, lambda id: SendCommand(id, parse_command(command, args)), sessions.keyboard, keyboard_readback)

    async def interrupt_session(session: str) -> Result:
        """Stop what a session is doing, as pressing Escape at its keyboard does. At a permission dialog that is the dialog's no.

        Say the returned readback to the user.

        Args:
            session: The session's id, from list_sessions.
        """
        return await _answer("interrupt_session", sessions, session, Interrupt, sessions.keyboard, keyboard_readback)

    # A barge-in must not cancel either part way: it would be typed without its readback heard.
    return [tool(body, completes=True) for body in (send_command, interrupt_session)]


def permission_tools(sessions: Sessions) -> list[Tool]:
    """answer_permission, answer_question, answer_plan: the only ways a voice answer reaches a session waiting on its dialog."""

    async def answer_permission(request: str, decision: str, message: str = "") -> Result:
        """Answer a session's permission request with what the user decided. Call it only after the user has said to allow or deny.

        Say the returned readback to the user.

        Args:
            request: The request id given with the permission request.
            decision: "allow" to let the tool run, or "deny" to refuse it.
            message: Only when denying: what the user wants the session to know or do instead, in their words.
        """
        return await _decide("answer_permission", sessions, request, lambda: parse_decision(decision, message))

    async def answer_question(request: str, answers: list[str]) -> Result:
        """Answer the questions a session asked with what the user chose. Call it only once the user has answered every one.

        Say the returned readback to the user.

        Args:
            request: The request id given with the questions.
            answers: One answer per question, in the order they were asked: the label of the option the user chose, or their own words when no option fits. Where more than one may be chosen, join the labels with ", ". An empty answer when the user chose none.
        """
        return await _decide("answer_question", sessions, request, lambda: parse_answers(answers))

    async def answer_plan(request: str, decision: str, message: str = "") -> Result:
        """Answer a session's plan with what the user decided. Call it only after the user has approved the plan or asked for changes.

        Say the returned readback to the user.

        Args:
            request: The request id given with the plan.
            decision: "approve" to approve the plan and go on in the mode the session had before it planned; "auto-accept edits" or "manually approve edits" only when the user says how edits should go; or "keep planning" to send it back.
            message: Only when it keeps planning: what the user wants changed, in their words.
        """
        return await _decide("answer_plan", sessions, request, lambda: parse_plan_decision(decision, message))

    # A barge-in must not cancel an answer part way: the user would never hear whether it went through.
    return [tool(body, completes=True) for body in (answer_permission, answer_question, answer_plan)]


async def _decide(name: str, sessions: Sessions, request: object, decision: Callable[[], Decision]) -> Result:
    # [LAW:no-silent-failure] the model hears a refused answer and says it; the log keeps it.
    try:
        outcome = await sessions.answer(_request_id(request), decision())
    except Rejected as error:
        logger.error(f"{name} refused its arguments: {error}")
        return {"error": str(error)}
    return {"readback": answer_readback(outcome, lambda id: spoken_name(sessions, id))}


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


def _unnamed(session: object) -> bool:
    """Whether the model named no session: empty, or null, which some models send for an argument left empty."""
    return session in ("", None)


def _session_id(session: object) -> SessionId:
    match session:
        case str():
            return SessionId(session)
        case other:
            raise Rejected(f"session should be a session id string, got {type(other).__name__}")


def parse_draft(text: object, resolutions: object) -> Staged:
    """The model's arguments, parsed once into a draft whose text is safe to type."""
    # [LAW:parse-dont-validate] PromptText is made here and nowhere else.
    return Staged(_prompt_text(text, "the draft text"), tuple(_resolution(item) for item in _items(resolutions, "resolutions")))


def parse_command(name: object, args: object) -> Command:
    """The model's arguments, parsed once into a command whose name and arguments are safe to type after a slash."""
    return Command(_command_name(name), _command_args(args))


def _command_name(name: object) -> CommandName:
    # [LAW:parse-dont-validate] CommandName is made here and nowhere else. A slash the model kept from what the user
    # said is the one the command is typed with, not a second.
    match name:
        case str():
            named = _COMMAND_NAME.fullmatch(name)
            if named is None:
                raise Rejected(f"command should be a slash command's name, such as compact, got {name!r}")
            return CommandName(named.group(1))
        case other:
            raise Rejected(f"command should be a string, got {type(other).__name__}")


def _command_args(args: object) -> PromptText | None:
    match args:
        case str() if not args.strip():
            return None
        case str() if "\n" in args:
            # A command is one line: what a line break does inside one, pasted, is unmeasured.
            raise Rejected("a command's arguments are one line, and these hold a line break")
        case _:
            return _prompt_text(args, "the argument string")


def _prompt_text(text: object, what: str) -> PromptText:
    match text:
        case str() if not text.strip():
            raise Rejected(f"{what} is empty")
        case str() if KEYSTROKES.search(text):
            # A tab is one: typed into a session it cycles the mode, and fritter refuses it by name at the socket.
            # Refusing it here instead means the model is told while it still has the words to fix.
            raise Rejected(f"{what} holds a control character, which would press a key when it is typed")
        case str() if text.endswith("\\"):
            raise Rejected(f"{what} ends with a backslash, which turns the Return that sends it into a newline")
        case str():
            return PromptText(text)
        case other:
            raise Rejected(f"{what} should be a string, got {type(other).__name__}")


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
