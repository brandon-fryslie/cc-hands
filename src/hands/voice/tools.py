"""The tools the brain can call.

Each is a `hands.voice.tool.Tool`: a plain async body, called with the model's arguments and returning what the model
is handed back. [LAW:decomposition] what each tool does knows nothing of who calls it: hands' MCP server is the adapter
over the bodies.
"""

import asyncio
import functools
import json
import re
from datetime import UTC, datetime, timedelta
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol, TypedDict

import aiohttp
from loguru import logger

from hands.voice.tool import Result, Tool, tool
from hands.brain.usage import Usage
from hands.core.drafts import AmendDraft, DiscardDraft, DraftAmended, DraftOutcome, DraftStaged, SendDraft, StageDraft
from hands.core.effects import Allow, Answers, Approve, Command, Decision, Deny, KeepPlanning, ModeAfterPlan
from hands.core.keyboard import Interrupt, SendCommand
from hands.core.progress import Doing, said
from hands.core.session import Blocker, Membership, CommandName, Opened, Held, Idle, LetGo, KEYSTROKES, Permission, Plan, PromptText, Question, RequestId, Resolution, Running, Session, SessionId, SessionState, Staged, Unreported, delegating
from hands.core.status import Busy, Going, Shell, Unknown, UnknownReason, Waiting
from hands.core.delta import Delta
from hands.core.attention import Attention, Kind, Overlay, Spoken, Withheld
from hands.core.drilldown import drill
from hands.core.sentences import Due, cut, turn_digest
from hands.core.tmux import InPane, NotInTmux, Pane, PaneUnread, Unanswered
from hands.core.spoken import counted
from hands.core.turn import Budget, Happening, Opening, body, describe, turns
from hands.sessions.backfill import Reading, read_transcript
from hands.sessions.backlog import BACKLOG, Backlog, Unread, Untracked, read_backlog
from hands.sessions import catchup, closesession, startsession, tmux
from hands.sessions.audit import Record
from hands.sessions.wide import annotate, fail, unit
from hands.sessions.focus import Unreadable, focused
from hands.sessions.payload import Rejected
from hands.sessions.home import Home
from hands.sessions.overlays import Overlays
from hands.sessions.registry import Listing, Sessions
from hands.sessions import attention as settings
from hands.core import playback, status
from hands.core.place import Modality
from hands.voice.trigger import Trigger, Triggers, described, readied
from hands.voice.wake import Unheard
from hands.voice.wakeword import Word
from hands.voice.narrator import Recount, Recounts, delivery, set_to
from hands.voice.player import Player
from hands.voice.sentences import SummaryStore
from hands.voice.readback import identifier, keyboard_readback, readback, spoken_mode, spoken_name
from hands.voice.refocus import NotRunning, Refocus, move_focus
from hands.voice.speakers import Room
from hands.voice.speech import answer_readback, told
from hands.voice.voices import VOICES, Voices, fetched, parse_voice, spoken
from hands.threads import off_loop
from hands import spotify

@dataclass(frozen=True)
class Called:
    """What the model gave a tool, and the result it was handed back: None while the body runs, and for a body that raised."""

    arguments: Mapping[str, object]
    result: Result | None


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
    """The tool, each call one wide event: the tool's name, what it was given, and what the model was handed back, ending
    failed where the body raised or the result refuses the call."""

    # [LAW:single-enforcer] one unit on every body, whichever adapter calls it, so no call goes unseen [LAW:nothing-unseen].
    @functools.wraps(tool.body)
    async def call(**arguments: object) -> Result:
        with unit("tool.run", record):
            annotate(tool=tool.name, called=Called(arguments, None))
            result = await tool.body(**arguments)
            annotate(called=Called(arguments, result))
            match result:
                case {"error": refused}:
                    fail(str(refused))
                case _:
                    pass
            return result

    return replace(tool, body=call)


def cued(tool: Tool, acting: Callable[[], None]) -> Tool:
    """The tool, saying by `acting` that hands is acting each time the model calls it, before its body runs."""

    @functools.wraps(tool.body)
    async def call(**arguments: object) -> Result:
        acting()
        return await tool.body(**arguments)

    return replace(tool, body=call)


def intermediary_tools(
    sessions: Sessions, store: SummaryStore, home: Home, recounts: Recounts, player: Player, refocus: Refocus, switch: Callable[[Modality], None], triggers: Triggers, wake: Word, own: "OwnModel", catalogue: spotify.Catalogue, room: Room, environment: Mapping[str, str], acting: Callable[[], None]
) -> list[Tool]:
    """Every tool the intermediary is given, in the order its schema lists them, each telling `acting` as it is called
    but staying silent, whose call is the choice not to act. `environment` is the run's, which says where tmux keeps its
    sockets and where tmux is.

    [LAW:one-source-of-truth] the daemon hands the model these, and the eval judges the prompt against these, so a
    tool added here is one the eval's model is offered too.
    """
    overlays = Overlays(home)
    # Every tool that acts on one session, each taking the focus for the session the user did not name.
    on_a_session = [
        *session_tools(sessions, store),
        tell_turn_tool(sessions, recounts, refocus),
        expand_tool(sessions, recounts),
        *backlog_tools(sessions, store),
        *draft_tools(sessions),
        *keyboard_tools(sessions),
        read_screen_tool(sessions, environment),
        set_overlay_tool(sessions, overlays),
    ]
    acts = [
        list_sessions_tool(sessions, overlays, home, environment),
        focus_session_tool(sessions, home),
        *(defaulting_to_focus(tool, home) for tool in on_a_session),
        *permission_tools(sessions),
        catch_up_tool(sessions, home, lambda: datetime.now(UTC)),
        attention_tool(home),
        modality_tool(switch),
        *trigger_tools(triggers, home.wake_word, wake),
        *voice_tools(Voices(home, player.lines, fetched)),
        *model_tools(own, player),
        *playback_tools(player),
        *spotify_tools(spotify.Player(), catalogue),
        name_voice_tool(room),
    ]
    return [*(cued(tool, acting) for tool in acts), stay_silent_tool()]


def playback_tools(player: Player) -> list[Tool]:
    """Going back over what was said: hands says it again from where the speaker was, never the model from memory.

    [LAW:nothing-unseen] each call's result is what hands said for it and how many cut-off readings still wait, so its
    event holds what was heard and where it left playback.
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


def name_voice_tool(room: Room) -> Tool:
    async def name_voice(voice: int, name: str) -> Result:
        """Give someone else in the room the name they told you, so that from now on, in this conversation and every
        later one, their words come to you marked with it.

        Args:
            voice: The number their words were marked with.
            name: Their name, as they said it.
        """
        try:
            await asyncio.to_thread(room.named, voice, name)
        except ValueError as error:
            return {"error": str(error)}
        return {"named": name}

    return tool(name_voice)


def stay_silent_tool() -> Tool:
    async def stay_silent() -> Result:
        """Say nothing in reply to what was just heard, because it was not said to you.

        Calling it is the whole reply: add no words of your own.
        """
        # [LAW:no-silent-failure] the choice not to answer is still a tool's event in the audit log, and with the model
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
                # [LAW:nothing-unseen] the session the focus stood in for rides on the result, so its event says where the call went.
                return {**await tool.body(session=focus, **arguments), "focused_session": focus}

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
            to = None if session == "" else SessionId(session)
            await move_focus(sessions, home, to)
        except (Rejected, NotRunning, OSError) as error:
            return {"error": str(error)}
        return {"readback": "No session is focused now." if to is None else f"Now on {spoken_name(sessions, to)}."}

    return tool(focus_session, completes=True)


def list_sessions_tool(sessions: Sessions, overlays: Overlays, home: Home, environment: Mapping[str, str]) -> Tool:
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
        none (focus_session), null when none is focused, or says why it cannot be read. `tmux` is the tmux pane a
        session runs in, as it is now: its server's `socket`, the `pane`, and the tmux `session` and `window` it is in,
        so keys reach it by `tmux -S <socket> send-keys -t <pane>`; or "not in tmux", or says why it cannot be read.
        """
        listings = sessions.live()
        panes, focus = await asyncio.gather(tmux.panes([listing.session.membership.pid for listing in listings], environment), _focus(home))
        return {
            "sessions": [{**describe_listing(listing), "overlay": await _overlay(overlays, listing.session.membership.id), "tmux": _in_pane(pane)} for listing, pane in zip(listings, panes, strict=True)],
            "focus": focus,
        }

    return tool(list_sessions)



def read_screen_tool(sessions: Sessions, environment: Mapping[str, str]) -> Tool:
    async def read_screen(session: str) -> Result:
        """What a session's terminal shows now, as text, read off the tmux pane it runs in.

        Call this before you send keys to a session at a dialog hands let go of, to see the dialog and its options, and
        when the user asks what such a dialog asks: say what it asks and its options in a sentence or two, never the
        screen whole. `screen` is the pane's text, down to its last line drawn on; `tmux` is the pane it was read
        from, as list_sessions names it. A session in no tmux pane has no screen hands can read, and the error says so.

        Args:
            session: The session's id, from list_sessions.
        """
        id = SessionId(session)
        live = sessions.live_session(id)
        if live is None:
            return {"error": f"no session {session} is running: list_sessions names the ones that are"}
        [pane] = await tmux.panes([live.membership.pid], environment)
        match pane:
            case Pane(socket=socket, id=pane_id):
                match await tmux.shown(environment, socket, pane_id):
                    case str(screen):
                        return {"screen": screen, "tmux": _in_pane(pane)}
                    case Unanswered(reason=reason):
                        return {"error": f"the screen of {spoken_name(sessions, id)} could not be read: {reason}", "tmux": _in_pane(pane)}
            case NotInTmux():
                return {"error": f"{spoken_name(sessions, id)} runs in no tmux pane, and hands reads a screen only off one"}
            case PaneUnread(reason=reason):
                return {"error": f"which tmux pane {spoken_name(sessions, id)} runs in could not be read: {reason}"}

    return tool(read_screen)


def _in_pane(pane: InPane) -> str | Mapping[str, str | int]:
    match pane:
        case Pane(socket=socket, id=id, session=session, window=window):
            return {"socket": str(socket), "pane": id, "session": session, "window": window}
        case NotInTmux():
            return "not in tmux"
        case PaneUnread(reason=reason):
            return {"cannot_read": reason}


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
        # status has not been read yet may be mid-turn, and a sentence of half a turn would be kept for good. One whose
        # only work is subagents in the background is at its prompt too, though Claude Code says busy.
        delegates = live is not None and delegating(live)
        over = live is None or isinstance(live.state, Idle) or delegates
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
            "working": live is not None and isinstance(live.state, Running) and not delegates,
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
        sentence is being written. `directory` is where this backlog was read; lit run there works its tracker.
        `tracked` false means lit has no workspace there, so the project has no backlog at all.

        Args:
            session: The id, from list_sessions, of a session working in the project.
        """
        match await _backlog(sessions, store, session):
            case str() as error:
                return {"error": error}
            case Untracked(project):
                return {"directory": str(project), "tracked": False, "items": []}
            case (project, backlog, said):
                roots = backlog.roots()
                return {
                    "directory": str(project),
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
            case Untracked(project):
                return {"error": f"{project} has no backlog, so no ticket {ticket}: lit has no workspace there"}
            case (_, backlog, said) if ticket not in backlog.tickets:
                return {"error": f"the backlog has no ticket {ticket}"}
            case (_, backlog, said):
                parent = backlog.parent.get(ticket)
                found = backlog.tickets[ticket]
                return {
                    **_ticket_line(backlog, said, ticket),
                    **({} if parent is None else {"parent": _ticket_line(backlog, said, parent)}),
                    "open_children": [_ticket_line(backlog, said, child) for child in backlog.open_children(ticket)],
                    **({"description": found.description, "comments": [{"by": comment.by, "at": comment.at, "body": comment.body} for comment in backlog.comments.get(ticket, ())]} if full else {}),
                }

    return [tool(read_backlog), tool(read_ticket)]


async def _backlog(sessions: Sessions, store: SummaryStore, session: str) -> tuple[Path, Backlog, Mapping[str, str]] | Untracked | str:
    """The session's project, its backlog read fresh, and every sentence already said of it; that lit has no workspace
    there, so it has no backlog; or why it could not be read."""
    member = sessions.membership(SessionId(session))
    if member is None:
        return f"there is no session {session}"
    try:
        backlog = await read_backlog(member.cwd)
    except (Unread, Rejected) as error:
        # [LAW:no-silent-failure] the model is told why it got nothing, and the log keeps it.
        logger.error(f"cannot read the backlog of session {session} in {member.cwd}: {error}")
        return f"the backlog in {member.cwd} could not be read: {error}"
    if isinstance(backlog, Untracked):
        return backlog
    # Every read is a sighting: what changed since the last pass is said in the background, never while this call waits.
    store.want(member.cwd)
    return member.cwd, backlog, store.reckon(backlog.thing()).said


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
    """Every live session by its id, name, state and mode, as list_sessions describes each before what it adds."""
    return [describe_listing(listing) for listing in sessions.live()]


def describe_listing(listing: Listing[Session]) -> dict[str, str]:
    return {
        "id": listing.session.membership.id,
        "name": identifier(listing),
        "state": _spoken_state(listing.session),
        "mode": "not reported yet" if listing.session.mode is None else spoken_mode(listing.session.mode),
    }


def _spoken_state(session: Session) -> str:
    match (session.dialog, session.state, session.turn):
        case (Held(on=on), _, _):
            # Said by what it asks, though the status saying it waits may not have been read yet.
            return _waiting_on(on)
        case (LetGo(on=on), _, _):
            return f"{_waiting_on(on)} on its screen, a dialog hands let go of and its answer tools no longer reach"
        case (None, Running(status=Busy()), Opened(latest=Doing() as latest)):
            # [LAW:one-source-of-truth] the progress a session that is not the focus is noted with, for the brain, whose
            # notes are this listing at the tail of its every request.
            return f"working; the last thing it set out to do: {said((latest,))}"
        case (None, Idle(status=status.Idle()), Opened()):
            # A turn opened at its prompt, before the status that says it is busy is read.
            return "working"
        case (None, _, _) if delegating(session):
            # No turn runs: a prompt typed at it runs at once, and each subagent's report opens a turn of its own.
            return f"idle, with {counted(len(session.background), 'subagent')} it started in the background still working"
        case (None, state, _):
            return _stated(state)


def _stated(state: SessionState) -> str:
    match state:
        case Unreported():
            return "not reported yet"
        case Idle(status=Shell()):
            # No turn runs: it does nothing until that task ends and its notification opens one.
            return "idle, with a shell command it started in the background still running"
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
        case Busy():
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


def tell_turn_tool(sessions: Sessions, recounts: Recounts, refocus: Refocus) -> Tool:
    async def tell_turn(session: str) -> Result:
        """What a session's last finished turn did, as hands tells a turn when it finishes, and how it stands now.

        Call this when the user asks what a session just did or how its last turn went. Tell them as the returned turn
        says to. `now` is how the session stands at this moment, as list_sessions says it: a question the turn ended
        on may have been answered at the keyboard since, and a session working again is no longer waiting on it.
        The session told becomes the focus, as a session whose turn hands tells as it finishes does.

        Args:
            session: The session's id, from list_sessions.
        """
        id = SessionId(session)
        live = sessions.live_session(id)
        if live is None:
            return {"error": f"no running session has the id {id!r}; take one from list_sessions"}
        name = spoken_name(sessions, id)
        now = _spoken_state(live)
        match recounts.of(id):
            case None:
                # [LAW:no-silent-failure] said as what it is, never as a turn that did nothing.
                return {"error": f"no turn of {name} has finished since hands started; read_session reads what it did before"}
            case Recount(tellings=(), unread=False):
                turn = f"[hands] The Claude Code session {name} finished a turn with nothing in it hands could tell. Tell the user so."
            case Recount(tellings=tellings, unread=unread):
                failed = (f"[hands] hands could not read {'the rest of ' if tellings else ''}the turn the Claude Code session {name} finished. Tell the user so.",)
                turn = "\n\n".join((*(told(id, name, (telling,), "full") for telling in tellings), *(failed if unread else ())))
        # Told as a turn that finishes is told, so the focus moves to it as it does then, recorded as that move is.
        await refocus(id)
        return {"turn": turn, "now": now}

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
        id = SessionId(session)
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
        # [LAW:nothing-unseen] the depth rides on the result, so the call's event says how far down this asking went.
        return {"parts": [{"part": topic, "told": told} for topic, told in drilled.told], "deeper": drilled.deeper, "depth": depth}

    return tool(expand)


# How much of the sessions' closing words one catch-up carries, shared among the sessions that finished: a morning away
# can be a hundred sessions, and every one of them is still named. Each gets its share, and never less than enough to
# say what came of its work.
CATCH_UP_CLOSINGS = 12000
CATCH_UP_LEAST = 150


def _shown(value: object) -> object:
    """A field of an occurrence as catch_up hands it on: text, or a call's input, bounded as a closing's least share is."""
    match value:
        case str():
            return cut(value, CATCH_UP_LEAST)
        case dict():
            return cut(json.dumps(value, ensure_ascii=False), CATCH_UP_LEAST)
        case _:
            return value


def catch_up_tool(sessions: Sessions, home: Home, now: Callable[[], datetime]) -> Tool:
    async def catch_up(minutes: int = 0) -> Result:
        """What the user missed: every session that finished work while they were away and the newest words each closed
        a turn with, the sessions that ended, what their hooks said happened, and what hands announced.

        Call this when the user asks what they missed, what happened while they were away, or what went on in the last
        while. Sum it up the way a colleague would after a break: each session that finished in a sentence, by name, then
        any that ended. Leave nothing out of `finished`: the user is asking because they heard none of it. `turns` is
        how many turns a session finished; read_session reads what each did, when they want more of one. `occurred` is
        what sessions' hooks said happened, each kind once a session with how many `times` and the newest one's
        fields, by its type: AutoDenied, auto mode refusing a call; SubagentStarted and SubagentStopped; TaskCompleted;
        ConfigChanged, its settings or skills changing; Compacting; Cleared, a /clear.

        Args:
            minutes: How far back to look, when the user says, such as 60 for "the last hour". 0 for since they last spoke to you before this.
        """
        if minutes < 0:
            return {"error": f"minutes is how far back to look, so it cannot be {minutes}"}
        at = now()
        opening = catchup.LastSpoke() if minutes == 0 else at - timedelta(minutes=minutes)
        try:
            missed = await asyncio.to_thread(catchup.missed, home.audit, opening)
        except OSError as error:
            return {"error": f"hands could not read its log: {error}"}
        share = max(CATCH_UP_LEAST, CATCH_UP_CLOSINGS // max(1, len(missed.finished)))
        return {
            # [LAW:nothing-unseen] where the window opened and what of it could not be read ride on the result, so the
            # call's event says what was read.
            "since_minutes_ago": None if missed.since is None else round((at - missed.since).total_seconds() / 60),
            "unreadable_lines": missed.unreadable,
            "finished": [
                {"session": spoken_name(sessions, done.session), "turns": done.turns, "closing": None if done.closing is None else cut(done.closing, share)}
                for done in missed.finished
            ],
            "ended": [spoken_name(sessions, session) for session in missed.ended],
            "occurred": [
                {"session": spoken_name(sessions, happened.session), "times": happened.times, **{key: _shown(value) for key, value in happened.newest.items()}}
                for happened in missed.occurred
            ],
            "announced": [{"text": said.text, "times": said.times} for said in missed.announced],
        }

    return tool(catch_up)


class Change(TypedDict):
    kind: Kind
    level: Literal["off", "brief", "full", "on"]


def attention_tool(home: Home) -> Tool:
    async def attention(changes: list[Change]) -> Result:
        """Change what you say to the user without being asked, or hear what is set: with no changes, it only says.

        Kinds, and their levels:
        - finished: each turn a session finishes. off holds it until they ask (tell_turn, catch_up), except a watched
          session's; brief tells it in a few words; full tells what it did.
        - progress: the focused session's steps as it works. off says none; brief says what it says it is doing; full
          says each step.
        - ended: a session ending, on or off.
        - Claude Code's hooks, each off, brief (what happened), or full (and what the hook says of it), and off until
          set: permission_denied, auto mode refusing a call; subagent_start and subagent_stop, a subagent the session
          started starting and finishing; task_completed, a task on its list marked done; config_change, its settings
          or skills changing; pre_compact, its context about to be compacted; clear, a /clear.
        - quiet: on holds everything above, whatever its level, until it is off again, and leaves the levels as they
          were. "Be quiet for a while" is quiet on; "you can talk again" is quiet off.
        What a session asks, a permission, a question, or a plan, is said whatever is set: it needs an answer.

        Call this as soon as the user says what they want to hear more or less of, mapping their words to the nearest
        kind and level yourself, several changes at once if they said several; never ask which they meant. "Stop telling
        me when sessions finish" is finished off; "just the headlines" is finished brief; "less detail while it works" is
        progress brief. Call it with no changes when they ask what you tell them. It lasts until they change it, across
        restarts, from the next thing you would say. Say the returned readback to the user; for quiet on, say in a few words
        that you will keep quiet, never nothing, so they know it took.

        Args:
            changes: each kind to set and its level, in the order said; none to hear what is set.
        """
        try:
            said = [(change["kind"], change["level"]) for change in changes]
            to = await asyncio.to_thread(settings.asked, home, said)
        except (Rejected, OSError) as error:
            return {"error": str(error)}
        return {"readback": settings.described(to)}

    return tool(attention, completes=True)


def modality_tool(switch: Callable[[Modality], None]) -> Tool:
    async def set_modality(modality: Modality) -> Result:
        """Take the user as able to see a screen, or as audio-only, from now on.

        Each of their turns says which they are: talking at the Mac starts as screen, and from the phone as audio-only.
        This switches it until hands next moves between the Mac and the phone, as a call comes or goes. It is a hint for choosing what to do, never a limit on what
        you can do. Call this when the user asks to go audio-only, or back to a screen. Say the returned readback.

        Args:
            modality: screen, or audio-only.
        """
        switch(modality)
        return {"modality": modality, "readback": _modality_readback(modality)}

    return tool(set_modality, completes=True)


def _modality_readback(modality: Modality) -> str:
    match modality:
        case "screen":
            return "Okay, you can see a screen."
        case "audio-only":
            return "Okay, audio only."


def trigger_tools(triggers: Triggers, models: Path, wake: Word) -> list[Tool]:
    """Which trigger opens the user's turns at the Mac: saying the one in use, and switching to another while hands runs;
    `wake` is the wake word, and `models` where its models are kept."""

    async def trigger_in_use() -> Result:
        """Say which trigger is in use: the way the user opens a turn at the Mac.

        Call this when the user asks how to talk to hands, or which trigger is on. Say the returned readback.
        """
        return {"trigger": triggers.in_use, "readback": described(triggers.in_use, wake)}

    async def set_trigger(trigger: Trigger) -> Result:
        """Switch the trigger the user opens their turns with at the Mac; their next turn opens the new way.

        Call this when the user asks for a trigger by name. A trigger not listed here is not built: tell them so, and
        that the one in use stays on. Say the returned readback.

        Args:
            trigger: the trigger to use from now on.
        """
        try:
            fetched = await readied(trigger, models, wake)
        except (aiohttp.ClientError, TimeoutError, OSError, Unheard) as error:
            return {"error": f"{trigger} could not be readied: {error}", "trigger": triggers.in_use, "readback": f"The {trigger} could not be set up, so the trigger stays as it was."}
        was = triggers.choose(trigger)
        # [LAW:nothing-unseen] the trigger it replaced and the files fetched for it land on the call's event, so a switch,
        # a no-op, and a first switch that fetched read apart.
        return {"trigger": trigger, "was": was, "fetched": list(fetched), "readback": f"{'Already on' if was == trigger else 'Okay'}. {described(trigger, wake)}"}

    return [tool(trigger_in_use), tool(set_trigger, completes=True)]


def voice_tools(voices: Voices) -> list[Tool]:
    """Choosing the voice hands speaks in, by ear: the user hears the voices said by hands, and keeps one.

    [LAW:nothing-unseen] each result names the voice hands speaks in after the call, and a hearing the voices it said,
    so the call's event holds what was heard and what was kept.
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


class OwnModel(Protocol):
    """The model hands itself runs on (hands.daemon.config.OwnModel): hands has no session id, so what targets it is a
    tool of its own, never a session's."""

    def running(self) -> str: ...

    def weigh(self, model: str) -> Callable[[], None]: ...


def model_tools(own: OwnModel, player: Player) -> list[Tool]:
    """Choosing the model hands runs on, which is a setting of hands' own and never a session's.

    [LAW:nothing-unseen] each result names the model hands runs on and, for a choice, the one it switches to, so the
    call's event holds both.
    """

    async def model_in_use() -> Result:
        """The model you, hands, run on now. Call this when the user asks what model you run on."""
        return {"running_on": own.running()}

    async def use_model(model: str) -> Result:
        """Run you, hands, on another model: hands restarts on it, back in a few seconds, and stays on it across restarts.

        Call this when the user asks to change your own model. It never touches a session; a session's model is changed
        by send_command running model in it. Hands says it is switching, and switches once the user has heard that:
        calling it is the whole reply, so add no words of your own.

        Args:
            model: the model's id, such as claude-opus-5-5.
        """
        try:
            keep = await off_loop(functools.partial(own.weigh, model), "weighing a model chosen by voice")
        except Rejected as error:
            return {"error": str(error)}
        line = f"Switching to {model.strip()}. I'll be back in a few seconds."
        # [LAW:no-ambient-temporal-coupling] written only once heard: the edit restarts hands, which would cut the line off.
        if not await player.heard(line):
            return {"error": f"The user spoke over hands saying it would switch to {model.strip()}, so it did not switch. Ask whether they still want it."}
        try:
            await off_loop(keep, "keeping a model chosen by voice")
        except (Rejected, OSError) as error:
            return {"error": str(error)}
        return {"said": line, "running_on": own.running(), "switching_to": model.strip()}

    # Completes through a barge-in: the barge-in is what `heard` reports, and the model is told the switch did not happen.
    return [tool(model_in_use), tool(use_model, then="silence", completes=True)]


def usage_tool(usage: Usage) -> Tool:
    """[LAW:nothing-unseen] the result is the reading, so the call's event holds it."""

    async def context_usage() -> Result:
        """How many tokens your main conversation holds now, and how many you have spent since hands started, as the API
        counted them on your replies.

        Call this when the user asks how much context or how many tokens you have used, or when you need to know how full
        your context is. These are the real figures; a token total in your system reminders is not your usage. A subagent
        you start has its own context, which this does not measure.
        """
        reading = usage.reading()
        if reading is None:
            return {"error": "No reply of yours has been counted yet."}
        return {
            "model": reading.model,
            "in_context_tokens": reading.in_context,
            "spent": {"input_tokens": reading.spent.input_tokens, "output_tokens": reading.spent.output_tokens, "replies": reading.spent.replies},
        }

    return tool(context_usage)


def spotify_tools(player: spotify.Player, catalogue: spotify.Catalogue) -> list[Tool]:
    """Spotify on this Mac: what it plays, played, paused, skipped, and set, and its catalogue searched for what to play.

    [LAW:nothing-unseen] each result is what Spotify said or did, or why it would not, so the call's event holds it.
    """

    async def spotify_now_playing() -> Result:
        """What Spotify is playing on this Mac: whether it is playing, paused, stopped, or not running, and the track."""
        return await _refusable(_now(player))

    async def spotify_play(link: str = "") -> Result:
        """Play music in Spotify on this Mac, starting Spotify if it is not running.

        To play something the user names, find it with spotify_search and pass the link of the one they meant.

        Args:
            link: The spotify: link of the track, album, artist, or playlist to play from its start, or an
                open.spotify.com link the user gave. Leave it out to go on with what was playing.
        """
        uri = spotify.link(link) if link else None
        if link and uri is None:
            return {"error": f"{link!r} is no Spotify link; find what to play with spotify_search"}
        return await _refusable(_done(player.play(uri), {"playing": uri or "what was playing"}))

    async def spotify_pause() -> Result:
        """Pause Spotify on this Mac."""
        return await _refusable(_done(player.pause(), {"paused": True}))

    async def spotify_skip(to: Literal["next", "previous"]) -> Result:
        """Skip Spotify on this Mac to the next track, or back to the previous one.

        Args:
            to: Which track to skip to.
        """
        return await _refusable(_done(player.skip(to), {"skipped_to": to}))

    async def spotify_volume(level: int) -> Result:
        """Set Spotify's own volume on this Mac, which is not the Mac's.

        Args:
            level: From 0, silent, to 100, its loudest.
        """
        if not 0 <= level <= 100:
            return {"error": f"Spotify's volume runs from 0 to 100, not {level}"}
        return await _refusable(_done(player.volume(level), {"volume": level}))

    async def spotify_shuffle(on: bool) -> Result:
        """Turn Spotify's shuffle on or off on this Mac.

        Args:
            on: True to shuffle, false to play in order.
        """
        return await _refusable(_done(player.shuffle(on), {"shuffle": on}))

    async def spotify_repeat(on: bool) -> Result:
        """Turn Spotify's repeat on or off on this Mac.

        Args:
            on: True to repeat, false not to.
        """
        return await _refusable(_done(player.repeat(on), {"repeat": on}))

    async def spotify_search(query: str, kind: spotify.Kind = "track") -> Result:
        """Search Spotify's catalogue for something to play: up to five, best match first, each with the link
        spotify_play takes.

        Args:
            query: What the user named, in their words: a song, an album, an artist, or a playlist, and who made it
                where they said.
            kind: What they asked for.
        """
        return await _refusable(_searched(catalogue, query, kind))

    return [
        tool(spotify_now_playing),
        tool(spotify_play, completes=True),
        tool(spotify_pause, completes=True),
        tool(spotify_skip, completes=True),
        tool(spotify_volume, completes=True),
        tool(spotify_shuffle, completes=True),
        tool(spotify_repeat, completes=True),
        tool(spotify_search),
    ]


async def _refusable(act: Awaitable[Result]) -> Result:
    """What `act` handed back, or, where Spotify refused it, why: a refusal the model is asked to answer."""
    try:
        return await act
    except spotify.Refused as refused:
        return {"error": str(refused)}


async def _now(player: spotify.Player) -> Result:
    return spotify.said(await player.now_playing())


async def _done(act: Awaitable[None], result: Result) -> Result:
    await act
    return result


async def _searched(catalogue: spotify.Catalogue, query: str, kind: spotify.Kind) -> Result:
    return {"found": [spotify.said(found) for found in await catalogue.search(query, kind)]}


def start_session_tool(home: Home, record: Record, environment: Mapping[str, str], members: Callable[[], Iterable[Membership]]) -> Tool:
    """[LAW:nothing-unseen] each start is its own session.start event, inside the call's."""

    async def start_session(folder: str, model: str = "") -> Result:
        """Start a new Claude Code session for the user: `claude` in the folder, in a new window of the tmux session named
        for the folder, as the user would start it at a terminal. Returns once the session has joined hands, with its id
        and the tmux pane it runs in, or says why it did not. A session started has been told nothing.

        Args:
            folder: The folder the session works in, its whole path or one from ~.
            model: The model it starts on, as the user named it (opus, sonnet, haiku, or a model id). Empty for Claude Code's own default.
        """
        try:
            started = await startsession.start(home, record, Path(folder), model or None, environment, members)
        except startsession.NotStarted as why:
            return {"error": str(why)}
        return {"session": started.session, "tmux_session": started.tmux_session, "pane": started.pane}

    # A barge-in never stops a start part way: the session would be running, and the user never told.
    return tool(start_session, completes=True)


def close_session_tool(home: Home, record: Record, sessions: Sessions) -> Tool:
    """[LAW:nothing-unseen] each close is its own session.close event, inside the call's."""

    async def closed(id: SessionId, asked: closesession.Asked) -> Result:
        # Its name as the user knows it, read while it is still listed.
        name = spoken_name(sessions, id)
        try:
            match await closesession.close(home, record, sessions.live_session, id, asked):
                case closesession.Closed():
                    return {"closed": name}
                case closesession.LeftRunning(session=found):
                    # [LAW:one-source-of-truth] what it was found doing, as list_sessions says it.
                    return {"left_running": name, "state": _spoken_state(found)}
        except closesession.NotClosed as why:
            return {"error": str(why)}

    async def close_session(sessions: list[str], asked: closesession.Asked) -> Result:
        """End Claude Code sessions the user is finished with: each one's claude exits, as at a closed terminal, and it
        leaves your session listing. Returns once they have, with each one's name, or says why one did not. A tmux window
        hands opened for one closes with it; a terminal the user started one from stays.

        Args:
            sessions: Each session's id, from list_sessions. They are closed together, so one call closes them all.
            asked: `named` when the user named these sessions: each ends whatever it is doing. `done` when they asked for
                the sessions that are done: each ends only at its prompt, with no dialog up and no shell or subagent
                running in the background, and is otherwise left running, which its result says.
        """
        return {"sessions": list(await asyncio.gather(*(closed(SessionId(session), asked) for session in sessions)))}

    # A barge-in never stops a close part way: the sessions would be ending, and the user never told.
    return tool(close_session, completes=True)


def set_overlay_tool(sessions: Sessions, overlays: Overlays) -> Tool:
    async def set_overlay(session: str, overlay: Overlay) -> Result:
        """Set how the user hears a session's finished turns: watched, normal, or muted.

        `watched` tells them each turn it finishes, as it finishes, even with finished turns off. `normal` tells its
        turns as finished turns are set to be told (attention), and otherwise when they ask (tell_turn); every session is
        normal until they change it. `muted` tells its turns only when they ask. Whatever its overlay, a session's
        permission requests, questions, and plans are said: they need an answer. Call this with watched when the user
        asks to be told when a session finishes, with muted when they ask to stop hearing about it or to mute it, and
        with normal when they unmute or unwatch it. It lasts until they change it, across restarts. Say the returned
        readback to the user: it says how the session's turns now reach them, what is set considered.

        Args:
            session: The session's id, from list_sessions.
            overlay: watched, normal, or muted.
        """
        try:
            id = SessionId(session)
            # [LAW:single-enforcer] the registry is the one judge of which sessions are running.
            live = sessions.live_session(id)
            if live is None:
                raise Rejected(f"no running session has the id {id!r}; take one from list_sessions")
            await asyncio.to_thread(overlays.set, id, overlay)
        except (Rejected, OSError) as error:
            return {"error": str(error)}
        # [LAW:one-source-of-truth] the delivery the narrator computes, from what is set as it reads it, so the readback
        # says what will happen to the session's next turn.
        return {"readback": _overlay_readback(spoken_name(sessions, id), await set_to(lambda: settings.attention(overlays.home)), overlay)}

    return tool(set_overlay, completes=True)


def _overlay_readback(name: str, attention: Attention, overlay: Overlay) -> str:
    match delivery(attention, overlay):
        case Spoken(why=why):
            return {
                "watched": f"I'll tell you each turn {name} finishes.",
                "finished": f"I'll tell you each turn {name} finishes, as I tell every session's.",
            }[why]
        case Withheld(why="quiet"):
            # What quiet holds is told as it is set once hands talks again, so that is said too.
            return f"For now I'm keeping quiet and holding {name}'s turns. After that, {_overlay_readback(name, replace(attention, quiet='off'), overlay)}"
        case Withheld(why=why):
            return {
                "off": f"I'll hold {name}'s turns until you ask for one.",
                "muted": f"{name} is muted: I'll hold its turns until you ask. It still speaks when it needs your answer.",
            }[why]


class Resolved(TypedDict):
    heard: str
    meant: str


def draft_tools(sessions: Sessions) -> list[Tool]:
    """stage_draft, amend_draft, discard_draft, send_draft: a prompt dictated for a session, read back until it is right, then sent."""

    def aloud(outcome: DraftOutcome, name: str) -> Result:
        """A staged or amended draft is read back by hands, as written: what the user checks it by is spelled for the
        ear, and a model asked to say it would say it in its own words. Any other outcome changed nothing, so it is the
        model's to answer, as a refusal: to retry, as with the session's id, or to say what went wrong."""
        match outcome:
            case DraftStaged() | DraftAmended():
                return {"says": readback(outcome, name)}
            case _:
                return {"error": readback(outcome, name)}

    async def stage_draft(session: str, text: str, resolutions: list[Resolved]) -> Result:
        """Stage a prompt the user dictated for a session. It is not sent until the user says to send it.

        Hands reads the draft back to the user as it will be typed. Calling it is the whole reply: add no words of your own.

        Args:
            session: The session's id, from list_sessions.
            text: The prompt.
            resolutions: Each spoken phrase you turned into something exact, such as a file name, with what you made of it. Empty when you resolved nothing.
        """
        return await _answer(sessions, session, lambda id: StageDraft(id, parse_draft(text, resolutions)), sessions.draft, aloud)

    async def amend_draft(session: str, text: str, resolutions: list[Resolved]) -> Result:
        """Replace a session's staged draft with a corrected one when the user changes it.

        Hands reads back what changed. Calling it is the whole reply: add no words of your own.

        Args:
            session: The session's id, from list_sessions.
            text: The whole corrected prompt, not only the changed words.
            resolutions: Every resolution the corrected prompt relies on.
        """
        return await _answer(sessions, session, lambda id: AmendDraft(id, parse_draft(text, resolutions)), sessions.draft, aloud)

    async def discard_draft(session: str) -> Result:
        """Throw away a session's staged draft without sending it.

        Args:
            session: The session's id, from list_sessions.
        """
        return await _answer(sessions, session, DiscardDraft, sessions.draft, _for_the_model(readback))

    async def send_draft(session: str) -> Result:
        """Type a session's staged draft into it and press Return. Call it only once the user has said to send it.

        Say the returned readback to the user.

        Args:
            session: The session's id, from list_sessions.
        """
        return await _answer(sessions, session, SendDraft, sessions.draft, _for_the_model(readback))

    # A barge-in must not cancel a draft call part way: the draft would change, or be sent, without its readback heard.
    return [
        tool(stage_draft, then="silence", completes=True),
        tool(amend_draft, then="silence", completes=True),
        tool(discard_draft, completes=True),
        tool(send_draft, completes=True),
    ]


def _for_the_model[O](say: Callable[[O, str], str]) -> Callable[[O, str], Result]:
    """An outcome's readback handed to the model, which says it."""
    return lambda outcome, name: {"readback": say(outcome, name)}


async def _answer[R, O](
    sessions: Sessions,
    session: str,
    request: Callable[[SessionId], R],
    apply: Callable[[R], Awaitable[O]],
    say: Callable[[O, str], Result],
) -> Result:
    # [LAW:no-silent-failure] the model hears each failure and says it; the tool's event keeps it.
    id = SessionId(session)
    try:
        outcome = await apply(request(id))
    except Rejected as error:
        return {"error": str(error)}
    return say(outcome, spoken_name(sessions, id))


def keyboard_tools(sessions: Sessions) -> list[Tool]:
    """send_command, interrupt_session: what the user would otherwise do at a session's keyboard besides send it a prompt."""

    async def send_command(session: str, command: str, args: str = "") -> Result:
        """Run a slash command in a session, such as compact, clear, or model. Call it only when the user asks for a command by name.

        What the user dictates as a prompt is a draft, even when it begins with a slash; this is only for commands.
        Hands itself is not a session: its own model is changed by use_model.
        Say the returned readback to the user.

        Args:
            session: The session's id, from list_sessions.
            command: The command's name, such as "compact" or "model".
            args: What follows the name, such as "opus" for model. Empty when the user gave nothing.
        """
        return await _answer(sessions, session, lambda id: SendCommand(id, parse_command(command, args)), sessions.keyboard, _for_the_model(keyboard_readback))

    async def interrupt_session(session: str) -> Result:
        """Stop what a session is doing, as pressing Escape at its keyboard does. At a permission dialog that is the dialog's no.

        Say the returned readback to the user.

        Args:
            session: The session's id, from list_sessions.
        """
        return await _answer(sessions, session, Interrupt, sessions.keyboard, _for_the_model(keyboard_readback))

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
        return await _decide(sessions, request, lambda: parse_decision(decision, message))

    async def answer_question(request: str, answers: list[str]) -> Result:
        """Answer the questions a session asked with what the user chose. Call it only once the user has answered every one.

        Say the returned readback to the user.

        Args:
            request: The request id given with the questions.
            answers: One answer per question, in the order they were asked: the label of the option the user chose, or their own words when no option fits. Where more than one may be chosen, join the labels with ", ". An empty answer when the user chose none.
        """
        return await _decide(sessions, request, lambda: Answers(tuple(answers)))

    async def answer_plan(request: str, decision: str, message: str = "") -> Result:
        """Answer a session's plan with what the user decided. Call it only after the user has approved the plan or asked for changes.

        Say the returned readback to the user.

        Args:
            request: The request id given with the plan.
            decision: "approve" to approve the plan and go on in the mode the session had before it planned; "auto-accept edits" or "manually approve edits" only when the user says how edits should go; or "keep planning" to send it back.
            message: Only when it keeps planning: what the user wants changed, in their words.
        """
        return await _decide(sessions, request, lambda: parse_plan_decision(decision, message))

    # A barge-in must not cancel an answer part way: the user would never hear whether it went through.
    return [tool(body, completes=True) for body in (answer_permission, answer_question, answer_plan)]


async def _decide(sessions: Sessions, request: str, decision: Callable[[], Decision]) -> Result:
    # [LAW:no-silent-failure] the model hears a refused answer and says it; the tool's event keeps it.
    try:
        outcome = await sessions.answer(_request_id(request), decision())
    except Rejected as error:
        return {"error": str(error)}
    return {"readback": answer_readback(outcome, lambda id: spoken_name(sessions, id))}


def parse_decision(decision: str, message: str) -> Decision:
    """The model's answer, parsed once into the only two things a person can decide."""
    # [LAW:parse-dont-validate] a Decision is made here and nowhere else, so nothing but "allow" runs a tool.
    match (decision, message):
        case ("allow", ""):
            return Allow()
        case ("allow", _):
            raise Rejected("a message goes only with deny; an allow carries none, so nothing was answered")
        case ("deny", ""):
            return Deny(DENIED_BY_VOICE)
        case ("deny", _):
            return Deny(message)
        case (other, _):
            raise Rejected(f"decision should be 'allow' or 'deny', got {other!r}")


def parse_plan_decision(decision: str, message: str) -> Approve | KeepPlanning:
    """The model's answer to a plan, parsed once into an approval for a mode or a plan sent back."""
    match (decision, message):
        case ("keep planning", ""):
            return KeepPlanning(SENT_BACK_BY_VOICE)
        case ("keep planning", _):
            return KeepPlanning(message)
        case (choice, "") if choice in _APPROVALS:
            return Approve(_APPROVALS[choice])
        case (choice, _) if choice in _APPROVALS:
            raise Rejected("a message goes only with keep planning; an approval carries none, so nothing was answered")
        case (other, _):
            raise Rejected(f"decision should be 'approve', 'auto-accept edits', 'manually approve edits', or 'keep planning', got {other!r}")


def _request_id(request: str) -> RequestId:
    if not request:
        raise Rejected(f"request should be the request id string, got {request!r}")
    return RequestId(request)


def _unnamed(session: object) -> bool:
    """Whether the model named no session: empty, or null, which some models send for an argument left empty."""
    return session in ("", None)


def parse_draft(text: str, resolutions: list[Resolved]) -> Staged:
    """The model's arguments, parsed once into a draft whose text is safe to type."""
    # [LAW:parse-dont-validate] PromptText is made here and nowhere else.
    return Staged(_prompt_text(text, "the draft text"), tuple(Resolution(heard=item["heard"], meant=item["meant"]) for item in resolutions))


def parse_command(name: str, args: str) -> Command:
    """The model's arguments, parsed once into a command whose name and arguments are safe to type after a slash."""
    return Command(_command_name(name), _command_args(args))


def _command_name(name: str) -> CommandName:
    # [LAW:parse-dont-validate] CommandName is made here and nowhere else. A slash the model kept from what the user
    # said is the one the command is typed with, not a second.
    named = _COMMAND_NAME.fullmatch(name)
    if named is None:
        raise Rejected(f"command should be a slash command's name, such as compact, got {name!r}")
    return CommandName(named.group(1))


def _command_args(args: str) -> PromptText | None:
    if not args.strip():
        return None
    if "\n" in args:
        # A command is one line: what a line break does inside one, pasted, is unmeasured.
        raise Rejected("a command's arguments are one line, and these hold a line break")
    return _prompt_text(args, "the argument string")


def _prompt_text(text: str, what: str) -> PromptText:
    if not text.strip():
        raise Rejected(f"{what} is empty")
    if KEYSTROKES.search(text):
        # A tab is one: typed into a session it cycles the mode, and fritter refuses it by name at the socket.
        # Refusing it here instead means the model is told while it still has the words to fix.
        raise Rejected(f"{what} holds a control character, which would press a key when it is typed")
    if text.endswith("\\"):
        raise Rejected(f"{what} ends with a backslash, which turns the Return that sends it into a newline")
    return PromptText(text)
