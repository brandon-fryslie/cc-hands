# Architecture

`hands` is a single Python process that runs in a terminal. It consists of four packages
with a strict one-way dependency order: `daemon` → `voice` → `sessions` → `core`. Every
decision below supports one of the nine needs in [needs.md](needs.md). The work that
implements each need is planned in [features.md](features.md). Known failures, and the
rule each one produced, are listed in [failure-modes.md](failure-modes.md).

The design follows one principle: **the pure core makes decisions, the edges perform
actions, and every fact is stored in exactly one place.** State, events, and effects are
typed unions. The reducer converts an event into a new state and a list of effects. It
performs no I/O, so every lifecycle transition can be tested with a unit test that uses
no mocks. The adapters that perform the effects are thin, and there is exactly one
adapter for each kind of effect: one replies to blocked hooks, one controls the speaker,
and one (not yet built) types into sessions through the fritter that wraps them.

## Shape

```
                                  hands daemon
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │  voice                                                                        │
 │  mic ─► gate ─► Whisper (MLX)       ─► LLM ─► pocket-tts ─► speakers          │
 │         ▲                        ▲  │        ▲                                │
 │   gate edges              notes, │  │ tool   │ system speech                  │
 │   terminal · held key     narrate│  │ calls  │ (straight to TTS)              │
 │   button · phone · wake word     │  ▼        │                                │
 │  ───────────────────────────────────────────────────────────────────────────  │
 │  sessions                  ┌──────────────────┐                               │
 │  hook socket · session     │  core (pure)     │  procs                        │
 │  files · JSONL tail · git  │  types · reducer │  audit log                    │
 │                            │  steps · policy  │                               │
 │                            └──────────────────┘                               │
 └────────▲──────────────────────────────────────────────┬───────────────────────┘
          │ hook shims, unix socket                       │ hook replies · fritter sockets
          │                                               ▼
       target Claude Code sessions (any terminal, started any way)
```

The audio path on the left is the Pipecat pipeline that the spike already runs. The
Claude Code path on the right is the sessions package. The two paths interact only
through `core`'s types: hook events enter the pipeline as frames, and the pipeline's
tools call the sessions API.

## Packages and the direction of dependency

**`core`** contains the domain logic: the state types, the reducer, the step
recognisers, the spoken-form transform, the narration tree, the attention policy, and
the coalescing of pending speech. It imports only the standard library. A test enforces
this: `core` must not import `pipecat`, `subprocess`, `socket`, or `asyncio` streams.
Because of that test, `[LAW:effects-at-boundaries]` is enforced for the package instead
of depending on its authors' discipline. Everything in `core` can be tested with plain
values and no mocks.

**`sessions`** contains every edge on the Claude Code side: the unix socket that the
hook shims POST to, the session files that the shims write, Claude Code's own status
files, the JSONL tail, the git delta reader, the process-liveness check, the audit log,
and the wire proxy that a Claude Code process uses to reach the API
(`ANTHROPIC_BASE_URL`). The wire proxy passes every byte through unchanged, parses a copy
into `core.wire`'s typed events, and writes one `Exchanged` audit line per request. The
exchanges of the working sessions reach the same observer by a different path, the tap
(`sessions/tap.py`). Each session's fritter acts as that session's proxy: it forwards
the session's requests to the API that the session is configured to use, and copies each
exchange to `<hands home>/wire.sock`, where it is parsed into the same values. As a
result, a session never waits on the daemon, and its requests never pass through the
daemon. The wire proxy has one listener: the brain's stage, together with the component
that manages the brain's context. This listener routes each request in one of two ways.
Either the request is sent with hands' changes applied, or hands holds the request and
answers it without contacting the API. The possible changes are: no change, hands' tail
appended to the newest message, old tool results reduced to one line each, or a
compaction's prompt replaced. The package parses hook input once, at the socket, into a
`HookEvent`. It rejects any input it does not recognise, logs an error, and returns a
non-2xx reply `[LAW:parse-dont-validate]`. It provides two interfaces to the packages
above it: an async stream of events, and a small API that the tools call.

**`voice`** contains every edge on the audio side: the Pipecat pipeline, the gate and
the inputs that open and close it, the audio transport, the STT, LLM, and TTS services,
and the tool functions. It consumes the sessions event stream and routes each event to
one of the three speech channels described below. It never imports from `daemon`.

**`daemon`** is the composition root. It parses the config file into a frozen
`Config`, builds the sessions package and the voice package from that config, runs both
under one supervisor, publishes the heartbeat, and provides the `hands` CLI. Only
`daemon` reads the config file `[LAW:one-source-of-truth]`. Environment variables are
read only at process entry points: the daemon, and the two `sessions` modules that
Claude Code runs as separate processes (the hook shim and the attention tool). Every
other module receives the values it needs as arguments (`tests/test_environment.py`).

Dependencies point in one direction and never form a cycle `[LAW:one-way-deps]`. If a
lower package appears to need something from a higher package, the missing piece is a
type that belongs in `core`.

## The types are the design

These unions define the interfaces between components. They are shown here as Python
because a prose description would drift out of sync with the code that defines them,
while the code is checked by the type checker on every run
`[LAW:types-are-the-program]`.

```python
SessionId = NewType("SessionId", str)
Uuid      = NewType("Uuid", str)            # a JSONL record id
NarrationId = NewType("NarrationId", str)
SegmentId   = NewType("SegmentId", str)
Instant   = float                           # monotonic seconds

# What the shim records at SessionStart. The name is not stored here: it is the newest
# custom-title record in the transcript, and it is read when a listing needs it.
@dataclass(frozen=True)
class Membership:
    id: SessionId
    pid: int                  # the shim's parent, which is the claude process
    cwd: Path
    transcript: Path          # the JSONL, from the hook payload
    fritter: Path | None      # the socket used to type into it; None if it is not wrapped

@dataclass(frozen=True)
class Session:
    membership: Membership
    state: SessionState       # the state that Claude Code reports
    mode: PermissionMode      # from each hook payload that carries it
    turn: Turn                # the current turn, as reported by hooks and records
    dialog: Dialog | None     # the open dialog, as reported by its hooks; an idle status ends it
    background: frozenset[AgentId]  # background subagents it launched that have not yet reported back

# An ended session has no status, turn, or dialog: its last turn has been reported and
# its held hook was released when it ended. If it starts again, it is a new Session.
@dataclass(frozen=True)
class Gone:      membership: Membership
Known = Session | Gone        # what the registry holds per session

# Only a status read moves a session between these states. Whether a session is running
# comes from Claude Code's status and is never inferred from a hook or a record. Claude
# Code's busy status also covers a subagent working in the background (2.1.289), so busy
# with no open turn and a subagent still running is read as delegating(session): the
# session is at its prompt. An idle status means no subagent is running.
SessionState = Unreported | Idle | Running
@dataclass(frozen=True)
class Idle:      status: Idle | Shell; stamp: Stamp; after: PromptId | None  # one idle period; Shell: a background shell is running
@dataclass(frozen=True)
class Running:   status: Busy | Waiting | Unknown; stamp: Stamp; idled: Stamp

# Each fact about the turn is stored on the phase in which it is true.
Turn = Opened | Untold | Told
@dataclass(frozen=True)
class Opened:    turn: PromptId; others: frozenset[PromptId]; queued: bool
@dataclass(frozen=True)
class Untold:    turn: PromptId; others: frozenset[PromptId]; by: Stamp
@dataclass(frozen=True)
class Told:      turn: PromptId | None; others: frozenset[PromptId]

Dialog = Held | LetGo
@dataclass(frozen=True)
class Held:
    on: Blocker
    request: RequestId                    # the shim call waiting for a reply
    deadline: Instant                     # derived from the hook config's timeout
    warned: bool                          # the deadline warning has been spoken
    continues: Instant | None             # when Claude Code continues past a question on its own, if it does

# Every reason Claude Code stops arrives through the same PermissionRequest hook.
Blocker = Permission | Question | Plan
@dataclass(frozen=True)
class Permission: tool: str; input: Mapping[str, object]; suggestions: Sequence[object]
@dataclass(frozen=True)
class Question:   questions: Sequence[AskedQuestion]
@dataclass(frozen=True)
class Plan:       text: str             # a finished ExitPlanMode is PlanApproved: its input no longer carries the plan

# Input typed into a session. The variant determines the escaping, so no code checks
# whether the input starts with a slash: Text always escapes a leading sigil, Command
# never does, and Key is a named key combination that carries no text.
Input = Text | Command | Key
Keystroke = Literal["escape", "enter", "ctrl_c", "ctrl_u", "up", "down", "tab", "shift_tab"]

# The complete set of effects the reducer can produce. Adapters perform these and nothing else.
Effect = Reply | Type | Speak | Narrate | Play | Summarise | Snapshot | Audit
@dataclass(frozen=True)
class Reply:    request: RequestId; reply: HookReply
@dataclass(frozen=True)
class Type:     session: SessionId; writer: Fritter | Pane; input: Input  # typed through fritter, or through tmux into its pane
@dataclass(frozen=True)
class Speak:    text: str; priority: Priority                # straight to TTS
@dataclass(frozen=True)
class Narrate:  event: Narration; priority: Priority         # sent to the brain as a separate turn
@dataclass(frozen=True)
class Play:     narration: NarrationId; segment: SegmentId   # straight to TTS, bookmarked
@dataclass(frozen=True)
class Summarise: narration: NarrationId; steps: Sequence[Step]; expand: SegmentId | None
@dataclass(frozen=True)
class Snapshot: session: SessionId; cwd: Path                # the repository state when the turn opens
@dataclass(frozen=True)
class Compare:  session: SessionId; again: bool               # the turn's changes, read when the turn stops
@dataclass(frozen=True)
class Audit:    record: AuditRecord

Priority = Literal["blocking", "result", "fyi"]

# A narration is a tree of spoken segments. The top level plays first. Each segment has
# children, and every segment lists the records it summarises, so requests such as
# "that part" and "more on that" are resolved by lookup.
@dataclass(frozen=True)
class Segment:
    id: SegmentId
    kind: Literal["headline", "question", "section", "detail"]
    spoken: str                     # already in spoken form; goes straight to TTS
    refs: Sequence[Uuid]            # the records this segment summarises
    children: Sequence[SegmentId]   # empty until built, on request or ahead of time

@dataclass(frozen=True)
class Narration:
    id: NarrationId
    session: SessionId
    title: str
    kind: Literal["stop", "progress", "blocked", "subagent", "idle", "gone"]
    top: Sequence[SegmentId]        # headline, then questions, then sections
    segments: Mapping[SegmentId, Segment]

# The current playback position, and the positions where the user interrupted. Resume pops the stack.
@dataclass(frozen=True)
class Bookmark: narration: NarrationId; segment: SegmentId
@dataclass(frozen=True)
class Playback:
    playing: Bookmark | None
    interrupted: Sequence[Bookmark]  # most recent last

Draft = NoDraft | Staged
@dataclass(frozen=True)
class Staged:   text: str; resolutions: Sequence[Resolution]
```

The block above is the target design. `hands.core.effects` currently defines fewer
types: `Effect = Audit | Reply | Heard | Story`. `Heard = Speak | Narrate` carries
permission announcements and requests. `Story = Summarise | SessionGone` carries
finished turns and ended sessions. The current `Summarise` contains a session, the
prompt id of the turn that ended, and the reply from that turn's `Stop` hook, instead of
a narration id and steps. `Type`, `Play`, and `Snapshot`, and the segment, narration,
and playback types, are planned.

Two things are intentionally omitted. There is no `Session.last_seen` timestamp,
because the absence of events does not indicate anything; liveness is determined by
the pid. There is also no queue of unsent inputs, because Claude Code has its own input
queue and the daemon must not keep a second one `[LAW:one-source-of-truth]`.

## The reducer and its effects

```
reduce(state: Registry, event: Event) -> tuple[Registry, list[Effect]]
```

`Event` is the union of parsed hook events, steps from the transcript tail, tool calls
from the intermediary, ticks from the single clock, results of `Summarise` and
`Snapshot`, playback reports from the output transport, and liveness reports. The
reducer is a pure function `[LAW:effects-at-boundaries]`: it never reads a file, checks
a process, or reads a clock. When it needs the current time, it uses the time in a
`Tick` event it has already received. When it needs a summary, it emits `Summarise` and
receives the segments back as an event. Currently, no event comes back. `Summarise` is
emitted for a live session's `Stop`, for an interrupt, and for a turn that is still
running when a prompt names a different turn. The narrator reads the turn and passes it
to the intermediary to speak, and nothing is returned to the reducer.

The adapters are in `sessions` and `voice`, and each one performs one kind of effect:
`Reply` writes to the blocked shim's socket connection, `Speak` becomes a Pipecat
`TTSSpeakFrame`, `Narrate` becomes a `Narrated` frame that the brain's stage handles as a
separate turn, `Play` sends a segment to TTS through the player, `Summarise` passes the
turn to the intermediary, `Snapshot` records or diffs the target's git state, and `Audit`
appends one JSONL line. An adapter that fails raises an exception. The supervisor logs
it, and the failure is spoken on the system channel. Nothing is retried silently and
there are no fallbacks `[LAW:no-silent-failure]`.

The block above describes the design, not the current code. `core/effects.py` currently
has seven of those eight effects - `Audit`, `Reply`, `Type`, `Speak`, `Narrate`,
`Summarise`, `Snapshot` - plus `SessionGone` and `Compare`, which the block above does
not include. `Play` is not implemented. `Input` currently has only `Text`; `Command` and
`Key` are tracked in `hands-keyboard-gxr.i5n`. `Type` is emitted by `core.drafts.decide`
for a send, not by `reduce`, and `Sessions.draft` performs it. The draft is removed from
the registry as soon as the send is decided, so it is sent at most once. The result of
the send is whether its writer typed the text.

Because every transition is a call to `reduce` on values, the session lifecycle tests are
a table: state before, event, state after, effects. These tests use no pipeline, no
socket, and no keyboard.

## Each timing fact has one owner

Correctness never depends on incidental ordering `[LAW:no-ambient-temporal-coupling]`.
Each timing fact has one owner.

| What must be ordered | Owner |
|---|---|
| When a user turn starts and ends | the gate, based on the key position |
| Which utterance plays next, and that two utterances never overlap | Pipecat's output transport |
| Where a narration resumes after the user interrupts | the player's bookmark stack, based on the segment the output transport was playing |
| That a session's end is announced after its last turn | the story queue: `Summarise` and `SessionGone` wait in one queue, and the narrator announces each one in order |
| When a permission deadline warning is spoken and when the deadline expires | the reducer, based on `Held.deadline`, driven by one `Tick` source |
| Whether text typed during a turn is queued or lost | Claude Code's own input queue, which was measured to queue it |
| When the daemon is running, and restarting it | the user, with `hands run` in a terminal |

Deadlines are stored as data. The `Held` dialog stores the time it expires and whether
the warning has been spoken. A single ticker sends `Tick(now)` once per second. The
reducer compares the times and emits `Speak("ten seconds on that permission")` exactly
once, because the change from `warned=False` to `warned=True` is a state change, not a
timer callback. At the deadline, it emits `Reply(deny)` for a permission and announces
the denial. A question cannot be answered by silence, so it is withdrawn instead and left
to its dialog, where the user may be answering it at the keyboard. Its dialog then
becomes `LetGo`: it is still waiting, but only on the keyboard, until the answered call
returns. The ticker's period only limits how late a deadline is announced; no correctness
property depends on a `sleep`.

The deadline comes from one number. `hands.sessions.hookconfig` declares the timeout of
the `PermissionRequest` hook (90 seconds) in the plugin's `hooks/hooks.json`. The shim
waits that long for the daemon, for that hook only, and the daemon denies the request 5
seconds earlier, so the denial reaches Claude Code before Claude Code kills the hook
`[LAW:single-enforcer]`.

A question can time out earlier. Claude Code's `askUserQuestionTimeout` (`60s`, `5m`,
`10m`, or `never`, the default) measures idle time. It starts when the request is made,
runs alongside the hook, restarts on any interaction with the dialog, and when it expires
Claude Code continues with whatever answers are selected. When a `PermissionRequest` for
`AskUserQuestion` arrives, `hands.sessions.questiontimeout` reads the setting the same way
2.1.289 does: first the session's `--settings` (found in the arguments of its `claude`
process), then the user's `settings.json` in the session's config directory, unless
`--setting-sources` excludes user settings. Managed policy takes precedence over both and
is not read. The warning is spoken 10 seconds before the earlier of that timeout and the
hook's deadline, so the warning for a `60s` question is spoken at 50 seconds. Only Claude
Code knows when it actually continued: its `PostToolUse` event includes `afkTimeoutMs` in
the tool input, and at that point hands announces that the question continued without an
answer. The hook's deadline still applies as hands' own deadline. The hook's event
records what was read (`question_timeout`: the seconds, the settings that set them, and
any settings that were not read).

Claude Code queues messages submitted while a turn is running and displays them with
"Press up to edit queued messages". Measured on 2.1.270: text pasted into a working
session and submitted is added to that queue and runs when the turn ends. A send to a
working target is therefore an ordinary send, and the daemon holds nothing. A permission
dialog is the exception: it consumes pasted text and treats the Enter as "Yes".
Therefore, when the drafts are sent, a send to a session whose status is `waiting` is
refused, the draft remains staged, and the user is told why.

The workspace-trust dialog also consumes a paste (measured on 2.1.278), but it does not
need a separate rule. Claude Code runs no hooks until a startup dialog is answered, so a
session at that dialog has no membership and nothing can be sent to it
(`hands-harness-5nb.xw8`, measured on 2.1.283).

### Typing into a session

hands types into a session through **fritter** (`fritter/`), which runs the session's
`claude` on a pseudo-terminal and listens on a unix socket next to it. hands uses a
wrapper instead of synthetic key events because a keyboard types into whichever window
has focus, and the requirement is to drive a session while the display is asleep: no
window, no permission grant, and no focus.

fritter publishes its socket address to the process it wraps in `FRITTER_SOCKET`. The
hook runs as a child of that process and inherits the variable, so the address reaches
`Membership.fritter` without either side deriving a path from a pid. A session started
outside fritter has no address. If such a session is in the foreground of a tmux pane,
tmux types into that pane instead (`hands.sessions.tmux.typed`): the pasted text,
bracketed, and the Return are sent in one tmux command list, so either all of them are
delivered or none are. A session that is not in a pane, or that is behind another program
in a pane (stopped, inside an editor or ssh, or started from inside another session), is
refused by name instead of typing into whatever program receives the pane's keys.
`core.reach.writer` chooses a session's writer from its membership and the program in the
foreground of its pane (`core.tmux.keyboard_of`), which `Sessions` reads just before
deciding. fritter is chosen whenever it is available. The user does not need to remember
to wrap a session: `hands install-fritter` installs a `claude` in `<hands home>/bin` that
runs every interactive claude under fritter, and runs a pipe, a script, a subcommand, or
`claude -p` as the real claude with no address (`hands.sessions.wrapper`). A session
started from inside a wrapped session also inherits the address, so an address alone does
not identify which session it reaches. Every request therefore names the session's
`Membership.pid`, and fritter refuses a request whose pid is not the process it wrapped.

A send is typing: the text, pasted, and Return, in one write, as a user at the keyboard
would type it. hands decides *whether* a session may be written to, based on state that
fritter cannot see, and `Input` defines what a leading `/` means. fritter types the text
it is given. Any text that the user at the keyboard had partly typed stays in front of
the sent text, as it would if they pasted the text themselves.

`fritter/README.md` documents the protocol and the measurements.

## Three output routes

Every event that reaches the pipeline uses one of three routes. A table chooses the
route, not code that inspects the event `[LAW:dataflow-not-control-flow]`.

- **Speak.** Text is sent directly to TTS as a `TTSSpeakFrame`. There is no model call,
  no interpretation, and no latency beyond synthesis. Used for facts that a template can
  express: "auth-refactor finished", "ten seconds on that permission", "cc-hands is gone,
  its terminal closed", "the language model is unreachable". This channel is also how the
  daemon reports its own failures, which is why it must not depend on the LLM.
- **Play.** A narration's segments are sent to TTS one at a time through the player,
  already in spoken form. There is no model call at playback, because the summariser has
  already done that work. The player tracks which segment is playing, so an interruption
  leaves a bookmark. Used for a turn's results, progress while a session works, and a
  subagent's report.
- **Narrate.** The event is passed to the brain as a separate turn. The model interprets
  it and speaks. Used when the content is a conversation turn: a permission request, a
  question from `AskUserQuestion`, a plan.

Information that the brain needs to know but not speak does not need a route: the tail of
its next request reports the state of the sessions.

Currently, no table chooses the route; two queues are used instead. `Heard` carries
permission announcements as `Speak` and permission requests as `Narrate`, and relays
them as soon as the reducer emits them. A relayed item is not necessarily heard yet:
everything hands says about the sessions without being asked reaches the floor
(`voice/floor.py`) before the user aggregator, as a value, a `Pending`
(`core/pending.py`), not yet as a frame. From the key press that opens the user's turn
until that turn is sent, the value waits on the floor and is spoken after the user's
words. A value that arrives when no turn is open is released immediately, in the same
way. In both cases the floor converts it to frames only when it releases it, after
`coalesce` (below). For each item it announces, `coalesce` records where each item it
holds stood, so the items it drops are known because no announcement names them.

Each item that a session gives hands to say without being asked is one `utterance` wide
event (`voice/utterance.py`). The event is opened when the relay or the narrator receives
the item, and emitted when its outcome is known: `noted` (the route, the delivery, or a
turn or burst with no new content prevented it from being spoken), `dropped` (no longer
true by the time it was to be announced: out of date when the floor released it, or
progress of a turn that ended before its summary), `silent` (passed on, but none of it
played), `played`, or `cut`. Its fields record what was received and from which session,
what decided its route, how long the floor held it (`held_ms`), what form it was
announced in and how many received items were combined into that announcement, and
`first_audio_ms`, the time from receipt to the first audio the speaker wrote for it. The
`Audible` observer reads the last three outcomes from the output transport. Each
utterance is delimited by an `Uttering` frame, which a barge-in drops together with the
words that follow it, and an `Uttered` frame, which is kept through a barge-in. Every
utterance that is passed on therefore reaches the output transport closed, in order with
its audio. The brain's stage sends these frames around what it takes from hands' lane.
After a barge-in that the turn continues through, it starts the rest of the turn's speech
with an uninterruptible `Resumed`. It fails each utterance of an announcement that the
brain failed. The brain turn that announces an utterance is a child of that utterance in
its trace.
`Story` carries finished turns and ended sessions in one ordered queue, because a
session's end that was announced immediately was heard before the last turn it ended.
Every finished turn is read once into a `News` (`voice/narrator.py`, `recount`): the
session's last message, what its record adds, and what it is waiting on. `speech.told` is
the only place where it is converted into the input given to the model, under the
session's name at the time it is announced, for the model to say in its own words. How
that summary reaches the user is its `Delivery`. It is delivered as a separate brain
turn, a `Narrated` frame, when finished turns are configured to be announced, briefly or
in full, or when the session is watched (`Spoken`). Otherwise it is held (`Withheld`,
with the reason), and `tell_turn` passes it to the model when the user asks. In both
cases, each session's last summary is stored in `Recounts` as its `News`, and its
utterance records its delivery. No part of a turn is spoken verbatim without passing
through the model, and nothing is spoken about a session that is at its prompt.
A mode change is never spoken: the tail of the brain's next request reports the mode. Each
session has an overlay, `normal`, `watched`, or `muted`, stored as one file per session
under `~/.hands/overlays` (`hands/core/attention.py`), which the narrator reads at every
finished turn. What hands says without being asked is controlled by one setting,
`Attention` (`hands/core/attention.py`), stored in `~/.hands/attention.json`. The file
stores only the kinds the user has set, so a kind that was never set uses its default.
The setting has a level for each kind of unprompted speech — finished turns, the focused
session's progress, a session ending — and a quiet flag, which holds all of them without
changing their levels. `delivery`, `progress_route`, and `ended_route` are the tables
over this setting and the overlay. A muted session's turn is held regardless of the
settings, until the user requests it through `tell_turn`. The user sets the overlay by
voice with `set_overlay`, and sets unprompted speech with `attention` or
`/hands:attention`. A muted session's permission requests, questions, and plans are still
narrated: if they were held without being spoken, each would wait until its deadline and
be refused. Progress already has its row of the table, `attention.progress_route`, based
on the settings, the focus, and the overlay (see "Streaming"); the player and the table
for every other event kind below are planned.

The routing table is a value in `core`:

```python
Route = Literal["speak", "play", "narrate", "note", "drop"]
DEFAULT_POLICY: Mapping[EventKind, Route] = {
    "stop": "play", "progress": "note", "blocked": "narrate", "subagent_stop": "note",
    "gone": "speak", "session_start": "note", ...
}
```

The per-session overlay is a second table applied over the first: a muted session's
`play` becomes `note`, and its `narrate` stays, because a session that asks a question
needs an answer. Adding a new event kind adds a row, and adding an overlay value adds a
column `[LAW:one-type-per-behavior]`.

Pending speech is ordered when the floor releases it, and nothing starts while the key is
held down. `coalesce` (`core/pending.py`) is a pure function. First, it drops items that
are no longer true, based on the live sessions at the time the floor releases them: a
request that was answered at the keyboard while the user was talking, and a deadline
countdown for that request, identified by the held dialog's request id; and progress of a
turn that ended in the meantime, whether or not its end was announced. Second, it
combines one session's finished turns into one `Finished` at the position of the first
one. Its headline covers all of them ("finished 3 turns"), and its announcements keep
their narration parts and therefore their record ids. As a result, three `Stop`s that
arrived while the user was talking start with one sentence, not three. A turn that could
not be read stays between the turns it arrived between, and a turn that is announced
again because it continued past its `Stop` still counts as one turn. Third, it orders the
remaining items `known` (notes and the briefing, never spoken) before `blocking` (what a
session asks) before `result` (finished turns) before `fyi` (a session gone), in arrival
order within each group. A session's own items keep the order in which they occurred:
anything it announced before a higher-priority item is announced together with that
item, so the request for its next turn is never heard before the turn that preceded it. A
combined announcement shares one `REPLY_SHOWN` limit among its turns. A `Pending`'s
priority is derived from its variant and is never stored separately. This queue is not
the player's bookmarks: resuming replays a bookmarked sentence and never re-enqueues an
announcement. Each item is a transition, never a state, so nothing is announced twice
`[LAW:one-source-of-truth]`: a request is narrated, warned of, and expired by its request
id, which the daemon generates for each hook delivery, so a second delivery is a second
request. A turn's `Summarise` requests from the tail only the records it has not already
announced. The same event received twice is spoken once (`tests/test_reducer.py`).

## Hooks report events; the transcript records turns

Hooks report when events happen: a session starts, a prompt is submitted, a turn stops,
a permission is needed, or a line of Claude's text is ready. They fire when the event
occurs, and the blocking hook is the only way to answer a permission request. The
transcript records what a turn did, including the record id of each step. Claude Code
appends to the transcript while the turn runs, so the daemon tails the transcript
instead of hooking every tool call. Every hook input includes `session_id`,
`transcript_path`, `cwd`, and `hook_event_name`. Of the events hands subscribes to,
`UserPromptSubmit`, `Stop`, `PermissionRequest`, `PostToolUse`, and
`PostToolUseFailure` also include `permission_mode`; `SessionStart`, `Notification`,
and `SessionEnd` do not (verified live on 2.1.281). That value is the session's mode:
each hook that includes it sets the mode, `list_sessions` reports it, and the tail of
the brain's next request reports it. A hook fired inside a subagent includes the
subagent's mode and an `agent_id`, and does not set anything. Pressing Shift-Tab fires
no hook, and the transcript writes its `permission-mode` record only when a prompt is
sent. As a result, a mode change made at an idle prompt is detected at the session's
next prompt, and a mode change made mid-turn is detected at the next tool call. The
event-specific fields below were read from the 2.1.263 bundle. Payloads captured from
2.1.270 on 2026-09-14 contain no session title, so a session's name is taken from the
newest `custom-title` record in its transcript. Hands sets that name itself: the reply
of a `UserPromptSubmit` hook can include `hookSpecificOutput.sessionTitle` (2.1.286).
Claude Code writes it as a `custom-title` record and shows it on the terminal tab, the
same way `/rename` does (verified live on 2.1.286). The most recent name takes effect,
regardless of what set it.

After a session finishes each turn, hands asks the summariser's model whether the
session's name still fits, and sends it the name and the turn's final reply
(`hands.voice.naming`). A new name, of at most three words, is stored in `Names` until
that session's next prompt. Each check is recorded as one `name.judged` event, which
contains the previous name, any name that was decided but not yet applied (`pending`),
the reply, the decided name, and the outcome (`judged`: renamed or kept, or, if the
event fails, unread, failed, or refused). A hook can set a title only at session start
or at a prompt, so Claude Code's tab shows the new name starting from the session's
next prompt. Hands lists the session under the new name immediately, unless a
`/rename` has set a different name since the new name was decided (`Names.current`).
When spoken, a session is identified by its project followed by its name, as one
identifier: "cc-hands, naming fix".

| Event | Payload fields |
|---|---|
| `SessionStart` | `source`, `agent_type`, `model` |
| `UserPromptSubmit` | `prompt`, `prompt_id` |
| `Stop` | `stop_hook_active`, `last_assistant_message`, `prompt_id` |
| `PermissionRequest` | `tool_name`, `tool_input`, `permission_suggestions` |
| `PostToolUse`, `PostToolUseFailure` | `tool_name`, `tool_input`, `tool_use_id`, and the response or the error |
| `Notification` | `message`, `title`, `notification_type` in `permission_prompt`, `idle_prompt`, `auth_success`, `elicitation_dialog` |
| `SubagentStart` | `agent_id`, `agent_type` |
| `SubagentStop` | `agent_id`, `agent_transcript_path`, `agent_type`, `last_assistant_message` (absent for a report returned through a tool, 2.1.289) |
| `PermissionDenied` | `tool_name`, `tool_input`, `tool_use_id`, `reason` |
| `TaskCompleted` | `task_id`, `task_subject`, and optionally `task_description`, `teammate_name` |
| `ConfigChange` | `source`, and optionally `file_path` |
| `PreCompact` | `trigger`, `custom_instructions` |
| `MessageDisplay` | `turn_id`, `message_id`, `index`, `final`, `delta` |
| `SessionEnd` | the common fields |

No hook fires when the user interrupts a turn with Escape or Ctrl-C (2.1.281): there
is no `Stop`, no `PostToolUse` for the interrupted tool, and no `idle_prompt`
afterwards. Instead, Claude Code writes a user record, `[Request interrupted by user]`,
or `[Request interrupted by user for tool use]` if a tool was running. The record
includes the `promptId` of the interrupted turn, which is the same as the `prompt_id`
in that turn's `UserPromptSubmit`. The tail reads that record as the `Interrupted`
event. Claude Code sets the session to `idle` before it writes the record (37 ms
before, 2.1.283), and that status change is what moves the session to its prompt (see
Sessions, below). The record indicates that the turn Claude was responding to has
ended, and the spoken report of the turn waits for it. An interrupt that flushes a
queued message includes that message's id instead, and the turn continues under that
id. Only the prompt identifies the turn: a background subagent's hooks keep the
`prompt_id` of the turn that started the subagent, even after that turn has ended.

Only Claude Code's status moves a session between running and being at its prompt.
Hooks and records identify which turn is current and what it did. A turn's states are
`Opened`, then `Untold` once Claude Code's idle status ends it, then `Told`. A turn
opens when a prompt is received while no turn is open. That prompt marks the turn,
whether or not the busy status it set has been read yet. Input submitted while a turn
runs, such as a queued message or a task notification, fires `UserPromptSubmit` with
the running turn's `prompt_id` and does not change the status (2.1.283), so it belongs
to the turn whose id it carries. Claude Code assigns a new id to a prompt only at its
prompt, so a prompt with a different id means the open turn has ended. This holds even
when the `idle` status between the two prompts was set and set again within one status
read (93 ms apart, 2.1.283). In that case, the open turn is reported to the user and
compared while the new prompt's hook holds Claude Code, and the new turn is marked.
Any other id that a record carries while a turn is open belongs to a flushed message.
Claude Code takes that message seconds before Claude responds under its id, and the id
is added to the ids of the open turn. A `Stop` includes the `prompt_id` of the turn it
ends, which is how the tail finds that turn. A `Stop` ends only an open turn with that
id, so a `Stop` applied late never ends the following turn. A `Stop` received while no
turn is open reports a turn that hands never had open. Examples are the turn a session
was in when hands attached to it, or the last turn stopping again after another Stop
hook blocked its Stop. Such a `Stop` does not end any part of a last turn that has
already been reported. The session continues running after a `Stop` until Claude Code
reports that it is idle. A turn opened by a background task's notification fires
`UserPromptSubmit` with its own id, the same as a typed prompt.

A `PermissionRequest` hook prints its reply on stdout as
`{"hookSpecificOutput": {"hookEventName": "PermissionRequest", "decision": ...}}`,
where the decision is `{"behavior": "allow", "updatedInput"?: object}` or
`{"behavior": "deny", "message": string}`; empty output makes no decision (read from
the 2.1.270 bundle, and verified live: an allow runs the tool, and the agent receives a
deny's message as the tool's error). Permission prompts, plan approval, and
`AskUserQuestion` all arrive through this one hook. This is why `Held.on` is a union of
three types and the answer path is one adapter. To answer a question, hands allows it
with its original input plus an added `answers` object. The object maps each
question's text to the chosen label or the user's own words, with multiple labels
joined by ", ". This is the same format Claude Code's own dialog uses (read from the
2.1.280 bundle, and verified live: the agent continued with the answers given by
voice). A plan is approved the same way its own dialog approves it: an allow with an
empty `updatedInput`, so the plan is read from its file as the user left it, and
`updatedPermissions` holding the mode to switch to when leaving plan mode. That value
is either nothing, which returns the session to the mode it had before planning
(including bypass or auto), or one `setMode` to `acceptEdits` or `default` when the
user names one. Claude Code ignores an allow without `updatedInput` for a tool that
asks the user something, and shows its dialog instead. "Keep planning" is a deny, and
the agent receives its message as feedback (read from the 2.1.281 bundle, and verified
live). A plan that runs is reported through `PostToolUse` without its text, as
`PlanApproved`. If an answer does not match what was asked, hands sends nothing and the
session keeps waiting. This covers the wrong number of answers, a plain allow to a
question (which would run it unanswered), a plain allow to a plan, and plan feedback
sent for a tool's request.

The documented nine events are not the complete set. There are 33:

```
ConfigChange CwdChanged DirectoryAdded Elicitation ElicitationResult FileChanged
InstructionsLoaded MessageDisplay Notification PermissionDenied PermissionRequest
PostCompact PostModelSwitch PostToolBatch PostToolUse PostToolUseFailure PreCompact
PreModelSwitch PreToolUse SessionEnd SessionStart Setup Stop StopFailure SubagentStart
SubagentStop TaskCompleted TaskCreated TeammateIdle UserPromptExpansion
UserPromptSubmit WorktreeCreate WorktreeRemove
```

The daemon subscribes to the events in the table. `PermissionDenied`, `SubagentStart`,
`SubagentStop`, `TaskCompleted`, `ConfigChange`, and `PreCompact` are only forwarded
(`hands.core.occurrences`), as is a `SessionStart` from `/clear`. Each of these is
reported to the user according to the kind set by `/hands:attention` (off until set),
and is otherwise recorded as a noted utterance. These hooks run in the background. The
subagent hooks match `.+`, because Claude Code's own internal agents (prompt
suggestions, `/btw`) fire them with an empty `agent_type`. Each hook posted, to the
socket or the display route, is recorded as one `hook` wide event, which stays open as
long as the hook holds its session. The event records the hook and session, and the
branch that answered it: for a Stop, `decided` or `let go` at the hold; for a
permission, the reply it received (`cancelled` when its session closed it); for a
prompt, the name it set or withheld. A hook that the daemon refuses is recorded as a
failed event with the reason.

`MessageDisplay` is the only live source of Claude's text. The transcript writes a text
block once, after the block is complete (checked on 2026-09-14), while `MessageDisplay`
fires with each batch of completed lines as the text streams. It is dispatched
synchronously for every batch, even to a hook that declares itself async, so it is
installed as an HTTP hook, which Claude Code 2.1.270 supports: the event is POSTed to
the daemon without spawning a process, with a short timeout. Measured on 2.1.280 on
2026-09-24, with a 40-line reply:

- An interactive session fired 35 to 40 batches, one for every one or two completed
  lines, 0.1 to 0.5 s apart. `claude -p` fired one final batch containing the whole text.
- At most one batch is in flight at a time. Against an endpoint that took 1 s to
  respond, 16 batches arrived, each containing every line since the previous batch, and
  the last batch arrived 4.8 s after the turn ended.
- The turn took the same time against a fast endpoint, a 1-second endpoint, a refused
  port, and a port that accepts connections and never responds: 11.3 to 12.6 s in every
  case. A daemon that is slow, down, or hung loses narration lines, but never slows down
  the agent.

The daemon reads tool calls and their results from the tail, not from a hook.
`PostToolUse` and `PostToolUseFailure` are subscribed for one fact that the tail would
report too late to be useful: that a tool the daemon is still waiting on a permission
for has run, because its dialog was answered at the keyboard. They are declared
`async`, so the shim they spawn on every tool call never delays the agent.

**Installing the hooks.** The repository is a Claude Code marketplace
(`.claude-plugin/marketplace.json`) whose only entry, `hands@cc-hands`, has a `command`
source: Claude Code runs `hands plugin` and copies the directory it prints into its
plugin cache, at install and again once per session, so the hooks always come from the
installed hands. The plugin's files are package data, in
`src/hands/sessions/plugin`: `.claude-plugin/plugin.json`, `hooks/hooks.json`, and the
skills. A package cannot specify which interpreter runs it, so `hands plugin`
(`hands.sessions.marketplace`) copies these files and writes the launcher
`hooks/python` next to them. The launcher execs the interpreter that ran `hands plugin`
with `-I`. This keeps the session's directory (where Claude Code runs the hook) and the
user's `PYTHON*` variables off the path, so a project's own `json.py` or `hands/`
cannot replace hands' modules. `hands plugin` writes the files under the hands home
directory, in `plugins/<digest>`, a directory named by its content. The directory is
staged in full and then renamed into place, so sessions that start at the same time
never give Claude Code a partially written directory. Installing the plugin installs
the hooks; disabling or uninstalling it removes them, and hands does not edit any
settings file. `hooks.json` is generated from `hookconfig` (`python -m
hands.sessions.hookconfig > src/hands/sessions/plugin/hooks/hooks.json`), and a test
fails when the checked-in file differs from what `hookconfig` declares.
`MessageDisplay` is an HTTP hook to `hookconfig.DISPLAY_URL`, a fixed loopback port,
because Claude Code does not support variables in a hook's URL (2.1.288). The daemon
serves only that route on that port, and does not start if the port is in use. Every
other hook uses exec form: `${CLAUDE_PLUGIN_ROOT}/hooks/python -m hands.sessions.shim`,
spawned by Claude Code without a shell. The launcher execs, so the shim runs as the
process Claude Code spawned and its parent is the claude process. A launcher that ran
Python as a child process, as `uv run` does, would record its own pid instead. The
shim's home is `HANDS_HOME`, or `~/.hands`, resolved the same way the `hands` CLI
resolves it. A relative `HANDS_HOME` is rejected, because a hook runs in its session's
directory.

**The shims.** Each hook runs one shim process, which POSTs stdin to the daemon socket
and exits. At `SessionStart` the shim also writes
`~/.hands/sessions/<session_id>.json` with its parent pid, `cwd`, and
`transcript_path`. That file is the only record of the session's membership, and it
has a single writer. The file is written whether or not the daemon is running, so a
daemon started later finds the sessions that are already running. The hooks are
installed whether or not hands is running, so a shim that cannot reach the socket
checks the heartbeat to find out why (`hands.sessions.heartbeat.look`, the same check
`hands status` uses). If hands was stopped or never ran, it is off, not broken: the
shim exits 0 and prints nothing, and a permission request falls through to Claude
Code's own dialog. The same applies while hands is still starting: `hands run` writes
its first heartbeat before it imports Pipecat and serves the socket, and it reads the
session files once it serves the socket. If the heartbeat shows that hands died,
refused to start, hung, or is running but not responding, or if the heartbeat cannot be
read, the shim exits 1 and prints the socket error and the heartbeat result on stderr.
Claude Code then shows the failure in the session where it happened, instead of a dead
daemon appearing to be an idle one `[LAW:no-silent-failure]`.

A session that was running before the plugin was installed, or that was reloaded with
`/reload-plugins`, never fires `SessionStart`. So every other hook writes the file when
no membership file names the shim's parent process, and the daemon attaches the session
named in the file before it applies the hook. Such a session joins on whichever hook it
fires first. Keying on the process prevents a late hook from a session the process has
since left (after a `/clear`, a resume, or a `/branch`) from writing a file that would
take precedence over the new session's file.

**Timing constraint.** Hooks run in the agent's critical path with a timeout, and
`MessageDisplay` and `SessionStart` are dispatched synchronously. A shim that waits for
anything delays the agent's own output. Every shim POSTs and returns. The only
exception is `PermissionRequest`, where blocking is the intended behavior. Its timeout
is declared in the hook config, and the daemon derives its default-deny deadline from
that same number, so the time budget is set in one place `[LAW:single-enforcer]`.

**The dialog and the hook run concurrently.** Measured on 2.1.270: Claude Code shows
its permission dialog immediately and runs the `PermissionRequest` hook at the same
time, and whichever answers first decides. An answer typed at the dialog does not end
the hook; the hook runs to completion and its output is ignored. So the daemon is never
notified of a keyboard answer directly. Instead, it detects that the session has moved
on. Any of the following withdraws the waiting reply, which then prints nothing: the
requested tool finishing (`PostToolUse` or `PostToolUseFailure` with the same tool and
input, or, for a question, the same questions, because the question comes back with the
answers added), or the next `UserPromptSubmit`, `Stop`, `SessionEnd`, or
`PermissionRequest`. If the hook's connection closes first, the wait ends with no reply,
so a user who answers by voice later is told that the request no longer exists, not
that it went through. This is how a keyboard refusal is detected: answering No or Esc
at the dialog fires no post-tool hook and no `Stop`, but Claude Code kills the waiting
hook, and the closed connection releases the session (measured on 2.1.270). The
remaining case is a tool approved at the keyboard that is still running at the
deadline. Its warning and its deny, which Claude Code ignores, are still spoken, so the
expiry is described as what hands did ("so I told it no"), never as what happened to
the tool. A permission request from a session the daemon does not know about first
adds that session, so the request is asked aloud like any other.

## Transcripts: the live tail, and backfill

Claude Code appends to a session's JSONL file while a turn runs. It writes one record per
content block, once the block is finished. While this design was being written, the
transcript of the session writing it contained a record that was 21 seconds old in the
middle of a turn. For this reason, the daemon reads transcripts continuously, not only
when a turn stops.

**The recognisers** are implemented. `hands.core.steps.recognise` takes one `Call` (a tool
call, the record it was written in, and the result that was returned) and returns the
`Step` that the call represents. It uses a single table keyed by tool name
`[LAW:one-type-per-behavior]`. A recogniser matches a call only when the record contains
the data that its step requires, so no code needs a special case for failure: a failed
edit records no patch, so it becomes an `Other` instead of an `Edited`.
`hands.core.turn.Opening` is `Asked | Notified | Commanded | Shelled` and records what
started the turn. A question that Claude asked the user is a `Questioned` step.

**The tail** is implemented in `hands.sessions.tail`. As soon as the registry lists a
session, `keep_tailing` reads the new content of that session's JSONL file ten times a
second and passes each new record to the recognisers. The file is never re-read from the
beginning. One `Following` object per session stores the byte offset, the turn that is
currently open, the steps recognised so far, and how many of those steps have already been
narrated. It also keeps up to eight earlier turns that have ended, each with every prompt id
found in its records, because the narrator can be several seconds behind. A turn that has
ended is discarded once it has been narrated, or once a later turn has been narrated. A
prompt id that does not belong to any kept turn produces no narration; it is never used to
narrate a different turn. In a live measurement on 2026-09-21, a record became a step 96 to
305 ms after Claude Code wrote it, with a median of 160 ms. This delay is the poll period
plus the read time, and it determines how far behind the narration of a turn *while it
runs* can be. A `Stop` does not wait for the next poll: `tell` first reads the rest of its
own transcript, so the turn is narrated from everything that has been written.

`tell` receives the prompt id of the turn that ended and narrates that turn, regardless of
what has been read since `[LAW:no-ambient-temporal-coupling]`. Otherwise, a `Stop` followed
quickly by the next prompt, or an interrupt followed by the next prompt, would cause the new
turn to be narrated in place of the old one. `tell` returns a `Telling`, which contains the
turn's number as well as its steps that have not been narrated. It marks nothing as narrated
until the narrator confirms that the narration was spoken. A summary that fails is never
marked as narrated. When a slow summary is marked, the mark is applied to the turn the
summary was made from, which is found by its number, and never to a turn that started while
the model was generating the summary.

The user hears what a turn intends to do while it runs (see "Streaming" below). The result
of the turn is narrated at `Stop`.

**Backfill.** When the daemon attaches to a session that has already been running for an
hour, `read_session(session)` and `read_turn(session, turn, since)` read the same file
through the same recognisers. They return `Happening = Opening | Step`: what started each
turn as well as each step of the response, because the steps alone describe how a session
spent an hour but not what the work was for. A `Turn` keeps openings and steps separate,
because it is summarised as a whole against its request. A reading of a session that was
not narrated has no single whole to summarise, so it returns them together in the order in
which they occurred. Both are described by the same `describe` function, so a request cannot
be described one way in a turn and a different way in a reading `[LAW:one-source-of-truth]`.

The whole file is folded first, and only then cut at the specified record. As a result, a
call made before the cut and answered after it is still one step that includes its result.
If reading started from the mark, that result would arrive without the call it belongs to. A
call that has not yet returned a result is shown, because it indicates that the session is
working on something, but it is not marked as read. A mark identifies a record, and the next
reading continues after it. If a call that is still in progress were marked, its result
would be lost: the reader would hear that the test suite was being run and never hear what
failed `[LAW:no-silent-failure]`. This applies only to a call that a reading *ends* on, so
that a call the session has already moved past cannot hold the mark back indefinitely. The
mark is the last record whose happenings are *all* settled, because one record can contain
a text and the call it introduces. Marking that record because of its text would make the
next reading continue after the whole record and lose the call's result. A reading reports
two facts separately: `more` for history it did not reach, and `working` for a call that has
not returned. If they were reported as one value, the intermediary could not distinguish
"read further" from "wait and ask again" `[LAW:types-are-the-program]`. A mark that this
transcript never contained is reported as such, instead of being treated as a mark at the
start, which would narrate the whole session again as if it were new.

```python
# One variant per kind of action in a turn. Every step identifies the record it came from, except
# when there is no record to identify: a record that has no uuid, and the final reply that the Stop
# hook passes on before Claude Code has written it.
Step = Said | Edited | Ran | Tested | Looked | Planned | Delegated | Questioned | Other

@dataclass(frozen=True)
class Said:       ref: Ref | None; text: str                 # assistant text; never spoken verbatim
@dataclass(frozen=True)
class Edited:     ref: Ref | None; path: str; created: bool; change: str   # Edit, Write, MultiEdit, NotebookEdit
@dataclass(frozen=True)
class Ran:        ref: Ref | None; command: str; purpose: str | None; failed: bool; output: str; git: tuple[GitChange, ...]
@dataclass(frozen=True)
class Tested:     ref: Ref | None; runner: str; passed: int | None; failed: int; failing: tuple[str, ...]
@dataclass(frozen=True)
class Looked:     ref: Ref | None; tool: str; target: str; found: str   # Read, Grep, Glob, web tools
@dataclass(frozen=True)
class Planned:    ref: Ref | None; task: str; change: str    # TaskCreate, TaskUpdate
@dataclass(frozen=True)
class Delegated:  ref: Ref | None; agent: str | None; description: str; report: str | None
@dataclass(frozen=True)
class Questioned: ref: Ref | None; questions: tuple[Question, ...]   # AskUserQuestion
@dataclass(frozen=True)
class Other:      ref: Ref | None; tool: str; input: str; result: str; failed: bool  # named and summarised, never dropped

GitChange = Committed | Pushed | Branched | PullRequested   # what a command did to the repository

@dataclass(frozen=True)
class Asked:      ref: Ref | None; text: str    # the user's own prompt, typed or sent through the SDK
@dataclass(frozen=True)
class Notified:   ref: Ref | None; text: str    # a background task's report, delivered as the next prompt
@dataclass(frozen=True)
class Commanded:  ref: Ref | None; name: str; args: str; output: str | None   # a slash command the user ran
@dataclass(frozen=True)
class Shelled:    ref: Ref | None; command: str; output: str | None           # a `!` command the user ran

Opening = Asked | Notified | Commanded | Shelled   # what started the turn, so a turn is never narrated as a user request when it was not one
Happening = Opening | Step          # the contents of a session reading; an opening can be marked like a step

@dataclass(frozen=True)
class Changed:  path: str; added: int | None; removed: int | None   # counts are None for a file that git treats as binary
@dataclass(frozen=True)
class Commit:   sha: str; subject: str
@dataclass(frozen=True)
class Delta:    files: tuple[Changed, ...]; commits: tuple[Commit, ...]; patch: str

# The result is empty in three cases: there are no changes, there is no repository, or git could not
# be read. In all three cases nothing is spoken, because the user is told only what happened, not what
# did not happen. The log records which case occurred.
```

**What the records actually contain**, based on 900 transcripts read on 2026-09-21. These
findings showed that three lines of the design above were wrong:

- `toolUseResult` is *absent* from about one result in five. Every error writes a string
  there instead of an object, and so does every result that Claude Code's own harness
  handled, such as output too large to include inline. The content of the `tool_result`
  block is the only part that is always present, so a recogniser can prefer the structured
  record but must never require it.
- A failed command is marked by `is_error` on the block, not by the `Error: Exit code N`
  prefix that this document previously stated: one result in 15,098 began with `Error:`,
  while 448 had `is_error`. A failing test run also has `is_error`, which is why
  `Ran.failed` is the command's exit status and not a reason to stop recognising the run.
- `gitOperation` is a *set* of operations, not a single operation: of 850 sampled, 70
  commit and push together, and 30 open a pull request and push. For this reason there is
  no `Committed` step. A commit is a `Ran` that lists what it did to the repository, and
  one record remains one step.
- This version of Claude Code has no `TodoWrite` and no `Task` tool (zero calls in the
  sample). It plans with `TaskCreate` and `TaskUpdate` and delegates with `Agent`. An update
  records the task's id and its new status but does not repeat its subject, so `Planned` is
  one task and the change made to it.
- A newly created file records an empty hunk list and stores its text in `content`, because
  there was no earlier version to diff it against. `Edited.change` is therefore the hunks
  for an edit and the whole file for a create.

`Tested` is a `Ran` whose output was written by a test runner. Each runner is defined by
four patterns in one table: patterns that detect that the runner ran, count what passed and
what failed, and extract the name of a failure. The patterns are fitted to output captured
from pytest, vitest, cargo test and go test, for both passing and failing runs, and stored
in `tests/fixtures/testruns/`. `go test` counts only failures, so `Tested.passed` is None for
it, and a run's failures are counted from the failure names it printed. A run is counted
only from the summary lines that its patterns matched, and from nothing else in the output,
with the counts added across all of those lines: a workspace prints one summary per test
binary, and vitest counts its files on the line above the line that counts its tests. A
command is identified as a test run because its output matches a runner's output, never
because of how the command was written, so `make test` and a script use the same table.
`Tested` is used only when those counts describe everything the command did. A command that
failed while no test is counted as failing did more than run a test suite, and so did a
command that also made a commit. Both remain a `Ran` that contains the output describing
what else happened. `go test` prints its package line in the same format for a package that
failed to build, with the reason in place of the elapsed time, so a package line is counted
only when it includes the elapsed time. Questions that Claude asks in plain text instead of
through `AskUserQuestion` are not a step: they are extracted from the turn's final text when
it is narrated.

Record shapes to be aware of, observed in transcripts on 2026-09-14:

- `ai-title` → `aiTitle`. Claude Code's own title. It is often twenty-five words long and is
  not shown anywhere the user looks. Hands never speaks it.
- `custom-title` → `customTitle`. The session's name, set by `/rename` or by a hook's
  `sessionTitle`. Use it for `list_sessions` labels.
- `assistant` → `.message.content[]`, blocks typed `text` | `thinking` | `tool_use`.
  `text` and `tool_use` are both content. A Bash `tool_use` includes a `description` of
  its purpose.
- `user` → `.message.content` is a plain string for real user turns, or an array of
  `tool_result`; `isMeta: true` marks injected reminders. The record's top-level
  `toolUseResult` holds the structured result: `structuredPatch` for edits and writes;
  `stdout`, `stderr`, and `interrupted` for commands; `gitOperation` for a commit, a
  push, a branch change, or a pull request. It describes one call, and it can be matched to
  that call only when the record contains a single result.
- `permission-mode` → the session mode. It is written only when a prompt is sent; the hooks
  report it earlier.
- Every record: `uuid`, `parentUuid`, `timestamp`, `cwd`, `gitBranch`, `sessionId`.

**Prose is the smallest part of a turn.** In one session sampled for this design, there
were 24 assistant text blocks and 157 tool calls. The text totalled 126 thousand
characters and the tool results nearly 2 million, most of which were screenshots. Happy
narrated only the text, which is the smallest part of what happened (failure mode 2). In
this design, the recognisers cover the rest, and the length budget applies to what is
spoken, after summarising.

**Subagents.** In the parent's transcript, a subagent appears as one call and its report,
which becomes `Delegated`: either an `Agent` call, or a `Skill` call whose result says
`forked`, as the result of /code-review does. Both kinds of result contain the subagent's
`agentId`. The subagent's own records are in `<session>/subagents/agent-<id>.jsonl`, and
each of them has `isSidechain` set to true; the parent's transcript contains none of them.
When the subagent reports back, either in a task notification whose `<task-id>` is its
`agentId` and whose `<summary>` reads `Agent
"<job>" ...`, or in the result of a call that ran it in the foreground, its transcript is
folded by the same fold that backfill uses. It is narrated as a separate part of that turn,
named by the job the parent assigned to it: "the subagent's work on <job>" (failure mode
21). The first record of the subagent's transcript is that job, not work: a subagent's
prompt, or, for a fork, a copy of the parent's launching call. A notification whose summary
names no agent comes from a background command or a monitor. The notification is read with
the narration that responds to it, and is not read again in a later narration of the same
turn.

**Results outside the transcript.** A formatter, a code generator, or a `sed` in a
shell command changes files that no `Edited` step names. So at `UserPromptSubmit` the
reducer emits `Snapshot`, and the reader records the target's `HEAD` and a tree of every
file git would keep. At `Stop`, the reducer emits `Compare`, which diffs that tree against
a new tree taken at that point, and lists the commits that are reachable from where the
turn ended but not from where it began. A turn that is narrated a second time is instead
read from where its first reading ended (see "Git changes are reported by hands, not by the model").
The summariser receives the `Delta` alongside the steps, because the delta is the combined
result of all of them, and the file that a `sed` changed does not belong to any step.

A mark is taken only from a session that is at the prompt, because a turn starts
there and nowhere else. Claude Code sends the hook for a prompt that is queued while a turn
runs, and a `Stop` that the daemon did not receive leaves the session in the working state
in the registry. If a mark were taken again at either point, the turn would be compared
against a point in the middle of its own work, and everything it changed before that second
prompt would be missing from the narration of that turn
`[LAW:no-ambient-temporal-coupling]`. A mark whose `HEAD` could not be read is not a valid
mark. `rev-parse` returns nothing both for a repository with no commits yet and for a
repository it could not read, and empty output has more causes than timing alone can
explain: a `HEAD` read in the middle of a rewrite by a checkout in another terminal, a ref
that cannot be read, or a deadline that has already run out. For this reason, an unborn
`HEAD` is checked explicitly instead of inferred: if `symbolic-ref` succeeds, `HEAD` names a
branch that has no commits, which is the state of every repository between `git init` and
its first commit, and no other result counts. If a mark that is actually unreadable were
treated as unborn, it would be compared against no commit. Every commit ever made in that
repository would then be reachable from where the turn ended and not from where it began,
and the turn would be reported as having made all of them `[LAW:parse-dont-validate]`.

The tree is written through a separate index owned by the daemon: the repository's index is
copied to a scratch file, then `git add -A` and `git write-tree` are run. As a result,
nothing is staged, stashed, or reverted, and the repository's own index is never written.
`git stash create` would do most of this, and this design originally specified it. However,
a stash does not include untracked files, and a file that a code generator just wrote is
exactly what a turn must be able to report. `git stash create` also modifies the
repository's index while it runs. The index is copied instead of being built from empty
because it contains what git already knows about every file: 0.105 s compared with 1.409 s
on a repository of 20,000 files, and this step runs while a prompt's hook waits for it
`[LAW:carrying-cost]`. The only side effect is one unreferenced object for each piece of
content that git has not already stored, and git's own housekeeping removes these objects.
Git identifies objects by their content, so a file that does not change costs its size only
once, however many turns read it: measured at 1.6 MB the first time two hundred untracked
files were seen, and nothing on the two following readings.

`Compare` is a separate effect, emitted and performed before `Summarise`, instead of being
read when the summary is generated. Summaries are generated one at a time and take seconds,
and by then the session may have started another turn, whose changes would be reported as
part of the previous turn `[LAW:no-ambient-temporal-coupling]`. `Compare` starts the reading
and does not wait for any of it to finish. Both hooks are blocking hooks. The shim stops
waiting after `POST_TIMEOUT_SECONDS` and then prints that it cannot reach the daemon. The
effect queued after `Compare` is the one that causes the turn to be spoken at all, so a
reading that is slow, fails, or has its handler cancelled must only lose the turn's delta
and never its narration `[LAW:no-silent-failure]`. Measured: the stop hook waits 0 ms, and
the prompt hook waits 40 ms here and 145 ms on 20,000 files, compared with a total budget
for a mark of `POST_TIMEOUT_SECONDS - 0.5`. A mark and a reading are both best effort, and
neither may cost the turn what it was read for, so both run under one shared guard instead
of one guard each `[LAW:single-enforcer]`: a mark must not fail the prompt hook that waits
for it, and a reading must not cost the turn the `Summarise` queued after it.

A mark is held from the moment its snapshot starts, so a reading that needs a mark that is
still being taken waits for it. If the mark could not be taken at all, the turn is narrated
without a delta. There is one reading per narration, in the order in which the narrations
are made.

Everything from a delta that is shown to the summariser is limited by the `Budget`,
including commits: a turn that pulls or rebases can bring in hundreds of commits, and in
that case the count matters more than the subjects. The reader keeps no more than
`MOST_COMMITS` commits for the same reason it keeps no more than `MOST` characters of patch.
It requests no patch at all above `MOST_LINES`, because git returns the whole diff before
any of it can be truncated, and a turn that generated a million-line file inside the
repository would otherwise load all of it into the daemon at once. The numstat counts used
for this decision cost one line per file and are already available. When the patch is
refused, the remaining data (the files and their counts) is everything from a diff of that
size that would have fit in the budget anyway. A numstat that could not be read is not the
same as a numstat that reports nothing: if it were treated as all counts being zero, the
limit would not apply to the one diff whose size was the reason for checking
`[LAW:no-silent-failure]`.

A reading is kept for every turn that stopped, and when there are more than `HELD`
readings, the *newest* one is dropped. Readings are limited, but the `Summarise` effects
they are paired with are not, so the two are kept aligned by position only. Dropping the
oldest reading would give every later narration the delta of the next turn instead of its
own, which is exactly what this pairing is designed to prevent. When readings are dropped
from the end, the turns beyond the limit are narrated from their steps only, and every turn
that receives a delta receives its own.

Within a reading, the commits are read before the tree, because reading the commits takes
two fast commands while `git add -A` is slow. If a commit is the most important thing about
a turn, it must not be lost because the tree step exceeded the reading's deadline.

One reading is made for every turn that stops. Readings are kept in the order the turns
stopped, and every narration takes exactly one reading, including a narration whose summary
failed. This is the only mechanism that keeps readings and narrations aligned. If a reading
were held back for a turn that could not be summarised, the next turn's narration would take
it. A user can act on changes they did not hear about, but can do nothing about changes
attributed to the wrong turn. The snapshot races the agent's first edit, with a margin equal
to the time the model takes to start, which is several seconds. An edit that happens before
the snapshot is still recorded as an `Edited` step.

## Summaries: spoken form, and the narration tree

hands does not read any text aloud verbatim. Claude writes for a screen, and markdown, code, tables,
paths, and hashes cannot be understood when read aloud as written. Every word sent to the speaker is
therefore summarised or transformed first.

**What runs today** is the top level of the tree. Each finished turn is passed to the intermediary
together with the changes git reports for the turn and, if the turn ended on a question, that
question, as described below. The sections are opened when the user asks for more detail (`expand`,
below). When a `Stop` arrives for a live session, the reducer emits `Summarise(session, closing)`,
and `narrate` in `hands.voice.narrator` requests from the tail the parts of that session that have
not yet been reported. A turn starts at the last user record that meets all of these conditions: it
is not `isMeta`, it is not `isCompactSummary`, its content is a string or a block list that contains
no tool result, and it does not follow a tool call or tool result. A message sent while a tool is
running belongs to the current turn. An image or document attached to a prompt is identified by name,
not read. The opening is `Commanded` if the record starts with Claude Code's slash-command markup
(`<command-name>` or `<command-message>`), `Shelled` if it starts with `<bash-input>`, `Notified` if
its `origin.kind` is `task-notification`, and `Asked` otherwise. A command's output
(`<local-command-stdout>`, `<bash-stdout>`, and their stderr counterparts, marked as stderr) is a
separate record that follows the command's record. It does not start a turn. It is attached to the
opening whose record it references as its `parentUuid`, and terminal escape sequences are removed
(`hands.core.turn.printed`). A command and its output are written as a `system` record of subtype
`local_command` about as often as they are written as a user record, and both forms are handled the
same way. A `system` record of this kind has no prompt id and is not a response from Claude. A command
that is recorded as the typed text, such as `/compact` before its compaction runs or a skill that runs
in its own fork, is also `Commanded`. It is identified as a slash followed by a name, in a record with
no `promptSource`. Claude Code writes `promptSource` on every prompt it sends to Claude, including a
prompt that starts with a slash. The record that `/compact` writes after it has run, under the same
prompt id, is treated as the same command (`hands.core.turn.recorded`), not as a second turn. The
steps of a turn are assistant text (`Said`) and tool calls, each call matched to its result by id,
and each step is passed to `recognise`. Thinking blocks are skipped, because they show how Claude
reached a result, not the result itself. A subagent's records are stored in a separate file and are
folded only from that file.

**A turn can stop twice.** Another hook can block a `Stop`, and the turn then continues until a later
`Stop`. For this reason, the tail stores two values per session: the number of the open turn's steps
that have already been reported, and any closing reply that was reported before its record existed.
Each report includes only the steps after those, so the first half of a long turn is not read aloud
twice. Both values are cleared when a new turn starts, and all stored state for a session is cleared
when the registry stops listing the session. This was verified live on 2026-09-21 with a second `Stop`
hook that blocks once: the first report described what the turn had done, and the second reported
only `echo second` and nothing before it. The time from `Stop` to the spoken summary was 1.41 s and
1.21 s.

The steps are not the only thing a second report must handle correctly. The opening belongs to the
whole turn, so it is sent to the summariser again. A small model that receives the same request twice
responds to it twice. This happened on 2026-09-21: the second summary restated the first half of the
turn, based on an opening that named all three original actions, although the steps it received
contained none of them. For this reason, the turn itself records which report this is, instead of the
report type being inferred separately. `Turn.standing` is `Answering` or `Continuing(told)`, and
`render` matches on it. For a continuing report, the opening is presented as context, together with
the number of steps already reported and an instruction to report only the steps that follow
`[LAW:types-are-the-program]`. A separate flag on the turn would have done the same job, but the type
checker would not have checked that both interpretations of an opening are handled. An eval case
covers the same real turn reported both ways.

The stand-in reply exists because the hook and the transcript are briefly out of sync. In measurements
over twelve live turns, the reply included in `Stop` was never yet in the transcript when the hook
fired, and its record was written 46 to 77 ms later. The hook's copy of the reply is therefore used as
the turn's last step while the record is missing. When the record arrives, it replaces the copy, and
the reply is never reported twice. This works because Claude Code only appends to the transcript, so
that record is always the first of the steps not yet reported `[LAW:one-source-of-truth]`. *Not*
covered: a tool result written after a turn was reported is never reported, because its call was
already reported as having no result. In all twelve turns, every call was paired with its result at
`Stop`, so this case cannot be handled at a `Stop`. Progress reporting does not cover it either:
progress reports what a call is about to do, never what it returned.

The narrator passes a finished turn to the intermediary as a separate turn, and the intermediary
describes it in its own words. As a result, what the user heard is in the intermediary's history, and
the intermediary can answer questions about it. A separate summariser, called with `/btw`, left this
information out of the brain's history: on 2026-09-30, hands told Brandon that a session had asked
about PR 68 and 69, and when he asked what those were, the brain replied that no session showed them.
The intermediary receives the turn's last message first. This is the session's own account, written by
the session that knows what PR 68 is. It is limited to `REPLY_SHOWN` characters, because every
reported turn adds to the brain's history and moves it closer to compaction. Next, it receives the
information hands read from the turn that the last message may not include, taken from the narration
tree. Last, it receives the question the turn is waiting on, and the intermediary is instructed to end
with that question. If the brain cannot accept the turn, hands speaks a fixed message saying that it
could not report the turn, and nothing else. At `SessionStart`, the plugin's shim instructs each
working session to end every turn with a concise overview that can be read aloud. A turn that the user
stopped before it did anything is passed on like any other turn, and its record notes that the user
interrupted it. Each summary and its delivery method are recorded as facts on the turn's utterance.
If a transcript cannot be read, hands speaks "cc-hands finished a turn, and I could not read it."
without using the model and without adding it to the context, and logs a `Failure` line.

For the brain, a narration waits in a dedicated `BrainStage` lane, never in Pipecat's context, and the
user's turn has priority over it. A narration that waited while the brain was answering is sent only
when no words from the user are waiting. Fixed text that the narrator speaks as written waits in the
same lane (`Aloud`), so the end of a session is announced after its last turn. Each turn emits one
`voice.turn` wide event, and its `queued_ms` records how long the turn waited in its lane.
Its `waited_ms` records how long the user waited from releasing the key to the turn's first word on the
wire. `transcribed_ms` and `queued_ms` show how that wait was spent before the input was written to the
brain. When several sets of words waited together behind a turn, they are timed from the last of them.
Each tool call made by the brain's replies (`tool.call`) is a child event of the turn, timed by when the
stage saw it on the wire. Each model round trip is recorded by the proxy's own `Exchanged` line, which
carries a span inside the turn's span (`span`) and reaches the collector as a `proxy.exchange` span
under the turn. No second record of a round trip is kept. The brain's other requests during a turn,
such as a subagent's or a fork's request, and a request that hands held, also belong to the turn. A
side question's own request belongs to its `brain.aside`. The router assigns each exchange's span when
it routes the exchange. An exchange that belongs to no unit of work, such as a wrapped session's
exchange through the tap or a brain exchange outside a turn, is recorded as a `proxy.exchange` span at
the root of its own trace, with its session, kind, and path. A turn that hands stopped partway through
still emits its event, marked as cancelled, with the work it had completed.
The event also records the text typed into the brain (`asked`). If the user barged in, it also records
the calls that were running at that point (`running`) and whether the brain was instructed to stop
immediately (`stopped`).

If a user's turn is still running tool calls `ACKNOWLEDGE_SECONDS` (2 s) after its first call, hands
acknowledges it, regardless of how many calls the work requires. hands does not acknowledge the turn if
the user has already heard something from it: a word, a permission question, or their own barge-in.
The acknowledgement is a short fixed phrase ("One moment.", "On it.", ...), chosen in rotation from
`ACKNOWLEDGEMENTS` so that two consecutive turns do not get the same phrase. It is sent to the turn's
speaker as a separate sentence, in the same way as a permission question, and is not added to the
context, because the brain did not say it. hands never acknowledges a call that returns immediately, a
turn that states what it will do before it makes a call, or a turn that hands narrates. The turn's
event records the phrase (`acknowledged`) and how long after the turn left its lane the phrase was sent
to the speaker (`acknowledged_ms`). These are recorded in the same way as the turn's own words: a
phrase that the user spoke over before it was played is not recorded.

The brain's own process emits four kinds of event. The launch event (`brain.launch`) covers the time
from the spawn until the brain's input is ready. It records the account, model, and config directory;
the fritter used (the one included in hands' package); and the conversation (`conversation`:
`resumed`, or `fresh`, with the previously held conversation that Claude Code has no transcript of
recorded as `untranscribed`). If the launch failed, it records the conversation that was released
(`let_go`), or the error that caused the failure. A hands build without its fritter is rejected with
the same message that `hands install-fritter` uses to reject it. The run event (`brain.run`) is a child
of the launch event. It covers the time from then until the process ends, and records the exit code and
the last output the process displayed. Each turn typed into the brain emits a `brain.turn` event, which
is a child of the `voice.turn` that sent it. It covers the time from typing to the end of the turn. It
records the prompt ids that Claude Code assigned to it (`prompts`), the number of times it was typed
(`typings`), the tools offered to the model in its latest request (`offered`), and the prompt ids of
other prompts that Claude Code accepted while the turn waited to be accepted, or that the turn's typing
or stop interrupted (`others`). If the turn failed, it records the error instead. A turn counts as
accepted only when a `UserPromptSubmit` arrives whose `prompt` matches the typed text as Claude Code
stores it: with the tags that wrap a long paste removed and trailing whitespace trimmed. If a turn
ended without being accepted and was accepted later, that acceptance is never attributed to the next
turn, unless both turns typed the same words. A prompt that Claude Code accepts but that belongs to no
turn, such as one for a request that already failed or one that nobody made, runs before the text typed
after it. A turn therefore stops such a prompt with Escape and Ctrl-C: once before the turn is typed,
and again, after the turn is typed a second time with its full time allowance, if such a prompt ran
during its wait. A prompt accepted late cannot prevent later turns from being accepted, however long it
would run. A turn typed twice is accepted by either typing and ends at the Stop of either one, because
the Escape may have stopped the first typing as it was accepted, and a stopped prompt does not post a
Stop. A turn continues to be processed until it ends, even if the caller that sent it stopped waiting.
Each dialog the brain posts to hands' listener emits one event: a `brain.permission` event, which
records the tool, the decision, and how long the dialog was held, or a `brain.elicitation` event;
elicitations are always declined. A dialog posted during the active turn is a child of that
`brain.turn`; any other dialog is a child of the launch. If hands cannot read a dialog's body, it
responds as the hook requests, and the event is marked as failed with that reason. Any other failure in
the brain's own work, such as typing a turn, handling a dialog, or processing a hook, is printed once to
the terminal with its traceback. The turn that the work was for, if any, is stopped in the same way an
Escape stops it, and ends with that error. A permission request whose handling failed is refused. A
hook is answered before it is processed, so a hook whose processing failed is not attributed to any
turn, and later hooks are still processed.
Each side question is sent to a separate Claude Code instance and emits one `brain.aside` event,
starting when the question is asked. The event records the purpose (`kind`), the Claude Code session
(`aside_session`), the question and its answer, and how long the question waited behind earlier
questions (`queued_ms`, absent for a question that was never processed; the rest of the event's
duration is the time the Claude Code instance spent answering). If there is no answer, it records the
reason (`unanswered`). One possible reason is that the caller's time ran out: a `Deadline` measured
from when the question was asked, or, for the line that an old result is delivered as, a `TimeLimit`
measured from its turn. In that case the event also records what the Claude Code instance displayed at
that point (`shown`). The side question's Claude Code instance is always shut down gracefully, after
the caller's time runs out if necessary, because it shares the brain's config directory. An explanation
is a child of the `utterance` it delays, a summary is a child of the `summary.backlog` or
`summary.turns` pass that requested it, and a name is a child of its `name.judged`. A line, which is
requested after its result's turn has ended, starts its own trace.

**What is in front.** Built: when the user's words reach the brain's stage, hands reads once what is
in front on the Mac's screen (`sessions/front.py`, with the decision made by the pure
`core/front.py`). The brain receives this after the user's words: the frontmost app and the session
shown in its front tab, or a note that the tab shows no session. The screen is not monitored between
turns, and narrations do not use it. The frontmost app is the one LaunchServices reports
(`lsappinfo front`), whether or not one of its windows is on screen. An app that reports through
AppleScript which terminal its front tab shows (iTerm2, Terminal) is queried. For any other app, every
terminal under it is considered, and this identifies the session when one of them contains it. For a
tab running tmux, the visible pane is the one its client is attached to. A session is in front when
that terminal is among its ancestors: the ancestors of its fritter, its tab, or its pane. If hands
cannot read the screen (no frontmost app, an AppleScript request that is refused or not answered, or
several sessions under the terminals shown), the screen information is left out of the turn. The
`voice.turn` event's `asker` fact, a `UserAsked`, records what was read or why it was not, and how
long the read took.

**Screen or audio-only.** Built: the user's turn also tells the brain whether the user can see a
screen (`core/place.py`'s `Modality`). This value is read when the user's words reach the stage and is
recorded on `UserAsked`. `PushToTalk` owns it together with the place: moving to the desk sets it to
`screen`, moving to the phone sets it to `audio-only`, and the brain's `set_modality` tool changes it
until the next move. The brain uses it as a hint when making choices; it never restricts what hands
does.

The rest of this section describes planned work: step summaries built as steps arrive, and streaming.

**Spoken form.** Built: `core/spoken.py` is a pure function that converts text into text that can be
read aloud. It is installed as the TTS service's only text filter `[LAW:single-enforcer]`. Pipecat
applies a TTS service's filters both to the text of a `TTSSpeakFrame` and to each aggregated sentence
of a model's streamed reply. As a result, summaries, announcements, system speech, and the
intermediary's own words all pass through the filter, and text that has not passed through it cannot
reach the speaker. This is why the filter is placed here instead of at each place a frame is built:
there are five such places, and the intermediary's own reply is not one of them, so four enforcers
would still have left the largest source unfiltered. Instructing a model to produce spoken form does
not solve the problem either. That approach is a rule expressed as an instruction, which the model may
or may not follow and which nothing checks. The model had already been observed reading a bare file
name aloud, and session titles never passed through it at all.

Headings are converted to section cues, and lists are converted to counted sequences. Identifiers are
split into words, so `authMiddleware` is read as "auth middleware". A path is replaced by its file
name, with its directory included only when two files mentioned together share the same name. Flags are
replaced by their names. Hashes, ids, and URLs are replaced by a description of what they refer to, or
removed. Code blocks, diffs, and tables never reach the filter as text, because they are summarised
first. If one does reach it, it is replaced by its kind and length, and the leak is logged.

Two principles shape the rules. Each rule requires a marker that ordinary English does not contain: a
diff must start with `@@` or `diff --git`, a table must have two rows rather than one line containing a
pipe, a hash must contain both a digit and a hex letter, a path must start at the root or end in an
extension from a closed list, and an id must be mixed case as well as long. This is because a rule that
corrupts a sentence costs more than the code name it fixes `[LAW:carrying-cost]`. A rule without such
a marker does more than fail to help. It removes a word from the middle of a sentence and leaves the
sentence grammatical, so nothing downstream can detect the error. Observed errors include "and/or"
read as "or", "24/7" as "7", a file of 1048576 bytes as "a commit bytes", and `base64_encode` as "an
id". For this reason, the markers are enforced by a table of ordinary sentences in
`tests/test_spoken.py` that must be returned unchanged, not by this paragraph, because a rule stated
only in documentation is not kept up to date.

In addition, the last step unconditionally removes every remaining markup character. This includes a
line that contains only markup characters, which is how a horizontal rule and the dashes under a
heading are written. This makes "no backtick, no pipe table, no fence reaches the speaker" a property
of the function rather than an assumption about the earlier rules `[LAW:parse-dont-validate]`. A fence
is parsed, not recognised by its first three characters. Its closing run must use the same character
and be at least as long as the opening run, because a model quotes a three-backtick block by wrapping
it in a four-backtick block. A closing check that ignored length ended the outer block at the inner
opening fence, and the quoted code was read aloud.

On the streaming path, the filter receives a reply in pieces. A list that the intermediary streams is
therefore processed one item at a time and is not read as a counted sequence, whereas the same list
inside a summary is. A fenced block cannot be split this way, because its continuation arrives without
a fence and would be read aloud as the ordinary text it resembles. For this reason, the reply is never
split inside a fenced block. Pipecat's `LLMTextProcessor` sits between the model and the TTS service
and uses `FenceAggregator` (`voice/spoken.py`). `FenceAggregator` splits the stream into sentences and
buffers a block in full from its opening fence to its closing fence. It detects an open fence using the
same rules as `spoken`. Pipecat flushes the aggregator when a reply ends and resets it on a barge-in.
A reply that ends inside a fence is therefore read as that block, and nothing carries over to the next
reply. Tracking the open fence in the filter instead was tried and reverted. Pipecat does not notify a
text filter when a reply ends, so after a reply ended inside a fence, every later utterance was replaced
by a block announcement.

The filtered text is also what the intermediary stores in its context, because Pipecat builds the frame
it appends to the assistant context from the filter's output. This is intentional. The context exists
so that the model can answer questions about what the user heard. Before this filter existed, the
context held the text sent to the speaker, which was never the same string. The trade-off is that the
model cannot retrieve an exact path or sha from its own context; it uses the session tools to look up
facts. A draft readback passes through the same point but should not be changed the way a summary is,
because it is read aloud so the user can check what will be sent. This is tracked separately, rather
than solved by adding a mode to the function.

The function is pure and uses only the standard library, because it is domain logic (what a developer
who is not looking at the screen can hear), and it therefore cannot log. It returns a leak instead of
logging it, and the voice edge logs the leak, because logging is an effect and `core` is not the edge
`[LAW:effects-at-boundaries]`. The filter is stateless, so a barge-in in the middle of a sentence
leaves no state to reset.

**The narration tree.** `core/narration.py` divides a finished turn into segments: whether the user
stopped it, what the repository did, the question the turn is waiting on, what the user already
answered, and one section per topic (the change, the tests, the commit, the commands, what it read, the
plan, the subagents, the other tools, and what it said). The topics are not defined by a separate table
of rules. They are a match over the `Step` union, which already makes exactly those distinctions, so a
new kind of step causes a compile error here instead of producing a result with no section
`[LAW:types-are-the-program]`. Every step is assigned to a section, so "more on that" can reach
anything the turn did. A segment contains the happenings it was created from, so `opened` renders them
through the same `body` used for the whole turn. A segment's `refs` are derived from those happenings
rather than stored separately: the record ids a segment references are the records it contains, so the
two cannot become inconsistent `[LAW:one-source-of-truth]`.

**More on that.** Each report of a turn is stored in `Recounts`, with the tree's parts stored next to
the text sent to the intermediary. That text includes the session's id, so `expand(session, part?)`
does not require a listing call first. When called without a part, it returns one line per part, such
as "The tests: one test run.", no matter how many times it is called, because expanding every part at
once would be more than one tool result can hold. When called for one part, it moves one level deeper
in a sequence of budgets (`hands.core.drilldown`); each level is rendered by `opened` with a larger
budget. The depth is the number of times that part was requested. The daemon counts it; the model does
not need to remember it. A Stop that finds nothing new keeps the current depth; a new report of the
turn resets it. Whether more detail is available is determined by checking whether the next level
renders any part differently, so a part that is complete at a short level is reported as complete. The
deepest level is still a rendering cut to its budget, which the intermediary rephrases in its own
words. There is no verbatim level, so requesting more detail never causes code or output to be read
aloud. A test run keeps the lines in which its runner explains a failure (`Runner.why`: pytest's `E`
lines, a Go test's `file.go:N:` lines, a Rust panic and its message, a vitest `FAIL` and its error),
so expanding "the tests" reports what failed and why. pytest's own summary line cannot be used for
this, because outside a terminal it truncates the reason to `- Asser...`.

No part of the tree is text generated by a model. It is computed from typed steps. This is a design
requirement, not a cost saving: on 2026-09-21, a summariser was observed reporting "version two point
seven point one" for a runner that printed 8.4.1, and a computed count cannot be fabricated.

**Git changes are reported by hands, not by the model.** Whether a turn made a commit is recorded in
two places: by the step, when Claude Code writes a `gitOperation`, and by the delta, which is read
against the repository's state at the start of the turn. Each source detects cases the other misses: a
`git commit` inside a heredoc produces no operation for a step to record, but the delta still reports
it. The narration therefore reports the commit from whichever source detected it, in wording that
contains no hash: "It committed and left five files different." When a summariser was asked to report
this instead, it was observed both omitting the commit entirely and reading its hash aloud, in the same
afternoon. This sentence is passed to the intermediary together with the reply, which may say the same
thing. This redundancy is intentional: omitting the git sentence whenever the reply claims a commit
would suppress it in exactly the case it exists for, a claimed commit that was never made
`[LAW:no-silent-failure]`. A branch name is converted by `spoken_ref` instead of being copied as is.
This is the only part of the top level that no model wrote, so the spoken-form instruction does not
apply to it, and the filter before the speaker deliberately does not treat a bare
`feature/narration-tree` as a path. A rule broad enough to match it would also match "and/or" and
"24/7" and remove a word from each. A ref is therefore converted at the point where its type already
identifies it as a ref. Its separators are replaced by spaces and every word is kept, because a branch
is named so that it can be distinguished from other branches `[LAW:single-enforcer]`.

A push, a branch, or a pull request changes no file and adds no local commit. Claude Code also writes
no `gitOperation` for one inside a heredoc, a script, or a compound command that it does not parse;
this applies to two in five commands that push, and to nearly every `checkout -b`. The delta therefore
detects each of these from the traces it does leave, and only for the branch that the session's
worktree is on, because every worktree and the terminal next to it share the repository's refs. The
traces are: the branch's own log showing that it was created from something other than the remote
branch of the same name (a checkout or a rename does not count as creating a branch); a remote-tracking
ref for the branch whose log shows `update by push` since the mark (a fetch moves the same ref and
logs `fetch`); and, for a pushed branch that is not the branch a remote's HEAD follows, a pull request
that the forge reports was opened since the mark. The forge is queried in parallel with building the
tree and must answer `SPARE` before the narrator's `PATIENCE` runs out, so a slow forge can cause the
pull request to be omitted, but never the commit. Both sources report changes as the same `GitChange`
values, so a push that both detect is reported once. Each mark emits one `delta.mark` wide event and
each reading emits one `delta.read`. Each event records which index its snapshot started from: a copy
of the repository's own index, an empty index, or none, because git did not report where the index is.
Each event also counts the git commands that did not respond, so a git timeout can be distinguished
from a repository with no changes to report.

For a turn that is reported twice, git is read twice, and each reading is compared against where the
previous reading ended. No prompt marks where the part after a blocked `Stop` started, so every reading
also stores the repository state it found, and `Compare(again=True)` compares against that state
instead of the prompt's mark. A commit made in the second half of the turn, such as a heredoc commit,
which no step records either, is reported with the second report, and nothing from the first report is
repeated. If the first reading could not determine the repository state, the second report has no
delta. Only a turn that hands already reported is read this way. A turn whose first `Stop` hands did
not receive is read against its prompt's mark, or not read at all if no prompt was received, because
the last reading in that case belongs to another turn.

**A question from the turn is always played once, and the daemon determines whether the turn asked
one.** A turn that ends on a question is waiting for the user whether or not a hook blocks, so
`open_questions` reads the question from the turn itself. It checks two things: an unanswered
`AskUserQuestion` followed by nothing except an interruption, and any question in the closing text.
The closing text is the turn's last step, because a question the turn continued past was either
answered or did not need an answer. When Claude continued past a dialog, the dialog had been declined with a
message or refused by a hook; this was the case for 5 of the 94 unanswered dialogs in this machine's
transcripts. The other 89 were dismissed with Escape and ended the turn on the question. `asked_in` is
the only function that detects questions in text, and the narration uses it on Claude's closing text
`[LAW:one-source-of-truth]`. A question addressed directly to the user counts wherever it appears (for
example, "want me to do it?" followed by two more sections). At the end of the text, an offer counts
("Say the word and I'll do it."), and so do a choice and any other question, unless the same list item
or block of prose answers it ("Why did it fail? The cache was stale.") and nothing later in the
paragraph refers to an answer still to come ("Once I have that I'll pin the interface."). A `?` inside
a code block, a code span, a quotation, or an italic aside is treated as mentioned, not asked. A `?`
immediately followed by a word is treated as part of an address or a name. A `?` followed by an arrow
is treated as answered on the same line. These patterns were derived from 3,038 closing texts on this
machine, and each pattern that decides a case and fits in a fixture has one. The only real closing text
that asks a question and answers it on the same line is from a 794 KB turn.

The question the turn is waiting on is a single question segment, placed last. It uses Claude's own
words, introduced with "It is asking:" or "It said:", with "(Recommended)" removed, and is passed
through `spoken`. It is passed to the intermediary, which ends with it. An `AskUserQuestion` that the
turn is no longer waiting on, because it was answered or the turn continued past it, is a separate
segment in `settled`. It can be opened but is never played. `tests/test_questions.py` requires
`open_questions` to produce no missed questions and no false positives on real turns copied in full
from real transcripts, stored under `tests/fixtures/turns`.

**Streaming.** Built for tool calls (`hands.core.progress`). Each call that the tail reads into a
running turn is described by what it is about to do, based on its input and never on its result. A
command is described by the description Claude Code asks for ("run the test suite"), an edit or a read
by its file name, and a search by its pattern only when the pattern is words. Code is never used: not a
command, a regular expression, or a glob. `AskUserQuestion` and `ExitPlanMode` are not reported as
progress, because their hook announces them when they are asked. The tail passes the calls on as a
`Progressed` event. It passes on no calls from a turn that it read from the start of a file, because
that turn started before hands began following the session. The reducer collects the calls on the
`Opened` turn. When a turn ends, the collected calls are discarded without being spoken, and the turn's
result is reported instead. The reducer's tick releases a burst as one `Progress` when no call has
arrived for `SETTLE` seconds, or when the first call in the burst has waited `LONGEST`
`[LAW:no-ambient-temporal-coupling]`. The relay routes it using `attention.progress_route`, a table
keyed on the focus and the overlay, and records each routing choice and its reason on the burst's
utterance. Progress from the focused session is routed as `Working` and played as written at `fyi`
("cc-hands: edit ten files, then run the test suite."). Progress from any other session, and from a
muted session even when it is focused, is left to the session listing, which reports what a working
session last started to do. The session listing is at the end of every request the brain makes.
`coalesce` combines a session's progress into one report and drops progress that a result from the
same turn describes better. Progress carries its turn's ids for this purpose, because a result reaches
the floor only after it is summarised, by which time the next turn's calls may already have reached it.

Text is collected the same way, from `MessageDisplay`. Each batch of lines that Claude Code displays is
a `Displayed` event in the turn named by its `prompt_id`, and is added to the burst on the `Opened`
turn. A line keeps the burst open in the same way a call does, and a long explanation is released at
`LONGEST`, while Claude is still writing it. Lines displayed after the turn's `Stop` have no effect.
Progress that the relay routes for playback uses a dedicated lane (`hands.voice.working`), because its
text waits for a summary and a permission request must not wait behind that. The summariser's model
rewrites the text as an imperative phrase, which is placed before the burst's calls ("cc-hands: explain
how DNS resolution works, then run the test suite."). Text that cannot be summarised is spoken as "write
something", and the text itself is never read out. If a turn ended while its progress was being summarised, the progress
is not played, and the turn's result is reported instead. The registry reports this when the summary
is ready. The burst's utterance records the phrase, or the summarisation failure; a failure marks the
utterance as failed, and the burst is still spoken.

A subagent's progress is reported the same way while it works, from its own transcript. The tail
detects a subagent by the `agent-<id>.meta.json` file that Claude Code writes next to the subagent's
transcript when it starts the subagent. For a subagent started before hands began following its
session, the tail reads from where the transcript ended at that time. Its calls are reported as a
`Progressed` event whose `of` is the subagent: an `AgentTask` named by the task that the starting call
assigned to it. The name is the meta file's `description`, or, for a skill that runs in its own
subagent and has no description, the single `Skill` call its parent is running ("/code-review high
152", read aloud as "code review high 152"). A subagent started by another subagent is stored in the
same folder, and its meta file names that subagent as `parentAgentId`. Its progress is reported as part
of the task that the session's own call assigned to the first subagent. A subagent that nothing names
is logged once and is never reported as part of another subagent's work. The record at the start of
its transcript is its task, not its work, as is also the case for its report. The reducer collects a
subagent's calls on the `Session`, not on the turn, because a subagent running in the background
continues working after the turn that started it has ended ("cc-hands, its subagent to review the
parser change: read tail.py, then run the test suite."). `coalesce` combines a subagent's progress
only with that subagent's own progress, and drops it wherever a report that covers that subagent is
present (`News.reported`). The subagent's work is reported with the turn it reports back to, and a
result that does not cover the subagent leaves its progress as news.

## Playback: bookmarks and resume

The user often interrupts a reading, for example to ask which file is meant or to answer
something else, and every interrupted reading must be resumable. For this reason, the
playback position is stored in the daemon, not in the model's memory (failure modes 8 and
25). The player holds a `Playback`, which contains the segment that is currently playing
and a stack of bookmarks that record where earlier readings were interrupted.

An interruption pushes a bookmark for the segment that was playing. "Go back to what you
were talking about" maps to `resume()`, which pops the bookmark and replays that segment
from the beginning. "Skip that" and "say that again" map to `skip()` and `repeat()`.
"That part" refers to the segment that is playing, or to the last segment played, and
"more on that" maps to `expand()` on the turn that the segment belongs to.

Pipecat's output transport reports text as its audio plays. On an interruption, only the
text that was played is added to the context. pocket-tts does not report word timings, so
the smallest unit of position is a sentence. In the current implementation
(`hands.core.playback`, `hands.voice.player`), a reading is the sequence of sentences sent
to the speaker since the speaker was last silent. The sentences can be the model's reply
or lines that hands speaks verbatim. An observer builds the reading from frames that two
processors push: the TTS service pushes each sentence as it generates it, and the output
transport pushes each sentence's `TTSTextFrame` and each barge-in as it releases them. A
processor placed in the pipeline instead captures a barge-in that arrives before the end
of a sentence that is still in its queue. "Go back" skips the reading that the user just
interrupted, because that reading is the side answer the user is leaving, and returns to
the reading that was interrupted before it. In a live measurement on 2026-10-03 with
pocket-tts output to BlackHole, all three sentences of a reply were generated by 2.6 s,
the first sentence's text frame left the transport at 4.2 s when its audio ended, and a
barge-in one second into the second sentence created a bookmark at the second sentence.
`resume`, `skip`, and `repeat` are tools whose call is the entire reply. hands speaks the
original sentences verbatim, so the model never restates them from memory. These lines are
inserted immediately before the TTS service, not through the model's stage, so a barge-in
drops them like any other sentence that has not played yet. Their reading contains all of
them from the start, so going back does not lose any of them.

A staged or amended draft is read back the same way: hands speaks it verbatim. The tool
returns `{"says": ...}`, and the brain's stage speaks that text after the brain's own words
finish, so a barge-in before that point does not lose it. The tool calls form the entire
reply only when every call in the reply is silent. A call that is refused (`error`) or a
call that requests a reply returns control to the model.

## Push pointers, pull content

The intermediary's context window holds the conversation with the user, not session
transcripts. Hook events and played segments are injected as small frames that contain the
session title, the event kind, the spoken text, and the segment id. Steps, records, and
unplayed segments stay in the daemon, and the model retrieves them with `expand`,
`read_session`, and `recall`. When the daemon starts or reconnects, the intermediary
receives one note that lists the live sessions by title, state, and focus. The note never
includes session history. It is modeled on the session directory that Happy sends at
connect. The brain does not receive this note. Instead, each request that the brain's turn
makes includes the same listing, built when the request is sent, as the final text block
after the block that contains Claude Code's cache marker (`hands.voice.briefing.tail`). The
cached prefix is exactly what Claude Code sent, and the history never contains an outdated
listing. The brain's history is also kept small by rewriting outgoing requests
(`hands.brain.context`). A long tool result that is more than K turns old is sent as
`<tool>: <sentence>`. These replacements are applied in a batch every K turns, so the
cached prefix changes once per batch. The sentence is requested when the result's turn
ends, as a side question that shows the result to a separate Claude Code instance started
only for that question (`hands.brain.asides`). The sentence is kept in the summary store,
keyed by the result's content. Each batch produces one `context.stubbing` event. The event
lists, with counts, the calls that are sent as a single line from then on (`stubbed`) and
the calls that are sent in full because no sentence existed for them yet (`whole`). The
brain's compaction is instructed to keep what a voice session needs rather than what a
coding session needs. A restart does not reset the conversation. Each brain stores its
session in `conversation` in its config directory, and the next brain resumes that session
as long as Claude Code still has its transcript. As a result, a restart, an upgrade, or a
model change continues the existing conversation, and the same compaction limits its
length. If a start fails, the stored conversation is discarded, so the next start begins a
new conversation instead of failing in the same way. The brain runs with
`--system-prompt-snapshot off`, so a resumed conversation receives the instruction written
by the current hands, not the instruction it started with. This design decision avoids
most of the problems that Happy had. Happy pushed history into the context and could not
pull it on demand, so it needed a bootstrap dump and an eviction policy (which it never
implemented), and its context window only grew.

Context window usage is also read from the API traffic (`hands.brain.usage`). The brain's
`context_usage` tool reads the usage that the API reports in each reply's `message_start`
and `message_delta` frames. These frames reach hands before Claude Code receives them, so
the figure includes the main-conversation request in which the tool was called. The token
figure that Claude Code shows in its system reminders stays fixed for the whole
conversation (15000000, hands-misc-itx.xak) and is not the source of this value.

For "give me the details of that part" to work, one requirement must be met: every segment
carries the `uuid`s of the records it summarises. Resolving "that part" is then a lookup
instead of a fuzzy search through earlier speech.

## Sessions: membership from files, state from events, liveness from the OS

Three facts about a session come from three different sources. The registry derives each
fact from its source and does not store a second copy of any of them
`[LAW:one-source-of-truth]`.

- **Membership** is the set of files in `~/.hands/sessions/`. The shim writes a file at
  `SessionStart`, or at the first hook of a session that did not fire `SessionStart`, and
  removes it at `SessionEnd`. The daemon scans the directory once before its models load
  and every 2 seconds after that (`hands.sessions.liveness`). A file whose process is
  running produces `Attached`. For a session that is not in the registry, `Attached`
  registers it in the `Idle` state. For a session that is already in the registry, it
  changes nothing, because a hook that arrived first carries more information than the
  file. A sweep and a hook can therefore arrive in either order. As a result, a daemon
  restart loses no sessions: its first sweep lists every session that the previous run
  listed, before anything is spoken.
- **State** is computed by the reducer from the hook events received since the daemon
  attached.
- **Status reported by Claude Code** comes from the file that Claude Code maintains for
  every interactive session: `sessions/<pid>.json` in the config directory that contains
  the session's transcript (2.1.280 to 2.1.289). `status` is `idle`, `busy`, `waiting`
  (with `waitingFor` set to `permission prompt` or `input needed`), or `shell`.
  `statusUpdatedAt` is the time the status was set, in epoch milliseconds. A `!` command
  reports `busy`. Claude Code writes `shell` instead of `idle` while a background bash task
  that the session started is still running (2.1.289). In this case the session is at its
  prompt and no turn is running. hands treats the session as `Idle` with that status, so
  everything below that applies to `idle` also applies to `shell`.
  `hands.sessions.statusfile` reads the file of every listed session ten times per second.
  It applies a `StatusReported` each time the file's timestamp differs from the timestamp
  in the registry. As a result, the daemon still detects a status that is set again to the
  same value, or an idle, busy, idle sequence that occurs between two reads. A status or
  reason that hands does not recognize is parsed as an unknown variant and logged. It is
  never treated as idle. A file that contains a different pid or a different session is
  rejected. A missing or rejected file is logged as an error, once per reason. Without
  this file, the daemon cannot detect the end of a turn stopped with Escape, because
  Escape does not fire a Stop. The registry stores this status as the session's state,
  `Idle` or `Running`, and only a status read moves a session between these two states.
  An `idle` ends the open turn regardless of how the turn was stopped. There is no special
  handling for any particular way of stopping a turn. If a prompt's turn is open but its
  record has not been read, one of two things happened: an Escape cancelled the prompt
  during its hooks, which sets `idle` about 70 ms later, or Claude Code accepted the prompt
  and stopped it before the tail read its record. In either case, the turn ends here, and
  it is reported as a turn only if a record shows that it ran. The session becomes `Idle`
  immediately. Claude Code sets `idle` before the transcript records how the turn ended.
  The interrupt record for an Escape is written 37 ms later (2.1.283), and the Stop for a
  turn ended by Escape can fire after that. For this reason, the turn is kept in the
  `Untold` state and is reported once, at the first of these five events: its Stop
  (reported with the reply that the Stop carries), its interrupt record (one that names its
  prompt), the start of a later turn (reported before that turn's mark), the end of the
  session, or a read of the transcript that reaches `UNTOLD` ms past the timestamp of the
  `idle`. The last event is how a double Escape, which leaves no record, is reported. That
  wait is measured on Claude Code's clock. Each read reports how far it read (`Read`), with
  a timestamp taken before it opened the file. If hands reads a transcript late, the report
  is delayed, but the record is never left out of it. Claude Code sets `idle` only after a
  Stop's hooks have returned, and the daemon responds to a Stop's hook only after it has
  decided the Stop, so a stopped turn is normally ended by its Stop. A Stop whose id does
  not yet appear in any record that has been read is held until such a record is read
  (`Holding`), for at most `STOP_HOLD_SECONDS`. After that time, its hook is released. The
  Stop still reports its turn when a record names it, or it becomes a single line if the
  transcript is read past it without such a record. Claude Code sets `busy` before a
  prompt's hooks run, so any `idle` read after a prompt is applied was set after that
  prompt. A single Escape that flushes a queued message sets `busy` again, not `idle`. All
  three behaviors were observed live on 2.1.282. This ordering holds only if a file is
  read at the moment its report is applied. For this reason, the reader reads each
  session's file lazily, looks it up by id, and reads it only after the previous report
  has been applied.
- **Unreported ends.** A process runs one session at a time, so the *holder* of a running
  process is the newest file that names that process and passes the start-time check. A
  file whose process is not running produces `Died`. The holder produces `Attached`. Any
  other file for a running process produces `MovedOn`. `MovedOn` covers a `/clear`, a
  resume, or a `/branch` inside the process whose end hook never arrived; that session
  ends without an announcement, and its file is removed. The shim removes a session's file
  before it posts `SessionEnd`, so a listed session with no file has ended, even if that
  post was lost. The daemon evaluates such a session only if its file was also missing at
  the previous sweep. This gives the end hook two seconds to report how the session ended.
  The session is then `MovedOn` if another session holds its pid, and `Died` if no session
  does. The sweep gets the list of sessions before it reads the directory, and a session
  is added only after its file is written. As a result, a session that is added during a
  sweep is never mistaken for one whose file is missing. A file with a pid that no macOS
  process can have (outside 1 to 99999) is reported and removed when it is read, in the
  same way as a file that cannot be parsed, so the kernel is never queried about it.
- **Liveness** comes from the kernel. On every sweep, the daemon queries the start time of
  each file's pid (the `kern.proc.pid` sysctl, about 10 µs per pid). A session that waits
  for input can be silent for hours and still be alive, and a session in a tool loop is
  never silent. Silence therefore indicates nothing about liveness. A process counts as the
  session's process only if it started before the file was written, so a pid that a later
  process reuses is treated as dead. A dead process produces `Died`: the session becomes
  `Gone`, a pending permission hook is released, the file is removed unless a new process
  has rewritten it, and hands says "The session cc-hands is gone" once. If the current run
  never listed a session, for example one that died while the daemon was stopped or before
  a reboot, its file is removed without an announcement, because the user was never told
  about that session in this run.
- **Ends** are reported through `SessionEnd`, whose `reason` indicates who ended the
  session. As measured on 2.1.270, `/exit` and a double Ctrl-C report `prompt_input_exit`,
  `/clear` reports `clear`, and a closed terminal reports `other`. hands does not announce
  an end that the user chose at the keyboard. For `other` and for any reason that hands
  does not recognize, hands announces that the session is gone, with the same sentence it
  uses for a dead process.

`list_sessions` returns a session while its pid is alive. Each entry shows the session's
project and name, its state, and whether it is the focus.

## Focus, drafts, and other state kept outside the model

Models do not reliably track "which session we are talking about" or "I am in the middle
of a draft" across a long conversation, and both failures are costly. For this reason,
both are stored in the daemon as typed state, and the tools use them as defaults.

**Focus** is `SessionId | None`, or `Unreadable` when its file cannot be read. It is
stored in the `focus` file in the hands home directory (`hands.sessions.focus`) and read
each time it is used. Every tool that acts on a session treats an omitted `session`
argument as the focus (`defaulting_to_focus`), and a session named in the argument takes
precedence over the focus. "Switch to cc-hands" maps to `focus_session` and requires no
confirmation. When hands reports a session's turn or question, that session also becomes
the focus, so the user's reply goes to it (`hands.voice.refocus`). The focus moves when the
model receives the report, after any words the user spoke before the report, so those
words still go to the session they were intended for. The brain's stage moves the focus
before it sends the request to the model. A barge-in during playback stops the audio but
does not undo the focus change. A session that has ended never becomes the focus. The
focus is included in the tail of every request to the brain, and `list_sessions` also
returns it, so hands, not the prompt, answers "which one did you mean".

**Drafts** are stored per target: `NoDraft | Staged(text, resolutions)`. The readback is
generated from the stored resolutions, never from the model repeating itself:
"Draft for cc-hands, reading 'auth middleware' as `authMiddleware.ts`: refactor the
auth middleware to use the new token helper." The readback reports what changed, not a
repeat of what was said. A draft can be staged, amended, or discarded, and `send_draft`
sends it as the Type effect. `send_command` and `interrupt_session` emit the same effect
with a `Command` or the Escape `Key` (`hands.core.keyboard`). fritter holds the session's
pseudo-terminal, and `Typist` types into it over a unix socket. hands therefore does not
need to locate a window, take window focus, or request a macOS permission. A send appends
a `Typing` audit record before it types, and a `TypingFailed` record with the same effect
if the typing fails. One file therefore answers both "did it send something I didn't
approve" and "did it arrive".

**Drives** are standing orders that the user gives once for a session and that apply to
many prompts (`hands.core.drive`). `drive_session` stores the drive in the registry next
to the drafts, which is the only place it is stored, and the drive is removed when the
session ends. While a drive is active, `delivery` passes each turn that the session
finishes to the brain as `Steering`, regardless of the quiet, overlay, or finished-turn
settings, so a drive never stalls because of a setting intended for audio output. Those
settings still control whether the brain speaks about the turn. The turn is processed in
hands' lane like any other report, and `speech.Handed` records the reason: `ToAct` leaves
the user's focus unchanged, and `ToAsk` (a driven turn that asks the user a question)
moves the focus and instructs the brain to stop and ask the user. The asker of the
`voice.turn` event carries this value. The brain responds with `drive_send`, which types
the next prompt through the same Type effect as a draft, or with `stop_driving`. When the
brain's turn ends, whether it was answered, failed, or interrupted by a barge-in, the
stage returns the drive (`HandedBack`). If the drive is unchanged from when it was handed
over, it ends and hands announces this, so no drive remains active without the brain
acting on it. A send that fails to type is not counted, and a turn that cannot be read
ends the drive with an announcement.
`drive.decide` is the only place where a prompt reaches a session without the user saying
"send it". It rejects any session that has no drive, never modifies a staged draft,
rejects a session that is showing a dialog (`reach.prompter`, shared with drafts and
commands), and ends the drive on the send that reaches its limit of `SENDS` prompts.
Giving the order again does not reset the count. The count resets only when the drive
ends.

## The summary store

A project's backlog is too large to load unprocessed into a spoken conversation: 132 open
links tickets are ~145K tokens through `lit show`, and one sentence per ticket is ~3K. For
this reason, `read_backlog` and `read_ticket` return one sentence per ticket from the
summary store, and return the ticket's full text only on request (`full`). The backlog is
read from `lit export` on every call (`hands.sessions.backlog`). The store holds only
sentences.

Each sentence is keyed by a digest of its inputs (`hands.core.sentences`): the
summariser's instruction, the item's own text, and the sentence of each item below it.
The tree has the backlog at the root. Below it are the unfinished tickets that have no
unfinished parent (epics, standalone tickets, and follow-ups filed under a closed ticket),
and each of those has its unfinished children below it. Editing a ticket changes its key
and the keys of its ancestors, and no others. The change stops propagating upward at the
first sentence that is regenerated unchanged. A rerank or a status change does not change
any key. An edit to the instruction changes every backlog key (a turn's key is its
identity, described below). The rows are stored in a single table in
`<home>/sentences.db` and are never updated.

A single task generates sentences outside the voice path (`hands.voice.summarising`). The
backlog for each live session's project is requested at startup and again on every read.
The task requests the due sentences from the summariser, twenty items per call, and
processes leaves before their parents, because parents are keyed by their children's
sentences. Each pass produces one `summary.backlog` event. The event records the number of
items in the backlog; how many were already summarised, newly summarised, and still
unsummarised at the end of the pass; the number of rounds, calls, failed calls, and
unexpected reply lines; the items that a reply omitted (`left_out`); the error that each
failed call raised (`errors`); and, after lit has responded, whether lit has a workspace in
that directory (`tracked`). A directory where lit has no workspace (because `lit init` was
never run, or the directory is not in a git repository) has no backlog. Its pass succeeds
with `tracked` false and every count zero, and `read_backlog` returns an empty backlog with
`tracked` false. A pass fails with lit's error if the export could not be read, and with
the error of the last failed call if a call failed, for either kind of pass. A tool never
waits for the task. An item that does not have a sentence yet is returned with its title
and counted in `unsummarised`. Measured on 2026-09-29 on this repository: 52 sentences in
5 calls and 45 s with a cold store, and 0 calls and 0.16 s with a warm store.

Sessions are served the same way. `read_session` splits the full transcript into turns,
one per request (`hands.core.turn.turns`), and returns the newest forty, with a sentence
for each finished turn, or the turn's request if its sentence does not exist yet. `before`
pages back to older turns, and `read_turn` pages through one turn's steps starting from a
mark. A finished turn never changes, so its key is based on its identity instead of its
text (`turn_digest`): the id of its first record and the number of happenings in it, with no
instruction included. Turn sentences are requested on each read and generated by the same
task, with one `summary.turns` event per pass. The task skips turns that were summarised
after they were queued (`known`) and counts the turns it requested, summarised, and failed
on. The registry determines whether the last turn is finished, and only an `Idle` or ended
session confirms it. An `Unreported` session may be in the middle of a turn.

## The intermediary's tools

```
list_sessions()
read_session(session, before?)   read_turn(session, turn, since?)
read_backlog(session)            read_ticket(session, ticket, full?)
focus_session(session)
end_session(session?)
interrupt_session(session?)
send_command(session?, command, args?)
stage_draft(session?, text)      amend_draft(session?, text)
discard_draft(session?)          send_draft(session?)
drive_session(session?, order)   stop_driving(session?)
drive_send(session?, text)
answer_permission(request, decision, message?)
answer_question(request, answers)
find_path(session?, query)
catch_up(minutes?)
recall(query, since?)
expand(session, part?)           resume()
skip()                           repeat()
stay_silent()
```

The boundary rule: the tools route, name, and read session records and summaries. No tool
writes to a repository, and the conversational model never receives a file's contents.
The daemon does read what a session changed, through the session's transcript and git
delta, because summarising results is its job. The summariser receives diffs so that it
can describe them. `find_path` returns paths from `git ls-files` in the target's `cwd` so
that a spoken "the auth middleware file" can be resolved to a real path in the draft. It
returns file names, never contents. `catch_up` and `recall` read the daemon's own audit
log. If the intermediary had an edit tool, it would eventually decide that editing the
file is faster than routing the user's request. The tools listed above are the complete
set `[LAW:no-mode-explosion]`.

The model uses `stay_silent` to decline to respond to speech that was not addressed to it.
It is based on Happy's `skip_turn`. Push-to-talk rarely needs it, but the wake-word edge,
which opens the microphone without a key press, does.

**Two people in the room.** hands identifies the speaker of each hold that contains speech
(`hands.voice.speakers`). A hold that the owner opened by hand belongs to the owner. A
hold that was opened by voice is attributed to the owner or to someone else based on the
cosine similarity between its CAM++ embedding and the owner's voiceprint (threshold 0.5).
The voiceprint is stored in the `speakers` directory in the hands home. There is no
enrolment step. The voiceprint is trained from holds of the desk's key and from each
engaged conversation that, after it ends, is found to contain only the owner's voice (at
least three such holds are required for the first voiceprint). The voiceprint is therefore
learned in conversations with only the owner and used in conversations with other people.
Each hold carries, from the gate, the conversation it was spoken in, because its identification can
arrive after the owner has disengaged.
The `Room` identifies everyone else (`room.json` next to the voiceprint, mode 0600; to
reset the other speakers, stop hands and delete the file): each person has their own
voiceprint, a number assigned in the order they were first heard, and the name they gave
through the brain's `name_voice` tool. Their speech reaches the brain as
`[Sam, someone else in the room: ...]`, `[someone else in the room,
voice 2, name not yet known: ...]`, or, for a hold too short to identify or when the Room
fails, `[someone else in the room: ...]`. Anyone can ask questions and discuss the work,
but only the owner's instructions trigger actions. The one exception is that a guest can
set their own name. Each identification is logged as a `Voiced` line with its similarity
score, which is used to tune the threshold.

**The prompt.** The brain's prompt is `voice/intermediary_instruction.py`. The prompt
states when to use a tool, and the tool's docstring explains how to use its result. The
prompt names only tools that the model has, so the ticket for each planned tool adds that
tool's line. When the prompt quotes a wrong reply, the model copies it almost word for word
(measured against Qwen on 2026-09-26: "The docs site is idle, so it can take this" came
back as "The auth refactor session is idle, so it can take this"). For this reason, the
prompt quotes a wrong reply only when copying it could not produce an acceptable reply, and
describes a plausible wrong reply as an action instead.

`send_command` exists so that `/clear`, `/compact`, and `/model` reach the target as
commands, with their sigil intact. In `stage_draft` text, a leading sigil is always
escaped. The two never share a code path that inspects the first character, because the
`Input` variant already identifies the kind of input. Claude Code recognizes three sigils
at the start of a prompt: `/` for a command, `@` for a file mention, and `!` for shell
mode. After a space, each sigil is plain text, so `Text` is always typed with a leading
space, regardless of its first character. Its newlines stay inside the prompt because
fritter pastes it.

**Starting a session.** The brain's `start_session` tool starts a session, as described in
the brain's `hands:start` skill (`hands.sessions.startsession`). Only the brain has this
tool, because finding the folder requires a shell. The daemon runs the tool in the
environment that hands was started in, which is the user's environment. The brain's shell
has an environment that hands created for the brain, without the user's credentials, so a
session started from it would not belong to the user. The tool runs hands' own `claude`
(the shim in hands' bin directory) in a new window of the tmux session named after the
folder, and creates that tmux session if it does not exist. The command runs through
`/bin/sh` with the variables removed that fritter's tap and a Claude Code session set for
the processes they run. This is necessary because a tmux server that is already running
gives a window the server's own environment, which contains a session's variables if the
server was started inside a session. The home that the session reports to is passed to its
window by name. The tool returns when the registry contains a session whose process the
window started, so a draft can be staged for it immediately. It returns the session's id
and pane, or an error that gives the reason: the folder is not a full path, tmux is not
available, `claude` exited first, the session did not join within 30 seconds (usually
because of a dialog, whose pane the error shows), or the session joined outside fritter. A
newly started session has received no instructions. The user's request for it is staged
and sent like any other prompt. Each start produces a `session.start` event.

**Closing a session.** The brain's `close_session` tool ends all of the given sessions
together, as described in the brain's `hands:close` skill (`hands.sessions.closesession`).
It sends SIGTERM to each session's `claude`, unless the liveness sweep has determined that the
process no longer holds the session (the process ended, its pid was reused, or a /clear moved it
to another session). Claude Code handles SIGTERM the same way as a closed terminal and fires
SessionEnd (2.1.289). The tool then waits until the registry
no longer lists the session, for at most 10 seconds. The program that ran that `claude`
behaves as it would after an exit at a terminal: a window that hands opened ran nothing
else, so tmux closes it, and a shell that the user ran `claude` from returns to its
prompt. A close is requested in one of two modes: `named`, which ends the session
regardless of what it is doing, or `done`, which ends the session only if it is at its
prompt with no dialog open, no turn started, and no background shell or subagent running.
Otherwise, `done` leaves the session running and reports what it is doing. The registry's
state is read when the close runs, so a session that started a turn after the brain listed
it is not closed, even if its busy status has not been read yet. Each close produces a
`session.close` event.

## The audio side

**The gate sets the turn boundaries and the mute.** Pipecat's turn strategies act on
voice-activity (VAD) frames, so the key acts as the VAD. Every microphone frame carries
the key state at the time it was captured. Whisper pushes VAD frames at the points where
the key state changes and numbers each hold. As a result, the turn that the strategies
see and the audio that Whisper transcribes start and end at the same frame. How a turn
ended is not represented as a key position. Instead, the gate counts the turns it has
sent and the turns it has discarded. Every frame carries both counts, and Whisper ends
its hold when either count changes. A frame is captured every 20 ms, so one turn can end
and the next can arm between two frames. The key also controls the mute. Unless the key
is pressed, microphone bytes are replaced with silence of the same length. Frames flow
at the full rate in both cases; only their content changes. The microphone opens when
the key is pressed, and a hold's audio starts at that point. As a result, words spoken
before the hold is confirmed as talk are kept. They are discarded if the press turns out
to be an ordinary Shift press. The hold opens the turn, and from then on the user has
the floor. The release ends the turn, and the release is final. The turn's cut is the
interruption. It flushes queued audio, cancels the reply that is still streaming from
the model along with its in-flight calls, and marks a line spoken through the player's
`heard()` as cut off. This is barge-in.

The edge that opened a turn's hold determines when the turn cuts (`turn_start` in
`hands.voice.trigger`, enforced by `EdgeTurnStart`). Each hold carries that edge from
the gate, in the same way that it carries the key. A hold that the user opens by hand,
with the held key or the phone's button, cuts as soon as it opens (Pipecat's VAD start).
To stop hands, the user presses and releases one of these without speaking. A hold
opened by the wake word also cuts as it opens, because the wake word is said on purpose
and its detector receives no audio while hands speaks. A hold opened by engaged
conversation cuts only after Whisper pushes words for it (Pipecat's transcription
start). In engaged conversation, the desk's detector receives audio through the echo
canceller, and residual sound from the reply, or a cough or a door, can open a hold.
Between 23:52 on 2026-10-05 and 02:53 UTC, on the MacBook's speakers, 14 of 33 voice
barge-ins during a reading contained no words for Whisper. Such a turn does not cut
anything, so nothing has to be undone. Until the turn cuts, no component after the user
aggregator is notified that the user started speaking. The reply in progress continues
to play, and continues to act on a tool's result, as if no turn had opened. Meanwhile,
Floor holds back the session updates hands reports from the moment the hold opened. A
cut can occur while the turn's words, and the note that accompanies them, are still
being sent to the model, so those messages are uninterruptible. An interruption stops
hands, never the user. The user aggregator does not accept any of them until the
assistant aggregator, at the end of the pipeline, reports that the interrupted reply has
been written to the context as cut off (`CutWritten`). As a result, the context always
lists that reply before the words that interrupted it, no matter how soon after the cut
those words arrive. Each turn is recorded as a `UserTurn` line in the audit log. The
line records how the turn's edge made it cut and how long after the turn opened the cut
happened, or that it never cut. A turn ends when Whisper has finished with every hold in
the turn (`KeyTurnStop`). A press that occurs while the last hold is still being
transcribed joins that turn, so no hold's words are left out. Nothing else ends a turn.
Pipecat's user aggregator would otherwise end a turn after 5 s with no speech and no
transcript, and a slow or queued transcription can take longer than that. For this
reason, that timeout is set to never (`user_turn_stop_timeout` in `build_voice`).
Instead, Whisper limits the length of a turn. A transcription that has not returned
within one minute fails, and the failure is announced aloud like any other failure. This
resolves the hold (`TRANSCRIBING_SECONDS`).

**The mute is applied where sound is captured, and the speaker's echo is cancelled.**
Measurements on 2026-09-14 with MacBook Pro speakers and microphone showed the
following. The interruption stops writes to the speaker within a few milliseconds of the
press. However, audio that was already written stays above the room's noise floor at the
microphone for about 185 ms. In every run where nobody spoke, Whisper transcribed that
tail as a word ("Wow.", "Well.", "What?"). A Pipecat input filter could not prevent
this, because a filter runs when the event loop reaches a frame, tens of milliseconds
after capture. For this reason, `hands.voice.microphone` replaces both halves of the
local transport. The two halves share the echo canceller of the open streams
(`hands.voice.echo`, WebRTC's AEC3 through LiveKit's binding). Each reopen opens the new
pair of streams with a new canceller, which learns the new room from scratch. The old
canceller is closed on the thread that closed its microphone stream, so a stream
callback that is still running never reaches a canceller that has been released. The
`Speaker`'s stream is a `Playout` that is fed from PortAudio's callback. The device
takes 20 ms of audio whenever it needs it: the written audio, followed by silence. The
canceller receives exactly that audio, on the device's clock. When the canceller was
instead given audio at the time it was written, the reference shifted relative to its
echo after every silence. The shift depended on where a reading's first write fell
between the two devices' callbacks, and AEC3 did not cancel the start of each reading. A
write fills at most 140 ms of queued sound, adding more as the device makes room, in the
same way that a write to a blocking stream fills its buffer. The `KeyedMicrophone`
passes every buffer through the canceller in PortAudio's capture callback, whether the
key is up or down and at either place, so the canceller keeps learning the room. The
callback then passes the buffer on only while the key is down. If the canceller raises
an exception there, the pipeline ends, and the run ends with it. Otherwise PyAudio would
abort the stream and leave hands unable to hear. AEC3 expects one frame of reference
audio for every frame of microphone audio, as a single device that plays and records at
the same time provides. The two devices have separate clocks, so the canceller buffers
the audio that the speaker's device took. Each 10 ms of microphone audio is paired with
the next 10 ms of that buffer, or with silence when the buffer is empty. The amount
buffered equals how far the reference lags its echo. At a reopen the speaker starts
before the microphone, and the speaker's clock may run fast. Reference audio that stays
buffered for a whole second without being needed by any microphone frame is slipped
(dropped), except for one spare buffer. Each released microphone stream produces one
`microphone.let_go` wide event. The event records how many frames its canceller
processed, how many of those had nothing playing, how many it dropped unprocessed
because the microphone had stopped taking them, and how many it slipped. Each released
speaker stream produces one `speaker.let_go` event. The event records how many frames
its device took, how many of those had nothing written, and how many callbacks PortAudio
reported as late. Every frame the microphone pushes also carries the same audio as it
was captured, before the canceller (`KeyedAudio.captured`). Whisper keeps this audio for
the frames in a hold, so each hold's `HoldHeard` line records the mean power in dBFS
before and after the canceller (`levels.captured_dbfs`, `levels.heard_dbfs`). A
transcript produced from residual echo of the reply shows a high captured level and a
low heard level. A transcript produced from the room with nobody speaking shows low
levels on both. A level is null for digital silence. At the phone, audio does not pass
through a canceller, so the two levels are equal. Measurements on 2026-10-03 through
this transport showed about 27 dB of echo removed. A press during a reply with nobody
speaking produced no words from the reply in 8 of 8 holds. The raw microphone produced
one in every hold. When "Stop. What time is it?" was spoken starting 50 ms after the
press, "Stop." was kept in 8 of 8 holds. The previous mute kept the microphone shut
until the reply's sound had faded, and lost that first word in 8 of 8 holds.

**When a device is lost, hands switches to the new default instead of waiting.**
PortAudio does not report when a device disappears. In a test on 2026-09-24 with a
CoreAudio aggregate device destroyed mid-stream, the microphone's callbacks stop, a
write to the speaker blocks indefinitely, and the stream still reports itself as active.
Before this change, unplugging a headset left the daemon running but unable to hear. The
heartbeat reported the daemon as up, the next turn produced nothing, and Pipecat's
10-second write timeout then marked the speaker as unusable for the rest of the run.
macOS changes the default devices as soon as the current default disappears. The
`hands.voice.coreaudio` listener reports this change within about 17 ms, and hands uses
it as the signal. The follower in `hands.voice.devices` then reopens the entire
transport in this order:

1. Both streams are detached, so a write that arrives during the reopen is reported as not written.
2. The speaker is closed before the microphone, because closing it releases the stuck write.
3. PortAudio is terminated and restarted, because it lists devices only when it starts.
4. Both streams are reopened on the new defaults, and the speaker is made usable again.

It then speaks `Audio moved: listening on …, speaking on …` through the system channel
on the new speaker, or posts the message if speech is unavailable. Closing a microphone
whose device is gone takes 3 to 4 s, so the whole move took about 5.5 s from unplugging
to the announcement. Each move produces one `devices.moved` event. The event records the
devices it moved away from, the defaults that triggered it, the devices it reopened on,
the defaults it read while reopening (the next move is measured against these), and how
long the reopen took. If the reopen fails, the event is marked failed with the exception
that was raised, even when the run was stopping at the same time. The next turn ran end
to end on the built-in devices. Every step that accesses a device runs outside the event
loop. If a reopen takes longer than 10 s or fails, the run stops and is reported as
down. The next `hands run` opens on whatever devices are available. The same process
handles a headset being plugged in, or a default device changed in Control Center. A Mac
with no built-in microphone (a Mac mini or Mac Studio) can lose its only input device.
In that case, the microphone holds an empty stream (`NoInput`), which is opened,
started, stopped, and closed like any other stream. The move is announced as
`No microphone: hands cannot hear you. Speaking on …`. A daemon that starts without a
microphone says `hands is up, but there is no microphone, so it cannot hear you.`
instead of failing setup and stopping. Without a stream, no frames reach Whisper and no
turn starts, so the key edge responds to a press directly with
`There is no microphone, so hands cannot hear you.` When a microphone is plugged in
later, it changes the default input, and the follower opens it as it does for any other
move. This behavior is tested against a PortAudio that lists no default input. A MacBook
cannot be put in that state, because macOS always falls back to the built-in microphone.

**The gate has one owner and several edges.** `PushToTalk` stores the key position. Any
component that reads physical input calls `PushToTalk.move` and identifies itself. The
edges are values of one type, `Edge` (`hands.voice.trigger`), not modes of the gate
`[LAW:one-type-per-behavior]`. The gate stores the edge that opened the last turn, and
each turn's `UserAsked` records it:

| Edge | Down | Up |
|---|---|---|
| `held key` | Right Shift held alone for 600 ms, in any app (`hands.voice.hold`) | released; pressing another key while it is held drops the turn without sending it |
| `engaged conversation` | engaged by one hold of Right Shift; then Silero confirms speech on the desk microphone (`hands.voice.engaged`) | Smart Turn determines that the speech is complete, or the silence after it exceeds its `stop_secs`; another hold disengages |
| `button` | a HID button or headset button pressed | released |
| `phone button` | the phone page's talk button pressed | released |
| `wake word` | openWakeWord detects the wake word, "Hey Jarvis" unless config.toml specifies another, on the desk microphone (`hands.voice.wake`) | same as `engaged conversation`, once the request has started after the wake word |

The wake-word edge is the only edge that opens the mic without the user's hand, and it
is half-duplex. While hands speaks, the wake-word detector receives silence instead of
room audio, because an open mic in a room with speakers picks up the pipeline's own
voice. The pause after the wake word does not end the turn. Smart Turn judges the wake
word alone as complete, so a stop is sent only after speech has started again after the
wake word. After the wake word, the driver starts listening again from a clean state
(`Ears.afresh`). Silero reports nothing until it is sure of speech again, and Smart Turn
holds no audio. As a result, the request starts like any other speech, whether after a
pause or in the same breath, and Smart Turn evaluates the request by itself. The edge
shares engaged conversation's driver and models (`hands.voice.engaged.drive`). The desk
listens for as long as the edge is in use, so the wake word is included in the turn's
audio for Whisper. `set_trigger` downloads the model's files into the `wake-word`
directory in the hands home directory before the switch (`hands.voice.trigger.readied`).
If the download fails, the switch is refused with a spoken message, and the current
trigger remains in use.

**The trigger is the desk's edge, a single setting changed by voice**
(`hands.voice.trigger`). The phone's button is available on every call. At the desk, the
edge that drives the gate is the current `Trigger`, a single value that `Triggers`
stores while hands runs. The brain's `trigger_in_use` reports it, and `set_trigger`
changes it. When it changes, the old edge stops and the new one starts
(`Triggers.drive`), so the next turn opens with the new trigger. Implemented triggers:
the `held key`, `engaged conversation`, and the `wake word`. A trigger that is not
implemented is rejected by the tool's closed set of values, and the current trigger
remains in use.

**An engaged desk listens between turns.** Engaged conversation moves the gate with the
same moves as the held key, plus two of its own: `listen` when it engages and `deafen`
when it disengages. Each is signaled with two tones. While the desk is listening, the
key stays at `listening` between turns instead of `up`. The microphone's bytes reach
Whisper, which keeps the last second of them, as Pipecat keeps audio in which nobody is
speaking. As a result, a turn opened by voice starts with the words spoken while Silero
was confirming speech (about 0.2 s). Listening applies only to the desk. At the phone,
the key stays at `up`, audio at the desk does not open a turn, and the desk starts
listening again when hands returns. Both models receive desk audio through the echo
canceller. In measurements on MacBook speakers, after a canceller had processed one
reply, no part of the next reply triggered Silero. With the raw microphone, every phrase
triggered it. Only during the first two seconds of a new canceller did a reply get
through.

**The phone is a second place, alongside the desk.** The desk is the Mac's own mic and
speakers. The phone is a page with a talk button that hands serves
(`hands.voice.phonepage`) and that a phone opens over the LAN or the tailnet. Both are
available for the whole run, and there is no setting to choose between them. The gate
stores the place where the last turn was opened, and hands is at that place. The
pipeline receives audio only from that place's microphone, so Whisper gets one stream of
frames. The speaker plays to that place, so a reply, a session update, and a turn's tone
go to where the user is (`hands.voice.ptt`, `hands.voice.microphone`). When a call
connects, hands moves to the phone. When the call ends, hands moves back to the desk in
the same step, so no audio is played to a phone that has disconnected. An offer that has
been answered but is not yet connected does not move hands, so a page that cannot reach
hands never holds its speech. The newest offer releases any older offer that is not yet
connected, and replaces the active call once it connects. A place cannot affect a turn
that belongs to the other place. Pressing Shift at the desk while the user is talking on
the phone arms nothing. If a turn opens at one place while a hold is open at the other,
both are dropped, in the same way that pressing a key while the talk key is held drops
the turn. The gate plays the cues for the moves and reports them in the order it
processed them. Only a turn opening moves hands. A press that is still arming never
moves hands, because it may be an ordinary Shift press. As a result, a hold at the place
where hands is not keeps none of the words spoken before it opened a turn, and typing at
the desk never moves a call's replies off the phone. Each move of hands between places
produces one `place.moved` event. The event records the old and new place, whether a
turn or a call caused the move, and whether a hold open at the old place was discarded.
A move caused by a call is part of that call's trace.

**Cues indicate what hands is doing while it is silent** (`hands.voice.cues`). A cue
plays at each edge of the talk key as the key moves, whether or not anything downstream
accepts the turn. Two more cues cover silent periods. A high rising pair plays when a
turn's words are written to the model's context; a hold that captured nothing or was
dropped never reaches this point. One steady tone plays when hands acts: at each call of
any of its tools except `stay_silent`, and at each burst of progress that is heard. It
therefore never plays for another session's progress, a muted session's progress, or any
progress while hands is quiet. Pending cues go into one queue. A cue plays once nobody
is speaking, as indicated by Pipecat's started-speaking and stopped-speaking frames for
hands and for the user, and once the cue's minimum interval since it last played has
passed. The working tone plays at most once every two seconds, the receipt cue is never
delayed behind it, and cues that accumulate in the meantime play as one. Each play is
recorded as a `Cued` line in the audit log, with how many cues it represented, how long
it was delayed, and where it was played. The destination is nowhere while no speaker is
attached.

The call (`hands.voice.phone`) is a single WebRTC connection made directly with aiortc.
hands' speech is sent to the phone on an audio track. The phone's microphone audio is
not. The page sends it as plain 16-bit audio over the call's ordered data channel, in
order with the button's presses and releases. The page holds back a release until every
block its microphone captured before the release has been sent. A release that arrived
ahead of the words spoken before it would cut off the end of the turn. On a single
ordered channel, the key state travels with the audio, as `KeyedAudio` makes it travel
at the desk. Earbuds keep hands' voice out of the phone's microphone, so the phone's
audio is gated only by its button.

The same channel carries the outcome of each hold in the other direction. The latency
observer (`hands.voice.latency`) sends the phone each mark of a turn as it logs the mark
(`hands.voice.mark`): `released` or `discarded`, `transcript` or `no words`,
`first LLM token`, `first audio`, and `failed` if a stage failed first. The page shows
the latest mark under its button. While a reply is pending, it shows a running count of
the time since the button was released. Once the first sound arrives, it shows the time
to that sound. Every time shown is measured on the page's own clock, from the button
release to the mark's arrival, so it is the wait the user experiences at the phone,
including the network.

`no words` applies to a turn, not to a hold. It means the user aggregator reported that
the turn ended and no hold in the turn contained words, so nothing is sent to the model.
A silent hold released after a hold whose words were captured is answered along with the
rest of its turn. `failed` is a pipeline error that leaves the release without a
response. It is either an error after the turn was sent, which means the reply failed,
or an error while the turn was being collected after which the turn ends with no words,
which means Whisper failed on it. Both end the turn's latency window, so whatever hands
says next, including the failure message itself, is not timed as the answer.
`first audio` is the first sound after the release, regardless of what hands said, and
the page shows only that hands spoke.

A mark does not include a hold number, so the page places it by its position in the
sequence. Each mark lists the marks it can follow. Until hands reports that it accepted
or discarded the hold shown, no other mark sent refers to that hold. A mark for a hold
that the page has stopped tracking is not shown. A call's `phone.call` event records how
many marks were sent to its page.

The channel also carries the transcript (`hands.voice.transcript`). The page keeps it
behind a button and hides it until the user opens it. Every message on the channel is
one JSON object whose `kind` identifies its type: a `mark` or a line. The user's words
are the line Whisper transcribed for each hold. Each of hands' sentences is sent twice:
when the output transport writes its first audio, and when the transport releases the
end of the sentence or a barge-in cuts it off. The sentence's text alone does not mark
its start, because the first sentence of a reply is released before its audio is
generated. pocket-tts does not report word timings, so hands measures its speaking rate
from each sentence that plays to the end, and sends that rate with the next sentence.
The page moves the highlight through a sentence's words at that rate, keeps it on the
last word until hands reports that the sentence is done, and strikes through any words
that a barge-in or the end of the call cut off. The `phone.call` event records the
number of lines sent, in addition to the marks.

**The conversation page shows the log's conversation, with a text box for typing**
(`hands.voice.conversationpage`). It is served alongside the phone's page, on the same
port and addresses, at `/conversation`, and requires the phone's key in the same way
(`carries_key` in `hands.voice.phoneaddress`, the only check of the key). It shows the
newest 200 moments of the audit log, folded the same way `hands recall` folds them
(`hands.sessions.recall.Moments`): what the user said, what hands said, what was sent to
a session and the session's answer, and each tool hands called (its `tool.run` event). A
tool call's arguments and result can be expanded on request. `hands recall` omits the
tool calls. The daemon keeps one fold of the newest 200 and continues reading from the
last log offset it reached. Each read from a page includes how many changes the page has
seen and waits up to 25 s for another change, so the read returns as soon as something
is said. A send that is later found to have failed, or a session that is named after the
send, is redrawn in its original position. Typed words are queued into the voice
pipeline as `Typed`, which interruptions do not drop. Whisper treats them as a separate
hold with the `typed` opener: the hold opens and ends at once, and is processed in order
after any key holds that are still being transcribed. This is not an edge. It does not
move the gate and does not change where hands is. A typed hold interrupts hands as it
opens, as a press does, and joins a turn that a key hold has open. From there, the words
are sent to the model and return to the page through the log, in the same way as spoken
words. Each read produces one `conversation.read` event, which records what the page had
seen, how long the read waited, and how many moments it returned. Each send produces one
`conversation.typed` event, which records the number of characters typed. A request
without the key or without words is rejected, and its event records the reason.

A browser allows a page to use the microphone only over HTTPS. Under the tailnet name,
hands presents the certificate that `tailscale cert` issues for that name, which the
phone trusts without any extra step. Under a LAN address, hands presents a self-signed
certificate. The phone prompts the user once to accept it, and hands generates a new one
at startup if the current one expires within 30 days. hands requests the tailnet
certificate from Tailscale again every day while the page is served. Serving the page
produces one `phone.served` event, which records the tailnet name it is served under, or
the reason it is served only on the LAN. This event occurs after the pipeline starts, so
it is not part of `hands.start`. Anyone who can reach the port can open the page, but
cannot start a call. An offer must include the phone's key, a secret stored in the hands
home directory. The key is passed in the fragment of the page address, which a browser
never sends with the page request. `hands phone` prints the addresses with the key, and
prints the first address as a QR code. aiortc does not detect a browser that is closed
abruptly, so hands hangs up a page that sends nothing for `QUIET_SECS`. An open page
sends its microphone audio every 20 ms. The page applies the same check in the other
direction. hands sends its audio every 20 ms, including silence. If the page has
received no audio for 3 s, or its channel closes or its connection fails, the page
treats the call as dropped and reconnects on its own. It retries after 1 s, then doubles
the interval up to every 8 s, and retries immediately when the page is brought to the
foreground. If a press starts a call that never opens, the failure is reported and the
call is not retried, because the user is present and can press again. The page keeps the
microphone open between calls, because a browser grants the microphone in response to a
press, not to a page that reconnects on its own. The page stops when microphone access
is taken away, since a call without it could not be heard. Every offer includes the
page's name, which the page generates when it opens, and the requested action. Pressing
Connect sends `take`, which takes the phone from any page. A page that reconnects sends
`resume`, which hands declines with a 409 while another page has the call or a pending
offer. In that case, the other page keeps the phone, and this page stops reconnecting
until Connect is pressed. As a result, two open pages never take the phone back and
forth, and a page whose previous call hands has not yet hung up replaces that call. Each
call produces one `phone.call` event, which is the root of its own trace from the offer
to the end of the call and is written when the call ends. The event records the outcome
and the reason: refused at the page because of a missing key or missing offer; declined,
as a resume; released before it connected; or ended after hands was at the call, with
how long after the offer hands arrived. Each offer that was read records the page and
the requested action (`asked`). An offer whose body could not be read, or that hands
could not answer, is marked failed with the exception that was raised.

**One audio owner.** Pipecat's output transport is the only component that plays sound.
When two sessions finish at the same time, their utterances queue behind it instead of
overlapping.

## Loud failure

In an audio system, the default output is silence, and a system that is thinking is
also silent. For this reason, every failure reaches the user through a path that does
not depend on the component that failed `[LAW:no-silent-failure]`:

1. **Speech.** The system channel speaks messages such as "the language model is
   unreachable", "speech recognition failed for that turn", and "the session cc-hands
   is gone". These are `Speak` effects and do not require a model. `hands.voice.system`
   renders each message from a template and queues it at the TTS processor, after the
   LLM and outside its context, so the model never reads a system message as one of its
   own replies. The worker's `on_pipeline_error` routes each error based on the
   processor that raised it. LLM errors become "unreachable", "usage limit reached,
   until <when it lifts>" (the time is read from the API's error message; that message
   is never spoken), or "failed: <category>". Whisper errors become "speech recognition
   failed for that turn". TTS errors are shown on the screen. Pipecat classifies an SDK
   connection error as UNKNOWN, so "unreachable" is detected from the exception type. A
   model reply that contains no text and no tool call is also a model failure, reported
   as "sent back nothing". The API services detect it from the frames they push
   (`EmptyReplyFails`), except for a reply to a tool call's result. The brain's stage
   detects it from the wire, across all of a turn's replies. A turn that the model ends
   with stay_silent made a tool call, so it stays silent. A reply cancelled mid-stream
   was abandoned and is not treated as empty. An Anthropic request that times out,
   which Pipecat reports only as an event, is announced as unreachable. A turn that
   Whisper transcribed to nothing is not announced (Brandon, 2026-09-27: "I do not need
   to hear it"); it is only logged. For the same reason, a muted microphone is not
   announced, because every turn it produces is empty. The `Whisper` subclass resolves
   the hold when transcription ends, so the turn closes at that point rather than after
   the user turn's 5-second stop timeout. The channel announces a burst of the same
   fault once: a fault is not announced again until ten seconds have passed since it
   was last announced. Faults tend to recur in bursts. On 2026-09-22, a held key queued
   hundreds of empty turns, and their reports were then sent every 0.43 s until the
   queue drained. That is not a loud failure but a jammed one, because nothing else
   could be heard while it ran. A burst ends after a period of quiet, not when a
   different announcement is made, because in the case this rule exists for there is
   never another announcement: with no microphone, waiting until something else was
   announced would leave the user pressing the key at a daemon that has gone
   permanently silent. Suppressing a repeated announcement does not hide the fault:
   every occurrence is still logged. `Announced` is written at the point where the
   sentence was accepted, either by a working TTS or by the screen, so a notification
   that the screen refused is logged as a failure rather than as an announcement. How
   often hands attempts an announcement and what the user actually received are two
   separate facts and are tracked separately. The window starts when the channel
   decides to speak, not after it has spoken, because Pipecat dispatches each pipeline
   error on its own task, so a failed TTS that raises once per queued frame creates
   hundreds of these decisions at the same time. The window is keyed on the fault, never
   on the sentence: Pipecat's text for a silent utterance contains a new context id
   every time, so `Post` stores that text as the part that varies and the fault it
   reports as the part that recurs. Only a closed set of values is ever used as a key: a
   `SystemFact`'s own sentence, or that fault. The LLM clients do not retry: against
   inferno, the SDK's two retries turned a refused connection into 4.6 s of silence.
   Measured against a refused port on inferno, the user hears the failure 1.75 s after
   releasing the key. The pipeline's start is also announced: "hands is up", or "hands
   is back after a crash" when the last heartbeat names a pid that no longer exists and
   did not record `stopped`.
2. **Screen.** On every heartbeat, the daemon writes `~/.hands/status.json` with its pid,
   uptime, pipeline state, last audio output, number of live sessions, whether a turn is
   open, and each way in which it is running degraded (the microphone is open on no
   device; the collector has not accepted all of a signal's last few batches), each
   described in its own message. `hands status` prints this file, including every
   degradation. The running daemon is the only writer: a run locks `~/.hands/hands.lock`
   before it reads the heartbeat or starts listening. A second `hands run` on a home that
   a daemon already holds is refused at that point. It reports that hands is already
   running and the pid recorded in the lock, and it does not modify the heartbeat or the
   sockets; only its failed `hands.start` is written to the log. When TTS itself is down,
   a macOS notification is posted through `osascript`. `hands indicator` is a menu-bar
   status item that runs in its own process. `hands run` starts it in its own session, so
   neither the daemon exiting nor Ctrl-C in the terminal stops it first. It runs as long
   as the process that started it is running. After that process exits, it keeps running
   until the light changes from up; it then posts that notification and exits. A restart
   keeps the same process. The new run stops the indicator that the previous run started
   (it sends SIGTERM to the indicator's process group, which the indicator handles the
   same way as a normal exit, then sends SIGKILL to the group if it has not exited within
   one second) and starts its own indicator, running the new run's code. Once per second,
   the indicator reads and evaluates the heartbeat through `heartbeat.look`.
   `hands status`, the crash check at startup, and the hook shim use the same function. Its title
   shows one of six lights: up, not
   responding, down, refused (a start that ended before it ran, with its reason), off (stopped or never ran), and unreadable. An unreadable heartbeat
   produces a warning as prominent as one for a dead daemon. A daemon that is up but
   degraded still shows the up light; its title is a warning that lists every
   degradation, and shows whether a turn is open. A daemon with no microphone never
   reports an open turn. The indicator posts a notification when the light changes from up
   to a warning light, or when a degradation appears that the previous check did not
   show. It posts at most one notification per minute, so a loop that repeatedly stalls
   and recovers does not produce a notification every time. A change within that minute
   is held and posted when the minute ends, if the held condition still applies: hands is
   still not up, or the degradation is still present. If the daemon is already down on
   the indicator's first check, it is shown but no notification is posted. A heartbeat
   whose pid is outside macOS's `1..99999` range fails to parse, because no process can
   have that pid. A pid counts as the daemon only if the process holding it started no
   later than the heartbeat's `started_at`, with one second of tolerance for a clock that
   was set back. The process start time comes from the kernel (the `kern.proc.pid` sysctl
   that `ps` itself reads, which takes about 10 µs). The session sweep uses the same rule,
   from `hands.sessions.processes`. A pid that a later process took over, after a crash
   or a reboot, reads as down rather than as not responding. `hands tmux-status` prints
   the same title for a tmux status line, styled by light (`indicator.segment`): green for
   up, yellow for not responding or degraded, red for down, refused, or unreadable, and
   grey for off. tmux runs it every `status-interval`, and it exits with code 0 regardless
   of the result, because tmux displays whatever is printed.
3. **Log.** Every effect and every failure is written as one line in the segmented log
   `~/.hands/audit/`. The daemon and each `hands` command write to it under one lock
   (`hands.sessions.audit`). Each segment is named by its base offset. When a line would
   make the active segment larger than 32 MiB, the log rolls to a new segment, which
   starts with a `Rolled` line, and retention deletes all segments except the one just
   closed and the new one. A segment is never renamed, and once a later segment exists it
   is never appended to, so `hands log` prints the newest lines and follows the log across
   a roll without losing any lines. Each line is a value encoded in a single format: its
   type under `"type"`, its fields next to it (the same applies to nested events and
   effects), and the wall-clock time under `"at"`. `Sessions` is the only writer for the
   session side. It writes one `applied` wide event for each event or voice answer that
   the registry applies (only those that changed the registry or required an effect, so a
   tick with no changes produces no line). The event stays open until its effects are
   performed. It contains what was applied, each effect as a `Performed` record with its
   outcome and time (including `Audit` records such as `Unregistered`), and the number of
   effects of each kind. When the event is applied inside a hook post, it is a child of
   that post; a unit that an effect opens is a separate unit. A single wrapper turns every
   tool call into a `tool.run` wide event that contains its arguments and the result
   returned to the model. When the call is made in a voice turn, the event is a child of
   that turn's `tool.call` span. The context aggregators write each user turn as
   `Transcribed` and each reply as `Replied`
   (a line that hands speaks verbatim and keeps in the context is written as its own
   `Replied` after it is spoken; if it is spoken inside the model's turn, it is part of
   the model's `Replied` instead, and the model's turn runs through its tool calls up to
   the reply that answers them:
   `hands.voice.conversation.AssistantTurns`);
   the system channel writes `Announced`, recording whether it spoke or posted the
   message; each message that a session gives hands to speak without being asked is an
   `utterance` wide event, covering it from when it is received to its final outcome; and
   a loguru sink converts every error that a `hands` module logs into a `Failure`. A
   dictation is traced from the spoken words to the readback: `Transcribed`, then the
   `tool.run` of `stage_draft` with the text that hands read back, which no model
   rewords. The log only records; it never controls behavior. If the disk does not accept
   a line, the line is lost and a warning is printed to stderr. If a value cannot be
   encoded, a `Failure` line is written instead. In both cases, the draft, question, or
   tick that the line described continues. `hands log` follows the file by inode and
   offset, so if the log is moved aside, it is read from its first line.

The daemon runs in the foreground of a terminal, so a crash is visible in the terminal,
as down in `hands status`, and as a notification from the indicator. Nothing restarts it
automatically: to run it again, use `hands run`, which re-reads the session files and
announces that hands is back. A hook shim that cannot reach the socket fails visibly in
the target session, unless the heartbeat shows that hands was stopped or never ran. As a
result, a daemon that crashed or hung produces an error in every session, while a daemon
that was stopped has no effect on them. Two sources can never disagree about whether the
daemon is up: only the daemon writes the heartbeat file, and every other component reads
it.

The heartbeat is accurate at both the start and the end of a run. `hands run` writes its
first heartbeat, `pipeline starting`, before it imports Pipecat. The models load off the
event loop, and the loop keeps writing heartbeats, so a new run shows its pid within half
a second of starting, and only a loop that is actually stuck reads as not responding.
SIGTERM, Ctrl-C, the `q` key, and a failed background task all set the same quit event.
Its handler is installed before the models load, so a stop during loading does not wait
for them to finish. During shutdown, every permission hook that is still waiting is left
undecided, including one that arrives as shutdown begins, so that session's own dialog
stays in place and the daemon exits in under a second. After the socket is released, a
stop writes a final heartbeat with the status `stopped`, and a stopped daemon reads as
stopped even if its pid is later reused. A crash writes nothing more, so its last
heartbeat names a pid that no longer exists, and it reads as down. A failed background
task or a pipeline that ended on its own counts as a crash: the run raises an exception,
exits with a nonzero code, and reads as down until it is run again.

A start is recorded as one `hands.start` event (`hands.daemon.starting.Start`), timed from
when `hands run` began to when the pipeline reported that it started, so its duration is
the time hands took to become ready. It records which run it is (`pid`, `restarted`,
`after_crash`, and on a restart, what happened to the `previous_indicator`: ended, killed,
exited on its own, or already reaped), which settings took effect (the file, the
collector, the backend, server, model, and account, and the voice), and what the run
listens on (the hook socket, the proxy and its upstream, the tap, and the display route).
Each value is added when the step that determines it runs, so a start that ended early
shows how far it got; for the display, the value is the address it was bound to. No single
code body runs a start, so its event is emitted when the start ends (`wide.ended`) rather
than held open as a unit for the duration of the run. If it were held open, every server
and task that the start creates would be inside it, and every unit they ran would be part
of the start's trace. A start that is told to stop before it finishes is cancelled. A
start that cannot be completed is failed with the reason, and the reason is also printed
in its terminal; a start that anything else ends first is failed with the exception that
was raised. If a run is refused at an early check (the talk key's grant, `config.toml`),
it does not yet hold the heartbeat and writes none, so the existing heartbeat, from a
running hands or from a crash, stays in place. A restarted run holds the heartbeat from
the beginning, because its predecessor already wrote a starting heartbeat under the same
pid. If a run is refused after it holds the heartbeat (the brain's login, the voice, the brain's
start), its last heartbeat records `refused` with the reason. `hands status` and the
indicator show it, and the next start reads it as not a crash.

## Wide events

A unit of work runs inside `hands.sessions.wide.unit`, which produces exactly one
`WideEvent` regardless of how the run ends: ok, failed (with the exception it raised and
the stack frames it passed through), or cancelled `[LAW:nothing-unseen]`. Code inside the
run calls `annotate` to add a fact and `count` to add to a count that the unit declared.
It never emits anything itself. A run that fails without raising an exception calls
`fail`, and its event ends as failed with that error. A part of the run that was timed
where it happened, such as a request made by another process, is emitted with `child` as
a separate event under the unit. A declared count that the run never added to is written
as 0, so a run that did nothing can be distinguished from a run that never happened. A
unit opened inside another unit shares its `trace_id`. A task that continues after its
unit has emitted the event cannot add to that event: the attempt is refused with a
`LookupError`.

The event is sent through the `emit` function that the unit was opened with; in the
daemon, this is the audit log's `record`. The audit log is the only export point, and an
event is one more line in it. Therefore, a run's facts are added to its event, never to a
separately constructed line next to it. When a unit of work that already writes such a
line is moved to wide events, that line type is deleted, as was done with `DeltaRead`.
Each event has its own `span_id`, and a unit opened inside another unit records the outer
unit's `span_id` as its `parent_id`, so all of a run's units form one trace.

When `[telemetry] collector` is set, the export point also sends each event to that
OpenTelemetry collector over OTLP/HTTP, with `service.name=hands` (`hands.sessions.otlp`).
Each event is sent twice: as a span, for the trace store, and as a log record, for the
event store. The log record's body is the event's name, and the record carries the span's
ids, so an event's row links to its trace. The event is written to the log first,
regardless of what happens with the collector, because hands reads the log back
(`catch_up`, `hands log`): the log's contents never depend on the network. Events are sent
in batches, and each signal is sent from its own thread, so a slow or unavailable
collector does not delay any unit of work, and neither signal is delayed by the other.
Each batch sent for each signal is recorded as an `Exported` line that names the signal,
lists each event by its span id, and records how long the send took. If the collector did
not accept the batch, the line records why. If the collector accepted it with a warning,
the line records the warning `[LAW:nothing-unseen]`. On stop, hands waits for the batches
still queued for a single timeout in total, and records any batches it leaves unsent.
These lines are aggregated, per signal, into the number of consecutive batches that the
collector has not fully accepted (`otlp.failing`). From the third such batch onward,
every heartbeat that hands writes while up, including during startup, includes this as a
degradation that names the signal and when the failures began. (The reason each batch was
not accepted is recorded on its own line, so a failure with differently worded error text
is not reported as new.) The degradation remains until a batch is accepted or the
collector is unset. The `hands.start` event names the collector. Only `hands run` sends to
it. Every other command except `hands tmux-status` is recorded as one `hands.command`
event. `hands tmux-status` is excluded because it reads the heartbeat, tmux runs it every
`status-interval`, and it would fill the log with copies of the same event. The event is
written by the dispatcher that those commands go through (`hands.daemon.cli.commanded`),
and it contains the command, the home it ran on, its parsed arguments, its exit code, and
its duration. It ends as failed if the command exits with a nonzero code, and the unit of
work the command ran, such as `plugin.render`, is in its trace. These events are written
only to the log: Claude Code waits for `hands plugin` before a session starts, and `hands
status` must respond regardless of the config, so no command waits on a collector or reads
a config.toml to find one. hands specifies only the collector; the homelab determines
which stores are behind it.

## Endurance

The intermediary talks with you for hours, and its own context is the only part of the
system that grows. Two mechanisms limit it, and both were implemented at the same time as
the feature they limit (failure mode 5).

Pipecat's context summarizer, configured with `LLMAutoContextSummarizationConfig`,
compacts the conversation when it exceeds a token threshold and keeps the recent turns
verbatim. This handles eviction. Because every narration, every tool call, and every user
transcript is also a line in the audit log, nothing that was evicted is lost:
`catch_up(minutes)` replays what was said while you were away, and the brain retrieves the
rest with its own Bash tool and its `hands:recall` skill, by running `hands recall
WORDS...` (`hands.sessions.recall`). That command prints, one line each, what you said,
what hands said, what was typed into a session, and how a session's permission request was
answered, keeping the newest lines that contain every word. hands adds no tool for this.
Each recall is logged as a `memory.recall` event, with the number of lines it read and the
number of moments it matched. The log is the long-term memory, and the context window is
the working memory. The same pull-not-push rule that applies to session transcripts
applies to the intermediary's own history.

## Configuration: one file, parsed once

`config.toml` in the home directory (`~/.hands`, or `HANDS_HOME`) is read once, by `hands
run` before its first heartbeat, and parsed into a frozen `Config` (`hands.daemon.config`).
An edit takes effect the same way as any other change on disk: the run restarts with it.
The watcher compares the file with the bytes that the start read. An edit that parses ends
the run in the same way as the restart signal `[LAW:single-enforcer]`, and is recorded as
`SettingsEdited` before the next `hands.start`, which records that the run was
`restarted`. An edit that does not parse, or one that the start's own `backend` check
rejects for the brain's login, is recorded as a refused `SettingsEdited`, and the run keeps
its current settings. The file is in the home directory, together with everything else a
hands instance stores, so a second home directory is a second hands instance with its own
settings. A missing file or a missing key uses the default. A misspelled key, or a key that
the selected variant does not use, stops the start and reports the key. hands stores no
secrets: the brain's login is in its own config directory. The spike's `HANDS_LLM*` and
`HANDS_WHISPER_MODEL` variables have been deleted, so there is one source
`[LAW:one-source-of-truth]`. `HANDS_HOME` is not a setting: it specifies where the settings
are, and it is the only value that can be passed to a hook, which Claude Code runs with no
hands arguments.

The settings are limited by `[LAW:no-mode-explosion]`: each config field specifies a
variant or a number. There are no boolean feature flags. A field that would be a flag is
either a variant with a real alternative or it does not exist. The fields:

| key | what it names |
|---|---|
| `[llm] backend` | `claude`, the brain: the default, and the only harness hands supports so far; a model API called with an API key is never a backend |
| `[llm] model` | the model, one of the four that hands offers |
| `[telemetry] collector` | the OTLP/HTTP address of the OpenTelemetry collector that each wide event is also sent to |
| `[talk] personality` | the tone hands uses, described in the user's own words: a section of the conversational model's instructions, placed last before the closing text, so it sets tone and never changes what hands does |
| `[talk] wake_word` | the phrase the `wake word` trigger listens for: one of openWakeWord's built-in phrases (Hey Jarvis, the default, Hey Mycroft, Hey Rhasspy, Alexa), or, with `wake_word_model`, the phrase that the user's model was trained on |
| `[talk] wake_word_model` | the full path of an ONNX model that the user trained with openWakeWord; it is checked when the user switches to that trigger, which is also when openWakeWord's built-in models are downloaded |

The voice is not a setting: the user selects it by voice while hands is running, and it is
stored in the `voice` file in the home directory. The model is a setting that the user can
also select by voice. hands has no session id, so a selection that applies to hands itself
uses its own tool (`use_model`), never `send_command` sent to a session. The selection is
an edit to `config.toml` (`OwnModel`): `[llm] model` is set, and every other line stays
unchanged. It is validated by the same `backend` check before it is written, so a
rejection is reported while the user is present, and it is written only after the user has
heard hands announce that it is switching (`Player.heard`). The run then restarts with it,
as with any edit. `hands run --model` specifies a model that overrides the file's model for
that run and every restart of it (each re-exec passes the flag again). It is applied where
the file is parsed, through the same offered-model check, and the watcher evaluates edits
with it applied, so while the flag is in effect, an edit that changes only the file's model
is not treated as an edit. A selection by voice is rejected, and the rejection names the
flag. The permission timeout is declared in the hook config, from which `hooks.json` is
generated.

## Stack

Pipecat provides every component the pipeline needs: an in-process pocket-tts service, a
local audio transport over PyAudio, a segmented STT service for upload APIs, the
SmallWebRTC transport, the Silero VAD, and `TTSSpeakFrame` for the system channel.
Verified against the installed Pipecat 1.10 on 2026-09-12. The model's stage is hands' own
brain stage (`brain/stage.py`), not one of Pipecat's LLM services.

pocket-tts is MIT-licensed, has 100M parameters, is CPU-only by design, reports no word
timings, and streams its output. Measured on this Mac, first audio arrives 87 ms after the
text arrives, and it runs at about 5.6x real time. Whisper runs large-v3-turbo on MLX, in
hands' own process (`hands.voice.transcription`), and is loaded when the pipeline is built
(the `whisper.loaded` event): the transcript is ready about 0.3 s after the key is
released. LowTalker (~/code/low-talker) serves the same weights from the Neural Engine and
took 0.6 to 0.7 s for the same holds, both by upload and through its Realtime socket
(2026-10-04, hands-dictation-2bs.kpm), so hands keeps its own. The model is the brain:
hands' own Claude Code instance, running through hands' proxy. Measured on 2026-09-12, full
voice-to-voice with Qwen3-30B-A3B on inferno (since retired): a turn with a tool call had
first audio 4.3 s after key release; a plain turn, 1.4 s.

Python, `uv`, pyright strict. State types are discriminated unions: frozen dataclasses
with a `Literal` kind or a union of frozen dataclasses.
