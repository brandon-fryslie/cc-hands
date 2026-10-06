# Architecture

`hands` is one Python process, run in a terminal, built from four packages in a strict
downhill order: `daemon` → `voice` → `sessions` → `core`. Every decision below serves
one of the nine needs in [needs.md](needs.md); the work that delivers it is planned in
[features.md](features.md). The catalogue of what goes wrong, and the rule each entry
produced, is [failure-modes.md](failure-modes.md).

The design has one organizing idea: **the pure core decides, the edges act, and every
fact has one home.** State, events, and effects are typed unions. The reducer that turns
an event into new state and a list of effects has no I/O, so every lifecycle transition
is a unit test with no mocks. The adapters that perform the effects are thin, and there
is exactly one of each: one replies to blocked hooks, one owns the speaker, and one -
once it is built - types into sessions, through the fritter that wrapped them.

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

The audio leg on the left is the Pipecat pipeline the spike already runs. The
Claude Code leg on the right is the sessions package. They meet only through `core`'s
types: hook events enter the pipeline as frames, and the pipeline's tools call the
sessions API.

## Packages and the direction of dependency

**`core`** holds the domain: the state types, the reducer, the step recognisers, the
spoken-form transform, the narration tree, the attention policy, and the coalescing of
pending speech. It imports the standard library and nothing else. A test asserts that:
`core` must not import `pipecat`, `subprocess`, `socket`, or `asyncio` streams. That
test is what makes `[LAW:effects-at-boundaries]` a property of the package rather than
a habit of its authors. Everything in `core` can be exercised with plain values and no
mocks.

**`sessions`** is every edge on the Claude Code side: the unix socket the hook shims
POST to, the session files the shims write, Claude Code's own status files, the JSONL tail, the git delta reader, the
process-liveness check, the audit log, and the wire proxy a Claude Code process reaches
the API through (`ANTHROPIC_BASE_URL`), which passes every byte unchanged and reads a
copy into `core.wire`'s typed events, one `Exchanged` audit line per request. The
working sessions' exchanges reach the same observer by another path, the tap
(`sessions/tap.py`): each session's fritter is its proxy, forwards its requests to the API it
still names, and copies each exchange to `<hands home>/wire.sock`, read into the same values, so a session
never waits on the daemon and has no route through it. The one
listener joined to it (the brain's stage, with the keeper of the brain's context) routes
each request: sent with the changes hands makes to it (none, hands' tail appended to its
newest message, old tool results as one line each, a compaction's prompt replaced), or
held and answered by hands without reaching the API. It
parses hook input once at the socket into a `HookEvent` and rejects anything it does
not recognise with a logged error and a non-2xx reply `[LAW:parse-dont-validate]`. It
exposes two things upward: an async stream of events, and a small API the tools call.

**`voice`** is every edge on the audio side: the Pipecat pipeline, the gate and the
edges that drive it, the audio transport, the STT, LLM, and TTS services, and the tool
functions. It consumes the sessions event stream and turns each event into one of the
three speech channels below. It never imports from `daemon`.

**`daemon`** is the composition root: it parses the config file into a frozen
`Config`, builds the sessions package and the voice package from it, runs both under
one supervisor, publishes the heartbeat, and provides the `hands` CLI. Only `daemon`
reads the config file `[LAW:one-source-of-truth]`, and the environment is read only
where a process starts: the daemon, and the two modules of `sessions` that Claude Code
runs as processes of their own, the hook shim and the attention tool. Every other module
is handed what it needs from it (`tests/test_environment.py`).

The arrows point one way and never loop `[LAW:one-way-deps]`. When a lower package
seems to need something from a higher one, the missing thing is a type that belongs in
`core`.

## The types are the design

These unions are the seams. They are given here as Python because the doc that
describes them in words would drift from the code that defines them, and the code is
the map the type checker redraws on every run `[LAW:types-are-the-program]`.

```python
SessionId = NewType("SessionId", str)
Uuid      = NewType("Uuid", str)            # a JSONL record id
NarrationId = NewType("NarrationId", str)
SegmentId   = NewType("SegmentId", str)
Instant   = float                           # monotonic seconds

# What the shim records at SessionStart. The name is not here: it is the newest
# custom-title record in the transcript, read when a listing needs it.
@dataclass(frozen=True)
class Membership:
    id: SessionId
    pid: int                  # the shim's parent, which is the claude process
    cwd: Path
    transcript: Path          # the JSONL, from the hook payload
    fritter: Path | None      # the socket to type into it; None if nobody wrapped it

@dataclass(frozen=True)
class Session:
    membership: Membership
    state: SessionState       # what Claude Code says it is doing
    mode: PermissionMode      # from each hook payload that carries it
    turn: Turn                # which turn it is, as hooks and records say
    dialog: Dialog | None     # the dialog it is at, as its hooks say; an idle ends it

# An ended session has no status, turn, or dialog: its last turn was told and its held
# hook let go as it ended. It starts again as a new Session.
@dataclass(frozen=True)
class Gone:      membership: Membership
Known = Session | Gone        # what the registry holds per session

# Only a status read moves a session between these: whether it runs is Claude Code's
# word, never inferred from a hook or a record.
SessionState = Unreported | Idle | Running
@dataclass(frozen=True)
class Idle:      status: Idle | Shell; stamp: Stamp; after: PromptId | None  # one idle period; Shell: a background shell runs
@dataclass(frozen=True)
class Running:   status: Busy | Waiting | Unknown; stamp: Stamp; idled: Stamp

# Each fact about the turn lives on the phase it is true in.
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

# Everything Claude Code stops for arrives through the same PermissionRequest hook.
Blocker = Permission | Question | Plan
@dataclass(frozen=True)
class Permission: tool: str; input: Mapping[str, object]; suggestions: Sequence[object]
@dataclass(frozen=True)
class Question:   questions: Sequence[AskedQuestion]
@dataclass(frozen=True)
class Plan:       text: str             # a finished ExitPlanMode is PlanApproved: its input no longer carries the plan

# What is typed into a session. The variant decides the escaping,
# so there is no "if it starts with a slash" anywhere: Text always escapes a leading sigil,
# Command never does, Key is a named chord and carries no text at all.
Input = Text | Command | Key
Keystroke = Literal["escape", "enter", "ctrl_c", "ctrl_u", "up", "down", "tab", "shift_tab"]

# The reducer's whole vocabulary of effects. Adapters perform these and nothing else.
Effect = Reply | Type | Speak | Narrate | Note | Play | Summarise | Snapshot | Audit
@dataclass(frozen=True)
class Reply:    request: RequestId; reply: HookReply
@dataclass(frozen=True)
class Type:     session: SessionId; socket: Path; pid: int; input: Input  # through fritter
@dataclass(frozen=True)
class Speak:    text: str; priority: Priority                # straight to TTS
@dataclass(frozen=True)
class Narrate:  event: Narration; priority: Priority         # LLM, run_llm on
@dataclass(frozen=True)
class Note:     event: Narration                             # LLM context, silent
@dataclass(frozen=True)
class Play:     narration: NarrationId; segment: SegmentId   # straight to TTS, bookmarked
@dataclass(frozen=True)
class Summarise: narration: NarrationId; steps: Sequence[Step]; expand: SegmentId | None
@dataclass(frozen=True)
class Snapshot: session: SessionId; cwd: Path                # where the turn's repository stands as it opens
@dataclass(frozen=True)
class Compare:  session: SessionId; again: bool               # what it changed, read when the turn stops
@dataclass(frozen=True)
class Audit:    record: AuditRecord

Priority = Literal["blocking", "result", "fyi"]

# A narration is a tree of spoken segments. The top level plays first; each segment
# opens into children, and every segment names the records it summarises, so
# "that part" and "more on that" are lookups.
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

# Where playback is, and where it was when you cut in. Resume pops the stack.
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

The block above is the target. `hands.core.effects` defines less today:
`Effect = Audit | Reply | Heard | Story`, where `Heard = Speak | Narrate` carries
permission announcements and requests and `Story = Summarise | SessionGone` carries
finished turns and sessions gone. Today's `Summarise` holds a session, the prompt id of
the turn that ended, and the reply its `Stop` hook carried, not a narration id and steps. `Type`, `Note`, `Play`, and `Snapshot`,
and the segment, narration, and playback types, are planned.

Two things are deliberately absent. There is no `Session.last_seen` timestamp,
because silence measures nothing; liveness is the pid. And there is no queue of unsent
inputs, because Claude Code has its own input queue and the daemon must not keep a
second one `[LAW:one-source-of-truth]`.

## The reducer and its effects

```
reduce(state: Registry, event: Event) -> tuple[Registry, list[Effect]]
```

`Event` is the union of parsed hook events, steps from the transcript tail, tool calls
from the intermediary, ticks from the one clock, results of `Summarise` and `Snapshot`,
playback reports from the output transport, and liveness reports. The reducer is a
pure function `[LAW:effects-at-boundaries]`: it never reads a file, checks a process,
or looks at a clock. When it needs the time it has already been handed one in a
`Tick`. When it needs a summary it emits `Summarise` and receives the segments back as
an event. Today nothing comes back: `Summarise` is emitted for a live session's `Stop`,
for an interrupt, and for a turn that a prompt naming another turn finds still running,
and the narrator reads the turn and hands it to the intermediary to say, without returning
to the reducer.

The adapters live in `sessions` and `voice` and each performs one effect kind: `Reply`
writes to the blocked shim's socket connection, `Speak` becomes a Pipecat `TTSSpeakFrame`,
`Narrate` and `Note` become
`LLMMessagesAppendFrame` with `run_llm` on or off, `Play` sends a segment to TTS
through the player, `Summarise` hands the turn to the intermediary, `Snapshot` records or diffs the
target's git state, `Audit` appends one JSONL line. An
adapter that fails raises; the supervisor logs it and the failure is spoken through
the system channel. Nothing is retried silently and nothing falls back
`[LAW:no-silent-failure]`.

That block is the design, not the code. `core/effects.py` has eight of those nine today -
`Audit`, `Reply`, `Type`, `Speak`, `Narrate`, `Note`, `Summarise`, `Snapshot` - plus
`SessionGone` and `Compare`, which the block above leaves out. `Play` is unbuilt. `Input` is
`Text` alone so far; `Command` and `Key` are `hands-keyboard-gxr.i5n`. `Type` is emitted by
`core.drafts.decide` for a send rather than by `reduce`, and `Sessions.draft` performs it:
the draft leaves the registry the moment the send is decided, so it is sent at most once,
and the send's answer is whether fritter typed it.

Because every transition is `reduce` on values, the test suite for the session
lifecycle is a table: state before, event, state after, effects. There is no pipeline,
no socket, and no keyboard in those tests.

## Time has named owners

Correctness never depends on incidental ordering `[LAW:no-ambient-temporal-coupling]`.
Each timing fact has one owner.

| What must be ordered | Owner |
|---|---|
| When a user turn starts and ends | the gate, through the key position |
| Which utterance plays next, and that two never overlap | Pipecat's output transport |
| Where a narration resumes after you cut in | the player's bookmark stack, from the segment the output transport was playing |
| That a session's end is heard after its last turn | the story queue: `Summarise` and `SessionGone` wait in one queue, and the narrator tells each in turn |
| When a permission deadline warns and expires | the reducer, from `Held.deadline`, driven by one `Tick` source |
| Whether text typed mid-turn is queued or lost | Claude Code's own input queue, measured to queue it |
| When the daemon is up, and restarting it | you, with `hands run` in a terminal |

Deadlines are data. The `Held` dialog carries the instant it expires and whether
the warning has been spoken. A single ticker sends `Tick(now)` once a second; the
reducer compares, and emits `Speak("ten seconds on that permission")` exactly once,
because the transition from `warned=False` to `warned=True` is a state change, not a
timer callback. At the deadline it emits `Reply(deny)` for a permission and says so;
a question, which silence cannot answer, is withdrawn instead and left to its dialog,
where the user may be answering it at the keyboard. Its dialog is then `LetGo`: still
waiting, now on the keyboard alone, until the answered call comes back. The ticker's
period only bounds how late a deadline is heard; no correctness property depends on
a `sleep`.

The deadline itself comes from one number. `hands.sessions.hookconfig` declares the
`PermissionRequest` hook's timeout (90 seconds) in the plugin's `hooks/hooks.json`; the shim
waits on the daemon that long for that hook alone, and the daemon denies 5 seconds
earlier, so the deny reaches Claude Code before Claude Code kills the hook
`[LAW:single-enforcer]`.

Claude Code queues messages submitted while a turn is running and shows them with
"Press up to edit queued messages". Measured on 2.1.270: text pasted into a working
session and submitted lands in that queue and runs when the turn ends, so a send to a
working target is an ordinary send and the daemon holds nothing. A permission dialog
is the exception: it swallows pasted text and takes the Enter as "Yes". So when the
drafts are sent, a send to a session whose status says `waiting` is refused, the draft
stays staged, and the user hears why.

The workspace-trust dialog swallows a paste the same way, measured on 2.1.278, and needs
no rule of its own: Claude Code runs no hook until a startup dialog is answered, so a
session at one has no membership and nothing can be sent to it (`hands-harness-5nb.xw8`,
measured on 2.1.283).

### Typing into a session

A session is typed into through **fritter** (`fritter/`), which runs its `claude` on a
pseudo-terminal and listens on a unix socket beside it. A wrapper and not synthetic key
events, because a keyboard types into whatever has focus, and the requirement is a
session driven with the display asleep: no window, no grant, no focus.

fritter publishes its socket's address to the process it wrapped in `FRITTER_SOCKET`.
The hook runs as a child of that process and inherits it, so the address reaches
`Membership.fritter` without either side deriving a path from a pid. A session started
outside fritter has no address, and a send to it is refused by name rather than written
into nothing. Nobody has to remember to wrap one: `hands install-fritter` puts a `claude`
in `<hands home>/bin` that runs every interactive claude under fritter, and runs a pipe, a
script, a subcommand, or `claude -p` as the real claude with no address (`hands.sessions.wrapper`). Inheritance also hands the address to a session started from inside a
wrapped one, so an address alone does not say which session it reaches: every request
names the session's `Membership.pid`, and fritter refuses one that is not the process it
wrapped.

A send is typing: the text, pasted, and Return, in one write, as someone at the keyboard
would type it. hands decides *whether* a session may be written to, from state fritter
cannot see, and what a leading `/` means is `Input`'s business; fritter types the text it
is given. Text the person at the keyboard had half-written stays in front of it, as it
would under their own paste.

`fritter/README.md` holds the protocol and what was measured.

## Four ways to reach the ear

Every event that reaches the pipeline takes one of four routes, and the route is
chosen by a table, not by code that looks at the event `[LAW:dataflow-not-control-flow]`.

- **Speak.** Text goes straight to TTS as a `TTSSpeakFrame`. No model call, no
  interpretation, no latency beyond synthesis. Used for facts a template can say:
  "auth-refactor finished", "ten seconds on that permission", "cc-hands is gone, its
  terminal closed", "the language model is unreachable". This channel is also how the
  daemon reports its own failures, which is why it must not depend on the LLM.
- **Play.** A narration's segments go to TTS one at a time through the player,
  already in spoken form. There is no model call at playback, because the
  summariser did that work first. The player knows which segment is on the speaker,
  so an interruption leaves a bookmark, and each played segment is appended to the
  intermediary's context as a note so it can answer about what you heard. Used for
  a turn's results, progress while a session works, and a subagent's report.
- **Narrate.** The event is appended to the intermediary's context with `run_llm`
  on. The model interprets and speaks. Used when the content is a conversation
  turn: a permission request, a question from `AskUserQuestion`, a plan.
- **Note.** Appended with `run_llm` off. The model knows, and says nothing until
  asked. Used for context that changes what a later answer should say: a focus
  change, a subagent finishing, a session going idle.

Today no table chooses; two queues stand in for it. `Heard` carries permission
announcements as `Speak` and permission requests as `Narrate`,
relayed as soon as the reducer emits them. Relayed is not heard: everything hands says
unprompted of the sessions, under either telling, reaches the floor (`voice/floor.py`) ahead
of the user aggregator as a value, a `Pending` (`core/pending.py`), not yet a frame. From the
press that opens the user's turn until that turn is sent it waits there and follows the user's
words; what arrives with no turn open is let go at once, the same way. Either way the floor
makes frames of it only as it lets it go, after `coalesce` (below), which says for each thing
it tells where each thing it holds stood, so what it drops is known by none naming it.

Each thing a session gives hands to say unasked is one `utterance` wide event
(`voice/utterance.py`), opened as the relay or the narrator hears it and emitted at its fate:
`noted` (the route, the delivery, or a turn or burst with nothing new kept it from being said),
`dropped` (no longer so by the time it was to be told: out of date as the floor let it go, or
progress of a turn that ended before its summary), `silent` (a note to the model's context,
settled as it is sent), `played`, or `cut`. Its facts are what was
heard and from which session, what decided its route, how long the floor held it (`held_ms`),
what it was told as and how many things heard were folded into that telling, and
`first_audio_ms`, from heard to the first audio the speaker wrote of it. The last three fates
are read off the output transport by the `Audible` observer: what says an utterance is sent
between an `Uttering` frame, dropped by a barge-in with the words it leads, and an `Uttered`
frame, kept through one, so every utterance handed on reaches the output transport closed, in
order with its audio. The brain's stage sends them around what it takes from hands' lane, leads
what its turn still has to say with an uninterruptible `Resumed` after a barge-in the turn goes
on through, and fails each utterance of a telling the brain failed; the brain's turn telling an
utterance is its child, in its trace.
`Story` carries finished turns and sessions gone in one ordered
queue, because an end spoken at once was heard before the last turn it ended. Every
finished turn is read once into a `News` (`voice/narrator.py`, `recount`): the session's last
words, what its record adds, and what it is waiting on. `speech.told` is the one place it
becomes what the model is handed, under the session's name as it is when told, for the model
to say in its own words. How that one summary reaches the user is its `Delivery`: as a turn of the intermediary's own
— an `LLMMessagesAppendFrame` with `run_llm` on for an API model, a `Narrated` frame for
the brain — when finished turns are set to be told, briefly or in full, or when the session is
watched (`Spoken`); otherwise it is held (`Withheld`, saying why) and `tell_turn` hands it
to the model when the user asks. Each session's last summary is held in `Recounts` either
way, as its `News`, and its utterance names its delivery. Nothing of a turn is said as
written past the model, and nothing is said of a session that sits at its prompt.
`Heard` also carries a mode change as a `Note`, which enters the intermediary's context
with `run_llm` off. Each session has an overlay, `normal`, `watched`, or `muted`, one file per
session under `~/.hands/overlays` (`hands/core/attention.py`), which the narrator reads at
every finished turn. What hands says unprompted is one control, `Attention`
(`hands/core/attention.py`), kept in `~/.hands/attention.json`: a level for each kind it
says unasked — finished turns, the focused session's progress, a session ending — and
quiet, which holds all of them without touching their levels. `delivery`, `progress_route`,
and `ended_route` are the tables over it and the overlay; a muted session's turn is held
whatever is set, until the user asks for it through `tell_turn`. The user sets the overlay
by voice with `set_overlay`, and what is said unprompted with `attention` or
`/hands:attention`. A muted session's permission requests,
questions, and plans are still narrated: held unsaid, each would wait out its deadline and
be refused. Progress has its row of the table today, `attention.progress_route` over what
is set, the focus, and the overlay (see "Streaming"); the player and the table over every other event kind
below are planned.

The routing table is a value in `core`:

```python
Route = Literal["speak", "play", "narrate", "note", "drop"]
DEFAULT_POLICY: Mapping[EventKind, Route] = {
    "stop": "play", "progress": "note", "blocked": "narrate", "subagent_stop": "note",
    "gone": "speak", "session_start": "note", ...
}
```

The per-session overlay is a second table over the first: a muted session's `play`
becomes `note`, and its `narrate` stays, since a session that asks needs an answer. Adding a new event kind is a new row, and adding an overlay
value is a new column `[LAW:one-type-per-behavior]`.

Pending speech is ordered as the floor lets it go, and nothing starts while the key is
down. `coalesce` (`core/pending.py`) is pure: it drops what is no longer so, read off the
live sessions as the floor lets go: a request answered at the keyboard while you talked and
a deadline counted down on it, by the held dialog's request id, and progress of a turn that
ended meanwhile, whether or not its ending was told; it folds one session's
finished turns into one `Finished` where the first stood, whose headline covers them all
("finished 3 turns") and whose tellings keep their narration parts and so their record ids,
so three `Stop`s that arrived while you were talking start with one sentence, not three, and
a turn that could not be read stays between the turns it came between, and a turn told again as it
went on past its `Stop` is still one turn; and it orders the
rest `known` (notes and the briefing, never spoken) before `blocking` (what a session asks)
before `result` (finished turns) before `fyi` (a session gone), arrival order within each.
A session's own story keeps the order it happened in: what it told before something sooner
is told with that sooner thing, so its next turn's request is never heard ahead of the turn
before it. A folded telling shares one `REPLY_SHOWN` bound among its turns. A `Pending`'s priority is read off its variant, never stored beside it. This
queue is not the player's bookmarks: resuming replays a bookmarked sentence and never
re-enqueues a telling. Each item is a transition, never a state, so nothing is announced
twice `[LAW:one-source-of-truth]`: a request is narrated, warned of, and expired by its
request id, which the daemon mints per hook delivery, so a second delivery is a second request; a turn's `Summarise` asks the
tail only for the records it has not told. The same event heard twice is said once
(`tests/test_reducer.py`).

## Hooks carry the moment; the transcript carries the record

Hooks say when things happen: a session starts, a prompt is submitted, a turn stops, a
permission is needed, a line of Claude's text is ready. They fire at the moment, and
the blocking one is the only way to answer a permission. What a turn did, with the
record id of each step, is in the transcript, which Claude Code appends to while the
turn runs, so the daemon tails it rather than hooking every tool call. Every hook
input carries `session_id`, `transcript_path`, `cwd`, and `hook_event_name`. Of the
events hands subscribes to, `UserPromptSubmit`, `Stop`, `PermissionRequest`,
`PostToolUse`, and `PostToolUseFailure` carry `permission_mode` as well, and `SessionStart`, `Notification`, and
`SessionEnd` do not (verified live on 2.1.281). That mode is the session's mode: each
hook that carries one sets it, `list_sessions` says it, and a change reaches the
intermediary as a `Note`. A hook fired inside a subagent carries the subagent's mode
and an `agent_id`, and sets nothing. Shift-tab fires no hook, and the transcript writes its
`permission-mode` record only as a prompt is sent, so a mode changed at an idle
prompt is heard at the session's next prompt, and one changed mid-turn at its next
tool call. The
event-specific fields below were read out of the 2.1.263 bundle. Payloads captured from
2.1.270 on 2026-09-14 carry no session title, so a session's name is the newest
`custom-title` record in its transcript. Hands sets that name itself: a `UserPromptSubmit`
hook's reply may carry `hookSpecificOutput.sessionTitle` (2.1.286), which Claude Code writes
as a `custom-title` record and shows on the terminal's tab, exactly as `/rename` does
(verified live on 2.1.286). The latest name wins, whoever set it.

After each turn a session finishes, hands asks the summariser's model whether the
session's name still fits, showing it the name and the turn's closing reply
(`hands.voice.naming`). A new name, three words at most, waits in `Names` for that
session's next prompt. Each judging is one `name.judged` event: the name before, the reply, the
name decided, and what came of it (`judged`: renamed or kept, or, failing the event, unread, failed, or
refused). A hook can set a title only at a start or a prompt, so a session
shows its new name from its next prompt on. Spoken, a session is its project and then its
name, as one identifier: "cc-hands, naming fix".

| Event | Payload fields |
|---|---|
| `SessionStart` | `source`, `agent_type`, `model` |
| `UserPromptSubmit` | `prompt`, `prompt_id` |
| `Stop` | `stop_hook_active`, `last_assistant_message`, `prompt_id` |
| `PermissionRequest` | `tool_name`, `tool_input`, `permission_suggestions` |
| `PostToolUse`, `PostToolUseFailure` | `tool_name`, `tool_input`, `tool_use_id`, and the response or the error |
| `Notification` | `message`, `title`, `notification_type` in `permission_prompt`, `idle_prompt`, `auth_success`, `elicitation_dialog` |
| `SubagentStop` | `agent_id`, `agent_transcript_path`, `agent_type`, `last_assistant_message` |
| `MessageDisplay` | `turn_id`, `message_id`, `index`, `final`, `delta` |
| `SessionEnd` | the common fields |

No hook fires when the user interrupts a turn with Escape or Ctrl-C (2.1.281): no
`Stop`, no `PostToolUse` for the tool it cut off, and no `idle_prompt` afterwards.
Claude Code writes a user record instead, `[Request interrupted by user]`, or
`[Request interrupted by user for tool use]` when a tool was running, carrying the
`promptId` of the turn it stopped, which is the `prompt_id` that turn's
`UserPromptSubmit` carried. The tail reads that record as the `Interrupted` event.
Claude Code sets the session `idle` before it writes the record (37 ms before, 2.1.283),
and that status is what moves the session to its prompt (see Sessions, below); the record
says the turn Claude was answering is over, and is what its telling waits for. An
interrupt that flushes a queued message names that message's id instead, and the turn
goes on under it. Only the prompt names the turn: a background
subagent's hooks keep the `prompt_id` of the turn that started it after that turn is over.

Only Claude Code's status moves a session between running and at its prompt. Hooks and
records say which turn it is and what it did: a turn is `Opened`, `Untold` once Claude
Code's idle ends it, and `Told`. A turn opens from a prompt heard with no turn open, which
marks it, whether or not the busy that prompt set has been read yet. What is submitted
while a turn runs, a queued message or a task's notification, fires `UserPromptSubmit`
with the running turn's `prompt_id` and leaves the status as it was (2.1.283), so it is in
the turn it names. Claude Code gives a prompt an id of its own only at its prompt, so a
prompt under another id means the open turn is over, even when the `idle` between them
was set and set again inside one status read (93 ms apart, 2.1.283): that turn is told,
compared while the new prompt's hook holds Claude Code, and the new turn is marked. Any
other id a record carries while a turn is open is a flush's, taken seconds before Claude
answers under it, and joins the ids the turn goes by. A `Stop` carries
the `prompt_id` of the turn it ends, which is how that turn is found in the tail, and it
ends only an open turn that goes by that id, so a `Stop` applied late never ends the turn
after it. A `Stop` with no turn open tells a turn hands never had open, such as the one a
session was in when it was attached, or the last one stopping again after another Stop
hook blocked its Stop, and ends nothing of a last turn told already. The session runs on
past a `Stop` until Claude Code says it is idle. A turn a background task's notification opens fires
`UserPromptSubmit` with an id of its own, as a typed prompt does.

The reply a `PermissionRequest` hook may give is printed on its stdout as
`{"hookSpecificOutput": {"hookEventName": "PermissionRequest", "decision": ...}}`,
where the decision is `{"behavior": "allow", "updatedInput"?: object}` or
`{"behavior": "deny", "message": string}`; empty output decides nothing (read out of
the 2.1.270 bundle, and verified live: an allow runs the tool, and the agent reads a
deny's message as the tool's error). Permission
prompts, plan approval, and `AskUserQuestion` all arrive through this one hook, which
is why `Held.on` is a union of three and the answer path is one adapter. A question
is answered by allowing it with its own input and an `answers` object added, each
question's text keying the label chosen or the user's own words, several labels
joined with ", " — the shape Claude Code's own dialog answers with (read out of the
2.1.280 bundle, and verified live: the agent went on with the answers given by voice).
A plan is approved as its own dialog approves it: allow with an empty `updatedInput`,
so the plan is read from its file as the user left it, and `updatedPermissions`
holding the mode to leave plan mode for — nothing, which returns the session to the
mode it had before it planned, bypass or auto included, or one `setMode` to
`acceptEdits` or `default` when the user names one. Claude Code ignores an allow
without `updatedInput` for a tool that asks the user something, and shows its dialog
instead. Keep planning is a deny, and the agent reads its message as the feedback
(read out of the 2.1.281 bundle, and verified live). A plan that
runs comes back through `PostToolUse` without its text, as `PlanApproved`.
An answer that does not fit what was asked — the wrong number of answers, a plain
allow to a question, which would run it unanswered, a plain allow to a plan, or a
plan's feedback sent against a tool's request — sends nothing, and the session still
waits.

The documented nine events are not the real set. There are 33:

```
ConfigChange CwdChanged DirectoryAdded Elicitation ElicitationResult FileChanged
InstructionsLoaded MessageDisplay Notification PermissionDenied PermissionRequest
PostCompact PostModelSwitch PostToolBatch PostToolUse PostToolUseFailure PreCompact
PreModelSwitch PreToolUse SessionEnd SessionStart Setup Stop StopFailure SubagentStart
SubagentStop TaskCompleted TaskCreated TeammateIdle UserPromptExpansion
UserPromptSubmit WorktreeCreate WorktreeRemove
```

The daemon subscribes to the events in the table. Each hook posted, to the socket or the display route, is one `hook`
wide event, open as long as the hook holds its session: which hook and session, and the branch it was answered by — a
Stop `decided` or `let go` at the hold, the reply a permission was given (`cancelled` when its session closed it), the
name a prompt gave or withheld. One the daemon refuses is a failed event saying why.

`MessageDisplay` is the only
live source of Claude's text: the transcript writes a text block once, after it
finishes (checked on 2026-09-14), while `MessageDisplay` fires with each batch of
finished lines as the text streams. It is dispatched synchronously for every batch,
even to a hook that declares itself async, so it is installed as an HTTP hook, which
Claude Code 2.1.270 supports: the event is POSTed to the daemon with no process
spawned, under a short timeout. Measured on 2.1.280 on 2026-09-24, with a 40-line
reply:

- An interactive session fired 35 to 40 batches, one per finished line or two,
  0.1 to 0.5 s apart. `claude -p` fired one final batch holding the whole text.
- At most one batch is in flight. Against an endpoint that took 1 s to answer, 16
  batches arrived, each carrying every line since the last, and the last one came
  4.8 s after the turn had ended.
- The turn itself took the same time against a fast endpoint, a 1-second one, a
  refused port, and a port that accepts and never answers: 11.3 to 12.6 s every
  time. A daemon that is slow, down, or hung costs narration lines, never the
  agent's speed.

The tail, not a hook, is how
the daemon reads tool calls and their results. `PostToolUse` and `PostToolUseFailure`
are subscribed for one fact the tail would give too late to use: that a tool the
daemon is still waiting on a permission for has run, because its dialog was answered
at the keyboard. They are declared `async`, so the shim they spawn on every tool call
never holds the agent up.

**Installing the hooks.** The repository is a Claude Code marketplace
(`.claude-plugin/marketplace.json`) whose one entry, `hands@cc-hands`, has a `command`
source: Claude Code runs `hands plugin` and copies the directory it prints into its
plugin cache, at install and again once per session, so the hooks are always the
installed hands' own. The plugin's files are package data, in
`src/hands/sessions/plugin`: `.claude-plugin/plugin.json`, `hooks/hooks.json`, and the
skills. What a package cannot carry is which interpreter runs it, so `hands plugin`
(`hands.sessions.marketplace`) copies them and writes beside them the launcher
`hooks/python`, which execs the interpreter that ran `hands plugin` with `-I`: the
session's directory, where Claude Code runs the hook, and the user's `PYTHON*`
variables stay off the path, so a project's own `json.py` or `hands/` cannot stand in
for hands' modules. It writes them under the home, in `plugins/<digest>`, a directory
named by its content: staged whole and renamed into place, so sessions starting together
never hand Claude Code a half-written one. Installing the plugin installs the hooks;
disabling or uninstalling it removes them, and no settings file is edited by hands.
`hooks.json` is generated from `hookconfig` (`python -m hands.sessions.hookconfig >
src/hands/sessions/plugin/hooks/hooks.json`), and a test fails when the checked-in file
differs from what `hookconfig` declares. `MessageDisplay` is an HTTP
hook to `hookconfig.DISPLAY_URL`, a fixed loopback port, since Claude Code puts no variable in a hook's URL
(2.1.288); the daemon serves that route alone there, and does not start when the port is taken. Every other hook is exec
form: `${CLAUDE_PLUGIN_ROOT}/hooks/python -m hands.sessions.shim`, spawned by Claude
Code with no shell between, and the launcher execs, so the shim runs as the process
Claude Code spawned and its parent is the claude process. A launcher that ran Python as
its child, as `uv run` does, would record its own pid instead. The shim's home is
`HANDS_HOME`, or `~/.hands`, found the way the `hands` CLI finds it; a relative
`HANDS_HOME` is refused, since a hook runs in its session's directory.

**The shims.** Each is one process per hook: POST stdin to the daemon socket, exit.
At `SessionStart` the shim also writes
`~/.hands/sessions/<session_id>.json` with its parent pid, `cwd`, and
`transcript_path`; that file is the one record of the session's membership, written
by one writer. The file is written whether or not the daemon is up, so a daemon started
later finds the sessions already running. The hooks are installed whether or not
hands is running, so a shim that cannot reach the socket asks the heartbeat why
(`hands.sessions.heartbeat.look`, the judge `hands status` uses). A hands that was
stopped or never ran is off, not broken: the shim exits 0 and prints nothing, and a
permission request falls through to Claude Code's own dialog. So is one still
starting: `hands run` writes its first heartbeat before it imports Pipecat and serves
the socket, and it reads the session files once it does. A hands whose heartbeat
says it died, refused to start, hung, or is up but not answering, or whose heartbeat cannot be read,
makes the shim exit 1 with the socket error and the verdict on stderr, so Claude Code
shows the failure in the session where it happened rather than letting a dead daemon
look like a quiet one `[LAW:no-silent-failure]`.

A session running before the plugin was installed, or reloaded with `/reload-plugins`,
never fires `SessionStart`. So every other hook writes the file when no membership file
names the shim's parent process, and the daemon attaches the session its file names
before it applies the hook: such a session joins on whatever it fires first. Keying on
the process keeps a late hook from a session the process has moved on from (a `/clear`,
a resume) from writing a file that would outrank the new session's.

**The constraint that will bite.** Hooks run in the agent's critical path with a
timeout, and `MessageDisplay` and `SessionStart` dispatch synchronously. A shim that
waits for anything stutters the agent's own output. Every shim POSTs and returns; the
sole exception is `PermissionRequest`, where blocking is the feature. Its timeout is
declared in the hook config, and the daemon derives its default-deny deadline from
that same number, so there is one place the budget is set `[LAW:single-enforcer]`.

**The dialog and the hook race.** Measured on 2.1.270: Claude Code shows its
permission dialog at once and runs the `PermissionRequest` hook beside it, and
whichever answers first decides. An answer typed at the dialog does not end the hook;
it runs on to its own end and its output is ignored. So the daemon never learns of a
keyboard answer directly. It learns that the session moved on: the asked-about tool
finishing (`PostToolUse` or `PostToolUseFailure` with the same tool and input, or for a
question the same questions, since it comes back with the answers added), or the
next `UserPromptSubmit`, `Stop`, `SessionEnd`, or `PermissionRequest`, withdraws the
waiting reply, which prints nothing. A hook whose connection closes first ends the wait with
no reply at all, so a later voice answer hears that the request is gone rather than
that it went through. That is how a keyboard refusal arrives: answering No or Esc at
the dialog fires no post-tool hook and no `Stop`, but Claude Code kills the waiting
hook, and the closed connection releases the session (measured on 2.1.270). What is left
is a tool approved at the keyboard that is still running at the deadline: its warning
and its deny, which Claude Code ignores, are still heard, so the expiry is spoken as
what hands did ("so I told it no"), never as what happened to the tool. A permission
request from a session the daemon has not heard of joins the session first, so it is
asked aloud like any other.

## Transcripts: the live tail, and backfill

Claude Code appends to a session's JSONL while the turn runs, one record per content
block, each written once when the block finishes. While this design was written, its own session's transcript held a record 21
seconds old in the middle of a turn. So the daemon reads transcripts continuously, not
only when a turn stops.

**The recognisers** run today. `hands.core.steps.recognise` takes one `Call` — a tool
call, the record it was written in, and the result that came back — and gives the `Step`
it is, through one table keyed by tool name `[LAW:one-type-per-behavior]`. A recogniser
claims a call only when the record carries what its step asserts, so failure needs no
case anywhere: a failed edit records no patch, and is therefore an `Other` rather than an
`Edited`. `hands.core.turn.Opening` is `Asked | Notified | Commanded | Shelled`, what opened
the turn; a question Claude put to the user is the `Questioned` step.

**The tail** runs today, in `hands.sessions.tail`. From the moment the registry lists a
session, `keep_tailing` reads what that session's JSONL has gained, ten times a second,
and hands each new record to the recognisers. Nothing re-reads the file from its start:
one `Following` per session holds the byte offset, the turn that is open in it, the steps
recognised so far, and how many of them the session has been told. It also keeps up to eight
turns that ended before it, each with every prompt id its records carry, because the narrator
can be seconds behind. A turn that ended is let go of once it is told, or once a later turn
is told. A prompt id that no kept turn carries tells nothing, rather than telling another turn. Measured live on
2026-09-21, a record becomes a step 96 to 305 ms after Claude Code wrote it, median 160 ms
— the poll period plus the read, which is what sets how late a turn narrated *while it
runs* can be. A `Stop` does not wait for that poll: `tell` reads the rest of its own
transcript first, so the turn is told from the whole of what has been written.

`tell` is given the prompt id of the turn that ended and tells that turn, whatever has
been read since `[LAW:no-ambient-temporal-coupling]`: a `Stop` and the next prompt in quick
succession, or an interrupt and the next prompt, would otherwise have the new turn told in the
old one's place. It returns a `Telling`, which carries the turn's number as well as its untold
steps, and marks nothing until the narrator says it was spoken. A summary that fails is never
marked told, and the mark a slow summary makes lands on the turn it was made of, found by its
number, never on one that opened while the model was answering.

What a turn sets out to do is heard while it runs (see "Streaming" below); what it came to is
told at `Stop`.

**Backfill.** When the daemon attaches to a session that has been running for an
hour, `read_session(session)` and `read_turn(session, turn, since)` read the same file
through the same recognisers. What it hands over is `Happening = Opening | Step`: what opened
each turn as well as each step of the answer, because the steps alone say how a session
spent an hour and never what for. A `Turn` keeps the two apart, because it is summarised
as a whole against its request; a reading of a session nobody heard has no whole to
summarise, and hands them over in the one order they make sense in. Both are put into
words by the same `describe`, so a request cannot be worded one way in a turn and another
in a reading `[LAW:one-source-of-truth]`.

The whole file is folded and only then cut at the record named, so a call made before the
cut and answered after it is still one step that knows its result; read from the mark on,
that result would arrive with no call to belong to. A call the session has not come back
from is shown — it is working on something, and that is worth saying — but it is not
marked as read: a mark names a record and a reading goes on from after it, so marking a
call still in flight would spend its result on nobody, and the reader would be told the
suite was being run and never told what failed `[LAW:no-silent-failure]`. Only a call a
reading *ends* on, so that one the session carried on past cannot hold the mark behind it
for ever. The mark is the last record *every* happening of which is settled, because one
record holds a text and the call it introduces, and marking it for the text would go on
from after the whole record and lose the call's result with it. A reading answers two
facts separately — `more` for history it did not reach, `working` for a call that has not
come back — because told as one the intermediary cannot tell "read on" from "wait and ask
again" `[LAW:types-are-the-program]`. A mark this transcript never held is said rather
than read as a mark at the start, which would tell the whole session over again as though
it were new.

```python
# One variant per kind of thing a turn does. Every step names the record it came from, except
# where there is no record to name: one that carries no uuid, and the closing reply the Stop hook
# hands over before Claude Code has written it.
Step = Said | Edited | Ran | Tested | Looked | Planned | Delegated | Questioned | Other

@dataclass(frozen=True)
class Said:       ref: Ref | None; text: str                 # assistant text; never spoken as is
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
class Notified:   ref: Ref | None; text: str    # a background task's report, handed over as the next prompt
@dataclass(frozen=True)
class Commanded:  ref: Ref | None; name: str; args: str; output: str | None   # a slash command the user ran
@dataclass(frozen=True)
class Shelled:    ref: Ref | None; command: str; output: str | None           # a `!` command the user ran

Opening = Asked | Notified | Commanded | Shelled   # who opened the turn, so nothing is told as something asked that was not
Happening = Opening | Step          # what a reading of a session is made of; an opening is markable like a step

@dataclass(frozen=True)
class Changed:  path: str; added: int | None; removed: int | None   # None counts for a file git reads as binary
@dataclass(frozen=True)
class Commit:   sha: str; subject: str
@dataclass(frozen=True)
class Delta:    files: tuple[Changed, ...]; commits: tuple[Commit, ...]; patch: str

# Empty says the same thing three ways — nothing changed, no repository, git unreadable — because all three
# are the same silence to a listener, who is told what happened rather than what did not. The log says which.
```

**What the records really say**, read out of 900 transcripts on 2026-09-21, against which
three lines of the design above were wrong:

- `toolUseResult` is *absent* from about one result in five — every error writes a string
  there instead of an object, and so does every result Claude Code's own harness handled,
  such as output too large to inline. The `tool_result` block's content is the only part
  always present, so a recogniser may prefer the structured record and never require it.
- A command that fails is marked by `is_error` on the block, not by the `Error: Exit code N`
  prefix this document used to claim: one result in 15,098 began with `Error:`, while 448
  were `is_error`. A failing test run is `is_error` too, which is why `Ran.failed` is the
  command's exit status and not a reason to stop recognising the run.
- `gitOperation` is a *set* of operations, not one: of 850 sampled, 70 commit and push
  together and 30 open a pull request and push. So there is no `Committed` step — a commit
  is a `Ran` that names what it did to the repository, and one record stays one step.
- This Claude Code has no `TodoWrite` and no `Task` at all (zero calls in the sample); it
  plans with `TaskCreate` and `TaskUpdate` and delegates with `Agent`. An update records the
  task's id and its new status and never its subject again, so `Planned` is one task and
  what happened to it.
- A file written from nothing records an empty hunk list and its text in `content`, because
  there was nothing to diff it against. `Edited.change` is therefore the hunks for an edit
  and the whole file for a create.

`Tested` is a `Ran` whose output a test runner wrote. Each runner is four patterns in one
table — what proves it ran, how it counts what passed and what failed, and how it names a
failure — fitted to output captured from pytest, vitest, cargo test and go test, passing and
failing, and kept in `tests/fixtures/testruns/`. `go test` counts nothing it did not fail, so
`Tested.passed` is None there and a run's failures are counted by the names it printed. A run
is counted by the summary lines its mark found and by nothing else in the scrollback, added up
across them: a workspace prints one summary per test binary, and vitest counts its files on the
line above the one that counts its tests. A command is a test run because its output is a
runner's, never because of how the command was spelled, so `make test` and a script reach the
same table — and `Tested` is claimed only where those counts are the whole story. A command
that failed while nothing is counted failing did more than run a suite, and so did one that
committed on its way; both stay the `Ran` that carries the output saying what else happened.
`go test` writes its package line the same way for a package that never built, with the reason
where the time goes, so a package line counts only when it carries the time it took. Questions
asked in plain text rather than through `AskUserQuestion` are not a step: they are read off the
turn's closing text when it is narrated.

Record shapes worth knowing, observed in transcripts on 2026-09-14:

- `ai-title` → `aiTitle`. Claude Code's own title, often twenty-five words long, and shown
  nowhere the user looks. Hands never speaks it.
- `custom-title` → `customTitle`. The session's name, set by `/rename` or by a hook's
  `sessionTitle`. Use it for `list_sessions` labels.
- `assistant` → `.message.content[]`, blocks typed `text` | `thinking` | `tool_use`.
  `text` and `tool_use` are both content. A Bash `tool_use` carries a `description` of
  its purpose.
- `user` → `.message.content` is a plain string for real user turns, or an array of
  `tool_result`; `isMeta: true` marks injected reminders. The record's top-level
  `toolUseResult` holds the structured result: `structuredPatch` for edits and writes;
  `stdout`, `stderr`, and `interrupted` for commands; `gitOperation` for a commit, a
  push, a branch change, or a pull request. It describes one call, so it says which call it
  belongs to only where the record carries a single result.
- `permission-mode` → the session mode, written only as a prompt is sent; the hooks report it sooner.
- Every record: `uuid`, `parentUuid`, `timestamp`, `cwd`, `gitBranch`, `sessionId`.

**Prose is the smallest part of a turn.** In one session sampled for this design there
were 24 assistant text blocks and 157 tool calls; the text came to 126 thousand
characters and the tool results to nearly 2 million, most of them screenshots. Happy
narrated only the text, which is the least of what happened (failure mode 2). Here the
recognisers cover the rest, and the length budget applies to what is spoken, after
summarising.

**Subagents.** On the parent's transcript a subagent is one call and its report, which
becomes `Delegated`: an `Agent` call, or a `Skill` call whose result says `forked`, as
/code-review's does. Either result carries the subagent's `agentId`. Its own records are
in `<session>/subagents/agent-<id>.jsonl`, every one with `isSidechain` true; the
parent's transcript holds none of them. Where the subagent reports back, in a task
notification whose `<task-id>` is its `agentId` and whose `<summary>` reads `Agent
"<job>" ...`, or in the result of a call that ran it in the foreground, its transcript is
folded by the backfill's own fold and told as a part of that turn of its own, named by the
job the parent gave it, "the subagent's work on <job>" (failure mode 21). The record the transcript
starts from is that job, not work: a subagent's prompt, or for a fork, the parent's
launching call copied in. A notification whose summary names no agent is a background
command or a monitor. The notification is read with the telling that answers it, never
again with a later telling of the same turn.

**Results outside the transcript.** A formatter, a code generator, or a `sed` in a
shell command changes files that no `Edited` step names. So at `UserPromptSubmit` the
reducer emits `Snapshot`, and the reader records the target's `HEAD` and a tree of
everything git would keep; at `Stop` it emits `Compare`, which diffs that tree against
one taken now, and lists the commits reachable from where the turn ended and not from
where it began. A turn told a second time is read from where its first reading left off
instead (see "What git says is not the model's to say"). The summariser gets the `Delta` beside the steps, because it is the
result of all of them together and the file a `sed` changed belongs to no step at all.

A mark is taken only from a session sitting at the prompt, because a turn opens there
and nowhere else. Claude Code sends the hook for a prompt queued while a turn runs, and
a `Stop` the daemon never heard leaves a session working as far as the registry knows;
marked again at either, a turn would be compared against the middle of its own work and
everything it changed before that second prompt would be missing from the one telling
that names it `[LAW:no-ambient-temporal-coupling]`. A mark whose `HEAD` could not be
read is no mark at all. `rev-parse` says nothing both for a repository with no commit
yet and for one it could not answer for, and silence has more causes than a clock can
account for — a `HEAD` caught mid-rewrite by a checkout in the next terminal along, a
ref that cannot be read, a deadline with nothing left on it. So unbornness is asked for
rather than inferred: `symbolic-ref` answering means `HEAD` names a branch that no
commit is on, which is what every repository looks like between `git init` and its first
commit, and nothing else counts. Read as unborn, a mark that is really unreadable
compares against no commit, so every commit ever made in that repository is reachable
from where the turn ended and not from where it began — and the turn is spoken as having
made all of them `[LAW:parse-dont-validate]`.

The tree is written through an index of the daemon's own — the repository's index
copied to a scratch file, `git add -A`, `git write-tree` — so nothing is staged,
stashed, or reverted, and the repository's own index is never written. `git stash
create` would do most of this and is what this design first said, but a stash holds no
untracked file, and the file a code generator just wrote is exactly what a turn must be
able to name; it also touches the repository's index to do its work. The index is
copied rather than started from nothing because it carries what git already knows about
every file: 0.105 s against 1.409 s on a repository of 20,000 files, and this is taken
while a prompt's hook waits on it `[LAW:carrying-cost]`. The one mark left behind is one
unreferenced object per content git has not stored already, which its own housekeeping
collects. Content is what git names an object by, so a file that does not change costs
its size once however many turns read it: measured at 1.6 MB the first time two hundred
untracked files were seen, and nothing at all on the two readings after.

`Compare` is its own effect, emitted before `Summarise` and performed before it, rather
than read when the summary is made: summaries are made one at a time and take seconds,
and by then the session may have begun another turn, whose changes would be told as
part of the one before it `[LAW:no-ambient-temporal-coupling]`. It starts the reading
and waits for none of it. Both hooks are blocking ones and the shim gives up after
`POST_TIMEOUT_SECONDS`, after which it prints that it cannot reach the daemon; the
effect queued after `Compare` is the one that has the turn spoken at all, so a reading
that is slow, that fails, or whose handler is cancelled must cost the turn its delta
and never its telling `[LAW:no-silent-failure]`. Measured: the stop hook waits 0 ms,
and the prompt hook 40 ms here and 145 ms on 20,000 files, against a mark's whole
budget of `POST_TIMEOUT_SECONDS - 0.5`. A mark and a reading are both best effort and
neither may cost the turn what it was read for, so both are performed under one guard
rather than one guard each `[LAW:single-enforcer]`: a mark may not fail the prompt hook
waiting on it, and a reading may not cost the turn the `Summarise` queued behind it.

A mark is held from the moment its snapshot starts, so a reading that needs one still
being taken waits for it; where it could not be taken at all, the turn is told without a
delta. Readings stay one per telling, in the order the tellings are made.

Everything the summariser is shown of a delta is bounded by the `Budget`, commits
included: a turn that pulls or rebases brings them by the hundred, and the count is the
story where the subjects are not. The reader keeps no more than `MOST_COMMITS` of them
for the same reason it keeps no more than `MOST` characters of patch — and asks for no
patch at all past `MOST_LINES`, because git hands back a whole diff before a character
of it is cut, and a turn that generated a million-line file inside the repository would
otherwise have all of it in the daemon at once. The numstat counts that decide this cost
one line a file and are already in hand, and what is left when the patch is refused —
the files and their counts — is all of a diff that size that would have survived the
budget anyway. A numstat that could not be read is not a numstat reading nothing: read
as the second every count is zero, and the bound is not a bound at all on the one diff
whose size was the reason to ask `[LAW:no-silent-failure]`.

A reading is held for every turn that stopped, and past `HELD` of them the *newest* is
the one dropped. Readings are bounded and the `Summarise` effects they pair with are
not, so the two stay in step by position alone; dropping the oldest would hand every
telling after it the delta of the turn after its own, which is the one thing this
pairing exists to prevent. Dropped from the back, the turns past the bound are told by
their steps and every turn handed a delta is handed its own.

Within a reading the commits are read before the tree, because they are two fast
commands where `git add -A` is the slow one: a turn whose commit is the one thing worth
saying about it should not lose that because the tree ran past the reading's deadline.

One reading is made for every turn that stops, held in the order the turns stopped, and
every telling takes exactly one — including a telling whose summary failed. That is the
whole of what keeps readings and tellings in step. Held back for a turn that could not
be summarised, a reading would be taken by the next turn's telling, and a listener can
do something about changes they did not hear and nothing about changes attributed to
the wrong turn. The snapshot races the agent's first edit by however long the model
takes to start, which is seconds, and an edit that wins the race is still an `Edited`
step.

## Summaries: spoken form, and the narration tree

Nothing is read verbatim. Claude writes for a screen, and markdown, code, tables, paths,
and hashes cannot be heard as written, so every word that reaches the speaker is
summarised or transformed first.

**What runs today** is the top level of the tree: each finished turn is handed to the
intermediary with what git says the turn did and the turn's question where it ended on
one, as described below. The sections are opened when the user asks for more (`expand`, below). When a live session's `Stop` arrives, the reducer emits
`Summarise(session, closing)`, and `narrate` in `hands.voice.narrator` asks the tail what
that session has not been told. A turn opens at the last user record that is
not `isMeta`, not `isCompactSummary`, whose content is a string or a block list with no
tool result in it, and that does not follow a tool call or tool result: a message sent
while a tool runs belongs to the turn under way, and an image or document attached to a
prompt is named rather than read. The opening is `Commanded` for a record that opens with
Claude Code's slash-command markup (`<command-name>` or `<command-message>`), `Shelled` for
`<bash-input>`, `Notified` when its `origin.kind` is `task-notification`, and `Asked`
otherwise. What a command printed (`<local-command-stdout>`, `<bash-stdout>`, and their
stderr, marked) is a record of its own after the command's; it opens nothing, and joins the opening
whose record it names as its `parentUuid`, with terminal escapes dropped (`hands.core.turn.printed`). A
command and its output are as often written as a `system` record of subtype `local_command` as a user one,
and are read the same either way; such a record carries no prompt id and is no answer of Claude's. A command
written as the words typed — `/compact` ahead of its compaction, a skill run in a fork of its own — is `Commanded`:
a slash and a name, in a record with no `promptSource`, which Claude Code writes on every prompt it sends Claude, one
that opens with a slash included. The record `/compact` writes once it has run, under the same prompt id, is that
same command (`hands.core.turn.recorded`), not a second turn. The steps are assistant text (`Said`) and tool
calls matched to their results by id, each handed to `recognise`; thinking blocks are
skipped, because thinking is how Claude reached a result rather than a result. A
subagent's records are in its own file, and are folded only from there.

**A turn can stop twice.** Another hook may block a `Stop`, and the same turn then runs
on to a later one. So the tail keeps, per session, how many of the open turn's steps were
told and a closing reply told before its record existed, and each telling holds only the
steps beyond them, which is why the first half of a long turn is not heard twice. A turn
that opens forgets both, and a session the registry stops listing is forgotten whole.
Verified live on 2026-09-21 with a second `Stop` hook that blocks once: the first stop was
heard as what the turn had done, and the second as `echo second` and nothing before it,
1.41 s and 1.21 s from `Stop` to the spoken summary.

The steps are only half of what a second telling has to get right. The opening belongs to
the whole turn, so it goes in front of the summariser again — and a small model handed a
request twice answers it twice, which is what was heard on 2026-09-21: a second summary
restating the first half, out of an opening naming all three of the original actions, over
steps that held none of them. So which telling this is, is carried in the turn rather than
inferred beside it. `Turn.standing` is `Answering` or `Continuing(told)`, `render` matches
on it, and a continuing telling is shown its opening framed as context, with how many steps
have already gone out and an instruction to report only what follows
`[LAW:types-are-the-program]`. A flag beside the turn would have done the same job and left
the two readings of one opening uncounted by the type checker; an eval case is the same
real turn told both ways.

The stand-in reply is kept because the hook and the transcript disagree for a moment:
measured over twelve live turns, the reply `Stop` carries is never yet in the
transcript when the hook fires, and its record lands 46 to 77 ms later. So the hook's
copy stands in as the turn's last step while the record is missing, and gives way to
the record — never telling the reply twice — because Claude Code only appends, which
puts that record first among the steps not yet told `[LAW:one-source-of-truth]`. What
is *not* covered: a tool result written after a turn was told is never told, because its
call was already told as having none. Every call was paired with its result at `Stop`
in all twelve turns, so this is out of reach at a `Stop`. Progress does not reach it either:
it says what a call sets out to do, never what came back.

The narrator hands a finished turn to the intermediary as a turn of its own, and the
intermediary says it in its own words, so what the user heard is in its history and it can
answer about it. A summariser beside it, asked with `/btw`, left the brain's history
without it: on 2026-09-30 hands told Brandon a session asked about PR 68 and 69, and asked
what those were, the brain said no session showed them. What is handed is the last thing
the turn said — the session's own account, whose author knows what PR 68 is — bounded at
`REPLY_SHOWN` characters, since every turn told grows the brain's history toward
compaction; then what hands read of the turn that those words may not say, from the
narration tree; then the question it is waiting on, which the intermediary is told to end
on. If the brain cannot take the turn, hands says as written that it could not tell it,
and nothing more. Each working session is told at `SessionStart`, by the plugin's shim, to end every
turn with a concise, speakable overview. A turn the user stopped before it did anything is handed
on like any other, its record adding that the user interrupted it. Each summary, and how it was
delivered, are facts on the turn's utterance.
A transcript that cannot be read is said as "cc-hands finished a turn, and I could not
read it." without the model and out of the context, and logged as a `Failure` line.

For the brain, a narration waits in a lane of `BrainStage`'s own, never in Pipecat's
context, and the user's turn goes ahead of it: a narration that waited while the brain
was answering is asked only once no words of the user's are waiting. What the narrator
says as written waits in the same lane (`Aloud`), so a session's end is heard after its
last turn. Each turn is one `voice.turn` wide event, and its `queued_ms` is how long it waited in its lane.
Its `waited_ms` is how long the user waited from letting go of the key to the turn's first word on the wire,
`transcribed_ms` and `queued_ms` are where that wait went before the brain was written to; words that waited behind a
turn together are timed from the last of them. Each call the brain's replies made (`tool.call`) is a child event under
it, timed as the stage heard it on the wire. Each model round trip is the proxy's own `Exchanged` line, which carries a
span inside the turn's (`span`), and reaches the collector as a `proxy.exchange` span under it: no second record of a
round trip is kept. The brain's other requests in a turn, a subagent's or a fork's, and one hands held, are the turn's
too; a side question's own request is under its `brain.aside`. The router decides each exchange's span as it routes it,
and an exchange made for no unit of work, a wrapped session's through the tap or the brain's outside a turn, is a
`proxy.exchange` span at the root of a trace of its own, saying its session, kind and path. A turn hands stopped mid-way
still emits its event, cancelled, with what it had done.
The event also says what was typed into the brain (`asked`), and, where the user barged in, the calls running then
(`running`) and whether the brain was told to stop at once (`stopped`).

The brain's own process is four kinds of event. Its launch (`brain.launch`) runs from the spawn until its input is
up, on which account, model, and config directory, under which fritter (the one hands' package carries), or with what
failed it: a hands built without its fritter is refused in the words `hands install-fritter` refuses it with. Its run
(`brain.run`), a part of the launch, runs from then until its process ends, with its exit code and the last of what it
showed. Each turn typed into it (`brain.turn`), a part of the `voice.turn` that asked it, runs from its typing to its
end, with the prompt ids Claude Code took it as (`prompts`), how many times it was typed (`typings`), the tools its
latest request offered the model (`offered`), and the prompt ids that were not its own that Claude Code took while the
turn waited to be taken, or that its typing or its stop stopped (`others`), or with the error that failed it. A turn
is taken only by a `UserPromptSubmit` whose `prompt` is the text it typed, as Claude Code keeps it: out of the tags a
long paste comes in, trailing whitespace trimmed. A turn that ended untaken and was taken later is never the next
turn's, unless the two typed the same words. A prompt Claude Code takes that is no turn's runs ahead of what was typed
behind it, for an ask that already failed or one nobody made, so a turn stops it with Escape and Ctrl-C: before it is
typed, and again, typed once more with its whole time, when one ran through its wait. One taken late cannot make the
turns after it untaken, however long it would run. A turn typed twice is taken by either typing and over at the Stop of
either: the Escape may have stopped the first as it was taken, and a stopped prompt posts no Stop. A turn whose asker stopped waiting is still heard to its end. Each dialog
it posts to hands' listener is one event: a `brain.permission`, with the tool, the decision, and how long it was held,
and a `brain.elicitation`, which is always declined. A dialog posted for the turn in flight is a part of that `brain.turn`; any other is a part of the
launch. A body hands cannot read is answered as its hook asks, and its event fails saying so. Whatever else fails in
the brain's own work, a turn's typing, a dialog, a hook heard, is said once on the terminal with its traceback. The turn
it was done for, if any, is stopped as an Escape stops it and ends with that error; a permission it broke is refused, and
a hook is answered before it is heard, so one whose hearing broke is no turn's and the hooks after it are still heard.
Each side question (`brain.aside`), asked of a Claude Code of its own, is one event from when it is asked: what it is
for (`kind`), its Claude Code's session (`aside_session`), the question and its answer, how long it waited behind the
ones before it (`queued_ms`, absent for a question that never had its turn; the rest of its duration is its Claude Code
answering), or why it has none (`unanswered`), its asker's time running out among them (a `Deadline` from asking, or for the
line an old result goes as a `TimeLimit` from its turn), with what its Claude Code showed then (`shown`). Its Claude
Code is always ended gracefully, after its asker's time if need be: it shares the brain's config directory. An explanation is a part of
the `utterance` it delays, a summary of the `summary.backlog` or `summary.turns` pass that asked it, and a name of
its `name.judged`; a line, asked once its result's turn has ended, begins a trace of its own.

**What is in front.** Built: as the user's words reach the brain's stage, hands reads once
what is in front on the Mac's screen (`sessions/front.py`, decided by the pure
`core/front.py`), and the brain reads it after the user's words: the app in front and the
session its front tab shows, or that it shows none. It is never watched between turns, and
a narration is not read against it. The app in front is the one LaunchServices says is
(`lsappinfo front`), whether or not a window of its is on screen; an app that says by AppleScript which terminal its front tab shows
(iTerm2, Terminal) is asked, and any other shows every terminal under it, which names the
session when it holds one. A tab running tmux shows the pane its client is on. A session is
in front when that terminal is on its line of ancestors: its fritter's, its tab's or its
pane's. A screen hands cannot read (no app in front, a refused or unanswered AppleScript,
several sessions under the terminals it shows) is left out of the turn, and the `voice.turn` event's `asker` fact, a
`UserAsked`, records what was read or why it was not, and how long the read took.

**Screen or audio-only.** Built: the user's turn also tells the brain whether they can see a
screen (`core/place.py`'s `Modality`), read as their words reach the stage and recorded on
`UserAsked`. `PushToTalk` owns it with the place: a move to the desk makes it `screen`, a
move to the phone `audio-only`, and the brain's `set_modality` tool switches it until the
next move. It is a hint the brain chooses by, never a limit on what hands does.

**The same notes on an API backend.** Built: `core/beside.py` composes both notes for
either model. Under an API model `voice/beside.py`'s `Noting` stands between the floor
and the user aggregator. As the key is let go on a hold it reads the same two readers,
the screen as a task beside the pipeline while Whisper transcribes; as the hold's words
arrive it passes them on and puts the notes into the context as a message behind them.
No frame waits on the read: words that arrive before it is done are noted with where the
user is alone. The hold is resolved behind its words, so the note is in the context ahead
of the user's words, which the turn writes as it ends, and what hands tells of the
sessions, which the floor holds until then, follows both and is followed by no note.
Holds let go before the first one's words arrive are noted once, as the last was let go.
Each hold let go is one `front.read` wide event: its `front` and `modality` facts are
what was read, its duration is the read, and its outcome is `cancelled` where the words
arrived first or a later hold's read took its place.

The rest of this section is planned: step summaries built as steps arrive, and streaming.

**Spoken form.** Built: `core/spoken.py` is a pure function from text to speakable
text, installed as the TTS service's one text filter `[LAW:single-enforcer]`. Pipecat
applies a TTS service's filters to the text of a `TTSSpeakFrame` and to each aggregated
sentence of a model's streamed reply alike, so summaries, announcements, system speech
and the intermediary's own words all meet there, and text that has not been through it
cannot reach the speaker at all. That last part is why the filter goes here rather than
at each place a frame is built: there are five of those and the intermediary's own reply
is not one of them, so four enforcers would still have left the largest source unpoliced.
Asking a model for spoken form does not settle it either — that is a rule held as an
instruction, obeyed or not, checked by nobody, and it had already been heard saying a
bare file name while session titles never passed through it at all.

Headings become section cues and lists become counted sequences. Identifiers are split
into words, so `authMiddleware` is "auth middleware". A path becomes its file name, with
its directory only when two files in the same breath share it. Flags become their names,
and hashes, ids, and URLs are named by what they point at or dropped. Code blocks,
diffs, and tables never reach it as text, because they are summarised first; one that
leaks through is replaced by its kind and length, and the leak is logged.

Two things the rules are shaped by. Each asks for a tell that ordinary English does not
have — a diff must announce itself with `@@` or `diff --git`, a table must be two rows
rather than one line with a pipe in it, a hash must carry a digit and a hex letter both,
a path must start at the root or end in an extension on a closed list, and an id must be
mixed case as well as long — because a rule that mangles a sentence costs more than the
code name it fixes `[LAW:carrying-cost]`. A rule with no tell does not merely fail to
help: it takes a word out of the middle of a sentence and leaves it grammatical, so
nothing downstream can notice. "and/or" was heard as "or", "24/7" as "7", a file of
1048576 bytes as "a commit bytes", and `base64_encode` as "an id". That is why the tells
are held by a table of ordinary sentences in `tests/test_spoken.py` that must come back
unchanged, rather than by this paragraph: a promise in prose is a map nobody redraws.

And the last step drops every mark left over, unconditionally — including a line with
nothing in it but marks, which is how a rule across the page and the dashes under a
heading are drawn — which is what makes "no backtick, no pipe table, no fence reaches
the speaker" a property of the function rather than a hope about the rules above it
`[LAW:parse-dont-validate]`. A fence is parsed rather than recognised by its first three
characters: its closing run must be its own character and at least as long, because a
four-backtick block is how a model quotes a three-backtick one, and a length-blind
closer ended the outer block at the inner opening and read the quoted code out loud.

On the streaming path the filter sees a reply a piece at a time, so a list the intermediary streams
is seen an item at a time and is not counted aloud as a sequence, where the same list inside a summary
is. A fenced block cannot survive that split — its continuation arrives carrying no fence and would be
read out as the ordinary text it resembles — so the reply is never split inside one. Pipecat's
`LLMTextProcessor` stands between the model and the TTS service with `FenceAggregator`
(`voice/spoken.py`), which breaks the stream into sentences and holds a block whole from its opening
fence to its close, reading where one is open by the same rules `spoken` uses. Pipecat flushes that
aggregator when a reply ends and resets it on a barge-in, so a reply that ends inside a fence is said as
its block and leaves nothing behind for the next one. Carrying the open fence in the filter instead was
tried and reverted: Pipecat never tells a text filter that a reply ended, and a reply that ended inside
a fence left every later utterance replaced by a block announcement.

The filtered text is also what the intermediary remembers, because Pipecat builds the frame it appends
to the assistant context out of what a filter returned. That is intended: of the five places a
`TTSSpeakFrame` is built, the three that keep their text say in their own comments that the context is
kept so the model can answer about what the user heard, and before this filter it held what was sent to
the speaker, which was never the same string. The cost is that the model cannot read an exact path or
sha back out of its own memory, and it has the session tools for facts. A draft readback crosses the
same seam and does not want a summary's liberties — it is read out so the user can check what will be
sent — which is tracked separately rather than solved by giving the function a mode.

The function is pure and stdlib-only because it is the domain — what a developer who is
not looking can hear — and it therefore cannot log. A leak is returned rather than
logged, and the voice edge logs it, because logging is an effect and `core` is not the
edge `[LAW:effects-at-boundaries]`. The filter is stateless, so a barge-in mid-sentence
leaves nothing to reset.

**The narration tree.** `core/narration.py` cuts a finished turn into segments: whether
the user stopped it, what the repository did, the question it is waiting on, what the user already
answered, and one section per topic — the change, the tests, the commit, the commands, what it read, the plan, the subagents,
the other tools, what it said. The topics are not a table of rules written beside the
steps; they are a match over the `Step` union, which already draws exactly those lines, so
a new kind of step is a compile error here rather than a result with nowhere to go
`[LAW:types-are-the-program]`. Every step lands in a section, which is what makes "more on
that" able to reach anything the turn did. A segment holds the happenings it was cut from,
so `opened` renders them through the same `body` the whole turn goes through, and its
`refs` are read off those happenings rather than stored beside them — the record ids a
segment names are the records it holds, and the two cannot come apart
`[LAW:one-source-of-truth]`.

**More on that.** Each telling of a turn is held in `Recounts` with the tree's parts beside the
words the intermediary was handed, and the handed words carry the session's id, so `expand(session,
part?)` needs no listing first. Asked with no part, it gives each part's line, "The tests: one test
run.", however often it is asked: every part told deeper at once is more than one tool result can
carry. Asked for one part, it opens a rung down a ladder of budgets (`hands.core.drilldown`), each
rung `opened` at a longer one, and the depth is how many times that part was asked for, counted in
the daemon rather than remembered by the model. A Stop that finds nothing new keeps the depth; a
new telling of the turn starts it over. Whether there is more
is measured, by whether the next rung tells any part differently, so a part told whole at a short
rung says so. The deepest rung is still a rendering cut to its budget, for the intermediary to put
in its own words: there is no verbatim rung, so asking for more never gets code or output read out.
A test run keeps the lines its runner says why in (`Runner.why`: pytest's `E` lines, a Go test's
`file.go:N:` lines, a Rust panic and its message, a vitest `FAIL` and its error), so "the tests"
opens into what failed and why; pytest's own summary line is no use for this, since outside a
terminal it cuts the reason to `- Asser...`.

None of the tree is prose from a model. It is arithmetic over typed steps, and that is the
point rather than an economy: a summariser was measured on 2026-09-21 reporting "version
two point seven point one" for a runner that printed 8.4.1, and a count that is computed
cannot be invented.

**What git says is not the model's to say.** Whether a turn committed is recorded twice —
by the step, when Claude Code writes a `gitOperation`, and by the delta read against where
the turn began — and each sees what the other misses: a `git commit` inside a heredoc
carries no operation for a step to hold, and the delta names it anyway. So the narration
says it, from whichever saw it, in words that carry no hash: "It committed and left five
files different." Asked for this instead, a summariser was measured both dropping the commit
entirely and reading its hash out loud, in the same afternoon. It is handed to the
intermediary beside the reply, which may say the same, and that redundancy is kept on
purpose — dropping git's clause whenever the reply claims a commit would suppress it in
exactly the case it exists for, a commit claimed that never landed
`[LAW:no-silent-failure]`. A branch is said by `spoken_ref` rather than copied: this is the
one clause of the top level no model wrote, so the instruction cannot reach it, and the
filter in front of the speaker deliberately will not read a bare `feature/narration-tree`
as a path — a rule loose enough to catch it also eats "and/or" and "24/7" and costs each of
them a word. A ref is therefore said where its type already knows what it is, with its
separators as spaces and every word kept, because a branch is named so it can be told from
the others `[LAW:single-enforcer]`.

A push, a branch, or a pull request changes no file and adds no local commit, and Claude Code
writes no `gitOperation` for one inside a heredoc, a script, or a compound command it does not
parse — two commands in five that push, and nearly every `checkout -b`. So the delta reads each
where it does leave a mark, and only for the branch the session's worktree is on, since every
worktree and the terminal beside it share the repository's refs: that branch's own log saying it
was created from something other than the remote branch of its name (a checkout or a rename is
not a branch made), a remote-tracking ref of it whose log says `update by push` since the mark (a
fetch moves the same ref and logs `fetch`), and, for a pushed branch that is not the one a
remote's HEAD follows, a pull request the forge says was opened since the mark. The forge is asked
beside the tree and must answer `SPARE` before the narrator's `PATIENCE` runs out, so a slow forge costs the
pull request and never the commit. Both sources speak in the same `GitChange` values, so a push
both saw is said once. Each mark is one `delta.mark` wide event and each reading one
`delta.read`, and each says which index its snapshot started from: the repository's
own, copied; an empty one; or none, because git would not say where the index is. Each
also counts the git commands that did not answer, so a git that timed out reads
differently from a repository with nothing to tell.

A turn told twice has git read twice, each telling against where the one before it left off.
No prompt marks where the part after a blocked `Stop` began, so every reading also keeps where
it found the repository, and `Compare(again=True)` reads against that instead of the prompt's
mark: a commit made in the turn's second half — a heredoc commit, which no step records
either — is told with the second telling, and nothing the first told is told again. A first
reading that could not say where the repository stood leaves the second telling without a
delta. Only a turn hands told before is read this way: one whose first `Stop` it never heard
is read against its prompt's mark, or not at all where no prompt was heard, since the last
reading is then another turn's.

**What the turn asked always plays, once, and the daemon decides whether it asked.** A turn
that ends on a question is waiting on the listener whether or not a hook blocks, so
`open_questions` reads it off the turn itself: an `AskUserQuestion` nobody answered that
nothing but an interruption followed, and what the closing text asks — the turn's last step,
since a question it worked past was answered or did not need one. A dialog Claude went on
past was declined with a message or refused by a hook, as 5 of the 94 unanswered in this machine's
transcripts were; the other 89 were escaped, and ended the turn on the question. `asked_in` is the one reading of a
text for questions, used on Claude's closing text by the narration
`[LAW:one-source-of-truth]`. A question put to the listener outright counts wherever
it stands ("want me to do it?" before two more sections). Where the text ends, an offer
counts ("Say the word and I'll do it."), and so do a choice and any other question, unless
its own list item or run of prose goes on to answer it ("Why did it fail? The cache was stale.") and nothing
later in the paragraph looks ahead to an answer still to come ("Once I have that I'll pin
the interface."). A `?` inside a code block, a code span, a quotation, or an italic aside is
written about rather than asked, one with a word straight after it is in an address or a
name, and one an arrow follows was answered on its line. Those shapes were read off 3,038
closing texts on this machine, and each that decides a case and fits in a fixture is one:
the only real closing that asks itself and answers on the same line is a 794 KB turn.

What the turn is waiting on is one question segment, last, in Claude's own words framed as
"It is asking:" or "It said:", with "(Recommended)" dropped and put through `spoken`. It is
handed to the intermediary to end on. An
`AskUserQuestion` the turn is no longer waiting on, answered or gone past, is its own
segment in `settled`, there to be opened and never played. `tests/test_questions.py` holds
`open_questions` to no miss and no false alarm over real turns lifted whole out of real
transcripts, under `tests/fixtures/turns`.

**Streaming.** Built for tool calls (`hands.core.progress`). Each call the tail reads into a
running turn is said by what it sets out to do, read off its input and never its result:
a command by the description Claude Code asks for ("run the test suite"), an edit or a read
by its file's name, a search by its pattern only when that is words, and never code: not a
command, a regular expression, or a glob. `AskUserQuestion` and `ExitPlanMode` are not
progress: their hook speaks them as they are asked. The tail hands the calls on as a
`Progressed` event, and none of a turn it read from a file's start, which began before hands
followed it. The reducer gathers them on the `Opened` turn, so a turn that ends lets go of
what it gathered unsaid, its result being told instead; its tick lets a burst go as one
`Progress` once no call has come for `SETTLE` seconds, or once its first has waited
`LONGEST` `[LAW:no-ambient-temporal-coupling]`. The relay routes it by
`attention.progress_route`, a table over the focus and the overlay, and records each choice
and what decided it on the burst's utterance: the focused session's is `Working`, played as written at `fyi`
("cc-hands: edit ten files, then run the test suite."), and kept out of a pushed context,
which keeps every message it is given. Any other's, and a muted one's even when focused, is
left to the session listing, which says what a working session last set out to do: the
`list_sessions` tool for an API model, and the tail of the brain's every request. `coalesce`
folds a session's progress into one telling and drops what a result of the same turn says
better; progress carries its turn's ids for this, since a result reaches the floor only once
it is summarised, after the next turn's calls may have.

Text is gathered the same way, from `MessageDisplay`: each batch of lines Claude Code displays is a `Displayed` event in
the turn its `prompt_id` names, joined to the burst on the `Opened` turn, so a line holds the burst open as a call does
and a long explanation is let go at `LONGEST`, while Claude is still writing it. Lines displayed after the turn's `Stop`
move nothing. Progress the relay routes to be played goes by a lane of its own (`hands.voice.working`), because its text
waits on a summary and a permission request must not wait behind that: the summariser's model says the text as an
imperative phrase, which goes ahead of the burst's calls ("cc-hands: explain how DNS resolution works, then run the test
suite."). Text that cannot be summarised is said as "write something", never read out. A turn that ended while its
progress was being summarised is not played, its result being told instead; the registry says so as the summary is
ready, and the burst's utterance records its phrase, or its failure, which fails the utterance and leaves it said.

A subagent is heard the same way while it works, from its own transcript. The tail knows one by the
`agent-<id>.meta.json` Claude Code writes beside that transcript as it starts it, and follows any started before hands
followed its session from where its transcript ended then. Its calls are a `Progressed` whose `of` is the subagent, an
`AgentTask` named by the job the call that started it gave it: the meta file's `description`, or, for a skill run in a
subagent of its own, which names none, the one `Skill` call its parent is running ("/code-review high 152", said as
"code review high 152"). A subagent a subagent started sits in the same folder, its meta naming that one as
`parentAgentId`, and is heard as the work of the job the session's own call gave the first. A subagent nothing names
is said in the log once and never told as another's. The record its transcript starts from is its job, not its work,
as for its report. The reducer gathers its calls on the `Session`, not the turn, since one run in the background
works on after the turn that started it ends ("cc-hands, its subagent to review the parser change: read tail.py, then
run the test suite."). `coalesce` folds a subagent's progress only with its own, and drops it wherever a telling that
reports that subagent stands (`News.reported`): its work is told with the turn it reports back to, and a result that
does not report it leaves it news.

## Playback: bookmarks and resume

You will cut the reading off often, to ask which file or to answer something else, and
every interrupted reading must be resumable. So where playback is lives in the daemon,
not in the model's memory (failure modes 8 and 25). The player holds a `Playback`: the
segment on the speaker, and a stack of bookmarks where earlier readings were cut off.

An interruption pushes a bookmark at the segment that was playing. "Go back to what you
were talking about" is `resume()`, which pops the bookmark and replays that segment from
its start. "Skip that" and "say that again" are `skip()` and `repeat()`. "That part" is
the segment playing, or the last one played, and "more on that" is `expand()` on the turn it belongs to.

Pipecat's output transport reports text as its audio plays, and on an interruption only
the text that played reaches the context. pocket-tts reports no word timings, so the
finest position is a sentence. As built (`hands.core.playback`, `hands.voice.player`), a
reading is the run of sentences handed to the speaker since it last fell quiet, whether
the model's reply or lines said as written. An observer reads it off two processors'
pushes: each sentence as the TTS service makes it, and each sentence's `TTSTextFrame` and
each barge-in as the output transport lets them go; a processor standing in the pipeline
instead takes a barge-in ahead of a sentence's end still in its queue. "Go back" goes back
past the reading the user just cut in on, which is the side answer they are leaving, to
the one cut off before it. Measured live on 2026-10-03 with pocket-tts into
BlackHole: all three sentences of a reply were made by 2.6 s, the first one's text frame
left the transport at 4.2 s as its audio ended, and a barge-in one second into the second
sentence bookmarked the second. `resume`, `skip`, and `repeat` are tools whose call is the
whole reply: hands says the sentences again as written, so a model never retells them from
memory. Those lines enter just ahead of the TTS service, not through the model's stage, so a
barge-in drops them like any sentence not yet played; their reading holds all of them from
the start, so none is lost to going back.

A staged or amended draft is read back the same way, by hands as written: the tool hands
back `{"says": ...}`, and the API path says it as the call returns while the brain's stage
says it once the brain's own words are done, so a barge-in before then does not lose it.
A reply is the whole of what is said only when every call in it was silent: one refused
(`error`) or one that asks a reply puts the model back on. Pipecat lets each result decide
and the last to finish wins, so `Replies` is told each reply's calls as the service starts
them (`RunsReplies.run_function_calls`) and has only the last answer for them all.

## Push pointers, pull content

The intermediary's context window holds the conversation with you, not session
transcripts. Hook events and played segments are injected as small frames carrying the
session title, the event kind, the spoken text, and the segment id. Steps, records, and
unplayed segments stay in the daemon, and `expand`, `read_session`, and `recall` pull
them. When the daemon starts or reconnects, the intermediary gets one note listing the
live sessions by title, state, and focus, and never their history; Happy's session
directory at connect is the model for it. The brain gets no note: each request its turn
makes carries the same listing, composed as the request leaves, as a last text block after
the block holding Claude Code's cache marker (`hands.voice.briefing.tail`). The cached
prefix is exactly what Claude Code sent, and its history never keeps a stale listing. The
brain's history is kept small on the way out the same way (`hands.brain.context`): a long
tool result more than K turns old goes as `<tool>: <sentence>`, a batch every K turns so
the cached prefix changes once a batch, with the sentence asked once the result's turn
ends, as a side question that shows the result to a Claude Code started for that question
alone (`hands.brain.asides`), and kept in the summary store by the result's content. Each batch reached
is one `context.stubbing` event, naming the calls that go as a line from then on (`stubbed`) and those
that go whole because no sentence had been said of them by then (`whole`), each counted; and its compaction is asked for what a voice
session keeps rather than what a coding session does. This is the single decision that avoids
most of Happy's trouble: it pushed history in and could not pull, so it needed a
bootstrap dump, an eviction policy it never wrote, and a window that only grew.

The one thing that must be preserved for "give me the details of that part" to
resolve: every segment carries the `uuid`s of the records it summarises. "That part"
is then a lookup rather than a fuzzy search back through what was said.

## Sessions: membership from files, state from events, liveness from the OS

Three facts about a session have three different sources, and the registry derives
from all three rather than storing any of them twice `[LAW:one-source-of-truth]`.

- **Membership** is the set of files in `~/.hands/sessions/`. The shim writes one at
  `SessionStart`, or at the first hook of a session that fired none, and removes it at
  `SessionEnd`. The daemon sweeps the directory once
  before its models load and every 2 seconds after (`hands.sessions.liveness`). A file
  whose process is running becomes `Attached`, which registers a session the registry
  has never heard of in `Idle` and changes nothing for one it knows: a hook heard first
  knows more than the file, so a sweep and a hook can land in either order. A daemon
  restart therefore loses nothing: its first sweep lists every session the last run
  listed, before a word is spoken.
- **State** comes from the reducer applied to hook events since the daemon attached.
- **What Claude Code says the session is doing** is the file Claude Code itself keeps
  for every interactive session, `sessions/<pid>.json` in the config directory that
  holds the session's transcript (2.1.280 to 2.1.282): `status` is `idle`, `busy`,
  `waiting` (with `waitingFor`, `permission prompt` or `input needed`), or `shell`, and
  `statusUpdatedAt` is when it was set, in epoch milliseconds. A `!` command reports
  `busy`. `hands.sessions.statusfile` reads every listed session's file ten times a
  second and applies a `StatusReported` each time its stamp differs from the one the registry holds, so a status set
  again to what it was, or an idle, busy, idle between two reads, is still heard. A
  status or reason hands does not know arrives as an unknown variant and is logged,
  never read as idle. A file that names another pid or another session is refused. A file
that is missing or refused is logged as an error, once per reason: without it a turn
stopped with Escape, which fires no Stop, is never heard to end. The
  registry holds it as the session's state, `Idle` or `Running`, and only a status read
  moves a session between them. An `idle` ends the open turn, however the turn was
  stopped, with no case for any one way of stopping it. A prompt whose turn is open and
  unread was cancelled by an Escape during its hooks, which sets `idle` ~70 ms later, or
  taken and stopped before the tail read its record: either way its turn ends here, and is
  told as itself only if a record says it ran. The session is `Idle` at once. Claude Code sets `idle`
  before the transcript says how the turn ended: an Escape's interrupt record is
  written 37 ms after (2.1.283), and an Escape'd turn's Stop can fire after it. So the turn is
  kept `Untold` and is told, once, at the first of five events: its Stop
  (told with the reply the Stop carries), its interrupt record (one naming its prompt), a turn after it
  opening (told before that turn's mark), the session ending, or a reading of the
  transcript through `UNTOLD` ms past the `idle`'s stamp, which is what tells a double
  Escape that leaves no record. That wait is on Claude Code's clock: each reading says how
  far it read (`Read`), stamped before it opened the file, so a transcript hands reads late
  delays the telling and never leaves the record out of it. Claude Code sets `idle` only once a Stop's hooks have returned, and the
  daemon answers a Stop's hook only once the Stop is decided, so a stopped turn is normally ended by its Stop. A Stop under an
  id no record read yet names is held until one is (`Holding`), for at most `STOP_HOLD_SECONDS`; past that its hook is let go
  and the Stop still tells its turn once a record names it, or is a line once the transcript is read through it without one. It sets `busy` before a prompt's hooks run, so no `idle`
  read after a prompt is applied predates it. A single Escape that flushes a queued
  message sets `busy` again, not `idle`. All three were seen live on 2.1.282. The
  ordering holds only if a file is read at the moment its report is applied, so the
  reader reads each session's file lazily, looked up by id, once the report before it
  has been applied.
- **Ends nobody heard.** One process holds one session, so a running process's
  *holder* is the newest file that names it and passes the start-time check. A file
  whose process is not running is `Died`; the holder is `Attached`; any other file on
  a running process is `MovedOn`: a `/clear` or a resume inside the process whose end
  hook never arrived, which ends without a word and has its file removed. The shim
  removes a session's file before it posts `SessionEnd`, so a listed session with no
  file ended even if that post was lost. It is judged only when its file was also gone
  at the sweep before, which leaves the end hook two seconds to say how the session
  ended: then it is `MovedOn` if another session holds its pid and `Died` if none does. The sweep takes the listed sessions before
  it reads the directory, and a session joins only after its file is written, so a
  session joining mid-sweep is never taken for one whose file is gone. A file whose
  pid no macOS process can have (outside 1 to 99999) is reported and removed as it is
  read, like one that does not parse, so the kernel is never asked about it.
- **Liveness** comes from the kernel: each file's pid is asked when its process started
  (the `kern.proc.pid` sysctl, about 10 µs a pid), every sweep. A session waiting for input is silent for hours and alive; a session in a
  tool loop is never silent. Silence measures nothing. A process counts as the
  session's only if it started before the file was written, so a pid reused by a later
  process reads as dead. A dead one becomes `Died`: the session is `Gone`, a waiting
  permission hook is let go, the file is removed unless a new process has rewritten
  it, and "The session cc-hands is gone" is spoken, once. A session this run never
  listed, such as one that died while the daemon was down or before a reboot, has its
  file removed without a word: the user was not told of it here.
- **Ends** arrive through `SessionEnd`, whose `reason` says who ended the session.
  Measured on 2.1.270: `/exit` and a double Ctrl-C report `prompt_input_exit`, `/clear`
  reports `clear`, and a closed terminal reports `other`. An end the user
  chose at the keyboard is not spoken; `other`, and any reason hands does not know, is
  spoken as gone, the same sentence a dead process gets.

`list_sessions` offers a session while its pid is alive, labelled with its project and
name, its state, and whether it is the focus.

## Focus, drafts, and other state that stays out of the model's head

Models are unreliable at holding "which session we are talking about" and "I am
mid-draft" across a long conversation, and both failures are expensive. So both live
in the daemon as typed state and the tools default to them.

**Focus** is `SessionId | None`, or `Unreadable` when its file cannot be read, kept in the home's `focus` file (`hands.sessions.focus`)
and read at each use. Every tool that acts on a session accepts the omission of its
`session` argument as "the focus" (`defaulting_to_focus`), and a session it names wins
over the focus. "Switch to cc-hands" is `focus_session`, with no confirmation. A
session whose turn or question hands tells becomes the focus too, so the reply that
follows reaches it (`hands.voice.refocus`). It moves as the model takes the telling,
after the user's words that came before it, so those still go to the session they
were meant for: the brain's stage moves it before it asks, and behind an API model a
`Told` frame follows the telling and is dropped with it by a barge-in that comes before
the model has said it. A barge-in while it plays stops what is heard, not the move. A
session that has ended is never focused. The
brain is told the focus at the tail of every request, a Pipecat model in its startup
note, and `list_sessions` gives it to either, so hands, not the prompt, answers
"which one did you mean".

**Drafts** are per target: `NoDraft | Staged(text, resolutions)`. The readback is
generated from the stored resolutions, never from the model repeating itself:
"Draft for cc-hands, reading 'auth middleware' as `authMiddleware.ts`: refactor the
auth middleware to use the new token helper." Speak what changed, not what you said.
A draft is staged, amended, discarded, and sent with `send_draft`, as the Type effect;
`send_command` and `interrupt_session` emit the same effect with a `Command` or the Escape
`Key` (`hands.core.keyboard`). fritter holds the session's
pseudo-terminal and `Typist` types into it over a unix socket, so there is no window to
find, no focus to steal and no macOS permission to ask for. A send appends a `Typing`
audit record before it types, and a `TypingFailed` with the same effect when the typing
fails, so "did it send something I didn't approve", and "did it arrive", are answered by
one file.

## The summary store

A project's backlog is too large to read raw into a spoken conversation: 132 open links
tickets are ~145K tokens through `lit show`, and one sentence each is ~3K. So
`read_backlog` and `read_ticket` serve a sentence per ticket from the summary store, and
the ticket's own words only when asked (`full`). The backlog is read fresh from
`lit export` at every call (`hands.sessions.backlog`); the store holds only sentences.

A sentence is keyed by a digest of what it was made from (`hands.core.sentences`): the
summariser's instruction, the thing's own text, and the sentence of each thing under it.
The tree is the backlog over its unfinished tickets with no unfinished parent (epics, loose
tickets, and follow-ups filed under a closed ticket), each over its unfinished children. Editing one ticket changes its key and its ancestors' and no other,
and stops rising at the first sentence that comes back unchanged; a rerank or a status
change changes no key; an edit to the instruction changes every backlog key (a turn's key
is its identity, below). The rows live in
`<home>/sentences.db`, one table, never updated.

Sentences are made off the voice path by one task (`hands.voice.summarising`). A
backlog is wanted at start for every live session's project and again at every read;
the task asks the summariser for what is due, twenty things to a call, leaves before the
parents keyed by their sentences. Each pass is one `summary.backlog` event: how many
things the backlog has, how many were known, said, and still unsaid at its end, its rounds,
calls, failed calls, and stray reply lines, and the things a reply left out (`left_out`) and
what each failed call raised (`errors`), and, once lit has answered, whether it has a workspace there (`tracked`). A
directory lit has none in — `lit init` never ran, or it is in no git repository — has no
backlog: its pass is ok, `tracked` false and every count zero, and `read_backlog` answers an
empty backlog with `tracked` false. A pass fails with lit's error when the export could not
be read, and with the last failed call's error when one failed, either kind of pass. A tool
never waits on it: a thing with no sentence yet is served by its title and counted in
`unsummarised`. Measured 2026-09-29 on this repo: 52 sentences in 5 calls and 45 s cold,
0 calls and 0.16 s warm.

A session is served the same way. `read_session` splits the whole transcript into turns,
one per request (`hands.core.turn.turns`), and hands over the newest forty with a sentence
for each finished turn, or its request while the sentence is unsaid; `before` pages back,
and `read_turn` pages one turn's steps from a mark. A finished turn never changes, so its key is not its text but its identity
(`turn_digest`): its first record's id and how many things happened in it, under no
instruction. The turns are wanted at each read and said by the same task, one `summary.turns` event per pass, which
skips what was said after it was queued (`known`) and counts what it asked, said, and failed on. Whether the last turn is finished is the
registry's to say, and only `Idle` or an ended session proves it: an `Unreported` one may
be mid-turn.

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
answer_permission(request, decision, message?)
answer_question(request, answers)
find_path(session?, query)
catch_up(minutes?)
recall(query, since?)
expand(session, part?)           resume()
skip()                           repeat()
stay_silent()
```

The boundary rule: the tools route, name, and read session records and summaries.
None writes to a repository, and the conversational model is never handed a file's
contents. The daemon does read what a session changed, through its transcript and its
git delta, because summarising results is the job; the summariser sees diffs so that it
can describe them. `find_path` returns paths from `git ls-files` in the
target's `cwd` so that a spoken "the auth middleware file" can be resolved to a real
path in the draft; it returns names, never contents. `catch_up` and `recall`
read the daemon's own audit log. Give the intermediary an edit tool and it will
eventually decide that editing the file is faster than routing your request; the
surface above is the whole surface `[LAW:no-mode-explosion]`.

`stay_silent` is how the model declines to answer words that were not addressed to
it, taken from Happy's `skip_turn`. Push-to-talk rarely needs it; the wake-word edge,
which opens the mic without a hand, does.

**The prompt and its eval.** The intermediary's prompt is `voice/intermediary_instruction.py`,
and `evals/intermediary.py` judges it over `evals/conversations`, one decision point a case,
asked through the daemon's own service, adapter, tool schemas, and start-up note. The prompt
says when to reach for a tool, and the tool's docstring says how to use what comes back; it
names only tools the model is given, so each planned tool's ticket adds its own line. Two traps,
both measured against Qwen on 2026-09-26: it copies a quoted wrong reply almost word for word
("The docs site is idle, so it can take this" came back as "The auth refactor session is
idle, so it can take this"), so a wrong reply is quoted only when no case could pass by
copying it, and a plausible one is described as an action instead; and mlx_lm.server samples at temperature 0, so its runs are identical and an edit to
one paragraph can flip a case that paragraph never mentions. Run the whole eval after every
edit, not the cases the edit was for.

`send_command` exists so that `/clear`, `/compact`, and `/model` reach the target as
commands, with their sigil intact. `stage_draft` text always has a leading sigil escaped. The two never share a code path that inspects the first character; the
`Input` variant already knows. Claude Code reads three sigils at the start of a
prompt: `/` a command, `@` a file mention, `!` shell mode. Behind a space each is
plain text, so `Text` is always typed with a leading space, whatever it starts with,
and its newlines stay inside the prompt because fritter pastes it.

## The audio side

**The gate is the turn boundary and the mute.** Pipecat's turn strategies act on
voice-activity frames, so the key is the VAD: every microphone frame carries the key
it was captured under, and Whisper pushes the VAD frames where those keys change,
numbering each hold, so the turn the strategies see and the audio Whisper transcribes
are cut at the same frame. How a turn ended is not a key position: the gate counts the
turns it has sent and thrown away, every frame carries both counts, and Whisper ends
its hold as they move. A frame is captured every 20 ms, and a turn can end and the next
arm between two of them. The key is also the mute: microphone bytes become silence
of the same length unless the key is pressed. Frames flow at full rate either way;
only their content changes. The microphone opens on the press, and a hold's audio
begins there, so words said before the hold means talk are kept; they are thrown
away if the press turns out to be Shift. The hold opens the turn, the release ends
it and is final, and a hold opened during playback broadcasts the interruption that
flushes queued audio. That is barge-in. A turn ends once Whisper is done with every
hold it took in (`KeyTurnStop`): a press while the last hold is still being
transcribed joins that turn, so no hold's words are left out of it. Nothing else ends
one: Pipecat's user aggregator would end a turn itself after 5 s with no speech and no
transcript, which a slow or queued transcription outlasts, so that timeout is set to
never (`user_turn_stop_timeout` in `build_voice`). What bounds a turn instead is Whisper:
a transcription that has not returned in a minute fails, said aloud like any other
failure, and that resolves its hold (`TRANSCRIBING_SECONDS`).

**The mute is decided where sound is captured, and the speaker's echo is cancelled.**
Measured on 2026-09-14 with MacBook Pro speakers and microphone: the interruption stops
writes to the speaker within a few milliseconds of the press, but what was already
written stays above the room's floor at the microphone for about 185 ms, and Whisper
turned that tail into a word ("Wow.", "Well.", "What?") in every run where nobody spoke.
A Pipecat input filter could not stop it, because it runs when the event loop reaches a
frame, tens of milliseconds after capture. So `hands.voice.microphone` replaces the
local transport's two halves, joined by the echo canceller of the streams open
(`hands.voice.echo`, WebRTC's AEC3 through LiveKit's binding). Each reopen opens the new
pair of streams on a new canceller, which learns the new room from nothing. The old
canceller is closed on the thread that closed its microphone stream, so a stream still
calling back never reaches a canceller that has been let go of. The `Speaker` gives the canceller everything it writes,
on its writer thread as each chunk goes to the device, so a chunk whose write an
interruption cancels still counts. The `KeyedMicrophone` hears every buffer through the
canceller in PortAudio's capture callback, key up or down and at either place, so the
canceller keeps learning the room. The callback then lets the buffer through only while
the key is down. If the canceller raises there, the pipeline is ended and the run with it,
instead of PyAudio aborting the stream and leaving hands deaf. AEC3
wants one frame of reference for every frame of microphone, as a device that plays and
records at once gives them, but the pipeline writes only while it speaks, and up to an
output buffer ahead. So the canceller holds what was written, and each 10 ms of
microphone takes the next 10 ms of it, or silence when there is none. Each microphone
stream let go of is one `microphone.let_go` wide event, carrying how many frames its
canceller heard, how many of those had nothing playing, and how much of the speaker's
sound it dropped unheard. Every frame the microphone pushes carries the same sound as it
was captured, before the canceller (`KeyedAudio.captured`), and Whisper keeps it for the
frames a hold is made of, so each hold's `HoldHeard` line says its mean power in dBFS
on both sides (`levels.captured_dbfs`, `levels.heard_dbfs`): a transcript made of the
reply's echo left over reads loud captured and quiet heard, one of the room with nobody
speaking reads quiet on both. A level is null for digital silence, and at the phone,
heard through no canceller, the two are equal. Measured on
2026-10-03 through this transport: about 27 dB of echo removed. A press mid-reply with
nobody speaking left no word of the reply in 8 holds of 8, where the raw microphone made
one in every hold. "Stop. What time is it?", said from 50 ms after the press, kept
"Stop." in 8 of 8. The mute this replaced held the microphone shut until the reply's
sound had died away, and lost that first word in 8 of 8.

**A lost device is followed, not waited on.** PortAudio does not report a device
that disappears. Measured on 2026-09-24 against a CoreAudio aggregate device destroyed
mid-stream, the microphone's callbacks simply stop, a write to the speaker blocks for
good, and the stream still reports itself active. Before this, an unplugged headset left
the daemon running and deaf: the heartbeat said up, the next turn produced nothing, and
Pipecat's 10-second write timeout then marked the speaker unusable for the rest of the run.
macOS moves the default devices the moment the default one disappears, so the
`hands.voice.coreaudio` listener, which reports a change within about 17 ms, is the
signal. The follower in `hands.voice.devices` then reopens the whole transport, in this
order:

1. Both streams are detached, so a write that comes meanwhile is reported unwritten.
2. The speaker is closed before the microphone, since closing it releases the stuck write.
3. PortAudio is ended and started again, since it lists devices only when it starts.
4. Both streams are reopened on the new defaults, and the speaker is made usable again.

It then says `Audio moved: listening on …, speaking on …` through the system channel,
on the new speaker, or posts it if speech is down. Closing a microphone whose device is
gone takes 3 to 4 s, so the whole move took about 5.5 s from unplug to the sentence.
Each move is one `devices.moved` event: the devices it left, the defaults that moved it,
the devices it reopened on and the defaults it read as it did, which the next move is
measured from, and how long the reopen took; failed with what raised where the reopen
did, even when the run was stopping meanwhile.
The next turn ran end to end on the built-in devices. Every step that touches a device runs off the event loop.
A reopen that takes longer than 10 s, or one that fails, stops the run, which reads as down, and the next
`hands run` opens on whatever devices there are. The same path follows a headset plugged in, or a default changed in
Control Center. A Mac with no built-in microphone (a mini, a Studio) can
lose its only input device. Then the microphone holds a stream of nothing (`NoInput`),
which is opened, started, stopped, and closed like any other. The move is said as
`No microphone: hands cannot hear you. Speaking on …`. A daemon that starts that way
says `hands is up, but there is no microphone, so it cannot hear you.` instead of
failing its setup and stopping. With no stream, no frame reaches
Whisper and no turn starts, so the key edge answers a press itself:
`There is no microphone, so hands cannot hear you.` A microphone plugged in later
changes the default input, so the follower opens it like any other move. This is
tested against a PortAudio that lists no default input. A MacBook cannot be put in
that state, since macOS always falls back to the built-in microphone.

**The gate has one owner and several edges.** `PushToTalk` holds the key position;
whatever reads the physical world calls `PushToTalk.move`, naming itself. The edges are
values of one type, `Edge` (`hands.voice.trigger`), not modes of the gate
`[LAW:one-type-per-behavior]`; the gate keeps the one that opened the last turn, and each
turn's `UserAsked` says it:

| Edge | Down | Up |
|---|---|---|
| `held key` | Right Shift held alone for 600 ms, in any app (`hands.voice.hold`) | released; another key pressed while it is held drops the turn unsent |
| `engaged conversation` | engaged by one hold of Right Shift; then Silero confirms speech on the desk microphone (`hands.voice.engaged`) | Smart Turn judges the speech complete, or the silence after it runs past its `stop_secs`; another hold disengages |
| `button` | a HID button or headset button pressed | released |
| `phone button` | the phone page's talk button pressed | released |
| `wake word` | openWakeWord hears "Hey Jarvis" on the desk microphone (`hands.voice.wake`) | as `engaged conversation`'s, once what is asked has started after the wake word |

The wake-word edge is the only one that opens the mic without a hand, and it is
half-duplex: while hands speaks, the wake-word detector hears silence in place of the
room, because an open mic in a room with speakers hears the pipeline's own voice. The
pause after "Hey Jarvis," is no end of the turn: Smart Turn judges the vocative
complete, so a stop goes to it only once speech has started again after the wake word.
The driver hears afresh from the wake (`Ears.afresh`): Silero is quiet until sure of speech
again and Smart Turn holds none, so what is asked starts as any speech does, after a
pause or in the same breath, and Smart Turn judges it alone. The edge shares engaged
conversation's driver and models (`hands.voice.engaged.drive`); the desk listens for as
long as it is in use, so the wake word is in the turn's audio for Whisper. The model's
files are fetched into the home's `wake-word` directory by `set_trigger` before the switch
(`hands.voice.trigger.readied`): a fetch that fails refuses the switch out loud and the
trigger in use stays.

**The trigger is the desk's edge, one choice switched by voice** (`hands.voice.trigger`).
The phone's button is there for every call; at the desk, the edge that drives the gate is
the `Trigger` in use, one value `Triggers` holds while hands runs: the brain's
`trigger_in_use` says it, and `set_trigger` switches it: the old edge stops and the new
one starts (`Triggers.drive`), so the next turn opens the new way. Built: the `held key`,
`engaged conversation`, and the `wake word`. A trigger not built is refused by the tool's closed set, and the one
in use stays.

**An engaged desk listens between turns.** Engaged conversation moves the gate with the
held key's own moves, plus two of its own: `listen` as it engages and `deafen` as it
disengages, each cued with two tones. While the desk listens, the key rests at `listening`
between turns rather than `up`: the microphone's bytes reach Whisper, which keeps the last
second of them as Pipecat keeps audio nobody is speaking in, so the turn the voice opens
starts with the words said while Silero made sure of them (about 0.2 s). Listening is the
desk's alone: at the phone the key rests at `up`, what the desk hears opens no turn there,
and the desk listens again once hands is back. Both models hear the desk through the echo canceller; measured on MacBook speakers, a
canceller that has heard one reply lets none of the next trip Silero, where the raw
microphone tripped it at every phrase, and only the first two seconds of a fresh
canceller's life let one through.

**The phone is a second place, beside the desk.** The desk is the Mac's own mic and
speakers; the phone is a page hands serves (`hands.voice.phonepage`) that a phone
opens over the LAN or the tailnet, with a talk button. Both are there for the whole
run, with no setting to choose between them: the gate holds the place the last turn
was opened at, and that place is where hands is. The pipeline hears only that place's
microphone, so Whisper gets one stream of frames, and the speaker plays to that place,
so a reply, a session's news, and a turn's tone go where the user is
(`hands.voice.ptt`, `hands.voice.microphone`). A call connecting moves hands to the
phone, and the call ending moves it back to the desk in the same step, so nothing is
played to a phone that has gone. An offer answered but not yet connected moves
nothing, so a page that cannot reach hands never holds its speech; the newest offer
lets go of any older one not yet up, and replaces the call that is up once it
connects. The other place cannot touch a turn that is not its own: a Shift typed at
the desk while the user talks on the phone arms nothing. A turn opened at one place
while a hold is open at the other drops both, as a key pressed while the talk key is
held drops the turn, and the gate cues and tells the moves as it took them. Only a
turn opening moves hands, never a press still arming, which may be Shift: so a hold
at the place hands is not at keeps none of the words said before it opened a turn,
and typing at the desk never takes a call's replies off the phone. Each move of hands
between the places is one `place.moved` event: from where to where, whether a turn or
a call made it, and whether a hold open at the place it left was thrown away. A move
a call made is part of that call's trace.

**Cues say what hands is doing while it is silent** (`hands.voice.cues`). The talk
key's edges are cued as the key moves, whether or not anything downstream takes the
turn. Two more cues are for silence: a high rising pair when a turn's words are written
to the model's context, which a hold that heard nothing or was dropped never reaches,
and one steady tone when hands acts: at each call of one of its tools but `stay_silent`,
and at each burst of progress that is heard, so never another session's, a muted one's,
or any while hands is quiet. They are owed into one queue and played once nobody is
speaking, as Pipecat's started and stopped speaking frames say for hands and for the
user, and once the cue's spacing since it last played is up: the working tone sounds at
most every two seconds, the receipt is never held behind it, and what is owed meanwhile
plays as one. Each play is a `Cued` line in the audit log, with how many it stood for,
how long it was held, and where it went, which is nowhere while no speaker is attached.

The call (`hands.voice.phone`) is one WebRTC connection made with aiortc directly.
hands' speech goes to the phone on an audio track. The phone's microphone does not:
the page sends it as plain 16-bit audio over the call's ordered data channel, in
order with its button's presses and releases, and holds a release back until every
block its microphone captured before it has been sent. A release that overtook the
words said before it would cut the end of the turn; on one ordered channel the key
travels with the audio, as `KeyedAudio` makes it travel at the desk. Earbuds keep
hands' voice out of the phone's microphone, so the phone's audio is gated by its
button alone.

The same channel carries, the other way, what became of each hold. The latency
observer (`hands.voice.latency`) tells the phone each mark of a turn as it logs it
(`hands.voice.mark`): `released` or `discarded`, `transcript` or `no words`,
`first LLM token`, `first audio`, and `failed` where a stage failed first. The page
shows the last one under its button, with the time since the button was let go
counting while a reply is waited on, and the time to the first sound once it
comes. Every time shown is the page's own clock, from the button let go to the
mark arriving, so it is the wait as felt at the phone, the network included.

`no words` is a fact of the turn and not of a hold: the user aggregator says the
turn ended, and no hold it took in had words, so nothing is sent to the model. A
silent hold released after one whose words are in is answered with the rest of its
turn. `failed` is a pipeline error that leaves the release unanswered: one after
the turn was sent, which is its reply failing, or one while the turn was taken in
where the turn then ends with no words, which is Whisper failing on it. Both end
the turn's latency window, so what hands says next, the failure itself included,
is not timed as the answer. `first audio` is the first sound after the release,
whatever hands said with it, and the page says only that hands spoke.

A mark carries no hold's number, so the page places it by where it falls: each
mark names the marks it may come after, and until hands says it took the hold
shown or threw it away, nothing else told is of that hold. A mark of a hold the
page has let go of is not shown. A call's `phone.call` event counts the marks its
page was told.

The channel also carries the transcript (`hands.voice.transcript`), which the page
keeps under a button, hidden until asked for. Every message on it is one JSON
object whose `kind` says which it is: a `mark`, or a line. The user's words are
the line Whisper heard of each hold. Each of hands' sentences is told twice: as the
output transport writes its first audio, and as it lets the sentence's end go, or a
barge-in cuts it off. Its text alone is no start: the first sentence of a reply is
let go before its audio is made. pocket-tts reports no word timings, so hands
measures its speaking rate off each sentence played to its end and tells it with
the next; the page sweeps the highlight through a sentence's words at that rate,
holds on the last word until hands says the sentence is done, and strikes through
the words a barge-in, or the call ending, cut off. The `phone.call` event counts the
lines told beside the marks.

A browser gives a page the microphone only over HTTPS. Under the tailnet name,
hands shows the certificate `tailscale cert` issues for it, which the phone trusts
as it is; under a LAN address it shows a self-signed one, which the phone is asked
once to accept, and which hands makes again at a start within 30 days of its end;
the tailnet's is asked of Tailscale again daily while the page is served. Serving it is one
`phone.served` event, naming the tailnet name it is served under, or why it is served on
the LAN alone; it follows the pipeline's start, so it is no part of `hands.start`. The page is open to anyone who reaches the port; a call is not: an
offer must carry the phone's key, a secret in the home that travels in the page
address's fragment, which a browser never sends with the page request. `hands phone`
prints the addresses with the key, the first as a QR code. aiortc never notices a
browser closed outright, so a page that sends nothing for `QUIET_SECS` is hung up:
it sends its microphone every 20 ms for as long as it is open. The page does the same the
other way: hands sends its voice every 20 ms, silence and all, so a page that has received
none for 3 s, or whose channel closes or connection fails, takes its call as dropped and
calls again on its own, after 1 s, then doubling to every 8 s, at once when the page is
brought back to the front. A press whose call never opens is said and not made again: the
user is there to press once more. It keeps the microphone open between calls, because a
browser opens one for a press, not for a page calling on its own, and ends when the
microphone is taken away, since no call over it would be heard. Every offer names
the page, by a name it makes itself as it opens, and what it asks: a press of Connect
`take`s the phone from any page, and a page calling again `resume`s it, which hands
declines with a 409 while another page has the call or an offer in. Then that page
keeps the phone and this one stops calling until Connect is pressed. So two pages left
open never take the phone back and forth, and a page whose old call hands has not yet
hung up replaces it. Each call is one
`phone.call` event, the root of a trace of its own, from its offer to its end, written
as it ends: refused at the page, for no key or no offer; declined, as a resume; let go
before it connected; or left after hands was at it, with how long after its offer it
arrived; and why. Each one read names the page and what it asked (`asked`). One whose
body could not be read, or that hands could not answer, is failed with what raised.

**One audio owner.** Pipecat's output transport is the only thing that plays sound.
When two sessions finish at once, their utterances line up behind it instead of
overlapping.

## Loud failure

In an audio system the default output is silence, and silence is what "thinking"
sounds like too. So every failure has a path to the user that does not depend on the
thing that failed `[LAW:no-silent-failure]`:

1. **Speech.** The system channel says "the language model is unreachable", "speech
   recognition failed for that turn", "the session cc-hands is gone". These are
   `Speak` effects and need no model. `hands.voice.system` renders each fact from a
   template and queues it at the TTS processor, past the LLM and out of its context,
   so the model never reads a system line as a reply it gave. The worker's
   `on_pipeline_error` routes every error by the processor that raised it: the LLM's
   become "unreachable", "usage limit reached, until <when it lifts>" (read from the
   API's message, never spoken from it), or "failed: <category>", Whisper's become "speech recognition failed for that turn",
   and a TTS error goes to the screen. Pipecat files an SDK connection error
   under UNKNOWN, so "unreachable" is recognised from the exception type. A model reply
   with no words and no call in it is the model's failure too, "sent back nothing": the
   API services read it off the frames they push (`EmptyReplyFails`), excusing the reply to a call's result, and
   the brain's stage off the wire across all of a turn's replies; a turn the model ends with stay_silent made a call, so it stays silent,
   and a reply cancelled mid-stream was abandoned, not empty. An Anthropic request that times out, which Pipecat
   drops with only an event, is said as unreachable. A turn Whisper
   transcribed to nothing is not said (Brandon, 2026-09-27: "I do not need to hear
   it"), only logged, and so neither is a muted microphone, whose every turn comes back
   empty. The `Whisper` subclass resolves the hold as the transcription ends, so the
   turn closes then rather than after the user turn's 5-second stop timeout. The
   channel says a burst once: a fault is not said again until ten seconds have passed
   since it last was. A fault that recurs recurs in bursts, and on 2026-09-22 a held
   key queued hundreds of empty turns whose reports then went out every 0.43 s for as
   long as they drained — which is not a loud failure but a
   jammed one, because nothing else could have been heard while it ran. What ends a
   burst is a span of quiet and not some other announcement, because in the case this
   exists for there is never another one: with no microphone, staying quiet
   until something else was said would leave a user pressing the key at a daemon that
   has gone permanently silent. What a burst costs is the saying and never the
   knowing — every occurrence is still a log line. `Announced` is written where the
   sentence was taken, handed to a working TTS or accepted by the screen, so a
   notification the screen refused is a logged failure rather than an announcement.
   How often hands tries and what the user was given are two facts and are kept
   apart: the window is taken when the channel decides to speak and not after it has,
   because Pipecat dispatches each pipeline error on its own task, so a dead TTS
   raising once per queued frame puts hundreds of those decisions in flight together.
   The window is taken on the fault and never on the sentence: Pipecat's text for a
   silent utterance carries a fresh context id every time, so `Post` holds that text
   as what varies and the fault it reports as what recurs, and only a closed set — a
   `SystemFact`'s own sentence, or that fault — is ever a key. The LLM clients do not
   retry: against inferno, the SDK's two retries stretched a refused connection into
   4.6 s of silence. Measured against a refused port on inferno, the failure is heard
   1.75 s after the key release. The pipeline's start is spoken too: "hands is up",
   or "hands is back after a crash" when the last heartbeat names a pid that is gone
   without having said `stopped`.
2. **Screen.** The daemon writes `~/.hands/status.json` every heartbeat with its pid,
   uptime, pipeline state, last audio out, the count of live sessions, whether a turn is
   open, and each way it is running degraded (the microphone is open on no device; the
   collector has not taken all of a signal's last few batches), each in its own words. `hands status` prints it, saying every degradation. The
   daemon running is its one writer: a run locks `~/.hands/hands.lock` before it reads the
   heartbeat or listens, so a second `hands run` on a home a daemon holds is refused there,
   says that hands is already running and the pid the lock names, and leaves the heartbeat
   and the sockets alone; only its failed `hands.start` reaches the log. When TTS itself is down, a macOS notification is posted through
   `osascript`. `hands indicator` is a menu-bar status item in a process of its own,
   which `hands run` starts in a session of its own, so neither the daemon dying nor the
   terminal's Ctrl-C takes it down first. It lives while the process that started it does,
   and once that is gone, until the light leaves up: it posts that notice and exits. A
   restart keeps the process, and the run after it ends the indicator the run before showed
   (SIGTERM to its process group, which it ends on as it would by itself, then SIGKILL to the
   group if it has not within a second) and starts its own, on the code it runs. Once a second it judges the heartbeat through `heartbeat.look`, the one
   read-and-judge that `hands status`, the crash check at start, and the hook shim also use. Its title
   shows one of six lights: up, not
   responding, down, refused (a start that ended before it ran, with its reason), off (stopped or never ran), and unreadable. An unreadable heartbeat
   is warned of as loudly as a dead daemon. Up with any degradation is still the up light, its title
   a warning naming every degradation, and a turn open beside it; a daemon with no microphone never says a turn is open. It posts a notification when the light
   leaves up for a warning, or a degradation the look before did not show arrives, at most once a minute, so a loop that
   stalls and recovers over and over is not announced every time. A departure inside that
   minute is held, and posted when the minute is up if what it held still holds: hands still not up, or the degradation still there. A daemon it finds already
   down on its first look is shown but not announced. A heartbeat whose pid is outside
   macOS's `1..99999` does not parse, since no process can have it. A pid counts as the
   daemon only while the process holding it started no later than the heartbeat's
   `started_at`, with a second of slack for a clock stepped back. The start comes from
   the kernel (the `kern.proc.pid` sysctl that `ps` itself reads, about 10 µs), and it
   is the rule the session sweep uses, from `hands.sessions.processes`. A pid that a
   later process took, after a crash or a reboot, reads as down rather than as not
   responding.
3. **Log.** Every effect and every failure is one line in the segmented log
   `~/.hands/audit/`, written by the daemon and by each `hands` command, under one lock
   (`hands.sessions.audit`). Each
   segment is named by its base offset; a line that would take the active segment past
   32 MiB rolls the log to a new segment, which opens with a `Rolled` line, and
   retention deletes all but the segment just closed and the new one. A segment is never
   renamed and, once a later one exists, never appended to, so `hands log` prints the
   newest lines and follows the log across a roll without losing one. Each line is a value encoded one way: its type
   under `"type"`, its fields beside it, nested events and effects alike, and the
   wall-clock time under `"at"`. `Sessions` is the single writer for the session
   side: one `applied` wide event for each event or voice answer the registry applies
   (only one that changed the registry or called for an effect, so a quiet tick is not
   a line), open until its effects are performed, carrying what was applied, each
   effect as a `Performed` with its outcome and time (`Audit` records such as
   `Unregistered` among them), and the effects counted by kind; applied inside a hook
   post, it is that post's child, and a unit an effect opens is its own. One wrapper makes every tool call a
   `tool.run` wide event with its arguments and the result the model was handed, and,
   called in a voice turn, the child of that turn's `tool.call` span; the context
   aggregators write each user turn as `Transcribed` and each reply as `Replied`
   (a line hands says as written and keeps in the context is a `Replied` of its own
   once it is said, and part of the model's where it is said inside the model's turn,
   which runs through its calls to the reply that answers them:
   `hands.voice.conversation.AssistantTurns`);
   the system channel writes `Announced` with whether it spoke or posted; each thing a
   session gives hands to say unasked is an `utterance` wide event, from heard to its fate; and a
   loguru sink turns every error a `hands` module logs into a `Failure`. A dictation is
   traced from the words to the readback: `Transcribed`, then the `tool.run` of
   `stage_draft` with what hands said back, which no model rewords. The log watches
   and never steers: a line the disk will not take is lost with a warning on stderr,
   a value it cannot encode is a `Failure` line instead, and the draft, the question, or the tick it described goes on. `hands log` follows
   the file by inode and offset, so a log moved aside is read from its first line.

The daemon runs in the foreground of a terminal, so a crash is seen there, as down by
`hands status`, and as a notice from the indicator. Nothing restarts it: running it
again is `hands run`, which re-reads the session files and speaks that it is back. A
hook shim that cannot reach the socket fails visibly in the target session, unless the
heartbeat says hands was stopped or never ran, so a daemon that died or hung is loud in
every session while one that was stopped costs them nothing. Two clocks are never allowed to
disagree about whether the daemon is up: the heartbeat file is written by the daemon
alone, and everything else reads it.

The heartbeat is honest at both ends of a run. `hands run` writes its first heartbeat,
`pipeline starting`, before it imports Pipecat. The models load off the event loop,
which keeps beating, so a new run shows its pid within half a second of starting,
and only a loop that is actually stuck reads as not responding. A SIGTERM,
Ctrl-C, the `q` key, and a failed background task all set one quit event. Its handler
is in place before the models load, so a stop during the load does not wait for them.
Shutdown lets every permission hook still waiting go undecided, including one that
arrives as shutdown begins, so that session's own dialog stands and the daemon exits
in under a second. After the socket is released, a stop writes a last heartbeat that
says `stopped`, and a stopped daemon reads as stopped even if its pid is later reused.
A crash writes nothing more, so its last heartbeat names a pid that is gone, and it
reads as down. A background task that failed or a pipeline that ended on its own
counts as a crash: the run raises, exits nonzero, and reads as down until it is run again.

A start is one `hands.start` event (`hands.daemon.starting.Start`), timed from the moment
`hands run` began to the moment the pipeline reported started, so its duration is how
long hands took to be ready. It says which run it is (`pid`, `restarted`, `after_crash`, and on a restart what became
of the `previous_indicator`: ended, killed, exited on its own, or reaped already),
which settings won (the file, the collector, the backend, server,
model, and account, the voice), and what the run listens on (the hook socket, the proxy
and its upstream, the tap, the display route), each added as the step that learns it is
taken, so a start that ended first says how far it got; the display's is the address it
was bound on. No one body runs a start, so its event is emitted as it ends (`wide.ended`)
rather than held open as a unit over the run: held open, every server and task the start
makes would be inside it, and every unit they ran part of the start's trace. A start told
to stop first is cancelled. A start that cannot be made is failed, with the reason, and
says why on its terminal too; one that anything else ends first is failed with what raised. Refused at
the door (the talk key's grant, `config.toml`), a run holds no heartbeat yet and writes
none, so the one there, a running hands' or a crash's, stands; a restart's run holds the
heartbeat from the outset, its predecessor having beat starting under the same pid.
Refused once it holds the heartbeat (the backend's key, the voice, the brain), its last
heartbeat says `refused` with the reason, which `hands status` and the indicator show,
and which the next start reads as no crash.

## Wide events

A unit of work runs inside `hands.sessions.wide.unit`, which leaves exactly one
`WideEvent` however the run ends: ok, failed with what it raised and the frames it
came up through, or cancelled `[LAW:nothing-unseen]`. Code inside the run calls
`annotate` to add a fact and `count` to add to a count the unit declared. It never
emits anything itself. A run that fails without raising calls `fail`, and its event
ends failed with that error. A part of the run that was timed where it happened, such
as a request another process made, is emitted with `child` as its own event under the
unit. A declared count the run never added to is written as 0, so a
run that did nothing reads differently from a run that never happened. A unit opened
inside another shares its `trace_id`. A task that outlives the unit it was started in
cannot add to the event once it is emitted: it is refused with a `LookupError`.

The event leaves through the `emit` the unit was opened with, which in the daemon is
the audit log's `record`. The audit log is the one export edge, and an event is one
more line in it. So a run's facts go on its event and never on a hand-built line beside
it. When a unit of work that already writes such a line is moved onto the floor, that
line type is deleted, as `DeltaRead` was. Each event carries its own `span_id`, and a
unit opened inside another names that one's as its `parent_id`, so a run's units are one
trace.

With `[telemetry] collector` set, the export edge also sends each event to that
OpenTelemetry collector over OTLP/HTTP, carrying `service.name=hands`
(`hands.sessions.otlp`), twice over: as a span, for the trace store, and as a log record
whose body is the event's name and which carries the span's ids, for the event store, so
an event's row opens its trace. The event is written to the log first, whatever becomes of the
collector, because hands reads the log back (`catch_up`, `hands log`): what it holds never
depends on the network. Events are sent in batches, as each signal from a thread of its
own, so a slow or absent collector costs a unit of work nothing and holds up neither signal
behind the other. Each batch sent as each signal is an `Exported` line naming the signal,
each event by its span id, how long the send took, where the collector did not take it,
why, and where it took it with a warning, the warning `[LAW:nothing-unseen]`; a stop waits
on the batches still queued for one timeout in all, and records those it leaves unsent.
Those lines are folded, per signal, into how many batches in a row the collector has not
taken all of (`otlp.failing`); from the third on, every heartbeat hands beats while up,
starting included, carries it as a degradation naming the signal and since when (why each
batch was not taken is its own line, so a failure worded anew is not news again), until a
batch is taken or the collector is unset. The `hands.start` event names the collector. Only
`hands run` sends to it. Every other command is one `hands.command` event, written by the
dispatcher every command passes through (`hands.daemon.cli.commanded`), carrying the command,
the home it ran on, its arguments as parsed, its exit code, and how long it took; it ends failed where the command
exits nonzero, and the unit of work the command ran, such as `plugin.render`, is in its trace.
Those events are in the log alone: Claude Code waits on `hands plugin` before a session starts,
and `hands status` must answer whatever the config says, so no command waits on a collector or
reads a config.toml to find one. hands names only the collector; which stores sit behind it is
the homelab's.

## Endurance

The intermediary talks with you for hours, and its own context is the one thing in
the system that grows. Two mechanisms bound it, and both were written at the same
time as the thing they bound (failure mode 5).

Pipecat's context summariser, configured with `LLMAutoContextSummarizationConfig`,
compacts the conversation when it crosses a token threshold, keeping the recent
turns verbatim. That is the eviction story. And because every narration, every tool
call, and every user transcript is also a line in the audit log, nothing that was
evicted is lost: `catch_up(minutes)` replays what was said while you were away, and
the brain recalls the rest with its own Bash and its `hands:recall` skill, running
`hands recall WORDS...` (`hands.sessions.recall`). That prints, one line each, what you
said, what hands said, what was typed into a session, and how a session's permission was
answered, keeping the newest that hold every word; hands adds no tool for it. Each recall
is a `memory.recall` event on the log, with how many lines it read and moments it matched. The log is the long memory; the context
window is the working memory; the same pull-not-push rule that governs session
transcripts governs the intermediary's own past.

## Configuration: one file, parsed once

`config.toml` in the home (`~/.hands`, or `HANDS_HOME`) is read once, by `hands run` before
its first heartbeat, and parsed into a frozen `Config` (`hands.daemon.config`). An edit is taken up the way
everything else on disk is, by the run starting again on it: the watch weighs the
file against the bytes the start read, and an edit that parses ends the run as the
restart signal does `[LAW:single-enforcer]`, said as `SettingsEdited` before the
next `hands.start`, which says it was `restarted`. One that does not parse, or names a backend whose key or login the
start's own `backend` check refuses, is said as a `SettingsEdited` that was refused,
and the run keeps what it has. It is in the home, beside
everything else one hands keeps, so a second home is a second hands with settings of
its own. A file left out, or a key, is the default; a key misspelled, or one its
variant has no use for, stops the start naming it. Secrets come from the environment
and nothing else does: an API key, or the keychain's when the server is Anthropic's
own. The spike's `HANDS_LLM*` and `HANDS_WHISPER_MODEL` variables are deleted, so there
is one source `[LAW:one-source-of-truth]`. `HANDS_HOME` is not a setting: it says where
the settings are, and it is the one thing a hook, run by Claude Code with no arguments
of hands', can be told.

The settings cap `[LAW:no-mode-explosion]`: each config field names a variant or a
number. There are no boolean feature flags. A field that would be a flag is either a
variant with a real alternative or it does not exist. The fields:

| key | what it names |
|---|---|
| `[llm] backend` | `anthropic` (the default), `openai`, or `claude`, the brain |
| `[llm] model` | the model, for any backend |
| `[llm] url` | another server that speaks the API, for `anthropic` and `openai` |
| `[telemetry] collector` | the OpenTelemetry collector's OTLP/HTTP address each wide event is also sent to |

The voice is not a setting: the user chooses it by voice while hands runs, and it is
kept in the home's `voice` file. The permission timeout is declared in the hook
config, which `hooks.json` is generated from.

## Stack

Pipecat ships every piece the pipeline needs: an in-process pocket-tts service, a
local audio transport over PyAudio, a segmented STT service for upload APIs, the
SmallWebRTC transport, the Silero VAD, an OpenAI-compatible LLM service and an
Anthropic one, function registration from the `LLMContext`, `TTSSpeakFrame` for the
system channel, and `LLMMessagesAppendFrame` with `run_llm` for the other two.
Verified against the installed Pipecat 1.10 on 2026-09-12.

pocket-tts is MIT, 100M parameters, CPU-only by design, reports no word timings, and
streams: measured on this
Mac, first audio 87 ms after the text arrives and about 5.6x real time. Whisper is
large-v3-turbo on MLX, in hands' own process (`hands.voice.transcription`), loaded as the
pipeline is built (the `whisper.loaded` event): the transcript lands about 0.3 s after the
key's release. LowTalker (~/code/low-talker) serves the same weights from the Neural
Engine, and took 0.6 to 0.7 s for the same holds, by upload and by its Realtime socket
alike (2026-10-04, hands-dictation-2bs.kpm), so hands keeps its own. The default
LLM is Claude Sonnet 5 through the Anthropic API, or any server that speaks it; OpenAI's
chat completions API is the other backend variant. Measured on 2026-09-12, full
voice-to-voice with Qwen3-30B-A3B on inferno, since retired: a turn with a tool call had
first audio 4.3 s after key release; a plain turn 1.4 s.

Python, `uv`, pyright strict. State types are discriminated unions: frozen dataclasses
with a `Literal` kind or a union of frozen dataclasses.
