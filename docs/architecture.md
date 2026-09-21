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
is exactly one of each: one replies to blocked hooks, one owns the speaker, and one
types into sessions, through the fritter that wrapped them.

## Shape

```
                                  hands daemon
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │  voice                                                                        │
 │  mic ─► gate ─► Whisper (MLX) ─► LLM ─► pocket-tts ─► speakers                │
 │         ▲                        ▲  │        ▲                                │
 │   gate edges              notes, │  │ tool   │ system speech                  │
 │   terminal · hotkey       narrate│  │ calls  │ (straight to TTS)              │
 │   button · web · wakeword        │  ▼        │                                │
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
process-liveness check, and the audit log. It
parses hook input once at the socket into a `HookEvent` and rejects anything it does
not recognise with a logged error and a non-2xx reply `[LAW:parse-dont-validate]`. It
exposes two things upward: an async stream of events, and a small API the tools call.

**`voice`** is every edge on the audio side: the Pipecat pipeline, the gate and the
edges that drive it, the audio transport, the STT, LLM, and TTS services, and the tool
functions. It consumes the sessions event stream and turns each event into one of the
three speech channels below. It never imports from `daemon`.

**`daemon`** is the composition root: it parses the config file into a frozen
`Config`, builds the sessions package and the voice package from it, runs both under
one supervisor, publishes the heartbeat, and provides the `hands` CLI. Nothing below
`daemon` reads the environment or the config file `[LAW:one-source-of-truth]`.

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

# What the shim records at SessionStart. The title is not here: it is the newest
# ai-title record in the transcript, read when a listing needs it.
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
    mode: PermissionMode      # from each hook payload that carries it
    state: SessionState

# One session is in exactly one of these. A session waiting on you and a session
# working are different things, and the type says which.
SessionState = Idle | Submitted | Working | Blocked | AtDialog | Gone

# Sent, but not taken until its UserPromptSubmit hooks finish; an Escape before then
# cancels it with no hook or record, so only the transcript's record of the turn makes
# it Working, and only Claude Code's idle ends it.
@dataclass(frozen=True)
class Submitted: since: Instant
@dataclass(frozen=True)
class Working:   since: Instant
@dataclass(frozen=True)
class Idle:      last: Uuid | None        # the record that ended the last turn
@dataclass(frozen=True)
class Blocked:
    on: Blocker
    request: RequestId                    # the shim call waiting for a reply
    deadline: Instant                     # derived from the hook config's timeout
    warned: bool                          # the deadline warning has been spoken
@dataclass(frozen=True)
class Gone:      reason: Literal["exited", "terminal_closed", "pid_dead"]

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
class Type:     session: SessionId; input: Input             # through fritter; unbuilt, hands-keyboard-gxr.i5n
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
class Compare:  session: SessionId; mark: Marked              # what it changed, read when the turn stops
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
and the narrator reads, summarises, and speaks the turn without returning to the
reducer.

The adapters live in `sessions` and `voice` and each performs one effect kind: `Reply`
writes to the blocked shim's socket connection, `Speak` becomes a Pipecat `TTSSpeakFrame`,
`Narrate` and `Note` become
`LLMMessagesAppendFrame` with `run_llm` on or off, `Play` sends a segment to TTS
through the player, `Summarise` calls the summariser, `Snapshot` records or diffs the
target's git state, `Audit` appends one JSONL line. An
adapter that fails raises; the supervisor logs it and the failure is spoken through
the system channel. Nothing is retried silently and nothing falls back
`[LAW:no-silent-failure]`.

That block is the design, not the code. `core/effects.py` has six of those ten today -
`Audit`, `Reply`, `Speak`, `Narrate`, `Summarise` - plus `SessionGone`, which the block
above leaves out. `Type`, `Note`, `Play`, `Snapshot` and `Launch` are declared and unbuilt,
and their adapters are named here in a tense the code has not earned yet. `Type` is the
nearest: what it will call is built and measured - `hands.sessions.typing.Typist` types
into a session's fritter - and only the effect and its place in the reducer are left, in
`hands-keyboard-gxr.i5n`.

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
| When a permission deadline warns and expires | the reducer, from `Blocked.deadline`, driven by one `Tick` source |
| Whether text typed mid-turn is queued or lost | Claude Code's own input queue, measured to queue it |
| When the daemon is up, and restarting it | you, with `hands run` in a terminal |

Deadlines are data. The `Blocked` state carries the instant it expires and whether
the warning has been spoken. A single ticker sends `Tick(now)` once a second; the
reducer compares, and emits `Speak("ten seconds on that permission")` exactly once,
because the transition from `warned=False` to `warned=True` is a state change, not a
timer callback. At the deadline it emits `Reply(deny)` for a permission and says so;
a question, which silence cannot answer, is withdrawn instead and left to its dialog,
where the user may be answering it at the keyboard. The session is then `AtDialog`: still
waiting, now on the keyboard alone, until the answered call comes back. The ticker's
period only bounds how late a deadline is heard; no correctness property depends on
a `sleep`.

The deadline itself comes from one number. `hands.sessions.hookconfig` declares the
`PermissionRequest` hook's timeout (90 seconds) in the plugin's `plugin/hooks/hooks.json`; the shim
waits on the daemon that long for that hook alone, and the daemon denies 5 seconds
earlier, so the deny reaches Claude Code before Claude Code kills the hook
`[LAW:single-enforcer]`.

Claude Code queues messages submitted while a turn is running and shows them with
"Press up to edit queued messages". Measured on 2.1.270: text pasted into a working
session and submitted lands in that queue and runs when the turn ends, so a send to a
working target is an ordinary send and the daemon holds nothing. A permission dialog
is the exception: it swallows pasted text and takes the Enter as "Yes". So when the
drafts are sent, a send to a `Blocked` target is refused, the draft stays staged, and
the user hears why.

The workspace-trust dialog swallows a paste the same way, measured on 2.1.278, and that
rule does **not** reach it. `Blocked` has one producer, the `PermissionRequest` hook, and
no hook fires for a trust prompt: the session reads as `Idle`, fritter sees an empty input
box and types into it, and the send is answered `ok` while the draft vanishes. It is a
second rule and it is unbuilt — `hands-harness-5nb.xw8`.

### Typing into a session

A session is typed into through **fritter** (`fritter/`), which runs its `claude` on a
pseudo-terminal and listens on a unix socket beside it. A wrapper and not synthetic key
events, because a keyboard types into whatever has focus, and the requirement is a
session driven with the display asleep: no window, no grant, no focus.

fritter publishes its socket's address to the process it wrapped in `FRITTER_SOCKET`.
The hook runs as a child of that process and inherits it, so the address reaches
`Membership.fritter` without either side deriving a path from a pid. A session started
outside fritter has no address, and `Typist.of` refuses it by name rather than writing
into nothing.

Two things are divided rather than duplicated. hands decides *whether* a session may be
written to, from state fritter cannot see. fritter decides only whether the person at
the keyboard has characters in the box they have not sent, which hands cannot see because
those keystrokes never reach it; text arriving then is refused with a reason. A key is
not, because a key does exactly what the person pressing it would do and cannot
interleave with anything, and because Enter and Ctrl-C are the keys that give the line
back - gating them would leave a held session reachable only by a human at the physical
keyboard, which is the case fritter exists to remove. And escaping stays here: what a
leading `/` means is `Input`'s business, and fritter types the text it is given.

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
announcements and the idle nudge as `Speak` and permission requests as `Narrate`,
relayed as soon as the reducer emits them. The nudge is the `idle_prompt`
notification, the only one the `Notification` hook's matcher lets through; it is
spoken once per idle period, because `Idle.nudged` turns true as it is said and every
way into `Idle` builds a fresh one. Claude Code sends no `idle_prompt` after an
interrupted turn, so that one idle period carries `Idle.due`, and the tick nudges it
when the notification would have come. `Story` carries finished turns and sessions gone in one ordered
queue, because a summary takes seconds, and an end spoken at once was heard before the
last turn it ended. A turn's summary reaches TTS as one `TTSSpeakFrame`, with no player
and no segments. `Heard` also carries a mode change as a `Note`, which enters the intermediary's context
with `run_llm` off. The player, the routing table, the overlays, the priority
queue, and `coalesce` below are planned.

The routing table is a value in `core`:

```python
Route = Literal["speak", "play", "narrate", "note", "drop"]
DEFAULT_POLICY: Mapping[EventKind, Route] = {
    "stop": "play", "progress": "note", "blocked": "narrate", "subagent_stop": "note",
    "idle_prompt": "speak", "gone": "speak", "session_start": "note", ...
}
```

A per-session overlay of `focused | normal | muted` is a second table over the first:
a muted session's `play` and `narrate` become `note`, and a focused session's
`progress` becomes `play`. Adding a new event kind is a new row, and adding an overlay
value is a new column `[LAW:one-type-per-behavior]`.

Pending speech is a priority queue in `voice`: `blocking` before `result` before
`fyi`, and nothing starts while the key is down. Before an utterance plays, a pure
`coalesce` pass folds pending items from one session into one narration whose headline
covers them all and whose segments keep their record ids, so three `Stop`s that
arrived while you were talking start with one sentence, not three. Each item is a
transition keyed by the record id that caused it, so nothing is announced twice
`[LAW:one-source-of-truth]`.

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
2.1.270 on 2026-09-14 carry no session title, so a session's title is the newest
`ai-title` record in its transcript.

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
`UserPromptSubmit` carried. The tail reads that record as the `Interrupted` event. It
ends nothing: Claude Code sets the session `idle` ~100 ms before it writes the record
(2.1.282), and that status is what ends the turn (see Sessions, below); the record is
what the turn's telling waits for. Only the prompt names the turn: a background
subagent's hooks keep the `prompt_id` of the turn that started it after that turn is over.

Only a `Stop` and Claude Code's status move a session out of `Submitted`, `Working`,
`Blocked`, or `AtDialog`. Hooks and records name turns and fill them in, and never end
them. A turn opens from the prompt: a `UserPromptSubmit` applied to a session at its
prompt opens one, sent, and marks it. A message queued into a running turn fires
`UserPromptSubmit` with the running turn's `prompt_id`, the id that turn went on under
after a flush included (2.1.281), so one applied to a busy session is in the turn it names.
Any other id it or a record carries while a turn runs joins the ids the turn goes by: a
flush's, taken seconds before Claude answers under it, which a message queued in between
carries; or, should Claude Code's idle go unread between two turns, the next turn's. A
`Stop` carries the `prompt_id` of the turn it ends, which is how that turn is found in the
tail, and it ends only a busy turn that goes by that id, so a `Stop` applied late never
ends the turn after it. A `Stop` at the prompt tells a turn hands never had running,
such as the one a session was in when it was attached, and ends nothing of the last one,
which was told already. A turn a background task's notification opens fires
`UserPromptSubmit` with an id of its own, as a typed prompt does.

The reply a `PermissionRequest` hook may give is printed on its stdout as
`{"hookSpecificOutput": {"hookEventName": "PermissionRequest", "decision": ...}}`,
where the decision is `{"behavior": "allow", "updatedInput"?: object}` or
`{"behavior": "deny", "message": string}`; empty output decides nothing (read out of
the 2.1.270 bundle, and verified live: an allow runs the tool, and the agent reads a
deny's message as the tool's error). Permission
prompts, plan approval, and `AskUserQuestion` all arrive through this one hook, which
is why `Blocked.on` is a union of three and the answer path is one adapter. A question
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

The daemon subscribes to the events in the table. `MessageDisplay` is the only
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
(`.claude-plugin/marketplace.json`) holding one plugin, `plugin/`:
`plugin/.claude-plugin/plugin.json`, `plugin/hooks/hooks.json`, the launcher
`plugin/hooks/python`, and `plugin/src`, a link to the repository's `src`. The plugin is
a directory of its own so that an install copies those and not the repository's venv.
Installing the plugin installs the hooks; disabling or uninstalling it removes them,
and no settings file is edited by hands. `hooks.json` is generated from `hookconfig`
(`python -m hands.sessions.hookconfig > plugin/hooks/hooks.json`), and a test fails
when the checked-in file differs from what `hookconfig` declares. Every hook is exec
form: `${CLAUDE_PLUGIN_ROOT}/hooks/python -m hands.sessions.shim`, spawned by Claude
Code with no shell between. The plugin has no venv, so the launcher takes the first of
`python3.14`, `python3.13`, `python3.12` on `PATH` (the name says the version, so none
is started to ask), then a `python3` that says it is 3.12 or newer. It puts the
plugin's `src` on `PYTHONPATH` and execs it with `-P`, so the session's directory,
where Claude Code runs the hook, is never on the path and a project's own `json.py`
cannot stand in for the standard library's. The shim runs as the process Claude Code
spawned, so its parent is the claude process. A launcher that ran Python as its child,
as `uv run` does, would record its own pid instead. The shim's home is `HANDS_HOME`,
or `~/.hands`, found the way the `hands` CLI finds it; a relative `HANDS_HOME` is
refused, since a hook runs in its session's directory. With no Python new enough, every
hook fails saying so: the plugin cannot run at all, which is not the same as hands
being off.

**The shims.** Each is one process per hook: POST stdin to the daemon socket, exit.
At `SessionStart` the shim also writes
`~/.hands/sessions/<session_id>.json` with its parent pid, `cwd`, and
`transcript_path`; that file is the one record of the session's membership, written
by one writer. It is written whether or not the daemon is up, so a daemon started
later finds the sessions already running. The hooks are installed whether or not
hands is running, so a shim that cannot reach the socket asks the heartbeat why
(`hands.sessions.heartbeat.look`, the judge `hands status` uses). A hands that was
stopped or never ran is off, not broken: the shim exits 0 and prints nothing, and a
permission request falls through to Claude Code's own dialog. So is one still
starting: `hands run` writes its first heartbeat before it imports Pipecat and serves
the socket, and it reads the session files once it does. A hands whose heartbeat
says it died, hung, or is up but not answering, or whose heartbeat cannot be read,
makes the shim exit 1 with the socket error and the verdict on stderr, so Claude Code
shows the failure in the session where it happened rather than letting a dead daemon
look like a quiet one `[LAW:no-silent-failure]`.

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
request from a session the daemon does not know, such as one started before the
daemon was, is let go at once.

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
`Edited`. `hands.core.turn.Opening` stays `Asked | Notified`, the prompt that opened the
turn; a question Claude put to the user is the `Questioned` step.

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

The reducer will receive steps as events and ask for their summaries as they arrive, so
that by the time `Stop` fires most of the turn's summary is built; that is the progress
ticket's, not this one's.

**Backfill.** When the daemon attaches to a session that has been running for an
hour, `read_session(session, since)` reads the same file from an earlier point through
the same recognisers. What it hands over is `Happening = Opening | Step`: what opened
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

Opening = Asked | Notified          # who opened the turn, so a notification is never told as something asked
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
asked in plain text rather than through `AskUserQuestion` are found when the turn is summarised.

Record shapes worth knowing, observed in transcripts on 2026-09-14:

- `ai-title` → `aiTitle`. Live session name, free, no model call. Use it for
  `list_sessions` labels.
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

**Subagents.** On the parent's tail, where `isSidechain` is false, a subagent is one
`Agent` call and its report, which becomes `Delegated`. Its own records are in
`<session>/subagents/agent-<id>.jsonl`, with `agentType`, `description`, and the
parent's `toolUseId` in the `.meta.json` beside it. Tailing those is later work
(failure mode 21).

**Results outside the transcript.** A formatter, a code generator, or a `sed` in a
shell command changes files that no `Edited` step names. So at `UserPromptSubmit` the
reducer emits `Snapshot`, and the reader records the target's `HEAD` and a tree of
everything git would keep; at `Stop` it emits `Compare`, which diffs that tree against
one taken now, and lists the commits reachable from where the turn ended and not from
where it began. The summariser gets the `Delta` beside the steps, because it is the
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

**What runs today** is the top level of the tree: each finished turn becomes a headline of
the configured length, followed by what git says the turn did, followed by the turn's
question where it ended on one. The sections below the headline are built and not yet
spoken. When a live session's `Stop` arrives, the reducer emits
`Summarise(session, closing)`, and `narrate` in `hands.voice.narrator` asks the tail what
that session has not been told. A turn opens at the last user record that is
not `isMeta`, not `isCompactSummary`, whose content is a string or a block list with no
tool result in it, and that does not follow a tool call or tool result: a message sent
while a tool runs belongs to the turn under way, and an image or document attached to a
prompt is named rather than read. The opening is `Asked`, or `Notified` when its
`origin.kind` is `task-notification`. The steps are assistant text (`Said`) and tool
calls matched to their results by id, each handed to `recognise`; subagent records and
thinking blocks are skipped, thinking because it is how Claude reached a result rather
than a result.

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
in all twelve turns, so this is out of reach at a `Stop`. The tail can complete a call it
has already told — the slot is still there — but a completed step told out of order only
makes sense once steps are told as they arrive, so the progress ticket owns it.

`render(turn, budget)` in `core` writes the turn as the summariser's message
under `TURN_BUDGET`: 600 characters of the opening, 1,500 of each text block, 200 of
each tool input, 400 of each result, and 40 steps, where a longer turn keeps its head
and tail and says how many steps in the middle were left out. The summariser is a
stateless call on the configured backend, OpenAI-compatible or Anthropic, capped at 200
tokens, with a 30-second timeout and no retries; an empty answer is a failure. Its
instruction asks for results rather than a play-by-play, no code names, paths, or
hashes, and an ending that asks the turn's question with the session as "it". The
narrator speaks the summary after the session's name, keeps it in the intermediary's
context, and writes it to the audit log as `Recounted`. When reading or summarising
fails, it says "cc-hands finished a turn, and I could not summarise it." without the model and
out of the context, and logs the reason, which is a `Failure` line. Measured live
against a real `claude -p` session, with Qwen3-30B-A3B on inferno: the summary was ready
1.3 s after `Stop` and first audio came at 1.36 s, and the session's end was heard
after its summary. Measured again against a session whose first `Stop` another hook
blocked: 1.22 s to the summary and 1.29 s to first audio, then 0.93 s and 1.00 s for
the second stop of the same turn, which was heard as what the turn did after the
first — nothing of it twice.

The rest of this section is planned: step summaries built as steps arrive, children
below the top level, and streaming.

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

Two limits of the seam rather than of the function, both from the streaming path, where Pipecat hands
the filter one aggregated sentence at a time. A list the intermediary streams is seen an item at a
time and so is not counted aloud as a sequence, where the same list inside a summary is. And a fenced
block spanning chunks is only seen in the chunk its fence lands in; the rest arrives carrying no fence
and is read out as the ordinary text it then resembles, which is tracked as `hands-narration-2mc.1zu`.

Carrying the open fence in the filter was tried and reverted. Pipecat does support a stateful filter —
`handle_interruption` is called on every filter when an interruption frame arrives — but it never tells
a text filter that a reply ended, so the carry has no bounded lifetime: a reply that legitimately ends
inside a fence leaves it set, and every later utterance is replaced by a block announcement until the
user interrupts. Muting the assistant is worse than the fault it fixes. Closing this means skipping the
block at the aggregator, where it is broken up, and not in the filter. Every rule that makes text
sayable at all still applies to every chunk.

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

The summary instruction still asks for spoken form, and that is not duplication: the
model does what only the model can, turning `created_at` into "the creation date" rather
than "created at". The filter guarantees the floor beneath it.

**The summariser** is a stateless call on the configured backend, separate from the
conversational context. It is asked for one thing only — prose about what the turn did —
because it is the only thing nothing else can supply. Code and diffs are described by what
they do: "adds a retry around the token refresh, three attempts with backoff," not the
lines. Step summaries requested as steps arrive, so that first audio does not wait for the
whole turn to be read, are planned and belong to the progress ticket.

**The narration tree.** `core/narration.py` cuts a finished turn into segments: the
headline, what the repository did, one segment per question, and one section per topic —
the change, the tests, the commit, the commands, what it read, the plan, the subagents,
the other tools, what it said. The topics are not a table of rules written beside the
steps; they are a match over the `Step` union, which already draws exactly those lines, so
a new kind of step is a compile error here rather than a result with nowhere to go
`[LAW:types-are-the-program]`. Every step lands in a section, which is what makes "more on
that" able to reach anything the turn did. A segment holds the happenings it was cut from,
so `opened` renders them through the same `body` the whole turn goes through, and its
`refs` are read off those happenings rather than stored beside them — the record ids a
segment names are the records it holds, and the two cannot come apart
`[LAW:one-source-of-truth]`.

Only the headline is prose from a model. Everything else the top level says is arithmetic
over typed steps, and that is the point rather than an economy: a summariser was measured
on 2026-09-21 reporting "version two point seven point one" for a runner that printed 8.4.1,
and a count that is computed cannot be invented. It is also free, so the sections cost
nothing at the `Stop` that matters.

**What git says is not the model's to say.** Whether a turn committed is recorded twice —
by the step, when Claude Code writes a `gitOperation`, and by the delta read against where
the turn began — and each sees what the other misses: a `git commit` inside a heredoc
carries no operation for a step to hold, and the delta names it anyway. So the narration
says it, from whichever saw it, in words that carry no hash: "It committed and left five
files different." Asked for this instead, the model was measured both dropping the commit
entirely and reading its hash out loud, in the same afternoon. The instruction now asks it
to leave commits alone; when it says one anyway the listener hears it twice, and that
redundancy is kept on purpose — dropping git's clause whenever the report claims a commit
would suppress it in exactly the case it exists for, a commit claimed that never landed
`[LAW:no-silent-failure]`. A branch is said by `spoken_ref` rather than copied: this is the
one clause of the top level no model wrote, so the instruction cannot reach it, and the
filter in front of the speaker deliberately will not read a bare `feature/narration-tree`
as a path — a rule loose enough to catch it also eats "and/or" and "24/7" and costs each of
them a word. A ref is therefore said where its type already knows what it is, with its
separators as spaces and every word kept, because a branch is named so it can be told from
the others `[LAW:single-enforcer]`.

Only a commit has both sources. A push, a branch, or a pull request is known solely
from the `gitOperation` Claude Code writes beside the step, because none of them changes a
file or adds a local commit for the delta to find — so `git commit -m x && git push`, or a
push inside a heredoc, is heard as a commit and never as a push. The instruction bans the
model from saying any of the four rather than just commits, which is what makes that gap a
silence; the trade is kept because the model was measured dropping a commit and reading a
hash aloud in the same afternoon, and a clause code did not write is a clause nothing can
hold to `[LAW:no-silent-failure]`. `hands-narration-k08` owns closing it.

What git says is *not* said of a turn told twice. `Compare` pops the mark and only
`UserPromptSubmit` sets one, so the second telling of a turn whose first `Stop` was blocked
is handed an empty delta, and a commit made in its second half — a heredoc commit, which no
step records either — is never spoken at all. The eval's `second-telling` case is given that
empty delta because it is what production hands it, so the gap is measured rather than
papered over, and `hands-narration-k4q` owns closing it.

**The length is a number, and it is enforced rather than requested.**
`HEADLINE_SENTENCES` lives beside the instruction it rewrites, because changing it means
re-rendering that text and the two cannot be apart. It starts at one. Asked for one
sentence the local model wrote two in nine tellings out of twelve, so `narration` cuts the
reply to the number and keeps a closing question whole whatever the number is — the same
reason the spoken-form filter is a filter and not an instruction: a rule held as an
instruction is obeyed or not and checked by nobody. Nothing cut is lost, because the
sections hold every step the headline was made from. The eval reports how often the model
overran, which is the signal for changing the number.

**A question is in the tree and does not yet play.** Every `AskUserQuestion` becomes its
own segment, carrying its record id, said as a question and not as a count. It is not
spoken at `Stop`, for two reasons that both point the same way: the segment still holds
what Claude typed at a screen — a path, a hash, a "(Recommended)" — and the headline has
already been asked to end on the turn's question in spoken form, so playing both would say
it once well and once verbatim. Making questions play at every length, and finding the ones
Claude asked in prose rather than through the tool, is `hands-narration-2mc.4mu`.

**The eval.** `evals/narration.py` runs real turns, lifted whole out of real transcripts,
through the daemon's own recognisers, `render`, and summariser, and judges what comes back:
that the facts a listener must have are in it, that nothing code-shaped reached the ear,
that the headline is within its number, that every number said is a number the turn showed,
and that nothing the case forbids was said. The code-shape judge is `core/spoken.py` itself
rather than a second table of patterns, so the eval cannot drift from what the daemon does.
It exits 0, 1, or 2 — every check held, a check failed, or the model could not be reached —
and a model is stochastic, so it tells each case several times and every telling must hold.
Measured live against `claude -p` sessions on inferno, twice on 2026-09-22: `Stop` to the
summary 2.01 s and 1.82 s, `Stop` to first audio 2.07 s and 1.88 s, which puts the speech
leg at 60 ms. The four fixture cases summarise in 0.92 to 2.0 s, median 1.25 s.

**Streaming.** Steps from the tail are events like any other, so a session's progress
can be played as it happens: "running the tests," "editing the auth middleware." Text
streams line by line from `MessageDisplay`, so a long explanation is summarised while
Claude is still writing it; the transcript record that lands when the block finishes
supplies its record id. The routing table makes `progress` a note by default and the
focus overlay makes it play; `coalesce` folds a burst of edits into one sentence.

## Playback: bookmarks and resume

You will cut the reading off often, to ask which file or to answer something else, and
every interrupted reading must be resumable. So where playback is lives in the daemon,
not in the model's memory (failure modes 8 and 25). The player holds a `Playback`: the
segment on the speaker, and a stack of bookmarks where earlier readings were cut off.

An interruption pushes a bookmark at the segment that was playing. "Go back to what you
were talking about" is `resume()`, which pops the bookmark and replays that segment from
its start. "Skip that" and "say that again" are `skip()` and `repeat()`. "That part" is
the segment playing, or the last one played, and "more on that" is `expand()` on it.

Pipecat's output transport reports text as its audio plays, and on an interruption only
the text that played reaches the context. pocket-tts reports no word timings, so the
finest position is a sentence, and a segment is one to a few sentences. The playback
ticket confirms when a sentence's text frame arrives relative to its audio.

## Push pointers, pull content

The intermediary's context window holds the conversation with you, not session
transcripts. Hook events and played segments are injected as small frames carrying the
session title, the event kind, the spoken text, and the segment id. Steps, records, and
unplayed segments stay in the daemon, and `expand`, `read_session`, and `recall` pull
them. When the daemon starts or reconnects, the intermediary gets one note listing the
live sessions by title, state, and focus, and never their history; Happy's session
directory at connect is the model for it. This is the single decision that avoids
most of Happy's trouble: it pushed history in and could not pull, so it needed a
bootstrap dump, an eviction policy it never wrote, and a window that only grew.

The one thing that must be preserved for "give me the details of that part" to
resolve: every segment carries the `uuid`s of the records it summarises. "That part"
is then a lookup rather than a fuzzy search back through what was said.

## Sessions: membership from files, state from events, liveness from the OS

Three facts about a session have three different sources, and the registry derives
from all three rather than storing any of them twice `[LAW:one-source-of-truth]`.

- **Membership** is the set of files in `~/.hands/sessions/`. The shim writes one at
  `SessionStart` and removes it at `SessionEnd`. The daemon sweeps the directory once
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
  registry keeps the last one as `Session.report`. It is the source of whether a
  session's turn is over: an `idle` applied to a session in `Submitted`, `Working`,
  `Blocked`, or `AtDialog` ends its turn, however the turn was stopped, with no case for
  any one way of stopping it. A prompt still `Submitted` was cancelled by an Escape during
  its hooks, which sets `idle` ~70 ms later, or taken and stopped before the tail read its
  record: either way its turn ends here, and is told as itself only if a record says it
  ran. The session is `Idle` at once, with its nudge timed by hands (no `idle_prompt`
  follows a double Escape). Claude Code sets `idle`
  before the transcript says how the turn ended: an Escape's interrupt record is
  written ~100 ms after, and an Escape'd turn's Stop can fire after it. So the turn is
  kept as `Session.untold` and is told, once, at the first of five events: its Stop
  (told with the reply the Stop carries), its interrupt record (one naming its prompt), a turn after it
  opening (told before that turn's mark), the session ending, or the tick `UNTOLD_SECONDS` after the
  status, which is what tells a double Escape that leaves no record. Which came first
  is decided by the order they are applied, with no stamp compared against hands'
  clock. Claude Code sets `idle` only once a Stop's hooks have returned, and the shim
  waits (up to its 2 s post timeout) until the Stop is applied, so a stopped turn is
  normally ended by its Stop. It sets `busy` before a prompt's hooks run, so no `idle`
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

`list_sessions` offers a session while its pid is alive, labelled with its `aiTitle`,
its state, and whether it is the focus.

## Focus, drafts, and other state that stays out of the model's head

Models are unreliable at holding "which session we are talking about" and "I am
mid-draft" across a long conversation, and both failures are expensive. So both live
in the daemon as typed state and the tools default to them.

**Focus** is `SessionId | None`. Every tool that takes a `session` argument accepts
its omission as "the focus". "Switch to cc-hands" is `focus_session`, and the change
is a `Note` so the model knows without announcing it. The registry, not the prompt,
answers "which one did you mean".

**Drafts** are per target: `NoDraft | Staged(text, resolutions)`. The readback is
generated from the stored resolutions, never from the model repeating itself:
"Draft for cc-hands, reading 'auth middleware' as `authMiddleware.ts`: refactor the
auth middleware to use the new token helper." Speak what changed, not what you said.
A draft is staged, amended, and discarded; sending it waits for the Type effect
(`hands-keyboard-gxr.i5n`). How the keys reach the right session is settled and built:
fritter holds that session's pseudo-terminal and `Typist` types into it over a unix
socket, so there is no window to find, no focus to steal and no macOS permission to
ask for. What is left open is confirming the send through the `UserPromptSubmit` hook.
Until then the model tells the user that sending is not built. The send will append an audit record before it types, so "did it send
something I didn't approve" is answered by one file.

## The intermediary's tools

```
list_sessions()
read_session(session?, since?)
focus_session(session)
end_session(session?)
interrupt_session(session?)
send_command(session?, command, args?)
stage_draft(session?, text)      amend_draft(session?, text)
discard_draft(session?)
answer_permission(request, decision, message?)
answer_question(request, answers)
find_path(session?, query)
catch_up(since?)
recall(query, since?)
expand(segment?)                 resume()
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

`send_command` exists so that `/clear`, `/compact`, and `/model` reach the target as
commands, with their sigil intact, once the Type effect can send them.
`stage_draft` text always has a leading sigil escaped. The two never share a code path that inspects the first character; the
`Input` variant already knows. Claude Code reads three sigils at the start of a
prompt: `/` a command, `@` a file mention, `!` shell mode. Behind a space each is
plain text, so `Text` is always typed with a leading space, whatever it starts with,
and its newlines must stay inside the prompt rather than submit it.
Typed text lands after whatever is already in the target's input box. Claude Code
2.1.270 has no key that empties the box safely: Ctrl-S stashes but restores an
existing stash when the box is empty, Ctrl-L only redraws, a burst of Ctrl-U is
dropped, and Escape and Ctrl-C interrupt a turn. So what was actually submitted is
read back from the transcript, which records every prompt, and a prompt that is not
the draft is spoken.

## The audio side

**The gate is the turn boundary and the mute.** Pipecat's turn strategies act on
voice-activity frames, so the key is the VAD: a `KeyVAD` whose confidence is 1.0
while the key is down and 0.0 otherwise. Whisper keeps a second of pre-roll, so the
key is also the mute: microphone bytes become silence of the same length while the
key is up. Frames flow at full rate either way; only their content changes. The press
starts the turn, the release ends it and is final, and a press during playback
broadcasts the interruption that flushes queued audio. That is barge-in.

**The mute is decided where sound is captured, and waits out the speaker.** Measured
on 2026-09-14 with MacBook Pro speakers and microphone: the interruption stops writes
to the speaker within a few milliseconds of the press, but what was already written
stays above the room's floor at the microphone for about 185 ms, and Whisper turned
that tail into a word ("Wow.", "Well.", "What?") in every run where nobody spoke. A
Pipecat input filter could not stop it, because it runs when the event loop reaches a
frame, tens of milliseconds after capture. So `hands.voice.microphone` replaces the
local transport's two halves: the `Speaker` records, on every non-silent write, when
that sound will have died away at the microphone (the chunk's own length, the output stream's latency, and a
measured 150 ms echo path, counted from when the chunk is handed over, since an interruption cancels the wait for a write but not the write), and the `KeyedMicrophone` decides each buffer in PortAudio's
capture callback, dated by the buffer's recording time, not the callback's: silence
while the key is up or while the speaker's sound is still in the room. With it, a press
during playback gave no transcript with nobody speaking, and exactly "What time is it?"
when that was said 250 ms after the press. The cost is half duplex: while the speaker
is sounding, and for about 265 ms after its last chunk was handed over, the user is not heard, so a word spoken on top of
the press is lost rather than mixed with the reply. A stalled event loop delays the
interruption itself, and the mute then covers the reply for as long as it plays.
Acoustic echo cancellation would lift the half duplex and is a separate ticket.

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
The next turn ran end to end on the built-in devices. Every step that touches a device runs off the event loop.
A reopen that takes longer than 10 s, or one that fails, stops the run, which reads as down, and the next
`hands run` opens on whatever devices there are. The same path follows a headset plugged in, or a default changed in
Control Center. A Mac with no built-in microphone (a mini, a Studio) can
lose its only input device. Then the microphone holds a stream of nothing (`NoInput`),
which is opened, started, stopped, and closed like any other. The move is said as
`No microphone: hands cannot hear you. Speaking on …`. A daemon that starts that way
says `hands is up, but there is no microphone, so it cannot hear you.` instead of
failing its setup and stopping. With no stream, no frame reaches the
VAD and no turn starts, so the key edge answers a press itself:
`There is no microphone, so hands cannot hear you.` A microphone plugged in later
changes the default input, so the follower opens it like any other move. This is
tested against a PortAudio that lists no default input. A MacBook cannot be put in
that state, since macOS always falls back to the built-in microphone.

**The gate has one owner and several edges.** `PushToTalk` holds the key position;
whatever reads the physical world calls `move_key`. The edges are variants of one
config value, not modes of the gate `[LAW:one-type-per-behavior]`:

| Edge | Down | Up |
|---|---|---|
| `terminal` | space bar press | next space bar press (a terminal cannot report key-up) |
| `hotkey` | global key down | global key up |
| `button` | a HID button or headset button pressed | released |
| `web` | the phone page's talk button pressed | released |
| `wakeword` | the wake word heard | Silero VAD reports silence for the configured gap |

The wake-word edge is the only one that opens the mic without a hand, and it is
half-duplex: while the output transport is playing, the wake-word detector is deaf,
because an open mic in a room with speakers hears the pipeline's own voice. Acoustic
echo cancellation would lift that restriction and is a separate, later ticket.

**The transport is a variant too.** `Local(input_device, output_device)` is the Mac's
own mic and speakers. `WebRTC(host, port)` is Pipecat's SmallWebRTC transport serving
a page that a phone opens over the LAN or Tailscale; the page's talk button is a
proper key-down/key-up gate edge, and earbuds on the phone make own-voice bleed
moot. The pipeline between the transport and the gate is the same object in both
cases; `build_voice` matches on the variant exactly the way `build_llm` matches on
the backend.

**One audio owner.** Pipecat's output transport is the only thing that plays sound.
When two sessions finish at once, their utterances line up behind it instead of
overlapping.

## Loud failure

In an audio system the default output is silence, and silence is what "thinking"
sounds like too. So every failure has a path to the user that does not depend on the
thing that failed `[LAW:no-silent-failure]`:

1. **Speech.** The system channel says "the language model is unreachable", "Whisper
   returned nothing for that turn", "the session cc-hands is gone". These are
   `Speak` effects and need no model. `hands.voice.system` renders each fact from a
   template and queues it at the TTS processor, past the LLM and out of its context,
   so the model never reads a system line as a reply it gave. The worker's
   `on_pipeline_error` routes every error by the processor that raised it: the LLM's
   become "unreachable" or "failed: <category>", Whisper's become "speech recognition
   failed", and a TTS error goes to the screen. Pipecat files an SDK connection error
   under UNKNOWN, so "unreachable" is recognised from the exception type. Pipecat drops
   a turn Whisper transcribed to nothing without a word, so a thin `Whisper` subclass
   reports it as the transcription ends, rather than after the user turn's 5-second
   stop timeout. The channel says a burst once: a fault is not said again until ten
   seconds have passed since it last was. A fault that recurs recurs in bursts, and
   on 2026-09-22 a held key queued hundreds of empty turns whose reports went out
   every 0.43 s for as long as they drained — which is not a loud failure but a
   jammed one, because nothing else could have been heard while it ran. What ends a
   burst is a span of quiet and not some other announcement, because in the case this
   exists for there is never another one: with the microphone muted, staying quiet
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
   uptime, pipeline state, last audio out, and the count of live sessions. `hands
   status` prints it. When TTS itself is down, a macOS notification is posted through
   `osascript`. `hands indicator` is a menu-bar status item in a process of its own,
   which `hands run` starts in a session of its own, so neither the daemon dying nor the
   terminal's Ctrl-C takes it down first. It lives while the process that started it does,
   and once that is gone, until the light leaves up: it posts that notice and exits. Once a second it judges the heartbeat through `heartbeat.look`, the one
   read-and-judge that `hands status`, the crash check at start, and the hook shim also use. Its title
   shows one of five lights: up, not responding, down, off (stopped or never ran), and
   unreadable. An unreadable heartbeat is warned of as loudly as a dead daemon. It
   posts a notification when the light leaves up, at most once a minute, so a loop that
   stalls and recovers over and over is not announced every time. A departure inside that
   minute is held, and posted when the minute is up if hands is still not up. A daemon it finds already
   down on its first look is shown but not announced. A heartbeat whose pid is outside
   macOS's `1..99999` does not parse, since no process can have it. A pid counts as the
   daemon only while the process holding it started no later than the heartbeat's
   `started_at`, with a second of slack for a clock stepped back. The start comes from
   the kernel (the `kern.proc.pid` sysctl that `ps` itself reads, about 10 µs), and it
   is the rule the session sweep uses, from `hands.sessions.processes`. A pid that a
   later process took, after a crash or a reboot, reads as down rather than as not
   responding.
3. **Log.** Every effect and every failure is one line in `~/.hands/audit.jsonl`,
   written by the daemon alone (`hands.sessions.audit`). `hands log` prints the
   newest lines and follows the file. Each line is a value encoded one way: its type
   under `"type"`, its fields beside it, nested events and effects alike, and the
   wall-clock time under `"at"`. `Sessions` is the single writer for the session
   side: an `Applied` event (only one that changed the registry or called for an
   effect, so a quiet tick is not a line), each `Audit` record as it is (`Unregistered`,
   `AfterEnd`), then `Performed` or
   `EffectFailed` for every other effect. One wrapper writes every tool call as
   `Called` with its arguments and the result the model was handed; the context
   aggregators write each user turn as `Transcribed` and each reply as `Replied`;
   the system channel writes `Announced` with whether it spoke or posted; the narrator
   writes each turn summary it speaks as `Recounted`; and a
   loguru sink turns every error a `hands` module logs into a `Failure`. A dictation is
   traced from the words to the readback: `Transcribed`, `Called stage_draft`,
   `Replied`. The log watches
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

## Endurance

The intermediary talks with you for hours, and its own context is the one thing in
the system that grows. Two mechanisms bound it, and both were written at the same
time as the thing they bound (failure mode 5).

Pipecat's context summariser, configured with `LLMAutoContextSummarizationConfig`,
compacts the conversation when it crosses a token threshold, keeping the recent
turns verbatim. That is the eviction story. And because every narration, every tool
call, and every user transcript is also a line in the audit log, nothing that was
evicted is lost: `recall(query, since)` searches the log and `catch_up(since)`
replays what was said while you were away. The log is the long memory; the context
window is the working memory; the same pull-not-push rule that governs session
transcripts governs the intermediary's own past.

## Configuration: one file, parsed once

`~/.config/hands/config.toml` is read by `daemon` at startup and parsed into a frozen
`Config` whose fields are the variants above: the LLM backend, the transport, the
gate edge, the Whisper model, the voice, the policy overrides, and the permission
timeout. Secrets come from the environment and nothing else does. The
spike's environment variables are deleted when the file arrives, so there is one
source `[LAW:one-source-of-truth]`.

The settings cap `[LAW:no-mode-explosion]`: each config field names a variant or a
number. There are no boolean feature flags. A field that would be a flag is either a
variant with a real alternative or it does not exist.

## Stack

Pipecat ships every piece the pipeline needs: an in-process pocket-tts service, a
local audio transport over PyAudio, a Whisper service with an MLX build, the
SmallWebRTC transport, the Silero VAD, an OpenAI-compatible LLM service and an
Anthropic one, function registration from the `LLMContext`, `TTSSpeakFrame` for the
system channel, and `LLMMessagesAppendFrame` with `run_llm` for the other two.
Verified against the installed Pipecat 1.10 on 2026-09-12.

pocket-tts is MIT, 100M parameters, CPU-only by design, reports no word timings, and
streams: measured on this
Mac, first audio 87 ms after the text arrives and about 5.6x real time. Whisper
large-v3-turbo on MLX transcribes a four-second clip in under a second. The default
LLM is Qwen3-30B-A3B-Instruct-2507 in MLX 8-bit served by `mlx_lm.server` on inferno,
the M4 Max on the LAN; Claude through the Anthropic API is the other backend
variant. Measured on 2026-09-12, full voice-to-voice: a turn with a tool call had
first audio 4.3 s after key release; a plain turn 1.4 s.

Python, `uv`, pyright strict. State types are discriminated unions: frozen dataclasses
with a `Literal` kind or a union of frozen dataclasses.
