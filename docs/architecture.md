# Architecture

`hands` is one launchd-supervised Python process built from four packages in a strict
downhill order: `daemon` → `voice` → `sessions` → `core`. Every decision below serves
one of the nine needs in [needs.md](needs.md); the work that delivers it is planned in
[features.md](features.md). The catalogue of what goes wrong, and the rule each entry
produced, is [failure-modes.md](failure-modes.md).

The design has one organizing idea: **the pure core decides, the edges act, and every
fact has one home.** State, events, and effects are typed unions. The reducer that turns
an event into new state and a list of effects has no I/O, so every lifecycle transition
is a unit test with no mocks. The adapters that perform the effects are thin, and there
is exactly one of each: one process types into tmux, one replies to blocked hooks, one
owns the speaker.

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
 │  hook socket · session     │  core (pure)     │  tmux · procs · repos         │
 │  files · JSONL reader      │  types · reducer │  audit log                    │
 │                            │  digest · policy │                               │
 │                            └──────────────────┘                               │
 └────────▲──────────────────────────────────────────────┬───────────────────────┘
          │ hook shims, unix socket                       │ send-keys · hook replies
          │                                               ▼
       target Claude Code sessions (any tmux pane, started any way)
```

The audio leg on the left is the Pipecat pipeline the spike already runs. The
Claude Code leg on the right is the sessions package. They meet only through `core`'s
types: hook events enter the pipeline as frames, and the pipeline's tools call the
sessions API.

## Packages and the direction of dependency

**`core`** holds the domain: the state types, the reducer, the turn digest, the
attention policy, and the coalescing of pending speech. It imports the standard
library and nothing else. A test asserts that: `core` must not import `pipecat`,
`subprocess`, `socket`, or `asyncio` streams. That test is what makes
`[LAW:effects-at-boundaries]` a property of the package rather than a habit of its
authors. Everything in `core` can be exercised with plain values and no mocks.

**`sessions`** is every edge on the Claude Code side: the unix socket the hook shims
POST to, the session files the shims write, the JSONL reader, the tmux adapter, the
process-liveness check, the repo registry, and the audit log. It parses hook input once
at the socket into a `HookEvent` and rejects anything it does not recognise with a
logged error and a non-2xx reply `[LAW:parse-dont-validate]`. It exposes two things
upward: an async stream of events, and a small API the tools call.

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
Instant   = float                           # monotonic seconds

@dataclass(frozen=True)
class Session:
    id: SessionId
    title: str                # Claude Code's own ai-title
    cwd: Path
    transcript: Path          # the JSONL, from the hook payload
    pane: TmuxPane
    pid: int                  # the shim's parent, reported at SessionStart
    mode: PermissionMode      # from the hook payload
    state: SessionState

# One session is in exactly one of these. A session waiting on you and a session
# working are different things, and the type says which.
SessionState = Working | Idle | Blocked | Gone

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
class Gone:      reason: Literal["exited", "pane_closed", "pid_dead"]

# Everything Claude Code stops for arrives through the same PermissionRequest hook.
Blocker = Permission | Question | Plan
@dataclass(frozen=True)
class Permission: tool: str; input: Mapping[str, object]; suggestions: Sequence[object]
@dataclass(frozen=True)
class Question:   questions: Sequence[AskedQuestion]
@dataclass(frozen=True)
class Plan:       text: str

# What goes into a target pane. The variant decides the escaping, so there is no
# "if it starts with a slash" anywhere: Text always escapes a leading sigil,
# Command never does, Key is a named chord and carries no text at all.
Input = Text | Command | Key
Keystroke = Literal["escape", "enter", "ctrl_c", "up", "down", "tab", "shift_tab"]

# The reducer's whole vocabulary of effects. Adapters perform these and nothing else.
Effect = Reply | Type | Speak | Narrate | Note | Audit | ReadTurn | Launch
@dataclass(frozen=True)
class Reply:    request: RequestId; reply: HookReply
@dataclass(frozen=True)
class Type:     pane: TmuxPane; input: Input
@dataclass(frozen=True)
class Speak:    text: str; priority: Priority                # straight to TTS
@dataclass(frozen=True)
class Narrate:  event: Narration; priority: Priority         # LLM, run_llm on
@dataclass(frozen=True)
class Note:     event: Narration                             # LLM context, silent
@dataclass(frozen=True)
class Audit:    record: AuditRecord
@dataclass(frozen=True)
class ReadTurn: session: SessionId; since: Uuid | None       # the JSONL slice
@dataclass(frozen=True)
class Launch:   repo: Path; title: str                       # a new tmux window

Priority = Literal["blocking", "result", "fyi"]

# Every narration carries the record it came from, so "that part" is a lookup.
@dataclass(frozen=True)
class Narration:
    session: SessionId
    title: str
    kind: Literal["stop", "blocked", "subagent", "idle", "gone"]
    ref: Uuid | None
    text: str
    ledger: Ledger | None      # what the turn did, computed from the transcript

Draft = NoDraft | Staged
@dataclass(frozen=True)
class Staged:   text: str; resolutions: Sequence[Resolution]
```

Two things are deliberately absent. There is no `Session.last_seen` timestamp,
because silence measures nothing; liveness is the pid. And there is no queue of unsent
inputs, because Claude Code has its own input queue and the daemon must not keep a
second one `[LAW:one-source-of-truth]`.

## The reducer and its effects

```
reduce(state: Registry, event: Event) -> tuple[Registry, list[Effect]]
```

`Event` is the union of parsed hook events, tool calls from the intermediary, ticks
from the one clock, results of `ReadTurn`, and liveness reports. The reducer is a
pure function `[LAW:effects-at-boundaries]`: it never reads a file, checks a process,
or looks at a clock. When it needs the time it has already been handed one in a
`Tick`. When it needs the last turn's records it emits `ReadTurn` and receives them
back as an event.

The adapters live in `sessions` and `voice` and each performs one effect kind:
`Type` becomes `tmux send-keys`, `Reply` writes to the blocked shim's socket
connection, `Speak` becomes a Pipecat `TTSSpeakFrame`, `Narrate` and `Note` become
`LLMMessagesAppendFrame` with `run_llm` on or off, `Audit` appends one JSONL line,
`ReadTurn` reads the transcript slice, `Launch` opens a tmux window. An adapter that
fails raises; the supervisor logs it and the failure is spoken through the system
channel. Nothing is retried silently and nothing falls back `[LAW:no-silent-failure]`.

Because every transition is `reduce` on values, the test suite for the session
lifecycle is a table: state before, event, state after, effects. There is no pipeline,
no socket, and no tmux in those tests.

## Time has named owners

Correctness never depends on incidental ordering `[LAW:no-ambient-temporal-coupling]`.
Each timing fact has one owner.

| What must be ordered | Owner |
|---|---|
| When a user turn starts and ends | the gate, through the key position |
| Which utterance plays next, and that two never overlap | Pipecat's output transport |
| When a permission deadline warns and expires | the reducer, from `Blocked.deadline`, driven by one `Tick` source |
| Whether text typed mid-turn is queued or lost | Claude Code's own input queue |
| When the daemon is up, and restarting it | launchd, with `KeepAlive` |

Deadlines are data. The `Blocked` state carries the instant it expires and whether
the warning has been spoken. A single ticker sends `Tick(now)` once a second; the
reducer compares, and emits `Speak("ten seconds on that permission")` exactly once,
because the transition from `warned=False` to `warned=True` is a state change, not a
timer callback. At the deadline it emits `Reply(deny)`. There are no `sleep` calls
in the daemon that a correctness property depends on.

Claude Code queues messages submitted while a turn is running and shows them with
"Press up to edit queued messages" (seen in the 2.1.263 bundle). Whether `tmux
send-keys` mid-turn lands in that queue is the one thing the drafts ticket verifies
by experiment. If it does, `send_draft` while a target is working is an ordinary
send. If it does not, `send_draft` returns a typed `RefusedBusy` result and the user
hears it. Either way the daemon holds nothing.

## Three ways to reach the ear

Every event that reaches the pipeline takes one of three routes, and the route is
chosen by a table, not by code that looks at the event `[LAW:dataflow-not-control-flow]`.

- **Speak.** Text goes straight to TTS as a `TTSSpeakFrame`. No model call, no
  interpretation, no latency beyond synthesis. Used for facts a template can say:
  "auth-refactor finished", "ten seconds on that permission", "cc-hands is gone, the
  pane closed", "the language model is unreachable". This channel is also how the
  daemon reports its own failures, which is why it must not depend on the LLM.
- **Narrate.** The event is appended to the intermediary's context with `run_llm`
  on. The model summarises and speaks. Used when the content needs interpretation:
  a `Stop` with its ledger, a permission request, a question.
- **Note.** Appended with `run_llm` off. The model knows, and says nothing until
  asked. Used for context that changes what a later answer should say: a focus
  change, a subagent finishing, a session going idle.

The routing table is a value in `core`:

```python
Route = Literal["speak", "narrate", "note", "drop"]
DEFAULT_POLICY: Mapping[EventKind, Route] = {
    "stop": "narrate", "blocked": "narrate", "subagent_stop": "note",
    "idle_prompt": "speak", "gone": "speak", "session_start": "note",
    "message_display": "drop", "post_tool_use": "drop", ...
}
```

A per-session overlay of `focused | normal | muted` is a second table over the
first: a muted session's `narrate` becomes `note`, a focused session's `note` becomes
`narrate`. Adding a new event kind is a new row, and adding an overlay value is a new
column `[LAW:one-type-per-behavior]`.

Pending speech is a priority queue in `voice`: `blocking` before `result` before
`fyi`, and nothing starts while the key is down. Before an utterance plays, a pure
`coalesce` pass folds pending items from one session into one narration carrying all
their record ids, so three `Stop`s that arrived while you were talking become one
sentence, not three. Each item is a transition keyed by the record id that caused it,
so nothing is announced twice `[LAW:one-source-of-truth]`.

## Hooks are the event source, not the transcripts

Hooks are lower latency than tailing JSONL and carry more data. Every hook input
carries `session_id`, `transcript_path`, `cwd`, `permission_mode`, and
`hook_event_name`; the event-specific fields below were read out of the 2.1.263 bundle.

| Event | Payload fields |
|---|---|
| `SessionStart` | `source`, `agent_type`, `model`, `session_title` |
| `UserPromptSubmit` | `prompt`, `session_title` |
| `Stop` | `stop_hook_active`, `last_assistant_message` |
| `PermissionRequest` | `tool_name`, `tool_input`, `permission_suggestions` |
| `Notification` | `message`, `title`, `notification_type` in `permission_prompt`, `idle_prompt`, `auth_success`, `elicitation_dialog` |
| `SubagentStop` | `agent_id`, `agent_transcript_path`, `agent_type`, `last_assistant_message` |
| `MessageDisplay` | `turn_id`, `message_id`, `index`, `final`, `delta` |
| `SessionEnd` | the common fields |

The reply a `PermissionRequest` hook may give is `{"behavior": "allow",
"updatedInput"?: object}` or `{"behavior": "deny", "message": string}`. Permission
prompts, plan approval, and `AskUserQuestion` all arrive through this one hook, which
is why `Blocked.on` is a union of three and the answer path is one adapter. Answering
a question by returning the answers in `updatedInput` is the path the questions
ticket verifies live.

The documented nine events are not the real set. There are 33:

```
ConfigChange CwdChanged DirectoryAdded Elicitation ElicitationResult FileChanged
InstructionsLoaded MessageDisplay Notification PermissionDenied PermissionRequest
PostCompact PostModelSwitch PostToolBatch PostToolUse PostToolUseFailure PreCompact
PreModelSwitch PreToolUse SessionEnd SessionStart Setup Stop StopFailure SubagentStart
SubagentStop TaskCompleted TaskCreated TeammateIdle UserPromptExpansion
UserPromptSubmit WorktreeCreate WorktreeRemove
```

The daemon subscribes to the eight in the table. `PostToolUse` is deliberately not
one of them: it would put a process spawn on the agent's critical path for every tool
call, and the same facts are in the transcript at `Stop` for free.

**The shims.** Each is a two-line script in the target session's hook config: POST
stdin to the daemon socket, exit. At `SessionStart` the shim also writes
`~/.hands/sessions/<session_id>.json` with its parent pid, `$TMUX_PANE`, `cwd`, and
`transcript_path`; that file is the one record of the session's membership, written
by one writer. A shim that cannot reach the socket exits non-zero with a message, so
Claude Code shows the failure in the session where it happened rather than letting a
dead daemon look like a quiet one `[LAW:no-silent-failure]`.

**The constraint that will bite.** Hooks run in the agent's critical path with a
timeout, and `MessageDisplay` and `SessionStart` dispatch synchronously. A shim that
waits for anything stutters the agent's own output. Every shim POSTs and returns; the
sole exception is `PermissionRequest`, where blocking is the feature. Its timeout is
declared in the hook config, and the daemon derives its default-deny deadline from
that same number, so there is one place the budget is set `[LAW:single-enforcer]`.

## Transcripts: backfill, and the turn slice at Stop

Transcripts are read in two situations and no others.

**Backfill.** When the daemon attaches to a session that has been running for an
hour, no hooks fired for those turns. `read_session(session, since)` reads the JSONL
at the `transcript_path` the shim recorded to answer "what happened before I got
here."

**The turn slice.** `Stop` carries the turn's final text but no record id and no
account of what the turn did. So at each `Stop` the reducer emits `ReadTurn(session,
since=last)`, and the adapter reads the records since the watermark: the `uuid` of the
final assistant record, and every `tool_use` and `tool_result` block in between. From
those a pure `digest` computes the `Ledger`:

```python
@dataclass(frozen=True)
class Ledger:
    files_edited: Sequence[Path]
    files_read: int
    commands: Sequence[CommandRun]      # command, exit code, one-line tail
    tests: TestSummary | None           # passed, failed, the failing names
    turns_in_slice: int
```

The ledger is deterministic, small, and attached to the narration, so the model
summarises "edited three files and the tests pass" from facts rather than from
Claude's prose about itself. Known tool shapes are parsed by one table of
recognisers `[LAW:one-type-per-behavior]`; an unrecognised tool counts and is named.

Record shapes worth knowing:

- `ai-title` → `aiTitle`. Live session name, free, no model call. Use it for
  `list_sessions` labels.
- `assistant` → `.message.content[]`, blocks typed `text` | `thinking` | `tool_use`.
  Narrate `text` only.
- `user` → `.message.content` is a plain string for real user turns, or an array of
  `tool_result`; `isMeta: true` marks injected reminders. Filter to string + non-meta.
- `permission-mode` → live session mode.
- Every record: `uuid`, `parentUuid`, `timestamp`, `cwd`, `gitBranch`, `sessionId`.

Two filters are non-negotiable: `isSidechain: false`, or you narrate every subagent's
internal chatter; and the block-type filter, because in a sampled real session only
6 of 69 assistant content blocks were `text`. Budget speakable blocks, never records.

## Push pointers, pull content

The intermediary's context window holds the conversation with you, not session
transcripts. Hook events are injected as small frames carrying the session title, the
event kind, the record's `uuid`, the turn's final text for `Stop`, and the ledger.
Anything older is a `read_session` query. This is the single decision that avoids
most of Happy's trouble: it pushed history in and could not pull, so it needed a
bootstrap dump, an eviction policy it never wrote, and a window that only grew.

The one thing that must be preserved for "give me the details of that part" to
resolve: every narration carries the `uuid` of the record it came from. "That part"
is then a lookup rather than a fuzzy search back through what was said.

## Sessions: membership from files, state from events, liveness from the OS

Three facts about a session have three different sources, and the registry derives
from all three rather than storing any of them twice `[LAW:one-source-of-truth]`.

- **Membership** is the set of files in `~/.hands/sessions/`. The shim writes one at
  `SessionStart`; the daemon reads the directory when it starts and watches it after.
  A daemon restart therefore loses nothing: it re-reads the files, checks each pid,
  and is back where it was, with each surviving session in `Idle(last=None)` until
  the first backfill read establishes the watermark.
- **State** comes from the reducer applied to hook events since the daemon attached.
- **Liveness** is `kill(pid, 0)`. A session waiting for input is silent for hours and
  alive; a session in a tool loop is never silent. Silence measures nothing. A dead
  pid moves the session to `Gone("pid_dead")`, the file is removed, and the change is
  spoken.

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
"Sending to cc-hands: refactor the auth middleware to use the new token helper. I read
'auth middleware' as `authMiddleware.ts`." Speak what changed, not what you said.
`send_draft` appends an audit record before it types, so "did it send something I
didn't approve" is answered by one file.

## The intermediary's tools

```
list_sessions()
read_session(session?, since?)
focus_session(session)
start_session(repo, title?)
end_session(session?)
interrupt_session(session?)
send_command(session?, command, args?)
stage_draft(session?, text)      amend_draft(session?, text)
discard_draft(session?)          send_draft(session?)
answer_permission(request, decision, message?)
answer_question(request, answers)
find_path(session?, query)
catch_up(since?)
recall(query, since?)
```

The boundary rule: the tools route, name, and read session records. None reads or
writes a file in a repository. `find_path` returns paths from `git ls-files` in the
target's `cwd` so that a spoken "the auth middleware file" can be resolved to a real
path before it is sent; it returns names, never contents. `catch_up` and `recall`
read the daemon's own audit log. Give the intermediary an edit tool and it will
eventually decide that editing the file is faster than routing your request; the
surface above is the whole surface `[LAW:no-mode-explosion]`.

`send_command` exists so that `/clear`, `/compact`, and `/model` reach the target as
commands, with their sigil intact. `stage_draft` text always has a leading sigil
escaped. The two never share a code path that inspects the first character; the
`Input` variant already knows.

## The audio side

**The gate is the turn boundary and the mute.** Pipecat's turn strategies act on
voice-activity frames, so the key is the VAD: a `KeyVAD` whose confidence is 1.0
while the key is down and 0.0 otherwise. Whisper keeps a second of pre-roll, so the
key is also the mute: a `KeyMute` input filter replaces microphone bytes with silence
of the same length while the key is up. Frames flow at full rate either way; only
their content changes. The press starts the turn, the release ends it and is final,
and a press during playback broadcasts the interruption that flushes queued audio.
That is barge-in, and because of the mute the pipeline cannot transcribe its own
speech.

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
   `Speak` effects and need no model.
2. **Screen.** The daemon writes `~/.hands/status.json` every heartbeat with its pid,
   uptime, pipeline state, last audio out, and the count of live sessions. `hands
   status` prints it, and a tmux status-line snippet shows one glyph from it. When
   TTS itself is down, a macOS notification is posted through `osascript`.
3. **Log.** Every effect and every failure is one line in the audit JSONL.
   `hands log` tails it.

The daemon runs under launchd with `KeepAlive`, so a crash is a restart, and the
restart re-reads the session files and speaks that it is back. A hook shim that cannot
reach the socket fails visibly in the target session. Two clocks are never allowed to
disagree about whether the daemon is up: the heartbeat file is written by the daemon
alone, and everything else reads it.

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
gate edge, the Whisper model, the voice, the policy overrides, the repo roots, and the
permission timeout. Secrets come from the environment and nothing else does. The
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

pocket-tts is MIT, 100M parameters, CPU-only by design, and streams: measured on this
Mac, first audio 87 ms after the text arrives and about 5.6x real time. Whisper
large-v3-turbo on MLX transcribes a four-second clip in under a second. The default
LLM is Qwen3-30B-A3B-Instruct-2507 in MLX 8-bit served by `mlx_lm.server` on inferno,
the M4 Max on the LAN; Claude through the Anthropic API is the other backend
variant. Measured on 2026-09-12, full voice-to-voice: a turn with a tool call had
first audio 4.3 s after key release; a plain turn 1.4 s.

Python, `uv`, pyright strict. State types are discriminated unions: frozen dataclasses
with a `Literal` kind or a union of frozen dataclasses.
