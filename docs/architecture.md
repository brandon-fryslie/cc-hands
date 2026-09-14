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
 │  files · JSONL tail · git  │  types · reducer │  audit log                    │
 │                            │  steps · policy  │                               │
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

**`core`** holds the domain: the state types, the reducer, the step recognisers, the
spoken-form transform, the narration tree, the attention policy, and the coalescing of
pending speech. It imports the standard library and nothing else. A test asserts that:
`core` must not import `pipecat`, `subprocess`, `socket`, or `asyncio` streams. That
test is what makes `[LAW:effects-at-boundaries]` a property of the package rather than
a habit of its authors. Everything in `core` can be exercised with plain values and no
mocks.

**`sessions`** is every edge on the Claude Code side: the unix socket the hook shims
POST to, the session files the shims write, the JSONL tail, the git delta reader, the
tmux adapter, the process-liveness check, the repo registry, and the audit log. It
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
    pane: TmuxPane | None     # None outside tmux
    cwd: Path
    transcript: Path          # the JSONL, from the hook payload

@dataclass(frozen=True)
class Session:
    membership: Membership
    mode: PermissionMode      # from each hook payload that carries it
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
Effect = Reply | Type | Speak | Narrate | Note | Play | Summarise | Snapshot | Audit | Launch
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
class Play:     narration: NarrationId; segment: SegmentId   # straight to TTS, bookmarked
@dataclass(frozen=True)
class Summarise: narration: NarrationId; steps: Sequence[Step]; expand: SegmentId | None
@dataclass(frozen=True)
class Snapshot: session: SessionId; point: Literal["turn_start", "turn_end"]  # git delta
@dataclass(frozen=True)
class Audit:    record: AuditRecord
@dataclass(frozen=True)
class Launch:   repo: Path; title: str                       # a new tmux window

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
an event.

The adapters live in `sessions` and `voice` and each performs one effect kind: `Type`
becomes `tmux send-keys`, `Reply` writes to the blocked shim's socket connection,
`Speak` becomes a Pipecat `TTSSpeakFrame`, `Narrate` and `Note` become
`LLMMessagesAppendFrame` with `run_llm` on or off, `Play` sends a segment to TTS
through the player, `Summarise` calls the summariser, `Snapshot` records or diffs the
target's git state, `Audit` appends one JSONL line, `Launch` opens a tmux window. An
adapter that fails raises; the supervisor logs it and the failure is spoken through
the system channel. Nothing is retried silently and nothing falls back
`[LAW:no-silent-failure]`.

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
| Where a narration resumes after you cut in | the player's bookmark stack, from the segment the output transport was playing |
| When a permission deadline warns and expires | the reducer, from `Blocked.deadline`, driven by one `Tick` source |
| Whether text typed mid-turn is queued or lost | Claude Code's own input queue, measured to queue it |
| When the daemon is up, and restarting it | launchd, with `KeepAlive` |

Deadlines are data. The `Blocked` state carries the instant it expires and whether
the warning has been spoken. A single ticker sends `Tick(now)` once a second; the
reducer compares, and emits `Speak("ten seconds on that permission")` exactly once,
because the transition from `warned=False` to `warned=True` is a state change, not a
timer callback. At the deadline it emits `Reply(deny)` and says so. The ticker's
period only bounds how late a deadline is heard; no correctness property depends on
a `sleep`.

The deadline itself comes from one number. `hands.sessions.hookconfig` declares the
`PermissionRequest` hook's timeout (90 seconds) in the settings it prints; the shim
waits on the daemon that long for that hook alone, and the daemon denies 5 seconds
earlier, so the deny reaches Claude Code before Claude Code kills the hook
`[LAW:single-enforcer]`.

Claude Code queues messages submitted while a turn is running and shows them with
"Press up to edit queued messages". Measured on 2.1.270: text pasted into a working
pane and submitted lands in that queue and runs when the turn ends, so `send_draft`
to a working target is an ordinary send and the daemon holds nothing. A permission
dialog is the exception: it swallows pasted text and takes the Enter as "Yes". So
`send_draft` to a `Blocked` target is refused as `AwaitingPermission`, the draft stays
staged, and the user hears why.

## Four ways to reach the ear

Every event that reaches the pipeline takes one of four routes, and the route is
chosen by a table, not by code that looks at the event `[LAW:dataflow-not-control-flow]`.

- **Speak.** Text goes straight to TTS as a `TTSSpeakFrame`. No model call, no
  interpretation, no latency beyond synthesis. Used for facts a template can say:
  "auth-refactor finished", "ten seconds on that permission", "cc-hands is gone, the
  pane closed", "the language model is unreachable". This channel is also how the
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
input carries `session_id`, `transcript_path`, `cwd`, `permission_mode`, and
`hook_event_name`, and every one but `SessionStart` carries `permission_mode`; the
event-specific fields below were read out of the 2.1.263 bundle. Payloads captured from
2.1.270 on 2026-09-14 carry no session title, so a session's title is the newest
`ai-title` record in its transcript.

| Event | Payload fields |
|---|---|
| `SessionStart` | `source`, `agent_type`, `model` |
| `UserPromptSubmit` | `prompt`, `prompt_id` |
| `Stop` | `stop_hook_active`, `last_assistant_message` |
| `PermissionRequest` | `tool_name`, `tool_input`, `permission_suggestions` |
| `PostToolUse`, `PostToolUseFailure` | `tool_name`, `tool_input`, `tool_use_id`, and the response or the error |
| `Notification` | `message`, `title`, `notification_type` in `permission_prompt`, `idle_prompt`, `auth_success`, `elicitation_dialog` |
| `SubagentStop` | `agent_id`, `agent_transcript_path`, `agent_type`, `last_assistant_message` |
| `MessageDisplay` | `turn_id`, `message_id`, `index`, `final`, `delta` |
| `SessionEnd` | the common fields |

The reply a `PermissionRequest` hook may give is printed on its stdout as
`{"hookSpecificOutput": {"hookEventName": "PermissionRequest", "decision": ...}}`,
where the decision is `{"behavior": "allow", "updatedInput"?: object}` or
`{"behavior": "deny", "message": string}`; empty output decides nothing (read out of
the 2.1.270 bundle, and verified live: an allow runs the tool, and the agent reads a
deny's message as the tool's error). Permission
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

The daemon subscribes to the events in the table. `MessageDisplay` is the only
live source of Claude's text: the transcript writes a text block once, after it
finishes (checked on 2026-09-14), while `MessageDisplay` fires with each batch of
finished lines as the text streams. It is dispatched synchronously for every batch,
even to a hook that declares itself async, so it is installed as an HTTP hook, which
Claude Code 2.1.270 supports: the event is POSTed to the daemon with no process
spawned, under a short timeout. Its cost per batch and Claude's behaviour when the
daemon is down are measured before it is relied on. The tail, not a hook, is how
the daemon reads tool calls and their results. `PostToolUse` and `PostToolUseFailure`
are subscribed for one fact the tail would give too late to use: that a tool the
daemon is still waiting on a permission for has run, because its dialog was answered
at the keyboard. They are declared `async`, so the shim they spawn on every tool call
never holds the agent up.

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

**The dialog and the hook race.** Measured on 2.1.270: Claude Code shows its
permission dialog at once and runs the `PermissionRequest` hook beside it, and
whichever answers first decides. An answer typed at the dialog does not end the hook;
it runs on to its own end and its output is ignored. So the daemon never learns of a
keyboard answer directly. It learns that the session moved on: the asked-about tool
finishing (`PostToolUse` or `PostToolUseFailure` with the same tool and input), or the
next `UserPromptSubmit`, `Stop`, `SessionEnd`, or `PermissionRequest`, withdraws the
waiting reply, which prints nothing. A hook whose connection closes first ends the wait with
no reply at all, so a later voice answer hears that the request is gone rather than
that it went through. That is how a keyboard refusal arrives: answering No or Esc at
the dialog fires no post-tool hook and no `Stop`, but Claude Code kills the waiting
hook, and the closed connection releases the session (measured on 2.1.270). An
`AskUserQuestion` answered at the keyboard, whose tool input comes back changed, is
released by the `Stop` that follows. What is left
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

**The tail.** From the moment a session registers, the adapter follows its JSONL from
the watermark and hands each new record to a pure `recognise`, which turns records into
`Step`s through one table of recognisers `[LAW:one-type-per-behavior]`. The reducer
receives steps as events and asks for their summaries as they arrive, so by the time
`Stop` fires most of the turn's summary is built.

**Backfill.** When the daemon attaches to a session that has been running for an
hour, `read_session(session, since)` reads the same file from an earlier point through
the same recognisers.

```python
# One variant per kind of thing a turn does. Every step names the record it came from.
Step = Said | Edited | Ran | Tested | Looked | Committed | Planned | Delegated | Asked | Other

@dataclass(frozen=True)
class Said:      ref: Uuid; markdown: str                  # assistant text; never spoken as is
@dataclass(frozen=True)
class Edited:    ref: Uuid; path: Path; patch: Patch       # Edit, Write, MultiEdit, NotebookEdit
@dataclass(frozen=True)
class Ran:       ref: Uuid; command: str; purpose: str | None; failed: bool; output: str
@dataclass(frozen=True)
class Tested:    ref: Uuid; runner: str; passed: int; failed: int; failing: Sequence[str]
@dataclass(frozen=True)
class Looked:    ref: Uuid; tool: str; target: str; found: str   # Read, Grep, Glob, web tools
@dataclass(frozen=True)
class Committed: ref: Uuid; operation: GitOperation        # commit, push, branch, pull request
@dataclass(frozen=True)
class Planned:   ref: Uuid; items: Sequence[TodoItem]      # TodoWrite and the task tools
@dataclass(frozen=True)
class Delegated: ref: Uuid; agent_type: str; description: str; report: str | None
@dataclass(frozen=True)
class Asked:     ref: Uuid; questions: Sequence[AskedQuestion]    # AskUserQuestion
@dataclass(frozen=True)
class Other:     ref: Uuid; tool: str; input: str; result: str    # named and summarised, never dropped
```

`Tested` is a `Ran` whose output a test-runner parser accepts: pytest, vitest, cargo
test, go test. Questions asked in plain text rather than through `AskUserQuestion` are
found when the turn is summarised.

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
  push, a branch change, or a pull request. A failed command's result is a string that
  begins `Error: Exit code N`.
- `permission-mode` → live session mode.
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
reducer emits `Snapshot(turn_start)`, and the adapter records the target's `HEAD` and a
`git stash create` object, which captures the working tree without changing it. At
`Stop`, `Snapshot(turn_end)` diffs against that baseline, lists the commits since
`HEAD`, and lists untracked files that appeared, which a stash object does not hold.
The summariser gets the delta with the steps. The snapshot races the agent's first edit
by however long the model takes to start, which is seconds, and an edit that wins the
race is still an `Edited` step.

## Summaries: spoken form, and the narration tree

Nothing is read verbatim. Claude writes for a screen, and markdown, code, tables, paths,
and hashes cannot be heard as written, so every word that reaches the speaker is
summarised or transformed first.

**Spoken form** is a pure function in `core` from text to speakable text, installed as
the TTS service's text transform so there is one place it is enforced
`[LAW:single-enforcer]`. It applies to summaries, to the intermediary's replies, and to
system speech alike. Headings become section cues and lists become counted sequences.
Identifiers are split into words, so `authMiddleware` is "auth middleware". A path
becomes its file name, with its directory only when two files share the name. Flags
become their names, and hashes, ids, and URLs are named by what they point at or
dropped. Code blocks, diffs, and tables never reach it as text, because they are
summarised first; one that leaks through is replaced by its kind and length, and the
leak is logged.

**The summariser** is a stateless call on the configured backend, separate from the
conversational context. It takes steps, the turn's final text, and the git delta, and
returns segments. Code and diffs are described by what they do: "adds a retry around
the token refresh, three attempts with backoff," not the lines. Step summaries are
requested as steps arrive from the tail, and the turn's narration at `Stop` assembles
them, so first audio does not wait for the whole turn to be read.

**The narration tree.** A turn's narration plays its top level: a headline, then any
questions, then one section per topic, such as the change, the tests, and the commit.
Each segment opens into children when asked, built from its records the first time.
Questions come from the final text and from choices Claude offered, and they are in the
top level at every length. The top level's length is a number in the config. It starts
at one sentence and is expected to change as soon as it is heard, so the eval script
measures it and changing it is cheap.

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
path before it is sent; it returns names, never contents. `catch_up` and `recall`
read the daemon's own audit log. Give the intermediary an edit tool and it will
eventually decide that editing the file is faster than routing your request; the
surface above is the whole surface `[LAW:no-mode-explosion]`.

`stay_silent` is how the model declines to answer words that were not addressed to
it, taken from Happy's `skip_turn`. Push-to-talk rarely needs it; the wake-word edge,
which opens the mic without a hand, does.

`send_command` exists so that `/clear`, `/compact`, and `/model` reach the target as
commands, with their sigil intact. `stage_draft` text always has a leading sigil
escaped. The two never share a code path that inspects the first character; the
`Input` variant already knows. Claude Code reads three sigils at the start of a
prompt: `/` a command, `@` a file mention, `!` shell mode. Behind a space each is
plain text, so `Text` is always typed with a leading space, whatever it starts with,
as one bracketed paste followed by Enter, which keeps its newlines inside the prompt.
The paste lands after whatever is already in the target's input box. Claude Code
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

The heartbeat is honest at both ends of a run. `hands run` writes its first heartbeat,
`pipeline starting`, before it imports Pipecat. The models load off the event loop,
which keeps beating, so a restart shows its new pid within half a second of the kill,
and only a loop that is actually stuck reads as not responding. launchd's SIGTERM,
Ctrl-C, the `q` key, and a failed background task all set one quit event. Its handler
is in place before the models load, so a stop during the load does not wait for them.
Shutdown lets every permission hook still waiting go undecided, including one that
arrives as shutdown begins, so that session's own dialog stands and the daemon exits
in under a second. After the socket is released, a stop writes a last heartbeat that
says `stopped`, and a stopped daemon reads as stopped even if its pid is later reused.
A crash writes nothing more, so its last heartbeat names a pid that is gone, and it
reads as down. A background task that failed or a pipeline that ended on its own
counts as a crash: the run raises, exits nonzero, and launchd starts it again.

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
