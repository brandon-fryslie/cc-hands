# Failure modes

Two kinds: ones observed in Happy (`~/code/happy`, read 2026-09-08, citations are
`file:line` there) and ones inherent to cc-hands' own design. Each carries the rule that
prevents it. The rules are the point — the catalogue exists to produce them.

**Read this as a defect list, not a verdict.** Happy's core workflow works well in
practice. Its approach — a conversational intermediary between you and running coding
agents — is proven, which is exactly why it's worth studying. What it was, in real use,
is unreliable and fiddly, and that is a different axis from most of what follows. See
§11 for where the actual pain was.

## Observed

### 1. History delivered backwards

`storage.ts:669` sorts messages newest-first for an inverted chat list
(`ChatList.tsx:120`). `formatHistory` slices off that array and never re-sorts
(`contextFormatters.ts:76-78`), so the agent reads the newest message first under a
heading that says "History." The sibling `formatNewMessages` *does* sort ascending
(`contextFormatters.ts:69`) — one path was written knowing the orientation and the other
wasn't.

**Rule:** array orientation is a property of the store, so state it once at the store and
never let a consumer infer it. Where a consumer must order, it sorts explicitly rather
than trusting what it was handed.

### 2. Budget counted in the wrong unit

`MAX_HISTORY_MESSAGES = 50` slices before filtering, and most records render to nothing —
`agent-event`s, tool results, and tool calls without descriptions all drop out. In a real
session, 6 of 69 assistant content blocks were speakable text. "50 messages" can mean two
sentences.

**Rule:** budget the thing you're actually spending, which for speech is spoken length,
and measure it after summarising, never on the records going in. The same number
carries the lesson Happy missed: if 6 of 69 blocks are prose, the other 63 hold most of
what the agent did. Tool calls and their results are the primary record of a turn, and
a filter that drops them drops the results.

### 3. The same event announced twice, one of them useless

`sync.ts:2101` sends a formatted permission request carrying `<request_id>` through the
speaking queue. `storage.ts:516` separately calls `sendTextMessage` with
`"Claude is requesting permission to use the ${toolName} tool"` — no request ID, so the
agent can't act on it, and it bypasses the queue so it interrupts.

**Rule:** one event, one emitter. If two code paths can announce the same thing, they will
disagree about the payload.

### 4. Repeating announcements for state that hasn't changed

`sync.ts:2097` fires on every `agentState` update where `requests` is non-empty, takes
`requestIds[0]`, and dedupes nothing. A still-pending request is re-announced on every
version bump; a second pending request is never announced at all.

**Rule:** announce transitions, not states. Diff against what was already said, keyed by
the event's own identifier.

### 5. Nothing is ever evicted

`shownSessions` (`voiceHooks.ts:34`) prevents redundant dumps but never removes one.
`onMessages` re-injects a message's full text on every streaming edit. Three focused
sessions means three full histories in one window with no priority and no decay.

**Rule:** anything pushed into a context window needs an eviction story written at the
same time. If you can't say what removes it, don't push it — make it a query instead.
(This is the one cc-hands avoids structurally; see "Push pointers, pull content" in
the README.)

### 6. A silent split between configurations

The BYO path sends neither system prompt nor first message
(`RealtimeSession.ts:78-84`), so those users' agents get context only through a dynamic
variable their dashboard prompt has to interpolate. Nothing errors; the agent is just
worse, invisibly.

**Rule:** configuration variants take the same code path or fail loudly at the fork. A
degraded mode that looks identical to the good one will not be reported as a bug.

### 7. Documentation describing a design two models old

`docs/voice-architecture.md` and Happy's root `CLAUDE.md` both describe tools named
`messageClaudeCode`/`processPermissionRequest` routing through
`getCurrentRealtimeSessionId()`. That stopped being true at commit `a7378808`, when
routing moved into explicit tool arguments.

**Rule:** docs that describe an interface are checked when the interface changes, or
they're worse than no docs — they're a map that confidently points the wrong way.

### 8. Instructing a behavior the system can't perform

The prompt tells the agent to "assume the user is just narrating what they will
eventually want to ask" (`voiceSystemPrompt.ts:11`) — correct instinct — and then gives
it nowhere to store a draft. The behavior exists only as a hope about the model's memory.

**Rule:** if the prompt asks for stateful behavior, give the state a home outside the
model. Otherwise you've documented an intention, not built a feature.

### 9. Dead paths behind live flags

`DISABLE_SESSION_STATUS: true` means `onSessionOnline`/`onSessionOffline` never run,
though both are fully implemented and maintained.

**Rule:** a flag that has been off since it was written is not configuration, it's
undeleted code.

### 10. Failures that produce silence instead of errors

A conversation token is a JWT signed by one provider's LiveKit keys. Present it to a
different SFU and nothing errors — the client joins a room the agent isn't in and the
user hears nothing.

Happy *caught* this one and fixed it well: `requireMintAndDialAgree`
(`voiceProvider.ts:73`) throws when the token's provider and the dialed SFU disagree,
turning an inaudible failure into a loud one.

**Rule:** in an audio system, silence is the default output. Anything that can fail
quietly must be made to fail loudly, because the user cannot tell "broken" from
"thinking."

### 11. The transport, not the content — where the pain actually was

Everything above was found by reading source. None of it generated a bug fix. The
subsystem's entire fix history is connection lifecycle, and four of the five commits are
the *same* bug:

```
cf145bea  second-session disconnect via provider re-key
c837a5e9  LiveKit stale Room reuse — fixes second-session disconnect
c252c326  use web hook on native — fresh Room per session
8353b4b5  force voice session remount between calls — second-session disconnect
fda9be00  force kill voice assistant when stuck in connecting state
```

Start a second call in one app session and it connected to a dead room. Four attempts at
four different layers — remount the component, swap the hook, patch LiveKit's Room reuse,
re-key the provider. The final fix even appears twice under two hashes (`632feb17` and
`cf145bea`, same day, same message), which is its own kind of evidence.

**The lesson is about where defects hide.** In an LLM-in-the-loop system, content defects
degrade gracefully — a reversed history or a duplicated announcement gets absorbed by the
model and shows up as vague low quality nobody files. Transport defects are hard failures
the user feels on the first try. So the bugs you find by reading are not the bugs that
determine whether the thing is pleasant to use.

**Rule:** budget engineering effort against lived failure, not against code smell. For
cc-hands that means session lifecycle, daemon liveness, and reconnect get tested first and
hardest — before any of the context refinements above. And it means "I found this by
reading" is a weaker signal than "this made me stop using it."

## Anticipated

These come from cc-hands' own architecture. No citations — they haven't happened yet.

### 12. A hook shim stalls the agent

Hooks run in Claude Code's critical path with a timeout (`timeoutMs`/`budgetMs`), and
`MessageDisplay` and `SessionStart` dispatch with `forceSyncExecution: true`. A shim that
waits on TTS synthesis stutters the agent's own output.

**Rule:** every shim POSTs to the daemon socket and returns immediately. The sole
exception is `PermissionRequest`, where blocking *is* the feature.

### 13. The permission timeout expires mid-sentence

`PermissionRequest` blocks while you decide out loud. You will sometimes be slow, or
across the room, or talking to someone else.

**Rule:** the budget is declared, not discovered. The shim's hook config sets the
`timeout` on its `PermissionRequest` entry, and the daemon's default-deny deadline
derives from that same number. Speak the timeout as it approaches rather than letting
the decision evaporate.

### 14. The daemon dies and everything stays quiet

The worst one, because it's invisible. The daemon is the single process that also holds
the pipeline, so a dead daemon is also a silent pipeline. Hooks POST and don't care
about the response, Claude Code runs normally, and you simply stop hearing things —
indistinguishable from "the agent is still working."

**Rule:** the daemon emits a heartbeat you can hear or see, and a shim that can't reach
the socket leaves a visible trace. Never let a dead pipeline look like a working one with
nothing to say. The design is the "Loud failure" section of `architecture.md`: launchd
restarts it, `status.json` is the heartbeat, the system speech channel needs no model,
and the shim exits non-zero.

### 15. Two sessions speak at once

**Rule:** one audio owner, one queue. Utterances line up; they don't mix. Pipecat's
output transport is that single owner.

### 16. The intermediary does the work itself

Give it `Edit` and eventually it will decide that editing the file is faster than routing
your request.

**Rule:** it has the tools listed in `architecture.md` and nothing else. They route,
name, and read session records; none reads or writes a file in a repository.

### 17. Something is sent that you didn't approve

The model loses track of whether it's mid-draft and calls `send_draft`.

**Rule:** the draft lives in the daemon, not the model's head. Its call log is the
audit trail, and the readback is generated from stored text rather than from the model
repeating itself.

### 18. Speech-to-text mangles an identifier

"auth middleware" becomes a filename guess; a flag becomes a word.

**Rule:** the readback speaks what *changed* — resolutions, guesses, inferred targets —
not a recitation of your sentence. If it guessed, you hear the guess.

### 19. `tmux send-keys` collides with the UI

This applies to target sessions, the only thing the daemon types into. Text beginning
with `/` or `@` triggers Claude Code's own completion; sending mid-turn races the input
box. Whether `tmux send-keys` mid-turn lands in Claude Code's own input queue is
unverified.

**Rule:** the daemon escapes leading sigils. The spike decides between sending
immediately and `send_draft` returning a typed refused-busy result; the daemon never
holds a hidden queue.

### 20. The session registry goes stale

A session dies without `SessionEnd` — crash, closed pane, killed terminal — and
`list_sessions` keeps offering it.

**Rule:** liveness is a process check. The shim reports its parent pid at
`SessionStart`, and `list_sessions` offers a session while that pid is alive. Silence
measures nothing: a session waiting for input is silent for hours and alive, and a
session in a tool loop is never silent.

### 21. Subagent work is narrated as noise, or lost

A subagent's records are its own conversation: a prompt, its tool calls, and a report.
Spliced into the parent's narration they are noise. Dropped, they lose work the parent
relied on, because the parent often says only "the review found three issues."

**Rule:** the parent's narration reads the parent transcript, where a subagent is one
`Agent` call and its report. A subagent's own transcript, at
`<session>/subagents/agent-<id>.jsonl` with its type and description in the `.meta.json`
beside it, is summarised as its own narration and linked to the parent call by
`toolUseId`.

### 22. TTS output leaks into the mic

Speakers and microphone share a room. An open mic during playback feeds the pipeline's
own speech back in as your next utterance.

**Rule:** the push-to-talk gate closes the mic unless the key is held; VAD is off.

### 23. "That part" can't be resolved

You ask for detail on something it narrated. If narration is just text, resolving that
means fuzzy-matching back through what it said.

**Rule:** every segment of a narration carries the `uuid`s of the records it summarises.
The daemon tails the session JSONL, so the ids are in hand when the summary is built.
"That part" becomes a lookup: the segment playing, or the last one played. Cheap at the
source, impossible to retrofit.

### 24. Something unspeakable reaches the speaker

Claude writes for a screen: fenced code, tables, nested bullets, backticked identifiers,
file paths, commit hashes, URLs. Sent to TTS as written, that becomes "backtick backtick
backtick python" or a minute of symbols, and the listener gives up.

**Rule:** nothing is read verbatim, and no text reaches TTS without passing through the
spoken-form transform. Code and diffs are summarised by what they do; identifiers are
split into words; a path is its file name; a hash, id, or URL is named by what it points
at or dropped. The transform runs in one place, the TTS service's text transform, so a
model reply that slips a backtick through is caught there too.

### 25. An interruption loses the thread

You cut in to ask "which file?" in the middle of a summary. The answer comes, and the
rest of the summary is gone, because the model's only memory of where it was is its own
context, which now ends at the interruption.

**Rule:** where playback stopped is daemon state, not model memory. A narration is a
sequence of segments; an interruption pushes a bookmark at the segment that was playing;
"go back to what you were talking about" pops it and replays that segment from its
start.
