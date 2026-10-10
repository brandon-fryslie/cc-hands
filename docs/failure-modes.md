# Failure modes

This document lists two kinds of failure: failures observed in Happy (`~/code/happy`, read
2026-09-08; citations are `file:line` in that repository) and failures that are inherent
to the design of cc-hands. Each entry includes the rule that prevents it. The rules are
the purpose of this document; the catalogue exists to produce them.

**Read this as a defect list, not a verdict.** Happy's core workflow works well in
practice. Its approach, a conversational intermediary between the user and running coding
agents, is proven, which is why it is worth studying. In real use, Happy was unreliable
and difficult to operate, and that is a separate issue from most of the entries below.
See §11 for the problems that caused the most trouble in practice.

## Observed

### 1. History delivered backwards

`storage.ts:669` sorts messages newest-first for an inverted chat list
(`ChatList.tsx:120`). `formatHistory` takes a slice of that array and never re-sorts it
(`contextFormatters.ts:76-78`), so the agent reads the newest message first under a
heading labeled "History." The related function `formatNewMessages` *does* sort in
ascending order (`contextFormatters.ts:69`). One code path was written with knowledge of
the array order, and the other was not.

**Rule:** array order is a property of the store. State it once, at the store, and do
not let a consumer infer it. When a consumer needs a specific order, it sorts the array
explicitly instead of relying on the order it receives.

### 2. Budget counted in the wrong unit

`MAX_HISTORY_MESSAGES = 50` truncates the list before filtering, and most records produce no
output: `agent-event`s, tool results, and tool calls without descriptions are all
dropped. In one real session, 6 of 69 assistant content blocks were text that could be
spoken. A limit of "50 messages" can result in two sentences.

**Rule:** set the budget in the unit that is actually consumed. For speech, that unit is
spoken length. Measure it after summarising, not on the input records. The same numbers
show what Happy missed: if 6 of 69 blocks are prose, the other 63 contain most of
what the agent did. Tool calls and their results are the primary record of a turn, and a
filter that drops them drops the results.

### 3. The same event announced twice, one of them useless

`sync.ts:2101` sends a formatted permission request that includes `<request_id>` through
the speech queue. Separately, `storage.ts:516` calls `sendTextMessage` with
`"Claude is requesting permission to use the ${toolName} tool"`. That message has no
request ID, so the agent cannot act on it. It also bypasses the queue, so it interrupts.

**Rule:** each event has exactly one emitter. If two code paths can announce the same
event, they will send different payloads.

### 4. Repeating announcements for state that hasn't changed

`sync.ts:2097` runs on every `agentState` update in which `requests` is non-empty. It
takes `requestIds[0]` and does not deduplicate anything. A request that is still pending
is announced again on every version increment, and a second pending request is never
announced.

**Rule:** announce state changes, not states. Compare against what was already announced,
using the event's own identifier as the key.

### 5. Nothing is ever evicted

`shownSessions` (`voiceHooks.ts:34`) prevents duplicate history dumps but never removes a
dump after it is added. `onMessages` re-injects the full text of a message on every
streaming edit. With three focused sessions, one window contains three full histories,
with no prioritization and no decay.

**Rule:** anything added to a context window needs an eviction plan, written at the same
time. If you cannot state what removes it, do not add it; make it a query instead. (This is the one failure
that cc-hands avoids through its structure; see "Push pointers, pull content" in
[docs/architecture.md](architecture.md#push-pointers-pull-content).)

### 6. A silent split between configurations

The BYO path sends neither a system prompt nor a first message
(`RealtimeSession.ts:78-84`). As a result, the agents of those users receive context only
through a dynamic variable that their dashboard prompt must interpolate. No error occurs;
the agent performs worse, and nothing shows it.

**Rule:** configuration variants use the same code path, or fail with a visible error at
the point where they diverge. A degraded mode that looks identical to the normal mode will
not be reported as a bug.

### 7. Documentation describing a design two models old

`docs/voice-architecture.md` and Happy's root `CLAUDE.md` both describe tools named
`messageClaudeCode`/`processPermissionRequest` that route through
`getCurrentRealtimeSessionId()`. This stopped being accurate at commit `a7378808`, when
routing moved into explicit tool arguments.

**Rule:** documentation that describes an interface must be checked when the interface
changes. Otherwise it is worse than no documentation, because it presents incorrect
information as if it were correct.

### 8. Instructing a behavior the system can't perform

The prompt tells the agent to "assume the user is just narrating what they will
eventually want to ask" (`voiceSystemPrompt.ts:11`), which is the correct approach, but it
gives the agent no place to store a draft. The behavior depends entirely on the model
remembering it.

**Rule:** if the prompt asks for stateful behavior, store the state outside the model.
Otherwise the prompt documents an intention; it does not implement a feature.

### 9. Dead paths behind live flags

`DISABLE_SESSION_STATUS: true` means `onSessionOnline`/`onSessionOffline` never run,
although both are fully implemented and maintained.

**Rule:** a flag that has been off since it was written is not configuration; it is code
that was not deleted.

### 10. Failures that produce silence instead of errors

A conversation token is a JWT signed with one provider's LiveKit keys. If the token is
presented to a different SFU, no error occurs: the client joins a room that the agent is
not in, and the user hears nothing.

Happy *detected* this case and fixed it correctly: `requireMintAndDialAgree`
(`voiceProvider.ts:73`) throws an error when the token's provider and the dialed SFU do
not match. This turns a silent failure into a visible one.

**Rule:** in an audio system, silence is the default output. Anything that can fail
silently must be changed to fail with a visible error, because the user cannot
distinguish "broken" from "thinking."

### 11. Transport defects, not content defects, caused the real problems

All of the defects above were found by reading the source code. None of them led to a bug
fix. The entire fix history of the subsystem concerns connection lifecycle, and four of
the five commits fix the *same* bug:

```
cf145bea  second-session disconnect via provider re-key
c837a5e9  LiveKit stale Room reuse — fixes second-session disconnect
c252c326  use web hook on native — fresh Room per session
8353b4b5  force voice session remount between calls — second-session disconnect
fda9be00  force kill voice assistant when stuck in connecting state
```

When a second call was started in the same app session, it connected to a dead room.
There were four attempts to fix this, at four different layers: remounting the component,
replacing the hook, patching LiveKit's Room reuse, and re-keying the provider. The final
fix appears twice, under two hashes (`632feb17` and `cf145bea`, on the same day, with the
same message), which is itself evidence of the problem.

**The lesson concerns where defects occur.** In a system with an LLM in the loop, content
defects degrade gracefully: the model absorbs a reversed history or a duplicated
announcement, and the result is a vague drop in quality that nobody reports. Transport
defects are hard failures that the user notices on the first attempt. Therefore, the bugs
found by reading code are not the bugs that determine whether the product is pleasant to
use.

**Rule:** allocate engineering effort based on failures experienced in use, not on code
smells. For cc-hands, this means session lifecycle, daemon liveness, and reconnection are
tested first and most thoroughly, before any of the context improvements above. It also
means that "I found this by reading" is a weaker signal than "this made me stop using it."

## Anticipated

These failures follow from the architecture of cc-hands. They have no citations because
they have not occurred yet.

### 12. A hook shim stalls the agent

Hooks run in Claude Code's critical path with a timeout (`timeoutMs`/`budgetMs`), and
`MessageDisplay` and `SessionStart` are dispatched with `forceSyncExecution: true`. A shim
that waits for TTS synthesis causes pauses in the agent's own output.

**Rule:** every shim sends a POST request to the daemon socket and returns immediately.
The only exception is `PermissionRequest`, where blocking *is* the intended behavior.
`MessageDisplay`, which fires for every batch of streamed lines, is an HTTP hook with a
short timeout, so no process is started for each batch.

### 13. The permission timeout expires mid-sentence

`PermissionRequest` blocks while the user decides aloud. The user will sometimes be slow
to respond, across the room, or talking to someone else.

**Rule:** the timeout is declared in configuration, not determined at runtime. The shim's
hook config sets the `timeout` on its `PermissionRequest` entry, and the daemon's
default-deny deadline is derived from that same value. Announce the timeout as it
approaches, so that the decision does not expire without warning.

### 14. The daemon dies and everything stays quiet

This is the worst case, because it is invisible. The daemon is the single process that
also runs the pipeline, so when the daemon dies, the pipeline also goes silent. Hooks
send POST requests and ignore the response, Claude Code runs normally, and the user stops
hearing output. To the user, this looks the same as "the agent is still working."

**Rule:** the daemon emits an audible or visible heartbeat, and a shim that cannot reach
the socket leaves a visible trace. A dead pipeline must never look like a working
pipeline that has nothing to report. The design is described in the "Loud failure"
section of `architecture.md`: the daemon runs in a terminal, where the user can see when
it exits; `status.json` is the heartbeat; the menu-bar indicator reports when the daemon
stops running; the system speech channel does not require a model; and the shim exits
non-zero when the heartbeat shows that the daemon died, is hung, or cannot be read. A
daemon that was stopped or never started is off, not dead, and its shim exits 0 without
output: the hooks are always installed, so a daemon that is off must not affect a session.

### 15. Two sessions speak at once

**Rule:** there is one audio owner and one queue. Utterances are queued; they do not
overlap. Pipecat's output transport is that single owner.

### 16. The intermediary does the work itself

If the intermediary has `Edit`, it will eventually decide that editing the file is faster
than routing the user's request.

**Rule:** it has only the tools listed in `architecture.md`. These tools route, name, and
read session records; none of them reads or writes a file in a repository.

### 17. A message is sent without the user's approval

Once the virtual keyboard makes sending possible, the model loses track of whether it is
in the middle of a draft and sends the draft.

**Rule:** the draft is stored in the daemon, not in the model's context. The daemon's call
log is the audit trail, and the readback is generated from the stored text, not from the
model repeating the draft.

### 18. Speech-to-text mangles an identifier

"auth middleware" is replaced by a guessed filename; a flag is transcribed as a word.

**Rule:** the readback reports what *changed* (resolutions, guesses, and inferred
targets), not a repetition of the user's sentence. If the system made a guess, the user
hears the guess.

### 19. The virtual keyboard collides with the UI

This failure applies to target sessions, which are the only sessions the daemon will type
into, through the planned virtual keyboard. Text that begins with `/` or `@` triggers
Claude Code's own completion. A send during a turn races with the input box. A permission
dialog interprets Enter as "Yes". Synthetic keystrokes add two more problems: keys typed
while the user is typing are interleaved with the user's keys, and keys sent while another
window has focus go to that window.

**Rule:** the daemon escapes leading sigils, refuses to send to a session that is blocked
on a permission prompt, and never keeps a hidden queue. Two questions are resolved before
the keyboard types anything: how keys reach the correct session's window without
interfering with the user's typing, and how a send is confirmed.

### 20. The session registry goes stale

A session ends without `SessionEnd` (because of a crash, a closed terminal, or a killed
process), and `list_sessions` continues to list it.

**Rule:** liveness is determined by a process check. The shim reports its parent PID at
`SessionStart`, and `list_sessions` lists a session while that PID is alive. Inactivity
does not indicate anything: a session that is waiting for input is inactive for hours but
alive, and a session in a tool loop is never inactive.

### 21. Subagent work is narrated as noise, or lost

A subagent's records are a separate conversation: a prompt, its tool calls, and a report.
If they are inserted into the parent's narration, they are noise. If they are dropped, the
narration loses work that the parent relied on, because the parent often reports only "the
review found three issues."

**Rule:** narration of the parent reads the parent transcript, where a subagent appears as
one `Agent` call and its report. The subagent's own transcript, at
`<session>/subagents/agent-<id>.jsonl`, is read at the point where the subagent reports
back, and is narrated as a separate part of that turn. It is linked by the `agentId` that
appears in both the parent's call and the notification. The `.meta.json` file next to it
is not read: for a foreground fork, that file has no description, and hundreds of
transcripts here have none at all, while the parent's call and the notification's
`<summary>` always identify the task.

### 22. TTS output leaks into the mic

The speakers and the microphone are in the same room. If the mic is open during playback,
the pipeline's own speech is fed back in as the user's next utterance.

**Rule:** the push-to-talk gate keeps the mic closed unless the key is held; VAD is off.

### 23. "That part" can't be resolved

The user asks for detail about something hands narrated. If the narration is only text,
resolving "that part" requires fuzzy-matching against what was said.

**Rule:** every segment of a narration includes the `uuid`s of the records it summarises.
The daemon tails the session JSONL, so the ids are available when the summary is built.
"That part" becomes a lookup: the segment currently playing, or the last segment played.
This is inexpensive to implement at the source and impossible to add later.

### 24. Text that cannot be spoken is sent to TTS

Claude formats its output for a screen: fenced code, tables, nested bullets, backticked
identifiers, file paths, commit hashes, and URLs. If this text is sent to TTS unchanged,
it is read as "backtick backtick backtick python" or as a minute of symbols, and the
listener stops listening.

**Rule:** no text is read verbatim, and no text reaches TTS without first passing through
the spoken-form transform. Code and diffs are summarised by what they do; identifiers are
split into words; a path is read as its file name; a hash, id, or URL is described by what
it refers to, or omitted. The transform runs in one place, the TTS service's text
transform, so a model reply that contains a backtick is also handled there.

### 25. An interruption loses the thread

The user interrupts in the middle of a summary to ask "which file?". The answer is given,
and the rest of the summary is lost, because the model's only record of its position is
its own context, which now ends at the interruption.

**Rule:** the playback position is stored in the daemon, not in the model's memory. A
narration is a sequence of segments. An interruption pushes a bookmark for the segment
that was playing, and "go back to what you were talking about" pops the bookmark and
replays that segment from its start.
