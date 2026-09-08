# Failure modes

Two kinds: ones observed in Happy (`~/code/happy`, read 2026-09-08, citations are
`file:line` there) and ones inherent to cc-hands' own design. Each carries the rule that
prevents it. The rules are the point — the catalogue exists to produce them.

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

**Rule:** budget the thing you're actually spending. Count speakable blocks or tokens,
never records, and measure after the filter.

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
(This is the one cc-hands avoids structurally; see "Pull, don't push" in the README.)

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

## Anticipated

These come from cc-hands' own architecture. No citations — they haven't happened yet.

### 11. A hook shim stalls the agent

Hooks run in Claude Code's critical path with a timeout (`timeoutMs`/`budgetMs`), and
`MessageDisplay` and `SessionStart` dispatch with `forceSyncExecution: true`. A shim that
waits on TTS synthesis stutters the agent's own output.

**Rule:** every shim POSTs to the daemon socket and returns immediately. The sole
exception is `PermissionRequest`, where blocking *is* the feature.

### 12. The permission timeout expires mid-sentence

`PermissionRequest` blocks while you decide out loud. You will sometimes be slow, or
across the room, or talking to someone else.

**Rule:** the daemon owns an explicit default (deny) and a budget measured against the
real hook timeout, not a guess. Speak the timeout as it approaches rather than letting
the decision evaporate.

### 13. The daemon dies and everything stays quiet

The worst one, because it's invisible. Hooks POST and don't care about the response,
Claude Code runs normally, and you simply stop hearing things — indistinguishable from
"the agent is still working."

**Rule:** the daemon emits a heartbeat you can hear or see, and a shim that can't reach
the socket leaves a visible trace. Never let a dead pipeline look like a working one with
nothing to say.

### 14. Two sessions speak at once

**Rule:** one audio owner, one queue. Utterances line up; they don't mix.

### 15. The controller does the work itself

Give it `Edit` and eventually it will decide that editing the file is faster than routing
your request.

**Rule:** `--allowed-tools 'mcp__hands__*'` and nothing else. No Bash, no Read, no Edit.

### 16. Something is sent that you didn't approve

The model loses track of whether it's mid-draft and calls `send_draft`.

**Rule:** the draft lives in the daemon, not the model's head. `send_draft` is the only
write path, its call log is the audit trail, and the readback is generated from stored
text rather than from the model repeating itself.

### 17. Speech-to-text mangles an identifier

"auth middleware" becomes a filename guess; a flag becomes a word.

**Rule:** the readback speaks what *changed* — resolutions, guesses, inferred targets —
not a recitation of your sentence. If it guessed, you hear the guess.

### 18. `tmux send-keys` collides with the UI

Text beginning with `/` or `@` triggers Claude Code's own completion; sending mid-turn
races the input box.

**Rule:** the daemon knows each session's turn state from hooks, so it holds input until
the target is idle, and it escapes leading sigils.

### 19. The session registry goes stale

A session dies without `SessionEnd` — crash, closed pane, killed terminal — and
`list_sessions` keeps offering it.

**Rule:** registry entries expire on silence. Liveness is a recent event, not a past
`SessionStart`.

### 20. Subagent chatter gets narrated

**Rule:** filter `isSidechain: false` on the narration path. `SubagentStop` is a separate,
deliberate announcement if you want one at all.

### 21. The controller inherits the personal environment

Left alone it pulls in `~/.claude`: per-turn hooks, a large skill catalog, several MCP
servers, and a global CLAUDE.md of git and ticket mandates.

**Rule:** `CLAUDE_CONFIG_DIR` plus `--setting-sources ''` plus `--strict-mcp-config`, and
run the controller from its own directory so no project `CLAUDE.md` is picked up from the
working directory.

### 22. "That part" can't be resolved

You ask for detail on something it narrated. If narration is just text, resolving that
means fuzzy-matching back through what it said.

**Rule:** every narration carries the `uuid` of the record it came from. "That part"
becomes a lookup. Cheap at the source, impossible to retrofit.
