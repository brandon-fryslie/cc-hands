# Happy's voice architecture — reference

Happy (`~/code/happy`) is the closest existing thing to cc-hands: hands-free control of
Claude Code through a conversational agent, shipped and in use. This documents how it
works so nobody has to re-read its source, and records which of its decisions to copy and
which to avoid.

**Its core workflow works well.** The approach is proven — that's why it's worth this
much attention. In real use its trouble was reliability and setup friction rather than
anything below; see `failure-modes.md` §11.

Everything below was read out of the source on 2026-09-08 and citations are `file:line`
relative to `~/code/happy`. Line numbers drift; the function names won't.

## It has no name for the intermediary

Worth stating first, because it explains a lot of what follows. Happy has no noun for the
agent in the middle. Its code names things after the *transport* — `voice`, `realtime`,
`VoiceSession`, `RealtimeSession`, `voiceHooks`, `realtimeClientTools`, `realtimeMode` —
and never after the role. Comments call it "the voice assistant" (five times); the docs
call it "the voice agent." To the user it introduces itself as "Happy," the app's own
name, so the intermediary is not distinguishable from the product.

The consequence shows up everywhere: it's treated as a feature of the audio stack rather
than an actor with a job. Nothing owns the question "what should this thing know, and
when." That question is the entire design of cc-hands.

## Shape

```
SessionView.tsx              mic button, starts/stops the call
RealtimeSession.ts           lifecycle, token fetch, routing state
RealtimeVoiceSession.tsx     native ElevenLabs bridge (useConversation)
RealtimeVoiceSession.web.tsx web bridge, same interface
voiceHooks.ts                app events → the agent
contextFormatters.ts         how a turn is rendered as text
realtimeClientTools.ts       tools the agent can call
voiceSystemPrompt.ts         the prompt, assembled client-side
voiceProvider.ts             which ConvAI service mints, which SFU to dial
voiceConfig.ts               flags and constants
```

All in `packages/happy-app/sources/`. **The server is a bouncer, not a broker.** It mints
a LiveKit JWT and rations seconds against ElevenLabs' own conversation list
(`packages/happy-server/sources/app/api/routes/voiceRoutes.ts`); no context passes through
it. Every decision about what the agent knows is made on the device.

## Context strategy: there isn't one

**Happy never summarizes.** There is no summarization step in the voice path. The strategy
is a bulk dump at connect, then a live firehose, and the ConvAI agent's own window is the
memory. No compaction, no eviction, no relevance selection.

### Bootstrap

`voiceHooks.onVoiceStarted(sessionId)` (`voiceHooks.ts:206`) builds one string:

1. **Session directory** — every active session as `- <uuid>: "<title>"`
   (`voiceHooks.ts:134`). This is the whole cross-session addressing table. Titles come
   from `session.metadata.summary.text`, which is Claude Code's *own* chat title, relayed
   by the CLI when a `type: 'summary'` record appears in the JSONL
   (`packages/happy-cli/src/api/apiSession.ts:615`) or when Claude calls the `change_title`
   MCP tool (`packages/happy-cli/src/claude/utils/startHappyServer.ts:64`). Happy generates
   no titles of its own.
2. **Full context for the focused session** via `formatSessionFull`
   (`contextFormatters.ts:88`): session ID, project path, title, then history.

That string is delivered twice, by two mechanisms: as
`dynamicVariables.initialConversationContext` (`RealtimeVoiceSession.tsx:59`), which only
lands if the dashboard-side agent prompt interpolates `{{initialConversationContext}}`;
and as a `# Conversation history so far` section on the system prompt that overrides the
dashboard prompt outright (`voiceSystemPrompt.ts:57`).

The BYO path sends **neither** system prompt nor first message
(`RealtimeSession.ts:78-84`), so a BYO user's agent only gets context through the dynamic
variable. Silent behavioral split between tiers.

### Live stream

After bootstrap, `shownSessions` (`voiceHooks.ts:34`) ensures each session is dumped at
most once per call. Everything after is incremental, over two channels:

- **`sendContext`** → `sendContextualUpdate`. Silent; the agent absorbs and does not
  speak. Immediate, never queued. Carries new messages, focus changes, attachment notices.
- **`sendPrompt`** → `sendTextMessage`. Injected as a user turn, so the agent responds.
  **Queued while anyone is speaking**, flushed as one joined blob on `idle`
  (`voiceHooks.ts:72`). Carries permission requests and ready events.

This split is the best idea in the codebase — see "What to steal."

## Routing

`sendMessageToSession` takes `{ sessionId, message }` (`realtimeClientTools.ts:22`); the
agent picks its own target from the directory. `processPermissionRequest` takes
`{ requestId, decision }` and reverse-looks-up the owning session by scanning every
session's `agentState.requests` (`realtimeClientTools.ts:71-78`).

`currentSessionId` survives but no longer routes anything — its only consumers are focus
dedup (`voiceHooks.ts:173`) and the duplicate permission path below.

The prompt tells the agent to address sessions by UUID in tool args while never speaking
an ID aloud (`voiceSystemPrompt.ts:14`), so it must maintain the UUID↔title mapping in
context, unassisted.

## Defects worth knowing about

**The bootstrap history is reverse-chronological.** `storage.ts:669` and `:749` sort
messages descending (newest first) because `ChatList` is an inverted FlatList
(`ChatList.tsx:120`). `formatHistory` takes `messages.slice(0, 50)`
(`contextFormatters.ts:76-78`) and never re-sorts. The agent reads the newest message
first and the oldest last, under a heading that says "History." Its sibling
`formatNewMessages` (`contextFormatters.ts:69`) *does* sort ascending — so the incremental
path is correct and the bootstrap path is not.

**The window is counted in the wrong unit.** `MAX_HISTORY_MESSAGES = 50` slices *before*
filtering, and `formatMessage` returns null for `agent-event` records and (because
`LIMITED_TOOL_CALLS` is true) for tool calls without a description. Tool results have no
branch at all. A tool-heavy session can deliver two or three real lines while reporting a
full history dump.

**Permission requests are announced twice.** `sync.ts:2101` calls
`voiceHooks.onPermissionRequested`, which sends the formatted block with `<request_id>`
through the speaking queue. Separately, `storage.ts:516` calls `sendTextMessage` directly
with `"Claude is requesting permission to use the ${toolName} tool"` — bypassing the
queue, bypassing `voiceHooks`, and carrying **no request ID**, so the agent cannot act on
it. For a focused session the agent gets both.

**Pending requests are re-announced.** `sync.ts:2097` fires on every `agentState` update
where `requests` is non-empty, always takes `requestIds[0]`, and dedupes nothing. A
still-pending request is re-announced on each version bump, and only the first pending
request is ever mentioned.

**Nothing is ever evicted.** `shownSessions` prevents redundant dumps but never removes
one. Focus three sessions in a call and three full histories sit in the window with no
priority and no decay. `onMessages` also re-injects a message's full text on every
streaming edit.

**The docs describe a two-model-old design.** `docs/voice-architecture.md` and the root
`CLAUDE.md` both say the tools are `messageClaudeCode`/`processPermissionRequest` routing
via `getCurrentRealtimeSessionId()`. That was true before commit `a7378808`.

## What to steal

**The two-channel split.** Silent-context versus speaks-now, with the speaking channel
queued while anyone is talking and flushed on idle. It is the correct distinction — "know
this" versus "say something about this" — and it's why Claude finishing three tasks
mid-sentence doesn't produce three interruptions. cc-hands wants the same thing as a
speech queue with a single audio owner, gated on push-to-talk.

**The read watermark — from the other feature.** Happy's TTS path
(`hooks/useTtsPlayer.ts`) tracks the last message spoken per session via
`setTtsPosition`/`getTtsPosition`, so "continue" resumes where it left off. The voice path
has no equivalent, and that absence is exactly why it repeats itself. Build the watermark
in from commit one. (Note that `sliceMessages` at `useTtsPlayer.ts:208` looks
order-confused in the same way `formatHistory` is — the newest-first convention isn't
documented anywhere near its consumers.)

**The audio-session owner.** One process holds the speaker; everything else asks. Happy
needed a dedicated commit for this (`ee65c1e1`). Trivial to build in, miserable to add
later.

**Three prompt lines** from `voiceSystemPrompt.ts`:

- *"by default assume the user is just narrating what they will eventually want to ask"* —
  the draft-buffer instinct. Happy states it and then gives the agent nowhere to *put* a
  draft; cc-hands closes that hole with `stage_draft`.
- *"You always answer using a single sentence... be very short until explicitly asked to
  elaborate."* Spoken output has a hard budget.
- *"Never mention internal session identifiers."* Use titles aloud, IDs in tool args.

**A working summarizer Happy doesn't use for voice.** `sync/llm/apiSummarize.ts` calls an
OpenAI-compatible endpoint with a well-written spoken-word system prompt. It exists for
the TTS feature; the voice path never touches it. Worth reading before writing our own.

## What to avoid

Don't dump at connect — cc-hands is incremental by nature and sidesteps the whole class of
bug. Don't budget in records. Don't reconstruct permission state from diffs when
`PermissionRequest` hands you the event. Don't let one context window be the only memory.
