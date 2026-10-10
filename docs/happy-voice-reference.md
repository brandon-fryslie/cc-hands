# Happy's voice architecture — reference

Happy (`~/code/happy`) is the existing project most similar to cc-hands: it provides
hands-free control of Claude Code through a conversational agent, and it is released and
in use. This document describes how it works, so that its source does not need to be read
again, and records which of its decisions to copy and which to avoid.

**Its core workflow works well.** The approach is proven, which is why this document
covers it in this much detail. In real use, its problems were reliability and setup
difficulty, not any of the issues described below; see `failure-modes.md` §11.

All information below was taken from the source on 2026-09-08. Citations use the format
`file:line`, relative to `~/code/happy`. Line numbers change over time; function names
do not.

## It has no name for the intermediary

This is stated first because it explains much of what follows. Happy has no name for the
agent in the middle. Its code names components after the *transport* — `voice`,
`realtime`, `VoiceSession`, `RealtimeSession`, `voiceHooks`, `realtimeClientTools`,
`realtimeMode` — and never after the role. Code comments call it "the voice assistant"
(five times), and the docs call it "the voice agent." The agent introduces itself to the
user as "Happy," which is the app's name, so the intermediary cannot be distinguished
from the product.

As a result, the code treats the agent as a feature of the audio stack, not as a
component with its own responsibilities. No component is responsible for deciding what
the agent should know and when. That decision is the entire design of cc-hands.

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

All of these files are in `packages/happy-app/sources/`. **The server only handles
authorization and quotas; it does not relay context.** It issues a LiveKit JWT and
limits usage in seconds, measured against ElevenLabs' own conversation list
(`packages/happy-server/sources/app/api/routes/voiceRoutes.ts`). No context passes
through the server. Every decision about what the agent knows is made on the device.

## Context strategy: none

**Happy never summarizes.** The voice path has no summarization step. Happy sends all
context in bulk when the call connects, then streams every update continuously, and
uses the ConvAI agent's own context window as its memory. It does no compaction, no
eviction, and no relevance selection.

### Bootstrap

`voiceHooks.onVoiceStarted(sessionId)` (`voiceHooks.ts:206`) builds one string:

1. **Session directory** — every active session, formatted as `- <uuid>: "<title>"`
   (`voiceHooks.ts:134`). This is the only table the agent has for addressing sessions.
   Titles come from `session.metadata.summary.text`, which is Claude Code's *own* chat
   title. The CLI relays it when a `type: 'summary'` record appears in the JSONL
   (`packages/happy-cli/src/api/apiSession.ts:615`) or when Claude calls the
   `change_title` MCP tool (`packages/happy-cli/src/claude/utils/startHappyServer.ts:64`).
   Happy does not generate any titles itself.
2. **Full context for the focused session** from `formatSessionFull`
   (`contextFormatters.ts:88`): session ID, project path, title, then history.

That string is sent twice, by two mechanisms. The first is
`dynamicVariables.initialConversationContext` (`RealtimeVoiceSession.tsx:59`), which takes
effect only if the agent prompt configured in the dashboard includes
`{{initialConversationContext}}`. The second is a `# Conversation history so far` section
in the system prompt, which replaces the dashboard prompt entirely
(`voiceSystemPrompt.ts:57`).

The BYO path sends **neither** a system prompt nor a first message
(`RealtimeSession.ts:78-84`), so for a BYO user, the agent receives context only through
the dynamic variable. The two tiers therefore behave differently, and nothing
indicates this.

### Live stream

After bootstrap, `shownSessions` (`voiceHooks.ts:34`) ensures that each session's full
context is sent at most once per call. All later updates are incremental and use two
channels:

- **`sendContext`** → `sendContextualUpdate`. Silent: the agent receives the information
  but does not respond aloud. Sent immediately, never queued. Carries new messages, focus
  changes, and attachment notices.
- **`sendPrompt`** → `sendTextMessage`. Sent as a user turn, so the agent responds.
  **Queued while anyone is speaking**, then sent as one combined message on `idle`
  (`voiceHooks.ts:72`). Carries permission requests and ready events.

This split is the best design decision in the codebase — see "What to adopt."

## Routing

`sendMessageToSession` takes `{ sessionId, message }` (`realtimeClientTools.ts:22`); the
agent selects the target session from the directory. `processPermissionRequest` takes
`{ requestId, decision }` and finds the session that owns the request by scanning every
session's `agentState.requests` (`realtimeClientTools.ts:71-78`).

`currentSessionId` still exists but is no longer used for routing. It is used only for
focus deduplication (`voiceHooks.ts:173`) and by the duplicate permission path described
below.

The prompt instructs the agent to identify sessions by UUID in tool arguments but never
to say an ID aloud (`voiceSystemPrompt.ts:14`). The agent must therefore keep track of
the UUID↔title mapping in its context, with no help from the code.

## Defects worth knowing about

**The bootstrap history is in reverse chronological order.** `storage.ts:669` and `:749`
sort messages in descending order (newest first) because `ChatList` is an inverted
FlatList (`ChatList.tsx:120`). `formatHistory` takes `messages.slice(0, 50)`
(`contextFormatters.ts:76-78`) and does not re-sort them. The agent reads the newest
message first and the oldest message last, under a heading labeled "History." The related
function `formatNewMessages` (`contextFormatters.ts:69`) *does* sort in ascending order,
so the incremental path is correct and the bootstrap path is not.

**The history limit is counted in the wrong unit.** `MAX_HISTORY_MESSAGES = 50` limits the
number of messages *before* filtering. `formatMessage` returns null for `agent-event`
records and, because `LIMITED_TOOL_CALLS` is true, for tool calls without a description.
Tool results are not handled at all. For a session with many tool calls, the agent can
receive only two or three actual lines in what is reported as a full history.

**Permission requests are announced twice.** `sync.ts:2101` calls
`voiceHooks.onPermissionRequested`, which sends the formatted block with `<request_id>`
through the speaking queue. Separately, `storage.ts:516` calls `sendTextMessage` directly
with `"Claude is requesting permission to use the ${toolName} tool"`. This call bypasses
the queue and `voiceHooks`, and includes **no request ID**, so the agent cannot act on it.
For a focused session, the agent receives both messages.

**Pending requests are announced repeatedly.** `sync.ts:2097` runs on every `agentState`
update where `requests` is not empty. It always takes `requestIds[0]` and does no
deduplication. A pending request is announced again on every version update, and only
the first pending request is ever announced.

**Nothing is ever removed from context.** `shownSessions` prevents duplicate full-context
sends but never removes a session's context. If the user focuses three sessions in one
call, three full histories remain in the context window, with no prioritization and no
expiry. `onMessages` also re-sends a message's full text on every streaming edit.

**The docs describe a design that is two versions out of date.** `docs/voice-architecture.md`
and the root `CLAUDE.md` both state that the tools are
`messageClaudeCode`/`processPermissionRequest` and that they route through
`getCurrentRealtimeSessionId()`. That was true before commit `a7378808`.

## What to adopt

**The two-channel split.** One channel provides silent context; the other prompts the
agent to speak immediately. The speaking channel is queued while anyone is speaking and
sent when the conversation is idle. This is the correct distinction — "know this" versus
"say something about this" — and it is why Claude completing three tasks while the user
is speaking does not cause three interruptions. cc-hands needs the same behavior,
implemented as a speech queue with a single audio owner, gated on push-to-talk.

**The read watermark, from a different feature.** Happy's TTS path
(`hooks/useTtsPlayer.ts`) records the last message spoken for each session with
`setTtsPosition`/`getTtsPosition`, so "continue" resumes from where playback stopped. The
voice path has no equivalent, which is why it repeats itself. Include the watermark
from the first commit. (Note that `sliceMessages` at `useTtsPlayer.ts:208` appears to
have the same ordering problem as `formatHistory`; the newest-first convention is not
documented near the code that depends on it.)

**The audio-session owner.** One process controls the speaker, and every other component
requests access from it. Happy needed a dedicated commit to add this (`ee65c1e1`). It is
easy to include from the start and difficult to add later.

**Three prompt lines** from `voiceSystemPrompt.ts`:

- *"by default assume the user is just narrating what they will eventually want to ask"* —
  this is the same idea as the draft buffer. Happy states it but gives the agent no place
  to store a draft; cc-hands provides one with `stage_draft`.
- *"You always answer using a single sentence... be very short until explicitly asked to
  elaborate."* Spoken output has a strict length limit.
- *"Never mention internal session identifiers."* Use titles when speaking and IDs in tool
  arguments.

**A working summarizer that Happy does not use for voice.** `sync/llm/apiSummarize.ts`
calls an OpenAI-compatible endpoint with a well-written system prompt for spoken output.
It is used by the TTS feature; the voice path does not use it. Read it before writing our
own.

## What to avoid

Do not send all context at connect. cc-hands is incremental by design, which avoids this
entire category of bug. Do not set limits in numbers of records. Do not reconstruct
permission state from diffs when `PermissionRequest` provides the event directly. Do not
rely on a single context window as the only memory.
